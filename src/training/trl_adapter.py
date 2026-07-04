"""TRL lane: map Hydra config onto SFTConfig/DPOConfig and run with alert instrumentation."""

from __future__ import annotations

import io
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from helper.display import is_interactive


if TYPE_CHECKING:
    from datasets import Dataset

_SAMPLE_PROMPTS = ("Once upon a time", "The weather this morning", "In a small village")


def _report_to(backend: str | None) -> str:
    return backend if backend in ("trackio", "wandb") else "none"


def _resolve_reward_funcs(paths: list[str]) -> list[Callable[..., Any]]:
    import importlib

    funcs: list[Callable[..., Any]] = []
    for path in paths:
        module_name, _, attr = path.partition(":") if ":" in path else path.rpartition(".")
        if not module_name or not attr:
            raise ValueError(
                f"Reward function path {path!r} must look like 'package.module:function' or 'package.module.function'"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ValueError(f"Cannot import module {module_name!r} for reward function {path!r}: {exc}") from exc
        target: Any = module
        for part in attr.split("."):
            target = getattr(target, part, None)
            if target is None:
                raise ValueError(f"Module {module_name!r} has no attribute {attr!r} (reward function {path!r})")
        if not callable(target):
            raise ValueError(f"Reward function {path!r} resolved to non-callable {type(target).__name__}")
        funcs.append(target)
    return funcs


def _grpo_reward_funcs(cfg: DictConfig) -> list[Callable[..., Any]]:
    paths = OmegaConf.select(cfg, "trainer.reward_funcs")
    if not paths:
        raise ValueError("GRPO needs at least one reward function — set trainer.reward_funcs to dotted import paths")
    return _resolve_reward_funcs([str(path) for path in paths])


def _require_prompt_column(dataset: Any, split: str) -> None:
    columns = list(getattr(dataset, "column_names", None) or [])
    if "prompt" not in columns:
        raise ValueError(
            f"GRPO {split} dataset must contain a 'prompt' column (got {columns}) — "
            "GRPOTrainer samples completions from prompts"
        )


def _apply_tracking_group(cfg: DictConfig) -> None:
    group = OmegaConf.select(cfg, "tracking.group")
    if group is None:
        return
    if OmegaConf.select(cfg, "tracking.backend") == "wandb":
        # wandb.init reads WANDB_RUN_GROUP; transformers' wandb callback has no group kwarg.
        os.environ.setdefault("WANDB_RUN_GROUP", str(group))
    # trackio: transformers' TrackioCallback never passes group to trackio.init — wandb here, Lightning lane wires
    # trackio grouping directly.


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def _smoke_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "smoke_test")) or os.environ.get("SMOKE_TEST") == "1"


def _apply_smoke(args_dict: dict[str, Any], dataset: Dataset | None, smoke: bool) -> tuple[dict[str, Any], Any]:
    if not smoke:
        return args_dict, dataset
    args_dict["max_steps"] = 1
    args_dict["save_strategy"] = "no"
    if dataset is not None:
        dataset = dataset.select(range(min(32, len(dataset))))
    return args_dict, dataset


def _build_args(cfg: DictConfig, cls: type, **overrides: Any) -> Any:
    d: dict[str, Any] = OmegaConf.to_container(cfg.trainer.args, resolve=True)
    d["report_to"] = _report_to(OmegaConf.select(cfg, "tracking.backend"))
    run_name = OmegaConf.select(cfg, "tracking.run_name")
    if run_name is not None:
        d["run_name"] = run_name
    space_id = OmegaConf.select(cfg, "tracking.space_id")
    if space_id is not None:
        d["trackio_space_id"] = str(space_id)
        private = OmegaConf.select(cfg, "tracking.private")
        d["hub_private_repo"] = True if private is None else bool(private)
    d["include_num_input_tokens_seen"] = True
    # Keep NaN losses visible to the alert callback; the default filter logs them as 0.0.
    d["logging_nan_inf_filter"] = False
    d.setdefault("disable_tqdm", not is_interactive())
    d.update(overrides)
    return cls(**d)


class _StderrTee(io.TextIOBase):
    """Duplicates writes to the real stderr and a log file."""

    def __init__(self, fh: TextIO) -> None:
        self._fh = fh

    def write(self, s: str) -> int:
        sys.__stderr__.write(s)
        self._fh.write(s)
        return len(s)

    def flush(self) -> None:
        sys.__stderr__.flush()
        self._fh.flush()


def _run_with_stderr_tee(train_fn: Callable[[], Any], experiment_dir: Path) -> Any:
    log_path = experiment_dir / "logs" / "stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # One stderr.log per run: append mode would let a previous path's traceback
    # permanently fail stderr_scan for every later retry.
    with log_path.open("w", encoding="utf-8") as fh:
        original = sys.stderr
        sys.stderr = _StderrTee(fh)
        try:
            return train_fn()
        except BaseException as exc:
            fh.write(traceback.format_exc())
            print(f"VERDICT: TRAIN_FAIL | {type(exc).__name__}: {exc}")
            raise
        finally:
            sys.stderr = original


def _load_split(dataset: str, split: str, for_eval: bool = False) -> Any:
    from datasets import DatasetDict, load_dataset, load_from_disk

    path = Path(dataset)
    if path.is_dir():
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            return ds[split]
        if for_eval:
            raise ValueError(
                f"{dataset} is a plain on-disk Dataset with no splits — eval_split would silently "
                "evaluate on the training data; save a DatasetDict or use one file per split"
            )
        return ds
    if path.is_file():
        fmt = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet", ".txt": "text"}.get(
            path.suffix.lower(), "json"
        )
        return load_dataset(fmt, data_files=str(path), split=split)
    return load_dataset(dataset, split=split)


def _load_model(name: str, dtype_str: str | None, quantization: dict[str, Any] | None = None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    # transformers v5 loads checkpoints in their stored dtype (often fp16) by default;
    # full-precision fp16 training diverges, so default to explicit float32.
    dtype = "auto" if dtype_str in (None, "auto") else getattr(torch, str(dtype_str))
    kwargs: dict[str, Any] = {"dtype": dtype}
    if quantization:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "trainer.quantization is set but bitsandbytes is not installed — run 'uv sync --group gpu'"
            ) from exc
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(**quantization)
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def _peft_config(cfg: DictConfig) -> Any:
    peft = OmegaConf.select(cfg, "trainer.peft")
    if not peft:
        return None
    from peft import LoraConfig

    return LoraConfig(**OmegaConf.to_container(peft, resolve=True))


def _write_meta(mlog: Any, param_count: int, vocab_size: int, cfg: DictConfig) -> None:
    mlog.append_event("meta", key="task", value=str(OmegaConf.select(cfg, "trainer.kind")))
    mlog.append_event("meta", key="param_count", value=param_count)
    mlog.append_event("meta", key="vocab_size", value=vocab_size)
    for key in ("target_params", "planned_tokens"):
        value = OmegaConf.select(cfg, f"trainer.{key}")
        if value is not None:
            mlog.append_event("meta", key=key, value=value)


def _final_train_loss(trainer: Any) -> float | None:
    for record in reversed(trainer.state.log_history):
        if "loss" in record:
            return float(record["loss"])
    return None


def _write_samples(
    model: Any, tokenizer: Any, experiment_dir: Path, prompts: tuple[str, ...] = _SAMPLE_PROMPTS, seed: int = 42
) -> None:
    import json

    import torch

    records = []
    model.eval()
    torch.manual_seed(seed)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Sampling, not greedy: sanity samples must reflect the model's distribution;
            # greedy decode loops even on healthy models and trips the repetition check.
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        records.append({"prompt": prompt, "text": tokenizer.decode(output[0], skip_special_tokens=True)})
    path = experiment_dir / "logs" / "samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    logger.info("Wrote {} generation samples to {}", len(records), path)


def _run_trl(
    cfg: DictConfig,
    config_cls: type,
    trainer_cls: type,
    dpo: bool = False,
    reward_funcs: list[Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from intern.callbacks import AlertRules, TRLAlertCallback
    from intern.metrics import MetricsLog

    grpo = reward_funcs is not None
    experiment_dir = Path(str(cfg.experiment_dir))
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    smoke = _smoke_enabled(cfg)
    main_process = _is_main_process()

    # run_start is the verify scope boundary — it must precede any load that can crash.
    mlog = MetricsLog(experiment_dir / "metrics.jsonl")
    if main_process:
        mlog.append_event(
            "run_start",
            task=str(OmegaConf.select(cfg, "trainer.kind")),
            run_name=OmegaConf.select(cfg, "tracking.run_name"),
            smoke=smoke,
        )

    dtype_str = OmegaConf.select(cfg, "trainer.dtype")
    quantization = OmegaConf.select(cfg, "trainer.quantization")
    quantization = OmegaConf.to_container(quantization, resolve=True) if quantization else None
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(str(cfg.trainer.model_name), dtype_str, quantization)

    train_dataset = _load_split(str(cfg.trainer.dataset), str(cfg.trainer.dataset_split))
    eval_split = OmegaConf.select(cfg, "trainer.eval_split")
    eval_dataset = _load_split(str(cfg.trainer.dataset), str(eval_split), for_eval=True) if eval_split else None
    if grpo:
        _require_prompt_column(train_dataset, "train")
        if eval_dataset is not None:
            _require_prompt_column(eval_dataset, "eval")

    smoke_overrides, train_dataset = _apply_smoke({}, train_dataset, smoke)
    if eval_dataset is not None and str(OmegaConf.select(cfg, "trainer.args.eval_strategy") or "no") == "no":
        smoke_overrides.setdefault("eval_strategy", "steps")
        logger.info("eval_split is set but eval_strategy='no' — overriding to 'steps' so eval actually runs")
    _apply_tracking_group(cfg)
    args = _build_args(cfg, config_cls, **smoke_overrides)

    callback = TRLAlertCallback(mlog, str(cfg.tracking.backend), AlertRules())

    param_count = sum(p.numel() for p in model.parameters())
    if main_process:
        _write_meta(mlog, param_count, len(tokenizer), cfg)

    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "callbacks": [callback],
        "peft_config": _peft_config(cfg),
    }
    if dpo:
        ref_model_name = OmegaConf.select(cfg, "trainer.ref_model_name")
        kwargs["ref_model"] = _load_model(str(ref_model_name), dtype_str, quantization) if ref_model_name else None
    if grpo:
        kwargs["reward_funcs"] = reward_funcs
    trainer = trainer_cls(**kwargs)

    if main_process:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        mlog.append_event("meta", key="trainable_param_count", value=trainable)
        if quantization:
            mlog.append_event("meta", key="quantized", value=True)

    lane = "GRPO" if grpo else ("DPO" if dpo else "SFT")
    logger.info("Starting {} run (smoke={}) in {}", lane, smoke, experiment_dir)
    _run_with_stderr_tee(trainer.train, experiment_dir)

    if not smoke and main_process:
        prompts = OmegaConf.select(cfg, "trainer.sample_prompts")
        _write_samples(
            trainer.model,
            tokenizer,
            experiment_dir,
            prompts=tuple(prompts) if prompts else _SAMPLE_PROMPTS,
            seed=int(OmegaConf.select(cfg, "seed") or 42),
        )

    final_train_loss = _final_train_loss(trainer)
    if main_process:
        print(f"VERDICT: TRAIN_OK | final_train_loss={final_train_loss}")
    return {
        "final_train_loss": final_train_loss,
        "steps": int(trainer.state.global_step),
        "param_count": param_count,
    }


def run_sft(cfg: DictConfig) -> dict[str, Any]:
    from trl import SFTConfig, SFTTrainer

    return _run_trl(cfg, SFTConfig, SFTTrainer)


def run_dpo(cfg: DictConfig) -> dict[str, Any]:
    from trl import DPOConfig, DPOTrainer

    return _run_trl(cfg, DPOConfig, DPOTrainer, dpo=True)


def run_grpo(cfg: DictConfig) -> dict[str, Any]:
    reward_funcs = _grpo_reward_funcs(cfg)
    from trl import GRPOConfig, GRPOTrainer

    return _run_trl(cfg, GRPOConfig, GRPOTrainer, reward_funcs=reward_funcs)
