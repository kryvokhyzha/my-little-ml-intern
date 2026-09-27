"""Run the TRL lanes with alert instrumentation and the run-boundary contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.loading import validate_columns
from training.models import load_model, load_ref_model, load_teacher_model, load_tokenizer, model_id, peft_config
from training.runtime import (
    apply_tracking_env,
    is_main_process,
    record_run_end,
    record_run_start,
    record_setup_failure,
    run_seed,
    run_with_stderr_tee,
    smoke_enabled,
)
from training.sampling import resolve_sample_prompts, write_samples
from training.trl.config import apply_smoke, build_args, final_train_loss, write_meta
from training.trl.rewards import environment_kwargs, reward_funcs_from_cfg


ExtraKwargs = Callable[[DictConfig], dict[str, Any]]


def _load_data_node(node: DictConfig, for_eval: bool = False) -> Any:
    from hydra.utils import instantiate

    container: dict[str, Any] = OmegaConf.to_container(node, resolve=True)
    # The eval-on-train guard is load_split's; custom _target_s own their split hygiene.
    if for_eval and container.get("_target_") == "data.loading.load_split":
        container.setdefault("for_eval", True)
    return instantiate(container)


def _silence_experimental() -> None:
    # trl.experimental warns once on import; the warning is noise in train logs.
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


def _build_trainer(
    cfg: DictConfig,
    trainer_cls: type,
    extra_kwargs: ExtraKwargs | None,
    model_as_id: bool,
    mlog: Any,
    smoke: bool,
    main_process: bool,
) -> tuple[Any, Any, Any, Any, int]:
    """Load, validate, and construct the trainer; return (trainer, tokenizer, train_ds, eval_ds, param_count)."""
    from transformers import set_seed

    from intern.callbacks import AlertRules, TRLAlertCallback

    task = str(OmegaConf.select(cfg, "trainer.kind"))
    # TRL applies get_peft_model before transformers' Trainer.__init__ seeds, so without this the
    # LoRA adapter init (and any random init) ignores cfg.seed.
    set_seed(run_seed(cfg))
    tokenizer = load_tokenizer(cfg)
    model = model_id(cfg) if model_as_id else load_model(cfg)

    train_node = OmegaConf.select(cfg, "data.train")
    train_dataset = _load_data_node(train_node) if train_node else None
    eval_node = OmegaConf.select(cfg, "data.eval")
    eval_dataset = _load_data_node(eval_node, for_eval=True) if eval_node else None
    extra = extra_kwargs(cfg) if extra_kwargs is not None else {}
    if "train_dataset" in extra:
        train_dataset = extra.pop("train_dataset")

    text_field = str(OmegaConf.select(cfg, "trainer.args.dataset_text_field") or "text")
    if train_dataset is not None:
        validate_columns(train_dataset, task, "train", text_field=text_field)
    if eval_dataset is not None:
        validate_columns(eval_dataset, task, "eval", text_field=text_field)

    smoke_overrides, train_dataset = apply_smoke({}, train_dataset, smoke)
    if eval_dataset is not None and str(OmegaConf.select(cfg, "trainer.args.eval_strategy") or "no") == "no":
        smoke_overrides.setdefault("eval_strategy", "steps")
        logger.info("data.eval is set but eval_strategy='no' — overriding to 'steps' so eval actually runs")
    apply_tracking_env(cfg)
    args = build_args(cfg, **smoke_overrides)

    callback = TRLAlertCallback(mlog, str(cfg.tracking.backend), AlertRules())

    param_count = None if model_as_id else sum(p.numel() for p in model.parameters())
    if main_process and param_count is not None:
        write_meta(mlog, param_count, len(tokenizer), cfg)

    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "callbacks": [callback],
        **extra,
    }
    # Optional kwargs travel only when set: the async trainers take no eval_dataset, and
    # AsyncDistillationTrainer takes no peft_config — a set value then fails loudly, not silently.
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    peft = peft_config(cfg)
    if peft is not None:
        kwargs["peft_config"] = peft
    trainer = trainer_cls(**kwargs)

    if param_count is None:
        param_count = sum(p.numel() for p in trainer.model.parameters())
        if main_process:
            write_meta(mlog, param_count, len(tokenizer), cfg)
    if main_process:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        mlog.append_event("meta", key="trainable_param_count", value=trainable)
        if "quantization_config" in cfg.model.main:
            mlog.append_event("meta", key="quantized", value=True)
    return trainer, tokenizer, train_dataset, eval_dataset, param_count


def _run_trl(
    cfg: DictConfig,
    trainer_cls: type,
    extra_kwargs: ExtraKwargs | None = None,
    model_as_id: bool = False,
) -> dict[str, Any]:
    """Shared lane body: run_start → load → validate → build args → train → samples → VERDICT.

    ``extra_kwargs(cfg)`` returns the trainer-specific kwargs (reference or teacher model, reward
    functions, environment). It runs after ``run_start``, so a crash in a teacher load stays
    inside the verify scope. A ``train_dataset`` key in its result replaces ``data.train`` (an
    env_spec owns its dataset). ``model_as_id`` passes the ``model.main`` repo id instead of a
    loaded model — the async trainers load the model themselves. A failure in any step prints
    ``VERDICT: TRAIN_FAIL`` and journals it.
    """
    from intern.metrics import MetricsLog

    task = str(OmegaConf.select(cfg, "trainer.kind"))
    experiment_dir = Path(str(cfg.experiment_dir))
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    smoke = smoke_enabled(cfg)
    main_process = is_main_process()

    # run_start is the verify scope boundary — it must precede any load that can crash.
    mlog = MetricsLog(experiment_dir / "metrics.jsonl")
    if main_process:
        record_run_start(experiment_dir, mlog, task, OmegaConf.select(cfg, "tracking.run_name"), smoke)

    with record_setup_failure(experiment_dir):
        trainer, tokenizer, train_dataset, eval_dataset, param_count = _build_trainer(
            cfg, trainer_cls, extra_kwargs, model_as_id, mlog, smoke, main_process
        )

    logger.info("Starting {} run (smoke={}) in {}", task, smoke, experiment_dir)
    run_with_stderr_tee(trainer.train, experiment_dir)

    if not smoke and main_process:
        # Training succeeded and the checkpoint is saved — a sampling failure (e.g. OOM on a
        # long probe prompt) must not sink the run, so it degrades to a skipped samples.jsonl.
        try:
            configured = OmegaConf.select(cfg, "data.sample_prompts")
            prompts, pre_rendered = resolve_sample_prompts(configured, eval_dataset, train_dataset, tokenizer=tokenizer)
            write_samples(
                trainer.model,
                tokenizer,
                experiment_dir,
                prompts=prompts,
                pre_rendered=pre_rendered,
                seed=run_seed(cfg),
            )
        except Exception as exc:
            logger.warning("Sample generation failed ({}) — training succeeded, samples skipped", exc)

    loss = final_train_loss(trainer)
    steps = int(trainer.state.global_step)
    if main_process:
        record_run_end(experiment_dir, loss, steps)
    return {"final_train_loss": loss, "steps": steps, "param_count": param_count}


def _ref_model(cfg: DictConfig) -> dict[str, Any]:
    return {"ref_model": load_ref_model(cfg)}


def _teacher_model(cfg: DictConfig) -> dict[str, Any]:
    return {"teacher_model": load_teacher_model(cfg)}


def _teacher_and_tools(cfg: DictConfig) -> dict[str, Any]:
    tools = environment_kwargs(cfg).get("tools")
    return {**_teacher_model(cfg), **({"tools": tools} if tools else {})}


def _rewards_and_environment(reward_funcs: list[Callable[..., Any]]) -> ExtraKwargs:
    """Reward functions resolved before run_start (fail fast), plus the environment wiring."""

    def extra(cfg: DictConfig) -> dict[str, Any]:
        env = environment_kwargs(cfg)
        # An env_spec may bring its own reward functions; the configured ones come first.
        spec_rewards = env.pop("reward_funcs", [])
        return {**env, "reward_funcs": reward_funcs + spec_rewards}

    return extra


def run_sft(cfg: DictConfig) -> dict[str, Any]:
    from trl import SFTTrainer

    return _run_trl(cfg, SFTTrainer)


def run_dpo(cfg: DictConfig) -> dict[str, Any]:
    from trl import DPOTrainer

    return _run_trl(cfg, DPOTrainer, _ref_model)


def run_kto(cfg: DictConfig) -> dict[str, Any]:
    """Unpaired preference alignment (KTO): per-example desirable/undesirable labels, no pairs needed."""
    from trl import KTOTrainer

    # KTO uses a reference model exactly like DPO (model.ref optional; TRL clones one when absent).
    return _run_trl(cfg, KTOTrainer, _ref_model)


def run_grpo(cfg: DictConfig) -> dict[str, Any]:
    """Online RL with reward functions and, optionally, a multi-turn tool environment."""
    reward_funcs = reward_funcs_from_cfg(cfg)
    # environment_factory / rollout_func emit TRL's experimental warning on the stable trainer.
    _silence_experimental()
    from trl import GRPOTrainer

    return _run_trl(cfg, GRPOTrainer, _rewards_and_environment(reward_funcs))


def run_rloo(cfg: DictConfig) -> dict[str, Any]:
    """Online RL with the REINFORCE leave-one-out baseline; reward functions only (no environment support)."""
    reward_funcs = reward_funcs_from_cfg(cfg)
    from trl import RLOOTrainer

    return _run_trl(cfg, RLOOTrainer, lambda _cfg: {"reward_funcs": reward_funcs})


def run_gkd(cfg: DictConfig) -> dict[str, Any]:
    """On-policy distillation: student samples, the teacher supervises token-level (GKD)."""
    _silence_experimental()
    from trl.experimental.gkd import GKDTrainer

    return _run_trl(cfg, GKDTrainer, _teacher_model)


def run_distill(cfg: DictConfig) -> dict[str, Any]:
    """Fully on-policy distillation with the stable DistillationTrainer (GKD's successor since TRL 1.10)."""
    from trl import DistillationTrainer

    return _run_trl(cfg, DistillationTrainer, _teacher_and_tools)


def run_gold(cfg: DictConfig) -> dict[str, Any]:
    """On-policy distillation across tokenizers (GOLD: ULD loss aligns teacher and student vocabularies)."""
    _silence_experimental()
    from trl.experimental.gold import GOLDTrainer

    return _run_trl(cfg, GOLDTrainer, _teacher_model)


def run_sdft(cfg: DictConfig) -> dict[str, Any]:
    """Self-distillation FT: the teacher is the model itself, conditioned on a privileged context."""
    _silence_experimental()
    from trl.experimental.sdft import SDFTTrainer

    return _run_trl(cfg, SDFTTrainer)


def run_sdpo(cfg: DictConfig) -> dict[str, Any]:
    """Self-distillation policy optimization: rewards pick successful rollouts; the EMA self distills them."""
    reward_funcs = reward_funcs_from_cfg(cfg)
    _silence_experimental()
    from trl.experimental.sdpo import SDPOTrainer

    return _run_trl(cfg, SDPOTrainer, lambda _cfg: {"reward_funcs": reward_funcs})


def run_ssd(cfg: DictConfig) -> dict[str, Any]:
    """Run simple self-distillation (SSD): SFT on the model's own raw samples, no teacher, no verifier."""
    _silence_experimental()
    from trl.experimental.ssd import SSDTrainer

    return _run_trl(cfg, SSDTrainer)


def run_async_grpo(cfg: DictConfig) -> dict[str, Any]:
    """Asynchronous GRPO: a vLLM server generates while the trainer trains (GPU + vLLM server only)."""
    reward_funcs = reward_funcs_from_cfg(cfg)
    _silence_experimental()
    from trl.experimental.async_grpo import AsyncGRPOTrainer

    return _run_trl(cfg, AsyncGRPOTrainer, _rewards_and_environment(reward_funcs), model_as_id=True)


def run_async_distill(cfg: DictConfig) -> dict[str, Any]:
    """Asynchronous on-policy distillation: vLLM student server + HTTP teacher server(s) (GPU only)."""
    _silence_experimental()
    from trl.experimental.async_distillation import AsyncDistillationTrainer

    return _run_trl(cfg, AsyncDistillationTrainer, model_as_id=True)
