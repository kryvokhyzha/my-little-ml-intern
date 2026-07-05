"""Run the TRL SFT/DPO/GRPO lanes with alert instrumentation and the run-boundary contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.loading import load_split, require_prompt_column

from ..models import load_model, load_ref_model, load_tokenizer, peft_config
from ..runtime import apply_tracking_group, is_main_process, run_with_stderr_tee, smoke_enabled
from ..sampling import resolve_sample_prompts, write_samples
from .config import apply_smoke, build_args, final_train_loss, write_meta
from .rewards import grpo_reward_funcs


def _run_trl(
    cfg: DictConfig,
    config_cls: type,
    trainer_cls: type,
    dpo: bool = False,
    reward_funcs: list[Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    from intern.callbacks import AlertRules, TRLAlertCallback
    from intern.metrics import MetricsLog

    grpo = reward_funcs is not None
    experiment_dir = Path(str(cfg.experiment_dir))
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    smoke = smoke_enabled(cfg)
    main_process = is_main_process()

    # run_start is the verify scope boundary — it must precede any load that can crash.
    mlog = MetricsLog(experiment_dir / "metrics.jsonl")
    if main_process:
        mlog.append_event(
            "run_start",
            task=str(OmegaConf.select(cfg, "trainer.kind")),
            run_name=OmegaConf.select(cfg, "tracking.run_name"),
            smoke=smoke,
        )

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)

    train_dataset = load_split(str(cfg.data.path), str(cfg.data.dataset_split))
    eval_split = OmegaConf.select(cfg, "data.eval_split")
    eval_dataset = load_split(str(cfg.data.path), str(eval_split), for_eval=True) if eval_split else None
    if grpo:
        require_prompt_column(train_dataset, "train")
        if eval_dataset is not None:
            require_prompt_column(eval_dataset, "eval")

    smoke_overrides, train_dataset = apply_smoke({}, train_dataset, smoke)
    if eval_dataset is not None and str(OmegaConf.select(cfg, "trainer.args.eval_strategy") or "no") == "no":
        smoke_overrides.setdefault("eval_strategy", "steps")
        logger.info("eval_split is set but eval_strategy='no' — overriding to 'steps' so eval actually runs")
    apply_tracking_group(cfg)
    args = build_args(cfg, config_cls, **smoke_overrides)

    callback = TRLAlertCallback(mlog, str(cfg.tracking.backend), AlertRules())

    param_count = sum(p.numel() for p in model.parameters())
    if main_process:
        write_meta(mlog, param_count, len(tokenizer), cfg)

    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "callbacks": [callback],
        "peft_config": peft_config(cfg),
    }
    if dpo:
        kwargs["ref_model"] = load_ref_model(cfg)
    if grpo:
        kwargs["reward_funcs"] = reward_funcs
    trainer = trainer_cls(**kwargs)

    if main_process:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        mlog.append_event("meta", key="trainable_param_count", value=trainable)
        if "quantization_config" in cfg.model.main:
            mlog.append_event("meta", key="quantized", value=True)

    lane = "GRPO" if grpo else ("DPO" if dpo else "SFT")
    logger.info("Starting {} run (smoke={}) in {}", lane, smoke, experiment_dir)
    run_with_stderr_tee(trainer.train, experiment_dir)

    if not smoke and main_process:
        # Training succeeded and the checkpoint is saved — a sampling failure (e.g. OOM on a
        # long probe prompt) must not sink the run, so it degrades to a skipped samples.jsonl.
        try:
            configured = OmegaConf.select(cfg, "data.sample_prompts")
            prompts, pre_rendered = resolve_sample_prompts(configured, eval_dataset, train_dataset)
            write_samples(
                trainer.model,
                tokenizer,
                experiment_dir,
                prompts=prompts,
                pre_rendered=pre_rendered,
                seed=int(OmegaConf.select(cfg, "seed") or 42),
            )
        except Exception as exc:
            logger.warning("Sample generation failed ({}) — training succeeded, samples skipped", exc)

    loss = final_train_loss(trainer)
    if main_process:
        print(f"VERDICT: TRAIN_OK | final_train_loss={loss}")
    return {"final_train_loss": loss, "steps": int(trainer.state.global_step), "param_count": param_count}


def run_sft(cfg: DictConfig) -> dict[str, Any]:
    from trl import SFTConfig, SFTTrainer

    return _run_trl(cfg, SFTConfig, SFTTrainer)


def run_dpo(cfg: DictConfig) -> dict[str, Any]:
    from trl import DPOConfig, DPOTrainer

    return _run_trl(cfg, DPOConfig, DPOTrainer, dpo=True)


def run_grpo(cfg: DictConfig) -> dict[str, Any]:
    reward_funcs = grpo_reward_funcs(cfg)
    from trl import GRPOConfig, GRPOTrainer

    return _run_trl(cfg, GRPOConfig, GRPOTrainer, reward_funcs=reward_funcs)
