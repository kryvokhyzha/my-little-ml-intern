"""Map Hydra config onto a TRL *Config object and record run metadata."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from helper.display import is_interactive


if TYPE_CHECKING:
    from datasets import Dataset


# Lanes whose batches carry input_ids, so transformers' num_input_tokens_seen counter works.
DICT_BATCH_TASKS = ("trl_sft", "trl_dpo", "trl_kto", "trl_gkd", "trl_gold")
# TRL 1.14's disable_gradient_checkpointing (trl/models/utils.py) re-enables checkpointing with the raw
# gradient_checkpointing_kwargs. That bypasses the Trainer, which pops these transformers >= 5.17 keys,
# so these lanes crash mid-run ("Unexpected keyword arguments"). Remove when TRL passes them through.
GC_REENABLE_TASKS = ("trl_grpo", "trl_rloo", "trl_dpo", "trl_kto")
TRAINER_ONLY_GC_KWARGS = ("offload", "every_n_layers")


def report_to(backend: str | None) -> str:
    return backend if backend in ("trackio", "wandb") else "none"


def apply_smoke(args_dict: dict[str, Any], dataset: Dataset | None, smoke: bool) -> tuple[dict[str, Any], Any]:
    if not smoke:
        return args_dict, dataset
    args_dict["max_steps"] = 1
    args_dict["save_strategy"] = "no"
    # A 1-step smoke must log its one step, or the verdict prints final_train_loss=None.
    args_dict["logging_steps"] = 1
    if dataset is not None:
        dataset = dataset.select(range(min(32, len(dataset))))
    return args_dict, dataset


def _resolve_bf16(d: dict[str, Any]) -> None:
    """Pin bf16=False where TRL's default would crash; never touch an explicit choice.

    TRL >= 1.7 defaults bf16 to True whenever fp16 is unset, regardless of hardware,
    and transformers then rejects bf16 on machines without support (CPU-only boxes,
    pre-Ampere GPUs). Uses transformers' own capability predicate so the resolution
    can never disagree with their validator. Capable hardware is left to TRL.
    """
    if d.get("bf16") is None and not d.get("fp16") and not d.get("use_cpu"):
        from transformers.utils import is_torch_bf16_gpu_available

        if not is_torch_bf16_gpu_available():
            d["bf16"] = False


def build_args(cfg: DictConfig, **overrides: Any) -> Any:
    """Instantiate the trainer's ``args`` node (``_target_: trl.*Config``) with runtime injection.

    On top of the declared config this layers the values this repo owns: tracking wiring,
    the alert-callback flags, agent-aware tqdm, hardware-aware bf16, and the smoke overrides.
    The config declares the class; this injection is the guardrail library's value-add over a
    bare ``instantiate``.
    """
    from hydra.utils import instantiate

    d: dict[str, Any] = OmegaConf.to_container(cfg.trainer.args, resolve=True)
    d["report_to"] = report_to(OmegaConf.select(cfg, "tracking.backend"))
    run_name = OmegaConf.select(cfg, "tracking.run_name")
    if run_name is not None:
        d["run_name"] = run_name
    project = OmegaConf.select(cfg, "tracking.project")
    if project is not None:
        # transformers' TrackioCallback logs under TrainingArguments.project (default
        # "huggingface") — without this, TRL-lane trackio runs ignore tracking.project.
        d.setdefault("project", str(project))
    space_id = OmegaConf.select(cfg, "tracking.space_id")
    if space_id is not None:
        d["trackio_space_id"] = str(space_id)
        private = OmegaConf.select(cfg, "tracking.private")
        d["hub_private_repo"] = True if private is None else bool(private)
    if d["report_to"] == "trackio":
        # transformers' TrackioCallback otherwise syncs the metrics to a static Space on every Hub push,
        # public unless hub_private_repo=True. An explicit trainer.args value still wins.
        d.setdefault("trackio_static_space_id", False)
    # transformers counts tokens only for dict batches; on the rollout-batch lanes it warns on every
    # micro-step and counts nothing, so those lanes rely on TRL's own `num_tokens` metric instead.
    kind = OmegaConf.select(cfg, "trainer.kind")
    d["include_num_input_tokens_seen"] = True if kind is None or kind in DICT_BATCH_TASKS else "no"
    # Keep NaN losses visible to the alert callback; the default filter logs them as 0.0.
    d["logging_nan_inf_filter"] = False
    d.setdefault("disable_tqdm", not is_interactive())
    if not is_interactive():
        # TRL's DPO/KTO reference-logprob loop reads only TQDM_DISABLE, not args.disable_tqdm.
        os.environ.setdefault("TQDM_DISABLE", "1")
    d.update(overrides)
    bad = sorted(set(TRAINER_ONLY_GC_KWARGS) & set(d.get("gradient_checkpointing_kwargs") or {}))
    if bad and kind in GC_REENABLE_TASKS:
        raise ValueError(f"gradient_checkpointing_kwargs {bad} crash {kind} on TRL 1.14: remove them for this lane")
    _resolve_bf16(d)
    # _convert_="all": nested kwargs (lr_scheduler_kwargs, gradient_checkpointing_kwargs)
    # land as plain dicts, not DictConfig — TrainingArguments' JSON serialization needs that.
    return instantiate(d, _convert_="all")  # d carries `_target_` from the args node


def write_meta(mlog: Any, param_count: int, vocab_size: int, cfg: DictConfig) -> None:
    mlog.append_event("meta", key="task", value=str(OmegaConf.select(cfg, "trainer.kind")))
    mlog.append_event("meta", key="param_count", value=param_count)
    mlog.append_event("meta", key="vocab_size", value=vocab_size)
    for key, path in (("target_params", "model.target_params"), ("planned_tokens", "trainer.planned_tokens")):
        value = OmegaConf.select(cfg, path)
        if value is not None:
            mlog.append_event("meta", key=key, value=value)
    if OmegaConf.select(cfg, "trainer.args.completion_only_loss"):
        mlog.append_event("meta", key="completion_only", value=True)


def final_train_loss(trainer: Any) -> float | None:
    for record in reversed(trainer.state.log_history):
        if "loss" in record:
            return float(record["loss"])
    return None
