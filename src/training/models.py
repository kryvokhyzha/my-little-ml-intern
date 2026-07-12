"""Model loading: instantiate the `cfg.model` group nodes with dtype/quantization guards, plus PEFT/LoRA config."""

from __future__ import annotations

from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _instantiate_model(node: DictConfig) -> Any:
    from hydra.utils import instantiate

    container: dict[str, Any] = OmegaConf.to_container(node, resolve=True)
    if "quantization_config" in container:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "model config sets quantization_config but bitsandbytes is not installed — run 'uv sync --group gpu'"
            ) from exc
    elif "dtype" not in container:
        # transformers v5 loads checkpoints in their stored dtype (often fp16) by default;
        # full-precision fp16 training diverges, so default to explicit float32.
        logger.warning("model config sets neither dtype nor quantization_config — injecting dtype=float32")
        container["dtype"] = "float32"
    return instantiate(container)


def load_model(cfg: DictConfig) -> Any:
    return _instantiate_model(cfg.model.main)


def load_ref_model(cfg: DictConfig) -> Any | None:
    ref = OmegaConf.select(cfg, "model.ref")
    return _instantiate_model(ref) if ref is not None else None


def load_teacher_model(cfg: DictConfig) -> Any:
    """Instantiate the `model.teacher` node (on-policy distillation); the GKD lane requires it."""
    teacher = OmegaConf.select(cfg, "model.teacher")
    if teacher is None:
        raise ValueError(
            "trainer.kind=trl_gkd requires a `model.teacher` node (same shape as `model.main`; "
            "teacher and student must share a tokenizer)"
        )
    return _instantiate_model(teacher)


def load_tokenizer(cfg: DictConfig) -> Any:
    from hydra.utils import instantiate

    tokenizer = instantiate(OmegaConf.to_container(cfg.model.tokenizer, resolve=True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def peft_config(cfg: DictConfig) -> Any:
    peft = OmegaConf.select(cfg, "trainer.peft")
    if not peft:
        return None
    from hydra.utils import instantiate

    # `_target_: peft.LoraConfig` node — same instantiate path as the model group.
    # _convert_="all" keeps target_modules / nested kwargs as plain list/dict.
    return instantiate(OmegaConf.to_container(peft, resolve=True), _convert_="all")
