"""Model loading: dtype resolution, optional 4-bit quantization, and PEFT/LoRA config."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_model(name: str, dtype_str: str | None, quantization: dict[str, Any] | None = None) -> Any:
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


def peft_config(cfg: DictConfig) -> Any:
    peft = OmegaConf.select(cfg, "trainer.peft")
    if not peft:
        return None
    from peft import LoraConfig

    return LoraConfig(**OmegaConf.to_container(peft, resolve=True))
