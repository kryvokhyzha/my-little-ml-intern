"""Resolve a model's true parameter count for the budget scale-ceiling gate.

The count comes from HF safetensors metadata (no weight download), and is the TRUE
parameter count — unlike a quantized load's `sum(p.numel())`, which under-counts 4-bit
packed weights.
"""

from __future__ import annotations

from pathlib import Path


def hf_param_count(repo_id: str) -> int | None:
    """Total parameters from the Hub's safetensors metadata; None if unavailable."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id, expand=["safetensors"])
    except Exception:
        return None
    safetensors = getattr(info, "safetensors", None)
    total = getattr(safetensors, "total", None) if safetensors else None
    return int(total) if total else None


def experiment_model_repo(configs_dir: Path, experiment_name: str) -> str | None:
    """Return the model repo id an experiment composes, from its `model` group pick.

    Reads `configs/<experiment>.yaml`, honours a `_self_` override of `model.main._args_`,
    else follows the `defaults` `model:` pick to `configs/model/<pick>.yaml`.
    """
    import yaml

    config_path = configs_dir / f"{experiment_name}.yaml"
    if not config_path.is_file():
        return None
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    self_args = (((data.get("model") or {}).get("main")) or {}).get("_args_")
    if self_args:
        return str(self_args[0])

    pick = None
    for entry in data.get("defaults", []):
        if isinstance(entry, dict) and "model" in entry:
            pick = entry["model"]
    if not pick:
        return None
    model_path = configs_dir / "model" / f"{pick}.yaml"
    if not model_path.is_file():
        return None
    model_data = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}
    args = ((model_data.get("main")) or {}).get("_args_")
    return str(args[0]) if args else None
