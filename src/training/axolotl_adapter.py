"""Axolotl lane: render a launchable axolotl YAML; axolotl itself is never a local dependency."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def render(cfg: DictConfig) -> Path:
    base_config = OmegaConf.select(cfg, "trainer.base_config")
    base = OmegaConf.load(str(base_config)) if base_config else OmegaConf.create({})

    overrides = OmegaConf.select(cfg, "trainer.overrides")
    # Resolve against the full cfg before merging: the standalone merge result has no parent to resolve from.
    resolved = OmegaConf.to_container(overrides, resolve=True) if overrides else {}
    merged = OmegaConf.merge(base, OmegaConf.create(resolved))

    rendered_path = Path(str(cfg.trainer.rendered_path))
    rendered_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged, rendered_path)

    logger.info("Rendered axolotl config to {}", rendered_path)
    print(f"uv run --with axolotl axolotl train {rendered_path}")
    return rendered_path
