"""Resolve GRPO reward functions from dotted import paths in the trainer config."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf


def resolve_reward_funcs(paths: list[str]) -> list[Callable[..., Any]]:
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


def grpo_reward_funcs(cfg: DictConfig) -> list[Callable[..., Any]]:
    paths = OmegaConf.select(cfg, "trainer.reward_funcs")
    if not paths:
        raise ValueError("GRPO needs at least one reward function — set trainer.reward_funcs to dotted import paths")
    return resolve_reward_funcs([str(path) for path in paths])
