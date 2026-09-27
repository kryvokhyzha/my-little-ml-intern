"""Resolve reward functions, tools, and environments from dotted import paths in the trainer config."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf


def resolve_callable(path: str, what: str = "reward function") -> Callable[..., Any]:
    """Import ``package.module:attr`` (or ``package.module.attr``) and return the callable."""
    module_name, _, attr = path.partition(":") if ":" in path else path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"{what} path {path!r} must look like 'package.module:function' or 'package.module.function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Cannot import module {module_name!r} for {what} {path!r}: {exc}") from exc
    target: Any = module
    for part in attr.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError(f"Module {module_name!r} has no attribute {attr!r} ({what} {path!r})")
    if not callable(target):
        raise ValueError(f"{what} {path!r} resolved to non-callable {type(target).__name__}")
    return target


def resolve_reward_funcs(paths: list[str]) -> list[Callable[..., Any]]:
    return [resolve_callable(path, "reward function") for path in paths]


def has_environment(cfg: DictConfig) -> bool:
    """Report whether the trainer config supplies an environment (a factory path or an env_spec node)."""
    return bool(OmegaConf.select(cfg, "trainer.environment_factory") or OmegaConf.select(cfg, "trainer.env_spec"))


def reward_funcs_from_cfg(cfg: DictConfig) -> list[Callable[..., Any]]:
    """Resolve ``trainer.reward_funcs`` (GRPO, RLOO, SDPO, async GRPO); empty only when an environment is configured.

    An environment can own the reward through ``get_reward()`` (TRL adds it as a reward source), so
    the empty-list guard defers to TRL's own "No reward source provided" check in that case.
    """
    paths = OmegaConf.select(cfg, "trainer.reward_funcs")
    if not paths and not has_environment(cfg):
        kind = OmegaConf.select(cfg, "trainer.kind") or "this lane"
        raise ValueError(
            f"{kind} needs at least one reward function — set trainer.reward_funcs to dotted import paths, "
            "or (GRPO lanes) set trainer.environment_factory to an environment that defines get_reward()"
        )
    return resolve_reward_funcs([str(path) for path in paths or []])


def environment_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Build the ``tools`` / ``environment_factory`` trainer kwargs (and an env_spec's dataset and rewards).

    ``trainer.environment_factory`` is a dotted path to a class or zero-argument callable; TRL calls it
    once per concurrent rollout and exposes the instance's public methods as tools. ``trainer.tools``
    lists dotted paths to standalone tool functions. ``trainer.env_spec`` is an instantiate node for a
    spec object (``trl.experimental.harbor.HarborSpec``, ``trl.experimental.openreward.OpenRewardSpec``)
    that carries ``train_dataset``, ``environment_factory``, and ``reward_funcs`` together.
    """
    kwargs: dict[str, Any] = {}
    tools = OmegaConf.select(cfg, "trainer.tools")
    if tools:
        kwargs["tools"] = [resolve_callable(str(path), "tool") for path in tools]
    factory = OmegaConf.select(cfg, "trainer.environment_factory")
    spec_node = OmegaConf.select(cfg, "trainer.env_spec")
    if factory and spec_node:
        raise ValueError("Set trainer.environment_factory OR trainer.env_spec, not both")
    if factory:
        kwargs["environment_factory"] = resolve_callable(str(factory), "environment factory")
    if spec_node:
        from hydra.utils import instantiate

        spec = instantiate(OmegaConf.to_container(spec_node, resolve=True), _convert_="all")
        kwargs["environment_factory"] = spec.environment_factory
        kwargs["train_dataset"] = spec.train_dataset
        # HarborSpec / OpenRewardSpec return ONE callable (a property), not a list.
        spec_rewards = getattr(spec, "reward_funcs", None)
        if callable(spec_rewards):
            spec_rewards = [spec_rewards]
        if spec_rewards:
            kwargs["reward_funcs"] = list(spec_rewards)
    return kwargs
