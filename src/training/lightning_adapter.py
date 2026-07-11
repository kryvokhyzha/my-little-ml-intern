"""Lightning lane: instantiate module/datamodule from Hydra config and run with alert instrumentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import hydra
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from helper.display import is_interactive
from training.runtime import run_with_stderr_tee, smoke_enabled


class TrackioLightningLogger(Logger):
    """Minimal Lightning logger forwarding metrics to trackio."""

    def __init__(
        self,
        project: str | None = None,
        run_name: str | None = None,
        space_id: str | None = None,
        private: bool | None = None,
        group: str | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._run_name = run_name
        self._space_id = space_id
        self._private = private
        self._group = group
        self._initialized = False

    @property
    def name(self) -> str:
        return self._run_name or "trackio"

    @property
    def version(self) -> str:
        return "0"

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        pass

    # Lightning's base Logger does NOT rank-gate; built-in loggers decorate log_metrics themselves.
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        try:
            import trackio

            if not self._initialized:
                init_kwargs: dict[str, Any] = {"project": self._project or "lightning", "name": self._run_name}
                if self._group is not None:
                    # trackio 0.28 init supports the group kwarg.
                    init_kwargs["group"] = str(self._group)
                if self._space_id is not None:
                    # trackio 0.28 init supports space_id and private kwargs.
                    init_kwargs["space_id"] = str(self._space_id)
                    init_kwargs["private"] = True if self._private is None else bool(self._private)
                trackio.init(**init_kwargs)
                self._initialized = True
            trackio.log(dict(metrics), step=step)
        except Exception as exc:
            logger.warning("trackio logging failed: {}", exc)


def _build_logger(cfg: DictConfig) -> Logger | bool:
    backend = OmegaConf.select(cfg, "tracking.backend")
    project = OmegaConf.select(cfg, "tracking.project")
    run_name = OmegaConf.select(cfg, "tracking.run_name")
    group = OmegaConf.select(cfg, "tracking.group")
    if backend == "trackio":
        return TrackioLightningLogger(
            project=project,
            run_name=run_name,
            space_id=OmegaConf.select(cfg, "tracking.space_id"),
            private=OmegaConf.select(cfg, "tracking.private"),
            group=group,
        )
    if backend == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        kwargs: dict[str, Any] = {"project": project, "name": run_name}
        if group is not None:
            # WandbLogger forwards extra kwargs to wandb.init.
            kwargs["group"] = str(group)
        return WandbLogger(**kwargs)
    return False


def _final_train_loss(trainer: Any) -> float | None:
    for key in ("train_loss", "train/loss", "loss"):
        value = trainer.callback_metrics.get(key)
        if value is not None:
            return float(value)
    return None


def run(cfg: DictConfig) -> dict[str, Any]:
    from intern.callbacks import AlertRules, LightningAlertCallback
    from intern.metrics import MetricsLog

    experiment_dir = Path(str(cfg.experiment_dir))
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    smoke = smoke_enabled(cfg)

    mlog = MetricsLog(experiment_dir / "metrics.jsonl")
    mlog.append_event("run_start", task="lightning", run_name=OmegaConf.select(cfg, "tracking.run_name"), smoke=smoke)

    module = hydra.utils.instantiate(cfg.trainer.module)
    datamodule = hydra.utils.instantiate(cfg.trainer.datamodule)

    args: dict[str, Any] = OmegaConf.to_container(
        cfg.trainer.args, resolve=True
    )  # `_target_: lightning.pytorch.Trainer`
    args.setdefault("enable_progress_bar", is_interactive())
    if smoke:
        # val_check_interval must not exceed the truncated train-batch count, or the
        # Trainer refuses to start; disable sanity val for the 1-step smoke too.
        args.update(
            max_steps=1,
            limit_train_batches=2,
            limit_val_batches=1,
            enable_checkpointing=False,
            val_check_interval=1.0,
            num_sanity_val_steps=0,
        )

    callback = LightningAlertCallback(mlog, str(cfg.tracking.backend), AlertRules())
    # instantiate the Trainer node with the objects the config can't hold (logger/callbacks) —
    # the lightning-lane mirror of build_args for the TRL lane. _convert_="all" → plain dicts.
    trainer = hydra.utils.instantiate(args, logger=_build_logger(cfg), callbacks=[callback], _convert_="all")

    param_count = sum(p.numel() for p in module.parameters())
    mlog.append_event("meta", key="task", value="lightning")
    mlog.append_event("meta", key="param_count", value=param_count)
    vocab_size = getattr(module, "vocab_size", None)
    if vocab_size is not None:
        mlog.append_event("meta", key="vocab_size", value=int(vocab_size))
    for key in ("target_params", "planned_tokens"):
        value = OmegaConf.select(cfg, f"trainer.{key}")
        if value is not None:
            mlog.append_event("meta", key=key, value=value)

    logger.info("Starting Lightning run (smoke={}) in {}", smoke, experiment_dir)
    run_with_stderr_tee(lambda: trainer.fit(module, datamodule=datamodule), experiment_dir)

    final_train_loss = _final_train_loss(trainer)
    print(f"VERDICT: TRAIN_OK | final_train_loss={final_train_loss}")
    return {
        "final_train_loss": final_train_loss,
        "steps": int(trainer.global_step),
        "param_count": param_count,
    }
