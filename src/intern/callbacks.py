"""Framework callbacks that stream metrics to MetricsLog and fire training alerts.

Alert messages follow the contract format: ``<metric>=<value> at step <N> — <hypothesis>, try <action>``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger


if TYPE_CHECKING:
    from intern.metrics import MetricsLog

try:  # transformers absent -> TRLAlertCallback unusable, LightningAlertCallback still importable
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:
    _TrainerCallbackBase = object

try:  # lightning absent -> LightningAlertCallback unusable, TRLAlertCallback still importable
    from lightning.pytorch import Callback as _LightningCallbackBase
except ImportError:
    _LightningCallbackBase = object


_LOGURU_LEVELS = {"ERROR": "ERROR", "WARN": "WARNING", "WARNING": "WARNING", "INFO": "INFO", "DEBUG": "DEBUG"}


@dataclass
class AlertRules:
    nan_streak: int = 5
    divergence_factor: float = 3.0
    plateau_evals: int = 5


def fire_alert(backend: str, level: str, message: str) -> None:
    """Send an alert through the tracking backend, falling back to a loguru log line."""
    if backend == "trackio":
        try:
            import trackio

            # trackio.alert signature: (title, text=None, level=AlertLevel.WARN, webhook_url=None)
            trackio.alert(title=message.split(" — ")[0][:120], text=message, level=level)
            return
        except Exception as exc:
            logger.debug("trackio alert failed ({}); falling back to log", exc)
    elif backend == "wandb":
        try:
            import wandb

            wandb.alert(title=level, text=message)
            return
        except Exception as exc:
            logger.debug("wandb alert failed ({}); falling back to log", exc)
    logger.log(_LOGURU_LEVELS.get(level, "INFO"), message)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (bool, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _AlertEngine:
    """Shared alert-rule state machine backing both framework callbacks."""

    def __init__(self, metrics_log: MetricsLog, backend: str, rules: AlertRules) -> None:
        self.metrics_log = metrics_log
        self.backend = backend
        self.rules = rules
        self._nan_streak = 0
        self._best_train_loss = math.inf
        self._best_eval_loss = math.inf
        self._stale_evals = 0

    def alert(self, level: str, message: str) -> None:
        fire_alert(self.backend, level, message)
        self.metrics_log.append_event("alert", level=level, message=message)

    def observe_train_step(self, loss: float | None, step: int, grad_norm: float | None = None) -> bool:
        """Run NaN/instability and divergence rules; return True when training must abort.

        transformers' logging_nan_inf_filter can mask a NaN loss as 0.0, so a
        non-finite grad_norm counts as instability in its own right.
        """
        bad_grad = grad_norm is not None and not math.isfinite(grad_norm)
        bad_loss = loss is not None and math.isnan(loss)
        if bad_grad or bad_loss:
            metric = "grad_norm" if bad_grad else "loss"
            self._nan_streak += 1
            self.alert("ERROR", f"{metric}=nan at step {step} — numerical instability, try skip step + halve lr")
            return self._nan_streak >= self.rules.nan_streak
        if loss is None:
            return False
        self._nan_streak = 0
        if math.isfinite(self._best_train_loss) and loss > self.rules.divergence_factor * self._best_train_loss:
            self.alert("WARN", f"loss={loss:g} at step {step} — lr likely too high, try lr*0.1")
        self._best_train_loss = min(self._best_train_loss, loss)
        return False

    def observe_eval_loss(self, loss: float, step: int, metric: str = "eval_loss") -> None:
        if math.isnan(loss):
            return
        if loss < self._best_eval_loss:
            self._best_eval_loss = loss
            self._stale_evals = 0
            return
        self._stale_evals += 1
        if self._stale_evals >= self.rules.plateau_evals:
            self.alert(
                "INFO",
                f"{metric}={loss:g} at step {step} — eval loss plateaued for {self._stale_evals} evals, "
                "try early stopping or lr decay",
            )
            self._stale_evals = 0


class TRLAlertCallback(_TrainerCallbackBase):
    """transformers/TRL Trainer callback: metrics to MetricsLog + NaN/divergence/plateau alerts."""

    def __init__(self, metrics_log: MetricsLog, backend: str, rules: AlertRules | None = None) -> None:
        super().__init__()
        self._engine = _AlertEngine(metrics_log, backend, rules or AlertRules())

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        # Distributed runs invoke callbacks on every rank; only rank zero may write metrics.jsonl.
        if state is not None and not getattr(state, "is_world_process_zero", True):
            return control
        if not logs:
            return control
        step = int(state.global_step)
        for name, raw in logs.items():
            value = _as_float(raw)
            if value is not None:
                self._engine.metrics_log.append_metric(step, name, value, split="train")
        loss = _as_float(logs.get("loss"))
        grad_norm = _as_float(logs.get("grad_norm"))
        if self._engine.observe_train_step(loss, step, grad_norm=grad_norm):
            if control is not None:
                control.should_training_stop = True
            else:
                raise RuntimeError(f"{self._engine.rules.nan_streak} consecutive unstable steps — aborting run")
        return control

    def on_evaluate(
        self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        if state is not None and not getattr(state, "is_world_process_zero", True):
            return control
        if not metrics:
            return control
        step = int(state.global_step)
        for name, raw in metrics.items():
            value = _as_float(raw)
            if value is not None:
                self._engine.metrics_log.append_metric(step, name, value, split="eval")
        eval_loss = _as_float(metrics.get("eval_loss"))
        if eval_loss is not None:
            self._engine.observe_eval_loss(eval_loss, step, metric="eval_loss")
        return control


class LightningAlertCallback(_LightningCallbackBase):
    """Lightning callback: metrics to MetricsLog + NaN/divergence/plateau alerts."""

    def __init__(self, metrics_log: MetricsLog, backend: str, rules: AlertRules | None = None) -> None:
        super().__init__()
        self._engine = _AlertEngine(metrics_log, backend, rules or AlertRules())

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any = None, batch: Any = None, batch_idx: int = 0
    ) -> None:
        metrics = trainer.callback_metrics
        loss = _as_float(metrics.get("loss", metrics.get("train_loss")))
        if loss is None:
            return
        step = int(trainer.global_step)
        self._engine.metrics_log.append_metric(step, "loss", loss, split="train")
        if self._engine.observe_train_step(loss, step):
            trainer.should_stop = True

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = trainer.callback_metrics
        step = int(trainer.global_step)
        for name in metrics:
            if not name.startswith("val"):
                continue
            value = _as_float(metrics[name])
            if value is not None:
                self._engine.metrics_log.append_metric(step, name, value, split="eval")
        val_loss = _as_float(metrics.get("val_loss"))
        if val_loss is not None:
            self._engine.observe_eval_loss(val_loss, step, metric="val_loss")
