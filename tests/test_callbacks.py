import math
import re
import sys
import types
from types import SimpleNamespace

from loguru import logger

from intern.callbacks import AlertRules, LightningAlertCallback, TRLAlertCallback, fire_alert


ALERT_RE = re.compile(r"^[a-z_/]+=\S+ at step \d+ — .+, try .+$")
NAN = float("nan")


class FakeMetricsLog:
    def __init__(self):
        self.metrics = []
        self.events = []

    def append_metric(self, step, name, value, split="train"):
        self.metrics.append({"step": step, "name": name, "value": value, "split": split})

    def append_event(self, event, **fields):
        self.events.append({"event": event, **fields})

    def alerts(self):
        return [e for e in self.events if e["event"] == "alert"]


def _state(step):
    return SimpleNamespace(global_step=step)


def _control():
    return SimpleNamespace(should_training_stop=False)


def _trl(rules=None):
    log = FakeMetricsLog()
    return log, TRLAlertCallback(log, backend="none", rules=rules)


def _lightning(rules=None):
    log = FakeMetricsLog()
    return log, LightningAlertCallback(log, backend="none", rules=rules)


def _trainer(metrics, step=0):
    return SimpleNamespace(callback_metrics=metrics, global_step=step, should_stop=False)


def test_alert_rules_defaults():
    rules = AlertRules()
    assert rules.nan_streak == 5
    assert rules.divergence_factor == 3.0
    assert rules.plateau_evals == 5


def test_trl_on_log_writes_numeric_metrics():
    log, cb = _trl()
    logs = {"loss": 2.5, "learning_rate": 3e-4, "epoch": 1, "run_name": "x", "flag": True}
    cb.on_log(None, _state(10), _control(), logs=logs)
    assert {m["name"] for m in log.metrics} == {"loss", "learning_rate", "epoch"}
    assert all(m["step"] == 10 and m["split"] == "train" for m in log.metrics)
    assert not log.alerts()


def test_trl_on_evaluate_writes_eval_metrics():
    log, cb = _trl()
    cb.on_evaluate(None, _state(100), _control(), metrics={"eval_loss": 2.0, "eval_runtime": 3.2})
    assert {m["name"] for m in log.metrics} == {"eval_loss", "eval_runtime"}
    assert all(m["step"] == 100 and m["split"] == "eval" for m in log.metrics)


def test_trl_nan_alerts_and_aborts_after_streak():
    log, cb = _trl()
    control = _control()
    for step in range(1, 5):
        cb.on_log(None, _state(step), control, logs={"loss": NAN})
        assert control.should_training_stop is False
    cb.on_log(None, _state(5), control, logs={"loss": NAN})
    assert control.should_training_stop is True
    alerts = log.alerts()
    assert len(alerts) == 5
    assert all(a["level"] == "ERROR" for a in alerts)
    assert all(a["message"].startswith("loss=nan at step ") for a in alerts)
    assert all(ALERT_RE.match(a["message"]) for a in alerts)


def test_trl_nan_streak_resets_on_finite_loss():
    log, cb = _trl(rules=AlertRules(nan_streak=3))
    control = _control()
    for step, loss in enumerate([NAN, NAN, 2.0, NAN, NAN], start=1):
        cb.on_log(None, _state(step), control, logs={"loss": loss})
    assert control.should_training_stop is False
    cb.on_log(None, _state(6), control, logs={"loss": NAN})
    assert control.should_training_stop is True


def test_trl_divergence_fires_warn():
    log, cb = _trl()
    control = _control()
    cb.on_log(None, _state(10), control, logs={"loss": 2.0})
    cb.on_log(None, _state(20), control, logs={"loss": 5.9})  # below 3.0 * 2.0
    assert not log.alerts()
    cb.on_log(None, _state(30), control, logs={"loss": 6.1})
    alerts = log.alerts()
    assert len(alerts) == 1
    assert alerts[0]["level"] == "WARN"
    assert "try lr*0.1" in alerts[0]["message"]
    assert ALERT_RE.match(alerts[0]["message"])
    assert control.should_training_stop is False


def test_trl_plateau_fires_info_after_n_evals():
    log, cb = _trl()
    control = _control()
    cb.on_evaluate(None, _state(100), control, metrics={"eval_loss": 2.0})
    for i in range(1, 5):
        cb.on_evaluate(None, _state(100 + i), control, metrics={"eval_loss": 2.05})
        assert not log.alerts()
    cb.on_evaluate(None, _state(105), control, metrics={"eval_loss": 2.05})
    alerts = log.alerts()
    assert len(alerts) == 1
    assert alerts[0]["level"] == "INFO"
    assert alerts[0]["message"].startswith("eval_loss=")
    assert ALERT_RE.match(alerts[0]["message"])


def test_trl_eval_improvement_resets_plateau():
    log, cb = _trl(rules=AlertRules(plateau_evals=2))
    control = _control()
    for step, loss in enumerate([2.0, 2.1, 1.9, 1.95], start=1):
        cb.on_evaluate(None, _state(step), control, metrics={"eval_loss": loss})
    assert not log.alerts()  # improvement at step 3 reset the stale counter
    cb.on_evaluate(None, _state(5), control, metrics={"eval_loss": 1.95})
    assert len(log.alerts()) == 1


def test_lightning_train_batch_end_writes_loss():
    log, cb = _lightning()
    cb.on_train_batch_end(_trainer({"loss": 2.25}, step=7), None)
    assert log.metrics == [{"step": 7, "name": "loss", "value": 2.25, "split": "train"}]


def test_lightning_train_loss_fallback_key():
    log, cb = _lightning()
    cb.on_train_batch_end(_trainer({"train_loss": 1.5}, step=3), None)
    assert log.metrics == [{"step": 3, "name": "loss", "value": 1.5, "split": "train"}]


def test_lightning_nan_streak_sets_should_stop():
    log, cb = _lightning()
    trainer = _trainer({"loss": NAN}, step=1)
    for step in range(1, 5):
        trainer.global_step = step
        cb.on_train_batch_end(trainer, None)
        assert trainer.should_stop is False
    trainer.global_step = 5
    cb.on_train_batch_end(trainer, None)
    assert trainer.should_stop is True
    alerts = log.alerts()
    assert len(alerts) == 5
    assert all(a["level"] == "ERROR" and ALERT_RE.match(a["message"]) for a in alerts)


def test_lightning_divergence_fires_warn():
    log, cb = _lightning()
    cb.on_train_batch_end(_trainer({"loss": 2.0}, step=1), None)
    cb.on_train_batch_end(_trainer({"loss": 7.0}, step=2), None)
    alerts = log.alerts()
    assert len(alerts) == 1
    assert alerts[0]["level"] == "WARN"
    assert ALERT_RE.match(alerts[0]["message"])


def test_lightning_validation_end_writes_val_metrics_and_plateaus():
    log, cb = _lightning()
    cb.on_validation_end(_trainer({"val_loss": 2.0, "val_acc": 0.5, "loss": 2.2}, step=10), None)
    eval_metrics = [m for m in log.metrics if m["split"] == "eval"]
    assert {m["name"] for m in eval_metrics} == {"val_loss", "val_acc"}
    for i in range(1, 6):
        cb.on_validation_end(_trainer({"val_loss": 2.3}, step=10 + i), None)
    alerts = log.alerts()
    assert len(alerts) == 1
    assert alerts[0]["level"] == "INFO"
    assert alerts[0]["message"].startswith("val_loss=")
    assert ALERT_RE.match(alerts[0]["message"])


def test_fire_alert_trackio(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "trackio", types.SimpleNamespace(alert=lambda **kw: captured.update(kw)))
    fire_alert("trackio", "ERROR", "loss=nan at step 3 — numerical instability, try skip step + halve lr")
    assert captured == {
        "title": "loss=nan at step 3",
        "text": "loss=nan at step 3 — numerical instability, try skip step + halve lr",
        "level": "ERROR",
    }


def test_grad_norm_nan_counts_as_instability():
    log, callback = _trl()
    callback.on_log(None, _state(7), _control(), logs={"loss": 0.0, "grad_norm": float("nan")})

    alerts = log.alerts()
    assert len(alerts) == 1
    assert alerts[0]["level"] == "ERROR"
    assert alerts[0]["message"].startswith("grad_norm=nan at step 7")


def test_grad_norm_nan_streak_stops_training():
    log, callback = _trl(AlertRules(nan_streak=2))
    control = _control()
    callback.on_log(None, _state(1), control, logs={"loss": 0.0, "grad_norm": float("nan")})
    assert not control.should_training_stop
    callback.on_log(None, _state(2), control, logs={"loss": 0.0, "grad_norm": float("inf")})
    assert control.should_training_stop


def test_fire_alert_wandb(monkeypatch):
    captured = {}

    def fake_alert(title, text):
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(alert=fake_alert))
    fire_alert("wandb", "WARN", "loss=9.8 at step 120 — lr likely too high, try lr*0.1")
    assert captured == {"title": "WARN", "text": "loss=9.8 at step 120 — lr likely too high, try lr*0.1"}


def test_fire_alert_trackio_falls_back_to_loguru(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("no active run")

    monkeypatch.setitem(sys.modules, "trackio", types.SimpleNamespace(alert=boom))
    records = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="DEBUG")
    try:
        fire_alert("trackio", "WARN", "loss=9.8 at step 120 — lr likely too high, try lr*0.1")
    finally:
        logger.remove(sink_id)
    assert any("try lr*0.1" in r for r in records)


def test_fire_alert_unknown_backend_logs(monkeypatch):
    records = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="INFO")
    try:
        fire_alert("none", "WARN", "loss=9.8 at step 120 — lr likely too high, try lr*0.1")
        fire_alert("none", "INFO", "eval_loss=2.1 at step 5 — plateau, try early stopping")
    finally:
        logger.remove(sink_id)
    assert sum("try lr*0.1" in r for r in records) == 1
    assert sum("try early stopping" in r for r in records) == 1


def test_nan_value_written_to_metrics_log_is_nan():
    log, cb = _trl(rules=AlertRules(nan_streak=99))
    cb.on_log(None, _state(1), _control(), logs={"loss": NAN})
    assert math.isnan(log.metrics[0]["value"])


def test_trl_on_log_noops_on_non_main_rank():
    log, cb = _trl()
    state = SimpleNamespace(global_step=5, is_world_process_zero=False)
    control = _control()
    result = cb.on_log(None, state, control, logs={"loss": NAN})
    assert result is control
    assert log.metrics == [] and log.events == []
    assert control.should_training_stop is False


def test_trl_on_evaluate_noops_on_non_main_rank():
    log, cb = _trl()
    state = SimpleNamespace(global_step=5, is_world_process_zero=False)
    cb.on_evaluate(None, state, _control(), metrics={"eval_loss": 2.0})
    assert log.metrics == [] and log.events == []


def test_trl_state_without_rank_attr_still_logs():
    log, cb = _trl()
    cb.on_log(None, _state(3), _control(), logs={"loss": 1.5})
    assert log.metrics == [{"step": 3, "name": "loss", "value": 1.5, "split": "train"}]
