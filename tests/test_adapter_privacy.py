import sys

import pytest
from omegaconf import OmegaConf

from training.lightning_adapter import TrackioLightningLogger
from training.models import load_model as _load_model
from training.runtime import is_main_process as _is_main_process
from training.trl.config import build_args as _build_args


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "trainer": {
                "args": {
                    "output_dir": str(tmp_path / "ckpts"),
                    "max_steps": 5,
                    "per_device_train_batch_size": 1,
                    "logging_steps": 1,
                    "learning_rate": 2.0e-5,
                    "seed": 42,
                }
            },
            "tracking": {"backend": "trackio", "run_name": "001-test", "space_id": None, "private": True},
        }
    )


class TestBuildArgsSpacePrivacy:
    def test_space_id_sets_trackio_space_and_private_repo(self, cfg):
        from trl import SFTConfig

        cfg.tracking.space_id = "me/trackio-dash"
        args = _build_args(cfg, SFTConfig)
        assert args.trackio_space_id == "me/trackio-dash"
        assert args.hub_private_repo is True

    def test_private_false_propagates(self, cfg):
        from trl import SFTConfig

        cfg.tracking.space_id = "me/trackio-dash"
        cfg.tracking.private = False
        args = _build_args(cfg, SFTConfig)
        assert args.hub_private_repo is False

    def test_missing_private_key_defaults_true(self, cfg):
        from trl import SFTConfig

        cfg.tracking = OmegaConf.create({"backend": "trackio", "space_id": "me/dash"})
        args = _build_args(cfg, SFTConfig)
        assert args.hub_private_repo is True

    def test_no_space_id_leaves_args_untouched(self, cfg):
        from trl import SFTConfig

        args = _build_args(cfg, SFTConfig)
        assert args.trackio_space_id is None
        assert args.hub_private_repo is None


class TestLoadModelQuantization:
    def test_missing_bitsandbytes_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)
        with pytest.raises(ImportError, match="uv sync --group gpu"):
            _load_model("gpt2", "float32", {"load_in_4bit": True})


class TestIsMainProcess:
    def test_rank_unset_is_main(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        assert _is_main_process() is True

    def test_rank_zero_is_main(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        assert _is_main_process() is True

    def test_rank_one_is_not_main(self, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        assert _is_main_process() is False


class TestTrackioLightningLoggerPrivacy:
    def test_space_id_and_private_forwarded_to_init(self, monkeypatch):
        import trackio

        calls = []
        monkeypatch.setattr(trackio, "init", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(trackio, "log", lambda metrics, step=None: None)

        lightning_logger = TrackioLightningLogger(project="proj", run_name="run-1", space_id="me/dash", private=False)
        lightning_logger.log_metrics({"loss": 1.0}, step=1)

        assert calls == [{"project": "proj", "name": "run-1", "space_id": "me/dash", "private": False}]

    def test_private_defaults_true_when_space_id_set(self, monkeypatch):
        import trackio

        calls = []
        monkeypatch.setattr(trackio, "init", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(trackio, "log", lambda metrics, step=None: None)

        lightning_logger = TrackioLightningLogger(project="proj", run_name="run-1", space_id="me/dash")
        lightning_logger.log_metrics({"loss": 1.0}, step=1)

        assert calls[0]["private"] is True

    def test_no_space_id_omits_privacy_kwargs(self, monkeypatch):
        import trackio

        calls = []
        monkeypatch.setattr(trackio, "init", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(trackio, "log", lambda metrics, step=None: None)

        lightning_logger = TrackioLightningLogger(project="proj", run_name="run-1")
        lightning_logger.log_metrics({"loss": 1.0}, step=1)

        assert calls == [{"project": "proj", "name": "run-1"}]
