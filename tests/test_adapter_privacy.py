import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from training.lightning_adapter import TrackioLightningLogger
from training.models import load_model as _load_model
from training.models import load_ref_model as _load_ref_model
from training.models import load_tokenizer as _load_tokenizer
from training.runtime import is_main_process as _is_main_process
from training.trl.config import build_args as _build_args


def fake_model_factory(repo: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(repo=repo, kwargs=kwargs)


def fake_quant_factory(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(kwargs=kwargs)


def fake_tokenizer_factory(repo: str, pad_token: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(repo=repo, pad_token=pad_token, eos_token="</s>")


def model_cfg(**main_extra: Any) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "main": {"_target_": f"{__name__}.fake_model_factory", "_args_": ["fake/repo"], **main_extra},
                "tokenizer": {
                    "_target_": f"{__name__}.fake_tokenizer_factory",
                    "_args_": ["${model.main._args_[0]}"],
                },
                "ref": None,
                "target_params": None,
            }
        }
    )


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "trainer": {
                "args": {
                    "_target_": "trl.SFTConfig",
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
        cfg.tracking.space_id = "me/trackio-dash"
        args = _build_args(cfg)
        assert args.trackio_space_id == "me/trackio-dash"
        assert args.hub_private_repo is True

    def test_private_false_propagates(self, cfg):
        cfg.tracking.space_id = "me/trackio-dash"
        cfg.tracking.private = False
        args = _build_args(cfg)
        assert args.hub_private_repo is False

    def test_missing_private_key_defaults_true(self, cfg):
        cfg.tracking = OmegaConf.create({"backend": "trackio", "space_id": "me/dash"})
        args = _build_args(cfg)
        assert args.hub_private_repo is True

    def test_no_space_id_leaves_args_untouched(self, cfg):
        args = _build_args(cfg)
        assert args.trackio_space_id is None
        assert args.hub_private_repo is None


class TestLoadModel:
    def test_explicit_dtype_passes_through(self):
        model = _load_model(model_cfg(dtype="bfloat16"))
        assert model.repo == "fake/repo"
        assert model.kwargs["dtype"] == "bfloat16"

    def test_missing_dtype_injects_float32_and_warns(self):
        messages: list[str] = []
        sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            model = _load_model(model_cfg())
        finally:
            logger.remove(sink)
        assert model.kwargs["dtype"] == "float32"
        assert any("float32" in message for message in messages)

    def test_missing_bitsandbytes_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)
        cfg = model_cfg(quantization_config={"_target_": f"{__name__}.fake_quant_factory", "load_in_4bit": True})
        with pytest.raises(ImportError, match="uv sync --group gpu"):
            _load_model(cfg)

    def test_quantized_load_skips_dtype_injection(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
        cfg = model_cfg(quantization_config={"_target_": f"{__name__}.fake_quant_factory", "load_in_4bit": True})
        model = _load_model(cfg)
        assert "dtype" not in model.kwargs
        assert model.kwargs["quantization_config"].kwargs == {"load_in_4bit": True}


class TestLoadRefModel:
    def test_null_ref_returns_none(self):
        assert _load_ref_model(model_cfg()) is None

    def test_ref_instantiated_when_set(self):
        cfg = model_cfg()
        cfg.model.ref = {"_target_": f"{__name__}.fake_model_factory", "_args_": ["ref/repo"], "dtype": "float32"}
        ref = _load_ref_model(cfg)
        assert ref.repo == "ref/repo"
        assert ref.kwargs["dtype"] == "float32"

    def test_ref_shares_dtype_guard(self):
        cfg = model_cfg()
        cfg.model.ref = {"_target_": f"{__name__}.fake_model_factory", "_args_": ["ref/repo"]}
        assert _load_ref_model(cfg).kwargs["dtype"] == "float32"


class TestLoadTokenizer:
    def test_repo_interpolated_and_pad_falls_back_to_eos(self):
        tokenizer = _load_tokenizer(model_cfg())
        assert tokenizer.repo == "fake/repo"
        assert tokenizer.pad_token == "</s>"

    def test_existing_pad_token_kept(self):
        cfg = model_cfg()
        cfg.model.tokenizer.pad_token = "<pad>"
        assert _load_tokenizer(cfg).pad_token == "<pad>"


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
