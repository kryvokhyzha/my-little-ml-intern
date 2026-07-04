import math
import os

import pytest
import yaml
from datasets import Dataset
from omegaconf import OmegaConf

from data.loading import require_prompt_column as _require_prompt_column
from training import axolotl_adapter
from training.lightning_adapter import TrackioLightningLogger, _build_logger
from training.runtime import apply_tracking_group as _apply_tracking_group
from training.runtime import smoke_enabled as _smoke_enabled
from training.trl.config import apply_smoke as _apply_smoke
from training.trl.config import build_args as _build_args
from training.trl.config import report_to as _report_to
from training.trl.rewards import grpo_reward_funcs as _grpo_reward_funcs
from training.trl.rewards import resolve_reward_funcs as _resolve_reward_funcs


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_dict({"text": [f"row {i}" for i in range(64)]})


class TestReportTo:
    @pytest.mark.parametrize(
        ("backend", "expected"),
        [("trackio", "trackio"), ("wandb", "wandb"), ("none", "none"), ("mlflow", "none"), (None, "none")],
    )
    def test_mapping(self, backend, expected):
        assert _report_to(backend) == expected


class TestApplySmoke:
    def test_smoke_slices_and_overrides(self, dataset):
        args, sliced = _apply_smoke({}, dataset, smoke=True)
        assert args == {"max_steps": 1, "save_strategy": "no"}
        assert len(sliced) == 32
        assert sliced[0] == dataset[0]

    def test_smoke_keeps_small_dataset(self):
        small = Dataset.from_dict({"text": ["a", "b", "c"]})
        _, sliced = _apply_smoke({}, small, smoke=True)
        assert len(sliced) == 3

    def test_no_smoke_is_identity(self, dataset):
        args, same = _apply_smoke({"max_steps": 500}, dataset, smoke=False)
        assert args == {"max_steps": 500}
        assert same is dataset

    def test_none_dataset(self):
        args, ds = _apply_smoke({}, None, smoke=True)
        assert args["max_steps"] == 1
        assert ds is None


class TestSmokeEnabled:
    def test_cfg_flag(self):
        assert _smoke_enabled(OmegaConf.create({"smoke_test": True}))
        assert not _smoke_enabled(OmegaConf.create({"smoke_test": False}))

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("SMOKE_TEST", "1")
        assert _smoke_enabled(OmegaConf.create({"smoke_test": False}))


class TestBuildArgs:
    @pytest.fixture
    def cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "trainer": {
                    "args": {
                        "output_dir": str(tmp_path / "ckpts"),
                        "max_steps": 5,
                        "per_device_train_batch_size": 1,
                        "logging_steps": 1,
                        "learning_rate": 2.0e-5,
                        "eval_strategy": "no",
                        "seed": 42,
                    }
                },
                "tracking": {"backend": "trackio", "run_name": "001-test"},
            }
        )

    def test_injection(self, cfg):
        from trl import SFTConfig

        args = _build_args(cfg, SFTConfig)
        assert args.report_to == ["trackio"]
        assert args.run_name == "001-test"
        assert bool(args.include_num_input_tokens_seen)
        assert args.max_steps == 5
        assert args.seed == 42

    def test_overrides_win(self, cfg):
        from trl import SFTConfig

        args = _build_args(cfg, SFTConfig, max_steps=1, save_strategy="no")
        assert args.max_steps == 1
        assert args.save_strategy == "no"

    def test_none_backend_without_run_name(self, cfg):
        from trl import SFTConfig

        cfg.tracking = OmegaConf.create({"backend": "none"})
        args = _build_args(cfg, SFTConfig)
        assert args.report_to == []


class TestResolveRewardFuncs:
    def test_colon_form(self):
        assert _resolve_reward_funcs(["math:sqrt"]) == [math.sqrt]

    def test_dotted_form(self):
        assert _resolve_reward_funcs(["math.sqrt"]) == [math.sqrt]

    def test_missing_module(self):
        with pytest.raises(ValueError, match="Cannot import module"):
            _resolve_reward_funcs(["definitely_no_such_module:reward"])

    def test_missing_attribute(self):
        with pytest.raises(ValueError, match="no attribute"):
            _resolve_reward_funcs(["math:no_such_fn"])

    def test_bare_name_rejected(self):
        with pytest.raises(ValueError, match="package.module"):
            _resolve_reward_funcs(["math"])

    def test_non_callable_rejected(self):
        with pytest.raises(ValueError, match="non-callable"):
            _resolve_reward_funcs(["math:pi"])


class TestGrpoRewardFuncs:
    @pytest.mark.parametrize("reward_funcs", [[], None])
    def test_empty_raises(self, reward_funcs):
        cfg = OmegaConf.create({"trainer": {"reward_funcs": reward_funcs}})
        with pytest.raises(ValueError, match="at least one reward function"):
            _grpo_reward_funcs(cfg)

    def test_resolves_paths(self):
        cfg = OmegaConf.create({"trainer": {"reward_funcs": ["math:sqrt", "math.dist"]}})
        assert _grpo_reward_funcs(cfg) == [math.sqrt, math.dist]


class TestRequirePromptColumn:
    def test_prompt_column_present(self):
        _require_prompt_column(Dataset.from_dict({"prompt": ["p"]}), "train")

    def test_missing_prompt_column_raises(self):
        with pytest.raises(ValueError, match="'prompt' column"):
            _require_prompt_column(Dataset.from_dict({"text": ["t"]}), "train")


class TestApplyTrackingGroup:
    @pytest.fixture(autouse=True)
    def isolated_environ(self, monkeypatch):
        env = {key: value for key, value in os.environ.items() if key != "WANDB_RUN_GROUP"}
        monkeypatch.setattr(os, "environ", env)

    def test_wandb_backend_sets_env(self):
        cfg = OmegaConf.create({"tracking": {"backend": "wandb", "group": "exp-042"}})
        _apply_tracking_group(cfg)
        assert os.environ["WANDB_RUN_GROUP"] == "exp-042"

    def test_existing_env_wins(self):
        os.environ["WANDB_RUN_GROUP"] = "keep-me"
        _apply_tracking_group(OmegaConf.create({"tracking": {"backend": "wandb", "group": "new"}}))
        assert os.environ["WANDB_RUN_GROUP"] == "keep-me"

    def test_trackio_backend_skips_env(self):
        _apply_tracking_group(OmegaConf.create({"tracking": {"backend": "trackio", "group": "g"}}))
        assert "WANDB_RUN_GROUP" not in os.environ

    def test_no_group_is_noop(self):
        _apply_tracking_group(OmegaConf.create({"tracking": {"backend": "wandb"}}))
        assert "WANDB_RUN_GROUP" not in os.environ


class TestBuildLoggerGroup:
    def test_trackio_logger_receives_group(self):
        cfg = OmegaConf.create({"tracking": {"backend": "trackio", "project": "proj", "group": "exp-042"}})
        built = _build_logger(cfg)
        assert isinstance(built, TrackioLightningLogger)
        assert built._group == "exp-042"

    def test_wandb_logger_receives_group(self):
        cfg = OmegaConf.create(
            {"tracking": {"backend": "wandb", "project": "proj", "run_name": "r1", "group": "exp-042"}}
        )
        built = _build_logger(cfg)
        assert built._wandb_init["group"] == "exp-042"

    def test_wandb_logger_without_group(self):
        cfg = OmegaConf.create({"tracking": {"backend": "wandb", "project": "proj", "run_name": "r1"}})
        assert "group" not in _build_logger(cfg)._wandb_init


class TestAxolotlRender:
    def test_base_plus_overrides_deep_merge(self, tmp_path, capsys):
        base = {
            "base_model": "gpt2",
            "micro_batch_size": 1,
            "optimizer": {"name": "adamw", "lr": 1.0e-4},
        }
        base_path = tmp_path / "base.yaml"
        base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        rendered_path = tmp_path / "out" / "axolotl.yaml"
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "base_config": str(base_path),
                    "rendered_path": str(rendered_path),
                    "overrides": {"micro_batch_size": 4, "optimizer": {"lr": 5.0e-5}},
                }
            }
        )

        result = axolotl_adapter.render(cfg)

        assert result == rendered_path
        data = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
        assert data["base_model"] == "gpt2"
        assert data["micro_batch_size"] == 4
        assert data["optimizer"] == {"name": "adamw", "lr": 5.0e-5}
        assert f"uv run --with axolotl axolotl train {rendered_path}" in capsys.readouterr().out

    def test_no_base_config(self, tmp_path):
        rendered_path = tmp_path / "axolotl.yaml"
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "base_config": None,
                    "rendered_path": str(rendered_path),
                    "overrides": {"base_model": "gpt2"},
                }
            }
        )

        result = axolotl_adapter.render(cfg)

        assert yaml.safe_load(result.read_text(encoding="utf-8")) == {"base_model": "gpt2"}


class TestTrackioLightningLogger:
    def test_log_metrics_forwards_to_trackio(self, monkeypatch):
        import trackio

        calls = []
        monkeypatch.setattr(trackio, "init", lambda **kwargs: calls.append(("init", kwargs)))
        monkeypatch.setattr(trackio, "log", lambda metrics, step=None: calls.append(("log", metrics, step)))

        lightning_logger = TrackioLightningLogger(project="proj", run_name="run-1")
        lightning_logger.log_metrics({"loss": 1.5}, step=3)
        lightning_logger.log_metrics({"loss": 1.2}, step=4)

        assert calls[0] == ("init", {"project": "proj", "name": "run-1"})
        assert calls[1:] == [("log", {"loss": 1.5}, 3), ("log", {"loss": 1.2}, 4)]

    def test_group_forwarded_to_trackio_init(self, monkeypatch):
        import trackio

        calls = []
        monkeypatch.setattr(trackio, "init", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(trackio, "log", lambda metrics, step=None: None)

        lightning_logger = TrackioLightningLogger(project="proj", run_name="run-1", group="exp-042")
        lightning_logger.log_metrics({"loss": 1.5}, step=0)

        assert calls[0] == {"project": "proj", "name": "run-1", "group": "exp-042"}

    def test_log_metrics_swallows_trackio_errors(self, monkeypatch):
        import trackio

        def boom(**kwargs):
            raise RuntimeError("no server")

        monkeypatch.setattr(trackio, "init", boom)
        lightning_logger = TrackioLightningLogger()
        lightning_logger.log_metrics({"loss": 1.0}, step=0)

    def test_name_and_version(self):
        lightning_logger = TrackioLightningLogger(run_name="run-x")
        assert lightning_logger.name == "run-x"
        assert TrackioLightningLogger().name == "trackio"
        assert lightning_logger.version == "0"
