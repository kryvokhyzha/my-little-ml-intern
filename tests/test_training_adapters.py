import math
import os

import pytest
import yaml
from datasets import Dataset
from omegaconf import OmegaConf

from data.loading import load_split as _load_split
from data.loading import validate_columns as _validate_columns
from training import axolotl_adapter
from training.lightning_adapter import TrackioLightningLogger, _build_logger
from training.runtime import apply_tracking_env as _apply_tracking_env
from training.runtime import smoke_enabled as _smoke_enabled
from training.trl.config import apply_smoke as _apply_smoke
from training.trl.config import build_args as _build_args
from training.trl.config import report_to as _report_to
from training.trl.config import write_meta as _write_meta
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

    def test_bf16_pinned_false_on_incapable_hardware(self, cfg, monkeypatch):
        # TRL >=1.7 defaults bf16=True when fp16 is unset; on a CPU-only box (the CI
        # runner) transformers then rejects it. The adapter must pin bf16=False there.
        import transformers.utils
        from trl import SFTConfig

        monkeypatch.setattr(transformers.utils, "is_torch_bf16_gpu_available", lambda: False)
        args = _build_args(cfg, SFTConfig)
        assert args.bf16 is False

    def test_explicit_bf16_false_survives(self, cfg):
        from trl import SFTConfig

        cfg.trainer.args.bf16 = False
        args = _build_args(cfg, SFTConfig)
        assert args.bf16 is False

    def test_tracking_project_reaches_trackio_field(self, cfg):
        # TrackioCallback logs under TrainingArguments.project (default "huggingface").
        from trl import SFTConfig

        cfg.tracking.project = "my-proj"
        args = _build_args(cfg, SFTConfig)
        assert args.project == "my-proj"

    def test_explicit_args_project_wins(self, cfg):
        from trl import SFTConfig

        cfg.tracking.project = "from-tracking"
        cfg.trainer.args.project = "explicit"
        args = _build_args(cfg, SFTConfig)
        assert args.project == "explicit"


class FakeMetricsLog:
    def __init__(self):
        self.events = []

    def append_event(self, event, **fields):
        self.events.append({"event": event, **fields})

    def meta(self, key):
        for record in reversed(self.events):
            if record.get("event") == "meta" and record.get("key") == key:
                return record.get("value")
        return None


class TestWriteMeta:
    def _cfg(self, args=None, target_params=None):
        return OmegaConf.create(
            {"trainer": {"kind": "trl_sft", "args": args or {}}, "model": {"target_params": target_params}}
        )

    def test_completion_only_meta_emitted_when_flag_set(self):
        log = FakeMetricsLog()
        _write_meta(log, param_count=124, vocab_size=32000, cfg=self._cfg({"completion_only_loss": True}))
        assert log.meta("completion_only") is True

    def test_completion_only_meta_absent_when_flag_unset(self):
        log = FakeMetricsLog()
        _write_meta(log, param_count=124, vocab_size=32000, cfg=self._cfg({"completion_only_loss": False}))
        assert log.meta("completion_only") is None
        _write_meta(FakeMetricsLog(), param_count=124, vocab_size=32000, cfg=self._cfg())

    def test_target_params_from_model_group(self):
        log = FakeMetricsLog()
        _write_meta(log, param_count=124, vocab_size=32000, cfg=self._cfg(target_params=135000000))
        assert log.meta("target_params") == 135000000

    def test_target_params_absent_when_null(self):
        log = FakeMetricsLog()
        _write_meta(log, param_count=124, vocab_size=32000, cfg=self._cfg())
        assert log.meta("target_params") is None


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


class TestValidateColumns:
    @pytest.mark.parametrize(
        "columns",
        [["text"], ["prompt", "completion"], ["messages"]],
    )
    def test_sft_accepts_each_format(self, columns):
        _validate_columns(Dataset.from_dict({c: ["x"] for c in columns}), "trl_sft", "train")

    def test_sft_honors_custom_text_field(self):
        _validate_columns(Dataset.from_dict({"body": ["x"]}), "trl_sft", "train", text_field="body")

    def test_sft_allows_extra_columns_like_tools(self):
        # Tool-calling SFT: messages + tools schemas; TRL forwards `tools` to the chat template.
        _validate_columns(Dataset.from_dict({"messages": [[]], "tools": [[]]}), "trl_sft", "train")

    def test_sft_rejects_unusable_columns(self):
        with pytest.raises(ValueError, match="trl_sft train"):
            _validate_columns(Dataset.from_dict({"question": ["q"], "answer": ["a"]}), "trl_sft", "train")

    def test_dpo_accepts_implicit_prompt(self):
        _validate_columns(Dataset.from_dict({"chosen": ["c"], "rejected": ["r"]}), "trl_dpo", "train")

    def test_dpo_rejects_sft_shaped_data(self):
        with pytest.raises(ValueError, match="chosen"):
            _validate_columns(Dataset.from_dict({"prompt": ["p"], "completion": ["c"]}), "trl_dpo", "train")

    def test_grpo_requires_prompt(self):
        _validate_columns(Dataset.from_dict({"prompt": ["p"]}), "trl_grpo", "train")
        with pytest.raises(ValueError, match="prompt"):
            _validate_columns(Dataset.from_dict({"text": ["t"]}), "trl_grpo", "train")

    def test_unknown_task_is_not_checked(self):
        _validate_columns(Dataset.from_dict({"anything": ["x"]}), "lightning", "train")


class TestLoadSplit:
    def test_forwards_loader_kwargs(self, tmp_path):
        csv = tmp_path / "rows.csv"
        csv.write_text("prompt;completion\np1;c1\n")
        ds = _load_split(str(csv), "train", delimiter=";")
        assert ds.column_names == ["prompt", "completion"]

    def test_plain_dataset_dir_refuses_eval(self, tmp_path):
        Dataset.from_dict({"text": ["a"]}).save_to_disk(str(tmp_path / "ds"))
        with pytest.raises(ValueError, match="training data"):
            _load_split(str(tmp_path / "ds"), "train", for_eval=True)

    def test_eval_node_gets_guard_injected(self, tmp_path):
        from hydra.errors import InstantiationException

        from training.trl.run import _load_data_node

        Dataset.from_dict({"text": ["a"]}).save_to_disk(str(tmp_path / "ds"))
        node = OmegaConf.create(
            {"_target_": "data.loading.load_split", "dataset": str(tmp_path / "ds"), "split": "train"}
        )
        assert _load_data_node(node) is not None
        # hydra wraps the guard's ValueError in InstantiationException; the message survives.
        with pytest.raises(InstantiationException, match="training data"):
            _load_data_node(node, for_eval=True)


class TestApplyTrackingEnv:
    @pytest.fixture(autouse=True)
    def isolated_environ(self, monkeypatch):
        env = {k: v for k, v in os.environ.items() if k not in ("WANDB_RUN_GROUP", "WANDB_PROJECT")}
        monkeypatch.setattr(os, "environ", env)

    def test_wandb_backend_sets_env(self):
        cfg = OmegaConf.create({"tracking": {"backend": "wandb", "group": "exp-042", "project": "my-proj"}})
        _apply_tracking_env(cfg)
        assert os.environ["WANDB_RUN_GROUP"] == "exp-042"
        assert os.environ["WANDB_PROJECT"] == "my-proj"

    def test_existing_group_env_wins(self):
        os.environ["WANDB_RUN_GROUP"] = "keep-me"
        _apply_tracking_env(OmegaConf.create({"tracking": {"backend": "wandb", "group": "new"}}))
        assert os.environ["WANDB_RUN_GROUP"] == "keep-me"

    def test_resolved_project_overwrites_env(self):
        # The config value already honors a WANDB_PROJECT env override via interpolation,
        # so the resolved config is authoritative here.
        os.environ["WANDB_PROJECT"] = "stale"
        _apply_tracking_env(OmegaConf.create({"tracking": {"backend": "wandb", "project": "resolved"}}))
        assert os.environ["WANDB_PROJECT"] == "resolved"

    def test_trackio_backend_skips_env(self):
        _apply_tracking_env(OmegaConf.create({"tracking": {"backend": "trackio", "group": "g", "project": "p"}}))
        assert "WANDB_RUN_GROUP" not in os.environ
        assert "WANDB_PROJECT" not in os.environ

    def test_no_keys_is_noop(self):
        _apply_tracking_env(OmegaConf.create({"tracking": {"backend": "wandb"}}))
        assert "WANDB_RUN_GROUP" not in os.environ
        assert "WANDB_PROJECT" not in os.environ


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
