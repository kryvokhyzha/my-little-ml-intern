import sys

import pytest
from omegaconf import OmegaConf

from helper.display import get_console, is_interactive, print_table, run_footer, run_header
from training import lightning_adapter
from training.trl import config as trl_config


_ENV_VARS = ("CLAUDECODE", "NO_COLOR", "FORCE_RICH", "JSON_LOGS", "COLORIZE")


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_console.cache_clear()
    yield monkeypatch
    get_console.cache_clear()


class _FakeTty:
    def isatty(self) -> bool:
        return True


class TestIsInteractive:
    def test_tty_and_clean_env_is_interactive(self, clean_env):
        clean_env.setattr(sys, "stdout", _FakeTty())
        assert is_interactive() is True

    def test_non_tty_is_not_interactive(self, clean_env):
        assert is_interactive() is False

    @pytest.mark.parametrize(
        ("name", "value"),
        [("CLAUDECODE", "1"), ("NO_COLOR", "1"), ("JSON_LOGS", "true"), ("COLORIZE", "false")],
    )
    def test_env_disables(self, clean_env, name, value):
        clean_env.setattr(sys, "stdout", _FakeTty())
        clean_env.setenv(name, value)
        assert is_interactive() is False

    def test_force_rich_overrides_everything(self, clean_env):
        clean_env.setenv("CLAUDECODE", "1")
        clean_env.setenv("JSON_LOGS", "true")
        clean_env.setenv("FORCE_RICH", "1")
        assert is_interactive() is True

    def test_benign_env_values_stay_interactive(self, clean_env):
        clean_env.setattr(sys, "stdout", _FakeTty())
        clean_env.setenv("JSON_LOGS", "false")
        clean_env.setenv("COLORIZE", "true")
        assert is_interactive() is True


class TestGetConsole:
    def test_cached_instances(self, clean_env):
        assert get_console() is get_console()
        assert get_console(stderr=True) is get_console(stderr=True)
        assert get_console(stderr=True) is not get_console()
        assert get_console(stderr=True).stderr is True


class TestPrintTable:
    def test_plain_mode_aligned_columns(self, clean_env, capsys):
        clean_env.setenv("CLAUDECODE", "1")
        print_table("My table", ["name", "value"], [["alpha", "1"], ["b", "10.5"]])
        lines = capsys.readouterr().out.splitlines()
        assert lines == ["My table", "name   value", "alpha  1", "b      10.5"]

    def test_plain_mode_without_title(self, clean_env, capsys):
        clean_env.setenv("CLAUDECODE", "1")
        print_table(None, ["a"], [["x"]])
        assert capsys.readouterr().out == "a\nx\n"

    def test_rich_mode_renders_table(self, clean_env, capsys):
        clean_env.setenv("FORCE_RICH", "1")
        print_table("My table", ["name"], [["alpha"]])
        out = capsys.readouterr().out
        assert "My table" in out
        assert len(out.splitlines()) > 2


class TestRunHeaderFooter:
    def test_header_plain_single_greppable_line(self, clean_env, capsys):
        clean_env.setenv("CLAUDECODE", "1")
        run_header("smoke", {"experiment": "001-toy", "steps": 5})
        assert capsys.readouterr().out == "RUN_START | title=smoke | experiment=001-toy | steps=5\n"

    def test_footer_plain_single_greppable_line(self, clean_env, capsys):
        clean_env.setenv("CLAUDECODE", "1")
        run_footer("passed", {"final_train_loss": 2.31})
        assert capsys.readouterr().out == "RUN_END | status=passed | final_train_loss=2.31\n"

    def test_rich_mode_renders_panel(self, clean_env, capsys):
        clean_env.setenv("FORCE_RICH", "1")
        run_header("smoke", {"experiment": "001-toy"})
        out = capsys.readouterr().out
        assert "smoke" in out
        assert len(out.splitlines()) >= 3


class _FakeTrainer:
    captured: dict | None = None

    def __init__(self, **kwargs):
        _FakeTrainer.captured = kwargs
        self.callback_metrics: dict = {}
        self.global_step = 0

    def fit(self, module, datamodule=None):
        pass


class TestAdapterProgressDefaults:
    @pytest.fixture
    def trl_cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "trainer": {"args": {"output_dir": str(tmp_path / "ckpts"), "max_steps": 1}},
                "tracking": {"backend": "none"},
            }
        )

    def test_trl_disable_tqdm_when_non_interactive(self, trl_cfg, monkeypatch):
        from trl import SFTConfig

        monkeypatch.setattr(trl_config, "is_interactive", lambda: False)
        args = trl_config.build_args(trl_cfg, SFTConfig)
        assert bool(args.disable_tqdm) is True

    def test_trl_tqdm_kept_when_interactive(self, trl_cfg, monkeypatch):
        from trl import SFTConfig

        monkeypatch.setattr(trl_config, "is_interactive", lambda: True)
        args = trl_config.build_args(trl_cfg, SFTConfig)
        assert bool(args.disable_tqdm) is False

    def test_trl_cfg_value_wins_over_default(self, trl_cfg, monkeypatch):
        from trl import SFTConfig

        monkeypatch.setattr(trl_config, "is_interactive", lambda: True)
        trl_cfg.trainer.args.disable_tqdm = True
        args = trl_config.build_args(trl_cfg, SFTConfig)
        assert bool(args.disable_tqdm) is True

    def test_lightning_progress_bar_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SMOKE_TEST", raising=False)
        monkeypatch.setattr(lightning_adapter, "is_interactive", lambda: False)
        monkeypatch.setattr("lightning.pytorch.Trainer", _FakeTrainer)
        cfg = OmegaConf.create(
            {
                "experiment_dir": str(tmp_path),
                "smoke_test": False,
                "trainer": {
                    "module": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
                    "datamodule": {"_target_": "builtins.dict"},
                    "args": {"max_steps": 1},
                },
                "tracking": {"backend": "none", "run_name": "t"},
            }
        )

        lightning_adapter.run(cfg)

        assert _FakeTrainer.captured is not None
        assert _FakeTrainer.captured["enable_progress_bar"] is False

    def test_lightning_cfg_value_wins_over_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SMOKE_TEST", raising=False)
        monkeypatch.setattr(lightning_adapter, "is_interactive", lambda: False)
        monkeypatch.setattr("lightning.pytorch.Trainer", _FakeTrainer)
        cfg = OmegaConf.create(
            {
                "experiment_dir": str(tmp_path),
                "smoke_test": False,
                "trainer": {
                    "module": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
                    "datamodule": {"_target_": "builtins.dict"},
                    "args": {"max_steps": 1, "enable_progress_bar": True},
                },
                "tracking": {"backend": "none", "run_name": "t"},
            }
        )

        lightning_adapter.run(cfg)

        assert _FakeTrainer.captured is not None
        assert _FakeTrainer.captured["enable_progress_bar"] is True
