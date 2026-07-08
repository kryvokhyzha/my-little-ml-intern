import os

import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _compose():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="prep-pi-mono-sft")


def test_config_composes_without_hf_user(monkeypatch):
    monkeypatch.delenv("HF_USER", raising=False)
    cfg = _compose()
    assert cfg.data.source._target_ == "data.pi_mono.download_sessions"
    assert cfg.data.source.dataset_id == "badlogicgames/pi-mono"
    assert cfg.private is True
    assert cfg.dry_run is False
    assert cfg.max_length == 4096
    assert cfg.eval_size == 256
    assert cfg.include_reasoning is False
    assert cfg.limit_sessions is None
    assert cfg.target_repo is None


def test_config_reads_hf_user_from_env(monkeypatch):
    monkeypatch.setenv("HF_USER", "someone")
    cfg = _compose()
    assert cfg.target_repo == "someone"
    os.environ.pop("HF_USER", None)
