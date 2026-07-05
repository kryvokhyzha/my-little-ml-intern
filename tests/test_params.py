from intern import params


def test_experiment_model_repo_from_defaults_pick(tmp_path):
    configs = tmp_path / "configs"
    (configs / "model").mkdir(parents=True)
    (configs / "007-demo.yaml").write_text("defaults:\n  - model: my_model\n  - _self_\nexperiment_name: 007-demo\n")
    (configs / "model" / "my_model.yaml").write_text('main:\n  _args_: ["org/my-model"]\n')
    assert params.experiment_model_repo(configs, "007-demo") == "org/my-model"


def test_experiment_model_repo_self_override_wins(tmp_path):
    configs = tmp_path / "configs"
    (configs / "model").mkdir(parents=True)
    (configs / "008-demo.yaml").write_text(
        'defaults:\n  - model: base\n  - _self_\nmodel:\n  main:\n    _args_: ["org/override"]\n'
    )
    (configs / "model" / "base.yaml").write_text('main:\n  _args_: ["org/base"]\n')
    assert params.experiment_model_repo(configs, "008-demo") == "org/override"


def test_experiment_model_repo_missing_returns_none(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    assert params.experiment_model_repo(configs, "nope") is None


def test_hf_param_count_reads_safetensors_total(monkeypatch):
    from types import SimpleNamespace

    class FakeApi:
        def model_info(self, repo_id, expand=None):
            return SimpleNamespace(safetensors=SimpleNamespace(total=5123178051))

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    assert params.hf_param_count("google/gemma-4-E2B-it") == 5123178051


def test_hf_param_count_returns_none_on_error(monkeypatch):
    from types import SimpleNamespace

    class FakeApi:
        def model_info(self, repo_id, expand=None):
            raise OSError("offline")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    assert params.hf_param_count("x/y") is None
