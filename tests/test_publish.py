"""Tests for intern.publish: the blocking publish gate against a faked HF Hub API (no network)."""

import json
import os
from pathlib import Path

import pytest

from intern import publish as publish_module
from intern.publish import publish_run


BUDGET_MD = """# Budget

max_paths: 2
max_retries_per_path: 2
compute_cap_gpu_h: 2.0
scale_ceiling_params: 200000000
token_budget: null

## Spent

paths_launched: 1
retries_used: 0
gpu_h_used: 0.5
"""

LEDGER_COLUMNS = (
    "path_id",
    "approach",
    "status",
    "final_train_loss",
    "final_eval_loss",
    "verify",
    "failure_cause",
    "retry_of",
    "gpu_min",
    "run_url",
)

SAMPLES = (
    '{"text": "The quick brown fox jumps over the lazy dog while seventeen violet umbrellas drift downstream."}\n'
    '{"text": "Bright copper kettles whistle merrily as autumn leaves scatter across the cobblestone plaza at dusk."}\n'
    '{"text": "Every morning the lighthouse keeper counts distant sails and records the tide in a logbook."}\n'
)

RESULTS_MD = """# Results — 001-demo

**Winner: path-1** — baseline SFT, verify PASS (5 passed, 0 failed, 3 skipped).

```bash
uv run python scripts/python/001-demo.py
uv run python scripts/python/intern.py verify --experiment 001
```

## Path comparison

| path   | change   | outcome |
| ------ | -------- | ------- |
| path-1 | baseline | pass    |
"""


def _ledger_md(status: str = "passed", verify: str = "pass") -> str:
    header = "| " + " | ".join(LEDGER_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(LEDGER_COLUMNS)) + " |"
    row = f"| path-1 | baseline | {status} | 2.31 | 2.5 | {verify} |  |  | 10 |  |"
    return "\n".join([header, separator, row]) + "\n"


def _metrics_jsonl(final_train_loss: float = 2.31) -> str:
    ts = "2026-07-04T12:00:00+00:00"
    records = [
        {"ts": ts, "event": "meta", "key": "param_count", "value": 124_000_000},
        {"ts": ts, "event": "meta", "key": "vocab_size", "value": 50_257},
        {"ts": ts, "event": "meta", "key": "planned_tokens", "value": 100_000},
        {"ts": ts, "step": 10, "split": "train", "name": "loss", "value": 4.0},
        {"ts": ts, "step": 30, "split": "train", "name": "loss", "value": final_train_loss},
        {"ts": ts, "step": 30, "split": "eval", "name": "loss", "value": final_train_loss + 0.2},
        {"ts": ts, "step": 30, "split": "train", "name": "num_input_tokens_seen", "value": 90_000},
    ]
    return "\n".join(json.dumps(record) for record in records) + "\n"


class FakeApi:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = set(existing or ())
        self.created: list[tuple[str, bool]] = []
        self.folder_uploads: list[dict] = []
        self.file_uploads: list[dict] = []
        self.whoami_calls = 0

    def whoami(self) -> dict:
        self.whoami_calls += 1
        return {"name": "tester"}

    def create_repo(self, repo_id: str, private: bool = True, exist_ok: bool = False, **kwargs) -> None:
        if repo_id in self.existing:
            raise RuntimeError(f"409 Client Error: repo {repo_id} already exists")
        self.created.append((repo_id, private))
        self.existing.add(repo_id)

    def upload_folder(self, folder_path: str, repo_id: str, path_in_repo: str | None = None, **kwargs) -> None:
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())
        self.folder_uploads.append({"repo_id": repo_id, "path_in_repo": path_in_repo, "files": files})

    def upload_file(self, path_or_fileobj, path_in_repo: str, repo_id: str, **kwargs) -> None:
        content = path_or_fileobj.decode("utf-8") if isinstance(path_or_fileobj, bytes) else str(path_or_fileobj)
        self.file_uploads.append({"repo_id": repo_id, "path_in_repo": path_in_repo, "content": content})


@pytest.fixture
def fake_api(monkeypatch) -> FakeApi:
    api = FakeApi()
    monkeypatch.setattr(publish_module, "_api", lambda: api)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("HF_USER", raising=False)
    monkeypatch.delenv("INTERN_SKIP_BUNDLE_SCRUB", raising=False)
    return api


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    root = tmp_path
    experiment = root / "experiments" / "001-demo"
    (experiment / "logs").mkdir(parents=True)
    (experiment / "budget.md").write_text(BUDGET_MD)
    (experiment / "ledger.md").write_text(_ledger_md())
    (experiment / "metrics.jsonl").write_text(_metrics_jsonl())
    (experiment / "logs" / "stderr.log").write_text("UserWarning: tokenizer parallelism disabled\n")
    (experiment / "logs" / "samples.jsonl").write_text(SAMPLES)
    (experiment / "task.md").write_text("# Task\n")
    (experiment / "plan.md").write_text("# Plan\n")
    (experiment / "results.md").write_text(RESULTS_MD)

    old = experiment / "ckpts" / "checkpoint-5"
    new = experiment / "ckpts" / "checkpoint-10"
    for ckpt in (old, new):
        ckpt.mkdir(parents=True)
        (ckpt / "model.safetensors").write_bytes(b"weights")
    (new / "trainer_state.json").write_text("{}")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))

    configs = root / "configs"
    configs.mkdir()
    (configs / "main.yaml").write_text("seed: 42\nproject_name: proj\n")
    (configs / "001-demo.yaml").write_text("experiment_name: 001-demo\n")
    return experiment


def test_refuses_failing_verify(experiment_dir: Path, fake_api: FakeApi) -> None:
    (experiment_dir / "metrics.jsonl").write_text(_metrics_jsonl(final_train_loss=0.5))
    assert publish_run(experiment_dir) == 1
    assert fake_api.created == []
    assert fake_api.folder_uploads == []


def test_refuses_without_results_md(experiment_dir: Path, fake_api: FakeApi) -> None:
    (experiment_dir / "results.md").unlink()
    assert publish_run(experiment_dir) == 2
    assert fake_api.created == []


def test_refuses_without_passed_ledger_row(experiment_dir: Path, fake_api: FakeApi) -> None:
    (experiment_dir / "ledger.md").write_text(_ledger_md(status="running", verify="pending"))
    assert publish_run(experiment_dir) == 1
    assert fake_api.created == []


def test_refuses_without_checkpoint(experiment_dir: Path, fake_api: FakeApi) -> None:
    for child in sorted((experiment_dir / "ckpts").rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    assert publish_run(experiment_dir) == 2
    assert fake_api.created == []


def test_refuses_without_hf_token(experiment_dir: Path, fake_api: FakeApi, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert publish_run(experiment_dir) == 2
    assert fake_api.created == []


def test_happy_path_uploads_model_bundle_and_card(experiment_dir: Path, fake_api: FakeApi) -> None:
    assert publish_run(experiment_dir) == 0

    assert fake_api.whoami_calls == 1
    assert fake_api.created == [("tester/proj-001-demo", True)]

    model_upload, bundle_upload = fake_api.folder_uploads
    assert model_upload["path_in_repo"] is None
    assert model_upload["files"] == ["model.safetensors", "trainer_state.json"]  # newest checkpoint dir
    assert bundle_upload["path_in_repo"] == "bundle"
    assert bundle_upload["files"] == [
        "001-demo.yaml",
        "budget.md",
        "ledger.md",
        "plan.md",
        "results.md",
        "samples.jsonl",
        "task.md",
        "verify.md",
    ]

    (card,) = fake_api.file_uploads
    assert card["path_in_repo"] == "README.md"
    assert card["repo_id"] == "tester/proj-001-demo"
    assert "Results — 001-demo" in card["content"]
    assert "**Winner: path-1**" in card["content"]
    assert "| path-1 | baseline | pass" in card["content"]
    assert "uv run python scripts/python/001-demo.py" in card["content"]
    assert "my-little-ml-intern" in card["content"]

    results = (experiment_dir / "results.md").read_text()
    assert results.endswith("## Published\n\n<https://huggingface.co/tester/proj-001-demo>\n")


def test_hf_user_env_skips_whoami(experiment_dir: Path, fake_api: FakeApi, monkeypatch) -> None:
    monkeypatch.setenv("HF_USER", "envuser")
    assert publish_run(experiment_dir) == 0
    assert fake_api.whoami_calls == 0
    assert fake_api.created == [("envuser/proj-001-demo", True)]


def test_repo_collision_appends_suffix(experiment_dir: Path, fake_api: FakeApi) -> None:
    fake_api.existing.add("org/custom")
    assert publish_run(experiment_dir, repo_id="org/custom", private=False) == 0
    assert fake_api.whoami_calls == 0
    assert fake_api.created == [("org/custom-2", False)]
    results = (experiment_dir / "results.md").read_text()
    assert results.endswith("## Published\n\n<https://huggingface.co/org/custom-2>\n")


def test_repo_collision_exhausted_returns_2(experiment_dir: Path, fake_api: FakeApi) -> None:
    fake_api.existing.update({"org/custom", *(f"org/custom-{n}" for n in range(2, 6))})
    assert publish_run(experiment_dir, repo_id="org/custom") == 2
    assert fake_api.created == []
    assert "## Published" not in (experiment_dir / "results.md").read_text()


def test_verify_exit_2_propagates(tmp_path: Path, fake_api: FakeApi) -> None:
    exp = tmp_path / "001-empty"
    exp.mkdir()

    assert publish_run(exp) == 2


def test_scrub_blocks_planted_token(experiment_dir: Path, fake_api: FakeApi) -> None:
    results = experiment_dir / "results.md"
    results.write_text(results.read_text() + "\nleak: hf_" + "A" * 34 + "\n")

    assert publish_run(experiment_dir) == 2
    assert fake_api.created == []
    assert fake_api.folder_uploads == []
    assert fake_api.file_uploads == []


def test_scrub_blocks_home_dir_path(experiment_dir: Path, fake_api: FakeApi) -> None:
    plan = experiment_dir / "plan.md"
    # The current machine's home leaking into an artifact is the real threat.
    plan.write_text(plan.read_text() + f"\nrun dir was {Path.home()}/work/run\n")

    assert publish_run(experiment_dir) == 2
    assert fake_api.created == []
    assert fake_api.folder_uploads == []


def test_scrub_allows_other_users_paths(experiment_dir: Path, fake_api: FakeApi) -> None:
    # Model generations echo training-data paths under other users — not an env leak.
    plan = experiment_dir / "plan.md"
    plan.write_text(plan.read_text() + "\nsample output mentions /Users/badlogic/workspaces/pi/x\n")

    assert publish_run(experiment_dir) == 0


def test_scrub_bypass_env(experiment_dir: Path, fake_api: FakeApi, monkeypatch) -> None:
    results = experiment_dir / "results.md"
    results.write_text(results.read_text() + "\nleak: hf_" + "A" * 34 + "\n")
    monkeypatch.setenv("INTERN_SKIP_BUNDLE_SCRUB", "1")

    assert publish_run(experiment_dir) == 0
    assert len(fake_api.folder_uploads) == 2
    assert len(fake_api.file_uploads) == 1


def test_model_card_adapter_frontmatter(tmp_path):
    from intern.publish import _model_card

    model_dir = tmp_path / "checkpoint-200"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "google/gemma-4-E2B-it"}')
    card = _model_card("# Results — 001\n\n**Winner: path-1** works.\n", "001-demo", model_dir, "me/adapter")

    assert card.startswith("---\n")
    assert "base_model: google/gemma-4-E2B-it" in card
    assert "library_name: peft" in card
    assert 'PeftModel.from_pretrained(base, "me/adapter")' in card


def test_model_card_full_model_has_no_adapter_frontmatter(tmp_path):
    from intern.publish import _model_card

    model_dir = tmp_path / "ckpts"
    model_dir.mkdir()  # no adapter_config.json -> full model
    card = _model_card("# Results\n\n**Winner: path-1**\n", "002-demo", model_dir, "me/model")

    assert not card.startswith("---")
    assert "library_name: peft" not in card
