"""End-to-end tests for scripts/python/intern.py and the bash helpers (subprocess, real exit codes)."""

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = ("uv", "run", "python", "scripts/python/intern.py")

# Exhausted on purpose: paths_launched == max_paths, so can-launch must be denied.
BUDGET_MD = """# Budget

max_paths: 1
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


def _ledger_md() -> str:
    header = "| " + " | ".join(LEDGER_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(LEDGER_COLUMNS)) + " |"
    row = "| path-1 | baseline | running | 2.31 | 2.5 | pending |  |  | 10 |  |"
    return "\n".join([header, separator, row]) + "\n"


def _metrics_jsonl() -> str:
    ts = "2026-07-04T12:00:00+00:00"
    records = [
        {"ts": ts, "event": "meta", "key": "param_count", "value": 124_000_000},
        {"ts": ts, "event": "meta", "key": "vocab_size", "value": 50_257},
        {"ts": ts, "event": "meta", "key": "planned_tokens", "value": 100_000},
        {"ts": ts, "step": 10, "split": "train", "name": "loss", "value": 4.0},
        {"ts": ts, "step": 20, "split": "train", "name": "loss", "value": 3.0},
        {"ts": ts, "step": 30, "split": "train", "name": "loss", "value": 2.31},
        {"ts": ts, "step": 10, "split": "eval", "name": "loss", "value": 4.2},
        {"ts": ts, "step": 30, "split": "eval", "name": "loss", "value": 2.5},
        {"ts": ts, "step": 10, "split": "train", "name": "num_input_tokens_seen", "value": 30_000},
        {"ts": ts, "step": 30, "split": "train", "name": "num_input_tokens_seen", "value": 90_000},
    ]
    return "\n".join(json.dumps(record) for record in records) + "\n"


@pytest.fixture
def experiments_root(tmp_path: Path) -> Path:
    root = tmp_path / "experiments"
    experiment = root / "001-demo"
    (experiment / "logs").mkdir(parents=True)
    (experiment / "budget.md").write_text(BUDGET_MD)
    (experiment / "ledger.md").write_text(_ledger_md())
    (experiment / "metrics.jsonl").write_text(_metrics_jsonl())
    (experiment / "logs" / "stderr.log").write_text("UserWarning: tokenizer parallelism disabled\n")
    (experiment / "logs" / "samples.jsonl").write_text(SAMPLES)
    return root


def run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *map(str, args)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_budget_status_exits_0(experiments_root: Path) -> None:
    result = run_cli("budget", "--experiment", "001", "status", "--experiments-root", experiments_root)
    assert result.returncode == 0, result.stderr
    assert "max_paths: 1" in result.stdout
    assert "paths_launched: 1" in result.stdout


def test_budget_can_launch_exhausted_exits_1(experiments_root: Path) -> None:
    result = run_cli("budget", "--experiment", "001", "can-launch", "--experiments-root", experiments_root)
    assert result.returncode == 1, result.stderr
    assert "denied" in result.stdout


def test_budget_record_gpu_h_negative_exits_2(experiments_root: Path) -> None:
    result = run_cli(
        "budget", "--experiment", "001", "record-gpu-h", "--hours=-0.5", "--experiments-root", experiments_root
    )
    assert result.returncode == 2, result.stderr
    assert "record-gpu-h" in result.stderr
    budget_after = (experiments_root / "001-demo" / "budget.md").read_text()
    assert "gpu_h_used: 0.5" in budget_after


def test_verify_passing_run_exits_0(experiments_root: Path) -> None:
    result = run_cli("verify", "--experiment", "001", "--experiments-root", experiments_root)
    assert result.returncode == 0, result.stderr
    report = experiments_root / "001-demo" / "verify.md"
    assert report.is_file()
    assert "OVERALL: PASS" in report.read_text()


def test_verify_missing_metrics_exits_2(experiments_root: Path) -> None:
    (experiments_root / "001-demo" / "metrics.jsonl").unlink()
    result = run_cli("verify", "--experiment", "001", "--experiments-root", experiments_root)
    assert result.returncode == 2, result.stderr


def test_unknown_experiment_exits_2(experiments_root: Path) -> None:
    result = run_cli("verify", "--experiment", "042", "--experiments-root", experiments_root)
    assert result.returncode == 2, result.stderr


def test_ambiguous_experiment_exits_2(experiments_root: Path) -> None:
    (experiments_root / "001-other").mkdir()
    result = run_cli("verify", "--experiment", "001", "--experiments-root", experiments_root)
    assert result.returncode == 2, result.stderr
    assert "Ambiguous" in result.stderr


def test_full_experiment_name_resolves(experiments_root: Path) -> None:
    result = run_cli("budget", "--experiment", "001-demo", "status", "--experiments-root", experiments_root)
    assert result.returncode == 0, result.stderr


def test_ledger_upsert_then_show(experiments_root: Path) -> None:
    result = run_cli(
        "ledger",
        "--experiment",
        "001",
        "upsert",
        "--path-id",
        "path-2",
        "--status",
        "queued",
        "--approach",
        "wider net",
        "--experiments-root",
        experiments_root,
    )
    assert result.returncode == 0, result.stderr
    result = run_cli("ledger", "--experiment", "001", "show", "--experiments-root", experiments_root)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert {"path-1", "path-2"} <= {row["path_id"] for row in rows}


def test_ledger_upsert_invalid_status_exits_2(experiments_root: Path) -> None:
    result = run_cli(
        "ledger",
        "--experiment",
        "001",
        "upsert",
        "--path-id",
        "path-3",
        "--status",
        "bogus",
        "--experiments-root",
        experiments_root,
    )
    assert result.returncode == 2, result.stderr


def test_publish_private_int_exits_2(experiments_root: Path) -> None:
    # fire parses `--private 0` as int 0 — must be refused before any network/Hub access.
    result = run_cli("publish", "--experiment", "001", "--private", "0", "--experiments-root", experiments_root)
    assert result.returncode == 2, result.stderr
    assert "true|false" in result.stderr


def test_notify_without_env_exits_0() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TG_BOT_TOKEN", "TG_CHAT_ID", "SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID"}
    }
    result = subprocess.run(
        ["bash", "scripts/bash/notify.sh", "plan_ready", "x"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_gpu_probe_prints_keys_and_exits_0() -> None:
    result = subprocess.run(
        ["bash", "scripts/bash/gpu_probe.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "cuda=" in result.stdout
    assert "mps=" in result.stdout
