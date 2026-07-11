"""Tests for intern.report: gates_summary parsing and the three render_gates output modes."""

import json
import subprocess
from pathlib import Path

import pytest

from helper.display import get_console
from intern.report import gates_summary, render_gates


REPO_ROOT = Path(__file__).resolve().parents[1]

BUDGET_MD = """# Budget

max_paths: 2
max_retries_per_path: 2
compute_cap_gpu_h: 2.0
scale_ceiling_params: 200000000
token_budget: null

## Spent

paths_launched: 1
retries_used: 0
gpu_h_used: 0.1
"""

VERDICT_LINES = (
    "VERDICT: loss_plausibility = PASS | value=2.31 | threshold=(1.04, 10.4) | final train loss inside ln(vocab) band",
    "VERDICT: eval_train_gap = SKIP | value=n/a | threshold=0.5 | no eval metrics logged",
)
OVERALL_LINE = "OVERALL: PASS (1 passed, 0 failed, 1 skipped)"
JUDGMENT_LINE = "JUDGMENT: generation_quality = PASS | story-like English, matches the training distribution"
VERIFY_MD = "\n".join([*VERDICT_LINES, OVERALL_LINE, JUDGMENT_LINE]) + "\n"

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


def _ledger_md() -> str:
    header = "| " + " | ".join(LEDGER_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(LEDGER_COLUMNS)) + " |"
    rows = [
        "| path-1 | baseline | running | 2.31 | 2.5 | pending |  |  | 10 |  |",
        "| path-2 | retry: lower lr | queued |  |  | pending |  | path-1 |  |  |",
    ]
    return "\n".join([header, separator, *rows]) + "\n"


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiments" / "001-demo"
    experiment.mkdir(parents=True)
    (experiment / "verify.md").write_text(VERIFY_MD)
    (experiment / "budget.md").write_text(BUDGET_MD)
    (experiment / "ledger.md").write_text(_ledger_md())
    for name in ("task.md", "plan.md", "run.md"):  # complete the required scaffold set
        (experiment / name).write_text("# stub\n")
    return experiment


@pytest.fixture
def plain_env(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.delenv("FORCE_RICH", raising=False)


class TestGatesSummary:
    def test_parses_all_sections(self, experiment_dir: Path) -> None:
        summary = gates_summary(experiment_dir)

        checks = summary["verify"]["checks"]
        assert [check["name"] for check in checks] == ["loss_plausibility", "eval_train_gap"]
        assert checks[0] == {
            "name": "loss_plausibility",
            "status": "PASS",
            "value": "2.31",
            "threshold": "(1.04, 10.4)",
            "detail": "final train loss inside ln(vocab) band",
        }
        assert summary["verify"]["overall"] == "PASS (1 passed, 0 failed, 1 skipped)"
        assert summary["verify"]["judgment"].startswith("generation_quality = PASS")

        assert summary["budget"]["caps"] == {
            "max_paths": 2,
            "max_retries_per_path": 2,
            "compute_cap_gpu_h": 2.0,
            "scale_ceiling_params": 200_000_000,
            "token_budget": None,
        }
        assert summary["budget"]["spent"] == {"paths_launched": 1, "retries_used": 0, "gpu_h_used": 0.1}

        assert [row["path_id"] for row in summary["ledger"]] == ["path-1", "path-2"]
        assert summary["ledger"][1]["retry_of"] == "path-1"

    def test_tolerates_missing_artifacts(self, tmp_path: Path) -> None:
        summary = gates_summary(tmp_path)
        assert summary == {
            "scaffold": ["task.md", "plan.md", "budget.md", "ledger.md", "run.md"],
            "verify": None,
            "budget": None,
            "ledger": [],
        }

    def test_verify_without_judgment(self, experiment_dir: Path) -> None:
        (experiment_dir / "verify.md").write_text("\n".join([*VERDICT_LINES, OVERALL_LINE]) + "\n")
        summary = gates_summary(experiment_dir)
        assert summary["verify"]["judgment"] is None
        assert summary["verify"]["overall"] == "PASS (1 passed, 0 failed, 1 skipped)"


class TestRenderGates:
    def test_plain_mode_greppable_lines(self, experiment_dir: Path, plain_env, capsys) -> None:
        render_gates(experiment_dir)
        lines = capsys.readouterr().out.splitlines()
        assert lines == [
            "SCAFFOLD | complete",
            *VERDICT_LINES,
            OVERALL_LINE,
            JUDGMENT_LINE,
            "BUDGET | paths=1/2 | retries=0/2 | gpu_h=0.1/2.0",
            "LEDGER | path-1 | status=running | verify=pending",
            "LEDGER | path-2 | status=queued | verify=pending",
        ]

    def test_plain_mode_empty_dir_shows_scaffold_missing(self, tmp_path: Path, plain_env, capsys) -> None:
        # A bare experiment dir still reports its missing scaffold — the dashboard never silently blanks.
        render_gates(tmp_path)
        assert capsys.readouterr().out == "SCAFFOLD | MISSING: task.md, plan.md, budget.md, ledger.md, run.md\n"

    def test_json_round_trips(self, experiment_dir: Path, capsys) -> None:
        render_gates(experiment_dir, as_json=True)
        out = capsys.readouterr().out
        assert json.loads(out) == gates_summary(experiment_dir)

    def test_rich_mode_renders_tables(self, experiment_dir: Path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("FORCE_RICH", "1")
        get_console.cache_clear()
        try:
            render_gates(experiment_dir)
        finally:
            get_console.cache_clear()
        out = capsys.readouterr().out
        assert "Verify checks" in out
        assert "Budget" in out
        assert "Ledger" in out
        assert "OVERALL: PASS" in out


def test_cli_status_json_exits_0(experiment_dir: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/python/intern.py",
            "status",
            "--experiment",
            "001",
            "--json",
            "--experiments-root",
            str(experiment_dir.parent),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["verify"]["overall"].startswith("PASS")
    assert summary["budget"]["spent"]["paths_launched"] == 1
    assert len(summary["ledger"]) == 2
