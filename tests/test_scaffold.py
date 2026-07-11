from pathlib import Path

from intern.scaffold import REQUIRED_FILES, missing_required


def _write(experiment: Path, names) -> None:
    experiment.mkdir(parents=True, exist_ok=True)
    for name in names:
        (experiment / name).write_text("# stub\n")


def test_complete_scaffold_has_no_missing(tmp_path: Path) -> None:
    _write(tmp_path / "001-demo", REQUIRED_FILES)
    assert missing_required(tmp_path / "001-demo") == []


def test_missing_run_md_is_reported(tmp_path: Path) -> None:
    _write(tmp_path / "001-demo", [f for f in REQUIRED_FILES if f != "run.md"])
    assert missing_required(tmp_path / "001-demo") == ["run.md"]


def test_missing_reported_in_canonical_order(tmp_path: Path) -> None:
    _write(tmp_path / "001-demo", ["budget.md"])
    assert missing_required(tmp_path / "001-demo") == ["task.md", "plan.md", "ledger.md", "run.md"]


def test_empty_dir_reports_all_required(tmp_path: Path) -> None:
    (tmp_path / "001-demo").mkdir()
    assert missing_required(tmp_path / "001-demo") == list(REQUIRED_FILES)


def test_gate_outputs_and_optional_cards_are_not_required(tmp_path: Path) -> None:
    # verify.md/results.md (gate outputs) and data.md (optional) must not be required.
    for name in ("verify.md", "results.md", "data.md", "research.md", "board.md"):
        assert name not in REQUIRED_FILES
    _write(tmp_path / "001-demo", REQUIRED_FILES)
    assert missing_required(tmp_path / "001-demo") == []
