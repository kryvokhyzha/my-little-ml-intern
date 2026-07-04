from pathlib import Path

import pytest

from intern.ledger import COLUMNS, Ledger


HEADER = (
    "| path_id | approach | status | final_train_loss | final_eval_loss | verify"
    " | failure_cause | retry_of | gpu_min | run_url |"
)
SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def test_upsert_creates_row_and_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    ledger = Ledger(path)
    ledger.upsert("path-1", approach="baseline", status="queued")
    assert path.exists()
    assert ledger.rows() == [
        {
            "path_id": "path-1",
            "approach": "baseline",
            "status": "queued",
            "final_train_loss": "",
            "final_eval_loss": "",
            "verify": "",
            "failure_cause": "",
            "retry_of": "",
            "gpu_min": "",
            "run_url": "",
        }
    ]


def test_upsert_updates_existing_row(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-1", status="queued")
    ledger.upsert("path-2", status="queued")
    ledger.upsert("path-1", status="running", final_train_loss=2.31)
    rows = ledger.rows()
    assert len(rows) == 2
    assert rows[0]["status"] == "running"
    assert rows[0]["final_train_loss"] == "2.31"
    assert rows[1]["path_id"] == "path-2"


def test_unknown_column_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    with pytest.raises(ValueError, match="unknown ledger columns: loss"):
        ledger.upsert("path-1", loss=1.0)


def test_path_id_as_field_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    with pytest.raises(ValueError, match="path_id"):
        ledger.upsert("path-1", path_id="path-2")


def test_invalid_status_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    with pytest.raises(ValueError, match="invalid status"):
        ledger.upsert("path-1", status="exploded")


def test_invalid_verify_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    with pytest.raises(ValueError, match="invalid verify"):
        ledger.upsert("path-1", verify="maybe")


def test_all_enum_values_accepted(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    for i, status in enumerate(("queued", "running", "passed", "failed", "dropped")):
        ledger.upsert(f"path-{i}", status=status)
    for i, verify in enumerate(("pass", "fail", "pending", "n/a")):
        ledger.upsert(f"path-{i}", verify=verify)
    assert len(ledger.rows()) == 5


def test_write_stable_format(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    ledger = Ledger(path)
    ledger.upsert("path-1", approach="baseline", status="passed", verify="pass", gpu_min="12")
    expected = f"{HEADER}\n{SEPARATOR}\n| path-1 | baseline | passed |  |  | pass |  |  | 12 |  |\n"
    assert path.read_text() == expected


def test_none_clears_cell(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-1", failure_cause="oom")
    ledger.upsert("path-1", failure_cause=None)
    assert ledger.rows()[0]["failure_cause"] == ""


def test_hand_edited_whitespace_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    path.write_text(
        "|  path_id |approach|   status | final_train_loss | final_eval_loss "
        "|verify | failure_cause | retry_of | gpu_min |  run_url  |\n"
        "|---|:---:|---|---|---|---|---|---|---|---|\n"
        "|  path-1  |  baseline  |running|  2.31 |   | pending |  | | 12  |   |\n"
    )
    rows = Ledger(path).rows()
    assert rows == [
        {
            "path_id": "path-1",
            "approach": "baseline",
            "status": "running",
            "final_train_loss": "2.31",
            "final_eval_loss": "",
            "verify": "pending",
            "failure_cause": "",
            "retry_of": "",
            "gpu_min": "12",
            "run_url": "",
        }
    ]


def test_hand_edited_short_row_padded_and_rewritten_stable(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    path.write_text(f"{HEADER}\n{SEPARATOR}\n| path-1 | baseline | queued |\n")
    ledger = Ledger(path)
    assert ledger.rows()[0]["run_url"] == ""
    ledger.upsert("path-1", status="running")
    assert path.read_text() == f"{HEADER}\n{SEPARATOR}\n| path-1 | baseline | running |  |  |  |  |  |  |  |\n"


def test_pipe_in_field_sanitized_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    ledger = Ledger(path)
    ledger.upsert("path-1", status="failed", failure_cause="CUDA error | device-side assert")
    assert ledger.rows()[0]["failure_cause"] == "CUDA error / device-side assert"
    rows = Ledger(path).rows()
    assert len(rows) == 1
    assert rows[0]["failure_cause"] == "CUDA error / device-side assert"
    assert rows[0]["status"] == "failed"


def test_newline_in_field_collapsed_to_single_space(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    ledger = Ledger(path)
    ledger.upsert("path-1", failure_cause="oom\nat step 12\r\n\nretry smaller batch ")
    assert ledger.rows()[0]["failure_cause"] == "oom at step 12 retry smaller batch"
    rows = Ledger(path).rows()
    assert len(rows) == 1
    assert rows[0]["failure_cause"] == "oom at step 12 retry smaller batch"


def test_clean_value_not_warned_and_unchanged(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-1", approach="baseline lr=3e-4")
    assert ledger.rows()[0]["approach"] == "baseline lr=3e-4"


def test_unexpected_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    path.write_text("| path_id | wrong |\n| --- | --- |\n| path-1 | x |\n")
    with pytest.raises(ValueError, match="unexpected ledger header"):
        Ledger(path)


def test_columns_match_contract() -> None:
    assert COLUMNS == (
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
