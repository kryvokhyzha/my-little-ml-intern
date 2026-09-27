"""The per-experiment journal: format, validation, and the runtime entries (run start + outcome)."""

import pytest

from intern.journal import KINDS, Journal
from intern.metrics import MetricsLog
from training.runtime import env_stamp, format_stamp, record_run_end, record_run_start


def test_append_creates_header_and_parses_back(tmp_path):
    journal = Journal(tmp_path / "001-demo" / "journal.md")
    journal.append("decision", "raise beta\n 0.0 -> 0.04", path_id="path-2")
    journal.append("lesson", "ceiling, not mechanism")

    text = journal.path.read_text()
    assert text.startswith("# Journal — 001-demo\n")
    entries = journal.entries()
    assert [e["kind"] for e in entries] == ["decision", "lesson"]
    assert entries[0]["path_id"] == "path-2"
    assert entries[0]["text"] == "raise beta 0.0 -> 0.04"  # one entry = one line
    assert entries[1]["path_id"] is None


@pytest.mark.parametrize(("kind", "text"), [("musing", "x"), ("decision", "   ")])
def test_append_rejects_unknown_kind_and_empty_text(tmp_path, kind, text):
    with pytest.raises(ValueError):
        Journal(tmp_path / "journal.md").append(kind, text)


def test_entries_of_missing_journal_is_empty(tmp_path):
    assert Journal(tmp_path / "journal.md").entries() == []


def test_kinds_cover_automatic_and_agent_entries():
    assert {"run", "gate"} <= set(KINDS) and {"decision", "lesson", "blocker"} <= set(KINDS)


def test_env_stamp_names_versions_and_device_but_no_paths():
    stamp = env_stamp()
    assert {"python", "torch", "transformers", "trl", "device"} <= set(stamp)
    assert not any(value.startswith("/") for value in stamp.values())


def test_record_run_start_and_end_write_metrics_and_journal(tmp_path, capsys):
    mlog = MetricsLog(tmp_path / "metrics.jsonl")
    record_run_start(tmp_path, mlog, "trl_sft", "run-1", smoke=True)
    record_run_end(tmp_path, 2.5, steps=10)

    start = mlog.read()[0]
    assert start["event"] == "run_start" and start["env"]["trl"] == env_stamp()["trl"]
    assert capsys.readouterr().out.strip() == "VERDICT: TRAIN_OK | final_train_loss=2.5"
    entries = Journal(tmp_path / "journal.md").entries()
    assert [e["kind"] for e in entries] == ["run", "run"]
    assert entries[0]["text"].startswith("trl_sft run started (smoke=True) — python=")
    assert entries[1]["text"] == "VERDICT: TRAIN_OK | final_train_loss=2.5 | steps=10"
    assert format_stamp({"a": "1", "b": "2"}) == "a=1 b=2"


def test_setup_failure_prints_and_journals_train_fail(tmp_path, capsys):
    from training.runtime import record_setup_failure

    with pytest.raises(ImportError), record_setup_failure(tmp_path):
        raise ImportError("vllm is not installed")

    assert "VERDICT: TRAIN_FAIL | ImportError: vllm is not installed" in capsys.readouterr().out
    assert (
        Journal(tmp_path / "journal.md").entries()[-1]["text"]
        == "VERDICT: TRAIN_FAIL | ImportError: vllm is not installed"
    )
