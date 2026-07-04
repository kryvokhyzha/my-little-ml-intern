import json
from datetime import UTC, datetime

import pytest

from intern.metrics import MetricsLog


@pytest.fixture
def log(tmp_path):
    return MetricsLog(tmp_path / "metrics.jsonl")


def test_append_metric_writes_one_json_line_with_utc_ts(log):
    log.append_metric(10, "loss", 2.31)

    lines = log.path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 10
    assert record["split"] == "train"
    assert record["name"] == "loss"
    assert record["value"] == 2.31
    ts = datetime.fromisoformat(record["ts"])
    assert ts.utcoffset() is not None
    assert ts.astimezone(UTC) == ts


def test_append_event_roundtrip(log):
    log.append_event("alert", level="WARN", message="loss=9.8 at step 120 — lr likely too high, try lr*0.1")
    log.append_event("meta", key="param_count", value=124_000_000)

    records = log.read()
    assert len(records) == 2
    assert records[0]["event"] == "alert"
    assert records[0]["level"] == "WARN"
    assert records[1] == {"ts": records[1]["ts"], "event": "meta", "key": "param_count", "value": 124_000_000}


def test_read_missing_file_returns_empty(log):
    assert log.read() == []


def test_read_skips_malformed_trailing_line(log):
    log.append_metric(1, "loss", 3.0)
    with log.path.open("a") as stream:
        stream.write('{"ts": "2026-07-04T12:00:00+00:00", "step": 2, "spl')  # simulated kill mid-write

    records = log.read()
    assert len(records) == 1
    assert records[0]["value"] == 3.0


def test_final_filters_by_name_and_split(log):
    log.append_metric(1, "loss", 2.5, split="train")
    log.append_metric(1, "loss", 3.0, split="eval")
    log.append_metric(2, "loss", 2.0, split="train")

    assert log.final("loss") == 2.0
    assert log.final("loss", split="train") == 2.0
    assert log.final("loss", split="eval") == 3.0
    assert log.final("missing") is None
    assert log.final("loss", split="test") is None


def test_series_returns_step_value_tuples_in_order(log):
    log.append_metric(1, "loss", 2.5)
    log.append_metric(2, "loss", 2.0)
    log.append_metric(2, "grad_norm", 1.1)
    log.append_metric(3, "loss", 1.8, split="eval")

    assert log.series("loss", split="train") == [(1, 2.5), (2, 2.0)]
    assert log.series("loss") == [(1, 2.5), (2, 2.0), (3, 1.8)]
    assert log.series("missing") == []


def test_meta_returns_latest_value(log):
    assert log.meta("param_count") is None

    log.append_event("meta", key="param_count", value=100)
    log.append_event("meta", key="vocab_size", value=32_000)
    log.append_event("meta", key="param_count", value=200)
    log.append_event("alert", level="INFO", message="not a meta")

    assert log.meta("param_count") == 200
    assert log.meta("vocab_size") == 32_000
    assert log.meta("planned_tokens") is None


def test_read_skips_non_dict_json_lines(log):
    log.append_metric(1, "loss", 2.5)
    with log.path.open("a") as stream:
        stream.write('[1, 2, 3]\n"just a string"\n42\nnull\n')

    records = log.read()
    assert len(records) == 1
    assert records[0]["value"] == 2.5


def test_final_and_series_skip_wrong_typed_records(log):
    log.append_metric(1, "loss", 2.5)
    with log.path.open("a") as stream:
        stream.write(json.dumps({"ts": "x", "step": 2, "split": "train", "name": "loss", "value": "2.0"}) + "\n")
        stream.write(json.dumps({"ts": "x", "step": 3, "split": "train", "name": ["loss"], "value": 1.9}) + "\n")
        stream.write(json.dumps({"ts": "x", "step": 4, "split": "train", "name": "loss", "value": True}) + "\n")
        stream.write(json.dumps({"ts": "x", "step": 5, "split": "train", "name": "loss", "value": None}) + "\n")

    assert log.final("loss") == 2.5
    assert log.series("loss") == [(1, 2.5)]


def test_series_skips_records_with_non_int_step(log):
    log.append_metric(1, "loss", 2.5)
    with log.path.open("a") as stream:
        stream.write(json.dumps({"ts": "x", "step": "2", "split": "train", "name": "loss", "value": 2.0}) + "\n")

    assert log.series("loss") == [(1, 2.5)]
    assert log.final("loss") == 2.0


def test_nan_metric_value_stays_parseable_and_is_skipped(log):
    log.append_metric(1, "loss", 2.5)
    log.append_metric(2, "loss", float("nan"))

    lines = log.path.read_text().splitlines()
    assert json.loads(lines[1])["value"] == "nan"
    assert log.final("loss") == 2.5
    assert log.series("loss") == [(1, 2.5)]


def test_infinite_values_serialized_as_strings(log):
    log.append_metric(1, "grad_norm", float("inf"))
    log.append_event("meta", key="weird", value=float("-inf"))

    records = log.read()
    assert records[0]["value"] == "inf"
    assert records[1]["value"] == "-inf"
    assert log.final("grad_norm") is None
    assert log.meta("weird") == "-inf"


def test_metric_named_like_event_field_is_not_confused(log):
    log.append_metric(1, "kl", 0.2)
    log.append_event("meta", key="kl", value=999)

    assert log.final("kl") == 0.2
    assert log.meta("kl") == 999
