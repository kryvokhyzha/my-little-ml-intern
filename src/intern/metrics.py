import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _jsonl_safe(value: Any) -> Any:
    """Replace non-finite floats with their string form so every line stays valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class MetricsLog:
    """Append-only JSONL stream of metric and event records for one experiment run.

    Two record kinds (see docs/001-architecture.md):
    metrics ``{"ts", "step", "split", "name", "value"}`` and
    events ``{"ts", "event", ...}`` (e.g. ``alert``, ``meta``).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append_metric(self, step: int, name: str, value: float, split: str = "train") -> None:
        self._append({"ts": _utc_now_iso(), "step": step, "split": split, "name": name, "value": value})

    def append_event(self, event: str, **fields: Any) -> None:
        self._append({"ts": _utc_now_iso(), "event": event, **fields})

    def _append(self, record: dict[str, Any]) -> None:
        record = {key: _jsonl_safe(value) for key, value in record.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A truncated trailing line from a killed run must not poison the whole stream.
                logger.warning("Skipping malformed JSONL line {} in {}", lineno, self.path)
                continue
            if not isinstance(record, dict):
                logger.warning("Skipping non-dict JSONL line {} in {}", lineno, self.path)
                continue
            records.append(record)
        return records

    def _metrics(self, name: str, split: str | None) -> list[dict[str, Any]]:
        return [
            record
            for record in self.read()
            if "event" not in record
            and record.get("name") == name
            and _is_number(record.get("value"))
            and (split is None or record.get("split") == split)
        ]

    def final(self, name: str, split: str | None = None) -> float | None:
        """Return the value of the last logged metric matching ``name`` (and ``split`` when given)."""
        matches = self._metrics(name, split)
        return matches[-1]["value"] if matches else None

    def series(self, name: str, split: str | None = None) -> list[tuple[int, float]]:
        return [
            (record["step"], record["value"])
            for record in self._metrics(name, split)
            if isinstance(record.get("step"), int) and not isinstance(record.get("step"), bool)
        ]

    def meta(self, key: str) -> Any | None:
        """Return the value of the LATEST ``meta`` event for ``key``, or None when never written."""
        value: Any | None = None
        for record in self.read():
            if record.get("event") == "meta" and record.get("key") == key:
                value = record.get("value")
        return value
