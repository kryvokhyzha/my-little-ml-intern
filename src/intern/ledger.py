"""Per-path experiment ledger backed by the 10-column markdown table in ledger.md (docs/001 contract)."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger


COLUMNS: tuple[str, ...] = (
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
STATUS_VALUES = frozenset({"queued", "running", "passed", "failed", "dropped"})
VERIFY_VALUES = frozenset({"pass", "fail", "pending", "n/a"})
_ENUM_COLUMNS: dict[str, frozenset[str]] = {"status": STATUS_VALUES, "verify": VERIFY_VALUES}


_LINE_BREAKS = re.compile(r"[\r\n]+")


def _format_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(cells) + " |"


def _sanitize_cell(value: str) -> str:
    """Make a cell value safe for the markdown table: no pipes, no line breaks."""
    return _LINE_BREAKS.sub(" ", value.replace("|", "/")).strip()


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._rows = self._read()

    def rows(self) -> list[dict[str, str]]:
        self._rows = self._read()
        return [dict(row) for row in self._rows]

    def upsert(self, path_id: str, /, **fields: object) -> dict[str, str]:
        """Create or update the row for `path_id` and persist the table.

        Raises:
            ValueError: On unknown columns or invalid status/verify values.

        """
        if "path_id" in fields:
            raise ValueError("path_id is the positional key; do not pass it as a field")
        unknown = sorted(set(fields) - set(COLUMNS))
        if unknown:
            raise ValueError(f"unknown ledger columns: {', '.join(unknown)}; allowed: {', '.join(COLUMNS)}")
        for column, allowed in _ENUM_COLUMNS.items():
            if column in fields and fields[column] is not None:
                value = str(fields[column])
                if value and value not in allowed:
                    raise ValueError(f"invalid {column} {value!r}; allowed: {', '.join(sorted(allowed))}")

        self._rows = self._read()
        row = next((r for r in self._rows if r["path_id"] == path_id), None)
        if row is None:
            row = dict.fromkeys(COLUMNS, "")
            row["path_id"] = path_id
            self._rows.append(row)
        for column, value in fields.items():
            if value is None:
                row[column] = ""
                continue
            raw = str(value)
            clean = _sanitize_cell(raw)
            if clean != raw:
                logger.warning("Sanitized ledger cell {} for {}: {!r} -> {!r}", column, path_id, raw, clean)
            row[column] = clean
        self.write()
        return dict(row)

    def write(self) -> None:
        lines = [_format_row(COLUMNS), _format_row(("---",) * len(COLUMNS))]
        lines += [_format_row(tuple(row[column] for column in COLUMNS)) for row in self._rows]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read(self) -> list[dict[str, str]]:
        """Parse the table back, tolerating hand-edited extra whitespace around cells."""
        if not self.path.exists():
            return []
        rows: list[dict[str, str]] = []
        header_seen = False
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and all(cell and set(cell) <= {"-", ":"} for cell in cells):
                continue  # alignment/separator row
            if not header_seen:
                if cells != list(COLUMNS):
                    raise ValueError(f"unexpected ledger header in {self.path}: expected {list(COLUMNS)}, got {cells}")
                header_seen = True
                continue
            if len(cells) > len(COLUMNS):
                raise ValueError(f"ledger row in {self.path} has {len(cells)} cells, expected {len(COLUMNS)}")
            cells += [""] * (len(COLUMNS) - len(cells))
            rows.append(dict(zip(COLUMNS, cells, strict=True)))
        return rows
