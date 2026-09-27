"""Per-experiment lab journal: an append-only, timestamped log of runs, gate outcomes, and decisions.

``experiments/NNN-<slug>/journal.md`` is the chronological record that the other artifacts lack:
the ledger holds one row per path, results.md holds the conclusion, and the journal holds WHEN
each run started (with its library versions), what each gate said, and why the plan changed.
The training adapters and the ``intern.py`` gate commands append entries automatically; agents
append ``decision`` / ``observation`` / ``lesson`` / ``blocker`` entries through
``intern.py journal ... add``. An agent that resumes after a context reset reads the journal
first.

Line format (one entry per line, greppable)::

    - 2026-09-27T10:40:12Z [decision] path-2: raise beta 0.0 -> 0.04 because kl_ref SKIPped
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


KINDS = ("run", "gate", "decision", "observation", "lesson", "blocker")
_LINE_RE = re.compile(r"^- (?P<ts>\S+) \[(?P<kind>[a-z]+)\] (?:(?P<path_id>path-[\w.-]+): )?(?P<text>.*)$")


class Journal:
    """Append-only journal.md for one experiment directory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, kind: str, text: str, path_id: str | None = None) -> str:
        """Append one entry and return the written line.

        Raises:
            ValueError: When ``kind`` is not in :data:`KINDS` or ``text`` is empty.

        """
        if kind not in KINDS:
            raise ValueError(f"journal kind must be one of {KINDS}, got {kind!r}")
        flat = " ".join(str(text).split())  # one entry = one line
        if not flat:
            raise ValueError("journal text must not be empty")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        prefix = f"{path_id}: " if path_id else ""
        line = f"- {ts} [{kind}] {prefix}{flat}"
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f"# Journal — {self.path.parent.name}\n\n", encoding="utf-8")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return line

    def entries(self) -> list[dict[str, str | None]]:
        """Parse every entry line; header and free-form lines are ignored."""
        if not self.path.is_file():
            return []
        entries = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            match = _LINE_RE.match(raw)
            if match is not None:
                entries.append(match.groupdict())
        return entries
