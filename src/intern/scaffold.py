"""Scaffold-completeness gate: refuse an experiment missing its required authored files.

Agents sometimes forget to create a required artifact (e.g. run.md). This is the code
that refuses — `intern.py check`, the `status` dashboard, and the publish gate all read
``missing_required``. Gate outputs (verify.md, results.md) and optional cards (data.md,
research.md, board.md) are deliberately NOT required here: the first appear after their
gates, the second only when applicable.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_FILES: tuple[str, ...] = ("task.md", "plan.md", "budget.md", "ledger.md", "run.md")


def missing_required(experiment_dir: Path | str) -> list[str]:
    """Return the REQUIRED_FILES absent from the experiment dir, in canonical order (empty = complete)."""
    experiment_dir = Path(experiment_dir)
    return [name for name in REQUIRED_FILES if not (experiment_dir / name).is_file()]
