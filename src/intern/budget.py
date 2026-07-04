"""Budget caps and spend tally for an experiment (parses/emits budget.md, docs/001 contract)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .ledger import Ledger

_LINE_RE = re.compile(r"^(?P<key>\w+):\s*(?P<value>.*?)\s*$")
_INT_FIELDS = ("max_paths", "max_retries_per_path", "scale_ceiling_params", "paths_launched", "retries_used")
_FLOAT_FIELDS = ("compute_cap_gpu_h", "gpu_h_used")


@dataclass
class Budget:
    max_paths: int = 2
    max_retries_per_path: int = 2
    compute_cap_gpu_h: float = 2.0
    scale_ceiling_params: int = 200_000_000
    token_budget: int | None = None
    paths_launched: int = 0
    retries_used: int = 0
    gpu_h_used: float = 0.0


def _fmt_float(value: float) -> str:
    return str(round(value, 6))


def load_budget(path: Path | str) -> Budget:
    """Parse a budget.md file into a Budget.

    Raises:
        ValueError: When a required `key: value` line is missing or unparsable.

    """
    raw: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE_RE.match(stripped)
        if match:
            raw[match["key"]] = match["value"]

    missing = sorted(field for field in (*_INT_FIELDS, *_FLOAT_FIELDS, "token_budget") if field not in raw)
    if missing:
        raise ValueError(f"budget file {path} is missing keys: {', '.join(missing)}")

    try:
        token_raw = raw["token_budget"]
        return Budget(
            **{field: int(raw[field]) for field in _INT_FIELDS},
            **{field: float(raw[field]) for field in _FLOAT_FIELDS},
            token_budget=None if token_raw.lower() in ("null", "none") else int(token_raw),
        )
    except (TypeError, ValueError) as err:
        raise ValueError(f"budget file {path} has an unparsable value: {err}") from err


def save_budget(budget: Budget, path: Path | str) -> None:
    token = "null" if budget.token_budget is None else str(budget.token_budget)
    text = (
        "# Budget\n"
        "\n"
        f"max_paths: {budget.max_paths}\n"
        f"max_retries_per_path: {budget.max_retries_per_path}\n"
        f"compute_cap_gpu_h: {_fmt_float(budget.compute_cap_gpu_h)}\n"
        f"scale_ceiling_params: {budget.scale_ceiling_params}\n"
        f"token_budget: {token}\n"
        "\n"
        "## Spent\n"
        "\n"
        f"paths_launched: {budget.paths_launched}\n"
        f"retries_used: {budget.retries_used}\n"
        f"gpu_h_used: {_fmt_float(budget.gpu_h_used)}\n"
    )
    Path(path).write_text(text, encoding="utf-8")


class BudgetGate:
    """Blocking gate over an experiment's budget.md; every denial carries a human-readable reason."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.budget = load_budget(self.path)

    def can_launch_path(self, params: int | None = None) -> tuple[bool, str]:
        budget = self.budget
        if budget.paths_launched >= budget.max_paths:
            return False, f"denied: paths_launched {budget.paths_launched} >= max_paths {budget.max_paths}"
        if params is not None and params > budget.scale_ceiling_params:
            return False, f"denied: params {params} > scale_ceiling_params {budget.scale_ceiling_params}"
        compute_denial = self._compute_cap_denial()
        if compute_denial:
            return False, compute_denial
        return True, (
            f"allowed: {budget.paths_launched}/{budget.max_paths} paths launched, "
            f"{_fmt_float(budget.gpu_h_used)}/{_fmt_float(budget.compute_cap_gpu_h)} GPU-h used"
        )

    def can_retry(self, path_id: str, ledger: Ledger) -> tuple[bool, str]:
        budget = self.budget
        rows = ledger.rows()
        parents = {row["path_id"]: row["retry_of"] for row in rows}

        def root_of(pid: str) -> str:
            seen: set[str] = set()
            while pid not in seen and parents.get(pid):
                seen.add(pid)
                pid = parents[pid]
            return pid

        # Retries are counted over the whole retry TREE, not direct children: a chain
        # A->B->C must not reset the counter at each link.
        root = root_of(path_id)
        retries = sum(1 for row in rows if row["path_id"] != root and root_of(row["path_id"]) == root)
        if retries >= budget.max_retries_per_path:
            return False, (
                f"denied: {path_id} already has {retries} retries >= max_retries_per_path {budget.max_retries_per_path}"
            )
        global_cap = budget.max_paths * budget.max_retries_per_path
        if budget.retries_used >= global_cap:
            return False, (
                f"denied: retries_used {budget.retries_used} >= global retry cap {global_cap} "
                f"(max_paths * max_retries_per_path)"
            )
        compute_denial = self._compute_cap_denial()
        if compute_denial:
            return False, compute_denial
        return True, f"allowed: {path_id} has {retries}/{budget.max_retries_per_path} retries used"

    def record_launch(self) -> None:
        self.budget.paths_launched += 1
        save_budget(self.budget, self.path)

    def record_retry(self) -> None:
        self.budget.retries_used += 1
        save_budget(self.budget, self.path)

    def record_gpu_h(self, hours: float) -> None:
        self.budget.gpu_h_used += hours
        save_budget(self.budget, self.path)

    def _compute_cap_denial(self) -> str | None:
        budget = self.budget
        if budget.gpu_h_used >= budget.compute_cap_gpu_h:
            return (
                f"denied: gpu_h_used {_fmt_float(budget.gpu_h_used)}"
                f" >= compute_cap_gpu_h {_fmt_float(budget.compute_cap_gpu_h)}"
            )
        return None
