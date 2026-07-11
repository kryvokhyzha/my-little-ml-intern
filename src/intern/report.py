"""Gates dashboard for one experiment: verify verdicts, budget caps/spent, ledger rows (docs/001 contract)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from helper.display import get_console, is_interactive, print_table
from intern.budget import load_budget
from intern.ledger import Ledger


_VERDICT_HEAD_RE = re.compile(r"^VERDICT:\s*(?P<name>\S+)\s*=\s*(?P<status>\S+)\s*$")

_VERIFY_COLUMNS = ["name", "status", "value", "threshold"]
_BUDGET_COLUMNS = ["cap", "limit", "spent"]
_LEDGER_COLUMNS = ["path_id", "status", "verify", "final_train_loss", "final_eval_loss", "retry_of"]


def _parse_verify(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    checks: list[dict[str, str]] = []
    overall: str | None = None
    judgment: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VERDICT:"):
            parts = line.split(" | ")
            head = _VERDICT_HEAD_RE.match(parts[0])
            if head is None:
                logger.warning("Unparsable VERDICT line in {}: {!r}", path, line)
                continue
            checks.append(
                {
                    "name": head["name"],
                    "status": head["status"],
                    "value": parts[1].removeprefix("value=") if len(parts) > 1 else "",
                    "threshold": parts[2].removeprefix("threshold=") if len(parts) > 2 else "",
                    "detail": " | ".join(parts[3:]),
                }
            )
        elif line.startswith("OVERALL:"):
            overall = line.removeprefix("OVERALL:").strip()
        elif line.startswith("JUDGMENT:"):
            judgment = line.removeprefix("JUDGMENT:").strip()
    return {"checks": checks, "overall": overall, "judgment": judgment}


def _parse_budget(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        budget = load_budget(path)
    except ValueError as err:
        logger.warning("Unparsable {}: {}", path, err)
        return None
    return {
        "caps": {
            "max_paths": budget.max_paths,
            "max_retries_per_path": budget.max_retries_per_path,
            "compute_cap_gpu_h": budget.compute_cap_gpu_h,
            "scale_ceiling_params": budget.scale_ceiling_params,
            "token_budget": budget.token_budget,
        },
        "spent": {
            "paths_launched": budget.paths_launched,
            "retries_used": budget.retries_used,
            "gpu_h_used": budget.gpu_h_used,
        },
    }


def _parse_ledger(path: Path) -> list[dict[str, str]]:
    try:
        return Ledger(path).rows()
    except ValueError as err:
        logger.warning("Unparsable {}: {}", path, err)
        return []


def gates_summary(experiment_dir: Path) -> dict[str, Any]:
    """Collect the gate states; each section tolerates a missing artifact (None / empty list)."""
    from intern.scaffold import missing_required

    experiment_dir = Path(experiment_dir)
    return {
        "scaffold": missing_required(experiment_dir),
        "verify": _parse_verify(experiment_dir / "verify.md"),
        "budget": _parse_budget(experiment_dir / "budget.md"),
        "ledger": _parse_ledger(experiment_dir / "ledger.md"),
    }


def _verdict_line(check: dict[str, str]) -> str:
    return (
        f"VERDICT: {check['name']} = {check['status']} "
        f"| value={check['value']} | threshold={check['threshold']} | {check['detail']}"
    )


def _render_plain(summary: dict[str, Any]) -> None:
    scaffold = summary["scaffold"]
    print("SCAFFOLD | complete" if not scaffold else f"SCAFFOLD | MISSING: {', '.join(scaffold)}")
    verify = summary["verify"]
    if verify is not None:
        for check in verify["checks"]:
            print(_verdict_line(check))
        if verify["overall"] is not None:
            print(f"OVERALL: {verify['overall']}")
        if verify["judgment"] is not None:
            print(f"JUDGMENT: {verify['judgment']}")
    budget = summary["budget"]
    if budget is not None:
        caps, spent = budget["caps"], budget["spent"]
        print(
            f"BUDGET | paths={spent['paths_launched']}/{caps['max_paths']}"
            f" | retries={spent['retries_used']}/{caps['max_retries_per_path']}"
            f" | gpu_h={spent['gpu_h_used']}/{caps['compute_cap_gpu_h']}"
        )
    for row in summary["ledger"]:
        print(f"LEDGER | {row['path_id']} | status={row['status']} | verify={row['verify']}")


def _render_rich(summary: dict[str, Any]) -> None:
    scaffold = summary["scaffold"]
    get_console().print("SCAFFOLD: complete" if not scaffold else f"SCAFFOLD: MISSING {', '.join(scaffold)}")
    verify = summary["verify"]
    verify_rows = [] if verify is None else [[c[column] for column in _VERIFY_COLUMNS] for c in verify["checks"]]
    print_table("Verify checks", _VERIFY_COLUMNS, verify_rows)
    if verify is not None:
        console = get_console()
        if verify["overall"] is not None:
            console.print(f"OVERALL: {verify['overall']}")
        if verify["judgment"] is not None:
            console.print(f"JUDGMENT: {verify['judgment']}")

    budget = summary["budget"]
    budget_rows = []
    if budget is not None:
        caps, spent = budget["caps"], budget["spent"]
        budget_rows = [
            ["paths", str(caps["max_paths"]), str(spent["paths_launched"])],
            ["retries", str(caps["max_retries_per_path"]), str(spent["retries_used"])],
            ["gpu_h", str(caps["compute_cap_gpu_h"]), str(spent["gpu_h_used"])],
            ["scale_ceiling_params", str(caps["scale_ceiling_params"]), ""],
            ["token_budget", str(caps["token_budget"]), ""],
        ]
    print_table("Budget", _BUDGET_COLUMNS, budget_rows)

    ledger_rows = [[row[column] for column in _LEDGER_COLUMNS] for row in summary["ledger"]]
    print_table("Ledger", _LEDGER_COLUMNS, ledger_rows)


def render_gates(experiment_dir: Path, as_json: bool = False) -> None:
    """Print the gates dashboard: JSON, rich tables (interactive), or greppable plain lines."""
    summary = gates_summary(experiment_dir)
    if as_json:
        print(json.dumps(summary))
        return
    if is_interactive():
        _render_rich(summary)
        return
    _render_plain(summary)
