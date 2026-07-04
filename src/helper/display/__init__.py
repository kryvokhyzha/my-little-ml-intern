"""Rich-or-plain display helpers: rich output on interactive terminals, greppable text otherwise."""

import os
import sys
from functools import lru_cache
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def is_interactive() -> bool:
    """Whether rich (colored, boxed) output should be used instead of plain greppable text."""
    if os.environ.get("FORCE_RICH") == "1":
        return True
    if "CLAUDECODE" in os.environ or "NO_COLOR" in os.environ:
        return False
    if os.environ.get("JSON_LOGS", "false").lower() == "true":
        return False
    if os.environ.get("COLORIZE", "true").lower() != "true":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


@lru_cache(maxsize=2)
def get_console(stderr: bool = False) -> Console:
    force_terminal = True if os.environ.get("FORCE_RICH") == "1" else None
    return Console(stderr=stderr, force_terminal=force_terminal)


def _kv_line(prefix: str, pairs: dict[str, Any]) -> str:
    return " | ".join([prefix, *(f"{k}={v}" for k, v in pairs.items())])


def print_table(title: str | None, columns: list[str], rows: list[list[str]]) -> None:
    if is_interactive():
        table = Table(title=title)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*(str(cell) for cell in row))
        get_console().print(table)
        return

    widths = [len(column) for column in columns]
    for row in rows:
        for i, cell in enumerate(row[: len(widths)]):
            widths[i] = max(widths[i], len(str(cell)))
    lines = [] if title is None else [title]
    for row in [columns, *rows]:
        lines.append("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip())
    print("\n".join(lines))


def run_header(title: str, meta: dict[str, Any]) -> None:
    if is_interactive():
        body = "\n".join(f"{k}: {v}" for k, v in meta.items())
        get_console().print(Panel(body, title=title))
        return
    print(_kv_line("RUN_START", {"title": title, **meta}))


def run_footer(status: str, meta: dict[str, Any]) -> None:
    if is_interactive():
        body = "\n".join(f"{k}: {v}" for k, v in meta.items())
        get_console().print(Panel(body, title=f"{status}"))
        return
    print(_kv_line("RUN_END", {"status": status, **meta}))


__all__ = ["is_interactive", "get_console", "print_table", "run_header", "run_footer"]
