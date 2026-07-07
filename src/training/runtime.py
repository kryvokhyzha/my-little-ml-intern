"""Framework-neutral run plumbing shared by every training lane."""

from __future__ import annotations

import io
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from omegaconf import DictConfig, OmegaConf


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def smoke_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "smoke_test")) or os.environ.get("SMOKE_TEST") == "1"


def apply_tracking_env(cfg: DictConfig) -> None:
    """Map tracking config onto the env vars the wandb callback reads; trackio takes kwargs instead."""
    if OmegaConf.select(cfg, "tracking.backend") != "wandb":
        # trackio: transformers' TrackioCallback takes project via TrainingArguments.project
        # (build_args forwards it) and never passes group — Lightning wires trackio directly.
        return
    project = OmegaConf.select(cfg, "tracking.project")
    if project is not None:
        # transformers' WandbCallback reads only WANDB_PROJECT (default "huggingface"); the
        # resolved config value already honors a WANDB_PROJECT env override via interpolation.
        os.environ["WANDB_PROJECT"] = str(project)
    group = OmegaConf.select(cfg, "tracking.group")
    if group is not None:
        # wandb.init reads WANDB_RUN_GROUP; transformers' wandb callback has no group kwarg.
        os.environ.setdefault("WANDB_RUN_GROUP", str(group))


class StderrTee(io.TextIOBase):
    """Duplicates writes to the real stderr and a log file."""

    def __init__(self, fh: TextIO) -> None:
        self._fh = fh

    def write(self, s: str) -> int:
        sys.__stderr__.write(s)
        if not self._fh.closed:
            self._fh.write(s)
        return len(s)

    def flush(self) -> None:
        sys.__stderr__.flush()
        # A logging handler may hold this tee past the file's close and flush it at GC time.
        if not self._fh.closed:
            self._fh.flush()


def run_with_stderr_tee(train_fn: Callable[[], Any], experiment_dir: Path) -> Any:
    log_path = experiment_dir / "logs" / "stderr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # One stderr.log per run: append mode would let a previous path's traceback
    # permanently fail stderr_scan for every later retry.
    with log_path.open("w", encoding="utf-8") as fh:
        original = sys.stderr
        sys.stderr = StderrTee(fh)
        try:
            return train_fn()
        except BaseException as exc:
            fh.write(traceback.format_exc())
            print(f"VERDICT: TRAIN_FAIL | {type(exc).__name__}: {exc}")
            raise
        finally:
            sys.stderr = original
