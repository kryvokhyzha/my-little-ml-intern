"""Framework-neutral run plumbing shared by every training lane."""

from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

from omegaconf import DictConfig, OmegaConf


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def smoke_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "smoke_test")) or os.environ.get("SMOKE_TEST") == "1"


def run_seed(cfg: DictConfig) -> int:
    """``cfg.seed``, or 42 when the config sets none. A seed of 0 stays 0."""
    seed = OmegaConf.select(cfg, "seed")
    return 42 if seed is None else int(seed)


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
            record_run_failure(experiment_dir, exc)
            raise
        finally:
            sys.stderr = original


def record_run_failure(experiment_dir: Path, exc: BaseException) -> None:
    """Print the one-line failure VERDICT and journal it (rank zero only)."""
    verdict = f"VERDICT: TRAIN_FAIL | {type(exc).__name__}: {exc}"
    print(verdict)
    if is_main_process():
        _journal(experiment_dir, verdict)


@contextmanager
def record_setup_failure(experiment_dir: Path) -> Iterator[None]:
    """Record a failure BEFORE training — model/data load, validation, trainer construction.

    Without it a setup crash printed no VERDICT and left the journal at "run started".
    """
    try:
        yield
    except Exception as exc:
        record_run_failure(experiment_dir, exc)
        raise


# Packages whose version decides whether two runs' numbers are comparable (see AGENTS.md
# "Upgrade dependencies", step 5: a trainer upgrade can change the loss or the masking).
_STAMP_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "lightning",
    "vllm",
    "kernels",
    "bitsandbytes",
)


def _git_revision() -> str | None:
    import subprocess

    cwd = os.environ.get("PROJECT_ROOT")
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}+dirty" if status else sha


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_stamp() -> dict[str, str]:
    """Record what produced a run: python and library versions, the git commit, and the device.

    The stamp rides on the ``run_start`` event, the journal, and the ``ENV:`` line of verify.md, so a
    recorded number always names the trainer version that produced it. It never holds a path.
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    stamp = {"python": platform.python_version()}
    for package in _STAMP_PACKAGES:
        try:
            stamp[package] = version(package)
        except PackageNotFoundError:
            continue
    git = _git_revision()
    if git is not None:
        stamp["git"] = git
    stamp["device"] = _device()
    return stamp


def format_stamp(stamp: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in stamp.items())


def _journal(experiment_dir: Path, text: str) -> None:
    from intern.journal import Journal

    Journal(experiment_dir / "journal.md").append("run", text)


def record_run_start(experiment_dir: Path, mlog: Any, task: str, run_name: Any, smoke: bool) -> None:
    """Append the ``run_start`` event (the verify scope boundary) with the env stamp, and journal it."""
    stamp = env_stamp()
    mlog.append_event("run_start", task=task, run_name=run_name, smoke=smoke, env=stamp)
    _journal(experiment_dir, f"{task} run started (smoke={smoke}) — {format_stamp(stamp)}")


def record_run_end(experiment_dir: Path, final_train_loss: float | None, steps: int) -> None:
    """Print the one-line success VERDICT and journal it with the step count."""
    verdict = f"VERDICT: TRAIN_OK | final_train_loss={final_train_loss}"
    print(verdict)
    _journal(experiment_dir, f"{verdict} | steps={steps}")
