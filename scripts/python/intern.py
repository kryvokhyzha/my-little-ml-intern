"""Enforcement CLI: verify / budget / ledger / deps gates."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import fire
import rootutils
from dotenv import find_dotenv, load_dotenv
from loguru import logger


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)

from intern.budget import BudgetGate, load_budget, load_budget_profile, save_budget
from intern.deps import check_project
from intern.ledger import Ledger
from intern.publish import publish_run
from intern.report import render_gates
from intern.verify import verify_run


_BUDGET_ACTIONS = ("init", "status", "can-launch", "can-retry", "record-launch", "record-retry", "record-gpu-h")


def _resolve_experiment(experiment: str | int, experiments_root: str | None) -> Path:
    """Resolve `001` or `001-slug` to a directory; exit 2 when missing or ambiguous."""
    base = Path(experiments_root) if experiments_root is not None else root / "experiments"
    # fire parses a bare `--experiment 1` as int; normalize back to the NNN prefix.
    name = f"{experiment:03d}" if isinstance(experiment, int) else str(experiment)
    exact = base / name
    if exact.is_dir():
        return exact
    matches = sorted(path for path in base.glob(f"{name}-*") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if matches:
        logger.error("Ambiguous experiment '{}': matches {}", name, ", ".join(path.name for path in matches))
    else:
        logger.error("No experiment matching '{}' under {}", name, base)
    raise SystemExit(2)


def _as_checks(checks: str | list | tuple | None) -> list[str] | None:
    if checks is None:
        return None
    if isinstance(checks, str):
        return [item.strip() for item in checks.split(",") if item.strip()]
    return [str(item) for item in checks]


def verify(
    experiment: str | int,
    vocab_size: int | None = None,
    checks: str | list | tuple | None = None,
    experiments_root: str | None = None,
) -> None:
    experiment_dir = _resolve_experiment(experiment, experiments_root)
    try:
        code = verify_run(experiment_dir, vocab_size=vocab_size, checks=_as_checks(checks))
    except ValueError as err:
        logger.error("{}", err)
        raise SystemExit(2) from None
    raise SystemExit(code)


def _resolve_params(experiment_dir: Path) -> int | None:
    """Compute the experiment model's true param count from HF metadata (no --params needed)."""
    from intern.params import experiment_model_repo, hf_param_count

    repo = experiment_model_repo(root / "configs", experiment_dir.name)
    if repo is None:
        logger.warning(
            "Could not resolve the model repo from {} config — scale ceiling not enforced", experiment_dir.name
        )
        return None
    count = hf_param_count(repo)
    if count is None:
        logger.warning("Could not read param count for '{}' from HF — scale ceiling not enforced", repo)
    else:
        logger.info("Resolved {} = {} params from HF safetensors metadata", repo, count)
    return count


def _budget_init(budget_path: Path, profile: str | None, force: bool) -> None:
    """Seed budget.md from a profile; refuse to clobber recorded spend unless forced."""
    if profile is None:
        logger.error("budget init requires --profile (a name under configs/budget/, e.g. lora)")
        raise SystemExit(2)
    try:
        seeded = load_budget_profile(root / "configs", str(profile))
    except (FileNotFoundError, ValueError) as err:
        logger.error("{}", err)
        raise SystemExit(2) from None
    if budget_path.is_file() and not force:
        try:
            existing = load_budget(budget_path)
        except ValueError:
            existing = None
        if existing is not None and (existing.paths_launched or existing.retries_used or existing.gpu_h_used):
            logger.error(
                "Refusing to overwrite {} with recorded spend "
                "(paths_launched={}, retries_used={}, gpu_h_used={}); pass --force to re-seed",
                budget_path,
                existing.paths_launched,
                existing.retries_used,
                existing.gpu_h_used,
            )
            raise SystemExit(1)
    save_budget(seeded, budget_path)
    logger.info(
        "Seeded {} from profile '{}': {} paths, {} retries/path, {} GPU-h cap, {} param ceiling",
        budget_path,
        profile,
        seeded.max_paths,
        seeded.max_retries_per_path,
        seeded.compute_cap_gpu_h,
        seeded.scale_ceiling_params,
    )
    raise SystemExit(0)


def budget(
    action: str,
    experiment: str | int,
    hours: float | None = None,
    path_id: str | None = None,
    params: int | None = None,
    profile: str | None = None,
    force: bool = False,
    experiments_root: str | None = None,
) -> None:
    if action not in _BUDGET_ACTIONS:
        logger.error("Unknown budget action '{}' (expected {})", action, "|".join(_BUDGET_ACTIONS))
        raise SystemExit(2)
    experiment_dir = _resolve_experiment(experiment, experiments_root)
    budget_path = experiment_dir / "budget.md"
    if action == "init":
        _budget_init(budget_path, profile, force)
    if not budget_path.is_file():
        logger.error("Missing {}", budget_path)
        raise SystemExit(2)
    try:
        gate = BudgetGate(budget_path)
    except ValueError as err:
        logger.error("{}", err)
        raise SystemExit(2) from None

    if action == "status":
        for key, value in asdict(gate.budget).items():
            print(f"{key}: {value}")
        raise SystemExit(0)
    if action == "can-launch":
        resolved = int(params) if params is not None else _resolve_params(experiment_dir)
        allowed, reason = gate.can_launch_path(params=resolved)
        print(reason)
        raise SystemExit(0 if allowed else 1)
    if action == "can-retry":
        if path_id is None:
            logger.error("budget can-retry requires --path-id")
            raise SystemExit(2)
        ledger_path = experiment_dir / "ledger.md"
        if not ledger_path.is_file():
            logger.error("Missing {}", ledger_path)
            raise SystemExit(2)
        allowed, reason = gate.can_retry(str(path_id), Ledger(ledger_path))
        print(reason)
        raise SystemExit(0 if allowed else 1)
    if action == "record-launch":
        gate.record_launch()
    elif action == "record-retry":
        gate.record_retry()
    else:  # record-gpu-h
        if hours is None:
            logger.error("budget record-gpu-h requires --hours")
            raise SystemExit(2)
        if float(hours) < 0:
            logger.error("budget record-gpu-h requires --hours >= 0, got {}", hours)
            raise SystemExit(2)
        gate.record_gpu_h(float(hours))
    raise SystemExit(0)


def ledger(
    action: str,
    experiment: str | int,
    experiments_root: str | None = None,
    **fields: object,
) -> None:
    experiment_dir = _resolve_experiment(experiment, experiments_root)
    ledger_path = experiment_dir / "ledger.md"
    if action == "show":
        if not ledger_path.is_file():
            logger.error("Missing {}", ledger_path)
            raise SystemExit(2)
        try:
            rows = Ledger(ledger_path).rows()
        except ValueError as err:
            logger.error("{}", err)
            raise SystemExit(2) from None
        for row in rows:
            print(json.dumps(row))
        raise SystemExit(0)
    if action == "upsert":
        clean = {str(key).replace("-", "_"): value for key, value in fields.items()}
        path_id = clean.pop("path_id", None)
        if path_id is None:
            logger.error("ledger upsert requires --path-id")
            raise SystemExit(2)
        try:
            row = Ledger(ledger_path).upsert(str(path_id), **clean)
        except ValueError as err:
            logger.error("{}", err)
            raise SystemExit(2) from None
        print(json.dumps(row))
        raise SystemExit(0)
    logger.error("Unknown ledger action '{}' (expected upsert|show)", action)
    raise SystemExit(2)


def check(experiment: str | int, experiments_root: str | None = None) -> None:
    """Scaffold gate: 0 all required files present, 1 missing a required file, 2 experiment absent."""
    from intern.scaffold import REQUIRED_FILES, missing_required

    experiment_dir = _resolve_experiment(experiment, experiments_root)
    if experiment_dir.name.startswith("999-"):
        logger.info("{} is 999- scratch — scaffold gate exempt", experiment_dir.name)
        raise SystemExit(0)
    missing = missing_required(experiment_dir)
    if missing:
        logger.error("Scaffold incomplete for {}: missing {}", experiment_dir.name, ", ".join(missing))
        raise SystemExit(1)
    print(f"scaffold complete: {', '.join(REQUIRED_FILES)}")
    raise SystemExit(0)


def status(experiment: str | int, json: bool = False, experiments_root: str | None = None) -> None:
    experiment_dir = _resolve_experiment(experiment, experiments_root)
    render_gates(experiment_dir, as_json=bool(json))
    raise SystemExit(0)


def publish(
    experiment: str | int,
    repo_id: str | None = None,
    private: bool | str = True,
    experiments_root: str | None = None,
) -> None:
    # Only bool or literal true|false — fire turns `--private 0` into int 0, and silently
    # publishing weights world-readable because of a typo'd flag is not acceptable.
    if isinstance(private, bool):
        resolved = private
    elif isinstance(private, str) and private.strip().lower() in ("true", "false"):
        resolved = private.strip().lower() == "true"
    else:
        logger.error("publish --private expects true|false, got {!r} ({})", private, type(private).__name__)
        raise SystemExit(2)
    if not resolved:
        logger.warning("publish --private false: the Hub repo will be WORLD-READABLE — anyone can download it")
    experiment_dir = _resolve_experiment(experiment, experiments_root)
    raise SystemExit(publish_run(experiment_dir, repo_id=repo_id, private=resolved))


def deps(min_age_days: int = 7) -> None:
    lines = check_project(root / "pyproject.toml", min_age_days=min_age_days)
    for line in lines:
        print(line)
    violations = [line for line in lines if not line.startswith("info:")]
    if violations:
        logger.error("{} dependency-age violation(s)", len(violations))
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    fire.Fire(
        {
            "verify": verify,
            "check": check,
            "budget": budget,
            "ledger": ledger,
            "status": status,
            "publish": publish,
            "deps": deps,
        }
    )
