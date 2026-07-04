"""Blocking publish gate: only verified runs leave the repo, always with their reproducibility bundle."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .ledger import Ledger
from .verify import verify_run


if TYPE_CHECKING:
    from huggingface_hub import HfApi

_BUNDLE_DOCS = ("task.md", "plan.md", "budget.md", "ledger.md", "verify.md", "results.md")
_MAX_REPO_TRIES = 5
_PROJECT_NAME_RE = re.compile(r"^project_name:\s*(?P<name>\S+)\s*$")

_SCRUB_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hf-token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("slack-token", re.compile(r"xox[abp]-")),
    ("aws-access-key-id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github-token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("home-dir-path", re.compile(r"/(?:Users|home)/[^/\s]+/")),
)
_SCRUB_SUFFIXES = frozenset({".md", ".jsonl", ".yaml"})


def _api() -> HfApi:
    from huggingface_hub import HfApi

    return HfApi()


def _repo_root(experiment_dir: Path) -> Path:
    return experiment_dir.resolve().parent.parent


def _default_repo_name(experiment_dir: Path) -> str:
    main_yaml = _repo_root(experiment_dir) / "configs" / "main.yaml"
    if main_yaml.is_file():
        for line in main_yaml.read_text(encoding="utf-8").splitlines():
            match = _PROJECT_NAME_RE.match(line.strip())
            if match:
                project = match["name"].strip("\"'")
                return f"{project}-{experiment_dir.name}"
    return experiment_dir.name


def _resolve_model_dir(ckpts: Path) -> Path | None:
    if not ckpts.is_dir():
        return None
    candidates = [child for child in ckpts.iterdir() if child.is_dir() and any(child.glob("*.safetensors"))]
    if candidates:
        return max(candidates, key=lambda child: child.stat().st_mtime)
    if any(ckpts.glob("*.safetensors")):
        return ckpts
    return None


def _is_conflict(err: Exception) -> bool:
    status = getattr(getattr(err, "response", None), "status_code", None)
    message = str(err).lower()
    return status == 409 or "409" in message or "exist" in message or "conflict" in message


def _create_repo(api: HfApi, repo_id: str, private: bool) -> str | None:
    for attempt in range(1, _MAX_REPO_TRIES + 1):
        candidate = repo_id if attempt == 1 else f"{repo_id}-{attempt}"
        try:
            api.create_repo(repo_id=candidate, private=private, exist_ok=False)
        except Exception as err:
            if _is_conflict(err):
                logger.warning("Repo {} already exists — trying the next suffix", candidate)
                continue
            logger.error("create_repo failed for {}: {}", candidate, err)
            return None
        logger.info("Created repo {} (private={})", candidate, private)
        return candidate
    logger.error("No free repo id after {} tries starting from {}", _MAX_REPO_TRIES, repo_id)
    return None


def _stage_bundle(experiment_dir: Path, staging: Path) -> Path:
    staging.mkdir(parents=True)
    sources = [experiment_dir / name for name in _BUNDLE_DOCS]
    sources.append(experiment_dir / "logs" / "samples.jsonl")
    sources.append(_repo_root(experiment_dir) / "configs" / f"{experiment_dir.name}.yaml")
    for source in sources:
        if source.is_file():
            shutil.copy2(source, staging / source.name)
        else:
            logger.warning("Bundle file missing, skipped: {}", source)
    return staging


def _scrub_bundle(staged_dir: Path) -> list[str]:
    """Return 'file: pattern-name' hits for token-shaped strings and home-dir paths; never the matched value."""
    hits: list[str] = []
    for path in sorted(staged_dir.rglob("*")):
        if not path.is_file() or path.suffix not in _SCRUB_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern_name, pattern in _SCRUB_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path.relative_to(staged_dir)}: {pattern_name}")
    return hits


def _winner_paragraph(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if "winner" in line.lower() and line.strip() and not line.startswith(("#", "|", "```")):
            paragraph: list[str] = []
            for candidate in lines[index:]:
                if not candidate.strip() or candidate.startswith(("#", "|", "```")):
                    break
                paragraph.append(candidate.strip())
            return " ".join(paragraph)
    return None


def _first_table(lines: list[str]) -> str | None:
    table: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            table.append(line.rstrip())
        elif table:
            break
    return "\n".join(table) if table else None


def _first_code_block(lines: list[str]) -> str | None:
    block: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("```"):
            block.append(line)
            if inside:
                return "\n".join(block)
            inside = True
        elif inside:
            block.append(line)
    return None


def _model_card(results_md: str, experiment_name: str) -> str:
    lines = results_md.splitlines()
    title = next((line.removeprefix("# ").strip() for line in lines if line.startswith("# ")), experiment_name)
    winner = _winner_paragraph(lines)
    table = _first_table(lines)
    commands = _first_code_block(lines)
    parts = [f"# {title}", ""]
    if winner:
        parts += [winner, ""]
    if table:
        parts += ["## Path comparison", "", table, ""]
    if commands:
        parts += ["## Reproduce", "", commands, ""]
    parts += [
        "---",
        "",
        f"Trained with my-little-ml-intern — the full reproducibility bundle for "
        f"`{experiment_name}` lives under `bundle/` in this repo.",
        "",
    ]
    return "\n".join(parts)


def publish_run(experiment_dir: Path | str, repo_id: str | None = None, private: bool = True) -> int:
    """Run the blocking publish gate; 0 published, 1 gate refused, 2 missing artifacts/credentials."""
    experiment_dir = Path(experiment_dir)

    verify_code = verify_run(experiment_dir)
    if verify_code == 2:
        logger.error("Publish refused: verify found missing artifacts (exit 2)")
        return 2
    if verify_code != 0:
        logger.error("Publish refused: live verify exited {} — never trust a stale verify.md", verify_code)
        return 1
    logger.info("Publish gate: live verify passed")

    results_path = experiment_dir / "results.md"
    if not results_path.is_file():
        logger.error("Publish refused: missing {}", results_path)
        return 2

    rows = Ledger(experiment_dir / "ledger.md").rows()
    passed = [row for row in rows if row["status"] == "passed" and row["verify"] == "pass"]
    if not passed:
        logger.error("Publish refused: no ledger row with status=passed and verify=pass")
        return 1
    logger.info("Publish gate: passed+verified ledger row(s): {}", ", ".join(row["path_id"] for row in passed))

    model_dir = _resolve_model_dir(experiment_dir / "ckpts")
    if model_dir is None:
        logger.error("Publish refused: no *.safetensors model dir under {}", experiment_dir / "ckpts")
        return 2
    logger.info("Publish gate: model dir {}", model_dir)

    if not os.environ.get("HF_TOKEN"):
        logger.error("Publish refused: HF_TOKEN is not set")
        return 2

    api = _api()
    if repo_id is None:
        user = os.environ.get("HF_USER")
        if not user:
            try:
                user = str(api.whoami()["name"])
            except Exception as err:
                logger.error("Publish refused: cannot resolve HF user ({}); set HF_USER or pass --repo-id", err)
                return 2
        repo_id = f"{user}/{_default_repo_name(experiment_dir)}"
    logger.info("Publishing {} to {} (private={})", experiment_dir.name, repo_id, private)

    final_repo_id: str | None = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            bundle = _stage_bundle(experiment_dir, staging / "bundle")
            card = _model_card(results_path.read_text(encoding="utf-8"), experiment_dir.name)
            (staging / "README.md").write_text(card, encoding="utf-8")

            if os.environ.get("INTERN_SKIP_BUNDLE_SCRUB") == "1":
                logger.warning("INTERN_SKIP_BUNDLE_SCRUB=1 — bundle secret/path scrub bypassed")
            else:
                hits = _scrub_bundle(staging)
                if hits:
                    for hit in hits:
                        logger.error("Bundle scrub hit: {}", hit)
                    logger.error(
                        "Publish refused: staged bundle contains token-shaped strings or home-dir paths "
                        "(set INTERN_SKIP_BUNDLE_SCRUB=1 to bypass)"
                    )
                    return 2

            final_repo_id = _create_repo(api, repo_id, private)
            if final_repo_id is None:
                return 2
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=final_repo_id,
                repo_type="model",
                commit_message=f"model from {experiment_dir.name}",
            )
            api.upload_folder(
                folder_path=str(bundle),
                path_in_repo="bundle",
                repo_id=final_repo_id,
                repo_type="model",
                commit_message=f"reproducibility bundle from {experiment_dir.name}",
            )
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=final_repo_id,
                repo_type="model",
                commit_message="model card",
            )
    except Exception as err:
        logger.error("Upload to {} failed: {}", final_repo_id or repo_id, err)
        return 2

    url = f"https://huggingface.co/{final_repo_id}"
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Published\n\n<{url}>\n")
    logger.info("Published {} -> {}", experiment_dir.name, url)
    return 0
