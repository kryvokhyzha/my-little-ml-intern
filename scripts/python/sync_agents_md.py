"""Sync the AGENTS.md '## Project skills' table from .claude/skills/*/SKILL.md frontmatter."""

import difflib
import re
import sys
from pathlib import Path

import fire
import rootutils
import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)

START_MARKER = "<!-- skills-table:start -->"
END_MARKER = "<!-- skills-table:end -->"
HEADING = "## Project skills"
INTRO_PREFIX = "Project-level skills under"
CLOSING_PREFIX = "Invoke via"
USE_WHEN_LIMIT = 90
RUN_HINT = "run `uv run python scripts/python/sync_agents_md.py` to regenerate"


def _frontmatter(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            parsed = yaml.safe_load("\n".join(lines[1:index]))
            return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_skills(skills_dir: Path) -> list[dict[str, str]]:
    skills = []
    for skill_md in sorted(Path(skills_dir).glob("*/SKILL.md")):
        frontmatter = _frontmatter(skill_md.read_text(encoding="utf-8"))
        name = str(frontmatter.get("name") or skill_md.parent.name).strip()
        description = str(frontmatter.get("description") or "").strip()
        skills.append({"name": name, "description": description})
    return sorted(skills, key=lambda skill: skill["name"])


def _use_when(description: str, limit: int = USE_WHEN_LIMIT) -> str:
    sentence = re.split(r"(?<=[.!?])\s", description.strip(), maxsplit=1)[0].strip().rstrip(".")
    if len(sentence) <= limit:
        return sentence
    truncated = sentence[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—–-")
    return f"{truncated}…"


def render_table(skills: list[dict[str, str]]) -> str:
    headers = ["Skill", "Use when"]
    rows = [[f"`{skill['name']}`", _use_when(skill["description"])] for skill in skills]
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header) for i, header in enumerate(headers)
    ]

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), separator, *(fmt(row) for row in rows)])


def render_block(skills: list[dict[str, str]]) -> str:
    return f"{START_MARKER}\n\n{render_table(skills)}\n\n{END_MARKER}"


def _normalized(block: str) -> list[str]:
    lines = []
    for raw in block.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        # Prettier re-pads table cells and separator dashes; compare shape, not padding.
        if re.fullmatch(r"[|\s:-]+", line):
            line = "|---|"
        lines.append(line)
    return lines


def _wrap_legacy_section(text: str, block: str) -> str | None:
    lines = text.splitlines()
    if HEADING not in lines:
        return None
    heading = lines.index(HEADING)
    intro = next((i for i in range(heading + 1, len(lines)) if lines[i].startswith(INTRO_PREFIX)), None)
    if intro is None:
        return None
    closing = next((i for i in range(intro + 1, len(lines)) if lines[i].startswith(CLOSING_PREFIX)), None)
    if closing is None:
        return None
    new_lines = [*lines[: intro + 1], "", block, "", *lines[closing:]]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")


def sync(agents_md_path: Path, skills_dir: Path, check: bool = False) -> int:
    agents_md_path, skills_dir = Path(agents_md_path), Path(skills_dir)
    if not agents_md_path.is_file():
        logger.error("AGENTS.md not found: {}", agents_md_path)
        return 2
    if not skills_dir.is_dir():
        logger.error("Skills directory not found: {}", skills_dir)
        return 2

    text = agents_md_path.read_text(encoding="utf-8")
    skills = parse_skills(skills_dir)
    table = render_table(skills)
    block = render_block(skills)

    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start != -1 and end > start:
        current = text[start + len(START_MARKER) : end]
        if _normalized(current) == _normalized(table):
            return 0
        if check:
            diff = "\n".join(
                difflib.unified_diff(_normalized(current), _normalized(table), "AGENTS.md", "rendered", lineterm="")
            )
            logger.error("AGENTS.md skills table out of sync — {}\n{}", RUN_HINT, diff)
            return 1
        new_text = text[:start] + block + text[end + len(END_MARKER) :]
        agents_md_path.write_text(new_text, encoding="utf-8")
        logger.info("Rewrote skills table in {} ({} skills)", agents_md_path, len(skills))
        return 0

    wrapped = _wrap_legacy_section(text, block)
    if wrapped is None:
        logger.error(
            "Neither skills-table markers nor the '{} / {}... / {}...' structure found in {}",
            HEADING,
            INTRO_PREFIX,
            CLOSING_PREFIX,
            agents_md_path,
        )
        return 2
    if check:
        logger.error("Skills-table markers missing in {} — {}", agents_md_path, RUN_HINT)
        return 1
    agents_md_path.write_text(wrapped, encoding="utf-8")
    logger.info("Inserted skills-table markers into {} ({} skills)", agents_md_path, len(skills))
    return 0


def main(check: bool = False, agents_md: str | None = None, skills_dir: str | None = None) -> None:
    agents_md_path = Path(agents_md) if agents_md else root / "AGENTS.md"
    skills_path = Path(skills_dir) if skills_dir else root / ".claude" / "skills"
    raise SystemExit(sync(agents_md_path, skills_path, check=check))


if __name__ == "__main__":
    fire.Fire(main)
