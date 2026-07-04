"""Validation of .claude/skills/*/SKILL.md against the conventions in docs/001-architecture.md."""

import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
SKILL_FILES = sorted(SKILLS_DIR.glob("*/SKILL.md"))
SKILL_IDS = [path.parent.name for path in SKILL_FILES]

FENCE_RE = re.compile(r"^\s*```(\S+)", re.MULTILINE)
REFERENCE_RE = re.compile(r"references/([\w.-]+\.md)")
REPO_PATH_RE = re.compile(
    r"scripts/python/[\w<>.-]+\.py"
    r"|scripts/bash/[\w<>.-]+\.sh"
    r"|configs/[\w<>./-]+\.yaml"
    r"|docs/[\w<>.-]+\.md"
)
# Templated or example-only paths: NNN/<> placeholders, `foo` example slugs, gitignored 999- scratch.
PLACEHOLDER_MARKERS = ("NNN", "<", "foo", "999-")

skill_case = pytest.mark.parametrize("skill_file", SKILL_FILES, ids=SKILL_IDS)


def _split_skill(skill_file: Path) -> tuple[dict, str]:
    lines = skill_file.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0].strip() == "---", f"{skill_file}: missing opening '---' frontmatter marker"
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    assert closing is not None, f"{skill_file}: missing closing '---' frontmatter marker"
    frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    assert isinstance(frontmatter, dict), f"{skill_file}: frontmatter is not a YAML mapping"
    return frontmatter, "\n".join(lines[closing + 1 :])


def test_skills_discovered():
    assert SKILL_FILES, f"no SKILL.md files found under {SKILLS_DIR}"


@skill_case
def test_frontmatter_has_name_and_description(skill_file):
    frontmatter, _ = _split_skill(skill_file)
    for key in ("name", "description"):
        value = frontmatter.get(key)
        assert isinstance(value, str) and value.strip(), f"{skill_file}: frontmatter '{key}' missing or empty"


@skill_case
def test_name_matches_directory(skill_file):
    frontmatter, _ = _split_skill(skill_file)
    assert frontmatter["name"] == skill_file.parent.name, (
        f"{skill_file}: frontmatter name '{frontmatter['name']}' != directory '{skill_file.parent.name}'"
    )


@skill_case
def test_body_under_300_lines(skill_file):
    _, body = _split_skill(skill_file)
    line_count = len(body.splitlines())
    assert line_count < 300, (
        f"{skill_file}: body has {line_count} lines (must stay under 300 — move detail to references/)"
    )


@skill_case
def test_references_exist_and_are_mentioned(skill_file):
    _, body = _split_skill(skill_file)
    skill_dir = skill_file.parent
    for name in set(REFERENCE_RE.findall(body)):
        assert (skill_dir / "references" / name).is_file(), f"{skill_file}: mentions missing references/{name}"
    references_dir = skill_dir / "references"
    if references_dir.is_dir():
        for ref in sorted(references_dir.glob("*.md")):
            assert f"references/{ref.name}" in body, f"{skill_file}: never mentions existing references/{ref.name}"


@skill_case
def test_no_markdown_code_fences(skill_file):
    text = skill_file.read_text(encoding="utf-8")
    for language in FENCE_RE.findall(text):
        assert language.lower() != "markdown", (
            f"{skill_file}: uses a ```markdown fence — prettier corrupts nested markdown fences; use ```text instead"
        )


@skill_case
def test_referenced_repo_paths_exist(skill_file):
    _, body = _split_skill(skill_file)
    for path in sorted(set(REPO_PATH_RE.findall(body))):
        if any(marker in path for marker in PLACEHOLDER_MARKERS):
            continue
        assert (REPO_ROOT / path).exists(), f"{skill_file}: references '{path}' which does not exist in the repo"
