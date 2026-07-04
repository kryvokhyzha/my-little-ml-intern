"""Unit tests for scripts/python/sync_agents_md.py (tmp fixtures only — never the real AGENTS.md)."""

import importlib.util
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    module_path = REPO_ROOT / "scripts" / "python" / "sync_agents_md.py"
    spec = importlib.util.spec_from_file_location("sync_agents_md", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

SKILL_TMPL = """---
name: {name}
description: {description}
---

# {name}

Body.
"""

AGENTS_WITH_MARKERS = """# AGENTS.md

Intro prose.

## Project skills

Project-level skills under `.claude/skills/` (auto-discoverable):

<!-- skills-table:start -->

| Skill         | Use when |
| ------------- | -------- |
| `stale-skill` | Old row  |

<!-- skills-table:end -->

Invoke via `Skill` (or `/name`) when the request matches.

## Next section

More prose.
"""

AGENTS_LEGACY = """# AGENTS.md

## Project skills

Project-level skills under `.claude/skills/` (auto-discoverable):

| Skill            | Use when |
| ---------------- | -------- |
| `old-row-skill`  | Old row  |

Invoke via `Skill` (or `/name`) when the request matches.

## Next section
"""


@pytest.fixture
def skills_dir(tmp_path):
    base = tmp_path / "skills"
    for name, description in (
        ("beta-skill", "Second skill for ordering checks. Trailing sentence ignored."),
        ("alpha-skill", "Scaffold things quickly. Use when the user asks for scaffolding."),
    ):
        skill = base / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(SKILL_TMPL.format(name=name, description=description))
    return base


@pytest.fixture
def agents_md(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text(AGENTS_WITH_MARKERS)
    return path


def test_parse_skills_sorted(skills_dir):
    skills = mod.parse_skills(skills_dir)
    assert [skill["name"] for skill in skills] == ["alpha-skill", "beta-skill"]
    assert skills[0]["description"].startswith("Scaffold things quickly.")


def test_use_when_takes_first_sentence():
    assert mod._use_when("Do the thing. Use whenever asked.") == "Do the thing"


def test_use_when_truncates_at_word_boundary():
    sentence = (
        "Alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november "
        "oscar papa quebec romeo sierra tango"
    )
    result = mod._use_when(f"{sentence}. Second sentence here.")
    assert result.endswith("…")
    assert len(result) <= mod.USE_WHEN_LIMIT + 1
    assert sentence.startswith(result[:-1])
    assert sentence[len(result) - 1] == " "


def test_render_table_pads_columns(skills_dir):
    lines = mod.render_table(mod.parse_skills(skills_dir)).splitlines()
    assert lines[0].startswith("| Skill")
    assert set(lines[1]) <= {"|", "-", " "}
    assert "`alpha-skill`" in lines[2]
    assert "`beta-skill`" in lines[3]
    assert len({len(line) for line in lines}) == 1


def test_sync_rewrites_block_between_markers(agents_md, skills_dir):
    assert mod.sync(agents_md, skills_dir) == 0
    text = agents_md.read_text()
    assert "`alpha-skill`" in text
    assert "stale-skill" not in text
    assert text.count(mod.START_MARKER) == 1
    assert text.count(mod.END_MARKER) == 1
    assert "Invoke via `Skill` (or `/name`) when the request matches." in text
    assert "## Next section" in text


def test_check_flags_stale_then_passes_after_sync(agents_md, skills_dir):
    assert mod.sync(agents_md, skills_dir, check=True) == 1
    assert agents_md.read_text() == AGENTS_WITH_MARKERS
    assert mod.sync(agents_md, skills_dir) == 0
    assert mod.sync(agents_md, skills_dir, check=True) == 0


def test_check_tolerates_reformatted_padding(agents_md, skills_dir):
    mod.sync(agents_md, skills_dir)
    agents_md.write_text(re.sub(r" \| ", "   |   ", agents_md.read_text()))
    assert mod.sync(agents_md, skills_dir, check=True) == 0


def test_sync_inserts_markers_into_legacy_section(tmp_path, skills_dir):
    agents = tmp_path / "AGENTS.md"
    agents.write_text(AGENTS_LEGACY)
    assert mod.sync(agents, skills_dir) == 0
    text = agents.read_text()
    assert "old-row-skill" not in text
    intro = text.index("Project-level skills under")
    assert intro < text.index(mod.START_MARKER) < text.index("`alpha-skill`") < text.index(mod.END_MARKER)
    assert text.index(mod.END_MARKER) < text.index("Invoke via `Skill`")
    assert "## Next section" in text
    assert mod.sync(agents, skills_dir, check=True) == 0


def test_check_fails_when_markers_absent(tmp_path, skills_dir):
    agents = tmp_path / "AGENTS.md"
    agents.write_text(AGENTS_LEGACY)
    assert mod.sync(agents, skills_dir, check=True) == 1
    assert agents.read_text() == AGENTS_LEGACY


def test_exit_2_when_structure_missing(tmp_path, skills_dir):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# AGENTS.md\n\nNo skills section here.\n")
    assert mod.sync(agents, skills_dir) == 2
    assert mod.sync(agents, skills_dir, check=True) == 2


def test_exit_2_when_agents_md_missing(tmp_path, skills_dir):
    assert mod.sync(tmp_path / "missing.md", skills_dir) == 2
