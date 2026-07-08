# 006 — Adopting this template for a new project

Checklist for creating a new repo from `my-little-ml-intern`. Work top to
bottom; each step says what to change, where, and how to verify. The whole pass
takes ~15 minutes plus however long your README description takes.

## 1. Create the repo

Use GitHub's "Use this template" button (or
`gh repo create <name> --template kryvokhyzha/my-little-ml-intern`), clone it,
and create the environment:

```bash
make uv_create_venv && make uv_install_deps && make pre_commit_install
```

## 2. Rename: every place the old name lives

The name appears in two forms — the repo slug (`my-little-ml-intern`) and the
GitHub owner (`kryvokhyzha`). Mechanical pass first:

```bash
git grep -l 'my-little-ml-intern' -- ':!uv.lock' | xargs sed -i '' 's/my-little-ml-intern/<your-repo-name>/g'
git grep -l 'kryvokhyzha'         -- ':!uv.lock' | xargs sed -i '' 's/kryvokhyzha/<your-github-user>/g'
uv lock   # regenerates the project name inside uv.lock — never sed the lockfile
```

(Linux `sed` drops the `''` after `-i`.) What that pass touches, so you can
review the diff consciously:

| File                                  | What the name does there                                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                      | `[project].name` — also flows into `uv.lock` on the next `uv lock`                                               |
| `configs/main.yaml`                   | `project_name:` — drives tracking project names, publish repo ids (`<project>-<experiment>`), notification cards |
| `README.md`                           | Title + CI badge URL (owner **and** repo segments)                                                               |
| `AGENTS.md`                           | Project-overview first line                                                                                      |
| `docs/001-architecture.md`            | Positioning paragraph                                                                                            |
| `src/intern/publish.py`               | Model-card tag + footer ("Trained with …") — `tests/test_publish.py` asserts this string; they change together   |
| `.claude/skills/track-experiments/**` | `trackio … --project <name>` example commands (12+ occurrences)                                                  |

Then the identity files the sed can't decide for you:

- `pyproject.toml` — `authors`, `description`, `keywords`.
- `LICENSE` — copyright holder (or a different license entirely).
- `.github/workflows/` — nothing name-bound inside, but confirm Actions are
  enabled on the new repo so the CI badge resolves.

Verify: `git grep -n 'my-little-ml-intern\|kryvokhyzha' -- ':!uv.lock'` returns
nothing, and `uv run pytest -q` is green.

## 3. Environment

```bash
cp .env.example .env
```

Minimum to fill: `HF_TOKEN` + `HF_USER` (publish gate), `GITHUB_TOKEN`
(literature research). Optional but recommended: `TG_*`/`SLACK_*` for
notifications, `WANDB_API_KEY` if you track with wandb. Project names for the
trackers default to `project_name` from `configs/main.yaml`; override per
machine with `TRACKIO_PROJECT` / `WANDB_PROJECT` in `.env` when needed.

## 4. Replace the example experiment

`000-tiny-sft-smoke` is the repo's plumbing fixture — tests and docs reference
it. **Keep it.** `001-pi-mono-sft` is a worked example (gemma-4-E2B QLoRA on
coding-agent traces); keep it as a reference until your first real experiment,
then remove it in one commit:

```bash
git rm scripts/python/001-pi-mono-sft.py scripts/python/prep-pi-mono-sft.py \
       configs/001-pi-mono-sft.yaml configs/prep-pi-mono-sft.yaml \
       configs/data/pi_mono_sft.yaml configs/data/pi_mono_raw.yaml \
       configs/model/gemma_4_e2b_it.yaml configs/model/gemma_4_e2b_it_4bit.yaml \
       src/data/pi_mono.py tests/test_pi_mono.py tests/test_prep_pi_mono_sft.py \
       docs/007-example-pi-mono-sft.md
git rm -r experiments/001-pi-mono-sft
uv run pytest -q   # must stay green after the removal
```

Your own experiments then start at `001` (or the next free number) via the
`new-experiment` skill — model and dataset are one-file additions under
`configs/model/` and `configs/data/` following the `_target_` pattern in
[docs/001-architecture.md](001-architecture.md).

## 5. README rewrite

The new README describes **your project only** — no mention of the "ML intern"
harness, no "built from my-little-ml-intern" attribution, no template lineage.
The machinery speaks through the workflow sections; where the repo came from is
not part of your project's story.

- Delete the `> [!NOTE]` template banner at the top — your repo is no longer the
  template.
- Replace every `<!-- template: … -->` marked section with your content — in
  particular, the About section's "personal ML intern" description gets replaced
  wholesale by what YOUR project trains and why.
- Delete the "Vendoring into your project" section — it's template-facing
  (people vendor from the template, not from your project).
- Fix the CI badge (done by the sed in step 2 if your owner/repo names were
  substituted).
- Sections worth adding for a real project: the datasets involved (and their
  privacy status), current experiment index with one-line results,
  hardware/lanes you actually use.
- Keep: "The loop" diagram, Skills table, Notifications — they document the
  workflow your collaborators (and agents) will actually use, without naming its
  origin.

## 6. Docs and agent guidance

- `AGENTS.md` — rewrite the "Project overview" paragraph for your project; the
  skills table below it is regenerated by the `sync_agents_md` pre-commit hook,
  don't edit it by hand.
- `docs/001-architecture.md` — the contract still holds; update only the
  Positioning paragraph. New docs take the next free `NNN-` prefix.
- `experiments/`, `docs/007` — covered by step 4.

## 7. Final verification

```bash
uv run pytest -q                                   # full suite green
make pre_commit_run                                # all hooks green
uv run python scripts/python/intern.py deps        # dependency-age gate
uv run python scripts/python/000-tiny-sft-smoke.py smoke_test=true   # VERDICT: TRAIN_OK
```

Push, confirm CI is green and the badge renders, and you're on your own
experiments from here — start with the `new-experiment` skill.
