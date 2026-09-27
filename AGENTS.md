# AGENTS.md

This file gives guidance to the AI coding agents that work in this repository.

## Project overview

`my-little-ml-intern` is a personal "ML intern". It gives you Claude Code skills
and a guardrail library that the code enforces for ML/LLM experiments. The stack
is Python 3.13, `uv`, Hydra configs, Loguru, and Fire. `rootutils` reads
`.project-root` and sets the project root from it.

Three parts work together:

- **`.claude/skills/`** — the skill pack. It covers experiment scaffolds, the
  training discipline, verification, tracking, and literature research.
- **`src/intern/`** — the enforcement library. It holds the verification,
  budget, ledger, and dependency-age gates. These gates exit with a nonzero
  code. The skills give instructions. These scripts refuse the run.
- **`src/training/`** — the lane adapters. They map Hydra configs onto TRL (SFT,
  preference, online RL with tool environments, on-policy distillation,
  self-distillation, and the async vLLM lanes), PyTorch Lightning, and axolotl.
  The axolotl lane takes rendered YAML for a remote GPU machine. axolotl is
  never a local dependency.

[docs/001-architecture.md](docs/001-architecture.md) holds the full contract:
the module APIs, the artifact formats, and the skill conventions. Read that
document before you change `src/intern`, `src/training`, or any skill.

## Tech stack and key conventions

- **Python**: `>=3.11,<3.14`, target `py313`.
- **Package manager**: `uv`. Never use `pip`, `poetry`, or `conda` directly
  here. The lockfile is `uv.lock`.
- **Build backend**: `hatchling`. The package lives under `src/`.
- **Configuration**: Hydra (`configs/`, entrypoint `configs/main.yaml`).
- **Logging**: Loguru via `src/helper/logging` (singleton `LoggerConfig`). Env
  vars: `ENV_MODE`, `LOG_LEVEL`, `JSON_LOGS`, `COLORIZE`. Custom levels:
  `WARNONCE`, `DEPRECATED`.
- **CLI**: `fire` for command-line entry points.
- **Display/UX**: `rich` for terminal output.
- **Path/root resolution**: `rootutils`. Use it instead of a path that you
  compute from `__file__`.
- **Training**: `torch`, `lightning`, `trl` (+`peft`), `transformers`,
  `datasets`, `accelerate`. axolotl runs remotely via rendered YAML only.
- **Tracking**: `trackio` (primary) / `wandb`. The Hydra `tracking` group
  selects the backend. Never hardcode a backend.
- **Dependency freshness rule**: use the latest versions, but only the releases
  published **≥ 1 week ago**. `intern.deps` checks this rule. Run
  `uv run python scripts/python/intern.py deps`.

## Repository layout

```
configs/        # Hydra configs: main.yaml + groups (hydra/, model/, data/, trainer/, tracking/, compute/, budget/)
scripts/        # python/ (incl. intern.py CLI) and bash/ (notify.sh, gpu_probe.sh)
src/            # Importable package code
  helper/       # display/ (rich) and logging/ (LoggerConfig singleton) — template-generic
  data/         # loading.py (split loading) + synthetic.py (smoke fixtures)
  intern/       # enforcement library: verify, budget, ledger, callbacks, deps, traces
  training/     # runtime/models/sampling shared; trl/ subpackage, lightning_adapter, axolotl_adapter
experiments/    # NNN-<slug>/ run artifacts (task/plan/budget/ledger/journal/verify/results)
docs/           # Plans, analyses, design notes (see "Docs conventions")
tests/          # pytest suite for src/
trash/          # Gitignored scratchpad for throwaway scripts/output
pyproject.toml  # Project + tool config (ruff, pytest, nbqa)
Makefile        # Common uv / pre-commit shortcuts
.env.example    # Committed template for required env vars; copy to .env
.pre-commit-config.yaml
.project-root   # Marker file for rootutils — do not delete
```

You create some of these directories on demand. Create a directory when you need
it. Do not scaffold an empty directory.

## Common commands

Prefer a `make` target when one exists:

| Task                      | Command                                              |
| ------------------------- | ---------------------------------------------------- |
| Create venv (Python 3.13) | `make uv_create_venv`                                |
| Install deps (frozen)     | `make uv_install_deps`                               |
| Upgrade deps              | `make uv_install_deps_with_upgrade`                  |
| Show installed            | `make uv_show_deps` / `make uv_show_deps_tree`       |
| Install pre-commit hooks  | `make pre_commit_install`                            |
| Run pre-commit on all     | `make pre_commit_run`                                |
| Run a script              | `uv run python scripts/python/<name>.py`             |
| Run tests                 | `uv run pytest`                                      |
| Lint / format (manual)    | `uv run ruff check --fix .` / `uv run ruff format .` |

Always run Python with `uv run ...`. This command uses the locked environment.

## Code style

Ruff enforces these rules (`pyproject.toml`):

- Line length **120**, target `py313`.
- Rules: `E, F, W, I, D` (pycodestyle, pyflakes, isort, pydocstyle).
- isort: 2 blank lines after imports.
- Docstrings: Ruff keeps the D100–D107 checks **disabled**, so a public module,
  class, or function does not need one. Use Google-style docstrings.
  `src/helper/logging/__init__.py` shows the style.
- `nbqa` lints the notebooks. `nbstripout` strips the outputs on commit.

Additional conventions:

- Prefer `pathlib.Path` over `os.path`.
- Use `loguru.logger` (already configured) — do **not** instantiate
  `logging.getLogger`.
- Use `pydantic` for data models / config validation.
- Use `joblib` for parallelism and caching when appropriate.
- Type-hint new code. Tests can use fewer type hints.

## Writing style (ASD-STE100, with Strunk for prose)

All prose in this repo follows ASD-STE100 Simplified Technical English. Agents
read these files and act on them, so ambiguity causes wrong actions.

Load the `writing-clearly-and-concisely` skill before you write or edit prose.
The scope includes AGENTS.md, skills, `docs/`, experiment artifacts, commit
messages, PR text, error messages, and docstrings. The skill holds the full STE
rule list (`references/asd-ste100.md`), Strunk's _The Elements of Style_, and
the AI-pattern list.

**Sentence rules** — apply everywhere:

- Write in the active voice. Name the actor: "the gate refuses the run", not
  "the run is refused".
- Give one instruction per sentence. Put each step in its own sentence.
- Keep procedural sentences to 20 words. Keep descriptive sentences to 25.
- Keep a procedural paragraph to 6 sentences. Keep a descriptive paragraph to 1
  topic.
- Use simple tenses. Prefer the present tense.
- Keep the articles. Write "the budget gate", not "budget gate".
- Do not use an `-ing` form as a verb. Write "run the smoke test", not "running
  the smoke test". An `-ing` form is allowed inside a technical name.
- Use one word for one meaning. This repo says **gate** for a blocking check,
  **lane** for a trainer or compute adapter, and **path** for one solution
  attempt inside an experiment. Do not swap in synonyms.
- Do not use slang, idioms, or metaphor. Write the literal fact.
- Put a complex condition in a vertical list, not in one long sentence.
- Start each instruction with the verb. Put each condition before its
  instruction: "If verify exits 1, stop the run."
- Do not make a noun cluster of more than three nouns.
- Do not use AI filler: puffery ("pivotal", "crucial"), promotional adjectives
  ("robust", "seamless"), or AI vocabulary ("delve", "leverage"). Give the
  number, the path, or the command instead.

**Two deliberate deviations from the standard:**

1. We do not adopt the STE controlled dictionary of about 900 words. This repo
   needs its own technical vocabulary. Any ML, Python, or tooling term is a
   Technical Name, and this repo allows it.
2. `docs/` explains tradeoffs and rationale, which STE procedures cannot carry.
   Those files keep the sentence rules above but may argue and qualify. Strunk's
   rules govern them: omit needless words, use the positive form, use concrete
   language, and put the emphatic word last.

Where the two guides conflict in an instruction file, STE wins. Where they
conflict in `docs/`, Strunk wins.

## Testing

- Framework: `pytest` + `pytest-env`.
- `pyproject.toml` sets `pythonpath = ["src", "."]` — import as
  `from helper... import ...`.
- Place the tests beside the code, or under a top-level `tests/` directory.
  Create that directory if it does not exist.
- For async code, the default fixture loop scope is `function`.

## Add dependencies

- Runtime: edit `[project].dependencies` in `pyproject.toml`. Then run
  `uv sync --all-extras --no-install-project`.
- Dev / lint / test / notebook: use the matching `[dependency-groups]` group.
- After any dependency change, update `uv.lock`. The pre-commit `uv-lock` hook
  enforces this rule.
- Pin with care. Prefer `~=` for a library that we track closely. Use `>=,<` for
  a broad range.

## Upgrade dependencies (read the changelog first)

Two mechanisms enforce the 1-week freshness rule. Both run at **pre-commit and
CI time**. There is no scheduled job. A stale dependency is only actionable when
someone edits the dependencies.

|                 | `exclude-newer = "7 days"` (`[tool.uv]`)                    | `intern.py deps`                                         |
| --------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| Role            | **enforcement** — blocks the resolution                     | **discovery** — explains it and reports upgrades         |
| Runs at         | `uv lock` (pre-commit `uv-lock` hook; CI `uv lock --check`) | pre-commit hook on `pyproject.toml` edits; CI `deps` job |
| Covers          | every package uv resolves, **incl. transitive**             | `[project].dependencies` floors only                     |
| Too-young floor | unsatisfiable-resolution error                              | names the package, its age, and its release date         |
| Blind to        | what you could upgrade to (it stays silent)                 | transitive deps (the real attack surface)                |

Neither mechanism replaces the other. `exclude-newer` is the supply-chain guard.
A compromised release arrives as a _transitive_ dependency far more often than
as a direct one. Only the resolver setting blocks that release. But on a
**range** specifier, `exclude-newer` reports a too-young floor only as
`Because only <pkg><=<old> is available …`. That message never names the
cooldown. An exact `==` pin does get a clear message. `exclude-newer` also stays
silent about the upgrades that you could take.

`uv run python scripts/python/intern.py deps` lists every dependency that has a
newer **eligible** release (latest, but published ≥ 1 week ago). It also
**prints that package's changelog URL**. Start an upgrade from the release
notes. Never start from the version number alone.

1. **Run the gate** to see what is eligible. A package that the list does not
   name is current, or it has only releases younger than the 1-week floor. The
   second case is a wait, not a block. Note the date when the package becomes
   eligible.
2. **Read the notes** between the pinned floor and the target. Look for breaking
   changes, removals, and behavior changes. Check each change against the code
   that this repo actually calls (`grep -rn "<pkg>" src/ scripts/ configs/`). Do
   not skim the notes for alarming words.
3. **Edit the specifier by hand.** A three-part `~=` pin means
   `>= 1.21.0, < 1.22.0`, so `uv lock --upgrade` alone will NOT move a minor
   version. This behavior is deliberate, because an upgrade is a decision. The
   version string in `pyproject.toml` is the control that you must edit.
4. **Re-lock and verify**. Run `uv sync --all-extras --no-install-project`. Then
   run `uv run pytest`. Then run a smoke run
   (`uv run python scripts/python/000-tiny-sft-smoke.py smoke_test=true`). The
   loss of that smoke run is the canonical cross-version regression check.
5. **Note behavior changes that invalidate recorded numbers.** If the upgrade
   changes the loss or the masking of a trainer, past experiment results are no
   longer comparable across that boundary. Write that fact in the affected
   `results.md`. Do not compare the numbers in silence. Re-run the baseline
   instead.

torch needs two more checks, because `intern.py deps` reads PyPI only. Before a
torch bump, confirm that the configured Linux index (`pytorch-cu129`) serves the
target version. Also confirm that a vLLM release that pins that exact torch
version is ≥ 1 week old. On 2026-09-27, torch 2.14 failed both checks.

A dated, self-expiring entry in `[tool.intern.deps.exceptions]`
(`package = "YYYY-MM-DD"`) is the deliberate exception to the 1-week floor. A
reviewer sees the entry in the diff, and the entry re-arms itself. Remove an
entry after it expires. The dated entry does not unblock `uv lock`. If `uv lock`
fails because every version that satisfies a specifier is too young, also add
the uv option `exclude-newer-package = { pkg = "<approval date>T00:00:00Z" }`.
The timestamp limits the opt-out to the releases up to the approval date. Do not
write `{ pkg = false }`: that form is a **permanent** opt-out with no limit.
Remove both entries together when the dated entry expires.

## Hydra configs

- Entry config: `configs/main.yaml`. Compose groups via `defaults:` lists.
- `configs/hydra/default.yaml` disables the output directory defaults, so Hydra
  creates no `outputs/` directory.
- To add a new config group, create a subdirectory under `configs/`. Then
  reference that subdirectory from `defaults`.

## ML experiments (the core convention)

One experiment number = three artifacts: `scripts/python/NNN-<slug>.py` +
`configs/NNN-<slug>.yaml` + `experiments/NNN-<slug>/`. Scaffold the triple with
the `new-experiment` skill. The `999-` prefix is gitignored scratch, and the
gates do not apply to it. [docs/001-architecture.md](docs/001-architecture.md)
gives the full formats.

Non-negotiable rules — the exit codes of `scripts/python/intern.py` enforce
them:

- **Smoke before scale**: start every training run with `smoke_test=true` (1
  step, tiny slice). The run must print `VERDICT: TRAIN_OK`.
- **Budget gate**: run `intern.py budget --experiment NNN can-launch` before you
  launch a path. Record the spend afterwards.
- **Verify gate**: never write `results.md` and never report success unless
  `intern.py verify` exited 0. A failed gate means the run failed, whatever the
  loss shows. A low loss number is never evidence that the model works.
- **One variable per path**: prefer one Hydra override per experiment path. Each
  hypothesis in `plan.md` needs a mechanism, an expected numeric delta, and a
  falsification condition.
- **Journal**: record each decision that changes the plan with
  `intern.py journal --experiment NNN add --kind decision --text "..."`. The
  gates and the training adapters write the `run` and `gate` entries themselves.
  Read `journal.md` first when you resume an experiment.
- **Comparable numbers only**: compare two runs only when the `ENV:` lines in
  their verify.md name the same `trl` and `transformers` versions. For a QLoRA
  path, the lines must also name the same `bitsandbytes` version.
- A script that imports from `src/` adds `sys.path.insert(0, str(root / "src"))`
  directly after `rootutils.setup_root(...)`. The library imports stay bare
  (`from intern.verify import ...`). Never write `from src....`. The same style
  holds **inside** the `src/` packages: use absolute bare imports
  (`from intern.scaffold import ...`, `from training.runtime import ...`). Do
  not use a relative import (`from .scaffold import ...`).

## Docs conventions (`docs/`)

`docs/` holds the plans, design notes, research summaries, and analyses from
your work. Put any content there that must outlive a single conversation.

- **Write a long analysis to `docs/`, not to the chat only.** When the user asks
  for a plan, an investigation write-up, or a comparison, save it as a file and
  reference that file.
- **Naming**:
  - `NNN-kebab-case-title.md` — numbered, ordered series of substantive docs
    (e.g. `001-data-pipeline.md`, `002-eval-metrics.md`). Use the next free
    3-digit prefix.
  - `099-*.md` — exploratory / experimental notes that are not part of the main
    numbered series.
  - `999-*.md` — temporary or scratch docs. You delete them later, or you fold
    them back into a numbered doc. Git **ignores** them (see `.gitignore`), so
    they are safe for the in-progress notes that you do not want to commit yet.
  - Unprefixed `kebab-case.md` — standalone reference notes that don't belong to
    a sequence (rationales, workflow guides, one-off analyses).
- **PR descriptions, patches, ad-hoc test scripts**: keep them in `trash/`, not
  `docs/`.
- Git ignores `docs/build/`, which holds generated output (e.g. Sphinx). Do not
  place a hand-written note there.
- Write clear Markdown. Include links to source files and lines (e.g.
  `[foo.py:42](src/foo.py)`) where they help.

## Scratch / throwaway work

The repo has two forms of "don't commit this yet". Pick the form that fits:

- **`999-*` prefix** — use it for _in-progress work that still uses the
  project's normal infrastructure_ (Hydra configs, `scripts/python/` layout,
  `docs/` Markdown). The `.gitignore` file excludes:

  - `scripts/python/999-*.py` — exploratory Hydra+Fire scripts (use the normal
    scaffold of the `new-script` skill, and pick `999-` as the prefix).
  - `docs/999-*.md` — temporary notes or drafts. You can promote them later to a
    real `NNN-*.md` doc. Use this prefix when the script or doc must _look like_
    a regular project artifact — paired config, proper imports, etc. — but you
    are not ready to commit it.

- **`trash/` directory** — a gitignored place for _anything that doesn't fit the
  project's structure at all_: `tmp_*.py`, `t.py`, `check_*.py`, `*.patch`,
  draft `pr-desc.md`, downloaded artifacts, one-off diagnostic snippets, etc.
  - Do **not** import from `trash/` in committed code.
  - Do not put long-lived notes here — promote them to `docs/` instead.

To promote a file, rename `999-<slug>.{py,md}` →
`<next-free-NNN>-<slug>.{py,md}`. Also rename the paired config to match the
script. To keep content from `trash/`, move it into the proper directory and
clean it up first. Do not run `git add` directly from `trash/`.

## Environment variables (`.env` / `.env.example`)

- `.env.example` is the committed source of truth for the env surface. It holds
  placeholder values. Keep every comment on its own line, because
  `scripts/bash/notify.sh` shell-sources the file.
- `.env` is **gitignored**. Never commit a secret.
- When you add a new env var that the code reads, also add it to `.env.example`.
  Give it a sensible default or an empty placeholder.

The full read surface, in the groups from `.env.example`:

- **Logging** (`LoggerConfig` reads these; `JSON_LOGS`/`COLORIZE` also switch
  `helper.display` to plain-text output): `ENV_MODE`, `LOG_LEVEL`, `JSON_LOGS`,
  `COLORIZE`; optional: `FORCE_RICH` (force rich output when stdout is not a
  TTY).
- **Hugging Face**: `HF_TOKEN` (write scope for `intern.py publish`; hf_jobs
  lane secret), `HF_USER` (publish repo-id default); optional:
  `HF_XET_HIGH_PERFORMANCE` (faster Xet transfers; replaces the removed
  hf-transfer path), `HF_HUB_DISABLE_TELEMETRY` (opt out of the TRL and Hub
  usage ping), `HF_HOME`, `HF_HUB_OFFLINE`, `HF_DATASETS_CACHE`, `HF_ENDPOINT`
  (private Hub mirror).
- **GitHub**: `GITHUB_TOKEN` (`gh search` in literature-recipe-research needs
  it; an unauthenticated `gh search` refuses; `GH_TOKEN` takes precedence when
  you set it; gh does not auto-load `.env`, so export it or run
  `gh auth login`).
- **Training**: `SMOKE_TEST` (forces the smoke gate in the training adapters),
  `TOKENIZERS_PARALLELISM`; optional: `PYTORCH_ENABLE_MPS_FALLBACK`,
  `CUDA_VISIBLE_DEVICES`.
- **RL environments**: optional `OPENREWARD_API_KEY` (TRL's `OpenRewardSpec`
  reads it when `trainer.env_spec` targets an OpenReward environment).
- **trackio**: optional `TRACKIO_PROJECT` (project name; defaults to
  `project_name` from `configs/main.yaml` via config interpolation),
  `TRACKIO_DIR` (local metrics DB location, defaults to `$HF_HOME/trackio`). You
  configure the HF Space sync with `tracking.space_id` in the configs (created
  private), not with an env var.
- **wandb**: `WANDB_API_KEY` (only when `tracking=wandb`); optional:
  `WANDB_PROJECT` (project name; defaults to `project_name` from
  `configs/main.yaml` via config interpolation), `WANDB_MODE`, `WANDB_DIR`.
- **Notifications** (`scripts/bash/notify.sh`; all optional, per-channel no-op):
  `TG_BOT_TOKEN`, `TG_CHAT_ID`, `SLACK_WEBHOOK_URL`, `SLACK_BOT_TOKEN`,
  `SLACK_CHANNEL_ID`; `PROJECT_NAME` overrides the project label on the cards.

## Project skills

Project-level skills under `.claude/skills/` (auto-discoverable):

<!-- skills-table:start -->

| Skill                           | Use when                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `autoresearch-loop`             | Generational orchestrator that runs many bounded training experiments to find the best…    |
| `distill-traces`                | Turn verified agent traces into training data and run the self-distillation loop — define… |
| `literature-recipe-research`    | Isolated-context literature research that returns a ranked table of training recipes with… |
| `new-doc`                       | Create a new numbered document in docs/ following the NNN-kebab-case-title.md convention   |
| `new-experiment`                | Scaffold the numbered experiment triple for a training run — entrypoint…                   |
| `new-script`                    | Scaffold a new runnable Python entrypoint under scripts/python/ together with a paired…    |
| `publish-model`                 | Publish a verified training run to the Hugging Face Hub through the blocking publish gate… |
| `track-experiments`             | Sets up experiment tracking and runs the alert-driven iteration loop for training runs in… |
| `train-llm`                     | Plan, launch, and monitor LLM training runs — SFT, DPO/KTO, LoRA/QLoRA, online RL…         |
| `verify-run`                    | Run the blocking verification gate on a finished training run and act on the result…       |
| `writing-clearly-and-concisely` | Write and edit the prose that people and agents read in this repo — AGENTS.md, SKILL.md…   |

<!-- skills-table:end -->

Invoke via `Skill` (or `/name`) when the request matches.

## Engineering discipline

These behavioral rules reduce common agent coding mistakes. They favor caution
over speed. For a trivial task, use your judgment.

- **Think before you code.** State each assumption explicitly. If the request
  has more than one interpretation, present all of them — never pick one in
  silence. Research first, and never ask what the repo can answer. If something
  stays unclear after that research, stop and name it. Then ask the user
  (interactive), or record the assumption and fire `notify.sh approval_required`
  (headless). Never guess in silence.
- **Simplicity first.** Write the minimum code that solves the problem. Add no
  feature beyond the request, no abstraction for single-use code, no speculative
  configurability, and no error handler for an impossible scenario.
  Infrastructure earns its place only when the code uses it. Ask: "would a
  senior engineer call this overcomplicated?" If the answer is yes, simplify.
- **Surgical changes.** Every changed line traces to the request. Do not
  "improve" adjacent code, comments, or formatting. Do not refactor code that is
  not broken. Remove the imports and variables that your own change orphaned.
  Report pre-existing dead code, and do not delete it unasked.
- **Verifiable goals.** Turn the task into a checkable criterion before you
  write code. For "fix the bug" → write a test that reproduces it, then passes.
  For "add validation" → write tests for the invalid inputs, then make them
  pass. The experiments already enforce this rule (`mechanism` /
  `expected_delta` / `falsification` in plan.md). For a code change, see "Finish
  a task" below.

## Finish a task (verify before you report done)

Always run an appropriate verification before you declare a task complete. Do
not rely on a diff that looks correct. Pick the lightest command that exercises
the change:

- **A code change in `src/` or `scripts/python/`** → `uv run pytest` (or a
  focused `uv run pytest tests/test_foo.py::test_bar` when the suite is slow).
- **Style / formatting / import cleanup** → `uv run ruff check --fix .` and
  `uv run ruff format .`.
- **Anything just before a commit** → `make pre_commit_run`.
- **New runnable script** → execute it with a minimal config
  (`uv run python scripts/python/<file>.py`) to confirm that it starts, even
  when the real workload is heavy.

If you cannot run a verification (missing dataset, GPU-only code, external
service), say so explicitly. Never imply success. Never suppress an error to
make a command pass. Fix the root cause instead.

## Pre-commit hooks (what runs on commit)

`pre-commit-hooks` basics, `ruff` (fix + format), `codespell`, `prettier`
(md/yaml/toml/json/sh — README excluded), `nbqa-ruff`, `nbstripout`, `uv-lock`.
Do not bypass the hooks with `--no-verify` unless the user explicitly asks.

## Working tips for Claude

- Edit an existing file by default. The layout above is intentional.
- When you write a new module, mirror the patterns in
  `src/helper/logging/__init__.py` (Google-style docstrings, type hints,
  singletons where appropriate).
- Place a runnable entrypoint under `scripts/python/`, which is exempt from
  `E402`. For a Hydra entrypoint, decorate `main` with `@hydra.main(...)`. Call
  `main()` directly under `if __name__ == "__main__":`. Do not wrap it in
  `fire.Fire(...)`, because both parse `sys.argv` and conflict. Use
  `fire.Fire(...)` only for a multi-command CLI that has no `@hydra.main`
  decorator.
- Do not add a `logging` config. Reuse `LoggerConfig` from `src/helper/logging`.
- Do not introduce an alternate config system (argparse, click, dynaconf). Use
  Hydra + Fire, which this repo already chose.
- Do not create a new top-level directory without a clear reason. Extend
  `src/<package>/...` instead.
- When you are unsure about repo-specific intent (TODOs in README, empty
  `scripts/`), ask before you scaffold a large structure.
- Save a substantive analysis, plan, or research summary to `docs/` with the
  `NNN-kebab-case.md` naming (see "Docs conventions"). Do not reply in the chat
  only. Use `trash/` for a throwaway script or patch.
