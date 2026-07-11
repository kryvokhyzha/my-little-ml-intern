# AGENTS.md

Guidance for AI coding agents when working in this repository.

## Project overview

`my-little-ml-intern` — a personal "ML intern": Claude Code skills plus a
code-enforced guardrail library for ML/LLM experimentation. Python 3.13, `uv`,
Hydra configs, Loguru, Fire. Project root is anchored by `.project-root` (used
via `rootutils`).

Three parts work together:

- **`.claude/skills/`** — the skill pack (experiment scaffolding, training
  discipline, verification, tracking, literature research).
- **`src/intern/`** — enforcement library: verification, budget, ledger,
  dependency-age gates with nonzero exit codes. Skills instruct; these scripts
  refuse.
- **`src/training/`** — lane adapters mapping Hydra configs onto TRL (SFT/DPO),
  PyTorch Lightning, and axolotl (rendered YAML for remote GPU boxes; never a
  local dependency).

The full contract (module APIs, artifact formats, skill conventions) lives in
[docs/001-architecture.md](docs/001-architecture.md) — read it before changing
`src/intern`, `src/training`, or any skill.

## Tech stack and key conventions

- **Python**: `>=3.11,<3.14`, target `py313`.
- **Package manager**: `uv` (never use `pip`/`poetry`/`conda` directly here).
  Lockfile is `uv.lock`.
- **Build backend**: `hatchling`; package lives under `src/`.
- **Configuration**: Hydra (`configs/`, entrypoint `configs/main.yaml`).
- **Logging**: Loguru via `src/helper/logging` (singleton `LoggerConfig`). Env
  vars: `ENV_MODE`, `LOG_LEVEL`, `JSON_LOGS`, `COLORIZE`. Custom levels:
  `WARNONCE`, `DEPRECATED`.
- **CLI**: `fire` for command-line entry points.
- **Display/UX**: `rich` for terminal output.
- **Path/root resolution**: `rootutils` (use it instead of computing paths from
  `__file__`).
- **Training**: `torch`, `lightning`, `trl` (+`peft`), `transformers`,
  `datasets`, `accelerate`. axolotl runs remotely via rendered YAML only.
- **Tracking**: `trackio` (primary) / `wandb`, selected by the Hydra `tracking`
  group — never hardcode a backend.
- **Dependency freshness rule**: latest versions, but only releases published
  **≥ 1 week ago** (checked by `intern.deps`; run
  `uv run python scripts/python/intern.py deps`).

## Repository layout

```
configs/        # Hydra configs: main.yaml + groups (hydra/, model/, data/, trainer/, tracking/, compute/, budget/)
scripts/        # python/ (incl. intern.py CLI) and bash/ (notify.sh, gpu_probe.sh)
src/            # Importable package code
  helper/       # display/ (rich) and logging/ (LoggerConfig singleton) — template-generic
  data/         # loading.py (split loading) + synthetic.py (smoke fixtures)
  intern/       # enforcement library: verify, budget, ledger, callbacks, deps, traces
  training/     # runtime/models/sampling shared; trl/ subpackage, lightning_adapter, axolotl_adapter
experiments/    # NNN-<slug>/ run artifacts (task/plan/budget/ledger/verify/results)
docs/           # Plans, analyses, design notes (see "Docs conventions")
tests/          # pytest suite for src/
trash/          # Gitignored scratchpad for throwaway scripts/output
pyproject.toml  # Project + tool config (ruff, pytest, nbqa)
Makefile        # Common uv / pre-commit shortcuts
.env.example    # Committed template for required env vars; copy to .env
.pre-commit-config.yaml
.project-root   # Marker file for rootutils — do not delete
```

Some of these directories are created on demand — create them when they're
needed, don't scaffold empty ones.

## Common commands

Prefer `make` targets when available:

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

Always run Python via `uv run ...` to ensure the locked environment is used.

## Code style

Enforced by Ruff (`pyproject.toml`):

- Line length **120**, target `py313`.
- Rules: `E, F, W, I, D` (pycodestyle, pyflakes, isort, pydocstyle).
- isort: 2 blank lines after imports.
- Docstrings: required on public modules/classes/functions is **disabled**
  (D100–D107 ignored). Use Google-style docstrings (already in use — see
  `src/helper/logging/__init__.py`).
- Notebooks are linted via `nbqa`; `nbstripout` strips outputs on commit.

Additional conventions:

- Prefer `pathlib.Path` over `os.path`.
- Use `loguru.logger` (already configured) — do **not** instantiate
  `logging.getLogger`.
- Use `pydantic` for data models / config validation.
- Use `joblib` for parallelism and caching when appropriate.
- Type-hint new code; tests can be lighter.

## Testing

- Framework: `pytest` + `pytest-env`.
- `pythonpath = ["src", "."]` (set in `pyproject.toml`) — import as
  `from helper... import ...`.
- Place tests alongside or under a top-level `tests/` directory (create if
  absent).
- For async code, default fixture loop scope is `function`.

## Adding dependencies

- Runtime: edit `[project].dependencies` in `pyproject.toml`, then
  `uv sync --all-extras --no-install-project`.
- Dev / lint / test / notebook: use the matching `[dependency-groups]` group.
- After any dependency change, `uv.lock` must be updated (pre-commit `uv-lock`
  hook enforces this).
- Pin reasonably: prefer `~=` for libraries we track closely, `>=,<` for broad
  ranges.

## Hydra configs

- Entry config: `configs/main.yaml`. Compose groups via `defaults:` lists.
- Output dir defaults are disabled (no `outputs/` clutter) — see
  `configs/hydra/default.yaml`.
- When adding a new config group, create a subdirectory under `configs/` and
  reference it from `defaults`.

## ML experiments (the core convention)

One experiment number = three artifacts: `scripts/python/NNN-<slug>.py` +
`configs/NNN-<slug>.yaml` + `experiments/NNN-<slug>/`. Scaffold with the
`new-experiment` skill; `999-` is gitignored scratch and exempt from gates. Full
formats in [docs/001-architecture.md](docs/001-architecture.md).

Non-negotiable rules (enforced by `scripts/python/intern.py` exit codes):

- **Smoke before scale**: every training run starts with `smoke_test=true` (1
  step, tiny slice) and must print `VERDICT: TRAIN_OK`.
- **Budget gate**: check `intern.py budget --experiment NNN can-launch` before
  launching a path; record spend after.
- **Verify gate**: never write `results.md` or report success unless
  `intern.py verify` exited 0. A failed gate means the run failed, regardless of
  loss. A low loss number is never evidence the model works.
- **One variable per path**: prefer one Hydra override per experiment path;
  hypotheses in `plan.md` need mechanism, expected numeric delta, and a
  falsification condition.
- Scripts that import from `src/` add `sys.path.insert(0, str(root / "src"))`
  right after `rootutils.setup_root(...)` — library imports stay bare
  (`from intern.verify import ...`), never `from src....`. The same style holds
  **inside** `src/` packages: absolute bare imports
  (`from intern.scaffold import ...`, `from training.runtime import ...`), no
  relative imports (`from .scaffold import ...`).

## Docs conventions (`docs/`)

`docs/` is the home for plans, design notes, research summaries, and analyses
produced during work — anything that should outlive a single conversation.

- **Prefer writing to `docs/` over dumping long analyses into chat.** When the
  user asks for a plan, investigation write-up, or comparison, save it as a file
  and reference it.
- **Naming**:
  - `NNN-kebab-case-title.md` — numbered, ordered series of substantive docs
    (e.g. `001-data-pipeline.md`, `002-eval-metrics.md`). Use the next free
    3-digit prefix.
  - `099-*.md` — exploratory / experimental notes that are not part of the main
    numbered series.
  - `999-*.md` — temporary / scratch docs intended to be deleted or folded back
    into a numbered doc later. **Gitignored** (see `.gitignore`), so safe for
    in-progress notes you don't want to commit yet.
  - Unprefixed `kebab-case.md` — standalone reference notes that don't belong to
    a sequence (rationales, workflow guides, one-off analyses).
- **PR descriptions, patches, ad-hoc test scripts**: keep them in `trash/`, not
  `docs/`.
- `docs/build/` is gitignored and reserved for generated output (e.g. Sphinx);
  do not place hand-written notes there.
- Write in clear Markdown; include links to source files/lines (e.g.
  `[foo.py:42](src/foo.py)`) where helpful.

## Scratch / throwaway work

Two flavors of "don't commit this yet" coexist in the repo — pick the one that
fits:

- **`999-*` prefix** — for _in-progress work that still uses the project's
  normal infrastructure_ (Hydra configs, `scripts/python/` layout, `docs/`
  Markdown). The `.gitignore` excludes:

  - `scripts/python/999-*.py` — exploratory Hydra+Fire scripts (use the
    `new-script` skill's normal scaffold; just pick `999-` as the prefix).
  - `docs/999-*.md` — temporary notes / drafts that may later be promoted to a
    real `NNN-*.md` doc. Use this when you want the script or doc to _look like_
    a regular project artifact — paired config, proper imports, etc. — but
    aren't ready to commit it.

- **`trash/` directory** — gitignored sink for _anything that doesn't fit the
  project's structure at all_: `tmp_*.py`, `t.py`, `check_*.py`, `*.patch`,
  draft `pr-desc.md`, downloaded artifacts, one-off diagnostic snippets, etc.
  - Do **not** import from `trash/` in committed code.
  - Do not put long-lived notes here — promote them to `docs/` instead.

When promoting: rename `999-<slug>.{py,md}` → `<next-free-NNN>-<slug>.{py,md}`
(and pair the script with a matching renamed config). For `trash/` content worth
keeping, move it into the proper directory and clean it up first — don't just
`git add` from `trash/`.

## Environment variables (`.env` / `.env.example`)

- `.env.example` is the committed source of truth for the env surface
  (placeholder values; comments always on their own lines —
  `scripts/bash/notify.sh` shell-sources the file).
- `.env` is **gitignored** — never commit secrets.
- When adding a new env var that the code reads, also add it to `.env.example`
  with a sensible default or empty placeholder.

The full read surface, grouped as in `.env.example`:

- **Logging** (consumed by `LoggerConfig`; `JSON_LOGS`/`COLORIZE` also switch
  `helper.display` to plain-text output): `ENV_MODE`, `LOG_LEVEL`, `JSON_LOGS`,
  `COLORIZE`; optional: `FORCE_RICH` (force rich output when stdout is not a
  TTY).
- **Hugging Face**: `HF_TOKEN` (write scope for `intern.py publish`; hf_jobs
  lane secret), `HF_USER` (publish repo-id default), `HF_HUB_ENABLE_HF_TRANSFER`
  (hf-transfer is installed but inert without it); optional: `HF_HOME`,
  `HF_HUB_OFFLINE`, `HF_DATASETS_CACHE`, `HF_ENDPOINT` (private Hub mirror).
- **GitHub**: `GITHUB_TOKEN` (`gh search` in literature-recipe-research —
  unauthenticated gh search refuses; `GH_TOKEN` takes precedence when set; gh
  does not auto-load `.env`, so export it or `gh auth login`).
- **Training**: `SMOKE_TEST` (forces the smoke gate in the training adapters),
  `TOKENIZERS_PARALLELISM`; optional: `PYTORCH_ENABLE_MPS_FALLBACK`,
  `CUDA_VISIBLE_DEVICES`.
- **trackio**: optional `TRACKIO_PROJECT` (project name; defaults to
  `project_name` from `configs/main.yaml` via config interpolation),
  `TRACKIO_DIR` (local metrics DB location, defaults to `$HF_HOME/trackio`); HF
  Space sync is configured via `tracking.space_id` in configs (created private),
  not env.
- **wandb**: `WANDB_API_KEY` (only when `tracking=wandb`); optional:
  `WANDB_PROJECT` (project name; defaults to `project_name` from
  `configs/main.yaml` via config interpolation), `WANDB_MODE`, `WANDB_DIR`.
- **Notifications** (`scripts/bash/notify.sh`; all optional, per-channel no-op):
  `TG_BOT_TOKEN`, `TG_CHAT_ID`, `SLACK_WEBHOOK_URL`, `SLACK_BOT_TOKEN`,
  `SLACK_CHANNEL_ID`; `PROJECT_NAME` overrides the project label on the cards.

## Project skills

Project-level skills under `.claude/skills/` (auto-discoverable):

<!-- skills-table:start -->

| Skill                        | Use when                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| `autoresearch-loop`          | Generational orchestrator that runs many bounded training experiments to find the best…    |
| `distill-traces`             | Turn verified agent traces into training data and run the self-distillation loop — define… |
| `literature-recipe-research` | Isolated-context literature research that returns a ranked table of training recipes with… |
| `new-doc`                    | Create a new numbered document in docs/ following the NNN-kebab-case-title.md convention   |
| `new-experiment`             | Scaffold the numbered experiment triple for a training run — entrypoint…                   |
| `new-script`                 | Scaffold a new runnable Python entrypoint under scripts/python/ together with a paired…    |
| `publish-model`              | Publish a verified training run to the Hugging Face Hub through the blocking publish gate… |
| `track-experiments`          | Sets up experiment tracking and runs the alert-driven iteration loop for training runs in… |
| `train-llm`                  | Plan, launch, and monitor LLM training runs — SFT, DPO, LoRA/QLoRA, pretraining — through… |
| `verify-run`                 | Run the blocking verification gate on a finished training run and act on the result…       |

<!-- skills-table:end -->

Invoke via `Skill` (or `/name`) when the request matches.

## Engineering discipline

Behavioral rules that reduce common agent coding mistakes. They bias toward
caution over speed — for trivial tasks, use judgment.

- **Think before coding.** State assumptions explicitly; if multiple
  interpretations exist, present them — never pick one silently. Research first
  (never ask what the repo can answer), but when something is still unclear
  after looking: stop, name what's confusing, and ask (interactive) — or record
  the assumption and fire `notify.sh approval_required` (headless). Never guess
  silently.
- **Simplicity first.** Minimum code that solves the problem: no features beyond
  what was asked, no abstractions for single-use code, no speculative
  configurability, no error handling for impossible scenarios. Infrastructure
  earns its place by being used. Test: "would a senior engineer call this
  overcomplicated?" — if yes, simplify.
- **Surgical changes.** Every changed line traces to the request. Don't
  "improve" adjacent code, comments, or formatting; don't refactor what isn't
  broken. Remove imports/variables your own change orphaned; mention
  pre-existing dead code, don't delete it unasked.
- **Verifiable goals.** Turn the task into a checkable criterion before coding:
  "fix the bug" → a test that reproduces it, then passes; "add validation" →
  tests for the invalid inputs, then make them pass. Experiments already enforce
  this (`mechanism` / `expected_delta` / `falsification` in plan.md); for code
  changes, see "Finishing a task" below.

## Finishing a task (verify before reporting done)

Always run an appropriate verification before declaring a task complete — don't
rely on the diff "looking right." Pick the lightest command that exercises the
change:

- **Code changes touching `src/` or `scripts/python/`** → `uv run pytest` (or a
  focused `uv run pytest tests/test_foo.py::test_bar` when the suite is slow).
- **Style / formatting / import cleanup** → `uv run ruff check --fix .` and
  `uv run ruff format .`.
- **Anything just before a commit** → `make pre_commit_run`.
- **New runnable script** → execute it with a minimal config
  (`uv run python scripts/python/<file>.py`) to confirm it boots, even if the
  real workload is heavy.

If verification can't be run (missing dataset, GPU-only code, external service),
say so explicitly rather than implying success. Don't suppress errors to make a
command pass — fix the root cause.

## Pre-commit hooks (what runs on commit)

`pre-commit-hooks` basics, `ruff` (fix + format), `codespell`, `prettier`
(md/yaml/toml/json/sh — README excluded), `nbqa-ruff`, `nbstripout`, `uv-lock`.
Do not bypass with `--no-verify` unless explicitly asked.

## Working tips for Claude

- Default to editing existing files; the layout above is intentional.
- When writing new modules, mirror the patterns in
  `src/helper/logging/__init__.py` (Google-style docstrings, type hints,
  singletons where appropriate).
- Place runnable entrypoints under `scripts/python/` (which is exempt from
  `E402`). For a Hydra entrypoint, decorate `main` with `@hydra.main(...)` and
  call `main()` directly under `if __name__ == "__main__":` — don't wrap it in
  `fire.Fire(...)` (both parse `sys.argv` and conflict). Use `fire.Fire(...)`
  only for multi-command CLIs that have no `@hydra.main` decorator.
- Don't add a `logging` config — reuse `LoggerConfig` from `src/helper/logging`.
- Don't introduce alternate config systems (argparse, click, dynaconf) — use
  Hydra + Fire as already chosen.
- Don't create new top-level directories without a clear reason; extend
  `src/<package>/...`.
- When unsure about repo-specific intent (TODOs in README, empty `scripts/`),
  ask before scaffolding large structures.
- For substantive analyses, plans, or research summaries: save them to `docs/`
  using the `NNN-kebab-case.md` naming (see "Docs conventions") rather than only
  replying in chat. Use `trash/` for throwaway scripts and patches.
