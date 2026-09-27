---
name: new-experiment
description:
  Scaffold the numbered experiment triple for a training run — entrypoint
  scripts/python/NNN-slug.py, config configs/NNN-slug.yaml, and
  experiments/NNN-slug/ with task.md, plan.md, budget.md, ledger.md skeletons.
  Use whenever the user says "new experiment", "start an experiment", "scaffold
  a training run", "set up experiment NNN", or asks to train or fine-tune
  anything that does not yet have an experiment number. All training work in
  this repo starts here — if no NNN triple exists for the task, run this skill
  before touching any training code.
---

# new-experiment

One experiment number = three artifacts. Create all three together:

```
scripts/python/NNN-<slug>.py      # hydra entrypoint (new-script scaffold + src import)
configs/NNN-<slug>.yaml           # composes main + model/data/trainer/tracking/compute/budget
experiments/NNN-<slug>/           # task.md, plan.md, budget.md, ledger.md, run.md
```

The skeletons below are the contract. Read `docs/001-architecture.md` when
anything is ambiguous — the sections "Experiment convention", "Config groups",
"budget.md format", and "ledger.md format". That doc wins.

## Posture

- Research before you ask. Never ask the user about anything you can look up.
  Look up the existing configs, the prior experiments, and the dataset names
  already in the repo.
- Interactive mode: ask one AskUserQuestion that bundles ≤ 4 questions for the
  unresolved unknowns. Headless mode: never wait for a reply. Write best-guess
  defaults. Record them under "Unknowns" in task.md. Fire
  `scripts/bash/notify.sh approval_required "<what you assumed>"`. Then proceed.
- Doom-loop guard: after 3 identical tool calls that return no new information,
  write `experiments/NNN-<slug>/blocker.md`. Fire
  `scripts/bash/notify.sh blocker "<summary>"`. Then stop.

## Steps

### 1. Resolve the slug

Take the slug from `$ARGUMENTS` or from the task description. Slugify it to
kebab-case: lowercase it, replace each non-alnum character with `-`, collapse
repeats, and trim.

### 2. Pick the number

Take the next free 3-digit prefix across **all three locations**:

```bash
ls scripts/python configs experiments 2>/dev/null \
  | grep -E '^[0-9]{3}-' | grep -v '^999-' | sort | tail -1
```

Increment the highest number by 1. Zero-pad the result to 3 digits. Start at
`001` when nothing matches. The user can name an explicit NNN, for example "set
up experiment 007". Use that number only when it is free in all three locations.
Otherwise take the next free prefix and tell the user.

Scratch: when the user asks for a draft, scratch, WIP, or temporary run, use the
`999-` prefix instead. Git ignores these files, and the gates do not block them.
Several `999-` files can coexist, so do not increment the number. `.gitignore`
already covers `scripts/python/999-*.py`, `configs/999-*.yaml`, and
`experiments/999-*/`. To promote a scratch experiment, rename `999-<slug>` to
the next free `NNN` in all three locations at once. All gates apply after the
rename.

### 3. Create the config

Create `configs/NNN-<slug>.yaml`. Compose the groups exactly like this:

```yaml
# @package _global_

defaults:
  - main
  - model: <pick> # ls configs/model/
  - data: <pick> # ls configs/data/
  - trainer: trl_sft
  - tracking: trackio
  - compute: local
  - budget: <pick> # smoke | default | lora | sft | dpo | grpo | pretrain | autoresearch
  - _self_

experiment_name: NNN-<slug>
```

Swap a group pick only when the task needs it. The picks are the file names in
`configs/<group>/` — run `ls configs/model configs/data configs/trainer`. Read
the header comment of a trainer file before you pick it: it names the columns
and the `model.teacher` / reward / server needs. The train-llm skill (step 3)
maps each trainer lane to its use.

**Pick the budget by task** — the caps must fit the work. A smoke run needs
minutes, and a GRPO run needs hours. Step 5 holds the catalog.

The `model` group holds the model identity, and the `data` group holds the
dataset identity. Never type a raw repo id or a dataset slug into a trainer key.
Override the `model:` or `data:` pick instead. A new model or a new dataset is a
one-file addition: `configs/model/<name>.yaml` / `configs/data/<name>.yaml`.
Follow the `_target_` pattern in `docs/001-architecture.md` ("Config groups").
Set other concrete overrides under `_self_` only when you know the values.

Never hardcode the tracking backend in code. The adapters read
`cfg.tracking.backend`.

### 4. Create the script

Create `scripts/python/NNN-<slug>.py` from the new-script scaffold. Add the
`sys.path` src insert right after `rootutils.setup_root(...)`. Dispatch on
`cfg.trainer.kind`:

```python
"""<one-line description of the experiment>."""

import sys

import hydra
import rootutils
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from omegaconf import DictConfig


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="NNN-<slug>")
def main(cfg: DictConfig) -> None:
    """Entry point."""
    logger.info("Starting NNN-<slug> with config:\n{}", cfg)
    kind = cfg.trainer.kind
    if kind == "trl_sft":
        from training.trl import run_sft

        run_sft(cfg)
    else:
        raise ValueError(f"unknown trainer.kind: {kind}")


if __name__ == "__main__":
    main()
```

Rules (from the new-script skill — read it when you are unsure):

- Dispatch ONLY the lanes that the config composes. The shipped scripts do
  exactly this. For another lane, the branch calls
  `training.trl.run_<kind without trl_>` (`run_grpo`, `run_distill`, `run_sdft`,
  `run_async_grpo`, …), `training.lightning_adapter.run`, or
  `training.axolotl_adapter.render`.
- Keep the adapter imports lazy inside the branches. `--cfg job` and the config
  composition then never need the training dependencies.
- Write the library imports bare (`from training...`, `from intern...`). Never
  write `from src....`.
- Never wrap a `@hydra.main` function in `fire.Fire(...)` — Hydra owns
  `sys.argv`.

### 5. Create the experiment directory

Create `experiments/NNN-<slug>/` with exactly these five files. Never create
`results.md` or `verify.md`. The gates write those two files later. The gates
and the training adapters also create `journal.md` on first use.

`task.md`:

```text
# Task — NNN-<slug>

## Restated task

<the task in your own words: model, data, objective, success criterion>

## Unknowns

- <open question or best-guess assumption made>

## Run mode

interactive # or: headless
```

`plan.md` — every hypothesis carries the full contract. Do not edit a production
file before the change is a named hypothesis:

```text
# Plan — NNN-<slug>

## Hypotheses

### H1: <short name>

- mechanism: <causal path from change to outcome>
- expected_delta: <numeric, e.g. final_eval_loss -0.15>
- falsification: <what result kills this hypothesis>

## Solution paths

One variable per path — prefer exactly one Hydra override per path.

| path_id | hypothesis | override                               |
| ------- | ---------- | -------------------------------------- |
| path-1  | H1         | <e.g. trainer.args.learning_rate=1e-4> |
```

`budget.md` — **do not hand-write the caps.** Seed the file from the same
profile that the config composes, so the two never drift:

```bash
uv run python scripts/python/intern.py budget --experiment NNN init --profile <name>
```

`init` writes the caps from `configs/budget/<name>.yaml` and sets every spend
counter to zero. `init` refuses to overwrite a budget.md that already records
spend. Pass `--force` to re-seed the file on purpose. The budget is task-keyed.
Choose the profile whose caps fit the work:

| profile        | paths × retries | GPU-h | param ceiling | use for                                   |
| -------------- | --------------- | ----: | ------------: | ----------------------------------------- |
| `smoke`        | 1 × 1           |  0.25 |          200M | plumbing / tiny-model checks              |
| `default`      | 2 × 2           |   2.0 |          200M | generic small run (safe fallback)         |
| `lora`         | 2 × 2           |   4.0 |           12B | LoRA / QLoRA on one GPU                   |
| `sft`          | 2 × 2           |   6.0 |            3B | full-parameter SFT (memory-bound)         |
| `dpo`          | 2 × 2           |   8.0 |           12B | preference tuning (ref model doubles fwd) |
| `grpo`         | 3 × 1           |  12.0 |           12B | online RL / GRPO (rollout-bound)          |
| `pretrain`     | 1 × 1           |  24.0 |            2B | from-scratch pretraining                  |
| `autoresearch` | 10 × 1          |   8.0 |          200M | autoresearch-loop orchestrator            |

When you need caps between two profiles, add a new `configs/budget/<name>.yaml`
with five cap keys. Copy an existing one. Do not hand-edit budget.md. The
profile stays the single source of truth.

`ledger.md` — an empty table with these header columns exactly:

```text
| path_id | approach | status | final_train_loss | final_eval_loss | verify | failure_cause | retry_of | gpu_min | run_url |
| ------- | -------- | ------ | ---------------- | --------------- | ------ | ------------- | -------- | ------- | ------- |
```

`run.md` — write the skeleton only. The **train-llm** skill fills in the exact
on-compute commands as it runs them. `docs/001-architecture.md` ("run.md
format") holds the full format. Scaffold `# Run — NNN-<slug>` and then the empty
sections `## Lane`, `## Setup`, `## Train`, and `## Benchmark`. Add
`## Provision` and `## Teardown` only for the remote lanes.

### 6. Verify the scaffold

```bash
uv run python scripts/python/NNN-<slug>.py --cfg job
uv run python scripts/python/intern.py budget --experiment NNN status
uv run python scripts/python/intern.py check --experiment NNN
```

All three commands must exit 0. `--cfg job` prints the composed config and does
not run the training. `check` refuses when any required file is missing:
task.md, plan.md, budget.md, ledger.md, or run.md.

Exit 1 means the gate denied the scaffold. Exit 2 means a usage error or a
missing artifact. Fix the scaffold before you report. Never report success on a
nonzero exit.

### 7. Report back

Link all three artifacts. Show the run commands:

```
uv run python scripts/python/NNN-<slug>.py smoke_test=true   # mandatory smoke gate before any long run
uv run python scripts/python/NNN-<slug>.py
```

The train-llm skill plans, launches, and monitors the paths. Hand off to that
skill.

## Gates

Never write results.md unless `intern.py verify` exited 0. Never report success
unless `intern.py verify` exited 0. A failed gate means the run failed,
regardless of the loss.

- Before you launch any path, run
  `uv run python scripts/python/intern.py budget --experiment NNN can-launch`.
  Exit 0 allows the launch. Exit 1 denies it — stop and do not launch.
- This skill only scaffolds. It never fires `notify.sh train_done`.
- The gates skip a `999-` scratch experiment until you promote it.

## Done conditions

- [ ] `scripts/python/NNN-<slug>.py` exists — new-script scaffold + `sys.path`
      src insert + `cfg.trainer.kind` dispatch.
- [ ] `configs/NNN-<slug>.yaml` exists and composes the six groups + `_self_`
      with `experiment_name` set.
      `uv run python scripts/python/NNN-<slug>.py     --cfg job` exits 0.
- [ ] `experiments/NNN-<slug>/` contains task.md, plan.md, budget.md, ledger.md,
      run.md (skeleton) — and no results.md or verify.md.
      `intern.py check --experiment NNN` exits 0 (the scaffold gate).
- [ ] Every hypothesis in plan.md has mechanism / expected_delta /
      falsification. Each path is one Hydra override.
- [ ] budget.md parses:
      `uv run python scripts/python/intern.py budget --experiment NNN status`
      exits 0.
- [ ] The same NNN prefix appears in all three locations and collides with
      nothing.
