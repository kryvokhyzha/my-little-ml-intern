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

One experiment number = three artifacts, created together:

```
scripts/python/NNN-<slug>.py      # hydra entrypoint (new-script scaffold + src import)
configs/NNN-<slug>.yaml           # composes main + model/data/trainer/tracking/compute/budget
experiments/NNN-<slug>/           # task.md, plan.md, budget.md, ledger.md
```

The skeletons below are the contract. Read `docs/001-architecture.md`
("Experiment convention", "Config groups", "budget.md format", "ledger.md
format") when anything is ambiguous — that doc wins.

## Posture

- Research-before-clarify: never ask the user about anything you could look up
  (existing configs, prior experiments, dataset names already in the repo).
- Interactive: unresolved unknowns → one AskUserQuestion, ≤ 4 bundled questions.
  Headless: never hang — write best-guess defaults, record them under "Unknowns"
  in task.md, fire
  `scripts/bash/notify.sh approval_required "<what you assumed>"`, proceed.
- Doom-loop guard: 3 identical tool calls with no new information → write
  `experiments/NNN-<slug>/blocker.md`, fire
  `scripts/bash/notify.sh blocker "<summary>"`, stop.

## Steps

### 1. Resolve the slug

Use `$ARGUMENTS` or the task description; slugify to kebab-case (lowercase,
non-alnum → `-`, collapse repeats, trim).

### 2. Pick the number

Next free 3-digit prefix across **all three locations**:

```bash
ls scripts/python configs experiments 2>/dev/null \
  | grep -E '^[0-9]{3}-' | grep -v '^999-' | sort | tail -1
```

Increment the highest by 1, zero-pad to 3 digits; start at `001` if nothing
matches. If the user named an explicit NNN ("set up experiment 007"), use it
only when free in all three locations; otherwise take the next free prefix and
say so.

Scratch: when the user signals draft/scratch/WIP/temporary, use the `999-`
prefix instead — gitignored, exempt from blocking gates, multiple `999-` files
may coexist (don't increment). `.gitignore` already covers
`scripts/python/999-*.py`, `configs/999-*.yaml`, and `experiments/999-*/`.
Promotion = rename `999-<slug>` to the next free `NNN` in all three locations at
once — from then on all gates apply.

### 3. Create the config

`configs/NNN-<slug>.yaml` — compose the groups exactly like this:

```yaml
# @package _global_

defaults:
  - main
  - model: <pick> # smollm2_135m | gemma_4_e2b_it | gemma_4_e2b_it_4bit
  - data: <pick> # tiny_synthetic | pi_mono_sft
  - trainer: trl_sft
  - tracking: trackio
  - compute: local
  - budget: default
  - _self_

experiment_name: NNN-<slug>
```

Swap a group pick only when the task calls for it: `model` ∈
`smollm2_135m|gemma_4_e2b_it|gemma_4_e2b_it_4bit`, `data` ∈
`tiny_synthetic|pi_mono_sft`, `trainer` ∈
`trl_sft|trl_sft_lora|trl_dpo|trl_grpo|lightning|axolotl`, `tracking` ∈
`trackio|wandb|none`, `compute` ∈ `local|ssh|modal|vast|hf_jobs` (see
`configs/*/`). Model identity lives in the `model` group and dataset identity in
the `data` group — never invent values by typing raw repo ids or dataset slugs
into trainer keys; override the `model:`/`data:` pick instead. A new model or
dataset is a one-file addition (`configs/model/<name>.yaml` /
`configs/data/<name>.yaml`) following the `_target_` pattern in
`docs/001-architecture.md` ("Config groups"). Set other concrete overrides under
`_self_` only when actually known. Tracking backend is never hardcoded in code;
adapters read `cfg.tracking.backend`.

### 4. Create the script

`scripts/python/NNN-<slug>.py` — the new-script scaffold **plus** the `sys.path`
src insert right after `rootutils.setup_root(...)`, dispatching on
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
    elif kind == "trl_dpo":
        from training.trl import run_dpo

        run_dpo(cfg)
    elif kind == "trl_grpo":
        from training.trl import run_grpo

        run_grpo(cfg)
    elif kind == "lightning":
        from training.lightning_adapter import run

        run(cfg)
    elif kind == "axolotl":
        from training.axolotl_adapter import render

        render(cfg)
    else:
        raise ValueError(f"unknown trainer.kind: {kind}")


if __name__ == "__main__":
    main()
```

Rules (from the new-script skill — read it when unsure):

- Keep the adapter imports lazy inside the branches so `--cfg job` and config
  composition never require training dependencies.
- Library imports stay bare (`from training...`, `from intern...`), never
  `from src....`.
- Never wrap a `@hydra.main` function in `fire.Fire(...)` — Hydra owns
  `sys.argv`.

### 5. Create the experiment directory

`experiments/NNN-<slug>/` with exactly these four files. Never create
`results.md` or `verify.md` — those are gate outputs written later.

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

`plan.md` — every hypothesis carries the full contract; no production file may
be edited before the change is a named hypothesis:

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

`budget.md` — read the current values from `configs/budget/default.yaml` (or
whichever budget group the config composes) and write them in exactly this
format (values shown match the group defaults at time of writing — always
re-read the YAML):

```text
# Budget

max_paths: 2
max_retries_per_path: 2
compute_cap_gpu_h: 2.0
scale_ceiling_params: 200000000
token_budget: null

## Spent

paths_launched: 0
retries_used: 0
gpu_h_used: 0.0
```

`ledger.md` — empty table, header columns exactly:

```text
| path_id | approach | status | final_train_loss | final_eval_loss | verify | failure_cause | retry_of | gpu_min | run_url |
| ------- | -------- | ------ | ---------------- | --------------- | ------ | ------------- | -------- | ------- | ------- |
```

### 6. Verify the scaffold

```bash
uv run python scripts/python/NNN-<slug>.py --cfg job
uv run python scripts/python/intern.py budget --experiment NNN status
```

Both must exit 0 (`--cfg job` prints the composed config without running
training). Exit 1 = gate denied, 2 = usage/missing artifacts — fix the scaffold
before reporting; never report success on a nonzero exit.

### 7. Report back

Link all three artifacts and show the run commands:

```
uv run python scripts/python/NNN-<slug>.py smoke_test=true   # mandatory smoke gate before any long run
uv run python scripts/python/NNN-<slug>.py
```

Planning, launching, and monitoring paths belong to the train-llm skill — hand
off there.

## Gates

Never write results.md or report success unless `intern.py verify` exited 0. A
failed gate means the run failed, regardless of loss.

- Before launching any path:
  `uv run python scripts/python/intern.py budget --experiment NNN can-launch`
  (exit 0 = allowed, 1 = denied — stop, do not launch).
- This skill only scaffolds; it never fires `notify.sh train_done`.
- `999-` scratch experiments are exempt from gates until promoted.

## Done conditions

- [ ] `scripts/python/NNN-<slug>.py` exists — new-script scaffold + `sys.path`
      src insert + `cfg.trainer.kind` dispatch.
- [ ] `configs/NNN-<slug>.yaml` exists and composes the six groups + `_self_`
      with `experiment_name` set;
      `uv run python scripts/python/NNN-<slug>.py     --cfg job` exits 0.
- [ ] `experiments/NNN-<slug>/` contains task.md, plan.md, budget.md, ledger.md
      — and no results.md or verify.md.
- [ ] Every hypothesis in plan.md has mechanism / expected_delta /
      falsification; each path is one Hydra override.
- [ ] budget.md parses:
      `uv run python scripts/python/intern.py budget --experiment NNN status`
      exits 0.
- [ ] The same NNN prefix is used in all three locations and collides with
      nothing.
