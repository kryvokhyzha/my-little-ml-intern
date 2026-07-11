---
name: train-llm
description:
  Plan, launch, and monitor LLM training runs — SFT, DPO, LoRA/QLoRA,
  pretraining — through this repo's trainer lanes (trl_sft, trl_dpo, lightning,
  axolotl) and compute lanes (local, ssh, hf_jobs, modal, vast), under the
  budget and verify gates. Use whenever the user says "train", "fine-tune",
  "launch the run", "start training", "run experiment NNN", "kick off the
  SFT/DPO run", or when an experiment scaffolded by new-experiment is ready to
  execute. Also use when a running or failed training run needs diagnosis, an
  OOM fix, or a retry.
---

# train-llm

**This SKILL.md is a router, not a manual.** It sequences the gates and points
into `references/` for the details. The contract lives in
`docs/001-architecture.md` — when anything here is ambiguous, that doc wins.

Training discipline in one line: validate before you spend, smoke before you
train, verify before you report.

## Posture

- Research-before-clarify: never ask the user about anything you could look up
  (dataset schemas, model configs, prior experiments, installed versions).
- Headless: never hang — write best-guess defaults into task.md, fire
  `scripts/bash/notify.sh approval_required "<assumptions>"`, proceed.
  Interactive: one AskUserQuestion, ≤ 4 bundled questions.
- Doom-loop guard: 3 identical tool calls with no new information → write
  `experiments/NNN-<slug>/blocker.md`, fire
  `scripts/bash/notify.sh blocker "<summary>"`, stop.
- Context discipline: read train logs with `tail`/`grep`/`head` only; never cat
  whole logs. Subagents return concise reports, never log dumps.

## Workflow

### 1. Preconditions

The experiment triple must exist: `scripts/python/NNN-<slug>.py`,
`configs/NNN-<slug>.yaml`, `experiments/NNN-<slug>/` with task.md, plan.md,
budget.md, ledger.md. If it doesn't, run the **new-experiment** skill first — no
training work happens outside an experiment number (`999-` scratch is the only
gate-exempt exception).

Then the budget gate:

```bash
uv run python scripts/python/intern.py budget --experiment NNN can-launch
```

`can-launch` resolves the model's true parameter count from the config's
`model:` group (HF safetensors metadata) and enforces the `scale_ceiling_params`
cap automatically — pass `--params <count>` only to override it (e.g. an
unpublished model). budget.md's caps come from its task-keyed profile
(`smoke`/`lora`/`sft`/`dpo`/`grpo`/`pretrain`), so a denial means either this
task needs a bigger-budget profile or the path is genuinely out of budget.

Exit 0 = allowed. Nonzero = **stop** — do not launch, do not "just try one quick
run". Report which cap is hit; caps never license quitting work already
budgeted.

### 2. Validate the dataset format — before any GPU spend

Read `references/dataset-formats.md` and verify the dataset's actual columns
match the training method (SFT / DPO / GRPO each expect different columns).
Format mismatch is the #1 training failure; validation costs seconds on CPU, a
failed run costs GPU-hours from the budget. If mapping is needed, write it into
the experiment script — never silently substitute a dataset.

If the dataset needed **preparation** (a prep script, an in-script mapping, a
filter/mix), document it in `experiments/NNN-<slug>/data.md` — source → target,
a mermaid pipeline, and in/kept/split row counts (see `docs/001-architecture.md`
"data.md format"). Datasets consumed unchanged off the Hub need no data.md.

### 3. Pick the trainer lane (Hydra `trainer` group)

- `trainer=trl_sft` — default for instruction tuning / LM fine-tuning on HF
  models via TRL `SFTTrainer`.
- `trainer=trl_dpo` — preference alignment on prompt/chosen/rejected data.
- `trainer=trl_grpo` — online RL (GRPO) on prompt-only data: completions are
  sampled in-loop and graded by `trainer.reward_funcs`; read
  `references/grpo-rewards.md` before writing any reward function.
- `trainer=lightning` — custom architectures or loops that don't fit an HF
  Trainer; module/datamodule instantiated from config.
- `trainer=axolotl` — YAML-recipe training rendered locally, executed on a
  remote box (`render(cfg)` only; axolotl is never a local dependency).

The trainer group carries only run mechanics. Model identity/loading — including
QLoRA, via the `_4bit` model variant (`model=gemma_4_e2b_it_4bit`) — is the
Hydra `model` group, and dataset identity the `data` group, never trainer keys.

- LoRA/QLoRA path (`trainer.peft` for the adapter; QLoRA composes the `_4bit`
  model variant) → read `references/lora.md` (the no-regret recipe) before
  setting any adapter knob.
- Pretraining or continued pretraining → read `references/pretraining.md`.
- Choosing `trainer.args` for a new path → read
  `references/hyperparameter-priors.md` for defensible starting values.

One Hydra override per solution path, per plan.md. The tracking backend is
always `cfg.tracking.backend` (`tracking=trackio|wandb|none`) — never hardcoded.

### 4. Smoke run — mandatory, no exceptions

```bash
uv run python scripts/python/NNN-<slug>.py smoke_test=true
```

This forces `max_steps=1`, slices the dataset to ≤ 32 rows, saves no checkpoint.
It must print `VERDICT: TRAIN_OK` before any long run is allowed.
`VERDICT: TRAIN_FAIL | <cause>` → fix the cause and re-smoke; launching a long
run on a failed or skipped smoke is a workflow violation. For remote lanes,
smoke locally first (CPU/MPS is fine for this), then re-smoke on the remote
hardware if precision/attention settings differ.

### 5. Pre-flight checklist

Read `references/preflight.md`, fill every line in, and print the completed
checklist in your response before launching. If any line cannot be filled, stop
and complete the missing step first. Consult `references/hardware.md` when
sizing GPU, precision (bf16/fp16), and attention implementation.

### 6. Launch

Read `references/compute-lanes.md` for the lane matching `compute.kind` (local
and ssh are v1-complete; hf_jobs, modal, vast are config stubs — the reference
says what a launch takes on each). Then:

```bash
uv run python scripts/python/intern.py budget --experiment NNN record-launch
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status running --approach "<one line>"
scripts/bash/notify.sh train_started "<paths> path(s) on <dataset> via <lane>" NNN-<slug>
```

For multi-path plans: launch ONE path first, confirm it trains (loss lines
appearing, no crash in the first minutes), then launch the rest — batch
submission multiplies a shared bug across the whole budget.

**Record what you run into `experiments/NNN-<slug>/run.md`** as you run it — the
exact, copy-pasteable commands executed on the compute instance, section by
section (`## Prerequisite` / `## Provision` / `## Setup` / `## Train` /
`## Benchmark` / `## Teardown`; see `docs/001-architecture.md` "run.md format").
Fill it live, not from memory afterward: a command absent from the transcript is
unreproducible. **Self-contained** — a reader reproduces this experiment from
run.md alone: embed every step (including the one-time dataset-prep command),
never cross-reference another experiment ("see 004 for the launch").
Placeholders for anything account-specific (`<HOST>`, `<PROJECT>`, `<YOUR_IP>`);
never paste a token — it belongs in `.env`. run.md is **ungated** — a crashed
run still gets its commands recorded so the failure can be reproduced.

### 7. Monitor

Hand off to the **track-experiments** skill for dashboards, alerts, and
metrics.jsonl reads. Quick checks from here: `tail -n 20` on
`experiments/NNN-<slug>/logs/train.log`, `grep -c "step"` for progress, alert
events in `metrics.jsonl`. On NaN streaks, divergence, or plateau alerts, follow
the alert message's suggested action as a new named hypothesis. On CUDA OOM:
read `references/oom-recovery.md` and follow the ladder — never change the
method, dataset, or sequence length without user approval.

### 8. Verify, record, report

When a path finishes, run the **verify-run** skill (or directly):

```bash
uv run python scripts/python/intern.py verify --experiment NNN
uv run python scripts/python/intern.py check  --experiment NNN   # scaffold gate: run.md etc. present
uv run python scripts/python/intern.py budget --experiment NNN record-gpu-h --hours <h>
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status passed --verify pass
```

`check` must exit 0 before you report done — it catches a forgotten required
file like run.md (a run you can't reproduce is not a finished run; data.md stays
optional). Never write results.md or report success unless `intern.py verify`
exited 0. A failed gate means the run failed, regardless of loss.

On verify failure: write `postmortems/path-<id>.md` (symptom → root-cause
hypothesis → fix), then retry only if
`intern.py budget --experiment NNN can-retry --path-id <id>` exits 0. When every
path is exhausted, fire `scripts/bash/notify.sh error "<cause>"` — not
train_done.

Close out `run.md`: append any separate `## Benchmark` commands you ran and, on
remote lanes, the `## Teardown` sequence — the instance is not torn down until
it is in run.md and confirmed (no orphaned disks/VMs left billing).

## Done conditions

- [ ] Budget gate passed before every launch; ledger has one row per path, none
      left `running`.
- [ ] Dataset columns were verified against the method table before launch.
- [ ] Every launched path printed `VERDICT: TRAIN_OK` on its smoke run first.
- [ ] Pre-flight checklist was printed filled-in before launch.
- [ ] `experiments/NNN-<slug>/verify.md` exists with `OVERALL: PASS` for at
      least one path, and generation samples were eyeballed, not just
      mechanically checked.
- [ ] `experiments/NNN-<slug>/run.md` holds the exact on-compute commands
      (setup/train, plus benchmark and teardown when they applied), and
      `intern.py check --experiment NNN` exits 0 (scaffold gate).
- [ ] results.md written only after verify exit 0, naming the winning path with
      the ledger comparison.
- [ ] `notify.sh train_done` fired only after a passing verify — never for a run
      with no passing path.
