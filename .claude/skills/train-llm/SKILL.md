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

**This SKILL.md is a router, not a manual.** It sequences the gates. It points
into `references/` for the details. The contract lives in
`docs/001-architecture.md` — when anything here is ambiguous, that doc wins.

The training discipline has three rules: validate before you spend, run the
smoke test before you train, and verify before you report.

## Posture

- Research-before-clarify: never ask the user about anything you can look up.
  This covers dataset schemas, model configs, prior experiments, and installed
  versions.
- Headless: do not hang. Write best-guess defaults into task.md. Fire
  `scripts/bash/notify.sh approval_required "<assumptions>"`. Then proceed.
  Interactive: ask one AskUserQuestion with ≤ 4 bundled questions.
- Doom-loop guard: after 3 identical tool calls with no new information, write
  `experiments/NNN-<slug>/blocker.md`. Fire
  `scripts/bash/notify.sh blocker "<summary>"`. Then stop.
- Context discipline: read train logs with `tail`/`grep`/`head` only. Never read
  a whole log file. Subagents return concise reports, never log dumps.

## Workflow

### 1. Preconditions

The experiment triple must exist: `scripts/python/NNN-<slug>.py`,
`configs/NNN-<slug>.yaml`, and `experiments/NNN-<slug>/` with task.md, plan.md,
budget.md, ledger.md. If a part is missing, run the **new-experiment** skill
first. No training work happens outside an experiment number. The `999-` scratch
prefix is the only gate-exempt exception.

Then run the budget gate:

```bash
uv run python scripts/python/intern.py budget --experiment NNN can-launch
```

`can-launch` resolves the model's true parameter count from the config's
`model:` group (HF safetensors metadata). It also enforces the
`scale_ceiling_params` cap automatically. Pass `--params <count>` only to
override the cap (for example an unpublished model). The caps in budget.md come
from its task-keyed profile (`smoke`/`lora`/`sft`/`dpo`/`grpo`/`pretrain`). A
denial therefore means one of two things: this task needs a bigger-budget
profile, or the path is out of budget.

Exit code 0 allows the launch. A nonzero exit code means **stop**. Do not
launch. Do not try one quick run. Report which cap the path hit. A cap is never
a reason to stop work that the budget already covers.

### 2. Validate the dataset format — before any GPU spend

Read `references/dataset-formats.md`. Verify that the dataset columns match the
training method: SFT, DPO, and GRPO each expect different columns. A format
mismatch is the most frequent training failure. The validation costs seconds on
CPU, but a failed run costs GPU-hours from the budget. If the dataset needs a
column mapping, write the mapping into the experiment script. Do not substitute
another dataset silently.

If the dataset needs **preparation** (a prep script, an in-script mapping, a
filter/mix), document that preparation in `experiments/NNN-<slug>/data.md`.
Record source → target, a mermaid pipeline, and the in/kept/split row counts
(see `docs/001-architecture.md` "data.md format"). A dataset that you use
unchanged from the Hub needs no data.md.

### 3. Pick the trainer lane (Hydra `trainer` group)

- `trainer=trl_sft` — the default for instruction tuning and LM fine-tuning on
  HF models through TRL `SFTTrainer`.
- `trainer=trl_dpo` — preference alignment on prompt/chosen/rejected data.
- `trainer=trl_kto` — preference alignment from UNPAIRED binary feedback
  (prompt/completion/label rows; thumbs-up/down data works directly, with no
  pairs). This lane uses a reference model like DPO (optional `model.ref`).
- `trainer=trl_grpo` — online RL (GRPO) on prompt-only data. The trainer samples
  the completions in-loop and grades them with `trainer.reward_funcs`. Read
  `references/grpo-rewards.md` before you write a reward function.
- `trainer=trl_gkd` — ON-policy distillation (GKD) on `messages` data. The
  student samples, and a live `model.teacher` (same tokenizer) supervises at the
  token level through generalized JSD. OFF-policy distillation is plain
  `trl_sft` on teacher-generated data. See the distill-traces skill for the
  choice.
- `trainer=lightning` — custom architectures or loops that do not fit an HF
  Trainer. The lane instantiates the module and datamodule from the config.
- `trainer=axolotl` — YAML-recipe training. The lane renders the recipe locally,
  and a remote machine runs it (`render(cfg)` only; axolotl is never a local
  dependency).

The trainer group carries only the run mechanics. The model identity and the
model loading — including QLoRA, through the `_4bit` model variant
(`model=gemma_4_e2b_it_4bit`) — belong to the Hydra `model` group. The dataset
identity belongs to the `data` group. Never put either one in the trainer keys.

- LoRA/QLoRA path (`trainer.peft` for the adapter; QLoRA composes the `_4bit`
  model variant) → read `references/lora.md` (the no-regret recipe) before you
  set any adapter option.
- Pretraining or continued pretraining → read `references/pretraining.md`.
- To choose `trainer.args` for a new path → read
  `references/hyperparameter-priors.md` for defensible starting values.
- Any `model=gemma_*` path (or the choice of which Gemma to use) → read
  `references/gemma.md` (family matrix, chat-template/masking trap, multimodal
  collators, export targets).

Use one Hydra override per path, as plan.md specifies. The tracking backend
always comes from `cfg.tracking.backend` (`tracking=trackio|wandb|none`). Never
hardcode the backend.

### 4. Smoke run — mandatory, no exceptions

```bash
uv run python scripts/python/NNN-<slug>.py smoke_test=true
```

This command forces `max_steps=1`, slices the dataset to ≤ 32 rows, and saves no
checkpoint. The smoke run must print `VERDICT: TRAIN_OK` before you start any
long run. On `VERDICT: TRAIN_FAIL | <cause>`, fix the cause and run the smoke
test again. A long run after a failed or skipped smoke test is a workflow
violation. For a remote lane, run the smoke test locally first (CPU or MPS is
enough). Then run the smoke test again on the remote hardware if the precision
or attention settings differ.

### 5. Pre-flight checklist

Read `references/preflight.md`. Fill in every line. Print the completed
checklist in your response before you launch. If you cannot fill in a line, stop
and complete the missing step first. Read `references/hardware.md` when you size
the GPU, the precision (bf16/fp16), and the attention implementation.

### 6. Launch

Read `references/compute-lanes.md` for the lane that matches `compute.kind`. The
local and ssh lanes are v1-complete. The hf_jobs, modal, and vast lanes are
config stubs; the reference states what a launch needs on each lane. Then run:

```bash
uv run python scripts/python/intern.py budget --experiment NNN record-launch
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status running --approach "<one line>"
scripts/bash/notify.sh train_started "<paths> path(s) on <dataset> via <lane>" NNN-<slug>
```

The notify.sh events are a fixed list in its usage header. An unknown event name
sends a degraded generic card. Write a multi-item body (for example a cycle
summary) as one `- ` bullet line per item, not as a `·`-separated one-liner.

For a multi-path plan, launch ONE path first. Confirm that it trains: loss lines
appear, and the run does not crash in the first minutes. Then launch the rest. A
batch submission multiplies one shared bug across the whole budget.

**Record what you run into `experiments/NNN-<slug>/run.md`** as you run it.
Record the exact, copy-pasteable commands that you run on the compute instance,
section by section (`## Prerequisite` / `## Provision` / `## Setup` / `## Train`
/ `## Benchmark` / `## Teardown`). See `docs/001-architecture.md` "run.md
format". Fill run.md live, not from memory afterward: a command that is absent
from the transcript is unreproducible.

Keep run.md **self-contained** — a reader reproduces this experiment from run.md
alone. Embed every step, including the one-time dataset-prep command. Never
cross-reference another experiment ("see 004 for the launch"). Use placeholders
for anything account-specific (`<HOST>`, `<PROJECT>`, `<YOUR_IP>`). Never paste
a token: a token belongs in `.env`. run.md is **ungated** — record the commands
of a crashed run too, so that anyone can reproduce the failure.

### 7. Monitor

Hand off to the **track-experiments** skill for the dashboards, the alerts, and
the metrics.jsonl reads. Run quick checks from here: `tail -n 20` on
`experiments/NNN-<slug>/logs/train.log`, `grep -c "step"` for progress, and the
alert events in `metrics.jsonl`. On a NaN streak, a divergence alert, or a
plateau alert, follow the suggested action in the alert message. Treat that
action as a new named hypothesis. On a CUDA OOM, read
`references/oom-recovery.md` and follow the ladder. Never change the method, the
dataset, or the sequence length without the user's approval.

### 8. Verify, record, report

When a path finishes, run the **verify-run** skill. You can also run the
commands directly:

```bash
uv run python scripts/python/intern.py verify --experiment NNN
uv run python scripts/python/intern.py check  --experiment NNN   # scaffold gate: run.md etc. present
uv run python scripts/python/intern.py budget --experiment NNN record-gpu-h --hours <h>
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status passed --verify pass
```

`check` must exit 0 before you report the work as done. It catches a missing
required file such as run.md. A run that you cannot reproduce is not a finished
run. data.md stays optional. Never write results.md or report success unless
`intern.py verify` exits 0. A failed gate means the run failed, whatever the
loss value is.

On a verify failure, write `postmortems/path-<id>.md` (symptom → root-cause
hypothesis → fix). Retry only if
`intern.py budget --experiment NNN can-retry --path-id <id>` exits 0. When every
path is exhausted, fire `scripts/bash/notify.sh error "<cause>"`. Do not fire
train_done.

Close out `run.md`. Append any separate `## Benchmark` commands that you ran. On
a remote lane, append the `## Teardown` sequence. The instance counts as torn
down only after run.md holds the sequence and you confirm it. Leave no orphaned
disk or VM that still bills.

## Done conditions

- [ ] The budget gate passed before every launch. The ledger has one row per
      path, and no row stays `running`.
- [ ] You verified the dataset columns against the method table before the
      launch.
- [ ] Every path that you launched printed `VERDICT: TRAIN_OK` on its smoke run
      first.
- [ ] You printed the filled-in pre-flight checklist before the launch.
- [ ] `experiments/NNN-<slug>/verify.md` exists with `OVERALL: PASS` for at
      least one path. You read the generation samples yourself, and did not rely
      on the mechanical checks alone.
- [ ] `experiments/NNN-<slug>/run.md` holds the exact on-compute commands
      (setup/train, plus benchmark and teardown when they apply), and
      `intern.py check --experiment NNN` exits 0 (scaffold gate).
- [ ] You wrote results.md only after verify exited 0, and it names the winning
      path with the ledger comparison.
- [ ] `notify.sh train_done` fired only after a verify that passed. It never
      fires for a run with no passing path.
