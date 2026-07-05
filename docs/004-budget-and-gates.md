# 004 — Budget and gates

<!-- What budget.md is, what every gate enforces, and how to react when a gate says no.
Companion to docs/001-architecture.md (the contract) and docs/003-experiment-lifecycle.md
(the contract in motion). -->

## Philosophy: caps are accounting documents, not vibes

A budget is the spend you negotiated _before_ the run, written down where a
program can enforce it. Every gate decision is a nonzero-exit-code refusal with
a printed reason — never a judgment call made mid-run when sunk cost is
whispering "one more retry".

Caps cut both ways:

- **Ceiling:** no new path or retry launches past a cap. "Just one quick run" is
  how budgets die.
- **Floor:** caps never license quitting work that is already budgeted. Hitting
  `max_paths` doesn't excuse a missing postmortem, a ledger row left `running`,
  or skipping verify on a finished run. Under-spend is not a goal either —
  budgeted paths exist to be run.

## budget.md anatomy

Each experiment carries its own
[budget.md](../experiments/000-tiny-sft-smoke/budget.md), seeded from a
**task-keyed** budget profile (`budget init`, below) and parsed by
`intern.budget`. Caps depend on the task — a smoke needs minutes and 200M
params; a GRPO run needs hours and multi-B params — so the profile you compose
sets the ceiling:

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

- `max_paths` — how many distinct solution paths (plan.md hypotheses) may be
  launched. `can-launch` denies when `paths_launched` reaches it.
- `max_retries_per_path` — retries are counted over the whole retry **tree**,
  not per link of a chain. A chain path-1 → path-2 → path-3 has _two_ retries
  charged against path-1's tree; renaming a retry does not reset the counter. On
  top of that sits a **global backstop**: any retry is denied once
  `retries_used ≥ max_paths × max_retries_per_path`, no matter how the trees are
  shaped.
- `compute_cap_gpu_h` — hard GPU-hour ceiling, checked by both `can-launch` and
  `can-retry` once `gpu_h_used` reaches it.
- `scale_ceiling_params` — model-size ceiling. `can-launch` resolves the model's
  true parameter count automatically from the config's `model:` group (HF
  safetensors metadata) and denies when it exceeds the ceiling; pass
  `--params N` to override the resolved value. If the count can't be resolved
  (offline, no metadata), the ceiling is skipped with a warning.
- `token_budget` — advisory for the orchestrating agent (LLM-token spend), not
  machine-enforced; `null` is fine.

## The spent tally — who updates it

The `## Spent` section is updated **only** by the CLI:

```bash
uv run python scripts/python/intern.py budget --experiment NNN record-launch
uv run python scripts/python/intern.py budget --experiment NNN record-retry
uv run python scripts/python/intern.py budget --experiment NNN record-gpu-h --hours 0.5
```

Never by hand. The tally is the audit trail — its whole value is that every
number in it corresponds to a `record-*` call at a known point in the lifecycle.
The `000-tiny-sft-smoke` fixture's tally showing `retries_used: 5` against a 2×2
backstop is an honest record of a supervised pipeline shakedown outrunning its
defaults; a hand-edited tally can show anything, which is to say nothing.

## Gate commands and exit codes

Everywhere: **0 = ok/allowed, 1 = gate failed/denied, 2 = usage error or missing
artifacts.** These exits are the blocking mechanism — skills must stop on
nonzero.

```bash
uv run python scripts/python/intern.py budget --experiment NNN init --profile lora [--force]
uv run python scripts/python/intern.py budget --experiment NNN status
uv run python scripts/python/intern.py budget --experiment NNN can-launch [--params N]
uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1
uv run python scripts/python/intern.py budget --experiment NNN record-launch|record-retry
uv run python scripts/python/intern.py budget --experiment NNN record-gpu-h --hours H
uv run python scripts/python/intern.py verify --experiment NNN [--vocab-size N] [--checks a,b]
uv run python scripts/python/intern.py ledger --experiment NNN show|upsert --path-id ... [--field value ...]
uv run python scripts/python/intern.py status --experiment NNN [--json]
uv run python scripts/python/intern.py publish --experiment NNN [--repo-id org/name] [--private true|false]
uv run python scripts/python/intern.py deps [--min-age-days 7]
```

| command              | exit 0            | exit 1                   | exit 2                                              |
| -------------------- | ----------------- | ------------------------ | --------------------------------------------------- |
| `budget init`        | budget.md seeded  | refused: existing spend  | unknown/missing `--profile`, unparsable profile     |
| `budget status`      | tally printed     | —                        | experiment/budget.md missing or unparsable          |
| `budget can-launch`  | launch allowed    | denied (reason printed)  | missing artifacts                                   |
| `budget can-retry`   | retry allowed     | denied (reason printed)  | missing `--path-id` or ledger.md                    |
| `budget record-*`    | spend recorded    | —                        | bad/missing arguments (e.g. negative `--hours`)     |
| `verify`             | all checks passed | ≥ 1 check FAILED         | metrics.jsonl absent, or current run logged nothing |
| `ledger show/upsert` | ok                | —                        | unknown column, invalid enum, missing file          |
| `status`             | dashboard printed | —                        | experiment missing                                  |
| `publish`            | published         | gate refused             | missing artifacts or credentials                    |
| `deps`               | no violations     | dependency-age violation | —                                                   |

## The verify gate

`verify` runs eight checks against the experiment's artifacts and writes
[verify.md](../experiments/000-tiny-sft-smoke/verify.md) (`VERDICT:` line per
check, `OVERALL:` line at the end). One line each:

| #   | check               | FAIL when                                                                                                                          |
| --- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `loss_plausibility` | final train loss outside (0.1·ln V, ln V) for vocab size V; **red flag**: loss < 1.0 on an LM task                                 |
| 2   | `eval_train_gap`    | \|final eval loss − final train loss\| ≥ 0.5 (eval = `loss(split=eval)`, `eval_loss`, or `val_loss`)                               |
| 3   | `data_consumption`  | final `num_input_tokens_seen` < 0.7 × `planned_tokens`                                                                             |
| 4   | `stderr_scan`       | `Traceback` / `RuntimeError` / `CUDA out of memory` in logs/stderr.log (warnings listed but PASS)                                  |
| 5   | `param_drift`       | opt-in (SKIP unless the experiment sets `model.target_params`): \|`param_count` − `target_params`\| > 15%                          |
| 6   | `generation_sanity` | logs/samples.jsonl: unique-token ratio < 0.3, one token > 50% of output, a sample < 50 chars — or the file is absent on an LM task |
| 7   | `reward_margin`     | (auto when DPO metrics present) final `rewards/margins` ≤ 0                                                                        |
| 8   | `kl_ref`            | (auto when KL metrics present) mean KL non-finite or ≤ 0                                                                           |

Checks SKIP when their inputs are absent (unless stated otherwise) — 001's
report is `PASS (5 passed, 0 failed, 3 skipped)` because no eval split and no
DPO metrics existed.

**Task-awareness.** Checks read the `task` meta from metrics.jsonl.
`loss_plausibility` applies to LM tasks only (`trl_sft` and unknown count as
LM); it SKIPs for `trl_dpo` and `lightning`, whose losses are not vocab
cross-entropy. `reward_margin` and `kl_ref` switch on automatically when DPO
metrics are present.

**run_start scoping.** metrics.jsonl accumulates across paths and retries; every
check is scoped to records at or after the _last_ `run_start` event (whole file
when none exists). That is why you never delete metrics.jsonl between retries —
see [003-experiment-lifecycle.md](003-experiment-lifecycle.md) §5. A `run_start`
with no metric records after it is a crashed pipeline → exit 2, not "done".

**The red-flag rule.** Final train loss < 1.0 on an LM task FAILs even when the
vocab size is unknown. A "too good" loss means data leakage, a broken loss, or
NaN collapse reported as 0.0 (001's path-2) far more often than it means a great
model. Waiving it is an explicit human decision, never a default.

**Scoping/waiving via `--checks`.** `--checks a,b` re-runs only the named checks
— useful to confirm a single fixed check without re-litigating the rest. A
scoped run only _prints_ its report; it never writes or overwrites verify.md.
Only a full (unscoped) run writes verify.md, so the on-disk verdict always
reflects the complete gate — you cannot launder a pass by cherry-picking checks.
And mechanical PASS is necessary, not sufficient: the human `JUDGMENT:` line on
the generation samples is mandatory before any success claim.

## Ledger semantics

[ledger.md](../experiments/000-tiny-sft-smoke/ledger.md) is a markdown table
managed by `intern.ledger` — one row per path, columns exactly:

```text
| path_id | approach | status | final_train_loss | final_eval_loss | verify | failure_cause | retry_of | gpu_min | run_url |
```

- `status ∈ queued|running|passed|failed|dropped`;
  `verify ∈ pass|fail|pending|n/a`. Unknown columns and invalid enum values are
  rejected by `upsert` (exit 2).
- `retry_of` records lineage: path-2 with `retry_of: path-1` is a node in
  path-1's retry tree, and it is that whole tree the budget gate counts. 001's
  ledger is a six-row single tree — read it top to bottom and you get the entire
  retry story with one `failure_cause` per row.
- No row may be left `running` when the experiment ends; a path that finished
  but failed verify is `failed`/`fail` with a `failure_cause`, not deleted.

## Budget profiles (task-keyed)

The caps a run needs depend on the task, so budgets live as a catalog of
profiles under [configs/budget/](../configs/budget/). Compose the matching one
in the experiment config (`budget: <name>`) and seed the enforced budget.md from
it:

```bash
uv run python scripts/python/intern.py budget --experiment NNN init --profile <name>
```

`init` writes the profile's caps with all spend at zero and **refuses to
overwrite a budget.md that already has recorded spend** (pass `--force` to
re-seed deliberately). This is the one supported way to create budget.md — it
keeps the profile and the enforced artifact from drifting, which hand-copying
numbers does not.

| profile        | paths × retries | GPU-h | param ceiling | sized for                                 |
| -------------- | --------------- | ----: | ------------: | ----------------------------------------- |
| `smoke`        | 1 × 1           |  0.25 |          200M | plumbing / tiny-model smoke checks        |
| `default`      | 2 × 2           |   2.0 |          200M | generic small run (safe fallback)         |
| `lora`         | 2 × 2           |   4.0 |           12B | LoRA / QLoRA on a single GPU (001)        |
| `sft`          | 2 × 2           |   6.0 |            3B | full-parameter SFT (memory-bound)         |
| `dpo`          | 2 × 2           |   8.0 |           12B | preference tuning (ref model doubles fwd) |
| `grpo`         | 3 × 1           |  12.0 |           12B | online RL / GRPO (rollout-bound)          |
| `pretrain`     | 1 × 1           |  24.0 |            2B | from-scratch pretraining                  |
| `autoresearch` | 10 × 1          |   8.0 |          200M | autoresearch-loop generational sweeps     |

Need caps between profiles? Add a new `configs/budget/<name>.yaml` (five cap
keys) rather than hand-editing budget.md — the profile stays the source of
truth. Pick the profile at scaffold time; changing budgets mid-experiment is a
cap edit (see FAQ) or a deliberate `init --force`, not a silent hand-edit.

## FAQ

**The gate denied me — now what?** Read the printed reason; it names the cap and
the numbers (`denied: retries_used 4 >= global retry cap 4 ...`). A denial is an
answer, not an obstacle: do not re-run the command hoping for drift (doom-loop
guard), do not launch anyway. Interactive → present the situation to the human,
who may raise the cap (below). Headless → write the postmortem, fire
`scripts/bash/notify.sh approval_required "<cap hit + what you'd do with more>"`,
and stop that path. If every path is exhausted, that's `notify.sh error` — the
experiment ends with results honestly reported as not achieved, never
`train_done`.

**Can I edit budget.md by hand?** Caps: prefer `init --force` with a fitting
profile, but a one-off cap edit is legitimate before the run they gate, as a
deliberate human decision (raising `max_retries_per_path` because retries so far
were pipeline bugs, not method failures — the `000-tiny-sft-smoke` fixture
effectively lived this). Spent: **never**. The `## Spent` lines are written only
by `record-launch` / `record-retry` / `record-gpu-h`; a hand-edited tally
destroys the audit trail the whole gate rests on.

**What does exit 2 mean?** "You asked the question wrong, or the artifacts
aren't there" — wrong experiment number, missing budget.md/ledger.md, a
malformed file, a missing required flag, or (for `verify`) a run that never
logged metrics. It is neither a pass nor a denial: fix the invocation or the
pipeline and ask again. In particular, never hand-craft the missing artifact (an
empty metrics.jsonl, a fabricated ledger row) to convert a 2 into a 0 — exit 2
on `verify` specifically means training itself is broken.
