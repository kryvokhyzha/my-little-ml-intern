# 003 — Experiment lifecycle

<!-- Worked walkthrough of the full gate chain — scaffold → budget → smoke → train →
verify → results → publish — using the committed experiments/000-tiny-sft-smoke as the
specimen. The contract lives in docs/001-architecture.md; this doc shows it in motion. -->

This is a guided tour of one real experiment,
[experiments/000-tiny-sft-smoke](../experiments/000-tiny-sft-smoke/task.md).
Every artifact referenced here is committed; you can open each file as you read.
The experiment took **six paths** to pass its own verification gate, and that is
the point: five of the six runs produced a loss number that looked fine, and the
gates failed them anyway.

`000-tiny-sft-smoke` is the repo's plumbing self-test fixture — the `000-`
prefix marks it as a fixture, not a task; real task examples start at `001` (the
`gemma-4-E2B-it` pi-mono QLoRA SFT run, walked through in
[docs/007-example-pi-mono-sft.md](007-example-pi-mono-sft.md)).

## 1. The triple

One experiment number = three artifacts, created together by the
`new-experiment` skill:

```text
scripts/python/000-tiny-sft-smoke.py    # hydra entrypoint, dispatches on cfg.trainer.kind
configs/000-tiny-sft-smoke.yaml         # composes main + trainer/tracking/compute/budget
experiments/000-tiny-sft-smoke/         # task.md, plan.md, budget.md, ledger.md, run.md + run artifacts
```

See [the script](../scripts/python/000-tiny-sft-smoke.py) and
[the config](../configs/000-tiny-sft-smoke.yaml). Nothing trains outside a
numbered triple (`999-` scratch is the only gate-exempt exception).

## 2. task.md and plan.md — the hypothesis contract

[task.md](../experiments/000-tiny-sft-smoke/task.md) restates the task, lists
unknowns, and fixes the run mode (interactive vs headless — this decides whether
gates ask a human or fire `notify.sh approval_required` and proceed).

[plan.md](../experiments/000-tiny-sft-smoke/plan.md) carries the hypothesis
contract: every hypothesis needs a `mechanism`, a numeric `expected_delta`, and
a `falsification` — what result kills it. 001's actual hypothesis, verbatim:

```text
| id     | change                                      | mechanism                                                       | expected_delta                             | falsification                            |
| path-1 | 40 SFT steps on 256-row toy corpus, lr 5e-4 | gradient descent on a tiny repetitive corpus reduces train loss | final loss < 10.0 (from ~ln(50257) ≈ 10.8) | loss stays ≥ 10.8 or NaN within 40 steps |
```

One variable — ideally one Hydra override — per solution path. No production
file may be edited before the change is a named hypothesis. Note how the story
plays out below: path-1's final loss (10.587) landed in the gray zone between
its expected delta and its falsification criterion; the verify gate, not the
loss number, was decisive.

## 3. The budget gate

Before any launch:

```bash
uv run python scripts/python/intern.py budget --experiment 001 can-launch
uv run python scripts/python/intern.py budget --experiment 001 can-launch --params 135000000
```

Exit 0 = allowed, 1 = denied (a reason line is printed), 2 = usage error or
missing artifacts. On launch, the spend is recorded — by the CLI, never by hand:

```bash
uv run python scripts/python/intern.py budget --experiment 001 record-launch
uv run python scripts/python/intern.py ledger --experiment 001 upsert --path-id path-1 --status running --approach "40-step toy SFT, lr 5e-4"
```

Caps, the spent tally, and retry-tree accounting are explained in
[004-budget-and-gates.md](004-budget-and-gates.md).

## 4. The smoke gate

Mandatory before any long run, no exceptions:

```bash
uv run python scripts/python/000-tiny-sft-smoke.py smoke_test=true
```

`smoke_test=true` (or env `SMOKE_TEST=1`) changes three things: `max_steps=1`,
the dataset is sliced to ≤ 32 rows, and no checkpoint is saved. The adapter must
print a machine-greppable verdict line:

```text
VERDICT: TRAIN_OK | final_train_loss=3.6047
```

`VERDICT: TRAIN_FAIL | <cause>` → fix the cause and re-smoke. Launching a long
run on a failed or skipped smoke is a workflow violation. In the smoke's
`metrics.jsonl` (gitignored — it regenerates on every run) you can see the smoke
run (`"smoke": true` at 15:17:50) eleven seconds before the real run
(`"smoke": false` at 15:18:01).

## 5. The real run and metrics.jsonl anatomy

The adapter streams everything into the experiment's append-only
`metrics.jsonl`. Real lines from the `000-tiny-sft-smoke` run (timestamps
shortened):

```text
{"ts": "...T15:18:01+00:00", "event": "run_start", "task": "trl_sft", "run_name": null, "smoke": false}
{"ts": "...", "event": "meta", "key": "param_count", "value": 134515008}
{"ts": "...", "event": "meta", "key": "vocab_size", "value": 49152}
{"ts": "...", "event": "meta", "key": "planned_tokens", "value": 5000}
{"ts": "...", "step": 1, "split": "train", "name": "loss", "value": 3.6047}
{"ts": "...", "step": 10, "split": "train", "name": "num_input_tokens_seen", "value": 5528.0}
```

Four record kinds: `run_start` events (run boundary — the adapter's FIRST
action, before anything that can crash), `meta` events (`task`, `param_count`,
`vocab_size`, `planned_tokens` — the verify gate's inputs), metric records
(step/split/name/value), and `alert` events fired by the callbacks
(`<metric>=<value> at step <N> — <hypothesis>, try <action>`).

**Never delete metrics.jsonl between retries.** The file accumulates across
paths and retries by design; verify scopes every check to records at or after
the _last_ `run_start`, so stale records cannot contaminate a new verdict —
deletion buys you nothing mechanically. What it costs you: the raw history that
postmortems and path comparisons are written from, and the crashed-pipeline
detector (a `run_start` with no metric records after it makes verify exit 2 —
"the run never logged anything" is a pipeline bug, not "done"). 001's file still
contains all its history; only the tail after the last `run_start` was judged.

`logs/stderr.log` is the opposite: truncated per run by the adapter's tee, so
`stderr_scan` only ever sees the current run. Don't shell-redirect into it.

## 6. The retry story: six paths to one passing run

This is the heart of the lifecycle — a real demonstration that **a low loss
number is never evidence the model works**. Each row of
[ledger.md](../experiments/000-tiny-sft-smoke/ledger.md) is a one-variable
mutation of its parent (`retry_of` lineage), each failure has a postmortem
(symptom → root-cause hypothesis → fix), and each was killed by a specific gate,
not by a human squinting at a curve.

**path-1 — plausible loss, garbage model.** 40 steps on `sshleifer/tiny-gpt2`,
final loss 10.587 — inside the ln(vocab) plausibility band. Verify exit 1:
`generation_sanity` FAIL — all three samples collapsed to a single repeated
token ("factors" = 96–97% of output). The checkpoint was a randomly-initialized
~100k-param test model that cannot learn language at any step count; loss barely
moved (10.61 → 10.59) yet looked fine in isolation.
[postmortems/path-1.md](../experiments/000-tiny-sft-smoke/postmortems/path-1.md).
Fix: a genuinely pretrained model, `EleutherAI/pythia-14m`.

**path-2 — the "best" loss of the whole experiment: 0.0.** pythia-14m at lr 5e-4
diverged: `grad_norm: nan` mid-run, weights went NaN, and the reported loss
collapsed to an impossible 0.0. Caught by the `loss_plausibility` **red-flag
rule** (loss < 1.0 on an LM task is a FAIL even when vocab size is unknown) plus
`generation_sanity` (empty continuations). Side finding: the NaN alert watched
the loss value, which reported 0.0 rather than NaN — `grad_norm` is now watched
too.
[postmortems/path-2.md](../experiments/000-tiny-sft-smoke/postmortems/path-2.md).
Fix: the canonical divergence mutation, lr 5e-4 → 5e-5.

**path-3 — the hypothesis dies, the real bug surfaces.** Identical NaN
divergence at lr 5e-5 — the lr hypothesis from path-2 was falsified, which is
exactly what the falsification field is for. Bisecting the adapter found the
root cause: the pythia-14m checkpoint is stored in fp16 and transformers v5
loads checkpoints in their **stored dtype** by default → full-precision fp16
AdamW → divergence on MPS, `RuntimeError: mixed dtype (CPU)` on CPU.
[postmortems/path-3.md](../experiments/000-tiny-sft-smoke/postmortems/path-3.md).
Fix in code, not config: the model loader now injects `dtype: float32` (with a
warning) when `model.main` carries neither `dtype` nor `quantization_config`.

**path-4 — fixed the bug, found two more truths.** With fp32, no NaN — but on
MPS the loss _rose_ 4.48 → 7.29 (GPTNeoX numerics are broken on MPS even in
fp32; `generation_sanity` FAIL). The healthy CPU rerun then failed verify too:
40 steps memorized the tiny 5-template corpus, final loss 0.60 — red-flag FAIL
again, this time the check working as designed on a toy task.
[postmortems/path-4.md](../experiments/000-tiny-sft-smoke/postmortems/path-4.md).
Fix: `use_cpu: true` and `max_steps: 10`.

**path-5 — four checks pass, the fifth says no.** CPU, 10 steps, loss 1.671 —
and `generation_sanity` still FAILs: one of three samples collapses into a
repeated token ("night" = 66% of output). A 14M model fine-tuned on a templated
corpus has a real degeneracy attractor; the check correctly reported partial
distribution collapse. A samples-format bug (blank-line-delimited samples.txt)
was found and fixed along the way — `samples.jsonl` replaced it.
[postmortems/path-5.md](../experiments/000-tiny-sft-smoke/postmortems/path-5.md).
Fix: `HuggingFaceTB/SmolLM2-135M`.

**path-6 — green, end to end.** SmolLM2-135M, CPU, 10 steps, lr 5e-5, fp32.
Final loss 1.779 — _worse_ than path-2's 0.0, path-4's 0.60, and path-5's 1.671
— and it is the only run that passed, because loss was never the criterion.
Verify: `OVERALL: PASS (5 passed, 0 failed, 3 skipped)`.

Sit with the scoreboard for a second: sorted by final loss, the ranking is
path-2 (0.0), path-4 (0.60), path-5 (1.671), path-6 (1.779), path-4-MPS (7.29),
path-1 (10.587). The _winner is fourth_. Every run above it was broken in a way
the loss could not show — NaN collapse, memorization, degenerate generations.
That is what the gates are for.

One budget footnote: [budget.md](../experiments/000-tiny-sft-smoke/budget.md)
records `retries_used: 5` against `max_retries_per_path: 2` — this interactive,
human-supervised pipeline shakedown ran past the default caps, and the tally
shows it. That overrun being _visible_ is the point of CLI-only spend
accounting. Headless, the gate's denial is final: postmortem, notify, stop.

## 7. verify-run and the JUDGMENT line

```bash
uv run python scripts/python/intern.py verify --experiment 001   # exit 0/1/2
```

A full run writes [verify.md](../experiments/000-tiny-sft-smoke/verify.md) — one
machine-greppable line per check plus an `OVERALL:` line:

```text
VERDICT: loss_plausibility = PASS | value=1.779 | threshold=(1.08, 10.8) | final train loss inside ln(vocab) band
VERDICT: generation_sanity = PASS | value=0.369 | threshold=unique_ratio>=0.3, max_token_share<=0.5, chars>=50 | 3 sample(s) pass mechanical proxies — eyeball them; this is necessary, not sufficient
OVERALL: PASS (5 passed, 0 failed, 3 skipped)
```

Mechanical PASS is necessary, not sufficient. The verify-run skill then requires
a human read of
[logs/samples.jsonl](../experiments/000-tiny-sft-smoke/logs/samples.jsonl) — is
this recognizable language for the training distribution? — recorded as one
appended line:

```text
JUDGMENT: generation_quality = PASS | story-like English continuations, consistent with the toy corpus
```

A FAIL judgment fails the run overall even when the exit code was 0. The check
thresholds, task-awareness, and exit codes are catalogued in
[004-budget-and-gates.md](004-budget-and-gates.md).

## 8. run.md and results.md

As the run executes, `train-llm` records the exact commands it ran on the
compute instance into [run.md](../experiments/000-tiny-sft-smoke/run.md) —
provision → setup → train → benchmark → teardown. It is **ungated** (a crashed
run still gets its commands, so the failure reproduces) and complements
`results.md`, which is the gated write-up.

results.md is forbidden unless verify exited 0 (and the judgment passed). Only
then, after the ledger row is updated to `--status passed --verify pass`, write
[results.md](../experiments/000-tiny-sft-smoke/results.md): the winner, the path
comparison table, and — the highest-value section of 001 — the lessons folded
back into the repo (the fp32 default, the MPS/GPTNeoX warning, seeded sampling
for sanity generations).

## 9. The status dashboard

Any time you want the state of an experiment in one read:

```bash
uv run python scripts/python/intern.py status --experiment 001          # human-readable
uv run python scripts/python/intern.py status --experiment 001 --json   # machine-readable
```

It shows the verify verdicts, budget caps vs spent, and the ledger rows in one
place. Exit 0 (2 when the experiment doesn't exist). It is a read-only
dashboard, not a gate — use it before deciding the next action.

## 10. Where publish fits

Publishing is the last gate, and it is blocking:

```bash
uv run python scripts/python/intern.py publish --experiment 001 [--repo-id org/name] [--private true|false]
```

`publish` re-runs verify (must exit 0), requires results.md plus a ledger row
with `status=passed` and `verify=pass`, then uploads the newest `ckpts/` model
directory and the reproducibility bundle (task/plan/budget/ledger/verify
/results.md, `logs/samples.jsonl`, `configs/NNN-<slug>.yaml`) to the HF Hub with
a model card generated from results.md, and appends a `## Published` section
with the URL to results.md. Exit 0 published, 1 gate refused, 2 missing
artifacts or credentials.

Honest note about the specimen: 001 trained with `save_strategy: "no"`, so its
`ckpts/` is empty and `publish` would exit 2. A real experiment that intends to
publish must save checkpoints into `experiments/NNN-<slug>/ckpts/`.

## Command crib sheet

```bash
# 0. scaffold the triple (new-experiment skill):
#    scripts/python/NNN-<slug>.py + configs/NNN-<slug>.yaml + experiments/NNN-<slug>/

# 1. budget gate — stop on nonzero
uv run python scripts/python/intern.py budget --experiment NNN can-launch [--params <target_params>]

# 2. smoke gate — must print "VERDICT: TRAIN_OK"
uv run python scripts/python/NNN-<slug>.py smoke_test=true

# 3. record, then launch the real run
uv run python scripts/python/intern.py budget --experiment NNN record-launch
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status running --approach "<one line>"
scripts/bash/notify.sh train_started "path-1 on <dataset> via <lane>"
uv run python scripts/python/NNN-<slug>.py

# 4. verify gate — blocking; then read logs/samples.jsonl and append the JUDGMENT line
uv run python scripts/python/intern.py verify --experiment NNN

# 5a. pass: record spend + outcome, then (and only then) write results.md
uv run python scripts/python/intern.py budget --experiment NNN record-gpu-h --hours <h>
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status passed --verify pass

# 5b. fail: postmortem + ledger, retry only if the gate allows
#     write experiments/NNN-<slug>/postmortems/path-1.md   (symptom → root cause → fix)
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status failed --verify fail --failure-cause "<one line>"
uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1
uv run python scripts/python/intern.py budget --experiment NNN record-retry
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-2 --status running --retry-of path-1 --approach "<mutation>"

# 6. dashboard + publish (blocking; re-runs verify, needs results.md + passing ledger row + ckpts/)
uv run python scripts/python/intern.py status --experiment NNN
uv run python scripts/python/intern.py publish --experiment NNN [--repo-id org/name] [--private true|false]
```
