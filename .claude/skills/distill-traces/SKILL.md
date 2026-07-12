---
name: distill-traces
description:
  Turn verified agent traces into training data and run the self-distillation
  loop — define a train/eval task split, collect rollouts, filter by
  deterministic verification, convert traces to an SFT dataset, train the
  student, and evaluate it on held-out tasks. Use when the user says "distill
  traces", "train on agent trajectories", "self-distillation", "learn from the
  agent's sessions", "convert traces to a dataset", or wants to fine-tune a
  model on Claude Code / Codex session logs or any agent's rollouts.
---

# distill-traces

**This SKILL.md is a router, not a manual.** It sequences the distillation loop
and points at machinery that already exists: `intern.traces` for storage and
conversion, **new-experiment** for scaffolding, **train-llm** for the run,
**verify-run** for the gate. The trace schema, conversion contracts, and
session-ingestion notes live in `references/trace-conversion.md`; on any
ambiguity, `docs/001-architecture.md` wins.

Distillation discipline in one line: split before you collect, verify before you
accept, hold out before you claim.

**Two distillation modes — pick deliberately.** This skill's loop is
**OFF-policy**: the student imitates fixed teacher traces via `trainer=trl_sft`
(cheap; no teacher at training time; the student never sees its own mistakes —
worked example: `002-distill-off-policy`). **ON-policy** is `trainer=trl_gkd`:
the student samples its own completions and a live `model.teacher` (same
tokenizer) grades them token-level via generalized JSD — costlier (generation +
two resident models) but free of train/inference distribution mismatch (worked
example: `003-distill-on-policy`; the 002/003 pair differs by exactly that one
variable). Start off-policy; go on-policy when the off-policy student plateaus
with exposure-bias symptoms (clean imitation that collapses on its own
rollouts). **SELF-distillation** (STaR/RFT — no external teacher: the model
trains on its OWN verifier-accepted rollouts) is this skill's loop with model =
teacher = student; worked end-to-end example: `004-self-distill`
(`src/data/self_distill.py` + `prep-self-distill.py`).

Blocking-gate rule: Never write results.md or report success unless
`intern.py verify` exited 0. A failed gate means the run failed, regardless of
loss.

## Posture

- Research-before-clarify: task pools, session locations, trace shapes, and
  prior experiments are all discoverable in `~/.claude/projects`,
  `~/.codex/sessions`, and `experiments/` — look before asking.
- Headless: never hang — write best-guess defaults into task.md, fire
  `scripts/bash/notify.sh approval_required "<assumptions>"`, proceed.
  Interactive: one AskUserQuestion, ≤ 4 bundled questions.
- Doom-loop guard: 3 identical tool calls with no new information → write
  `experiments/NNN-<slug>/blocker.md`, fire
  `scripts/bash/notify.sh blocker "<summary>"`, stop.
- Context discipline: session files and trace JSONL are big — read them with
  `head`/`grep`/`jq` only; never cat a whole session into context.

## Workflow

### 1. Define the task pool and the split — BEFORE collecting anything

Write the task list and a train/eval split into the experiment's task.md before
the first rollout is collected. Contamination guard: **eval tasks never yield
training traces** — a record whose `task_id` belongs to the eval split is
written with `split="eval"` and must never reach a converter call. A split
invented after collection is not a split; it is a leak with paperwork.

### 2. Collect rollouts into the trace store

Traces live in `experiments/NNN-<slug>/traces/*.jsonl` (gitignored — they carry
prompts, tool output, and local paths). No experiment number yet → run
**new-experiment** first; traces need a numbered home. Every trace is one
`TraceRecord` appended via `intern.traces.TraceStore` with `task_id`, `split`,
and the full conversation — schema in `references/trace-conversion.md`.

Two sources:

- Fresh rollouts: run the agent on train-split tasks, append one record per
  episode.
- Mined sessions: Claude Code `~/.claude/projects`, Codex `~/.codex/sessions`.
  REVIEW AND REDACT secrets/PII before ingestion — the redaction checklist is in
  `references/trace-conversion.md`. Redaction happens before data enters the
  store, not before publishing.

### 3. Acceptance filtering — verification first, judge second

- Deterministic verification FIRST: did the task actually succeed — tests pass,
  output matches, artifact exists. Record `verifier_output` and `accepted` on
  every trace, including rejects.
- Judge critique SECOND, and only for quality ranking among already-verified
  traces (store it in `judge_critique`). A judge is never the sole acceptance
  signal — teacher and judge can share the same blind spot, and distilling
  unverified outputs reinforces it.
- Keep rejected traces (`accepted=false`) — accepted/rejected pairs are
  preference-data material later.

### 4. Decide the imitation target — one per path

Pick what the student should imitate: final answers | tool-call decisions |
repair behavior | full transcripts. One target per experiment path, per the
one-variable rule in the plan.md contract — name it as the path's hypothesis.
`references/trace-conversion.md` maps each target to its converter.

### 5. Convert traces to a dataset

Use the converters from `intern.traces` — worked snippet in
`references/trace-conversion.md`:

- `to_sft_messages(records)` → `messages` + `tools` rows.
- `to_prompt_completion(records, tokenizer)` → `prompt`/`completion` rows,
  rendered per final assistant turn.

Both default to `only_accepted=True`; additionally filter to `split == "train"`.
The prompt/completion converter asserts chat-template prefix consistency — a
`ValueError` naming a prefix mismatch means the template rewrites history when a
turn is appended, and completion-loss boundaries would corrupt silently. Fix the
template or model choice; never bypass the assert. Eyeball 5 converted rows
before training.

### 6. Train through the normal gates

Scaffold the training experiment via **new-experiment** (reuse the collection
experiment when this run is its first path) and hand off to **train-llm** on the
`trainer=trl_sft` lane. The dataset-formats reference in train-llm applies
verbatim — including its tool-calling checks when traces contain tool calls.
Budget gate, smoke gate, and verify gate apply unchanged; a distilled dataset
buys no exemptions.

### 7. Evaluate the student — distillation guardrails

- Evaluate on the HELD-OUT eval task split, never on training tasks.
- Compare the student's task success rate against the teacher's on the same eval
  tasks — that comparison belongs in results.md.
- A student that only matches the teacher on train-split tasks is memorization,
  not distillation — report it as a failure, not a partial success.
- **verify-run** still gates the training run itself; the held-out comparison is
  on top of, not instead of, `intern.py verify`.

## Done conditions

- [ ] Task pool and train/eval split written down before the first trace was
      collected; no eval task produced a training trace.
- [ ] Traces in `experiments/NNN-<slug>/traces/*.jsonl`, every record carrying
      `split`, `accepted`, and `verifier_output`; mined sessions redacted before
      ingestion.
- [ ] Acceptance decided by deterministic verification; judge critique used only
      for ranking.
- [ ] Exactly one imitation target per path, named in plan.md.
- [ ] Conversion ran without a prefix-mismatch ValueError; 5 converted rows
      eyeballed.
- [ ] Training ran through train-llm with budget/smoke/verify gates; verify
      exited 0 before any success claim.
- [ ] results.md contains the student-vs-teacher comparison on the held-out
      split.
