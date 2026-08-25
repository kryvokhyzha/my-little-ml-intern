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

**This SKILL.md is a router, not a manual.** It puts the distillation loop in
order. It points at machinery that already exists: `intern.traces` for storage
and conversion, **new-experiment** for the scaffold, **train-llm** for the run,
**verify-run** for the gate. The trace schema, the conversion contracts, and the
session-ingestion notes live in `references/trace-conversion.md`. If a rule is
ambiguous, `docs/001-architecture.md` wins.

Distillation discipline has three rules. Split the task pool before you collect
the traces. Verify a trace before you accept it. Hold out the eval tasks before
you claim a result.

**Two distillation modes — pick one deliberately.** This skill's loop is
**OFF-policy**: the student imitates fixed teacher traces through
`trainer=trl_sft`. Off-policy is cheap. It needs no teacher at training time.
The student never sees its own mistakes. Worked example:
`002-distill-off-policy`. **ON-policy** is `trainer=trl_gkd`: the student
samples its own completions, and a live `model.teacher` (same tokenizer) grades
them token-level through generalized JSD. On-policy costs more: it runs
generation and it keeps two resident models. On-policy has no train/inference
distribution mismatch. Worked example: `003-distill-on-policy`. The 002/003 pair
differs by exactly that one variable. Start off-policy. Go on-policy when the
off-policy student plateaus with exposure-bias symptoms: the student imitates
cleanly but collapses on its own rollouts. **SELF-distillation** (STaR/RFT) uses
no external teacher: the model trains on its OWN verifier-accepted rollouts.
Self-distillation is this skill's loop with model = teacher = student. Worked
end-to-end example: `004-self-distill` (`src/data/self_distill.py` +
`prep-self-distill.py`).

Blocking-gate rule: Never write results.md or report success unless
`intern.py verify` exited 0. A failed gate means the run failed, regardless of
loss.

## Posture

- Research before you clarify. You can find the task pools, the session
  locations, the trace shapes, and the prior experiments in
  `~/.claude/projects`, `~/.codex/sessions`, and `experiments/`. Look before you
  ask.
- Headless: do not wait for an answer. Write the best-guess defaults into
  task.md. Fire `scripts/bash/notify.sh approval_required "<assumptions>"`. Then
  continue. Interactive: ask one AskUserQuestion with ≤ 4 bundled questions.
- Doom-loop guard: when 3 identical tool calls return no new information, write
  `experiments/NNN-<slug>/blocker.md`. Fire
  `scripts/bash/notify.sh blocker "<summary>"`. Then stop.
- Context discipline: session files and trace JSONL files are large. Read them
  only with `head`, `grep`, or `jq`. Never load a whole session file into the
  context.

## Workflow

### 1. Define the task pool and the split — BEFORE you collect anything

Write the task list and a train/eval split into the experiment's task.md. Do
this before you collect the first rollout.

Contamination guard: **eval tasks never produce training traces**. If a record's
`task_id` belongs to the eval split, write the record with `split="eval"`. Never
pass such a record to a converter call. A split that you create after the
collection is not a split. It only documents a leak.

### 2. Collect rollouts into the trace store

Traces live in `experiments/NNN-<slug>/traces/*.jsonl`. Git ignores them,
because they carry prompts, tool output, and local paths. If the experiment has
no number yet, run **new-experiment** first. The traces need a numbered
experiment directory.

Every trace is one `TraceRecord`. Append each record with
`intern.traces.TraceStore`. Each record carries `task_id`, `split`, and the full
conversation. `references/trace-conversion.md` holds the schema.

Two sources:

- Fresh rollouts: run the agent on the train-split tasks. Append one record for
  each episode.
- Mined sessions: Claude Code writes to `~/.claude/projects`. Codex writes to
  `~/.codex/sessions`. REVIEW AND REDACT the secrets and the PII before you
  ingest a session. `references/trace-conversion.md` holds the redaction
  checklist. Redact the data before it enters the store, not before you publish
  it.

### 3. Filter for acceptance — verification first, judge second

- Run the deterministic verification FIRST: did the task succeed? Check that the
  tests pass, that the output matches, and that the artifact exists. Record
  `verifier_output` and `accepted` on every trace, also on the rejected traces.
- Run the judge critique SECOND. Use the judge only to rank the quality of the
  traces that already passed the verification. Store the critique in
  `judge_critique`. A judge is never the only acceptance signal. The teacher and
  the judge can share the same blind spot. If you distill unverified outputs,
  you make that blind spot stronger.
- Keep the rejected traces (`accepted=false`). An accepted/rejected pair can
  become preference data later.

### 4. Decide the imitation target — one per path

Pick what the student imitates: final answers | tool-call decisions | repair
behavior | full transcripts. Use one target for each experiment path. This
follows the one-variable rule in the plan.md contract. Name the target as the
path's hypothesis. `references/trace-conversion.md` maps each target to its
converter.

### 5. Convert traces to a dataset

`references/trace-conversion.md` holds a worked snippet. Use these converters
from `intern.traces`:

- `to_sft_messages(records)` → `messages` + `tools` rows.
- `to_prompt_completion(records, tokenizer)` → `prompt`/`completion` rows,
  rendered per final assistant turn.

Both converters default to `only_accepted=True`. Also filter the records to
`split == "train"`.

The prompt/completion converter asserts chat-template prefix consistency. A
`ValueError` that names a prefix mismatch means that the template rewrites the
history when you append a turn. The completion-loss boundaries then corrupt
silently. Fix the template or the model choice. Never bypass the assert. Read 5
converted rows before you train.

### 6. Train through the normal gates

Scaffold the training experiment with **new-experiment**. Reuse the collection
experiment when this run is its first path. Then hand off to **train-llm** on
the `trainer=trl_sft` lane.

The dataset-formats reference in train-llm applies verbatim. It also applies its
tool-calling checks when the traces contain tool calls. The budget gate, the
smoke gate, and the verify gate apply unchanged. A distilled dataset gets no
exemption.

### 7. Evaluate the student — distillation guardrails

- Evaluate the student on the HELD-OUT eval task split. Never evaluate the
  student on the train-split tasks.
- Compare the student's task success rate with the teacher's rate on the same
  eval tasks. Write that comparison in results.md.
- A student that matches the teacher only on the train-split tasks memorized the
  traces. This result is not distillation. Report it as a failure, not as a
  partial success.
- **verify-run** still gates the training run itself. The held-out comparison is
  an addition to `intern.py verify`, not a replacement for it.

## Done conditions

- [ ] You wrote the task pool and the train/eval split before you collected the
      first trace. No eval task produced a training trace.
- [ ] The traces are in `experiments/NNN-<slug>/traces/*.jsonl`. Every record
      carries `split`, `accepted`, and `verifier_output`. You redacted the mined
      sessions before you ingested them.
- [ ] The deterministic verification decided the acceptance. You used the judge
      critique only to rank the traces.
- [ ] Each path has exactly one imitation target. plan.md names the target.
- [ ] The conversion ran without a prefix-mismatch ValueError. You read 5
      converted rows.
- [ ] The training run went through train-llm with the budget gate, the smoke
      gate, and the verify gate. verify exited 0 before any success claim.
- [ ] results.md contains the student-vs-teacher comparison on the held-out
      split.
