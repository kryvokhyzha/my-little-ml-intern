---
name: autoresearch-loop
description:
  Generational orchestrator that runs many bounded training experiments to find
  the best config — seed diverse hypotheses, critique them before spending
  compute, execute survivors through the budget and verify gates, promote a
  champion on a shared board, and climb a stagnation ladder instead of quitting.
  Use whenever the user says "autoresearch", "run many experiments", "sweep for
  the best config", "beat metric X", "keep experimenting overnight", or "ablate"
  — any request to search a config space empirically rather than execute one
  known recipe. Runs long, on its own initiative, until a budget cap or a real
  stop condition.
---

# autoresearch-loop

**This skill sequences other skills — it never restates their steps.**
Scaffolding is **new-experiment**, recipe research is
**literature-recipe-research**, launching/monitoring is **train-llm**, the gate
is **verify-run**, dashboards are **track-experiments**. This skill owns only
the loop around them: seed → critique → execute → share → escalate → stop. The
artifact contract (board.md format, plan.md `## Loop` / `## Next levers`,
budget/ledger formats) lives in `docs/001-architecture.md` — that doc wins on
any ambiguity.

Never write results.md or report success unless `intern.py verify` exited 0. A
failed gate means the run failed, regardless of loss.

## Posture

- Research-before-clarify: never ask the user about anything you could look up
  (prior experiments, ledger state, dataset schemas, the current champion).
- Headless: never hang — write best-guess defaults, fire
  `scripts/bash/notify.sh approval_required "<assumptions>"`, proceed.
  Interactive: one AskUserQuestion, ≤ 4 bundled questions.
- Doom-loop guard: 3 identical tool calls with no new information → write
  `experiments/NNN-<slug>/blocker.md`, fire
  `scripts/bash/notify.sh blocker "<summary>"`, stop.
- Context discipline: subagents (critics, monitors) return concise reports only;
  read logs and metrics.jsonl with head/tail/grep.
- The budget is a floor as well as a ceiling: caps stop new paths — they never
  license quitting with budget remaining. Never end a turn asking "should I
  continue?" while `can-launch` exits 0 and live hypotheses remain.

## Workflow

### 0. Preconditions

- The experiment triple exists, scaffolded via **new-experiment**, with the
  config composing `budget: autoresearch` (`configs/budget/autoresearch.yaml`:
  10 paths, 1 retry per path, 8 GPU-h) — the default group's 2 paths cannot feed
  a generational loop.
- Method, dataset, or hyperparameters not already fixed → run
  **literature-recipe-research** first; its table lands in
  `experiments/NNN-<slug>/research.md` and seeds angles below.
- Create `experiments/NNN-<slug>/board.md` from the board.md format in
  `docs/001-architecture.md` when absent (empty sections, no champion yet).

### 1. SEED

Write 3-6 hypotheses into plan.md, each carrying the full plan contract —
`mechanism`, `expected_delta`, `falsification` — PLUS an angle tag from
`references/idea-angles.md`. Rules:

- Anti-convergence: no angle may hold more than 30% of live hypotheses. Replace
  the excess with ideas from under-represented angles (spin variants per the
  reference when the pool is homogeneous).
- Exactly ONE Hydra override per hypothesis = one solution path in the plan.md
  paths table.
- Add/extend the plan.md `## Loop` section: generation counter and which angles
  are covered vs untouched.

### 2. CRITIQUE — before any compute

Score every seeded hypothesis 0-10 on mechanism plausibility × novelty against
board.md `## Dead ends` (a near-duplicate of a dead end scores 0 on novelty).
Additional criteria (SmolLM playbook ablation discipline — perfect ablations on
irrelevant choices waste as much compute as sloppy ablations):

- **Derisked bar** — the hypothesis names its expected improvement OR a
  side-benefit (speed/memory/stability) tied to its falsification condition.
- **Two-questions filter** — drop hypotheses that neither address a known
  weakness nor exploit a known lever.
- **Information value** — `expected_delta` smaller than the success metric's
  observed run-to-run noise = unfalsifiable: score 0, never spend compute on it.

In-context self-critique by default; when subagents are available, spawn 2-3
parallel critics and average their scores. Drop everything below 6 — dropped
hypotheses never reach the paths table and cost no budget. Record the drop and
the score in plan.md.

### 3. EXECUTE

Run each survivor as a path through the **train-llm** skill in full — budget
`can-launch` gate, dataset validation, smoke, launch, monitor — then the
**verify-run** skill. Do not shortcut either skill's gates from here. Set
`tracking.group=gen-<N>` on every path so a generation clusters on the dashboard
(**track-experiments** documents the group knob).

One path at a time locally. Parallel fan-out is allowed only when compute lanes
differ (e.g. one local + one ssh path) — never batch-launch on a shared lane.

### 4. SHARE

After each path's verify-run completes (verify-run already updated the ledger):

- Update board.md. Champion = best path by the plan.md success metric, and
  `verify=pass` is required — a path without a passing verify can never be
  champion, whatever its number.
- Mechanism audit: did the observed delta match `expected_delta`? Confirmed →
  `## Verified wins`, one line. Refuted mechanism — no delta, or a real delta
  from a different cause — goes to `## Dead ends` with the why; a refuted
  mechanism often seeds a better next-generation hypothesis than the win itself.
- Every `## Dead ends` line names its bottleneck class: model capacity | data
  quality | reward design | environment behavior | evaluator coverage |
  infrastructure. The next lever should change the bottleneck, not only the
  learning rate — never repeat a failed variant without a documented change of
  method/reward/model/data/evaluator.
- Increment the generation counter in plan.md `## Loop`; append fresh ideas to
  `## Next levers` (board.md and plan.md).

### 5. STAGNATION LADDER

No champion improvement for 2 generations → climb exactly one rung; log every
climb in board.md `## Stagnation log`:

1. **Rung 1 — tweak the champion.** One-variable variations of the champion's
   own override (magnitude, schedule, neighbor values).
2. **Rung 2 — orthogonal angle.** Hypotheses from angles the champion's family
   and the dead-ends list do not touch (see the rung-2 mapping in
   `references/idea-angles.md`).
3. **Rung 3 — structural reframe.** A different method, not a tweak. A new
   method is a new plan.md baseline — re-run **literature-recipe-research**
   first, then re-enter at step 1 with the new baseline.

### 6. STOP

Stop ONLY when one of these holds:

- **Budget caps:** `can-launch` / `can-retry` denials (exit 1). Report which cap
  was hit.
- **Rung 3 dry for 2 rounds:** two consecutive structural-reframe attempts
  produce no hypothesis that survives critique.
- **Explicit success criterion met:** the success metric from task.md/plan.md is
  reached by a champion with `verify=pass`.

Caps are a floor as well as a ceiling — never quit early with budget remaining
and live hypotheses; "out of small tweaks" means climb a rung, not stop.
results.md is written only through verify-run's pass route. On stop with no
passing path, fire `scripts/bash/notify.sh error "<cause>"` — never train_done.

## Done conditions

- [ ] board.md exists with a `## Champion` entry backed by ledger `verify=pass`.
- [ ] Every launched path has a ledger row (none left `running`) and a verify
      verdict.
- [ ] budget.md `## Spent` tally is consistent with the ledger (paths_launched
      matches rows, retries and GPU-h recorded).
- [ ] plan.md carries `## Loop` with the final generation count and angle
      coverage; dead ends and stagnation climbs are on the board.
- [ ] `notify.sh train_done` fired only when a champion has `verify=pass`.
- [ ] No results.md unless `intern.py verify` exited 0.
