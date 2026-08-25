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

**This skill sequences other skills — it never restates their steps.** The
**new-experiment** skill scaffolds the experiment. The
**literature-recipe-research** skill finds the recipe. The **train-llm** skill
launches and monitors the run. The **verify-run** skill runs the gate. The
**track-experiments** skill owns the dashboards. This skill owns only the loop
around them: seed → critique → execute → share → escalate → stop. The artifact
contract (board.md format, plan.md `## Loop` / `## Next levers`, budget/ledger
formats) lives in `docs/001-architecture.md`. That doc wins on any ambiguity.

Never write results.md unless `intern.py verify` exited 0. Never report success
unless that gate exited 0. A failed gate means the run failed, regardless of the
loss.

## Posture

- Research-before-clarify: never ask the user about anything you can look up
  (prior experiments, ledger state, dataset schemas, the current champion).
- Headless: never hang. Write the best-guess defaults. Fire
  `scripts/bash/notify.sh approval_required "<assumptions>"`. Then proceed.
  Interactive: ask one AskUserQuestion with ≤ 4 bundled questions.
- Doom-loop guard: after 3 identical tool calls with no new information, write
  `experiments/NNN-<slug>/blocker.md`. Then fire
  `scripts/bash/notify.sh blocker "<summary>"`. Then stop.
- Context discipline: the subagents (critics, monitors) return concise reports
  only. Read the logs and metrics.jsonl with head/tail/grep.
- The budget sets a minimum as well as a maximum. A cap stops a new path. A cap
  never permits you to stop while budget remains. Never end a turn with the
  question "should I continue?" while `can-launch` exits 0 and live hypotheses
  remain.

## Workflow

### 0. Preconditions

- The experiment triple exists. The **new-experiment** skill scaffolds it. The
  config composes `budget: autoresearch` (`configs/budget/autoresearch.yaml`: 10
  paths, 1 retry per path, 8 GPU-h). The default group gives 2 paths, which
  cannot feed a generational loop.
- If the method, the dataset, or the hyperparameters are not already fixed, run
  **literature-recipe-research** first. Its table lands in
  `experiments/NNN-<slug>/research.md` and seeds the angles below.
- Create `experiments/NNN-<slug>/board.md` when it is absent. Use the board.md
  format in `docs/001-architecture.md` (empty sections, no champion yet).

### 1. SEED

Write 3-6 hypotheses into plan.md. Each hypothesis carries the full plan
contract — `mechanism`, `expected_delta`, `falsification`. Each hypothesis also
carries an angle tag from `references/idea-angles.md`. Rules:

- Anti-convergence: no angle may hold more than 30% of the live hypotheses.
  Replace the excess with ideas from under-represented angles. Spin variants per
  the reference when the pool is homogeneous.
- Give each hypothesis exactly ONE Hydra override. One override is one solution
  path in the plan.md paths table.
- Add or extend the plan.md `## Loop` section. Record the generation counter.
  Record which angles are covered and which are untouched.

### 2. CRITIQUE — before any compute

Score every seeded hypothesis 0-10 on mechanism plausibility × novelty against
board.md `## Dead ends`. A near-duplicate of a dead end scores 0 on novelty. The
SmolLM playbook ablation discipline also applies: a perfect ablation on an
irrelevant choice wastes as much compute as a sloppy ablation. Apply these
additional criteria:

- **Derisked bar** — the hypothesis names its expected improvement OR a
  side-benefit (speed/memory/stability) tied to its falsification condition.
- **Two-questions filter** — drop hypotheses that neither address a known
  weakness nor exploit a known lever.
- **Information value** — an `expected_delta` smaller than the success metric's
  observed run-to-run noise is unfalsifiable. Score it 0. Never spend compute on
  it.

Run the self-critique in context by default. When subagents are available, spawn
2-3 parallel read-only critics. Put this rubric verbatim in each prompt, plus
the experiment directory. Average their scores. Drop every hypothesis below 6. A
dropped hypothesis never reaches the paths table and costs no budget. Record the
drop and the score in plan.md.

### 3. EXECUTE

Run each survivor as a path through the **train-llm** skill in full. That skill
covers the budget `can-launch` gate, the dataset validation, the smoke run, the
launch, and the monitor step. Then run the **verify-run** skill. Do not shortcut
either skill's gates from here. Set `tracking.group=gen-<N>` on every path. The
dashboard then clusters one generation together. The **track-experiments** skill
documents the group knob.

Run one path at a time on the local lane. Fan out in parallel only when the
compute lanes differ (e.g. one local + one ssh path). Never batch-launch on a
shared lane.

### 4. SHARE

The verify-run skill already updated the ledger. After each path's verify-run
completes, do these steps:

- Update board.md. The champion is the best path by the plan.md success metric.
  The champion must also have `verify=pass`. A path without a passing verify can
  never be the champion, whatever its number.
- Audit the mechanism: did the observed delta match `expected_delta`? When it
  matches, write one line in `## Verified wins`. A mechanism is refuted when
  there is no delta, or when a real delta comes from a different cause. Write a
  refuted mechanism in `## Dead ends` with the reason. A refuted mechanism often
  seeds a better next-generation hypothesis than the win itself.
- Every `## Dead ends` line names its bottleneck class: model capacity | data
  quality | reward design | environment behavior | evaluator coverage |
  infrastructure. The next lever changes the bottleneck, not only the learning
  rate. Never repeat a failed variant without a documented change of
  method/reward/model/data/evaluator.
- Increment the generation counter in plan.md `## Loop`. Append fresh ideas to
  `## Next levers` in board.md and plan.md.

### 5. STAGNATION LADDER

When the champion does not improve for 2 generations, climb exactly one rung.
Log every climb in board.md `## Stagnation log`:

1. **Rung 1 — tweak the champion.** Vary the champion's own override one
   variable at a time (magnitude, schedule, neighbor values).
2. **Rung 2 — orthogonal angle.** Take hypotheses from the angles that the
   champion's family and the dead-ends list do not touch (see the rung-2 mapping
   in `references/idea-angles.md`).
3. **Rung 3 — structural reframe.** Choose a different method, not a tweak. A
   new method is a new plan.md baseline. Re-run **literature-recipe-research**
   first. Then re-enter at step 1 with the new baseline.

### 6. STOP

Stop ONLY when one of these conditions is true:

- **Budget caps:** `can-launch` or `can-retry` denies the request (exit 1).
  Report which cap stopped the loop.
- **Rung 3 produces nothing for 2 rounds:** two consecutive structural-reframe
  attempts produce no hypothesis that survives critique.
- **Explicit success criterion met:** a champion with `verify=pass` reaches the
  success metric from task.md/plan.md.

The caps set a minimum as well as a maximum. Never quit early while budget and
live hypotheses remain. When you run out of small tweaks, climb a rung. Do not
stop. Only the verify-run pass route writes results.md. When you stop with no
passing path, fire `scripts/bash/notify.sh error "<cause>"`. Never fire
train_done.

## Done conditions

- [ ] board.md exists with a `## Champion` entry, and the ledger records
      `verify=pass` for it.
- [ ] Every launched path has a ledger row and a verify verdict. No row stays
      `running`.
- [ ] The budget.md `## Spent` tally is consistent with the ledger:
      paths_launched matches the rows, and the tally records the retries and the
      GPU-h.
- [ ] plan.md carries `## Loop` with the final generation count and the angle
      coverage. The board holds the dead ends and the stagnation climbs.
- [ ] `notify.sh train_done` fired only when a champion has `verify=pass`.
- [ ] No results.md unless `intern.py verify` exited 0.
