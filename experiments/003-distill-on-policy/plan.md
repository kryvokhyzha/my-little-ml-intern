# Plan — 003-distill-on-policy

## Hypotheses

### H1: on-policy teacher supervision beats off-policy imitation at equal budget

- mechanism: GKD trains on sequences the STUDENT samples (lmbda=0.5 of steps),
  with the teacher's per-token distribution as the target (generalized JSD,
  beta=0.5) — the student is corrected on its own mistakes instead of only
  imitating teacher text it would never emit, removing the exposure-bias
  mismatch that caps 002.
- expected_delta: final held-out eval_loss ≤ 002-distill-off-policy's final
  eval_loss at the same 300-step budget (strictly lower = confirmed; the
  comparison lands in results.md).
- falsification: 003's held-out eval is clearly worse than 002's at equal
  steps, or generations degrade vs the student baseline — then on-policy at
  this teacher/student gap is not worth the generation cost; the next
  one-variable path is a stronger teacher (SmolLM2-1.7B-Instruct) or lmbda.

## Solution paths

One variable per path — prefer exactly one Hydra override per path.

| path_id | hypothesis | override                       |
| ------- | ---------- | ------------------------------ |
| path-1  | H1         | (baseline config, no override) |
