# Idea angles

This file gives the angle taxonomy for the hypothesis tags. Every hypothesis in
plan.md carries exactly one tag. A healthy seed set covers ≥ 4 distinct angles.
No angle holds more than 30% of the live hypotheses.

| tag | angle            | example one-override hypotheses                                                                     |
| --- | ---------------- | --------------------------------------------------------------------------------------------------- |
| A   | optimization     | learning rate, warmup ratio, schedule, optimizer choice, gradient clipping                          |
| B   | regularization   | weight decay, dropout, label smoothing, NEFTune noise alpha                                         |
| C   | architecture     | layer count, hidden dim, attention variant, norm placement (base-model or lightning module choice)  |
| D   | data             | mixture ratios, curriculum order, packing, dedup/filter threshold, sampling strategy                |
| E   | loss             | objective variant, auxiliary loss weight, DPO beta, distillation temperature                        |
| F   | efficiency       | precision (bf16), batch packing, gradient accumulation, torch.compile — more signal per GPU-hour    |
| G   | cross-domain     | a technique transplanted from an adjacent field (CV/RL/audio/bio) into this task                    |
| H   | scaling          | wider vs deeper, more tokens vs more params, batch size, context length — mind scale_ceiling_params |
| I   | repo-mined       | a concrete trick from a reference implementation or top repo (research.md rows), not from a paper   |
| J   | counterintuitive | test the negation of a community default ("larger batch helps" → try tiny batch)                    |

## Tag rules

- Tag by the **mechanism**. Do not tag by the config key's Hydra group. A
  batch-size change that you justify by throughput is F. The same change that
  you justify by gradient noise is A.
- Every hypothesis stays one Hydra override. An idea that needs two coupled
  overrides is two hypotheses, or a rung-3 reframe. It is not one path.
- Cite the source when one exists: a research.md row, a board.md verified win,
  or a repo (angle I requires one).
- An `expected_delta` smaller than the success metric's observed run-to-run
  noise is unfalsifiable. Score it 0 in the critique, whatever the angle. Never
  spend compute on it.

## Spin an idea — when the seed pool is homogeneous

Apply a transformation to an existing idea. Keep the result only when it is in a
different angle:

1. **Scale it** — 0.1× / 10× the magnitude (dropout 0.1 → 0.5).
2. **Invert it** — test the antithesis (angle J).
3. **Transplant it** — find the analogous technique in an adjacent field (angle
   G).
4. **Simplify it** — remove 80% of the idea. Keep the core mechanism.
5. **Combine** — merge two ideas from different angles that no previous path
   used together.
6. **Shift in time** — apply the idea only during the warmup, only at the end,
   or in alternation.
7. **Negate the assumption** — name the idea's implicit assumption. Then remove
   it.

## Ladder mapping

- **Rung 1 (tweak the champion):** Stay inside the champion's angle. Vary the
  magnitude or the schedule of its override.
- **Rung 2 (orthogonal angle):** List the angles in the champion's lineage. Add
  the angles in board.md `## Dead ends`. Propose only from the angles that this
  list does not name. Angles G and I are always-valid fallbacks. Re-mine
  research.md and the reference repos with the champion's technique as the
  query. Do this before you declare an angle exhausted.
- **Rung 3 (structural reframe):** This rung is not an angle. It is a new method
  and a new plan.md baseline. Run literature-recipe-research first.
