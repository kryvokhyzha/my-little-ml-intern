# Idea angles

Angle taxonomy for hypothesis tags. Every hypothesis in plan.md carries exactly
one tag. A healthy seed set covers ≥ 4 distinct angles; no angle holds more than
30% of live hypotheses.

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

## Tagging rules

- Tag by the **mechanism**, not the config key's Hydra group: a batch-size
  change justified by throughput is F; the same change justified by gradient
  noise is A.
- Every hypothesis stays one Hydra override. An idea needing two coupled
  overrides is two hypotheses or a rung-3 reframe — not one path.
- Cite the source when one exists: a research.md row, a board.md verified win,
  or a repo (angle I requires one).
- An `expected_delta` smaller than the success metric's observed run-to-run
  noise is unfalsifiable — score it 0 in critique, whatever the angle; never
  spend compute on it.

## Idea spinning — when the seed pool is homogeneous

Apply a transformation to an existing idea and keep the result only if it lands
in a different angle:

1. **Scale it** — 0.1× / 10× the magnitude (dropout 0.1 → 0.5).
2. **Invert it** — test the antithesis (angle J).
3. **Transplant it** — find the analogous technique in an adjacent field (angle
   G).
4. **Simplify it** — strip 80% of the idea, keep the core mechanism.
5. **Combine** — merge two ideas from different angles never tried together.
6. **Shift in time** — apply only during warmup, only at the end, alternating.
7. **Negate the assumption** — name the idea's implicit assumption, remove it.

## Ladder mapping

- **Rung 1 (tweak champion):** stay inside the champion's angle; vary its
  override's magnitude or schedule.
- **Rung 2 (orthogonal angle):** list the angles present in the champion's
  lineage plus board.md `## Dead ends`; propose only from angles absent from
  that list. Angles G and I are always-valid fallbacks — re-mine research.md and
  reference repos with the champion's technique as the query before declaring an
  angle exhausted.
- **Rung 3 (structural reframe):** not an angle — a new method and a new plan.md
  baseline. Run literature-recipe-research before proposing.
