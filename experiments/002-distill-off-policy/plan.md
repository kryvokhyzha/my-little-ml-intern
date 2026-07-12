# Plan — 002-distill-off-policy

## Hypotheses

### H1: off-policy imitation transfers teacher chat behavior to the tiny student

- mechanism: cross-entropy on fixed teacher-written completions (smoltalk is
  distilled data — completions authored by larger models) shifts the student's
  conditional distribution toward the teacher's response style. Supervision
  scope caveat, known and accepted: without masking, messages-format SFT trains
  on the ENTIRE rendered conversation (user turns included), while 003's GKD
  supervises only the final assistant turn. `assistant_only_loss: true` would
  narrow the gap but hard-fails on SmolLM2's chat template (no
  `{% generation %}` markers; TRL cannot auto-patch — verified by smoke), so
  the 002/003 comparison is method + supervision scope, not method alone.
- expected_delta: final_eval_loss ≥ 0.25 lower than the student's step-0 eval
  loss on the held-out test split (eval_on_start baseline).
- falsification: eval improvement < 0.1, or generation samples degenerate
  (repetition, non-chat continuations) — then imitation at this scale/steps is
  not transferring, and more steps or a different subset is a NEW hypothesis.

## Solution paths

One variable per path — prefer exactly one Hydra override per path.

| path_id | hypothesis | override                             |
| ------- | ---------- | ------------------------------------ |
| path-1  | H1         | (baseline config, no override)       |
