# Task — 003-distill-on-policy

## Restated task

On-policy distillation example (GKD, `trainer=trl_gkd`): distill
`HuggingFaceTB/SmolLM2-360M-Instruct` (teacher, live at training time) into
`HuggingFaceTB/SmolLM2-135M-Instruct` (student) on
`HuggingFaceTB/smoltalk` everyday-conversations. The student **samples its own
completions** and the teacher supervises token-level via generalized JSD — no
train/inference distribution mismatch, unlike 002's imitation of fixed text.
Success: held-out eval improves over the student baseline, and the comparison
against `002-distill-off-policy` (same student, same data, same step budget) is
recorded in results.md.

Teacher and student share the SmolLM2 tokenizer — a GKD requirement.

## Unknowns

- Wall-clock per step: lmbda=0.5 means half the steps generate student samples
  (max_new_tokens=128) — generation dominates; measured at smoke.
- Whether 360M is a strong-enough teacher to show a visible on- vs off-policy
  gap at 300 steps (if not: teacher size is the next one-variable path).

## Run mode

interactive
