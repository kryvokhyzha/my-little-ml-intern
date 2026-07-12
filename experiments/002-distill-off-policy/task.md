# Task — 002-distill-off-policy

## Restated task

Off-policy distillation example: fine-tune the tiny student
`HuggingFaceTB/SmolLM2-135M-Instruct` with plain SFT on
`HuggingFaceTB/smoltalk` (everyday-conversations) — completions written by
larger teacher models, so the student **imitates fixed teacher outputs** it
never sampled itself. Success: held-out eval loss improves clearly over the
student's pre-training baseline and samples stay coherent chat.

Controlled pair with `003-distill-on-policy` (same student, same dataset; only
the distillation method differs) — together they are the repo's worked
off-policy vs on-policy comparison.

## Unknowns

- Student baseline eval loss on smoltalk-everyday test (measured at run start —
  the eval_on_start point of the first path).
- Whether 300 steps saturates a 135M student on a 2.3k-row subset (watch the
  eval curve; multi-epoch is a legitimate H2 if it is still falling).

## Run mode

interactive
