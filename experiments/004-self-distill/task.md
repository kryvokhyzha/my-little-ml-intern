# Task — 004-self-distill

## Restated task

Self-distillation example (STaR/RFT — the third distillation mode after 002
off-policy and 003 on-policy): `HuggingFaceTB/SmolLM2-360M-Instruct` samples k
answers per 2-digit-addition TRAIN task, a **deterministic verifier** (parse
last integer, compare to the sum) accepts the correct ones, and the SAME model
is fine-tuned on its own accepted rollouts. No external teacher anywhere.
Success: held-out eval-task **success rate** (measured by the same verifier via
`prep-self-distill.py eval_model_path=<ckpt>`) improves clearly over the
pre-train baseline printed at collection time. The loss is only a proxy — the
verifier metric is the claim.

Exercises the full distill-traces machinery: split-before-collect,
`intern.traces.TraceStore` (rejects kept), deterministic-verification-first
acceptance, `to_prompt_completion` conversion, gates unchanged.

## Unknowns

- Baseline success rate of 360M on 2-digit addition (printed by the collect
  run) — if it is ~0 there is nothing to train on; if ~1.0 there is no headroom
  (then 3-digit addition is the next one-variable task change).
- Acceptance rate at k=4, temperature 0.7 — drives the training-set size.

## Run mode

interactive
