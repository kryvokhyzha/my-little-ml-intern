# Results — 004-self-distill

**Run: path-1 passed verification (one waived check); hypothesis PARTIALLY
confirmed.** STaR/RFT self-distillation: `SmolLM2-360M-Instruct` sampled k=4
answers per train task, a deterministic verifier accepted the correct ones,
and the same model fine-tuned 100 steps on its own 726 accepted rollouts.
Held-out success rate (60 tasks, greedy, same verifier): **0.867 → 0.950**
(+8.3pp). The plan expected ≥ 10pp; falsification ("within noise of baseline
or worse") did NOT trigger — +8.3pp is 5/60 tasks, ≈ 2.1σ under a binomial
noise model.

## The loop, with actuals

| Stage        | Actual                                                        |
| ------------ | ------------------------------------------------------------- |
| Collect      | 240 train tasks × k=4 @ T=0.7 → 960 traces, 726 accepted (75.6%) |
| Baseline     | eval success **0.867** (52/60)                                 |
| Train        | 100 steps / 2.18 epochs, 68 s, completion-only train loss 0.0591 |
| Post-train   | eval success **0.950** (57/60) on checkpoint-100               |

Total 0.25 GPU-h of the 6.0 budget (`budget: sft` profile).

## The waived check (eval_train_gap = 2.02)

Structural, mostly: gold eval completions are bare answers (`"139"`) while
training completions carry the model's own rollout style (`"48 + 91 = 139"`).
The **untrained base model already scores 1.483 CE** on the same gold rows, so
1.42 of the 2.02 gap pre-exists training; the remaining ~0.6 is the model
committing to its own answer format — while the format-agnostic verifier
metric improved. That is the opposite quadrant from the plan's falsification
("eval CE improves while verifier rate does not = fitting format"). Full
reasoning in [verify.md](verify.md).

## Why the expected_delta was missed (ceiling, not mechanism)

Baseline 0.867 left only 13.3pp of headroom, so the ≥ 10pp bar required
reaching 0.967 — the mechanism (reallocating mass from wrong modes to the
model's own verified ones) delivered a 62.5% error reduction (8 wrong → 3
wrong). Next one-variable paths per plan.md: harder task pool (3-digit
addition drops the baseline and reopens headroom) or higher k. More steps is
explicitly NOT the next path.

## Reproduce

```bash
uv run python scripts/python/prep-self-distill.py        # collect + verify + dataset + baseline
uv run python scripts/python/004-self-distill.py smoke_test=true
uv run python scripts/python/intern.py budget --experiment 004 can-launch
uv run python scripts/python/004-self-distill.py
uv run python scripts/python/intern.py verify --experiment 004
uv run python scripts/python/prep-self-distill.py eval_model_path=experiments/004-self-distill/ckpts/checkpoint-100
```

Exact GCP L4 commands with the 2026-07-12 actuals: [run.md](run.md). Data
provenance: [data.md](data.md).
