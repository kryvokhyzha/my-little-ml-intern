# Results — 002-distill-off-policy

**Winner: path-1** — plain SFT imitation of teacher-written completions
(`HuggingFaceTB/smoltalk` everyday-conversations) into
`SmolLM2-135M-Instruct`, 300 steps on a single L4. verify exit 1 → **waived**
(loss_plausibility red flag, documented mechanism below) + generation judgment
PASS. Hypothesis **confirmed**: expected_delta ≥ 0.25 eval-loss improvement,
observed **0.75**.

## Metrics

| Anchor                          | CE eval loss (smoltalk test) |
| ------------------------------- | ---------------------------- |
| step-0 baseline (base student)  | 1.5669                       |
| step 50                         | 0.8506                       |
| step 150                        | 0.8202                       |
| step 300 (final)                | **0.8151**                   |

Final train loss 0.6681; eval–train gap 0.147. 300 steps / 2.11 epochs in
**4.9 min** on the L4 (1.2M tokens seen, 82% of planned). Total path cost
0.15 GPU-h of the 2.0 budget.

## The waived red flag (read before copying this setup)

`loss_plausibility` FAILs when train loss < 1.0 on an LM task. Here the
mechanism is benign and known: smoltalk was in SmolLM2-135M-Instruct's **own
SFT mix** (re-training on in-distribution data is exactly what the imitation
baseline is), 2.1 epochs over a 2,260-row corpus, and the unmasked loss also
covers templated user turns. The eval–train gap is small and generations are
coherent, so no broken-loss signature. Full reasoning in
[verify.md](verify.md).

## 002 vs 003 (the controlled pair)

Shared metric — CE on the smoltalk test split, same student, same data, same
300 steps:

| Model                             | CE eval loss |
| --------------------------------- | ------------ |
| base 135M student                 | 1.5669       |
| **002 SFT student**               | **0.8151**   |
| 003 GKD student                   | 1.4857       |
| 360M teacher (reference)          | 1.2952       |

Each method won its own objective and barely moved the other's: SFT halved
dataset CE (it trains on exactly that); GKD left dataset CE almost untouched
(1.4857) because it optimizes JSD toward the live teacher's distribution, not
the dataset text — its own metric (eval JSD to teacher) improved instead. See
[003 results.md](../003-distill-on-policy/results.md) for that side and the
caveats (supervision-scope confound documented in plan.md).

## Reproduce

```bash
uv run python scripts/python/002-distill-off-policy.py smoke_test=true   # VERDICT: TRAIN_OK
uv run python scripts/python/intern.py budget --experiment 002 can-launch
uv run python scripts/python/002-distill-off-policy.py
uv run python scripts/python/intern.py verify --experiment 002
```

Exact GCP L4 commands (provision → train → teardown) with the 2026-07-12
actuals: [run.md](run.md).
