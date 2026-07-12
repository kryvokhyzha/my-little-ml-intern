# Results — 003-distill-on-policy

**Run: path-1 passed verification; hypothesis FALSIFIED by its own criterion.**
GKD (`trl_gkd`, lmbda 0.5 / beta 0.5) distilling the live
`SmolLM2-360M-Instruct` teacher into the 135M student, 300 steps on a single
L4. verify **PASS** (4 passed, 0 failed; loss_plausibility auto-skipped — JSD
is not vocab CE) + generation judgment PASS. The plan's expected_delta was
"final held-out eval ≤ 002's at equal steps"; on the shared CE metric 003
landed **clearly worse** (1.4857 vs 0.8151) → falsification condition fired.

## Metrics

GKD's own objective (generalized JSD to the teacher, final assistant turn)
did improve:

| Step        | Eval JSD   |
| ----------- | ---------- |
| 50          | 0.0724     |
| 150         | 0.0689     |
| 300 (final) | **0.0672** |

Final train JSD 0.0678; eval–train gap 0.0006. 300 steps / 2.11 epochs in
**29.2 min** (~5.7 s/step, generation-bound: half the steps sample up to 128
tokens from the student; 002's plain SFT ran ~1 s/step). 0.55 GPU-h of the
2.0 budget.

## The shared-metric table (what falsified it)

CE eval loss on the smoltalk everyday test split (`trash/ce_eval_local.py`
pattern — same pipeline for all four):

| Model                    | CE eval loss |
| ------------------------ | ------------ |
| base 135M student        | 1.5669       |
| 360M teacher             | 1.2952       |
| 003 GKD student          | 1.4857       |
| **002 SFT student**      | **0.8151**   |

## Interpretation (why this is the expected physics, and what it teaches)

Each method optimized its own target: SFT halved dataset CE because dataset CE
IS its loss; GKD moved the student **toward the teacher's distribution** — from
1.5669 to 1.4857, in the direction of the teacher's own 1.2952, which is a
floor-ish anchor for distribution matching (a student matching the teacher
cannot model the dataset much better than the teacher does). The comparison
metric in plan.md therefore structurally favored 002: it measures corpus
imitation, which is not what GKD maximizes. Falsified-as-written, with a
metric-design lesson attached.

Caveats already on record: the supervision-scope confound (002 trains on whole
conversations, GKD on the final assistant turn — plan.md), and this
teacher/student gap (360M→135M) is small; the paper's wins show up against
exposure bias on generation tasks, which dataset CE cannot see.

## Next one-variable paths (per plan.md, not launched)

- Stronger teacher: `SmolLM2-1.7B-Instruct` (same tokenizer, one config line).
- A generation-quality claim metric (e.g. teacher-judged or verifier-based)
  instead of dataset CE, so the on-policy mechanism is measured on-policy.

## Reproduce

```bash
uv run python scripts/python/003-distill-on-policy.py smoke_test=true   # VERDICT: TRAIN_OK
uv run python scripts/python/intern.py budget --experiment 003 can-launch
uv run python scripts/python/003-distill-on-policy.py
uv run python scripts/python/intern.py verify --experiment 003
```

Exact GCP L4 commands with the 2026-07-12 actuals: [run.md](run.md). Off-policy
side of the pair: [002 results.md](../002-distill-off-policy/results.md).
