# Path-7 — Colab T4 smoke verification

## Symptom

The full verify gate exits 1. The smoke command exits 0 and prints `TRAIN_OK`.
The run executes one CUDA step with loss 3.6047396659851074.

The failed checks are:

- `data_consumption`: the run consumes 544 tokens; the gate requires 3500.
- `generation_sanity`: the smoke run produces no generation samples.

Read the [recovered report](../logs/colab-20261020/artifacts/experiments/000-tiny-sft-smoke/verify.md)
and the [run transcript](../run.md#colab-t4-smoke--2026-10-20).

## Root-cause hypothesis

The user requests a one-step runtime smoke test. The experiment retains its
5000-token plan for a complete training run. Smoke mode limits training to one
step and skips generation samples. The full verification criteria therefore
remain unmet. The CUDA probe, finite gradients, and empty stderr log support
the narrower execution evidence.

## Fix

Retain the failed report without a waiver. Keep path-7 marked failed in the ledger.
Preserve the CPU baseline separately. Claim no model-quality result from this smoke run.
For a future full verification, obtain an appropriate budget before a full training path.
Keep the token-consumption and generation checks unchanged.

The artifacts are recovered. The T4 is stopped. The final Colab API responses
report zero active assignments and a zero usage rate.
