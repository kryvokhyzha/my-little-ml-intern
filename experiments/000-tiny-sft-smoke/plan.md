# Plan — 000-tiny-sft-smoke

## Hypotheses

| id     | change                                   | mechanism                                                       | expected_delta                             | falsification                          |
| ------ | ---------------------------------------- | --------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| path-1 | 40 SFT steps on 256-row toy corpus, lr 5e-4 | gradient descent on a tiny repetitive corpus reduces train loss | final loss < 10.0 (from ~ln(50257) ≈ 10.8) | loss stays ≥ 10.8 or NaN within 40 steps |

One Hydra override per path; this experiment has a single path (plumbing check).

## Success criterion

`intern.py verify --experiment 001` exits 0.

## Colab runtime smoke — 2026-10-20

### H7: Execute one smoke step on a Colab T4

- mechanism: The Colab CLI transfers the locked project and executes the existing SFT entrypoint on CUDA.
- expected_delta: Complete one optimizer step and return remote exit 0 with a finite loss.
- falsification: Fail this path if any condition holds:
  - The CUDA tensor probe fails or the allocated GPU is not a T4.
  - The run exits nonzero or omits `VERDICT: TRAIN_OK`.
  - The run records a step count other than 1 or a non-finite loss.
  - Artifact recovery fails or runtime teardown remains unconfirmed.

| path_id | hypothesis | overrides |
| ------- | ---------- | --------- |
| path-7 | H7 | `compute=colab smoke_test=true trainer.args.use_cpu=false` |

This path tests remote execution, not model quality. Smoke mode omits generation
samples and consumes one batch. Retain the full verifier outcome without a waiver.
Preserve the existing CPU baseline and its results.

## Full Colab validation — 2026-10-20

### H8: Verify the complete 10-step recipe on a Colab T4

- mechanism: Ten CUDA steps consume the planned token budget and enable the adapter's generation probe.
- expected_delta: Complete 10 optimizer steps, consume at least 3500 tokens, and produce 3 coherent generation samples.
- expected_delta: Reduce the full verification failure count from the smoke run's 2 failures to 0.
- falsification: Reject the path if training exits nonzero, verification exits nonzero, or a generation lacks coherent English.

| path_id | hypothesis | overrides |
| ------- | ---------- | --------- |
| path-8 | H8 | `compute=colab smoke_test=false trainer.args.use_cpu=false` |

The user authorizes one additional path. Set the path cap to 3 through the
`tiny-sft-colab` profile. Preserve recorded spend and retry counts. Run a fresh
one-step smoke on the allocated T4 before the 10-step run. Keep the 5000-token
plan and all verification thresholds. Preserve the prior CPU and Colab smoke artifacts.
