# Plan — 000-tiny-sft-smoke

## Hypotheses

| id     | change                                   | mechanism                                                       | expected_delta                             | falsification                          |
| ------ | ---------------------------------------- | --------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| path-1 | 40 SFT steps on 256-row toy corpus, lr 5e-4 | gradient descent on a tiny repetitive corpus reduces train loss | final loss < 10.0 (from ~ln(50257) ≈ 10.8) | loss stays ≥ 10.8 or NaN within 40 steps |

One Hydra override per path; this experiment has a single path (plumbing check).

## Success criterion

`intern.py verify --experiment 001` exits 0.
