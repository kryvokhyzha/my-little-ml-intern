# Plan — 004-self-distill

## Hypotheses

### H1: rejection-sampled self-training raises verifiable task success

- mechanism: sampling k=4 answers at temperature 0.7 finds correct solutions the
  greedy policy misses; cross-entropy on the verifier-accepted subset (STaR/RFT)
  reallocates probability mass from the model's wrong modes to its own correct
  ones — no external teacher, so every gradient target is a behavior the model
  already produced and a verifier certified.
- expected_delta: held-out eval-task success rate (deterministic verifier,
  greedy decoding) improves by ≥ 10 percentage points over the pre-train
  baseline printed by the collect run.
- falsification: post-train success rate within noise of baseline (or worse),
  or the eval loss improves while the verifier rate does not — then the model
  is fitting answer FORMAT, not addition, and the next one-variable path is
  more tasks or higher k, not more steps.

## Solution paths

One variable per path — prefer exactly one Hydra override per path.

| path_id | hypothesis | override                       |
| ------- | ---------- | ------------------------------ |
| path-1  | H1         | (baseline config, no override) |
