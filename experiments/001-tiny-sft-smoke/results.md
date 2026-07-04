# Results — 001-tiny-sft-smoke

**Winner: path-6** — `HuggingFaceTB/SmolLM2-135M`, CPU, 10 steps, lr 5e-5, fp32.
Final train loss 1.779; verify **PASS** (5 passed, 0 failed, 3 skipped) + generation
judgment PASS. Reproduce with:

```bash
uv run python scripts/python/001-tiny-sft-smoke.py smoke_test=true   # gate: VERDICT: TRAIN_OK
uv run python scripts/python/001-tiny-sft-smoke.py                   # ~40 s on CPU
uv run python scripts/python/intern.py verify --experiment 001       # exit 0
```

## Path comparison

| path   | change                             | outcome | what it proved                                                       |
| ------ | ---------------------------------- | ------- | -------------------------------------------------------------------- |
| path-1 | tiny-gpt2 (random init, 100k)      | fail    | generation check catches word salad at plausible loss                |
| path-2 | pythia-14m, lr 5e-4                | fail    | red-flag check catches loss=0.0 from NaN divergence                  |
| path-3 | lr 5e-5                            | fail    | falsified the lr hypothesis → led to the real bug                    |
| path-4 | adapter fp32 fix                   | fail    | transformers v5 loads checkpoint dtype (fp16) — real adapter bug fixed; also exposed GPTNeoX-on-MPS numerics |
| path-5 | CPU, 10 steps                      | fail    | 14M-model degeneracy attractor; samples.jsonl format bug found+fixed |
| path-6 | SmolLM2-135M                       | **pass** | full pipeline green end-to-end                                       |

## Lessons folded back into the repo

- `trainer.dtype: float32` default in TRL configs (fp16-checkpoint divergence).
- GPTNeoX/pythia diverges on MPS even in fp32 → documented in train-llm hardware reference.
- Sanity samples use seeded sampling, not greedy (greedy loops on healthy models).
- `samples.jsonl` replaced blank-line-delimited samples.txt (paragraph breaks collided
  with the delimiter).
- Sample prompts are configurable per experiment (`trainer.sample_prompts`) and must match
  the training distribution; pangram-bait prompts removed from defaults.
