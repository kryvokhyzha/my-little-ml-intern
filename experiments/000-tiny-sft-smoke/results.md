# Results — 000-tiny-sft-smoke

**Latest verified path: path-8**, the full 10-step recipe on one Colab Tesla T4.
The remote and recovered local verify gates exit 0. Direct review of all three
completions gives `generation_quality=PASS` for the synthetic text corpus.
The ledger records `status=passed` and `verify=pass`.

| Measurement | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M`, 134515008 parameters |
| Recipe | fp32, learning rate 5e-5, batch size 8, 10 optimizer steps |
| Final-step loss | 1.7793203592300415 |
| Run-average train loss | 2.3757517337799072 |
| Input tokens | 5528; plan 5000, gate minimum 3500 |
| Trainer runtime | 4.8337 seconds; excludes setup and generation |
| Verification | 4 passed, 0 failed, 7 skipped |
| Generation samples | 3; mechanical gate and direct language review pass |
| Allocation upper bound | 0.146404 GPU-hours, or 8.78424 GPU-minutes |

The four passing checks cover loss plausibility, token consumption, stderr,
and generation sanity. The skipped checks require held-out evaluation, a
parameter target, or metrics from other training lanes. The experiment defines
none of these inputs. No waiver changes the outcome.

## Generation review and limits

The completions contain recognizable simple English consistent with the toy
corpus. The weather prompt produces a complete sentence about a dog and the
beach. The longer samples repeat scene motifs and end at the 100-token cap.
These samples support execution and output sanity. They establish no held-out
quality result or improvement over the CPU baseline.

The recipe sets `save_strategy=no`, so this run produces no model checkpoint.
The recovered artifacts contain the metrics, data, logs, samples, journal,
budget, ledger, and verification reports.

## Path comparison

| Path | Outcome | Evidence |
| --- | --- | --- |
| path-6 | Historical CPU pass | The archived baseline contains the original results and report. |
| path-7 | One-step CUDA smoke; full verify fails | 544 tokens and no generation samples cause two failures. |
| path-8 | Full T4 pass | 5528 tokens and three coherent samples reduce the full failure count to zero. |

The historical CPU report lacks an `ENV:` stamp. Its loss is not an equivalent
environment comparison. Paths 1–5 and their failure causes remain in the ledger
and postmortems.

## Evidence and teardown

- [Verification report](verify.md), including the environment and language judgment.
- [Exact commands and recovery record](run.md#full-colab-t4-run--2026-10-20).
- [Metrics](metrics.jsonl) and [generation samples](logs/samples.jsonl).
- [Recovered full-run artifacts](logs/colab-full-20261020-path8/artifacts/experiments/000-tiny-sft-smoke/).
- [Original CPU results](logs/cpu-baseline-before-path-8/results.md).
- [Path ledger](ledger.md) and [budget](budget.md).

The raw artifact archive has SHA256
`3b3a94078276a7000d8b67bd84d41605fcbe7ebccfa3fd68b6f5a0e2485bf6b6`.
The owned session `intern-000-full-fc75ffb8` terminates at
2026-10-20 15:34:59 UTC. Colab confirms zero active assignments and a zero usage
rate after teardown.

The user-authorized budget profile raises `max_paths` from 2 to 3.
The final tally is 3 paths, 5 historical retries, and 0.364045 GPU-hours.
The 2 GPU-hour cap and all prior spend remain intact.

## Lessons folded back into the repo

- `trainer.dtype: float32` default in TRL configs (fp16-checkpoint divergence).
- GPTNeoX/pythia diverges on MPS even in fp32 → documented in train-llm hardware reference.
- Sanity samples use seeded sampling, not greedy (greedy loops on healthy models).
- `samples.jsonl` replaced blank-line-delimited samples.txt (paragraph breaks collided
  with the delimiter).
- Generation probe: raw-text data (this experiment) falls back to the built-in
  `SAMPLE_PROMPTS`; prompt/completion data samples held-out eval prompts instead
  (`resolve_sample_prompts`), and `data.sample_prompts` is an optional override.
