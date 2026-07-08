# Run — 000-tiny-sft-smoke

Exact commands executed. `local` lane — no Provision/Teardown.

## Lane

compute=local · instance: CPU (GPTNeoX/pythia diverges on MPS even in fp32; the
winning path pins CPU + fp32 — see the ledger retry chain).

## Setup

```bash
uv sync
```

## Train

```bash
uv run python scripts/python/000-tiny-sft-smoke.py smoke_test=true   # VERDICT: TRAIN_OK
uv run python scripts/python/000-tiny-sft-smoke.py                   # winner: SmolLM2-135M, CPU, 10 steps, lr 5e-5, fp32
uv run python scripts/python/intern.py verify --experiment 000-tiny-sft-smoke
```

## Benchmark

Inline only: 3 seeded generation samples (`logs/samples.jsonl`) + the smoke-scale
train loss, checked by the verify gate. No held-out eval (`data.eval: null` for
the tiny synthetic corpus).
