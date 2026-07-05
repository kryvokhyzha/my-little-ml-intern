# Task — 001-pi-mono-sft

## Restated task

SFT `google/gemma-4-E2B-it` on `badlogicgames/pi-mono` coding-agent traces via
QLoRA (nf4 4-bit, LoRA r=16) to teach the model the pi coding-agent interaction
format (chat + tool calls) and basic task behavior. This is a reproduction of
the reference run, not a sweep.

The dataset is the **materialized** `${HF_USER}/pi-mono-sft` produced by
`scripts/python/prep-pi-mono-sft.py` (raw session JSONL → prompt/completion rows
→ train/test splits pushed to the Hub). **Run that prep script FIRST** — this
experiment composes the `data: pi_mono_sft` group, i.e.
`data.path=${HF_USER}/pi-mono-sft` with splits `train` (fit) and `test`
(held-out eval). Prep config: `configs/prep-pi-mono-sft.yaml`.

Success criterion: `intern.py verify --experiment 001` exits 0 and generation
samples follow the pi tool-call format; held-out eval loss near the reference
`~0.55`.

## Unknowns

- `trainer.planned_tokens` is an estimate (~3,000,000). Tune it from the
  `num_input_tokens_seen` series after run 1; the reference kept 6,471 train rows
  at `max_length=4096` for 200 steps × effective batch 16.
- Exact converted example count depends on the prep run (session availability,
  overlength filtering at 4096); the reference kept 6,727 examples from 627 raw
  files.

## Compute

QLoRA is CUDA-only: it loads the base model in 4-bit through **bitsandbytes**,
which has no MPS/CPU path — so this experiment **cannot smoke on Mac**. Run
`uv sync --group gpu` on a `>=24GB` GPU box (an ssh / vast / modal / hf_jobs
lane; the reference used an L40S 48GB, ~65 min) and smoke THERE:

```bash
uv run python scripts/python/001-pi-mono-sft.py smoke_test=true   # on the GPU box
```

Optional **local plumbing-smoke** — validate the data + trl_sft path without a
GPU by swapping the model group to a tiny non-quantized proxy (no bitsandbytes):

```bash
uv run python scripts/python/001-pi-mono-sft.py \
  model=smollm2_135m \
  'trainer.peft.target_modules=all-linear' \
  smoke_test=true
```

The data group stays `pi_mono_sft`, so this still requires the prep'd
`${HF_USER}/pi-mono-sft` dataset. It exercises dataset loading,
prompt/completion rendering, and the trl_sft wiring; it does NOT validate the
QLoRA/gemma-4 path (do that on the GPU box).

## Run mode

interactive
