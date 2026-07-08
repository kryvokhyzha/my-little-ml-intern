# Results — 001-pi-mono-sft

**Winner: path-1** — `google/gemma-4-E2B-it` QLoRA (nf4 4-bit, LoRA r=16/α=32 on
the language tower), 200 steps on a single NVIDIA L4, completion-only SFT of the
`kryvokhyzha/pi-mono-sft` coding-agent traces. verify **PASS** (4 passed, 0
failed, 5 skipped) + generation judgment PASS.

## Metrics

| Step | Eval loss |
| ---- | --------- |
| 50   | 0.792     |
| 100  | 0.720     |
| 150  | 0.691     |
| 200  | **0.688** |

Final train loss **0.625**; eval–train gap **0.063** (no overfitting). 24.2M
trainable LoRA params on the frozen 4-bit base. Training wall-clock **2.58 h**
on the L4 (2.58 / 4.0 GPU-h budget). Reference (burtenshaw, L40S): eval 0.55 /
train 0.646 — same ballpark on a smaller converted split (5,046 train / 256
test here vs 6,471 / 256).

Generation judgment: samples emit coherent pi coding-agent output including the
exact tool-call syntax (`call:read{...}`, `response:...`, `call:bash{...}`).

## Reproduce

```bash
# 1. Materialize the dataset (once, needs HF_TOKEN):
uv run python scripts/python/prep-pi-mono-sft.py            # pushes private <HF_USER>/pi-mono-sft

# 2. Train on a >=24GB CUDA GPU (uv sync --group gpu first):
uv run python scripts/python/001-pi-mono-sft.py smoke_test=true       # VERDICT: TRAIN_OK
uv run python scripts/python/intern.py budget --experiment 001-pi-mono-sft can-launch
uv run python scripts/python/001-pi-mono-sft.py
uv run python scripts/python/intern.py verify --experiment 001-pi-mono-sft
```

The exact commands that ran on the GCP L4 (provision → setup → train →
teardown) are in [run.md](run.md). Trained adapter
(`adapter_model.safetensors`, 48 MB) pulled from the run; publish with the
`publish-model` skill when desired.

## Lessons folded back into the repo

- **pi-mono conversion is skip-tolerant**: 5/15,256 traces hit the chat-template
  `startswith` assert on template edges (consecutive assistant turns) — the
  converter now skips them with a warning instead of aborting the whole run.
- **`write_samples` is memory-safe and non-fatal**: post-training generation on
  long held-out prompts OOM'd on the L4 (training memory still resident). Now it
  frees the cache, truncates the probe prompt, guards per-prompt OOM, and — at
  the call site — a sampling failure degrades to a skipped samples.jsonl instead
  of sinking a run whose checkpoint is already saved.
- **Budget is task-keyed**: this run uses the `lora` profile (`budget: lora` →
  4.0 GPU-h cap, 12B param ceiling), sized for single-GPU LoRA/QLoRA. The L4 took
  ~2.5× the reference L40S (~65 min), so 2.58 h landed well under the 4.0 h cap.

## Published

<https://huggingface.co/kryvokhyzha/gemma-4-E2B-it-pi-mono-lora>
