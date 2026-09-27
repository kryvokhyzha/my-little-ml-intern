# Size the hardware

Size the GPU from the param count × the method BEFORE you pick a lane tier.
These are the AdamW estimates. A full fine-tune needs ≈ 16–18 bytes/param +
activations. LoRA needs ≈ 2 bytes/param for the frozen base (bf16), plus a small
adapter/optimizer overhead. QLoRA needs ≈ 0.5–1 byte/param for the base. The
activations scale with batch × sequence length. The OOM ladder turns those
knobs.

| Model size | Method       | Minimum viable              | Comfortable       | hf_jobs flavor guide      |
| ---------- | ------------ | --------------------------- | ----------------- | ------------------------- |
| ≤ 1B       | full or LoRA | 16 GB                       | 24 GB             | t4-small (no flash-attn)  |
| 1–3B       | LoRA/QLoRA   | 24 GB (RTX 3090/4090, A10G) | 2× 24 GB          | a10g-large / a10g-largex2 |
| 1–3B       | full SFT     | 48 GB + grad ckpt           | 80 GB             | a100-large                |
| 7–13B      | QLoRA        | 24 GB                       | 48 GB             | a10g-large                |
| 7–13B      | LoRA         | 48–80 GB                    | 80 GB (A100/H100) | a100-large                |
| 7–13B      | full         | multi-GPU 80 GB             | 4× 80 GB          | a100x4                    |
| 30B+       | QLoRA        | 80 GB                       | 2× 80 GB          | l40sx4 / a100x4           |
| 70B        | any          | multi-GPU only              | 4–8× 80 GB        | a100x8                    |

Notes:

- `a10g-small` and `a10g-large` have the SAME 24 GB GPU. Only the CPU and the
  RAM differ. Do not pay for `large` in the expectation of more VRAM.
- This repo's default `scale_ceiling_params` is 200M. budget.md must already
  justify any bigger model before you size the hardware for it.
- Keep the effective batch (`per_device × grad_accum × gpus`) constant across
  the paths. ~128 is a defensible SFT default. LoRA paths cap it at < 32 (see
  lora.md).
- The QLoRA rows are real lanes. bitsandbytes ships in the `gpu` dependency
  group (`uv sync --group gpu` on the CUDA box). lora.md's QLoRA subsection
  holds the recipe and the `_4bit` model variant (nested `BitsAndBytesConfig`
  node).

## Flash attention

- **Never run `pip install flash-attn`** (source compile). It fails on most
  CUDA/torch combinations. It wastes an hour before it fails. Use the prebuilt
  Hub kernels through the `kernels` library:
  `attn_implementation="kernels-community/flash-attn2"` (or `vllm-flash-attn3`)
  in `from_pretrained`.
- **Never use flash attention on a pre-Ampere GPU.** T4, V100, and the GTX
  10/16-series cannot run flash-attention 2 at all. Use Ampere or newer only
  (A10G, A100, RTX 30/40, L4/L40S, H100). On a pre-Ampere GPU, use the default
  `sdpa`. Pick a non-flash configuration. Do not try flash attention anyway.
- If you are not sure, use `sdpa` (the transformers default). It is correct
  everywhere, and only modestly slower. Never let an attention kernel fail a
  run.

## bf16 vs fp16 vs fp32

- **bf16** — the preferred mixed precision. It runs on Ampere+ only
  (`gpu_probe.sh` → `gpu_name` tells you). It has the fp32 dynamic range, it
  needs no loss scaling, and it gives far fewer NaN surprises. The trainer
  groups ship `bf16: false`. Set `trainer.args.bf16=true` only after the probe
  confirms Ampere+.
- **fp16** — the only mixed-precision option on T4/V100. It needs loss scaling,
  and the trainers handle that. It is NaN-prone at a high LR. If a fp16 run
  produces a NaN, suspect the precision before the data. Never enable bf16 on
  these cards. bf16 silently falls back, or it crashes. The result depends on
  the stack.
- The locked Linux torch (2.13, `cu129` index) has no V100 or Pascal kernels (it
  builds sm_75 and newer). A V100 needs the `cu126` torch build. A T4 (sm_75)
  works.
- **fp32** — always safe, ~2× memory. It is the correct default for a smoke run
  on CPU/MPS. It is also the correct default when you debug a NaN streak. Rerun
  the step that failed in fp32, to separate a precision bug from a data bug.

## MPS (Apple Silicon) caveats

- **Smoke runs and tiny runs only.** Never budget real training GPU-hours on
  MPS.
- **MPS gives no bf16 autocast guarantees.** Keep the smoke runs in fp32
  (`bf16: false`, no fp16). Precision-sensitive ops on MPS have known
  divergences. Run the smoke test in fp32 on MPS. Then run the smoke test again
  on the CUDA target. This pair is the reliable sequence.
- Some ops still miss the MPS kernels. If a smoke run crashes with an
  unimplemented-op error, set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Accept the CPU
  fallback. This is a smoke run, and correctness matters more than speed.
- `gpu_probe.sh` reports `mps=true` with `cuda=false`. Treat the local machine
  as a smoke lane. Pick a remote lane for the long run.
- **GPTNeoX/pythia diverges on MPS even in fp32.** Experiment 001 recorded it:
  the grad norms reached the thousands, and the loss climbed at lr 5e-5. The
  identical CPU run trained cleanly. For a GPTNeoX-family model on Apple
  Silicon, set `+trainer.args.use_cpu=true`. A tiny model trains in seconds on
  CPU anyway.
- bitsandbytes ≥ 0.50 runs 4-bit layers on CPU, so a QLoRA path can run its
  smoke test on a CPU box (verified with 0.50.2). MPS support is unverified. Run
  the real QLoRA run on the CUDA target.
