# Hardware sizing

Size the GPU from param count × method BEFORE picking a lane tier. Rules of
thumb (AdamW): full fine-tune ≈ 16–18 bytes/param + activations; LoRA ≈ 2
bytes/param frozen base (bf16) + small adapter/optimizer overhead; QLoRA ≈ 0.5–1
byte/param base. Activations scale with batch × sequence length — the knobs the
OOM ladder turns.

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

- `a10g-small` and `a10g-large` have the SAME 24 GB GPU — the difference is
  CPU/RAM only. Don't pay for `large` expecting more VRAM.
- This repo's default `scale_ceiling_params` is 200M — anything bigger must
  already be justified in budget.md before sizing hardware for it.
- Keep effective batch (`per_device × grad_accum × gpus`) constant across paths;
  ~128 is a sane SFT default (LoRA paths cap it at < 32 — see lora.md).
- The QLoRA rows are real lanes: bitsandbytes ships in the `gpu` dependency
  group (`uv sync --group gpu` on the CUDA box) — recipe and the
  `trainer.quantization` block live in lora.md's QLoRA subsection.

## Flash attention

- **Never `pip install flash-attn`** (source compile) — it fails on most
  CUDA/torch combos and wastes an hour before failing. Use prebuilt Hub kernels
  via the `kernels` library:
  `attn_implementation="kernels-community/flash-attn2"` (or `vllm-flash-attn3`)
  in `from_pretrained`.
- **Never on pre-Ampere GPUs** — T4, V100, GTX 10/16-series cannot run
  flash-attention 2 at all. Ampere or newer only (A10G, A100, RTX 30/40,
  L4/L40S, H100). On pre-Ampere, use the default `sdpa` and pick a non-flash
  configuration rather than "trying anyway".
- When in doubt, `sdpa` (the transformers default) is correct everywhere and
  only modestly slower. It is never worth failing a run over an attention
  kernel.

## bf16 vs fp16 vs fp32

- **bf16** — preferred mixed precision. Ampere+ only (`gpu_probe.sh` →
  `gpu_name` tells you). fp32 dynamic range, no loss scaling, far fewer NaN
  surprises. The trainer groups ship `bf16: false`; set `trainer.args.bf16=true`
  only after the probe confirms Ampere+.
- **fp16** — only option for mixed precision on T4/V100. Needs loss scaling (the
  trainers handle it) but is NaN-prone at high LR; if a fp16 run NaNs, suspect
  precision before data. Never enable bf16 on these cards — it silently falls
  back or crashes depending on the stack.
- **fp32** — always safe, ~2× memory. Correct default for smoke runs on CPU/MPS
  and for debugging NaN streaks (rerun the failing step in fp32 to separate
  precision bugs from data bugs).

## MPS (Apple Silicon) caveats

- **Smoke and tiny runs only** — never budget real training GPU-hours on MPS.
- **No bf16 autocast guarantees** — keep smoke runs fp32 (`bf16: false`, no
  fp16). Precision-sensitive ops on MPS have known divergences; a smoke pass in
  fp32 on MPS + a re-smoke on the CUDA target is the reliable combo.
- Some ops still miss MPS kernels; if a smoke crashes with an unimplemented-op
  error, set `PYTORCH_ENABLE_MPS_FALLBACK=1` and accept the CPU fallback — it's
  a smoke run, correctness beats speed.
- `gpu_probe.sh` reports `mps=true` with `cuda=false` — treat that as "local is
  a smoke lane" and pick a remote lane for the long run.
- **GPTNeoX/pythia diverges on MPS even in fp32** (observed in experiment 001:
  grad norms in the thousands, loss climbing at lr 5e-5, while the identical CPU
  run trains cleanly). For GPTNeoX-family models on Apple Silicon set
  `+trainer.args.use_cpu=true`; tiny models train in seconds on CPU anyway.
- bitsandbytes quantization (QLoRA) is CUDA-only — QLoRA paths cannot even smoke
  on MPS; smoke them on the CUDA target directly.
