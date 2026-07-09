# LoRA / QLoRA — the no-regret recipe

Distilled from "LoRA Without Regret" (Schulman et al., Thinking Machines, 2025:
https://thinkingmachines.ai/blog/lora/) and its TRL reproduction
(https://huggingface.co/docs/trl/lora_without_regret). The SFT column ships as a
preset: `configs/trainer/trl_sft_lora.yaml`, with `trl_sft_qlora.yaml` layering
the paged optimizer + bf16 on top for 4-bit bases (compose a `_4bit` model).

The adapter is a `_target_: peft.LoraConfig` node under `trainer.peft` — set
knobs as plain keys (`r`, `lora_alpha`, `target_modules`, …); the loader
instantiates it (same path as the model group). Override per experiment under
`_self_`; a null `trainer.peft` = full fine-tune.

## Recipe

| Knob             | SFT (post-training scale)                          | RL (policy gradient)        |
| ---------------- | -------------------------------------------------- | --------------------------- |
| `target_modules` | `all-linear` — always                              | `all-linear` — always       |
| `r`              | 256                                                | 1–32 (rank 1 already works) |
| `lora_alpha`     | 16                                                 | 32                          |
| `lora_dropout`   | 0.0                                                | 0.0                         |
| learning rate    | 10× the full-FT optimum (2e-4 vs the 2e-5 default) | 10× full-FT (1e-5 vs 1e-6)  |
| effective batch  | < 32                                               | < 32                        |

Why each row:

- **`all-linear`, always.** Attention-only LoRA significantly underperforms
  MLP-inclusive targeting, and higher rank does NOT compensate (attn-only r=256
  lost to MLP-only r=128). LoRA only tracks full-FT dynamics when applied to the
  layers holding most of the parameters — MLPs, and MoE experts where they
  exist.
- **Custom `target_modules` (anything but a preset):** resolve the spec against
  `model.named_modules()` before training — fail hard on zero matches, and log
  the match count plus a sample of matched module types. Suffix-only matching
  silently targeted unsupported wrapper modules on Gemma-4's multi-tower
  architecture (verified failure); explicit `model.language_model.layers.*`
  projection matching fixed it. `google/gemma-4-E2B-it` is a multi-tower
  `Gemma4ForConditionalGeneration` — set `target_modules` to the language-tower
  regex (or omit it to use PEFT's gemma4 default `q_proj`/`v_proj`);
  `all-linear` attaches adapters to the vision/audio towers (experiment
  001-pi-mono-sft).
- **Keep `lora_alpha` fixed, tune `r`.** The 1/r scaling in W' = W +
  (alpha/r)·B·A makes the optimal LR approximately rank-independent — never
  scale alpha with rank.
- **LR = 10× full-FT.** Fitted multiplier ~9.8 across 14 Llama/Qwen models, SFT
  and RL alike; ~15× only for very short (~100-step) runs.
- **Effective batch < 32.** LoRA tolerates large batches worse than full FT; the
  penalty grows with batch size and is a property of the B·A parametrization —
  rank does not fix it. This OVERRIDES hardware.md's "~128 effective batch" SFT
  guidance on LoRA paths (the trl_sft default 2×8=16 is already compliant).
- RL sizing intuition: policy-gradient learning absorbs ~1 bit/episode, so even
  rank 1 carries a ~10× capacity margin on typical RL datasets — spend rank on
  SFT, not RL.

## When full FT instead

Capacity rule: a LoRA adapter stores ~2 bits/param; SFT data carries ~1
bit/token (RL: ~1 bit/episode). LoRA "falls off the minimum-loss learning curve"
when the dataset outgrows the adapter.

| Situation                                          | Verdict                                                    |
| -------------------------------------------------- | ---------------------------------------------------------- |
| dataset tokens × 1 bit ≳ adapter params × 2 bits   | full FT (or raise r until capacity clears the dataset)     |
| large-batch regime required (effective batch ≥ 32) | full FT — the batch penalty is inherent, rank doesn't help |
| post-training-scale SFT within adapter capacity    | LoRA at r=256 matches full FT at ~2/3 the FLOPs            |
| policy-gradient RL                                 | LoRA, r=1–32                                               |

## QLoRA (the `_4bit` model variant)

On top of the LoRA preset, compose `model: <name>_4bit` (e.g.
`model=gemma_4_e2b_it_4bit`) — the variant's `model.main` nests a
`quantization_config:` node with `_target_: transformers.BitsAndBytesConfig`
(see `configs/model/gemma_4_e2b_it_4bit.yaml`). Quantization is model identity,
not a trainer key; a new quantized model is a new `<name>_4bit.yaml` file.

- Requires bitsandbytes: run `uv sync --group gpu` on the CUDA box (remote
  lanes: compute-lanes.md ssh step 2). Missing it fails at model load with an
  error naming that exact command.
- CUDA-only — QLoRA paths cannot even smoke on MPS/CPU; smoke directly on the
  CUDA target (hardware.md).
- `verify` auto-SKIPs `param_drift` on quantized runs: 4-bit storage packs two
  elements per byte, so `numel()` undercounts ~2× and the comparison is
  meaningless.

## Verify / tracking

- `param_count` meta = the BASE model, pre-peft — logged before the trainer
  applies the adapter, so `param_drift` stays honest on plain LoRA runs.
- `trainable_param_count` meta (requires_grad numel after trainer construction)
  is what marks a LoRA run — expect orders of magnitude below `param_count`.
- `quantized` meta is true whenever `model.main` carries a
  `quantization_config`.
