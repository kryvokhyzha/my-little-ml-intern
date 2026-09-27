# LoRA / QLoRA — the no-regret recipe

This recipe comes from "LoRA Without Regret" (Schulman et al., Thinking
Machines, 2025: https://thinkingmachines.ai/blog/lora/) and from its TRL
reproduction (https://huggingface.co/docs/trl/lora_without_regret). The SFT
column ships as a preset: `configs/trainer/trl_sft_lora.yaml`. For a 4-bit base,
`trl_sft_qlora.yaml` adds the paged optimizer and bf16 on top of that preset.
Compose a `_4bit` model with it.

The adapter is a `_target_: peft.LoraConfig` node under `trainer.peft`. Set the
knobs as plain keys (`r`, `lora_alpha`, `target_modules`, …). The loader
instantiates the node, exactly as it instantiates the model group. Override the
knobs per experiment under `_self_`. A null `trainer.peft` means a full
fine-tune.

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

- **`all-linear`, always.** Attention-only LoRA performs much worse than a
  target set that includes the MLPs. A higher rank does NOT compensate for it:
  attn-only r=256 lost to MLP-only r=128. LoRA tracks the full-FT dynamics only
  when you apply it to the layers that hold most of the parameters. Those layers
  are the MLPs, and the MoE experts where they exist.
- **Custom `target_modules` (anything but a preset):** resolve the spec against
  `model.named_modules()` before the training run. Fail hard on zero matches.
  Log the match count. Log a sample of the matched module types. A suffix-only
  match silently hit unsupported wrapper modules on the multi-tower architecture
  of Gemma-4 (verified failure). An explicit `model.language_model.layers.*`
  projection match fixed that failure. `google/gemma-4-E2B-it` is a multi-tower
  `Gemma4ForConditionalGeneration`. Set `target_modules` to the language-tower
  regex. As an alternative, omit `target_modules` to use PEFT's gemma4 default
  `q_proj`/`v_proj`. `all-linear` attaches adapters to the vision and audio
  towers (experiment 001-pi-mono-sft).
- **Keep `lora_alpha` fixed, and tune `r`.** The 1/r scale factor in W' = W +
  (alpha/r)·B·A makes the optimal LR approximately rank-independent. Never scale
  alpha with the rank.
- **LR = 10× full-FT.** The fitted multiplier is ~9.8 across 14 Llama/Qwen
  models, for SFT and RL alike. The multiplier is ~15× only for a very short
  (~100-step) run.
- **Effective batch < 32.** LoRA tolerates a large batch worse than full FT. The
  penalty grows with the batch size. The penalty is a property of the B·A
  parametrization, and the rank does not fix it. This rule OVERRIDES the "~128
  effective batch" SFT guidance of hardware.md on a LoRA path. The trl_sft
  default 2×8=16 already meets this rule.
- RL rank size: policy-gradient learning absorbs ~1 bit/episode. Even rank 1
  therefore carries a ~10× capacity margin on a typical RL dataset. Spend the
  rank on SFT, not on RL.

## When full FT instead

Capacity rule: a LoRA adapter stores ~2 bits/param. SFT data carries ~1
bit/token (RL: ~1 bit/episode). LoRA "falls off the minimum-loss learning curve"
when the dataset exceeds the capacity of the adapter.

| Situation                                          | Verdict                                                    |
| -------------------------------------------------- | ---------------------------------------------------------- |
| dataset tokens × 1 bit ≳ adapter params × 2 bits   | full FT (or raise r until capacity clears the dataset)     |
| large-batch regime required (effective batch ≥ 32) | full FT — the batch penalty is inherent, rank doesn't help |
| post-training-scale SFT within adapter capacity    | LoRA at r=256 matches full FT at ~2/3 the FLOPs            |
| policy-gradient RL                                 | LoRA, r=1–32                                               |

## QLoRA (the `_4bit` model variant)

On top of the LoRA preset, compose `model: <name>_4bit` (for example
`model=gemma_4_e2b_it_4bit`). The `model.main` node of that variant nests a
`quantization_config:` node with `_target_: transformers.BitsAndBytesConfig`
(see `configs/model/gemma_4_e2b_it_4bit.yaml`). Quantization is model identity,
not a trainer key. A new quantized model is a new `<name>_4bit.yaml` file.

- QLoRA requires bitsandbytes. Run `uv sync --group gpu` on the CUDA box (for a
  remote lane, see compute-lanes.md ssh step 2). Without bitsandbytes, the model
  load fails with an error that names that exact command.
- Run the real QLoRA run on CUDA. bitsandbytes ≥ 0.50 also runs the smoke test
  on CPU: run `uv sync --group gpu`, then add `++trainer.args.use_cpu=true`. MPS
  is unverified (hardware.md).
- `verify` auto-SKIPs `param_drift` on a quantized run. 4-bit storage packs two
  elements per byte. `numel()` therefore undercounts ~2×, and the comparison is
  meaningless.

## Verify / tracking

- The `param_count` meta holds the count of the BASE model, pre-peft. The lane
  logs it before the trainer applies the adapter, so `param_drift` stays valid
  on a plain LoRA run.
- The `trainable_param_count` meta marks a LoRA run. It holds the requires_grad
  numel after the trainer construction. Expect a value orders of magnitude below
  `param_count`.
- The `quantized` meta is true whenever `model.main` carries a
  `quantization_config`.
