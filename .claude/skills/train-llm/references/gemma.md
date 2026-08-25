# Gemma family: training and tuning notes

Read this file when an experiment composes a `configs/model/gemma_*` group. Read
it also when you plan such an experiment. Provenance tags: **[001]** = verified
by experiment 001-pi-mono-sft through this repo's gates. **[checked]** =
verified directly (template/tokenizer inspection, 2026-07-18). **[skills]** =
imported from the community
[google-gemma/gemma-skills](https://github.com/google-gemma/gemma-skills) pack
(its README disclaims official Google support). Where the sources disagree,
trust [001]/[checked] over [skills].

## Pick the model (before budget.md is written)

Default to the Gemma 4 generation [skills]. All Gemma 4 models have thinking
mode. The context is 256K, and 128K for the E-series:

| Model             | Repo                         | Modalities         | Niche                              |
| ----------------- | ---------------------------- | ------------------ | ---------------------------------- |
| Gemma 4 26B A4B   | `google/gemma-4-26B-A4B-it`  | text+image         | MoE — heavy reasoning, fast        |
| Gemma 4 31B       | `google/gemma-4-31B-it`      | text+image         | dense flagship; distill teacher    |
| Gemma 4 12B       | `google/gemma-4-12B-it`      | text+image+audio   | laptop-class multimodal            |
| Gemma 4 E2B / E4B | `google/gemma-4-E2B-it` …    | text+image+audio   | on-device; student models          |
| Gemma 3 270M–27B  | `google/gemma-3-*-it`        | text (+image ≥ 4B) | legacy; 270M/1B for tiny-scale     |
| EmbeddingGemma    | `google/embeddinggemma-300m` | text → vector      | RAG / retrieval (not a lane here)  |
| ShieldGemma 2     | `google/shieldgemma-2-4b-it` | classifier         | safety filtering (not a lane here) |

Budget gate: EVERY model in this table exceeds the default
`scale_ceiling_params` of 200M — even gemma-3-270M. Any Gemma path needs a
bigger-budget profile first. Justify that profile in budget.md (001 used
`budget: lora`, ceiling 12B).

## Load the model — what the model group must know

- `google/gemma-4-E2B-it` is a **multi-tower** `Gemma4ForConditionalGeneration`
  (language + vision + audio towers) [001]. Plain
  `AutoModelForCausalLM.from_pretrained` works for text SFT
  (`configs/model/gemma_4_e2b_it.yaml`) [001]. [skills] instead uses
  `AutoModelForMultimodalLM` + `AutoProcessor` as its default HF loader for
  Gemma 4, for every modality.
- QLoRA is the `_4bit` model variant (`model=gemma_4_e2b_it_4bit` — nested
  `BitsAndBytesConfig` node, nf4 + double quant). The quantization is model
  identity, never a trainer key [001]. lora.md holds the full recipe.
- To train a base (non `-it`) checkpoint, [skills] loads the processor from the
  `-it` counterpart repo.

## Chat template and loss masking — the #1 Gemma trap

- Gemma 4 turns are `<|turn>role … <turn|>`, and the assistant role is named
  **`model`** [checked]. Standard `messages` rows with `assistant` roles work,
  because the template maps `assistant` → `<|turn>model` [checked]. Never
  hand-format a turn. Always call `apply_chat_template`.
- The template has **no `{% generation %}` markers** [checked]. TRL's
  `assistant_only_loss` therefore hard-fails on Gemma 4. This is the same
  failure class that SmolLM2 hit in 002 (docs/008-example-distillation.md). The
  [skills] pack avoids the failure with a custom collator that token-searches
  `<|turn>model\n` and masks everything before that marker. The paths that work
  here without a custom collator:
  - `prompt` + `completion` data with `completion_only_loss: true` —
    template-free masking, verified on gemma-4-E2B [001].
  - `messages` data, unmasked full-conversation loss (the 002 posture) —
    document the supervision-scope confound in plan.md.

## Recipe priors (one override per path; see the conflict note)

001 verified these values on gemma-4-E2B QLoRA, and they match the [skills]
defaults: `r=16`, `lora_alpha=32`, dropout 0.05, LR `2e-4`, cosine + ~3% warmup
[001].

**Known conflict:** lora.md's no-regret recipe says r=256, `lora_alpha=16` held
FIXED (never scale alpha with rank), and dropout 0.0. The shipped `trl_sft_lora`
preset encodes those values. [skills]' `alpha = 2·r` rule is exactly the tuning
method that lora.md forbids. For a Gemma path, 001's values are the verified
starting point. From there, explore r-vs-alpha per lora.md: tune `r`, and keep
alpha fixed.

- `target_modules` — **the presets are a trap on Gemma**: `trl_sft_lora` /
  `trl_sft_qlora` ship `target_modules: all-linear`. An experiment config that
  omits the key therefore inherits `all-linear`. `all-linear` attaches the
  adapters to the vision and audio towers. This is a real failure from the
  reference sweep that 001 was built on (docs/007). Override the key with 001's
  explicit language-tower regex. Or set the key to `null`, which falls back to
  PEFT's Gemma-4 default (scoped to the LM layers [skills]; `q_proj`/`v_proj`
  per lora.md). Fail hard on zero matches — lora.md has the check.
- `max_length` 2048–8192 for local/single-GPU runs. The model's full context
  window (256K, 128K E-series) is a deployment feature, not a training default.
  The activations are the OOM lever [skills].
- Full fine-tune LR: [skills] says `2e-5`, but that value is a tiny-model
  default. At Gemma scales (all ≥ 1B; E2B is ~5.1B total [001]),
  hyperparameter-priors.md says to sweep downward into 3e-6–1e-5 first.
- DPO: run SFT into your format FIRST. DPO directly on an out-of-domain base
  degrades the format [skills]. Use `beta=0.1` (0.1–0.5). With PEFT adapters,
  `ref_model=None` (implicit reference = base + disabled adapter) saves a full
  model of VRAM.
- Reward modeling: `AutoModelForSequenceClassification` had **no Gemma 4
  support** at the [skills] snapshot. Check the support before you plan an RM
  path.

## Multimodal SFT (vision/audio) — when a path needs it

Multimodal SFT is not a shipped lane. Plan it as a custom-collator `trl_sft`
path [skills]:

- The rows keep `messages`, but `content` becomes a block list:
  `{"type": "image", "url": …}` / `{"type": "audio", "url": …}` +
  `{"type": "text", "text": …}`. Audio needs 16 kHz mono (librosa), and
  E2B/E4B/12B only. Vision works on any Gemma 4.
- The collator itself does `apply_chat_template` and the processor batching. Set
  `remove_unused_columns: false` and
  `dataset_kwargs: {skip_prepare_dataset: true}`. Otherwise TRL's default prep
  destroys the block structure.
- [skills]' QLoRA-audio runs install a `masked_scatter` dtype-coercion patch
  (labeled an audio dtype fix). Expect dtype problems there.

## Distillation ladder

`tokenizer.json` is bit-identical across gemma-4-E2B-it and gemma-4-31B-it (same
HF blob id) [checked]. A teacher 31B/26B-A4B → student E2B/E4B therefore works
with the token-level `trl_gkd` lane. This is the same pattern as the SmolLM2
ladder in 002/003 (docs/008). Text-level distillation (the [skills]
`distill_dataset.py` flow: the teacher generates the strings, then the student
SFTs on them) is our `trl_sft`-on-teacher-data lane, or the distill-traces
skill.

## After publish: export targets [skills]

The publish-model gate ships HF safetensors. Use a downstream conversion when an
experiment's deliverable needs one:

- **GGUF** (llama.cpp / LM Studio / Ollama): convert the merged model with
  llama.cpp's converter. Official `{model}-qat-q4_0-gguf` variants exist. They
  are quantization-aware-trained, and better than a post-hoc Q4 of the BASE
  model. Your fine-tune still needs its own conversion.
- **vLLM / SGLang**: `{model}-qat-w4a16-ct` compressed-tensors variants.
- **Faster eval/serving**: Gemma 4 MTP speculative decoding. Pair the target
  with its `{model}-assistant` drafter repo.
- **On-device**: LiteRT-LM (`.litertlm`, E2B/E4B).

## Deliberately not adopted

- **Unsloth** ([skills] recommends it first for local single-GPU): Unsloth
  patches the models at import time, and it forks the training code. This repo's
  lanes stay plain TRL under the uv lock. Real runs go to remote CUDA boxes
  anyway (compute-lanes.md). Revisit Unsloth only if a single-GPU-local
  experiment is VRAM-blocked on the TRL lane.
- **gemma-dev app tooling** (Gradio/Vertex/transformers.js serving): outside
  this repo's scope. An experiment ends at publish.

## Doc lookup

Index: fetch `https://ai.google.dev/gemma/docs/llms.txt`. Then fetch the
per-page `….md.txt` URLs (e.g. `core/prompt-formatting-gemma4.md.txt`,
`capabilities/text/function-calling-gemma4.md.txt`). This method is cheaper and
fresher than a scrape of the HTML.
