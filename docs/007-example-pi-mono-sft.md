# 007 — Example: pi-mono SFT (gemma-4-E2B QLoRA)

<!-- Walkthrough of the repo's first real task example: SFT google/gemma-4-E2B-it on
badlogicgames/pi-mono agent traces via QLoRA. Two-step flow (prep the dataset, then run
experiment 001), the gemma-4 LoRA-target gotcha, why QLoRA can't smoke on Mac, and the
completion-only-loss note. Contract lives in docs/001-architecture.md; lifecycle in
docs/003-experiment-lifecycle.md. -->

This is the repo's first **real task example** (000-tiny-sft-smoke is a plumbing
fixture — see [docs/003](003-experiment-lifecycle.md)). It reproduces the
`burtenshaw/training-agents` gemma-4 pi-mono SFT recipe inside this project's
gate chain: fine-tune `google/gemma-4-E2B-it` on `badlogicgames/pi-mono`
coding-agent traces via QLoRA to teach the model the pi interaction format
(chat + tool calls) and basic task behavior.

## The two-step flow

The dataset and the training run are separated. The Hub dataset is raw session
JSONL — `datasets.load_dataset` on it can fail on mixed session content — so a
prep script converts it once and materializes clean `{prompt, completion}` rows
to your own Hub namespace, and the experiment trains against that.

**Step 1 — materialize the dataset** (runs anywhere; CPU-only, needs the
tokenizer but no GPU):

```text
uv run python scripts/python/prep-pi-mono-sft.py
```

This downloads the raw `*.jsonl` traces, converts every visible user / assistant
/ tool-call / tool-result turn into prompt/completion rows (completion = the
assistant turn), filters examples over `max_length=4096`, splits `train`/`test`
with seed 42, and pushes `${HF_USER}/pi-mono-sft` (private) to the Hub. Config:
[configs/prep-pi-mono-sft.yaml](../configs/prep-pi-mono-sft.yaml); script:
[scripts/python/prep-pi-mono-sft.py](../scripts/python/prep-pi-mono-sft.py). The
full data-provenance card — pipeline diagram, format, and per-stage row counts —
is the experiment's own [data.md](../experiments/001-pi-mono-sft/data.md).

**Step 2 — run the experiment on a GPU lane:**

```text
uv sync --group gpu                                          # on the CUDA box
uv run python scripts/python/001-pi-mono-sft.py smoke_test=true   # gate: VERDICT: TRAIN_OK
uv run python scripts/python/001-pi-mono-sft.py                   # ~65 min on an L40S
uv run python scripts/python/intern.py verify --experiment 001   # blocking gate, exit 0
```

Experiment 001 composes the `data: pi_mono_sft` group —
`data.path=${HF_USER}/pi-mono-sft` with `train` for fitting and `test` for
held-out eval. Config:
[configs/001-pi-mono-sft.yaml](../configs/001-pi-mono-sft.yaml).

## The gemma-4 LoRA-target gotcha

`google/gemma-4-E2B-it` is a multi-tower `Gemma4ForConditionalGeneration` (~5.1B
params): a language decoder tower plus vision and audio towers. **`all-linear`
is wrong here** — it attaches LoRA adapters to the vision/audio towers, which
the reference sweep verified as a real failure (broad targeting hit unsupported
wrapper modules). The fix is to target only the language tower's projection
modules with an explicit regex:

```text
.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$
```

That matches 205 modules (the 7 projections per decoder layer). The
`configs/trainer/trl_sft_lora.yaml` preset ships `target_modules: all-linear`
for the no-regret single-tower recipe; experiment 001 **overrides it** with the
gemma-4 regex under `_self_`. Custom target specs must be resolved against
`model.named_modules()` and fail hard on zero matches. Full rationale:
[.claude/skills/train-llm/references/lora.md](../.claude/skills/train-llm/references/lora.md).

## QLoRA can't smoke on Mac

The run uses QLoRA: the 5.1B base loads in **4-bit nf4** (double-quant, bf16
compute) so the whole thing fits a single 24GB GPU — the
`model: gemma_4_e2b_it_4bit` group nests the `quantization_config`
BitsAndBytesConfig node inside `model.main`. 4-bit loading goes through
**bitsandbytes**, which is CUDA-only — there is no MPS/CPU path — so this
experiment **cannot smoke on a Mac**. A `quantization_config` in `model.main`
fails at model load with a clear error naming `uv sync --group gpu` when
bitsandbytes is absent. Smoke directly on the CUDA target with `smoke_test=true`
(`max_steps=1`, dataset sliced to ≤ 32 rows, no checkpoint save).

To validate the data + trl_sft plumbing without a GPU, swap the model group to a
tiny non-quantized proxy (see task.md):
`model=smollm2_135m 'trainer.peft.target_modules=all-linear' smoke_test=true`
(the data group stays `pi_mono_sft`, so the prep'd dataset is still required).
That exercises loading, rendering, and wiring — not the QLoRA/gemma-4 path.

## The completion-only-loss note

`completion_only_loss: true` masks the loss to the assistant completion — user
prompts and tool outputs are not training targets. A consequence: the reported
loss is **prompt/completion loss over the completion only**, so a "low" number
like the reference `eval_loss ~= 0.55` is legitimate, not a red flag. The
adapter writes a `completion_only` meta when `trainer.args.completion_only_loss`
is truthy (see `src/training/trl/config.py:write_meta`), and `verify` reads it —
so `loss_plausibility` is judged with that context rather than flagging the low
completion-masked loss as suspicious. Cross-ref:
[docs/004-budget-and-gates.md](004-budget-and-gates.md) for the check catalogue.

## Reference published numbers

The reference recipe (`lr2e4-r16-len4k`, chosen by lowest held-out eval loss)
reported, on an L40S 48GB in ~65 min:

| Metric                    | Value  |
| ------------------------- | ------ |
| final train loss          | 0.6465 |
| final eval loss           | 0.5506 |
| final eval token accuracy | 0.8624 |
| HumanEval pass@1 (t=0.0)  | 0.744  |
| MBPP pass@1 (t=0.5)       | 0.651  |

Training metrics are held-out prompt/completion loss and token accuracy; the
HumanEval / MBPP numbers come from a separate Inspect AI coding-benchmark pass
against the final adapter, not from the training loop. Experiment 001's H1
expects `final_eval_loss ~= 0.55` and generation samples that follow the pi
tool-call format; either failing kills the hypothesis
([experiments/001-pi-mono-sft/plan.md](../experiments/001-pi-mono-sft/plan.md)).

## Known limits

- Overlength filtering at `max_length=4096` drops many long-context examples.
- Raw traces should be audited for secrets and private code before treating this
  as a production recipe.
- The materialized dataset and any pushed adapters are private Hub repos by
  default (repo-wide private-by-default; [docs/005](005-security.md)).
