# Postmortem — path-3

**Symptom:** identical NaN divergence at lr 5e-5 — so the lr hypothesis from path-2 is
falsified.

**Diagnosis:** bare-TRL probe (model passed as string) trained perfectly on the same
data/args; bisecting the adapter's additions crashed on CPU with
`RuntimeError: mixed dtype (CPU)` in layer_norm. Root cause: the pythia-14m checkpoint
is stored in fp16 and transformers v5 loads checkpoints in their **stored dtype** by
default. The adapter loaded the model object in fp16 → full-precision fp16 AdamW →
divergence on MPS, crash on CPU. Bare TRL was unaffected because its internal model
creation forces fp32.

**Fix (code, not config):** `trainer.dtype: float32` added to the TRL trainer configs;
`_load_model` passes it explicitly to `from_pretrained`. Checkpoint dtype is now
opt-in via `trainer.dtype: auto`.
