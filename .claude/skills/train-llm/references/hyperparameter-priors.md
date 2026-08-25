# Hyperparameter priors (post-training)

These priors come from the survey "frontier model training methodologies". The
survey covers the training of SmolLM3, Kimi K2, DeepSeek-R1, gpt-oss, Hermes 4,
Intellect-3 and Trinity:
https://djdumpling.github.io/2026/01/31/frontier_training.html. These values are
PRIORS for plan.md hypotheses and `trainer.args`, not laws. Use one override per
path. The verify gate arbitrates.

## SFT learning rate

- Band: set the LR about one order of magnitude below the pretraining LR. At ~3B
  scale, 3e-6–1e-5 gave the best results. An LR > 1e-5 degraded the reasoning
  evals there. The trl_sft default (2e-5) is a tiny-model default. At 1B+, sweep
  downward first.
- **SFT is short. Sweep the LR fully.** A full log-spaced sweep [1e-6 .. 1e-4]
  costs minutes to hours. Run the sweep. Do not argue about the priors.
- LoRA paths: use 10× the full-FT optimum instead. See lora.md.
- AdamW β1 0.9 / β2 0.95, weight decay 0.1 (or 0.01), grad clip 1.0, warmup 1–5%
  of steps. Every frontier run still uses these standard defaults.
- LR schedule: set `trainer.args.lr_scheduler_type`. Cosine ships as the SFT
  default. Set `trainer.args.lr_scheduler_kwargs` for the scheduler-specific
  params. For example, use `cosine_with_min_lr` with `{min_lr_rate: 0.1}` to put
  a floor under the LR. Or use `cosine_with_restarts` with `{num_cycles: 2}`.
  Both are native `SFTConfig` fields. Set them as plain keys under
  `trainer.args`.

## Multi-epoch SFT

- 2–3 epochs can help on a small dataset. SmolLM3's LiveCodeBench score nearly
  doubled from epoch 2 to epoch 3. `num_train_epochs` is a legitimate hypothesis
  lever, not a sign of a mistake.
- Caveat: multi-epoch training on a small set widens the train/eval gap. The
  verify gate's `eval_train_gap` check can therefore FAIL for a legitimate
  reason. That FAIL is the check at work, not noise. Investigate before you
  treat the FAIL as waivable: does the held-out eval metric still improve? The
  eval metric arbitrates, not the train loss.

## Masking

- Mask the user turns (assistant-only loss). The gains are small but real, and
  they are largest on the instruction-following evals. In the TRL lane, set
  `trainer.args.assistant_only_loss=true` for a conversational dataset. Check
  the key against the installed TRL version first (preflight mistake #1/#2).

## Packing — decide per-path and record it

- Packing gives 3–5× throughput. For a fixed token budget, it also gives FEWER
  optimizer updates. Packing raises the effective batch, and nothing in the run
  reports the change.
- An effective batch > 32 measurably degraded small-dataset SFT (IFEval −10
  points at 128). Packing helps on a LARGE dataset only. Disable packing for a
  small curated set. Lower the LR when packing is on.
- A packing change between two paths is a hidden second variable. Record the
  decision in the preflight checklist every time.

## Batch size

- Batch ×k → LR ×√k (the gradient-variance argument). A hypothesis that changes
  the effective batch but keeps the LR changes two variables at once. Nothing in
  the run reports the second change.
- A LoRA path caps the effective batch at < 32 in every case. See lora.md.

## DPO / preference optimization

- LR: set the DPO LR 10–20× below your SFT LR. Zephyr used 10×. SmolLM3 used
  20×, which gave 1e-6 at 3B.
- beta ≥ 0.1. In a 0.01–0.5 sweep, 0.1 was the best value, and it is the trl_dpo
  default. More than one epoch over the preference data overfits. Partition the
  data and iterate instead of a second epoch over the same data.
- The preference-set size has little effect. 2k pairs already help, and 100k+
  degraded the reasoning mode. Spend the budget on pair quality, not on volume.
