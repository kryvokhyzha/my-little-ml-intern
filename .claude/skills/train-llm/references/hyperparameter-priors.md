# Hyperparameter priors (post-training)

Distilled from "frontier model training methodologies" (survey of how SmolLM3,
Kimi K2, DeepSeek-R1, gpt-oss, Hermes 4, Intellect-3 and Trinity were trained):
https://djdumpling.github.io/2026/01/31/frontier_training.html. These are PRIORS
for plan.md hypotheses and `trainer.args` — not laws. One override per path;
verify arbitrates.

## SFT learning rate

- Band: roughly one order of magnitude below the pretraining LR; 3e-6–1e-5 won
  at ~3B scale, and LR > 1e-5 tanked reasoning evals there. The trl_sft default
  (2e-5) is a tiny-model default — sweep downward first at 1B+.
- **SFT is short — sweep LR fully.** A full log-spaced sweep [1e-6 .. 1e-4]
  costs minutes-to-hours; run it instead of arguing about priors.
- LoRA paths: LR is 10× the full-FT optimum instead — see lora.md.
- AdamW β1 0.9 / β2 0.95, weight decay 0.1 (or 0.01), grad clip 1.0, warmup 1–5%
  of steps: the boring defaults every frontier run still uses.

## Multi-epoch SFT

- 2–3 epochs can genuinely help on small datasets (SmolLM3's LiveCodeBench
  nearly doubled from epoch 2 to 3). `num_train_epochs` is a legitimate
  hypothesis lever, not a smell.
- Caveat: multi-epoch training on a small set widens the train/eval gap, so
  verify's `eval_train_gap` check may legitimately FAIL — that is the check
  doing its job, not noise. Investigate (is the held-out eval metric still
  improving?) before treating the FAIL as waivable; the eval metric, not train
  loss, arbitrates.

## Masking

- Mask user turns (assistant-only loss): small but real gains, largest on
  instruction-following evals. In the TRL lane this is
  `trainer.args.assistant_only_loss=true` for conversational datasets — verify
  the key against the installed TRL version first (preflight mistake #1/#2).

## Packing — decide per-path and record it

- Packing gives 3–5× throughput but, for a fixed token budget, FEWER optimizer
  updates — it silently raises the effective batch.
- Effective batch > 32 measurably hurt small-dataset SFT (IFEval −10 points at
  128). Packing pays off on LARGE datasets only; disable it for small curated
  sets, and lower the LR when it is on.
- A packing flip between paths is a hidden second variable — record the decision
  in the preflight checklist every time.

## Batch size

- Batch ×k → LR ×√k (gradient-variance argument). A hypothesis that changes
  effective batch without touching LR silently changes two variables.
- LoRA paths cap effective batch < 32 regardless — see lora.md.

## DPO / preference optimization

- LR: 10–20× below your SFT LR (Zephyr used 10×; SmolLM3 20× → 1e-6 at 3B).
- beta ≥ 0.1 (0.1 won a 0.01–0.5 sweep; the trl_dpo default). More than one
  epoch over preference data overfits — partition the data and iterate instead
  of re-epoching.
- Preference-set size barely matters (2k pairs already help; 100k+ degraded
  reasoning mode) — spend budget on pair quality, not volume.
