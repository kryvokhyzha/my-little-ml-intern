# Pretraining / continued pretraining

Distilled from the Smol Training Playbook (Hugging Face — the full SmolLM3
build: 3B params, 11T tokens, 384×H100 for ~a month):
https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook. Scaled to
this repo's budgets — the discipline transfers even when "pretraining" here
means a SmolLM2-135M-class model on a tiny corpus.

First question: should you pretrain at all? Prompting or fine-tuning an existing
strong model is the mandatory first attempt — "fine-tuning for 1T tokens is
cheaper than pretraining 10T+". Legit reasons: a concrete research question at a
defined scale, production constraints (domain / deployment / governance), or a
strategic open gap. Write the reason into task.md.

## Sizing

- Pick N (params) FIRST from the deployment target, then D (tokens) from the
  compute you actually have via C ≈ 6·N·D FLOPs at ~30% MFU. SmolLM3: 3B for
  on-device → 384 H100 × 1 month at 30% MFU → 11T tokens.
- GPU count = total FLOPs / (per-GPU peak × MFU × target wall-clock).
- Overtraining past Chinchilla-optimal (~20 tok/param) is normal and correct
  when inference cost matters (Qwen3 trained on 36T tokens). Chinchilla
  minimizes training compute, not lifetime cost.
- Repo reality: `scale_ceiling_params` in budget.md still gates the launch, and
  a 135M model at Chinchilla is ~2.7B tokens — size the corpus honestly in
  plan.md before writing hypotheses.

## Ablation rig (derisk at small scale)

- ML is an experimental science; intuition fails (arXiv data HURTS small
  models). "Never change anything unless you've tested that it helps."
- The playbook's rig: a 1B proxy on 45B tokens (~1.3× Chinchilla), 1.5 days on
  one 8×H100 node, fixed data mix. Scaled to our budgets: a SmolLM2-135M-class
  model on a tiny corpus IS the rig — minutes-to-hours per run, same logic.
- One variable per ablation — plan.md's one-override-per-path rule is exactly
  this. When a change shifts param count (tied embeddings, GQA), re-match via
  layers/hidden and track counts explicitly.
- A validated change becomes the NEW baseline; test the next change against it.
  Battle-tested features first; no grid searches.
- Asymmetry of transfer: negative small-scale results reliably kill ideas;
  positive ones must be re-confirmed near target scale.
- Budget honestly: SmolLM3 spent ~58% as much on ablations + debugging as on the
  main run. Ablation coverage is also debugging speed — when the main run
  misbehaves, the untested component is the suspect list.

## Compass metrics (loss is not enough)

- Loss alone misleads: Wikipedia-heavy data lowers loss without a better model;
  tokenizer changes make losses incomparable; downstream ability can improve
  after loss plateaus.
- A good ablation benchmark is: monotonic over training, low-noise, above-random
  EARLY, and ranking-consistent across checkpoints.
- Use cloze formulation (per-character-normalized log-prob over answer choices)
  early; multiple-choice A/B/C/D formulations stay at random chance until far
  past small-scale budgets, and free-form generation is even later.
- If two runs differ by less than seed-to-seed noise, the ablation says nothing
  — rerun with different seeds or drop the idea.

## Data staging

- Multi-stage curriculum is standard practice; final behavior is dominated by
  data seen LATE. Save small high-quality sets for the LR-decay/annealing phase
  instead of diluting them from step 0.
- Change the mixture on signal: a lagging benchmark = inject better data for
  that domain; plan the high-quality injection into the decay window.
- Don't exceed ~5 epochs over any small dataset (data-constrained scaling).
- Automated mixture methods (DoReMi, RegMix) converged to roughly the natural
  distribution and did not beat manual ablations.

## Architecture defaults worth stealing

- GQA with ratio ~4 (e.g. 32Q/8KV heads) matches MHA and slashes KV cache; MQA /
  ratio-16 hurt.
- Tied embeddings for small models — at fixed params, extra depth beats untying.
- Intra-document masking from the START — neutral at short context, crucial for
  long-context training speed later.
- NoPE (RoPE removed every 4th layer) matches RoPE at short context and is the
  better long-context foundation.
- Optimizer: AdamW β1 0.9 / β2 0.95, wd 0.1, clip 1.0 — unchanged across
  Llama/DeepSeek generations. WSD schedule matches cosine but allows mid-run
  extension and decay-tail-only ablations.

## Marathon gotchas (long-run ops)

- **Throughput is a red-flag channel**, not a vanity stat: know the expected
  tok/s band and treat any sustained deviation as an incident. Playbook
  culprits: storage cache eviction, a dataloader index growing with total steps,
  one thermally-throttled GPU collapsing all-reduce bandwidth across 16 nodes.
- **Node-local storage for the dataset** — network-storage cache eviction
  produced a 40% throughput collapse hours into the run.
- **Reference-curve comparison**: intermediate checkpoints of a comparable prior
  model are how SmolLM3 caught a silent convergence tax (same RNG seed on all TP
  ranks) — and restarted from scratch at 1T tokens. Keep a comparable run's
  metrics around and compare trajectories, not endpoints (preflight.md has the
  checklist line).
- **Checkpoint + resume discipline**: save on a schedule, upload off-box, delete
  local only after the next save lands; verify auto-resume BEFORE the long run;
  never leave destructive commands (`rm -rf $CKPT_DIR`) in run scripts.
- "If you can automate only one thing, automate evaluations."
