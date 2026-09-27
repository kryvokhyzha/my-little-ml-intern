# Pretraining / continued pretraining

These notes come from the Smol Training Playbook (Hugging Face). The playbook
covers the full SmolLM3 build: 3B params, 11T tokens, 384×H100 for ~a month:
https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook. This
reference scales the playbook to this repo's budgets. The discipline still
transfers when "pretraining" here means a SmolLM2-135M-class model on a tiny
corpus.

Answer one question first: does this task need a pretraining run at all? A
prompt or a fine-tune on an existing strong model is the mandatory first
attempt. The playbook gives the reason: "fine-tuning for 1T tokens is cheaper
than pretraining 10T+". Three reasons are valid: a concrete research question at
a defined scale, a production constraint (domain / deployment / governance), or
a strategic open gap. Write the reason into task.md.

## Size the run

- Pick N (params) FIRST from the deployment target. Then pick D (tokens) from
  the compute that you actually have. Use C ≈ 6·N·D FLOPs at ~30% MFU. SmolLM3
  is the worked example: 3B for on-device → 384 H100 × 1 month at 30% MFU → 11T
  tokens.
- GPU count = total FLOPs / (per-GPU peak × MFU × target wall-clock).
- You can train past the Chinchilla-optimal point (~20 tok/param). That choice
  is normal and correct when the inference cost matters. Qwen3 trained on 36T
  tokens. Chinchilla minimizes the training compute, not the lifetime cost.
- In this repo, `scale_ceiling_params` in budget.md still gates the launch. A
  135M model at Chinchilla needs ~2.7B tokens. Size the corpus honestly in
  plan.md before you write the hypotheses.

## Ablation rig (derisk at small scale)

- ML is an experimental science, and intuition fails. arXiv data HURTS small
  models. The playbook states the rule: "Never change anything unless you've
  tested that it helps."
- The playbook's rig is a 1B proxy on 45B tokens (~1.3× Chinchilla), 1.5 days on
  one 8×H100 node, with a fixed data mix. At this repo's budgets, a
  SmolLM2-135M-class model on a tiny corpus IS the rig. Each run takes minutes
  to hours, and the logic is the same.
- Change one variable per ablation. The one-override-per-path rule in plan.md
  says exactly this. When a change shifts the param count (tied embeddings,
  GQA), re-match the count through the layers or the hidden size. Track the
  counts explicitly.
- A validated change becomes the NEW baseline. Test the next change against that
  baseline. Test the proven features first. Do not run a grid search.
- Transfer is asymmetric. A negative small-scale result reliably rules out an
  idea. You must re-confirm a positive result near the target scale.
- Budget honestly. SmolLM3 spent ~58% as much on the ablations and the debugging
  as on the main run. Ablation coverage also sets the debugging speed. When the
  main run misbehaves, the untested component is the first suspect.

## Compass metrics (loss is not enough)

- The loss alone misleads you. Wikipedia-heavy data lowers the loss but does not
  give a better model. A tokenizer change makes two losses incomparable. The
  downstream ability can improve after the loss plateaus.
- A good ablation benchmark has four properties:
  - monotonic over the training run
  - low-noise
  - above-random EARLY
  - ranking-consistent across the checkpoints
- Use the cloze formulation early (per-character-normalized log-prob over the
  answer choices). A multiple-choice A/B/C/D formulation stays at random chance
  until far past the small-scale budgets. Free-form generation becomes usable
  even later.
- If two runs differ by less than the seed-to-seed noise, the ablation says
  nothing. Run the ablation again with different seeds, or drop the idea.

## Data stages

- A multi-stage curriculum is standard practice. The data that the model sees
  LATE dominates the final behavior. Save the small high-quality sets for the
  LR-decay/annealing phase. Do not dilute them from step 0.
- Change the mixture on a signal. A benchmark that lags means one action: inject
  better data for that domain. Plan the high-quality injection into the decay
  window.
- Do not exceed ~5 epochs over any small dataset (data-constrained scaling).
- The automated mixture methods (DoReMi, RegMix) converged to roughly the
  natural distribution. They did not beat the manual ablations.
- Chunk long documents in Python: tokenize once, then slice the ids. Do not use
  `return_overflowing_tokens=True` with a `stride`. tokenizers 0.23.x drops
  tokens from the overflow windows (verified on 0.23.2: 30 tokens gave windows
  of 16 and 8, not 16 and 16).

## Architecture defaults to reuse

- GQA with ratio ~4 (e.g. 32Q/8KV heads) matches MHA and reduces the KV cache.
  MQA and ratio-16 hurt.
- Use tied embeddings for a small model. At fixed params, extra depth beats
  untied embeddings.
- Turn on intra-document masking from the START. It is neutral at short context.
  It is crucial for the long-context training speed later.
- NoPE (RoPE removed every 4th layer) matches RoPE at short context. NoPE is
  also the better long-context foundation.
- Optimizer: AdamW β1 0.9 / β2 0.95, wd 0.1, clip 1.0. These values stay
  unchanged across the Llama and DeepSeek generations. The WSD schedule matches
  cosine, and it also allows a mid-run extension and a decay-tail-only ablation.

## Long-run ops — known failure modes

- **Throughput is a failure signal**, not a status number. Know the expected
  tok/s band. Treat any sustained deviation as an incident. The playbook names
  three causes:
  - storage cache eviction
  - a dataloader index that grows with the total step count
  - one thermally-throttled GPU that collapses the all-reduce bandwidth across
    16 nodes
- **Use node-local storage for the dataset.** A network-storage cache eviction
  produced a 40% throughput collapse hours into the run.
- **Reference-curve comparison**: SmolLM3 caught a silent convergence tax with
  the intermediate checkpoints of a comparable prior model. The cause was the
  same RNG seed on all TP ranks. The SmolLM3 team then restarted from scratch at
  1T tokens. Keep the metrics of a comparable run. Compare the trajectories, not
  the endpoints (preflight.md has the checklist line).
- **Checkpoint and resume discipline**: save on a schedule. Upload each
  checkpoint off-box. Delete the local copy only after the next save lands.
  Verify the auto-resume BEFORE the long run. Never leave a destructive command
  (`rm -rf $CKPT_DIR`) in a run script.
- "If you can automate only one thing, automate evaluations."
