# Postmortem — path-1

**Symptom:** verify exit 1 — `generation_sanity` FAIL. All three samples collapse to a
single repeated token ("factors" = 96–97% of output) despite train loss 10.59 sitting
inside the ln(vocab) plausibility band.

**Root-cause hypothesis:** `sshleifer/tiny-gpt2` is a randomly-initialized ~100k-param
test checkpoint (hidden size 2). It cannot model language at any step count; greedy
decoding from a near-uniform distribution collapses to one token. Loss barely moved
(10.61 → 10.59) because the model has no capacity — the loss value alone looked
plausible, which is precisely the failure mode the generation check exists for.

**Fix:** retry with a small but genuinely pretrained model — `EleutherAI/pythia-14m`
(one-variable change: `trainer.model_name`; `trainer.target_params=14000000` for the
param-drift check). Everything else unchanged.
