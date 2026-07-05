# Postmortem — path-2

**Symptom:** verify exit 1 — `loss_plausibility` red-flag FAIL (final loss exactly 0.0)
plus `generation_sanity` FAIL (empty continuations). Training log shows `grad_norm: nan`
and `entropy: nan` from mid-run onward.

**Root-cause hypothesis:** lr 5e-4 is far too high for fine-tuning a pretrained 14M-param
model at batch 8 — gradients blew up, weights went NaN, and the reported loss collapsed
to an impossible 0.0. The loss value alone looked like a great run; the plausibility
band's `< 1.0` red flag is what caught it.

**Gap noted for callbacks:** the NaN alert watches the loss value, which here reported
0.0 rather than NaN; `grad_norm` NaN should also be watched. Flagged for review.

**Fix:** canonical divergence mutation — one variable: `trainer.args.learning_rate`
5e-4 → 5e-5.
