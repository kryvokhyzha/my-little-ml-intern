# OOM recovery ladder

CUDA out-of-memory is a resource problem. Fix it with resource knobs, in this
order, one rung at a time — re-smoke (`smoke_test=true` on the failing hardware)
after each change before relaunching the long run.

1. **Reduce per-device batch size.** Halve
   `trainer.args.per_device_train_batch_size` (floor: 1).
2. **Raise grad accumulation to compensate.** Increase
   `trainer.args.gradient_accumulation_steps` proportionally so
   `per_device_train_batch_size × gradient_accumulation_steps` (× num GPUs)
   stays identical — the effective batch size is a hyperparameter of the
   hypothesis; changing it silently invalidates the comparison.
3. **Gradient checkpointing.** `trainer.args.gradient_checkpointing=true`
   (lightning: model-side). Costs ~20–30% step time, saves activation memory.
4. **Smaller or quantized variant.** QLoRA / 4-bit base, or a smaller checkpoint
   from the same family. This changes what the user gets, so it is NOT yours to
   decide: interactive → ask; headless → fire
   `scripts/bash/notify.sh approval_required "<proposed variant>"`, record the
   assumption in task.md, and only then proceed.
5. **Bigger GPU tier.** Next lane tier up (see `hardware.md` and
   `compute-lanes.md`). Re-check the budget first — a bigger tier burns
   `compute_cap_gpu_h` faster;
   `intern.py budget --experiment NNN can-retry --path-id <id>` must exit 0
   before the relaunch.

Rungs 1–3 and 5 are within this skill's authority (the architecture contract:
OOM recovery may change batch size / grad accumulation / GPU tier — never the
method, dataset, or sequence length without user approval). Rung 4 requires
approval as described.

## Anti-scope-drift rule (binding, from HF ml-intern)

> SCOPE-CHANGING FIXES: Avoid at all costs! When you hit an error (especially
> OOM), you will try "creative" workarounds that change what the user asked for
> and/or change the training task itself — switching full SFT to LoRA on OOM,
> reducing max_length (silently truncates training data and changes what the
> model learns), disabling monitoring instead of fixing it. Do not do this. Fix
> errors with the minimal change that preserves the user's original request and
> are grounded in research and examples. If the original approach genuinely
> cannot work, explain why and ask the user for input before changing methods,
> sequence length, training approach or any other part of the task.

Concretely forbidden without approval: SFT→LoRA, lowering `max_length`, swapping
the dataset or model, dropping eval, disabling tracking/alerts.

## OOM converts parallel paths to serial

If one path OOMs while sibling paths run on the same device, the fix for the
retry is "reduce concurrency", not just "reduce batch": run the remaining paths
one at a time. Concurrent paths sharing a GPU mask each other's true memory
footprint and turn one bug into N failed rows in the ledger.

## Bookkeeping — every OOM retry is a retry

```bash
uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1   # exit 0 or stop
uv run python scripts/python/intern.py budget --experiment NNN record-retry
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1b --status queued --retry-of path-1 --failure-cause "CUDA OOM"
```

Write `experiments/NNN-<slug>/postmortems/path-<id>.md` first: symptom (the
actual OOM line from logs/stderr.log) → root-cause hypothesis (which tensor,
which rung addresses it) → fix applied. If `can-retry` denies, the path is
`dropped` — let sibling paths stand; do not steal their budget.
