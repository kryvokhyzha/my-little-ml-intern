# OOM recovery ladder

CUDA out-of-memory is a resource problem. Fix it with the resource knobs below,
in this order, one rung at a time. After each change, run the smoke test again
(`smoke_test=true` on the failing hardware). Relaunch the long run only after
that smoke run.

1. **Reduce per-device batch size.** Halve
   `trainer.args.per_device_train_batch_size` (floor: 1).
2. **Raise grad accumulation to compensate.** Increase
   `trainer.args.gradient_accumulation_steps` in proportion.
   `per_device_train_batch_size × gradient_accumulation_steps` (× num GPUs) must
   stay identical. The effective batch size is a hyperparameter of the
   hypothesis. A silent change to it invalidates the comparison.
3. **Gradient checkpointing.** Set `trainer.args.gradient_checkpointing=true`
   (lightning: model-side). This option costs ~20–30% step time. It saves
   activation memory.
4. **Smaller or quantized variant.** Use a QLoRA / 4-bit base, or a smaller
   checkpoint from the same family. This rung changes what the user gets, so it
   is NOT yours to decide. Interactive: ask the user. Headless: fire
   `scripts/bash/notify.sh approval_required "<proposed variant>"`, then record
   the assumption in task.md. Proceed only after that.
5. **Bigger GPU tier.** Move up to the next lane tier (see `hardware.md` and
   `compute-lanes.md`). Check the budget again first. A bigger tier burns
   `compute_cap_gpu_h` faster.
   `intern.py budget --experiment NNN can-retry --path-id <id>` must exit 0
   before the relaunch.

Rungs 1–3 and 5 are within this skill's authority. The architecture contract
allows OOM recovery to change the batch size, the grad accumulation, or the GPU
tier. It never allows a change to the method, the dataset, or the sequence
length without the user's approval. Rung 4 needs the approval described above.

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

Concretely forbidden without approval: SFT→LoRA, a lower `max_length`, a swap of
the dataset or the model, a dropped eval, and disabled tracking/alerts.

## OOM converts parallel paths to serial

One path can hit an OOM while sibling paths run on the same device. The fix for
the retry is "reduce concurrency", not only "reduce batch". Run the remaining
paths one at a time. Concurrent paths that share a GPU mask each other's true
memory footprint. They also turn one bug into N failed rows in the ledger.

## Bookkeeping — every OOM retry is a retry

```bash
uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1   # exit 0 or stop
uv run python scripts/python/intern.py budget --experiment NNN record-retry
uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1b --status queued --retry-of path-1 --failure-cause "CUDA OOM"
```

Write `experiments/NNN-<slug>/postmortems/path-<id>.md` first. The file records
symptom → root-cause hypothesis → fix applied. The symptom is the actual OOM
line from logs/stderr.log. The root-cause hypothesis names which tensor, and
which rung addresses it. If `can-retry` denies the retry, the path is `dropped`.
Let the sibling paths stand. Do not steal their budget.
