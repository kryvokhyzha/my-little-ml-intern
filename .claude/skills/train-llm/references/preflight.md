# Pre-flight checklist

Fill every line in and print the completed checklist in your response before
launching a long run. An unfillable line is a missing step, not a formality —
stop and complete it. (Smoke-scale and `999-` scratch runs are exempt.)

```
PRE-FLIGHT — NNN-<slug> / path-<id>
- Reference implementation: <URL or research.md row this run is based on>
- Dataset format verified: cfg.data.path=<value> — columns <...> match <method>
  (how: datasets load / datasets-server; see dataset-formats.md)
- Smoke run: VERDICT: TRAIN_OK | final_train_loss=<v>  (paste the actual line)
- Trainer lane: trainer=<trl_sft|trl_dpo|lightning|axolotl>, override: <the one Hydra override for this path>
- Tracking: backend=<trackio|wandb|none>, alerts on;
  run_name=<experiment_name> (first path) or <experiment_name>-path-<id> (retries / later paths)
- Budget headroom: can-launch exit 0; estimated <h> GPU-h of <cap> gpu_h remaining
- Wall-clock / timeout: <estimate> for <param count> on <hardware>, +30% buffer
  (justify from hardware.md; never leave a remote lane's default timeout unexamined)
- Precision & attention: <bf16|fp16|fp32>, <sdpa|prebuilt kernel> — allowed on
  this hardware per hardware.md
- Reference curve: <comparable prior run (run_url / metrics.jsonl) to check the
  trajectory against, or "none — first of its kind">
- Expected throughput: <tok/s or steps/min band from the smoke or reference
  run> — sustained deviation during the run = red flag
- Packing: <on|off> and why (small curated set → off; see
  hyperparameter-priors.md)
- Artifacts: output_dir=experiments/NNN-<slug>/ckpts, stderr tee to
  logs/stderr.log, samples to logs/samples.jsonl, metrics to metrics.jsonl
- Ledger: path-<id> row upserted (status=running once launched)
```

## Mistakes you WILL make (unless you check)

Adapted from HF ml-intern's failure log — each of these has burned real runs.

1. **Hallucinated imports.** Your internal knowledge of trl / transformers /
   peft / trackio APIs is stale. Renamed trainer classes, removed arguments,
   wrong config field names. Fix: check the installed version
   (`uv run python -c "import trl, transformers; print(trl.__version__, transformers.__version__)"`)
   and read a current example (literature-recipe-research skill, or the
   library's own docs for that version) before writing adapter overrides.
2. **Wrong trainer arguments.** You will pass args that don't exist in the
   installed version (classic: `max_seq_length` vs `max_length` on `SFTConfig`).
   Everything under `trainer.args` is mapped straight onto
   `SFTConfig`/`DPOConfig`/`L.Trainer` — an invalid key fails at start. Fix:
   verify each nonstandard key against the installed version's signature.
3. **Wrong dataset format.** You will assume column names without checking and
   get a KeyError 40 seconds into a paid run. Fix: step 2 of the workflow, see
   `dataset-formats.md`. Never skip it for "obvious" datasets.
4. **Default timeout kills jobs.** Remote lanes have wall-clock limits
   (`configs/compute/hf_jobs.yaml` defaults to 3h). Training takes hours; a
   killed job loses everything. Fix: size the timeout from model × hardware
   (hardware.md), add 20–30% buffer, and write the justification in the
   checklist. Minimum 2h for any real training.
5. **Lost artifacts.** Remote filesystems are ephemeral (hf_jobs) or get
   recycled (vast). If checkpoints/logs don't land back in
   `experiments/NNN-<slug>/` (rsync back, or Hub push on hf_jobs), the run never
   happened. Fix: plan the artifact return path before launch — it's a checklist
   line.
6. **Batch failures.** You will launch all paths at once and they will all die
   from the same bug. Fix: ONE path first, confirm loss lines are appearing,
   then the rest.
7. **Silent dataset substitution.** When the requested dataset fails to load you
   will quietly switch to a similar one. Never. Interactive: ask. Headless: fire
   `notify.sh approval_required` with the proposed substitute and record it
   under Unknowns in task.md before proceeding.
8. **Compiled flash-attn.** You will `pip install flash-attn` and lose an hour
   to a CUDA/torch version mismatch, or run it on pre-Ampere hardware where it
   cannot work. Fix: prebuilt kernels only, Ampere+ only — rules in
   `hardware.md`.
9. **Scope-changing fixes.** On error (especially OOM) you will reach for
   "creative" workarounds that change what the user asked for. The binding rule
   is in `oom-recovery.md` — read it before touching any config in response to a
   failure.
10. **Trusting the loss number.** A perfect loss curve proves nothing — label
    off-by-one, mask leaks, and EOS-only batches all produce beautiful curves on
    broken models. Only `intern.py verify` exit 0 plus eyeballed samples counts.
    Never write results.md before that.
11. **Progress bars eating logs.** tqdm output is useless in a teed log and
    floods context. The adapters log per-step lines (`logging_steps: 1` in the
    trainer groups); read them with `tail`/`grep`, not by re-running.
12. **Gemma-family softcapping silently broken in training.** Logit softcapping
    (Gemma-2-style) is incompatible with SDPA/flash-attention fused kernels
    during TRAINING — the run "works" and quietly trains wrong. Set
    `attn_implementation="eager"` when fine-tuning these families; inference can
    keep sdpa.
