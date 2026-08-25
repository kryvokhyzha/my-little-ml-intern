# Pre-flight checklist

Fill in every line. Print the completed checklist in your response before you
launch a long run. A line that you cannot fill in is a missing step, not a
formality. Stop and complete that step first. (Smoke-scale runs and `999-`
scratch runs are exempt.)

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

This list comes from the HF ml-intern failure log. Each item caused a real run
to fail.

1. **Hallucinated imports.** Your internal knowledge of trl / transformers /
   peft / trackio APIs is stale. The libraries rename trainer classes, remove
   arguments, and change config field names. Fix: check the installed version
   (`uv run python -c "import trl, transformers; print(trl.__version__, transformers.__version__)"`).
   Then read a current example — the literature-recipe-research skill, or the
   library's own docs for that version. Do this before you write any adapter
   override.
2. **Wrong trainer arguments.** You will pass args that do not exist in the
   installed version. A frequent example is `max_seq_length` versus `max_length`
   on `SFTConfig`. The lane maps everything under `trainer.args` straight onto
   `SFTConfig`/`DPOConfig`/`L.Trainer`. An invalid key fails at the start. Fix:
   verify each nonstandard key against the installed version's signature.
3. **Wrong dataset format.** You will assume the column names and skip the
   check. A KeyError then stops a paid run 40 seconds after the start. Fix: do
   step 2 of the workflow. Read `dataset-formats.md`. Never skip that step for
   an "obvious" dataset.
4. **Default timeout kills jobs.** A remote lane has a wall-clock limit
   (`configs/compute/hf_jobs.yaml` defaults to 3h). A training run takes hours,
   and a job that hits the limit loses all of its work. Fix: size the timeout
   from model × hardware (hardware.md). Add a 20–30% buffer. Write the
   justification in the checklist. Use a minimum of 2h for any real training
   run.
5. **Lost artifacts.** A remote filesystem is ephemeral (hf_jobs), or the
   provider recycles it (vast). The checkpoints and the logs must land back in
   `experiments/NNN-<slug>/`, through an rsync back or through a Hub push on
   hf_jobs. If they do not, the run leaves no usable result. Fix: plan how the
   artifacts come back before the launch. That plan is a checklist line.
6. **Batch failures.** You will launch all of the paths at once, and the same
   bug will fail all of them. Fix: launch ONE path first. Confirm that the loss
   lines appear. Then launch the rest.
7. **Silent dataset substitution.** When the requested dataset fails to load,
   you will switch to a similar one without a report. Never do this.
   Interactive: ask the user. Headless: fire `notify.sh approval_required` with
   the proposed substitute. Record the substitute under Unknowns in task.md
   before you proceed.
8. **Compiled flash-attn.** You will run `pip install flash-attn`. You will then
   lose an hour to a CUDA/torch version mismatch. Or you will run it on
   pre-Ampere hardware, where it cannot work. Fix: use prebuilt kernels only.
   Use Ampere+ hardware only. The rules are in `hardware.md`.
9. **Scope-changing fixes.** On an error, and especially on an OOM, you will use
   a "creative" workaround that changes what the user asked for. The binding
   rule is in `oom-recovery.md`. Read that rule before you change any config in
   response to a failure.
10. **Trust in the loss number.** A perfect loss curve is not evidence. A label
    off-by-one, a mask leak, and an EOS-only batch all produce a smooth curve on
    a broken model. Only two facts count: `intern.py verify` exits 0, and you
    read the generation samples yourself. Never write results.md before that.
11. **Progress bars in the logs.** tqdm output is useless in a teed log, and it
    fills the context window. The adapters log per-step lines
    (`logging_steps: 1` in the trainer groups). Read those lines with
    `tail`/`grep`. Do not re-run the job to see them.
12. **Gemma-family softcapping breaks silently in training.** Logit softcapping
    (Gemma-2-style) is incompatible with the SDPA/flash-attention fused kernels
    during TRAINING. The run raises no error, and it trains the model wrong. Set
    `attn_implementation="eager"` when you fine-tune these families. Inference
    can keep sdpa.
