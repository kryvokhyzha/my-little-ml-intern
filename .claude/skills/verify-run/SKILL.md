---
name: verify-run
description:
  Run the blocking verification gate on a finished training run and act on the
  result — mechanical checks plus a mandatory human read of generation samples.
  Use after ANY training run completes, BEFORE reporting results, claiming
  success, or writing results.md — even when the loss curve looks perfect. Also
  use when the user says "verify the run", "is the model actually good", "check
  training results", "did training work", "sanity-check the model", or doubts
  whether a run's numbers are real. If a training run just finished in this
  session, invoke this skill without being asked.
---

# verify-run

Run `intern.py verify` on an experiment. Judge the generations yourself. Then
route the outcome. On a pass, update the ledger and write results.md. On a fail,
write a postmortem and update the ledger. Retry only when the budget gate allows
it.

**A low loss number is never evidence the model works.**

Blocking-gate rule: never write results.md unless `intern.py verify` exited 0.
Never report success unless `intern.py verify` exited 0. A failed gate means the
run failed, whatever the loss shows.

## Workflow

1. **Locate the experiment (NNN).** Research before you clarify. Never ask the
   user what you can look up:

   - If you trained the experiment in this session, use that experiment number.
   - Otherwise, run `ls experiments/`. Prefer the directory whose
     `metrics.jsonl` is the newest, or whose ledger has a `running` or `queued`
     row
     (`uv run python scripts/python/intern.py ledger --experiment NNN show`).
   - Note the `path_id` of the row that you verify. You need it in step 5.
   - The blocking gates do not apply to a `999-` scratch experiment. Verify such
     an experiment when the user asks. The results.md prohibition does not apply
     to it.

2. **Run the gate.**

   ```
   uv run python scripts/python/intern.py verify --experiment NNN
   ```

   Capture the exit code. Options:

   - `--vocab-size N` — pass this option only when the effective tokenizer vocab
     differs from the `vocab_size` meta in metrics.jsonl (resized embeddings,
     swapped tokenizer). Otherwise omit it.
   - `--checks a,b` — scope the gate to specific checks by name
     (comma-separated). Use this option when you re-verify a single fixed check.
     The default is all applicable checks. A scoped run only prints its report.
     A scoped run never writes verify.md and never overwrites it. Only a full
     (unscoped) run writes verify.md.

   metrics.jsonl accumulates across the paths and the retries. The gate scopes
   itself to the last `run_start` event. So do NOT delete metrics.jsonl between
   the retries.

3. **Read the report** at `experiments/NNN-<slug>/verify.md`. For each `FAIL`
   line, state in one line what the check measures. Also state in one line why
   this run failed it. For the thresholds and the exact semantics, read the
   "verify.py" section of `docs/001-architecture.md` when a name is unfamiliar.
   The default checks are:

   | check                      | one-line meaning                                                                   |
   | -------------------------- | ---------------------------------------------------------------------------------- |
   | `loss_plausibility`        | final train loss inside the ln(vocab) band; < 1.0 on an LM task is a red-flag FAIL |
   | `eval_train_gap`           | eval and train loss within 0.5 of each other                                       |
   | `data_consumption`         | model actually saw ≥ 70% of planned tokens                                         |
   | `stderr_scan`              | no Traceback / RuntimeError / CUDA OOM in logs/stderr.log                          |
   | `param_drift`              | actual param count within 15% of target                                            |
   | `generation_sanity`        | samples.jsonl exists, not degenerate (mechanical proxies only)                     |
   | `reward_margin` / `kl_ref` | DPO-only: positive reward margin, finite KL                                        |

4. **MANDATORY human-judgment step — even on a mechanical PASS.** Read
   `experiments/NNN-<slug>/logs/samples.jsonl`. Judge whether the generations
   are recognizable language for the training distribution. A TinyStories model
   must produce story-like English. A code model must produce code-like text.
   Text that uses a valid vocabulary but carries no meaning is a fail, not a
   partial pass. Append your judgment to verify.md as one line:

   ```
   JUDGMENT: generation_quality = PASS|FAIL | <one-line reasoning against the training distribution>
   ```

   A FAIL judgment fails the whole run, even when the exit code was 0.

5. **Route the outcome.**

   **Overall pass** (exit 0 AND judgment PASS):

   ```
   uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status passed --verify pass
   ```

   Write `experiments/NNN-<slug>/results.md` only after you update the ledger.
   That file holds the winner and the comparison, as the experiment convention
   requires. Then confirm that the scaffold is complete.
   `uv run python scripts/python/intern.py check --experiment NNN` must exit 0.
   That command catches a forgotten run.md before you report done.

   **Fail** (exit 1, or judgment FAIL):

   - Write `experiments/NNN-<slug>/postmortems/path-<id>.md` with exactly:
     symptom → root-cause hypothesis → fix.
   - Update the ledger:

     ```
     uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status failed --verify fail --failure-cause "<one line>"
     ```

   - Before ANY retry, run the budget gate. Stop when the gate denies the retry:

     ```
     uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1
     ```

     A nonzero exit means no retry. Report the postmortem and stop. Never fire
     `notify.sh train_done` for a run that has no passing path. (The notify.sh
     events are a fixed list — read its usage header. An unknown name sends a
     degraded generic card. For a summary, write one `- ` bullet line per item.)

6. **Interpret the exit codes plainly.**
   - `0` — all the checks passed. Step 4 still applies before you claim success.
   - `1` — at least one check FAILED. The run FAILED, whatever the loss shows.
     Say so in those words. Do not write "mostly passed". Do not write "partial
     success".
   - `2` — the artifacts are missing. The training run did not produce
     metrics.jsonl, or the experiment directory is wrong. That result is a
     pipeline bug, not "done". Fix the training script or the callback wiring.
     Then rerun the training. Never hand-craft the missing files to satisfy the
     gate.

## Posture

- **Headless:** never hang. Some calls are ambiguous: which path_id to verify,
  whether to waive a red-flag loss, and whether to retry. For such a call, take
  the conservative default. Then fire
  `scripts/bash/notify.sh approval_required "<what you assumed>"`. Then proceed.
  Interactive: use AskUserQuestion, and bundle ≤ 4 questions.
- **Research-before-clarify:** you can discover the experiment number, the
  path_id, the vocab size, and the planned tokens in `experiments/`, ledger.md,
  and metrics.jsonl. Look before you ask.
- **Doom-loop guard:** after 3 identical tool calls that return no new
  information, write `blocker.md` in the experiment directory. Then fire
  `scripts/bash/notify.sh blocker "<summary>"`. Then stop.
- **Context discipline:** read metrics.jsonl and the logs with head, tail, or
  grep. Do not dump a whole file.

## Done conditions

- [ ] verify.md exists and contains an `OVERALL:` line.
- [ ] You appended a `JUDGMENT:` line to verify.md after you read
      logs/samples.jsonl.
- [ ] You updated the ledger `verify` column to `pass` or `fail` for the
      verified path. On a fail, you also set `failure_cause`.
- [ ] On a pass only: you wrote results.md, and `intern.py check` exits 0.
- [ ] On a fail: postmortems/path-<id>.md exists. You launched no retry unless
      `budget can-retry` exited 0.
