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

Run `intern.py verify` on an experiment, judge the generations yourself, and
route the outcome: pass → ledger + results.md; fail → postmortem + ledger, retry
only if the budget gate allows.

**A low loss number is never evidence the model works.**

Blocking-gate rule: Never write results.md or report success unless
`intern.py verify` exited 0. A failed gate means the run failed, regardless of
loss.

## Workflow

1. **Locate the experiment (NNN).** Research before clarifying — never ask the
   user what you can look up:

   - Just trained in this session → use that experiment number.
   - Otherwise: `ls experiments/`, prefer the dir whose `metrics.jsonl` is
     newest or whose ledger has a `running`/`queued` row
     (`uv run python scripts/python/intern.py ledger --experiment NNN show`).
   - Note the `path_id` of the row being verified — you need it in step 5.
   - `999-` scratch experiments are exempt from blocking gates; verify anyway
     when asked, but the results.md prohibition does not apply to them.

2. **Run the gate.**

   ```
   uv run python scripts/python/intern.py verify --experiment NNN
   ```

   Capture the exit code. Options:

   - `--vocab-size N` — only when the effective tokenizer vocab differs from the
     `vocab_size` meta in metrics.jsonl (resized embeddings, swapped tokenizer);
     otherwise omit.
   - `--checks a,b` — scope to specific checks by name (comma-separated) when
     re-verifying a single fixed check; default is all applicable checks. A
     scoped run only prints its report — it never writes or overwrites
     verify.md; only a full (unscoped) run does.

   metrics.jsonl accumulates across paths/retries — verify scopes itself to the
   last `run_start` event, so do NOT delete metrics.jsonl between retries.

3. **Read the report** at `experiments/NNN-<slug>/verify.md`. For each `FAIL`
   line, state in one line what the check measures and why this run failed it.
   Default checks (thresholds and exact semantics: read
   `docs/001-architecture.md` section "verify.py" when a name is unfamiliar):

   | check                      | one-line meaning                                                                   |
   | -------------------------- | ---------------------------------------------------------------------------------- |
   | `loss_plausibility`        | final train loss inside the ln(vocab) band; < 1.0 on an LM task is a red-flag FAIL |
   | `eval_train_gap`           | eval and train loss within 0.5 of each other                                       |
   | `data_consumption`         | model actually saw ≥ 70% of planned tokens                                         |
   | `stderr_scan`              | no Traceback / RuntimeError / CUDA OOM in logs/stderr.log                          |
   | `param_drift`              | actual param count within 15% of target                                            |
   | `generation_sanity`        | samples.jsonl exists, not degenerate (mechanical proxies only)                     |
   | `reward_margin` / `kl_ref` | DPO-only: positive reward margin, finite KL                                        |

4. **MANDATORY human-judgment step — even on mechanical PASS.** Read
   `experiments/NNN-<slug>/logs/samples.jsonl` and judge whether the generations
   are recognizable language for the training distribution (a TinyStories model
   should produce story-like English; a code model, code-like text). Word salad
   with valid vocabulary is a fail, not a partial pass. Append your judgment as
   one line to verify.md:

   ```
   JUDGMENT: generation_quality = PASS|FAIL | <one-line reasoning against the training distribution>
   ```

   A FAIL judgment fails the run overall even when the exit code was 0.

5. **Route the outcome.**

   **Overall pass** (exit 0 AND judgment PASS):

   ```
   uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status passed --verify pass
   ```

   Only after the ledger update, write `experiments/NNN-<slug>/results.md`
   (winner + comparison per the experiment convention).

   **Fail** (exit 1, or judgment FAIL):

   - Write `experiments/NNN-<slug>/postmortems/path-<id>.md` with exactly:
     symptom → root-cause hypothesis → fix.
   - Update the ledger:

     ```
     uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status failed --verify fail --failure-cause "<one line>"
     ```

   - Before ANY retry, check the budget gate and stop if it denies:

     ```
     uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1
     ```

     Nonzero exit = no retry; report the postmortem and stop. Never fire
     `notify.sh train_done` for a run with no passing path.

6. **Interpret exit codes plainly.**
   - `0` — all checks passed. Step 4 still applies before any success claim.
   - `1` — at least one check FAILED. The run FAILED regardless of loss — say so
     in those words. No "mostly passed", no "partial success".
   - `2` — missing artifacts: training did not produce metrics.jsonl (or the
     experiment dir is wrong). That is a pipeline bug, not "done" — fix the
     training script / callback wiring, then rerun training; do not hand-craft
     the missing files to appease the gate.

## Posture

- **Headless:** never hang. Ambiguous call (which path_id, waive a red-flag
  loss, retry or not) → take the conservative default, fire
  `scripts/bash/notify.sh approval_required "<what you assumed>"`, proceed.
  Interactive: AskUserQuestion, ≤ 4 bundled questions.
- **Research-before-clarify:** experiment number, path_id, vocab size, and
  planned tokens are all discoverable in `experiments/`, ledger.md, and
  metrics.jsonl. Look before asking.
- **Doom-loop guard:** 3 identical tool calls with no new information → write
  `blocker.md` in the experiment dir, fire
  `scripts/bash/notify.sh blocker "<summary>"`, stop.
- **Context discipline:** read metrics.jsonl and logs with head/tail/grep, not
  whole-file dumps.

## Done conditions

- [ ] verify.md exists and contains an `OVERALL:` line.
- [ ] `JUDGMENT:` line appended to verify.md after actually reading
      logs/samples.jsonl.
- [ ] Ledger `verify` column updated to `pass` or `fail` for the verified path
      (with `failure_cause` on fail).
- [ ] On pass only: results.md written.
- [ ] On fail: postmortems/path-<id>.md exists; no retry launched without
      `budget can-retry` exiting 0.
