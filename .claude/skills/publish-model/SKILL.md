---
name: publish-model
description:
  Publish a verified training run to the Hugging Face Hub through the blocking
  publish gate — newest checkpoint plus the reproducibility bundle and a model
  card generated from results.md. Use when the user says "publish the model",
  "push to hub", "upload the checkpoint", or "ship it", and after any verified
  experiment whose weights need to leave this repo. The gate re-runs verify and
  refuses unverified runs; not published = not shipped.
---

# publish-model

Route a finished experiment through `intern.py publish`. The gate uploads the
files and writes the record. This skill only checks the preconditions, runs the
gate, reads the exit code, and confirms the upload. The gate semantics live in
`docs/001-architecture.md` ("Publish gate"). That doc wins on any ambiguity.

Do not write results.md unless `intern.py verify` exits 0. Do not report success
unless that same command exits 0. A failed gate means that the run failed. The
loss value does not change that result.

## Workflow

### 1. Locate the experiment and confirm a passed path

Research the answer before you ask the user. If this session just trained or
verified a run, use that NNN. Otherwise run `ls experiments/` and check the
dashboards. Then run this command:

```bash
uv run python scripts/python/intern.py status --experiment NNN
```

The ledger must show at least one row with `status=passed` and `verify=pass`. If
no such row exists, you cannot publish this run. Route back to **verify-run**,
or to **train-llm** if no run has trained yet. Then stop here.

### 2. Preconditions

- `HF_TOKEN` must exist in `.env` with **write** scope. Check only that the
  variable is present (`grep -c '^HF_TOKEN=' .env`). Never print the token.
- `experiments/NNN-<slug>/results.md` must exist. The verify.md file must carry
  a `JUDGMENT: generation_quality = PASS` line. A missing JUDGMENT line means
  that nobody finished verify-run. Run verify-run. Do not skip it.
- If the training data is private or proprietary, review `logs/samples.jsonl`
  before you publish. The generations can reproduce the training data, and the
  bundle ships them.
- Decide the optional flags: `--repo-id` (default
  `<HF_USER or whoami>/<project_name>-<experiment_name>`) and `--private`
  (default true). In a headless session with no stated preference, keep the
  defaults. Then fire
  `scripts/bash/notify.sh approval_required "publishing NNN as <repo-id>, private"`.

### 3. Run the publish gate

```bash
uv run python scripts/python/intern.py publish --experiment NNN [--repo-id org/name] [--private true|false]
```

The gate re-runs verify, which must exit 0. The gate also requires results.md
and the passed ledger row. The gate uploads the newest model directory under
`ckpts/` together with the reproducibility bundle. That bundle holds
task/plan/budget/ledger/verify/results.md, `logs/samples.jsonl`, and
`configs/NNN-<slug>.yaml`. The gate builds the model card from results.md. The
gate then appends `## Published` and the URL to results.md itself.

Exit codes:

- `0` — the gate published the run. The CLI already appended `## Published` to
  results.md. Do not append anything by hand.
- `1` — the gate refused the run. Either verify no longer passes, or the ledger
  holds no row that is both passed and verified. Never hand-edit ledger.md,
  verify.md, or results.md to make the gate pass. Fix the underlying run with
  **verify-run** or **train-llm**. If you cannot fix it, accept that the run
  stays unpublished and say so plainly.
- `2` — the gate found missing artifacts or credentials. The cause is a missing
  results.md, a missing checkpoint under `ckpts/`, or an HF_TOKEN that is absent
  or read-only. Fix the missing piece. Run the command again. Do not build the
  missing artifacts by hand.

### 4. Verify the upload

Confirm that the repo holds the files. The exit code is necessary, but it is not
sufficient. Run this command:

```bash
uv run hf models info <repo-id>
```

This call authenticates with `HF_TOKEN`, so it sees private repos. The anonymous
`https://huggingface.co/api/models` endpoint returns 401 or 404 for a private
repo. That result is EXPECTED, and it is NOT a publish failure. Never make a
repo public to force a check to pass.

Check that the model weights are present. Check that the reproducibility bundle
is present. Check that results.md now ends with `## Published` and the URL.
Record in your report whether you ran the load-check
(`AutoModelForCausalLM.from_pretrained("<repo-id>")`) or skipped it on purpose
because the download is large.

### 5. Notify — after publish, not before

An unpublished run is not a shipped run. When the user asks you to ship a run,
fire `train_done` only after the publish gate exits 0.

```bash
scripts/bash/notify.sh train_done "published <repo-id>" NNN-<slug>
```

If the gate refuses the run with exit 1 or exit 2, fire
`scripts/bash/notify.sh error "<one-line cause>"` instead. Never fire train_done
after a refusal.

## Posture

- Research-before-clarify: you can find the experiment number, the path_id, and
  the repo name inputs (`project_name`, `experiment_name`) in the configs and in
  `experiments/`. Look there before you ask.
- In a headless session, never hang. Keep the defaults, fire
  `notify.sh approval_required`, and proceed. In an interactive session, ask one
  AskUserQuestion with ≤ 4 bundled questions.
- Doom-loop guard: after 3 identical tool calls that return no new information,
  write `blocker.md` in the experiment directory. Then fire
  `scripts/bash/notify.sh blocker "<summary>"` and stop. A re-run of publish
  against the same refusal is one such call.
- Never print or echo `HF_TOKEN`. Never pass it as a CLI argument.

## Done conditions

- [ ] `status` showed a ledger row with `status=passed` and `verify=pass` before
      the publish attempt.
- [ ] The publish CLI exited 0, or you reported the refusal (exit 1 or exit 2)
      plainly. You made zero hand-edits to ledger.md, verify.md, or results.md.
- [ ] results.md contains `## Published` with the Hub URL. The CLI wrote that
      line, not you.
- [ ] You verified the upload. The repo listing shows the model files and the
      reproducibility bundle.
- [ ] `notify.sh train_done` fired only after exit 0. `error` fired on a
      refusal.
