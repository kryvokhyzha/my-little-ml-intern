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

Route a finished experiment through `intern.py publish` — the gate does the
uploading and the record-keeping; this skill only checks preconditions, runs it,
interprets the exit, and confirms the upload. The gate semantics live in
`docs/001-architecture.md` ("Publish gate") — that doc wins on any ambiguity.

Never write results.md or report success unless `intern.py verify` exited 0. A
failed gate means the run failed, regardless of loss.

## Workflow

### 1. Locate the experiment and confirm a passed path

Research before clarifying: just trained/verified in this session → that NNN;
otherwise `ls experiments/` and check the dashboards. Then:

```bash
uv run python scripts/python/intern.py status --experiment NNN
```

The ledger must show at least one row with `status=passed` and `verify=pass`. No
such row → this run cannot be published; route back to **verify-run** (or
**train-llm** if nothing has trained) and stop here.

### 2. Preconditions

- `HF_TOKEN` present in `.env` with **write** scope. Check presence only
  (`grep -c '^HF_TOKEN=' .env`) — never print the token.
- `experiments/NNN-<slug>/results.md` exists, and verify.md carries a
  `JUDGMENT: generation_quality = PASS` line — a missing JUDGMENT means
  verify-run was never finished; run it, don't skip it.
- Training data private/proprietary → review `logs/samples.jsonl` before
  publishing: generations can regurgitate training data, and they ship in the
  bundle.
- Optional flags decided: `--repo-id` (default
  `<HF_USER or whoami>/<project_name>-<experiment_name>`) and `--private`
  (default true). Headless with no stated preference → keep the defaults and
  fire
  `scripts/bash/notify.sh approval_required "publishing NNN as <repo-id>, private"`.

### 3. Run the publish gate

```bash
uv run python scripts/python/intern.py publish --experiment NNN [--repo-id org/name] [--private true|false]
```

The gate re-runs verify (must exit 0), requires results.md plus the passed
ledger row, uploads the newest `ckpts/` model dir plus the reproducibility
bundle (task/plan/budget/ledger/verify/results.md, `logs/samples.jsonl`,
`configs/NNN-<slug>.yaml`) with a model card generated from results.md, and
appends `## Published` with the URL to results.md itself.

Exit codes:

- `0` — published. The CLI already appended `## Published` to results.md; append
  nothing manually.
- `1` — gate refused (verify no longer passes, or no passed+verified ledger
  row). NEVER hand-edit ledger.md, verify.md, or results.md to appease the gate
  — fix the underlying run via **verify-run**/**train-llm**, or accept
  unpublished and say so plainly.
- `2` — missing artifacts or credentials (no results.md, no checkpoint under
  `ckpts/`, HF_TOKEN absent or read-only). Fix the missing piece, re-run the
  command. Do not hand-craft missing artifacts.

### 4. Verify the upload

Confirm the repo actually holds the files — the exit code is necessary, not
sufficient:

```bash
uv run hf models info <repo-id>
```

This call is authenticated via `HF_TOKEN`, so it sees private repos. A 401/404
from the anonymous `https://huggingface.co/api/models` endpoint is EXPECTED for
private repos and is NOT a publish failure — never flip a repo public to make a
check pass.

Check: model weights present, reproducibility bundle present, and results.md now
ends with `## Published` + the URL. Note in your report whether a load-check
(`AutoModelForCausalLM.from_pretrained("<repo-id>")`) was run or deliberately
skipped (large download).

### 5. Notify — after publish, not before

Not published = not shipped: for a shipping request, `train_done` fires only
once the publish gate exited 0.

```bash
scripts/bash/notify.sh train_done "published <repo-id>" NNN-<slug>
```

On refusal (exit 1/2) fire `scripts/bash/notify.sh error "<one-line cause>"`
instead — never train_done.

## Posture

- Research-before-clarify: experiment number, path_id, repo naming inputs
  (`project_name`, `experiment_name`) are all discoverable in configs and
  `experiments/` — look before asking.
- Headless: never hang — defaults + `notify.sh approval_required`, proceed.
  Interactive: one AskUserQuestion, ≤ 4 bundled questions.
- Doom-loop guard: 3 identical tool calls with no new information (e.g.
  re-running publish against the same refusal) → write `blocker.md` in the
  experiment dir, fire `scripts/bash/notify.sh blocker "<summary>"`, stop.
- Never print or echo `HF_TOKEN`; never pass it as a CLI argument.

## Done conditions

- [ ] `status` showed a ledger row with `status=passed` and `verify=pass` before
      the publish attempt.
- [ ] Publish CLI exit 0 — or the refusal (1/2) reported plainly, with zero
      hand-edits to ledger.md/verify.md/results.md.
- [ ] results.md contains `## Published` with the Hub URL, written by the CLI,
      not by hand.
- [ ] Upload verified: repo listing shows the model files and the
      reproducibility bundle.
- [ ] `notify.sh train_done` fired only after exit 0; `error` fired on refusal.
