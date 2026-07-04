---
name: track-experiments
description:
  Sets up experiment tracking and runs the alert-driven iteration loop for
  training runs in this repo (trackio primary, wandb secondary). Use whenever
  the user says "track the run", "set up wandb", "set up trackio", "check
  training progress", "read the metrics", "what do the alerts say", or "iterate
  on the experiment" — and proactively whenever a training run has started or
  finished and the next step depends on its metrics. Covers picking the Hydra
  tracking group, run naming, dashboard access, polling alerts without tailing
  logs, and mapping each alert to exactly one Hydra override for the next run.
---

# track-experiments

This SKILL.md is a router, not a manual. Instrumentation lives in the Hydra
`tracking` config group and `src/intern/callbacks.py`; this skill wires them up
and closes the loop from fired alerts to the next run's config. Detailed alert
semantics, polling commands, and the wandb fallback live in
`references/alerts.md` — read it when polling alerts, adding custom alerts to
training code, or working on the wandb backend.

## Ground rules

- Tracking is ALWAYS the Hydra `tracking` group (`trackio` | `wandb` | `none`) —
  never hardcoded in scripts or adapters. Code reads `cfg.tracking.backend`; if
  you find a hardcoded `report_to=` or `wandb.init(` in a training script, that
  is a bug to fix, not a pattern to copy.
- Never write results.md or report success unless `intern.py verify` exited 0. A
  failed gate means the run failed, regardless of loss.
- Research-before-clarify: never ask the user about anything you could look up —
  `ls configs/tracking/`, `tail metrics.jsonl`, and
  `uv run trackio list runs --project my-little-ml-intern --json` answer most
  questions before they need asking.
- Headless posture: never hang. Write best-guess defaults, fire
  `scripts/bash/notify.sh approval_required "<message>"`, proceed. Interactive
  mode: AskUserQuestion, at most 4 bundled questions.
- Doom-loop guard: 3 identical tool calls with no new information (e.g. polling
  alerts and getting the same empty list while the run has clearly stalled) →
  write `blocker.md` in the experiment dir, fire
  `scripts/bash/notify.sh blocker "<message>"`, stop.

## 1. Instrument the run

1. Pick the tracking group in the experiment config (`configs/NNN-<slug>.yaml`
   defaults list) or as a CLI override:
   - `tracking=trackio` — default. Local-first, no account or API key,
     CLI-queryable alerts. Optional HF Space sync via `tracking.space_id` — the
     adapters create the Space and its metrics bucket PRIVATE by default
     (`tracking.private`); a pre-existing public Space stays public — check
     before reusing one.
   - `tracking=wandb` — when the user already lives in W&B or needs team
     dashboards. Requires `WANDB_API_KEY` in `.env`; set `tracking.entity`
     explicitly for org work — project visibility is inherited from the entity's
     server-side default, so verify it creates private projects before the first
     run. Alerts are NOT CLI-readable — see `references/alerts.md`.
   - `tracking=none` — smoke tests and offline debugging only. Alerts still land
     in `metrics.jsonl` via the loguru fallback.
2. Run naming: `tracking.run_name` interpolates `${experiment_name}` (set by the
   experiment config, e.g. `001-tiny-sft`), and `tracking.project` interpolates
   `${project_name}` (`my-little-ml-intern`). Leave both alone for the first
   path. For retries and additional paths, override ONLY the run name —
   `tracking.run_name=001-tiny-sft-path-2` — never `experiment_name`, because
   `experiment_dir` derives from it and artifacts would scatter.
3. `tracking.group` (default null) clusters related runs on the dashboard — set
   it per sweep or autoresearch generation (e.g. `tracking.group=gen-2`);
   trackio groups natively, wandb receives it via `WANDB_RUN_GROUP`.
4. That is the whole job. The adapters (`src/training/trl_adapter.py`,
   `src/training/lightning_adapter.py`) attach the `intern.callbacks` alert
   callbacks and set `report_to` from `cfg.tracking.backend` automatically. Do
   not add `trackio.init()`/`wandb.init()` calls to experiment scripts.

## 2. Dashboard access

```bash
uv run trackio show --project my-little-ml-intern     # local dashboard
```

For wandb, the run URL is printed at launch — record it in the ledger (`run_url`
column). For a persistent trackio dashboard, set
`tracking.space_id=<user>/<space>` and metrics sync to an HF Space.

On remote boxes never bind the dashboard to all interfaces
(`GRADIO_SERVER_NAME=0.0.0.0`) — port-forward instead:
`ssh -L 7860:localhost:7860 <host>`, then open `localhost:7860` locally.

## 3. Monitoring etiquette (while a run is live)

- `experiments/NNN-<slug>/metrics.jsonl` is the local source of truth. Access it
  with head/tail/grep ONLY — never read the whole file:

  ```bash
  tail -3 experiments/NNN-<slug>/metrics.jsonl
  grep '"event": "alert"' experiments/NNN-<slug>/metrics.jsonl | tail -5
  ```

- Check run state through the tracker, not the process:

  ```bash
  uv run trackio list runs --project my-little-ml-intern --json
  uv run trackio get run --project my-little-ml-intern --run <run_name> --json
  uv run trackio get metric --project my-little-ml-intern --run <run_name> --metric loss --json
  ```

- NEVER tail an active training log (`logs/train.log`) in a loop — it floods
  context and tells you less than the alerts do. Poll alerts on an interval
  instead (commands and cadence in `references/alerts.md`). Record a `--since`
  timestamp right before launch so every poll is incremental.

## 4. The iteration loop (the core of this skill)

After every run — completed, killed, or interrupted by an ERROR alert:

1. **Read the alerts.**

   ```bash
   uv run trackio list alerts --project my-little-ml-intern --json --since <launch_ts>
   ```

   On the wandb backend there is no alert CLI — grep the `"event": "alert"`
   lines from `metrics.jsonl` (written for every backend) or use the wandb API
   one-liner in `references/alerts.md`.

2. **Parse them.** Every `intern.callbacks` alert message is machine-parseable:

   ```
   <metric>=<value> at step <N> — <hypothesis>, try <action>
   ```

   e.g. `loss=9.8 at step 120 — lr likely too high, try lr*0.1`. The `<action>`
   is a suggestion, not an order — check it against the mutation table below and
   the run's metrics before adopting it.

3. **Run the verify gate.**

   ```bash
   uv run python scripts/python/intern.py verify --experiment NNN
   ```

   Never write results.md or report success unless `intern.py verify` exited 0.
   A failed gate means the run failed, regardless of loss. Exit 0 with no
   actionable alerts → the experiment may be done; go to Done conditions.

4. **Map each alert to EXACTLY ONE Hydra override** (the one-variable rule — one
   changed variable per new run, or you cannot attribute the delta). Canonical
   mutation table:

   | Alert                                                                    | Level | The one override for the next run                                                                                                    |
   | ------------------------------------------------------------------------ | ----- | ------------------------------------------------------------------------------------------------------------------------------------ |
   | Diverged (loss > 3× best)                                                | WARN  | `trainer.args.learning_rate=<prev*0.1>`                                                                                              |
   | NaN loss                                                                 | ERROR | `trainer.args.learning_rate=<prev*0.5>` — and audit the data batch around the failing step first (diagnosis, not a second variable)  |
   | Overfitting (eval−train gap, from alert or `eval_train_gap` verify FAIL) | WARN  | raise `trainer.args.weight_decay`, raise dropout, or add data — pick ONE                                                             |
   | Plateau (no eval improvement for N evals)                                | INFO  | `trainer.args.lr_scheduler_type=cosine` (or another schedule) — or check the scale ceiling in budget.md before spending more compute |

   Multiple alerts → address the most severe first (ERROR > WARN > INFO); the
   rest usually follow from the same root cause.

5. **Record the override as a new hypothesis row in
   `experiments/NNN-<slug>/plan.md` BEFORE running** — with `mechanism`,
   `expected_delta`, and `falsification` per the plan.md contract in
   `docs/001-architecture.md`. No mutated run launches without its row.

6. **Pass the budget gate, update the ledger, relaunch.**

   ```bash
   uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1
   uv run python scripts/python/intern.py budget --experiment NNN record-retry
   uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-2 --status queued --retry-of path-1
   uv run python scripts/python/<NNN-slug>.py trainer.args.learning_rate=2e-5 tracking.run_name=NNN-<slug>-path-2 smoke_test=true
   uv run python scripts/python/<NNN-slug>.py trainer.args.learning_rate=2e-5 tracking.run_name=NNN-<slug>-path-2
   ```

   Nonzero exit from `can-retry` is a hard stop: report the denial, do not work
   around it. The smoke run is mandatory (train-llm step 4 ordering): it must
   print `VERDICT: TRAIN_OK` before the full relaunch — a failed or skipped
   smoke means no full run. Then loop back to step 1 for the new run.

## Done conditions

Do not report this workflow finished until every box checks:

- [ ] Tracking group and run_name are set in the experiment config or as
      recorded CLI overrides — no tracking code in scripts.
- [ ] Alerts for the finished run were read and quoted (or "no alerts" confirmed
      via the poll command).
- [ ] `intern.py verify` was run and its exit code reported.
- [ ] Every actionable alert maps to exactly one Hydra override, each logged as
      a hypothesis row in plan.md before its run launched.
- [ ] Ledger row for each run updated via `intern.py ledger upsert` (including
      `run_url` when the backend provides one).
- [ ] results.md exists only if `intern.py verify` exited 0.
