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

This SKILL.md is a router, not a manual. The instrumentation lives in the Hydra
`tracking` config group and in `src/intern/callbacks.py`. This skill connects
them. It also maps each fired alert to the config of the next run.
`references/alerts.md` holds the detailed alert semantics, the poll commands,
and the wandb fallback. Read that file when you poll alerts. Also read it when
you add custom alerts to the training code, or when you work on the wandb
backend.

## Ground rules

- The Hydra `tracking` group (`trackio` | `wandb` | `none`) ALWAYS selects the
  tracker. Never hardcode the tracker in a script or an adapter. The code reads
  `cfg.tracking.backend`. If you find a hardcoded `report_to=` or `wandb.init(`
  in a training script, fix that bug. Do not copy that pattern.
- Never write results.md unless `intern.py verify` exited 0. Never report
  success unless `intern.py verify` exited 0. A failed gate means the run
  failed, whatever the loss is.
- Research before you clarify. Never ask the user a question that you can answer
  yourself. Run `ls configs/tracking/`, `tail metrics.jsonl`, and
  `uv run trackio list runs --project my-little-ml-intern --json` first — they
  answer most questions.
- Headless posture: never wait for a reply. Write the best-guess defaults. Fire
  `scripts/bash/notify.sh approval_required "<message>"`. Then continue. In
  interactive mode, use AskUserQuestion with at most 4 bundled questions.
- Doom-loop guard: stop after 3 identical tool calls that give no new
  information. An example is a poll of the alerts that returns the same empty
  list while the run is stalled. Write `blocker.md` in the experiment directory.
  Fire `scripts/bash/notify.sh blocker "<message>"`. Then stop.

## 1. Instrument the run

1. Pick the tracking group in the experiment config (`configs/NNN-<slug>.yaml`
   defaults list), or as a CLI override:
   - `tracking=trackio` — the default. It runs local-first, needs no account or
     API key, and gives CLI-queryable alerts. `tracking.space_id` adds optional
     HF Space sync. The adapters create the Space and its metrics bucket PRIVATE
     by default (`tracking.private`). A Space that already exists and is public
     stays public. Check the visibility before you reuse a Space.
   - `tracking=wandb` — use it when the user already works in W&B, or when the
     user needs team dashboards. It requires `WANDB_API_KEY` in `.env`. Set
     `tracking.entity` explicitly for org work. The entity's server-side default
     sets the project visibility. Confirm that the entity creates private
     projects before the first run. The alerts are NOT CLI-readable — see
     `references/alerts.md`.
   - `tracking=none` — use it for smoke tests and offline debug work only. The
     loguru fallback still writes the alerts to `metrics.jsonl`.
2. Run names: `tracking.run_name` interpolates `${experiment_name}`, which the
   experiment config sets, for example `001-tiny-sft`. `tracking.project`
   interpolates `${project_name}` (`my-little-ml-intern`). Do not change either
   value for the first path. For a retry or an additional path, override ONLY
   the run name — `tracking.run_name=001-tiny-sft-path-2`. Never override
   `experiment_name`, because `experiment_dir` derives from it, and the
   artifacts would then go to more than one directory.
3. `tracking.group` (default null) clusters the related runs on the dashboard.
   Set it for each sweep or each autoresearch generation, for example
   `tracking.group=gen-2`. trackio groups the runs natively. wandb receives the
   group through `WANDB_RUN_GROUP`.
4. That is the whole job. The adapters (`src/training/trl/`,
   `src/training/lightning_adapter.py`) attach the `intern.callbacks` alert
   callbacks automatically. The adapters also set `report_to` from
   `cfg.tracking.backend`. Do not add a `trackio.init()` or `wandb.init()` call
   to an experiment script.

## 2. Dashboard access

```bash
uv run trackio show --project my-little-ml-intern     # local dashboard
```

For wandb, the launch prints the run URL. Record that URL in the ledger
(`run_url` column). For a persistent trackio dashboard, set
`tracking.space_id=<user>/<space>`. The metrics then sync to an HF Space.

On a remote box, never bind the dashboard to all interfaces
(`GRADIO_SERVER_NAME=0.0.0.0`). Forward the port instead with
`ssh -L 7860:localhost:7860 <host>`. Then open `localhost:7860` on the local
machine.

## 3. Monitor a live run

- `experiments/NNN-<slug>/metrics.jsonl` is the local source of truth. Read it
  with head, tail, or grep ONLY. Never read the whole file:

  ```bash
  tail -3 experiments/NNN-<slug>/metrics.jsonl
  grep '"event": "alert"' experiments/NNN-<slug>/metrics.jsonl | tail -5
  ```

- Check the run state through the tracker, not through the process:

  ```bash
  uv run trackio list runs --project my-little-ml-intern --json
  uv run trackio get run --project my-little-ml-intern --run <run_name> --json
  uv run trackio get metric --project my-little-ml-intern --run <run_name> --metric loss --json
  ```

- NEVER tail an active training log (`logs/train.log`) in a loop. The loop fills
  the context window, and it gives you less information than the alerts do. Poll
  the alerts on an interval instead. `references/alerts.md` holds the commands
  and the cadence. Record a `--since` timestamp immediately before the launch,
  so that every poll is incremental.

## 4. The iteration loop (the core of this skill)

Do these steps after every completed run, every killed run, and every run that
an ERROR alert interrupted:

1. **Read the alerts.**

   ```bash
   uv run trackio list alerts --project my-little-ml-intern --json --since <launch_ts>
   ```

   The wandb backend has no alert CLI. Grep the `"event": "alert"` lines from
   `metrics.jsonl`, which every backend writes. You can also use the wandb API
   one-liner in `references/alerts.md`.

2. **Parse them.** Every `intern.callbacks` alert message is machine-parseable:

   ```
   <metric>=<value> at step <N> — <hypothesis>, try <action>
   ```

   For example: `loss=9.8 at step 120 — lr likely too high, try lr*0.1`. The
   `<action>` is a suggestion, not an order. Check it against the mutation table
   below and against the run's metrics before you adopt it.

3. **Run the verify gate.**

   ```bash
   uv run python scripts/python/intern.py verify --experiment NNN
   ```

   Never write results.md unless `intern.py verify` exited 0. Never report
   success unless `intern.py verify` exited 0. A failed gate means the run
   failed, whatever the loss is. If the gate exits 0 and no alert is actionable,
   the experiment may be done. Go to the Done conditions.

4. **Map each alert to EXACTLY ONE Hydra override.** The one-variable rule
   allows one changed variable for each new run. With more than one change, you
   cannot attribute the delta. Canonical mutation table:

   | Alert                                                                    | Level | The one override for the next run                                                                                                         |
   | ------------------------------------------------------------------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------- |
   | Diverged (loss > 3× best)                                                | WARN  | `trainer.args.learning_rate=<prev*0.1>`                                                                                                   |
   | NaN loss                                                                 | ERROR | `trainer.args.learning_rate=<prev*0.5>` — first audit the data batch near the step that failed (this is diagnosis, not a second variable) |
   | Overfitting (eval−train gap, from alert or `eval_train_gap` verify FAIL) | WARN  | raise `trainer.args.weight_decay`, raise dropout, or add data — pick ONE                                                                  |
   | Plateau (no eval improvement for N evals)                                | INFO  | `trainer.args.lr_scheduler_type=cosine` (or another schedule) — or check the scale ceiling in budget.md before you spend more compute     |

   If more than one alert fires, address the most severe alert first (ERROR >
   WARN > INFO). The other alerts usually come from the same root cause.

5. **Record the override as a new hypothesis row in
   `experiments/NNN-<slug>/plan.md` BEFORE you launch the run.** The row needs
   `mechanism`, `expected_delta`, and `falsification`, per the plan.md contract
   in `docs/001-architecture.md`. Never launch a mutated run without its row.

6. **Pass the budget gate. Update the ledger. Relaunch.**

   ```bash
   uv run python scripts/python/intern.py budget --experiment NNN can-retry --path-id path-1
   uv run python scripts/python/intern.py budget --experiment NNN record-retry
   uv run python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-2 --status queued --retry-of path-1
   uv run python scripts/python/<NNN-slug>.py trainer.args.learning_rate=2e-5 tracking.run_name=NNN-<slug>-path-2 smoke_test=true
   uv run python scripts/python/<NNN-slug>.py trainer.args.learning_rate=2e-5 tracking.run_name=NNN-<slug>-path-2
   ```

   A nonzero exit from `can-retry` is a hard stop. Report the denial. Do not
   work around it. The smoke run is mandatory (train-llm step 4 order). The
   smoke run must print `VERDICT: TRAIN_OK` before the full relaunch. If the
   smoke run fails, or if you skip it, do not start the full run. Then go back
   to step 1 for the new run.

## Done conditions

Do not report this workflow complete until you can check every box:

- [ ] The experiment config sets the tracking group and run_name, or recorded
      CLI overrides set them. No script contains tracking code.
- [ ] You read and quoted the alerts of the finished run, or the poll command
      confirmed "no alerts".
- [ ] You ran `intern.py verify` and you reported its exit code.
- [ ] Every actionable alert maps to exactly one Hydra override. A hypothesis
      row in plan.md records each override before its run launched.
- [ ] `intern.py ledger upsert` updated the ledger row of each run, with
      `run_url` when the backend provides one.
- [ ] results.md exists only if `intern.py verify` exited 0.
