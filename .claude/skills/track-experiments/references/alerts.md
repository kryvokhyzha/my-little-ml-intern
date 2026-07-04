# Alerts reference

Alert levels, the canonical messages fired by `intern.callbacks`, the polling
commands, the wandb fallback, and how to write additional custom alerts.

## Levels and canonical messages (fired by intern.callbacks)

`src/intern/callbacks.py` ships `TRLAlertCallback` (transformers/TRL) and
`LightningAlertCallback` (Lightning). Both write every logged metric to
`metrics.jsonl` and fire alerts through `fire_alert(backend, level, message)`,
which routes to `trackio.alert` / `wandb.alert` / a loguru fallback depending on
`cfg.tracking.backend`. Every alert is ALSO appended to `metrics.jsonl` as an
event line, so the local record exists for all backends including `none`:

```json
{
  "ts": "...",
  "event": "alert",
  "level": "WARN",
  "message": "loss=9.8 at step 120 — lr likely too high, try lr*0.1"
}
```

Thresholds come from
`AlertRules(nan_streak=5, divergence_factor=3.0, plateau_evals=5)`.

| Condition                                                        | Level | Canonical message                                                                             | Built-in behavior                                           |
| ---------------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| NaN loss                                                         | ERROR | `loss=nan at step <N> — numerical instability, try skip step + halve lr`                      | Aborts the run after `nan_streak` (5) consecutive NaN steps |
| Divergence: loss > `divergence_factor` (3.0) × best loss so far  | WARN  | `loss=<v> at step <N> — lr likely too high, try lr*0.1`                                       | Run continues; you decide whether to kill it                |
| Plateau: no eval improvement for `plateau_evals` (5) evaluations | INFO  | `eval_loss=<v> at step <N> — eval loss plateaued for 5 evals, try early stopping or lr decay` | Run continues; counter resets after firing                  |

Every message follows the parseable format:

```
<metric>=<value> at step <N> — <hypothesis>, try <action>
```

Split on `—` to get `(observation, suggestion)`; split the observation on `=`
and `at step` to get metric, value, step. Never fire or accept an alert message
that a future call could not parse and act on.

Level semantics when deciding what to do:

- **ERROR** — stop and change approach (NaN, divergence past recovery, OOM).
- **WARN** — finish or kill the run, then tweak exactly one hyperparameter.
- **INFO** — milestone or soft signal; note it, keep watching.

## Trackio polling in detail

Record the launch timestamp once, then poll incrementally. Poll every few
minutes (2–5 min for short runs, 10–15 min for multi-hour runs) — never in a
tight loop, and never by tailing `logs/train.log`.

```bash
# Before launching the run:
SINCE="$(date -u +%Y-%m-%dT%H:%M:%S)"

# Each poll — empty "alerts": [] means healthy so far:
uv run trackio list alerts --project my-little-ml-intern --json --since "$SINCE"

# Narrow by run or severity:
uv run trackio list alerts --project my-little-ml-intern --run <run_name> --json
uv run trackio list alerts --project my-little-ml-intern --level error --json
```

Alert JSON items carry `run`, `title`, `text`, `level`, `step`, `timestamp` —
the canonical message is in `text`.

When an alert fires at step N, inspect the neighborhood before deciding:

```bash
# All metrics in a ±10-step window around the alert:
uv run trackio get snapshot --project my-little-ml-intern --run <run_name> --around <N> --window 10 --json

# One metric's trajectory around the alert:
uv run trackio get metric --project my-little-ml-intern --run <run_name> --metric loss --around <N> --window 20 --json
```

Discovery and comparison across runs:

```bash
uv run trackio list projects --json
uv run trackio list runs --project my-little-ml-intern --json
uv run trackio get run --project my-little-ml-intern --run <run_name> --json   # metrics list, config, last_step
uv run trackio get metric --project my-little-ml-intern --run <run_name> --metric eval_loss --json
```

`get run` returns the run's `config` — read the previous run's config from here
and mutate only the key the alert justifies changing.

## wandb fallback

`wandb.alert()` delivers to Slack/email via W&B notification settings; there is
no CLI or public-API endpoint that returns alert history. Two options, in order
of preference:

1. **Backend-independent (preferred):** read the alert events from the local
   `metrics.jsonl` — `intern.callbacks` writes them regardless of backend:

   ```bash
   grep '"event": "alert"' experiments/NNN-<slug>/metrics.jsonl | tail -10
   ```

2. **Run state and metrics via the API** (needs `WANDB_API_KEY` in `.env`):

   ```bash
   uv run python -c "import wandb; [print(r.name, r.id, r.state, r.summary.get('train/loss')) for r in wandb.Api().runs('<entity>/my-little-ml-intern')]"
   ```

   Then drill into one run by id:

   ```bash
   uv run python -c "import wandb; r = wandb.Api().run('<entity>/my-little-ml-intern/<run_id>'); print(r.state); print(dict(r.summary))"
   ```

## Writing additional custom alerts

The built-in callbacks cover NaN / divergence / plateau. Add task-specific
alerts (reward collapse, KL spike, grad-norm blowup, accuracy target reached)
directly in training code when the task calls for them.

Rules:

- One metric, one threshold per `if`. Simple conditions stay easy to adjust
  between runs.
- The message MUST carry a numeric value and an actionable suggestion in the
  canonical format — `<metric>=<value> at step <N> — <hypothesis>, try <action>`
  — so a future call can parse it and act without rereading the code.
- Prefer `intern.callbacks.fire_alert` over calling a backend directly: it
  respects `cfg.tracking.backend` and keeps `metrics.jsonl` in sync.

```python
from intern.callbacks import fire_alert

if grad_norm > 100.0:
    fire_alert(
        cfg.tracking.backend,
        "WARN",
        f"grad_norm={grad_norm:.1f} at step {step} — optimization unstable, try max_grad_norm=1.0",
    )
```

Under a `Trainer`/`SFTTrainer` you don't own the loop — add a small
`TrainerCallback` next to the built-in one and pass it via `callbacks=[...]`.
Training metrics (loss, reward, kl) arrive in `on_log`; eval metrics ONLY in
`on_evaluate`:

```python
from transformers import TrainerCallback
from intern.callbacks import fire_alert

class RewardCollapseAlert(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        margin = (logs or {}).get("rewards/margins")
        if margin is not None and margin < 0.0 and state.global_step > 100:
            fire_alert(
                "trackio",
                "WARN",
                f"rewards/margins={margin:.3f} at step {state.global_step} — chosen not preferred over rejected, try beta*2 or check pair labels",
            )
```

Calling a backend directly (custom loops outside the adapters only):

```python
import trackio
trackio.alert(
    title="grad_norm spike",
    text=f"grad_norm={gn:.1f} at step {step} — optimization unstable, try max_grad_norm=1.0",
    level=trackio.AlertLevel.WARN,   # INFO | WARN | ERROR
)

import wandb
wandb.alert(
    title="grad_norm spike",
    text=f"grad_norm={gn:.1f} at step {step} — optimization unstable, try max_grad_norm=1.0",
    level=wandb.AlertLevel.WARN,
)
```

If you add a custom alert, add its condition and canonical message to the
experiment's plan.md notes so the next iteration knows what can fire.

## Loss-spike triage (usual suspects)

Before mutating a hyperparameter, match the spike's signature:

| Signature                                 | Usual suspect → next move                                                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spike recurs at the same step/data region | Bad data batch — skip or filter that region (still one override)                                                                                              |
| fp16 lane                                 | Rerun the failing step in fp32 to split precision bugs from data bugs                                                                                         |
| `grad_norm` climbing before the spike     | Optimization instability — tighten `trainer.args.max_grad_norm`                                                                                               |
| Spike right after a resume                | Optimizer-state/data-order mismatch with the checkpoint — check resume plumbing before blaming the config                                                     |
| NaN with loss logged as 0.0               | Already covered by the grad_norm watch (adapters set `logging_nan_inf_filter=False`) — see the fp16 note in `.claude/skills/train-llm/references/hardware.md` |
