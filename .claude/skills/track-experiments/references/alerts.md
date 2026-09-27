# Alerts reference

This file covers the alert levels and the canonical messages that
`intern.callbacks` fires. It also covers the poll commands, the wandb fallback,
and the rules for additional custom alerts.

## Levels and canonical messages (fired by intern.callbacks)

`src/intern/callbacks.py` ships `TRLAlertCallback` (transformers/TRL) and
`LightningAlertCallback` (Lightning). Both callbacks write every logged metric
to `metrics.jsonl`. Both fire the alerts through
`fire_alert(backend, level, message)`, which routes to `trackio.alert`,
`wandb.alert`, or a loguru fallback. The value of `cfg.tracking.backend` selects
the route. Both callbacks ALSO append every alert to `metrics.jsonl` as an event
line. The local record therefore exists for every backend, `none` included:

```json
{
  "ts": "...",
  "event": "alert",
  "level": "WARN",
  "message": "loss=9.8 at step 120 — lr likely too high, try lr*0.1"
}
```

The thresholds come from
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

Split the message on `—` to get `(observation, suggestion)`. Split the
observation on `=` and `at step` to get the metric, the value, and the step.
Never fire an alert message that a future call cannot parse and act on. Never
accept such a message.

Use the level semantics when you decide what to do:

- **ERROR** — stop the run. Change the approach (NaN, divergence past recovery,
  OOM).
- **WARN** — finish the run, or kill it. Then change exactly one hyperparameter.
- **INFO** — a milestone or a soft signal. Note it. Continue to watch the run.

## Poll trackio in detail

Record the launch timestamp once. Then poll incrementally. Poll every few
minutes (2–5 min for short runs, 10–15 min for multi-hour runs). Never poll in a
tight loop. Never tail `logs/train.log`.

```bash
# Before launching the run:
SINCE="$(date -u +%Y-%m-%dT%H:%M:%S)"

# Each poll — empty "alerts": [] means healthy so far:
uv run trackio list alerts --project my-little-ml-intern --json --since "$SINCE"

# Narrow by run or severity:
uv run trackio list alerts --project my-little-ml-intern --run <run_name> --json
uv run trackio list alerts --project my-little-ml-intern --level error --json
```

Alert JSON items carry `run`, `title`, `text`, `level`, `step`, and `timestamp`.
The `text` field holds the canonical message.

When an alert fires at step N, inspect the neighborhood before you decide:

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

`get run` returns the run's `config`. Read the config of the previous run from
here. Then mutate only the key that the alert justifies.

## wandb fallback

`wandb.alert()` delivers to Slack or email through the W&B notification
settings. No CLI or public-API endpoint returns the alert history. You have two
options, in order of preference:

1. **Backend-independent (preferred):** read the alert events from the local
   `metrics.jsonl`. `intern.callbacks` writes them for every backend:

   ```bash
   grep '"event": "alert"' experiments/NNN-<slug>/metrics.jsonl | tail -10
   ```

2. **Run state and metrics via the API** (needs `WANDB_API_KEY` in `.env`):

   ```bash
   uv run python -c "import wandb; [print(r.name, r.id, r.state, r.summary.get('train/loss')) for r in wandb.Api().runs('<entity>/my-little-ml-intern')]"
   ```

   Then inspect one run by id:

   ```bash
   uv run python -c "import wandb; r = wandb.Api().run('<entity>/my-little-ml-intern/<run_id>'); print(r.state); print(dict(r.summary))"
   ```

## Write additional custom alerts

The built-in callbacks cover NaN, divergence, and plateau. Add a task-specific
alert directly in the training code when the task needs one. Examples: reward
collapse, KL spike, grad-norm blowup, accuracy target reached.

Rules:

- Use one metric and one threshold per `if`. A simple condition stays easy to
  adjust between runs.
- The message MUST carry a numeric value and an actionable suggestion. Use the
  canonical format —
  `<metric>=<value> at step <N> — <hypothesis>, try <action>`. A future call
  then parses the message and acts without a reread of the code.
- Prefer `intern.callbacks.fire_alert` over a direct call to a backend. It
  respects `cfg.tracking.backend`. It also keeps `metrics.jsonl` in sync.

```python
from intern.callbacks import fire_alert

if grad_norm > 100.0:
    fire_alert(
        cfg.tracking.backend,
        "WARN",
        f"grad_norm={grad_norm:.1f} at step {step} — optimization unstable, try max_grad_norm=1.0",
    )
```

Under a `Trainer` or an `SFTTrainer`, you do not own the loop. Add a small
`TrainerCallback` next to the built-in one. Pass it through `callbacks=[...]`.
The training metrics (loss, reward, kl) arrive in `on_log`. The eval metrics
arrive ONLY in `on_evaluate`:

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

Call a backend directly (in a custom loop outside the adapters only):

```python
import trackio

trackio.alert(
    title="grad_norm spike",
    text=f"grad_norm={gn:.1f} at step {step} — optimization unstable, try max_grad_norm=1.0",
    level=trackio.AlertLevel.WARN,  # INFO | WARN | ERROR
)

import wandb

wandb.alert(
    title="grad_norm spike",
    text=f"grad_norm={gn:.1f} at step {step} — optimization unstable, try max_grad_norm=1.0",
    level=wandb.AlertLevel.WARN,
)
```

If you add a custom alert, record its condition and its canonical message in the
experiment's plan.md notes. The next iteration then knows what can fire.

## Loss-spike triage (usual suspects)

Before you mutate a hyperparameter, match the signature of the spike:

| Signature                                 | Usual suspect → next move                                                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spike recurs at the same step/data region | Bad data batch — skip or filter that region (still one override)                                                                                              |
| fp16 lane                                 | Rerun the failing step in fp32 to split precision bugs from data bugs                                                                                         |
| `grad_norm` climbing before the spike     | Optimization instability — tighten `trainer.args.max_grad_norm`                                                                                               |
| Spike right after a resume                | Optimizer-state/data-order mismatch with the checkpoint — check resume plumbing before blaming the config                                                     |
| NaN with loss logged as 0.0               | Already covered by the grad_norm watch (adapters set `logging_nan_inf_filter=False`) — see the fp16 note in `.claude/skills/train-llm/references/hardware.md` |
