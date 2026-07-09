# 001 — Architecture contract

<!-- Single source of truth for module APIs, artifact formats, and skill conventions.
Implementation must match this document exactly; change the document first, then the code. -->

## Positioning

`my-little-ml-intern` is a template-conforming reference project and an
installable pack: `.claude/skills/` + `src/intern/` are designed to be vendored
into other projects. The agent brain is Claude Code's own loop — skills
instruct, and the `src/intern` library **enforces**: verification, budget, and
dependency-age gates are Python with nonzero exit codes. Skills must refuse to
proceed when a gate fails.

## Security defaults

Private-by-default is a repo-wide rule: the publish gate creates private repos
unless explicitly overridden, the trackio Space and its metrics bucket are
forced private, and hf_jobs pushes to the Hub set `hub_private_repo=True`.
Key-only SSH guidance lives in the train-llm compute-lanes reference; secrets
enter only via `.env`, never configs or code.

## Experiment convention

One experiment number = three artifacts:

```
scripts/python/NNN-<slug>.py      # hydra entrypoint (new-script scaffold)
configs/NNN-<slug>.yaml           # composes main + trainer/tracking/compute/budget groups
experiments/NNN-<slug>/           # run artifacts (below)
```

`999-` prefix = gitignored scratch, exempt from blocking gates. Promotion =
rename to next free `NNN` across all three locations.

```
experiments/NNN-<slug>/
├── task.md          # restated task, unknowns, run mode (interactive|headless)
├── plan.md          # hypotheses + solution paths (see plan contract)
├── budget.md        # caps + spent tally (parsed by intern.budget)
├── ledger.md        # per-path ledger table (managed by intern.ledger)
├── run.md           # exact commands run on compute (train/benchmark); ungated, filled by train-llm
├── data.md          # optional: dataset(s) used + how they were prepared (text + mermaid)
├── research.md      # optional recipe table from literature-recipe-research
├── board.md         # autoresearch shared board (see board.md format)
├── metrics.jsonl    # gitignored: append-only metric/event stream (intern.callbacks)
├── verify.md        # written ONLY by intern.verify
├── results.md       # winner + comparison; forbidden unless verify exit 0
├── postmortems/     # path-<id>.md: symptom → root-cause hypothesis → fix
├── traces/          # gitignored: agent rollout JSONL for distillation
├── ckpts/           # gitignored
└── logs/            # gitignored: train.log, stderr.log, samples.jsonl
```

### plan.md contract

Every hypothesis must carry: `mechanism` (causal path), `expected_delta`
(numeric), `falsification` (what result kills it). Each solution path is one
variable — prefer one Hydra override per path. No production file may be edited
before the change is a named hypothesis.

Under the autoresearch loop, plan.md MAY additionally carry a `## Loop` section
(generation counter, angle coverage) and a `## Next levers` section (candidate
ideas for the next generation).

### board.md format

The autoresearch shared board, written by the `autoresearch-loop` skill:

```text
# Board — NNN-<slug>
## Champion
<path_id> | <metric>=<value> | override: <hydra override>
## Verified wins
- <path_id>: <one line>
## Dead ends
- <path_id>: <one line + why>
## Next levers
- rung 1|2|3: <idea>
## Stagnation log
- gen <N>: <no-improvement note>
```

Only a path with ledger `verify=pass` may hold `## Champion`. Refuted mechanisms
(delta real but the causal story wrong, or no delta) go to `## Dead ends` with
the why.

### budget.md format

```text
# Budget

max_paths: 2
max_retries_per_path: 2
compute_cap_gpu_h: 2.0
scale_ceiling_params: 200000000
token_budget: null

## Spent

paths_launched: 0
retries_used: 0
gpu_h_used: 0.0
```

`key: value` lines; the `## Spent` section is the tally the orchestrator updates
via the CLI. Caps stop new paths/retries; they never license quitting work that
is already budgeted.

### ledger.md format

Markdown table, one row per path, columns exactly:

```
| path_id | approach | status | final_train_loss | final_eval_loss | verify | failure_cause | retry_of | gpu_min | run_url |
```

`status ∈ queued|running|passed|failed|dropped`,
`verify ∈ pass|fail|pending|n/a`.

### run.md format

The copy-pasteable record of the exact commands executed **on the compute
instance** — the reproducibility spine between `plan.md` (what to try) and
`results.md` (what happened). Written by the `train-llm` skill as the run
proceeds; **ungated** — recorded even when a run fails, because reproducing a
failure needs its commands too. Placeholders in `<angle brackets>`; never inline
a token or account-specific secret (use `<PROJECT>`, `<HOST>`, `.env`).

**Self-contained.** A reader reproduces this experiment from this one file — no
"see 004 for the command" cross-references to other experiments. Embed every
step, including one-time local prerequisites like dataset materialization (the
exact `scripts/python/prep-*.py` invocation) even though they run off the
compute box. Referencing this repo's own scripts/configs by path is fine;
pointing at another experiment's artifacts is not.

```text
# Run — NNN-<slug>

## Lane

compute=<local|ssh|hf_jobs|modal|vast> · instance: <e.g. 1x NVIDIA L4 24GB> · region: <if remote>

## Prerequisite   (one-time, local; omit if none)

<dataset materialization, e.g. `uv run python scripts/python/prep-<slug>.py` — the exact command, not a pointer>

## Provision   (remote lanes only; omit for local)

<instance create, IP-locked firewall, key-only SSH — mirrors compute-lanes.md>

## Setup

<uv sync [--group gpu], gpu_probe>

## Train

<smoke_test=true, budget can-launch, the real training invocation(s), record-gpu-h, verify>

## Benchmark   (if a separate eval/benchmark ran; else note the inline eval)

<eval-harness / benchmark commands, or "held-out eval + generation samples, run inline by the training script">

## Teardown   (remote lanes only; omit for local)

<instance delete, firewall/key removal, confirm no orphaned disks>
```

Sections that do not apply (Provision/Teardown on `local`) are omitted, not left
empty. Bundled into the published model's `bundle/` by the publish gate.

### data.md format

**Optional** data-provenance card — write one when the experiment's dataset
needed preparation (conversion, filtering, mixing, materialization) rather than
being consumed as-is off the Hub. It answers "what did this train on and how was
it built" so the data is auditable and rebuildable. Free-form Markdown; a
`mermaid` flowchart of the prep pipeline is encouraged. Cover: **source(s)**
(upstream id + license/access), **target** (the materialized dataset id +
private/public), the **pipeline** (each transform → the `data.*`/prep function
that does it), the **format** (columns + loss shape), **stats** (row counts in →
kept → split; what was dropped and why), the **key knobs** (`max_length`,
`eval_size`, seed, …), and the exact **reproduce** command. Bundled into the
published model's `bundle/` by the publish gate. Datasets consumed unchanged
need no data.md — the `configs/data/<name>.yaml` node is the whole story.

### metrics.jsonl schema

One JSON object per line, two record kinds:

```json
{"ts": "2026-07-04T12:00:00+00:00", "step": 10, "split": "train", "name": "loss", "value": 2.31}
{"ts": "...", "event": "alert", "level": "WARN", "message": "loss=9.8 at step 120 — lr likely too high, try lr*0.1"}
{"ts": "...", "event": "meta", "key": "param_count", "value": 124000000}
```

Required meta keys written by adapters: `task` (the trainer `kind`),
`param_count`, `vocab_size`; when known: `planned_tokens`. Optional meta written
after trainer construction: `trainable_param_count` (post-peft `requires_grad`
numel) and `quantized` (bool, set by the quantization path). Optional
`completion_only` (bool, set when `SFTConfig.completion_only_loss` is truthy —
signals the train loss is over assistant-target tokens only, not corpus
cross-entropy). Token consumption is the `num_input_tokens_seen` metric series.
Alert messages follow
`<metric>=<value> at step <N> — <hypothesis>, try <action>`.

**Run boundary:** adapters append a `run_start` event (fields: `task`,
`run_name`, `smoke`) as their FIRST action, before anything that can crash.
metrics.jsonl accumulates across paths/retries; verify scopes every check to
records at or after the last `run_start` (whole file when none exists).
`logs/stderr.log` is truncated per run by the adapter's tee.

### verify.md format

Machine-greppable, one line per check:

```
VERDICT: loss_plausibility = PASS | value=2.31 | threshold=(1.04, 10.4) | final train loss inside ln(vocab) band
VERDICT: eval_train_gap = SKIP | value=n/a | threshold=0.5 | no eval metrics logged
...
OVERALL: PASS (5 passed, 0 failed, 2 skipped)
```

## src/intern API

### metrics.py

- `class MetricsLog(path: Path)` —
  `append_metric(step, name, value, split="train")`,
  `append_event(event, **fields)`, `read() -> list[dict]`,
  `final(name, split=None)`,
  `series(name, split=None) -> list[tuple[int, float]]`, `meta(key)`.

### verify.py

- `@dataclass CheckResult`: `name`, `status ∈ PASS|FAIL|SKIP`, `value`,
  `threshold`, `detail`.
- `class RunVerifier(experiment_dir: Path, vocab_size: int | None = None)` —
  `run(checks: list[str] | None = None) -> list[CheckResult]`,
  `write_report(results) -> Path`, `passed(results) -> bool`.
- `verify_run(experiment_dir, **kw) -> int` — 0 all pass, 1 any FAIL, 2 missing
  artifacts (metrics.jsonl absent, or a `run_start` exists with no metric
  records after it — a run that crashed before logging is a pipeline bug, not
  "done"). A full run writes verify.md; a scoped `checks=[...]` run only logs
  its report and never overwrites verify.md.

Checks are task-aware via the `task` meta (`trl_sft` and unknown count as LM
tasks) and scoped to the last `run_start`. Defaults (SKIP when inputs absent
unless stated):

1. `loss_plausibility` — LM tasks only (SKIP for `trl_dpo`/`trl_grpo`/
   `lightning` — their loss is not vocab cross-entropy): final train loss ∈
   (0.1·ln(V), ln(V)), V = `vocab_size`; loss < 1.0 is a red-flag FAIL even when
   V is unknown. SKIPs (before the task/vocab logic) when the `completion_only`
   meta is truthy — completion-only SFT loss is over assistant-target tokens
   only, so the band and the <1.0 red-flag do not apply; lean on
   `eval_train_gap`, `generation_sanity`, and held-out eval instead.
2. `eval_train_gap` — |final eval loss − final train loss| < 0.5; eval loss is
   `loss(split=eval)`, `eval_loss`, or `val_loss` (Lightning).
3. `data_consumption` — final `num_input_tokens_seen` ≥ 0.7 × `planned_tokens`.
4. `stderr_scan` — grep `logs/stderr.log`: FAIL on
   `Traceback|RuntimeError|CUDA out of memory`, listed-but-PASS on warnings;
   SKIP when file absent.
5. `param_drift` — opt-in: SKIP unless the experiment sets `model.target_params`
   (omitted from shipped configs). When set: |`param_count` − `target_params`| ≤
   15%. LoRA runs compare base-model numel (`param_count` is measured pre-peft,
   `target_params` stays the base size). Also SKIP when the `quantized` meta is
   truthy — 4-bit storage breaks numel comparability.
6. `generation_sanity` — `logs/samples.jsonl` mechanical proxies: unique-token
   ratio ≥ 0.3 (character-level fallback for non-space-delimited scripts), no
   token > 50% of output, each sample ≥ 50 chars. File absent → FAIL for LM
   tasks, SKIP otherwise; the agent must ALSO eyeball samples — mechanical pass
   is necessary, not sufficient.
7. `reward_margin` (auto when DPO metrics present) — final
   `rewards/margins` > 0.
8. `kl_ref` (auto when present) — mean KL over the current run finite and > 0.
9. `reward_variance` (auto for GRPO runs) — current-run scope: when a
   `reward_std` series exists, FAIL when every value == 0 ("no reward variance —
   the run optimized nothing"), PASS otherwise (value = final `reward_std`);
   else when a `reward` series exists, FAIL when its population std is 0
   (constant reward), PASS otherwise; SKIP when no GRPO reward metrics are
   logged.

A low loss number is never evidence the model works.

### budget.py

- `@dataclass Budget` — caps + spent fields mirroring budget.md.
- `load_budget(path) -> Budget`, `save_budget(budget, path)`.
- `load_budget_profile(configs_dir, profile) -> Budget` — build a fresh Budget
  (spend zeroed) from `configs/budget/<profile>.yaml`; backs `budget init`.
- `class BudgetGate(path)` — `can_launch_path(params=None) -> tuple[bool, str]`
  (denies when `params` exceeds `scale_ceiling_params`),
  `can_retry(path_id, ledger) -> tuple[bool, str]`, `record_launch()`,
  `record_retry()`, `record_gpu_h(hours)`.
- Retry accounting counts the whole retry TREE rooted at the chain's first path
  — a chain A→B→C does not reset the counter at each link — plus a global
  backstop: deny when `retries_used ≥ max_paths × max_retries_per_path`.
  `token_budget` is advisory for the orchestrating agent, not machine-enforced.

### ledger.py

- `class Ledger(path)` — `upsert(path_id, **fields)`, `rows() -> list[dict]`,
  `write()`. Unknown columns rejected; enums validated.

### callbacks.py

- `class AlertRules(nan_streak=5, divergence_factor=3.0, plateau_evals=5)`.
- `fire_alert(backend, level, message)` — `trackio.alert` / `wandb.alert` /
  loguru fallback.
- `class TRLAlertCallback(transformers.TrainerCallback)` and
  `class LightningAlertCallback(lightning.pytorch.Callback)` — both write every
  logged metric to `MetricsLog` and fire alerts: NaN loss OR non-finite
  `grad_norm` → ERROR ("skip step
  - halve lr"; abort after `nan_streak` — transformers' NaN filter can log a NaN
    loss as 0.0, so grad_norm is watched too and adapters set
    `logging_nan_inf_filter=False`), divergence (loss > factor × best) → WARN
    (suggest lr×0.1), plateau → INFO.

### traces.py

- `@dataclass TraceRecord` — `task_id: str`, `split: str` (`train`|`eval`),
  `messages: list[dict]`, `tools: list[dict] | None`, `model_id: str | None`,
  `gen_params: dict | None`, `steps: list[dict] | None`
  (`[{action, observation}]`), `terminal_status: str | None`,
  `verifier_output: dict | None`, `judge_critique: str | None`,
  `reward_components: dict | None`, `accepted: bool`.
- `class TraceStore(path)` — `append(record)`, `read() -> list[TraceRecord]`,
  `accepted() -> list[TraceRecord]` (filtered to `accepted=True`).
- Converters:
  `to_sft_messages(records, only_accepted=True) -> list[{"messages", "tools"}]`
  and
  `to_prompt_completion(records, tokenizer, only_accepted=True) -> list[{"prompt", "completion"}]`.
  Prompt/completion renders per final-assistant-turn:
  `prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False)`
  and `full = apply_chat_template(context + [assistant], tokenize=False)`;
  unless `full.startswith(prompt)` raise ValueError naming the chat-template
  prefix mismatch — this catches silently-corrupted completion-loss boundaries.
- Traces live in `experiments/NNN-<slug>/traces/*.jsonl` (gitignored).

### deps.py

- `latest_eligible(package, min_age_days=7) -> tuple[str, date]` — newest
  non-prerelease PyPI version whose earliest upload is ≥ `min_age_days` old.
- `check_project(pyproject_path, min_age_days=7) -> list[str]` — violation when
  a declared floor/pin version is younger than `min_age_days`; informational
  when a newer eligible version exists. Nonzero exit on violations.

## src/training API

Layout: `training/trl/` is the TRL subpackage (`rewards.py`, `config.py`,
`run.py`; `run_sft`/`run_dpo`/`run_grpo` re-exported from `training.trl`).
Framework-neutral plumbing is shared, not owned by a lane: `training/runtime.py`
(stderr tee, `is_main_process`, `smoke_enabled`, `apply_tracking_env` — maps
`tracking.project`/`tracking.group` onto `WANDB_PROJECT`/`WANDB_RUN_GROUP` for
the wandb callback; trackio takes `TrainingArguments.project` via `build_args`),
`training/models.py` (`load_model` with dtype/quant, `peft_config`),
`training/sampling.py` (`SAMPLE_PROMPTS`, `write_samples`). Dataset loading
lives in `data/` (`data.loading.load_split` / `validate_columns`,
`data.synthetic.build_tiny_text_dataset`). The `lightning_adapter` imports the
shared modules directly — never TRL internals.

- `training.trl.run_sft(cfg) -> dict` / `run_dpo(cfg) -> dict` — instantiate
  model/tokenizer from the `cfg.model` group (`training.models.load_model(cfg)`
  / `load_tokenizer(cfg)` wrap `hydra.utils.instantiate` of `model.main` /
  `model.tokenizer`; when `main` carries neither `dtype` nor
  `quantization_config` the loader injects `dtype: float32` and warns —
  transformers v5 otherwise inherits the checkpoint's stored dtype, and fp16
  full-precision training diverges; the tokenizer wrapper applies the pad→eos
  fallback), instantiate the datasets from the `cfg.data` group's `train`/`eval`
  `_target_` nodes (`for_eval=True` is injected for the `load_split` target so a
  plain on-disk Dataset can never silently eval on training data) and fail fast
  via `data.loading.validate_columns` when the columns cannot feed the task,
  build the trainer args by instantiating the `cfg.trainer.args` node
  (`_target_: trl.SFTConfig`/`DPOConfig`/`GRPOConfig`) via `build_args`, which
  layers the runtime values on top (`report_to` from `cfg.tracking.backend`,
  run*name/project, hardware-aware bf16, smoke overrides, the alert-callback
  flags), and the LoRA adapter by instantiating `cfg.trainer.peft` (`\_target*:
  peft.LoraConfig`) via `peft_config`, attach `TRLAlertCallback`, set `report_to`from`cfg.tracking.backend`, `include_num_input_tokens_seen=True`, write `param_count`/`vocab_size`meta, tee stderr to`logs/stderr.log`, generate 3 sampled continuations (seeded sampling; probe prompts from `resolve_sample_prompts`— held-out`prompt`column for prompt/completion data, else defaults) to`logs/samples.jsonl`(one`{"prompt",
  "text"}`object per line) after training, print one line:`VERDICT: TRAIN_OK |
  final_train_loss=<v>`(or`TRAIN_FAIL | <cause>`).
- `training.trl.run_grpo(cfg) -> dict` — mirrors `run_sft`/`run_dpo` via
  `GRPOConfig`/`GRPOTrainer`. The dataset must contain a `prompt` column
  (ValueError naming the column contract otherwise). Reward functions come from
  `cfg.trainer.reward_funcs`: a list of dotted import paths
  (`package.module:function` or `module.path.function`) resolved via importlib
  at run time; an empty list raises ValueError ("GRPO needs at least one reward
  function"); each callable must tolerate malformed completions — return the
  floor reward, never raise. `TRLAlertCallback` attached, meta written
  (`task=trl_grpo`), samples still generated post-run (the policy is an LM),
  same rank-zero + stderr-tee + `VERDICT` contract.
- Quantization: a `quantization_config:` nested
  `_target_: transformers.BitsAndBytesConfig` node inside `model.main` (use a
  dedicated `<name>_4bit.yaml` model variant); requires the `gpu` dependency
  group — missing bitsandbytes raises a clear error naming
  `uv sync --group gpu`. The DPO `ref` model (when set) shares the same load
  path. Meta: `target_params` comes from `cfg.model.target_params` and the
  `quantized` meta is emitted when `model.main` carries a `quantization_config`.
- Tracking wiring: when `cfg.tracking.space_id` is set, adapters pass
  `trackio_space_id` and `hub_private_repo` (from `cfg.tracking.private`,
  default true) so the trackio Space and its metrics dataset bucket are created
  private by default.
- Rank-zero rule: instrumentation writes — metrics.jsonl (`run_start` + meta),
  `logs/samples.jsonl`, and the final `VERDICT` print — happen only on the main
  process (`RANK` env, unset counts as 0); `TRLAlertCallback` acts only when
  `state.is_world_process_zero`.
- Smoke gate (all TRL lanes): when `cfg.smoke_test` or env `SMOKE_TEST=1` —
  `max_steps=1`, dataset sliced to ≤ 32 rows, no checkpoint save. Mandatory
  before any long run.
- `lightning_adapter.run(cfg) -> dict` — `hydra.utils.instantiate`
  module/datamodule and the `cfg.trainer.args` node
  (`_target_: lightning.pytorch.Trainer`), injecting `logger`/`callbacks`/
  `enable_progress_bar` + the smoke overrides (the lightning-lane mirror of
  `build_args`).
- `axolotl_adapter.render(cfg) -> Path` — merge `base_config` YAML + `overrides`
  → `rendered_path`, return path; print launch command for the remote box
  (`uv run --with axolotl axolotl train <path>`). axolotl is never a local
  dependency.

## CLI surface

`scripts/python/intern.py` (fire, rootutils preamble; `experiment` accepts `001`
or full `001-slug`):

```
uv run python scripts/python/intern.py verify --experiment 001 [--vocab-size N] [--checks a,b]
uv run python scripts/python/intern.py budget --experiment 001 init --profile lora [--force]
uv run python scripts/python/intern.py budget --experiment 001 status|can-launch|record-launch|record-retry
uv run python scripts/python/intern.py budget --experiment 001 can-launch --params 135000000
uv run python scripts/python/intern.py budget --experiment 001 can-retry --path-id path-1
uv run python scripts/python/intern.py budget --experiment 001 record-gpu-h --hours 0.5
uv run python scripts/python/intern.py ledger --experiment 001 upsert --path-id path-1 --status running ...
uv run python scripts/python/intern.py ledger --experiment 001 show
uv run python scripts/python/intern.py status --experiment 001 [--json]
uv run python scripts/python/intern.py publish --experiment 001 [--repo-id org/name] [--private true|false]
uv run python scripts/python/intern.py deps [--min-age-days 7]
```

Exit codes: 0 ok/allowed, 1 gate failed/denied, 2 usage or missing artifacts.
These exits are the blocking mechanism — skills must stop on nonzero.

`status` is the gates dashboard for one experiment: verify verdicts, budget
caps/spent, and ledger rows; exit 0 (2 when the experiment is missing).

### Publish gate

`publish` is the BLOCKING publish gate. It re-runs verify (must exit 0) and
requires results.md plus a ledger row with `status=passed` and `verify=pass`. It
uploads the newest `ckpts/` model dir plus the reproducibility bundle —
task/plan/budget/ledger/verify/results.md, `logs/samples.jsonl`,
`configs/NNN-<slug>.yaml` — to the HF Hub with a model card generated from
results.md, then appends `## Published` with the URL to results.md. Exit 0
published, 1 gate refused, 2 missing artifacts/credentials.

API: `src/intern/publish.py` —
`publish_run(experiment_dir, repo_id=None, private=True) -> int`. Repo id
defaults to `<HF_USER or whoami>/<project_name>-<experiment_name>`.

`--private` accepts ONLY a bool or the exact strings `true`/`false`
(case-insensitive); anything else — including fire-parsed ints — exits 2. A
resolved `false` logs a loud warning that the repo will be world-readable.
Before upload, the bundle scrub scans staged text files for token-shaped strings
(`hf_`, `sk-`, `xox[abp]-`, `AKIA`, `ghp_` prefixes) and the CURRENT machine's
home path (`Path.home()` — an environment leak, not any `/Users/x/` which model
generations legitimately echo from training data); a hit logs the file and
pattern name (never the value) and exits 2. Env `INTERN_SKIP_BUNDLE_SCRUB=1`
bypasses the scrub.

`scripts/bash/notify.sh <event> "<message>"` — events exactly: `plan_ready`,
`code_ready`, `train_started`, `train_done`, `error`, `blocker`,
`approval_required`. Telegram + Slack from env (`TG_BOT_TOKEN`/`TG_CHAT_ID`,
`SLACK_WEBHOOK_URL` or `SLACK_BOT_TOKEN`/`SLACK_CHANNEL_ID`); no-op safe, always
exit 0.

`scripts/bash/gpu_probe.sh` — `key=value` output (`cuda=`, `mps=`, `gpu_count=`,
`gpu_name=`, `vram_gb=`), always exit 0.

## Config groups

Experiment configs are `# @package _global_` and compose:

```yaml
defaults:
  - main
  - model: smollm2_135m
  - data: tiny_synthetic
  - trainer: trl_sft
  - tracking: trackio
  - compute: local
  - budget: default # task-keyed: smoke|default|lora|sft|dpo|grpo|pretrain|autoresearch
  - _self_

experiment_name: 001-<slug>
```

Groups land under `model.*`, `data.*`, `trainer.*`, `tracking.*`, `compute.*`,
`budget.*`. Seed budget.md from the composed profile with
`intern.py budget --experiment NNN init --profile <name>` (see
[docs/004](004-budget-and-gates.md)).

**When to use `_target_` (the rule for adding configs).** A node is a Hydra
`_target_` instantiate node **iff it maps 1:1 to constructing a Python object**.
Those are: `model.main`/`model.tokenizer`/`model.ref` (+ nested
`quantization_config`), `data.*` (`load_split` / `download_sessions`),
`trainer.args` (`trl.*Config` / `lightning.pytorch.Trainer`), `trainer.peft`
(`peft.LoraConfig`), and lightning `trainer.module`/`datamodule`. The adapters
instantiate these and — for `trainer.args` — inject the runtime values the
config can't hold (tracking wiring, bf16, smoke overrides, logger/callbacks);
that injection layer is the guardrail library's value-add, not a reason to avoid
`_target_`. Everything else stays **plain keys**, because it is not an object
construction: `tracking` is a backend _selector_ (`backend: none|trackio|wandb`)
plus a parameter sheet that code branches on; `compute` is an agent-read
parameter sheet no code instantiates; `budget` is a caps sheet parsed from
budget.md by `intern.budget`. Do not add `_target_` to those — it would be
cargo-culting a construction pattern onto config that selects behavior.

**model group** (`configs/model/<name>.yaml`) — Hydra-instantiable nodes,
consumed via
`hydra.utils.instantiate(OmegaConf.to_container(..., resolve=True))`: `main:`
(`_target_: transformers.AutoModelForCausalLM.from_pretrained`,
`_args_: [<repo_id>]`, kwargs incl. explicit `dtype` — omit it only for
quantized variants whose `quantization_config:` nested
`_target_: transformers.BitsAndBytesConfig` node sets compute dtype; the loader
injects `dtype: float32` and warns when both are absent, per the
checkpoint-dtype trap); `tokenizer:` (interpolates the model repo via
`${model.main._args_[0]}`). Quantized variants are separate files
(`<name>_4bit.yaml`), mirroring the trainer-preset pattern.

Two optional model keys are omitted from the shipped configs and added per
experiment only when needed: `ref:` (same shape as `main`, the DPO reference
model — DPO experiments set it under `_self_` or a dedicated model file; absent
elsewhere) and `target_params:` (an int enabling the opt-in `param_drift` verify
check; absent means that check SKIPs).

**data group** (`configs/data/<name>.yaml`) — `train:` and `eval:` are Hydra
`_target_` instantiate nodes (mirroring the model group); `eval: null` = no
eval. The default target is `data.loading.load_split` (`dataset` = Hub id or
local dir/file, `split`; any extra key — `revision`, `name`, `data_files`, … —
is forwarded to the underlying `datasets` loader), with a plain `path:` key kept
for interpolation (`${data.path}`) and for scripts that materialize local data.
Exotic loading points `_target_` at `datasets.load_dataset` directly. The
adapter injects `for_eval=True` into `load_split` eval nodes (plain on-disk
Dataset + eval = refused) and validates columns per task
(`data.loading.validate_columns`: SFT needs `text`/`dataset_text_field`,
`prompt`+`completion`, or `messages`; DPO needs `chosen`+`rejected`; GRPO needs
`prompt`). Optional `sample_prompts` overrides the generation-sanity probe with
fixed raw strings; omit it (shipped configs do) and
`training.sampling.resolve_sample_prompts` picks the probe automatically —
held-out prompts from the eval/train `prompt` column for prompt/completion data
(rendered in the model's chat format, so `add_special_tokens=False`), else the
built-in `SAMPLE_PROMPTS` for raw-text data.

A data config that feeds a **prep** pipeline rather than a trainer uses a
`source:` `_target_` node instead of `train:`/`eval:` — e.g.
[`pi_mono_raw.yaml`](../configs/data/pi_mono_raw.yaml) instantiates
`data.pi_mono.download_sessions` (raw Hub session JSONL, not
`datasets`-loadable), which `scripts/python/prep-pi-mono-sft.py` converts and
pushes as the trainable `pi_mono_sft`. Every dataset the repo uses — trainable
or raw source — has a `configs/data/*.yaml` with a `_target_`; likewise every
model in `configs/model/*.yaml` (`main:` + `tokenizer:`, plus a
`quantization_config` node for `_4bit` QLoRA variants). Nothing loads a model or
dataset from a hard-coded id in code.

The trainer group carries only run mechanics: `kind`, `planned_tokens`, `peft`,
`reward_funcs` (GRPO), `args`. Like the model and data groups, `args` and `peft`
are Hydra `_target_` instantiate nodes — `args._target_: trl.SFTConfig`
(/`DPOConfig`/`GRPOConfig`, or `lightning.pytorch.Trainer` on the lightning
lane), `peft._target_: peft.LoraConfig` — so every group declares the class it
builds. `build_args`/`peft_config` (and the lightning adapter) instantiate them
and inject the runtime values the config can't know (tracking wiring,
hardware-aware bf16, smoke overrides, logger/callbacks/alert flags), which is
the guardrail layer's value-add. Presets stack by inheritance: `trl_sft` (full
FT) → `trl_sft_lora` (LoRA) → `trl_sft_qlora` (paged optimizer + bf16; compose a
`_4bit` model). Model identity/loading lives in `model`; dataset identity in
`data`. `main.yaml` provides `seed`, `project_name`, `experiment_name`,
`experiment_dir`, `smoke_test`. Tracking backend is never hardcoded in code —
always `cfg.tracking.backend` (`trackio` primary, `wandb`, `none`).
`tracking.group` (default null) clusters related runs on the dashboard — wandb
via the `WANDB_RUN_GROUP` env var (both lanes); trackio via its
`init(group=...)` kwarg on the Lightning lane only (the TRL lane's
TrackioCallback hardcodes its init call and cannot forward it).

## Skill conventions

- Frontmatter: `name` (matches dir), `description` (third person, trigger-heavy:
  what + when, key phrases first). Body < 300 lines, router-not-manual: workflow
  steps + pointers into `references/` ("read X when Y"). Scripts referenced by
  exact `uv run` command.
- Blocking-gate rule verbatim in every training-related skill: "Never write
  results.md or report success unless `intern.py verify` exited 0. A failed gate
  means the run failed, regardless of loss."
- Research-before-clarify: never ask the user about anything you could look up.
- Headless posture: never hang — write best-guess defaults, fire
  `notify.sh approval_required`, proceed. Interactive: AskUserQuestion, ≤ 4
  bundled questions.
- Doom-loop guard: 3 identical tool calls with no new information → write
  `blocker.md`, fire `notify.sh blocker`, stop.
- Context discipline: subagents return concise reports only (no log dumps); read
  big files with head/tail/grep only.
- Done-conditions: every workflow skill ends with a checklist of artifacts that
  must exist. Never fire `train_done` for a run with no passing path.
- OOM recovery may change batch size / grad accumulation / GPU tier — never the
  method, dataset, or sequence length without user approval.
- No custom agents: the pack is skills-only (decided 2026-07-05). Orchestrating
  skills stay inline (they need AskUserQuestion, user visibility, and the Skill
  tool); isolated leaf work uses `context: fork` in skill frontmatter
  (literature-recipe-research) or ad-hoc read-only subagents with the skill's
  rubric in the prompt (autoresearch-loop critics).

## v1 scope and roadmap

v1 skills: `new-experiment`, `train-llm`, `verify-run`, `track-experiments`,
`literature-recipe-research`, `autoresearch-loop`, `publish-model`,
`distill-traces` (+ template's `new-doc`, `new-script`). Trainer lanes:
`trl_sft`, `trl_dpo`, `trl_grpo`, `lightning`, `axolotl`. Roadmap:
`eval-harness`, OpenEnv env-GRPO (environment-based rollouts), `add-dependency`,
`compare-experiments` CLI, `promote` CLI, `sync-agents-md` (AGENTS.md table
generator + pre-commit staleness check). Dropped: `experiment-ledger`
(superseded — within-experiment discipline lives in
train-llm/verify-run/track-experiments; cross-path orchestration is
autoresearch-loop) and `notify-milestones` (covered by notify.sh and the skills
that call it).
