# Trace schema and conversion

Contract for `experiments/NNN-<slug>/traces/*.jsonl` (gitignored), implemented
by `src/intern/traces.py`. The authoritative schema lives in
`docs/001-architecture.md` — that doc wins on any drift.

## TraceRecord fields

| field               | type                 | meaning                                                                     |
| ------------------- | -------------------- | --------------------------------------------------------------------------- |
| `task_id`           | `str`                | stable id of the task in the pool; ties the trace to the train/eval split   |
| `split`             | `str`                | `"train"` or `"eval"` — assigned at collection time, never after            |
| `messages`          | `list[dict]`         | full chat-format conversation (`{"role": ..., "content": ...}` turns)       |
| `tools`             | `list[dict] \| None` | tool schemas available to the agent during the episode                      |
| `model_id`          | `str \| None`        | teacher/policy that produced the trace                                      |
| `gen_params`        | `dict \| None`       | generation parameters (temperature, top_p, ...)                             |
| `steps`             | `list[dict] \| None` | agent-loop steps as `[{action, observation}]`                               |
| `terminal_status`   | `str \| None`        | how the episode ended (success, failure, timeout, ...)                      |
| `verifier_output`   | `dict \| None`       | deterministic check result (tests passed, output diff, ...)                 |
| `judge_critique`    | `str \| None`        | judge/teacher quality critique — ranking signal only, never sole acceptance |
| `reward_components` | `dict \| None`       | per-component reward values, if a reward was computed                       |
| `accepted`          | `bool`               | acceptance-filter verdict; converters skip `false` rows by default          |

`TraceStore(path)` — `append(record)`, `read() -> list[TraceRecord]`,
`accepted() -> list[TraceRecord]` (only `accepted=True`). One JSONL file per
collection batch under `experiments/NNN-<slug>/traces/`.

## Conversion targets — which converter for which imitation target

| imitation target                 | converter                                | output rows                          |
| -------------------------------- | ---------------------------------------- | ------------------------------------ |
| full transcripts, tool decisions | `to_sft_messages(records)`               | `{"messages": ..., "tools": ...}`    |
| final answers, repair behavior   | `to_prompt_completion(records, tok)`     | `{"prompt": ..., "completion": ...}` |
| preference / reward data         | no converter yet — map in the run script | DPO or GRPO columns                  |

1. **`to_sft_messages(records, only_accepted=True)`** — use when every assistant
   turn in the trace is target behavior (full-transcript imitation, tool-call
   decisions). Feeds the trl_sft `messages` format directly; `tools` ride along
   for chat templates that render them. When traces contain tool calls, the
   tool-calling checks in train-llm's `dataset-formats.md` reference apply
   before any GPU spend.
2. **`to_prompt_completion(records, tokenizer, only_accepted=True)`** — use when
   exactly one turn is the training target. It renders per **final assistant
   turn**: context = everything before it, completion = that turn. For
   repair-behavior targets, truncate each record's `messages` so the repair turn
   is the final assistant turn before converting. Feeds trl_sft
   `prompt`/`completion` with completion-only loss.
3. **Preference / reward data** — accepted vs comparable rejected traces on the
   same `task_id` become `prompt`/`chosen`/`rejected` for `trainer=trl_dpo`;
   prompt-only tasks plus verifier-derived reward functions become `prompt` rows
   for `trainer=trl_grpo` (reward functions are dotted import paths in
   `trainer.reward_funcs`). No converter exists yet — write the mapping in the
   experiment script per the dataset-formats rules.

One target per experiment path. Mixing targets in one dataset makes the
resulting delta unattributable.

## Prompt/completion rendering contract

`to_prompt_completion` renders each record as:

```python
prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False)
full = tokenizer.apply_chat_template(context + [assistant_turn], tokenize=False)
completion = full[len(prompt):]
```

It then asserts `full.startswith(prompt)` and raises a `ValueError` naming the
chat-template prefix mismatch when the assert fails. Why this matters: some chat
templates rewrite earlier turns when a new turn is appended (strip or merge
system messages, inject the current date, reformat tool JSON). Then `full` is
not `prompt + completion`, the completion-loss boundary lands mid-history, and
training silently optimizes the wrong tokens while the loss curve looks
perfectly normal. The assert turns that silent corruption into a loud failure.

On a prefix-mismatch ValueError: do NOT strip the assert or hand-slice strings.
Switch to a template/model whose rendering is append-only, or pin a chat
template that is, and re-convert. Always convert with the SAME tokenizer the
training run will use — a different tokenizer renders different boundaries.

## Session ingestion

Local agent sessions worth mining:

- Claude Code: `~/.claude/projects/` (one dir per project, JSONL per session)
- Codex: `~/.codex/sessions/`

Sessions contain prompts, tool inputs, command output, and file contents — treat
every line as tainted until grepped. REVIEW AND REDACT before a record enters
the store (converted datasets and published bundles inherit whatever you keep):

- API keys and token-shaped strings — `hf_`, `sk-`, `xox[abp]-`, `AKIA`, `ghp_`
  prefixes (the same pattern list the publish bundle scrub enforces).
- Absolute home paths — `/Users/<name>/`, `/home/<name>/`.
- Private URLs — internal hosts, signed URLs, tracking links.
- Personal data — emails, real names, customer content pasted into prompts.

The traces dir is gitignored, but gitignore is not redaction: redact at
ingestion time, not at publish time.

## Worked conversion

From the repo root (bare `intern` imports need `src` on the path):

```bash
uv run python -c "
import json, sys
sys.path.insert(0, 'src')
from pathlib import Path
from transformers import AutoTokenizer
from intern.traces import TraceStore, to_prompt_completion

store = TraceStore(Path('experiments/NNN-<slug>/traces/rollouts.jsonl'))
records = [r for r in store.accepted() if r.split == 'train']
tokenizer = AutoTokenizer.from_pretrained('<model_name>')
rows = to_prompt_completion(records, tokenizer)
out = Path('experiments/NNN-<slug>/data/train.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')
print(f'{len(rows)} rows -> {out}')
"
```

For `to_sft_messages`, swap the converter call and drop the tokenizer. Notes:

- The `split == 'train'` filter is the contamination guard applied a second time
  at conversion — keep it even though collection already tagged splits.
- `experiments/NNN-<slug>/data/` is gitignored, like the traces dir.
- After converting, re-run the 5-row dataset inspection from train-llm's
  dataset-formats reference on the output file before the smoke run.
- Held-out eval tasks are evaluated by RUNNING the student on them and scoring
  success, not by converting their teacher traces into a loss set.
