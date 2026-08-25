# Trace schema and conversion

This document is the contract for `experiments/NNN-<slug>/traces/*.jsonl`. Git
ignores those files. `src/intern/traces.py` implements the contract. The
authoritative schema lives in `docs/001-architecture.md` — that document wins on
any drift.

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

`TraceStore(path)` gives `append(record)`, `read() -> list[TraceRecord]`, and
`accepted() -> list[TraceRecord]` (only `accepted=True`). Write one JSONL file
for each collection batch under `experiments/NNN-<slug>/traces/`.

## Conversion targets — which converter for which imitation target

| imitation target                 | converter                                | output rows                          |
| -------------------------------- | ---------------------------------------- | ------------------------------------ |
| full transcripts, tool decisions | `to_sft_messages(records)`               | `{"messages": ..., "tools": ...}`    |
| final answers, repair behavior   | `to_prompt_completion(records, tok)`     | `{"prompt": ..., "completion": ...}` |
| preference / reward data         | no converter yet — map in the run script | DPO or GRPO columns                  |

1. **`to_sft_messages(records, only_accepted=True)`** — use this converter when
   every assistant turn in the trace is target behavior (full-transcript
   imitation, tool-call decisions). It feeds the trl_sft `messages` format
   directly. It also carries `tools` for the chat templates that render them.
   When the traces contain tool calls, apply the tool-calling checks in
   train-llm's `dataset-formats.md` reference before any GPU spend.
2. **`to_prompt_completion(records, tokenizer, only_accepted=True)`** — use this
   converter when exactly one turn is the training target. It renders one row
   for each **final assistant turn**. The context is everything before that
   turn. The completion is that turn. For a repair-behavior target, truncate
   each record's `messages` first. The repair turn must be the final assistant
   turn before you convert. It feeds trl_sft `prompt`/`completion` with
   completion-only loss.
3. **Preference / reward data** — an accepted trace and a comparable rejected
   trace on the same `task_id` become `prompt`/`chosen`/`rejected` rows for
   `trainer=trl_dpo`. Prompt-only tasks plus verifier-derived reward functions
   become `prompt` rows for `trainer=trl_grpo`. The reward functions are dotted
   import paths in `trainer.reward_funcs`. No converter exists yet. Write the
   mapping in the experiment script, and follow the dataset-formats rules.

Use one target for each experiment path. If you mix targets in one dataset, you
cannot attribute the resulting delta to a target.

## Prompt/completion rendering contract

`to_prompt_completion` renders each record as:

```python
prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False)
full = tokenizer.apply_chat_template(context + [assistant_turn], tokenize=False)
completion = full[len(prompt):]
```

It then asserts `full.startswith(prompt)`. When the assert fails, it raises a
`ValueError` that names the chat-template prefix mismatch. This matters, because
some chat templates rewrite the earlier turns when you append a new turn. Such a
template strips or merges the system messages, injects the current date, or
reformats the tool JSON. Then `full` is not `prompt + completion`. The
completion-loss boundary lands in the middle of the history. The training run
then silently optimizes the wrong tokens, and the loss curve looks perfectly
normal. The assert changes that silent corruption into a visible failure.

When you get a prefix-mismatch ValueError, do NOT strip the assert. Do NOT slice
the strings by hand. Switch to a template or a model that renders append-only,
or pin a chat template that renders append-only. Then convert the records again.
Always convert with the SAME tokenizer that the training run uses. A different
tokenizer renders different boundaries.

## Session ingestion

You can mine these local agent sessions:

- Claude Code: `~/.claude/projects/` (one dir per project, JSONL per session)
- Codex: `~/.codex/sessions/`

A session contains prompts, tool inputs, command output, and file contents.
Treat every line as tainted until you grep it. The converted datasets and the
published bundles inherit whatever you keep. REVIEW AND REDACT these items
before a record enters the store:

- API keys and token-shaped strings — `hf_`, `sk-`, `xox[abp]-`, `AKIA`, `ghp_`
  prefixes (the publish bundle scrub enforces the same pattern list).
- Absolute home paths — `/Users/<name>/`, `/home/<name>/`.
- Private URLs — internal hosts, signed URLs, tracking links.
- Personal data — emails, real names, customer content pasted into prompts.

Git ignores the traces dir, but gitignore is not redaction. Redact the data at
ingestion time, not at publish time.

## Worked conversion

Run this snippet from the repo root. The bare `intern` imports need `src` on the
path:

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

For `to_sft_messages`, swap the converter call. Then drop the tokenizer. Notes:

- The `split == 'train'` filter applies the contamination guard a second time at
  conversion. Keep the filter even though the collection step already tagged the
  splits.
- Git ignores `experiments/NNN-<slug>/data/`, like the traces dir.
- After the conversion, re-run the 5-row dataset inspection from train-llm's
  dataset-formats reference on the output file. Do this before the smoke run.
- To evaluate the held-out eval tasks, RUN the student on them. Score its
  success. Do not convert their teacher traces into a loss set.
