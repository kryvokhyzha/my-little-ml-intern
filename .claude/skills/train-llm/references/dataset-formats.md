# Dataset formats per training method

Format mismatch is the most common training failure. Verify actual column names
against this table before any GPU spend — the check costs seconds.

| Method | Trainer group      | Required columns                                                                                                           |
| ------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| SFT    | `trainer=trl_sft`  | `messages` (list of `{"role": ..., "content": ...}` dicts) — preferred; OR plain `text`; OR `prompt` + `completion`        |
| DPO    | `trainer=trl_dpo`  | `chosen` + `rejected` required, exact names; `prompt` strongly recommended (TRL's implicit-prompt format works without it) |
| GRPO   | `trainer=trl_grpo` | `prompt` only — prompt-only data; completions are generated during training and reward functions grade them                |

Notes:

- **SFT** auto-detects which of the three shapes it got. `messages` rows must
  alternate roles the model's chat template accepts; a `text` column is used
  verbatim; `prompt`/`completion` are concatenated with completion-only loss.
- **DPO** is where ~90% of public preference datasets need mapping —
  `instruction`/`chosen_response`/`question`/`response_j` style columns are
  common and all wrong. Check every DPO dataset, no exceptions.
- **GRPO** has no completion column to validate — the contract moves to the
  reward functions; read `references/grpo-rewards.md` before writing one.
- Extra columns beyond the required ones are usually tolerated by TRL but map
  them away anyway — silent column pickup has caused wrong-field training.

## Tool-calling SFT

Tool-use rows are `messages` plus a `tools` column of JSON tool schemas when the
model's chat template consumes it — check the template before assuming it does.
Checks before spend:

- Every tool name appearing in `messages` exists in the `tools` schemas.
- Tool-call arguments parse as JSON (or the model's expected structured format).
- Tool-result observations must not leak held-out labels.
- Include success AND recovery-after-failure examples — a model trained only on
  clean successes cannot repair a failed call.
- Evaluate tool-call validity separately from final answer quality.

## Inspect a dataset quickly (CPU, seconds)

Preferred — load 5 rows and eyeball schema plus one full example:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('<hub-slug>', split='train[:5]')
print(ds)
print(ds[0])
"
```

No-download alternative via the datasets-server API:

```bash
curl -s 'https://datasets-server.huggingface.co/first-rows?dataset=<hub-slug>&config=default&split=train' | head -c 3000
```

Also check split names (`train` is not guaranteed) and row counts — confirm the
splits named in the `cfg.data.train` / `cfg.data.eval` nodes exist. Column
mismatches are also enforced in code at load time
(`data.loading.validate_columns` raises before any GPU step), but that check
sees only column names — this audit is about what's IN them: class imbalance,
empty strings, duplicated rows, wildly long outliers. Looking at data is the
cheapest performance win available and prevents failed jobs. Also verify
`max_length` truncation does not cut the decisive assistant/tool turn — measure
sampled tokenized lengths against `trainer.args` before launch.

## When mapping is needed

Write the mapping in the experiment script (or a preprocessing step it calls),
keyed to the actual columns you observed:

```python
def to_dpo(example):
    return {
        "prompt": example["instruction"],
        "chosen": example["chosen_response"],
        "rejected": example["rejected_response"],
    }

dataset = dataset.map(to_dpo, remove_columns=dataset.column_names)
```

Rules:

- `remove_columns=dataset.column_names` — leave only the target schema.
- Re-run the 5-row inspection on the mapped dataset before smoking.
- The mapping is part of the run's reproducibility: it lives in the committed
  experiment script, not in a throwaway shell one-liner.
- If the dataset cannot be mapped to the method (missing signal, e.g. no
  rejected answers for DPO), that is a blocker, not an invitation to swap
  datasets — never substitute silently. Interactive: ask. Headless: fire
  `scripts/bash/notify.sh approval_required "<proposed substitute>"` and record
  the assumption in task.md.

The smoke gate (`smoke_test=true`) slices to ≤ 32 rows, so a mapped dataset gets
exercised end-to-end by the smoke run before any long spend.
