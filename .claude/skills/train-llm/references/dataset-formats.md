# Dataset formats per training method

A format mismatch is the most common training failure. Verify the actual column
names against this table before any GPU spend. The check costs seconds.

| Method | Trainer group      | Required columns                                                                                                           |
| ------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| SFT    | `trainer=trl_sft`  | `messages` (list of `{"role": ..., "content": ...}` dicts) — preferred; OR plain `text`; OR `prompt` + `completion`        |
| DPO    | `trainer=trl_dpo`  | `chosen` + `rejected` required, exact names; `prompt` strongly recommended (TRL's implicit-prompt format works without it) |
| KTO    | `trainer=trl_kto`  | `prompt` + `completion` + `label` (bool) — unpaired per-example feedback; no chosen/rejected pairing needed                |
| GRPO   | `trainer=trl_grpo` | `prompt` only — prompt-only data; completions are generated during training and reward functions grade them                |
| GKD    | `trainer=trl_gkd`  | `messages` — conversational; the student samples completions in-loop and the live `model.teacher` grades them token-level  |

Notes:

- **SFT** auto-detects which of the three shapes it receives. `messages` rows
  must alternate the roles that the model's chat template accepts. SFT uses a
  `text` column verbatim. SFT concatenates `prompt` and `completion`, and
  applies a completion-only loss.
- **DPO** needs a column mapping for ~90% of public preference datasets. Columns
  in the `instruction`/`chosen_response`/`question`/`response_j` style are
  common, and every one of them is wrong. Check every DPO dataset. There are no
  exceptions.
- **GRPO** has no completion column to validate. The contract moves to the
  reward functions. Read `references/grpo-rewards.md` before you write one.
- TRL usually tolerates extra columns beyond the required ones. Map the extra
  columns away anyway. A silent column pickup has caused training on the wrong
  field.

## Tool-calling SFT

A tool-use row holds `messages` plus a `tools` column of JSON tool schemas. The
row needs the `tools` column only when the model's chat template consumes it.
Check the template first. Do not assume that it consumes the column. Checks
before the spend:

- Every tool name that appears in `messages` exists in the `tools` schemas.
- Tool-call arguments parse as JSON (or the model's expected structured format).
- Tool-result observations must not leak held-out labels.
- Include success examples AND recovery-after-failure examples. A model that you
  train only on clean successes cannot repair a failed call.
- Evaluate tool-call validity separately from final answer quality.

## Inspect a dataset quickly (CPU, seconds)

Preferred method — load 5 rows. Read the schema and one full example:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('<hub-slug>', split='train[:5]')
print(ds)
print(ds[0])
"
```

No-download alternative through the datasets-server API:

```bash
curl -s 'https://datasets-server.huggingface.co/first-rows?dataset=<hub-slug>&config=default&split=train' | head -c 3000
```

Also check the split names and the row counts. The dataset does not always have
a `train` split. Confirm that the dataset has the splits that `cfg.data.train`
and `cfg.data.eval` name.

The code also enforces the column names at load time:
`data.loading.validate_columns` raises before any GPU step. That check reads
only the column names. This inspection reads what is inside them: class
imbalance, empty strings, duplicated rows, and very long outliers. It is the
cheapest performance win available, and it prevents failed jobs. Also verify
that `max_length` truncation does not cut the decisive assistant/tool turn.
Measure the sampled tokenized lengths against `trainer.args` before the launch.

## When the dataset needs a mapping

Write the mapping in the experiment script. A preprocessing step that the script
calls also works. Key the mapping to the actual columns that you observed:

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
- Re-run the 5-row inspection on the mapped dataset before the smoke run.
- The mapping is part of the run's reproducibility. It lives in the committed
  experiment script, not in a throwaway shell one-liner.
- If you cannot map the dataset to the method, that is a blocker. A missing
  signal is one cause, for example a DPO dataset with no rejected answers. Never
  substitute another dataset silently. Interactive: ask. Headless: fire
  `scripts/bash/notify.sh approval_required "<proposed substitute>"`. Then
  record the assumption in task.md.

The smoke gate (`smoke_test=true`) slices the dataset to ≤ 32 rows. The smoke
run therefore exercises a mapped dataset end-to-end before any long spend.
