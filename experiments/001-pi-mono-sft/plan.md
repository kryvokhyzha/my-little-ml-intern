# Plan — 001-pi-mono-sft

## Hypotheses

### H1: completion-only QLoRA SFT teaches the pi agent format

- mechanism: masking loss to the assistant completion and fine-tuning the
  gemma-4 language-tower LoRA adapter on real pi-mono agent traces makes the
  model emit the pi chat/tool-call format and reduces held-out prompt/completion
  loss on unseen sessions.
- expected_delta: `final_eval_loss ~= 0.55` (reference `lr2e4-r16-len4k`:
  eval_loss `0.5506`, eval token accuracy `0.8624`; downstream HumanEval `74.4%`).
- falsification: `final_eval_loss` stays `> 1.0` after 200 steps, OR generation
  samples do not follow the pi tool-call format (no structured tool calls,
  degenerate or off-format output) — either kills H1.

## Solution paths

One variable per path — this is a reproduction, so a single path.

| path_id | hypothesis | override                                                                  |
| ------- | ---------- | ------------------------------------------------------------------------- |
| path-1  | H1         | model + data reproduction (`trainer.model_name=google/gemma-4-E2B-it`, `trainer.dataset=${HF_USER}/pi-mono-sft`) — the config defaults |

## Success criterion

`intern.py verify --experiment 001` exits 0, and the human read of
`logs/samples.jsonl` confirms pi tool-call format. Reference eval_loss `0.5506`.
