# Distillation modes — pick the lane before you collect or train

Read this file when you choose how a student learns from a teacher, or from
itself. Each row is one lane. The worked examples are experiments 002, 003, and
004 (see `docs/008-example-distillation.md`).

## The mode table

| Mode                            | Lane                                   | Teacher at training time                                | Dataset columns                                                                                                          | Use when                                                                     |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Off-policy imitation            | `trl_sft`                              | none — it wrote the data offline                        | `messages` or `prompt`+`completion`                                                                                      | Start here. Any teacher works, including an API-only model. (002)            |
| On-policy, same tokenizer       | `trl_distill`                          | live `model.teacher`, same vocabulary                   | `prompt`                                                                                                                 | The off-policy student shows exposure bias.                                  |
| On-policy with a dataset mix    | `trl_gkd`                              | live `model.teacher`, same tokenizer                    | `messages`                                                                                                               | You want `lmbda` < 1: part dataset text, part student samples. (003)         |
| On-policy, another tokenizer    | `trl_gold` + `use_uld_loss: true`      | live `model.teacher`, any tokenizer                     | `messages` or `prompt`+`completion`                                                                                      | The best open teacher is from another model family.                          |
| On-policy, asynchronous         | `trl_async_distill`                    | vLLM HTTP server(s), same tokenizer                     | `prompt` (+ `teacher_id` for several teachers)                                                                           | Generation dominates the time and a GPU box with vLLM is ready.              |
| Self, verified (STaR / RFT)     | `prep-*` script + `trl_sft`            | none — a deterministic verifier accepts own samples     | `prompt`+`completion` of accepted rollouts                                                                               | The task has a verifier. (004)                                               |
| Self, privileged context (SDFT) | `trl_sdft`                             | the model itself, shown a privileged context            | `prompt` + `privileged_context`                                                                                          | You have demonstrations or documents and must not cause forgetting.          |
| Self, reward-selected (SDPO)    | `trl_sdpo`                             | EMA self, re-prompted with a successful sibling rollout | `prompt` (+ `privileged_context` only with `include_environment_feedback: true`; environment feedback, never the answer) | A reward function exists and successes are rare but real.                    |
| Self, unverified (SSD)          | `trl_ssd`                              | none                                                    | `prompt`                                                                                                                 | No verifier exists. Cheap baseline; it can reinforce the model's own errors. |
| RL from a reward                | `trl_grpo`, `trl_rloo`, `trl_grpo_env` | none — reward functions or an environment               | `prompt`                                                                                                                 | The reward is verifiable, or the task is a multi-turn tool environment.      |

## Rules

- Change one variable per path. The mode is a variable: 002 vs 003 differ only
  by the mode.
- Keep the train/eval task split from step 1 of the skill. The on-policy and
  self lanes sample from `prompt` rows, so an eval prompt in the train split is
  a leak even when no trace was collected.
- A teacher for `trl_distill` / `trl_gkd` must share the student vocabulary. TRL
  checks `vocab_size` and refuses a mismatch. Use `trl_gold` or a text-level
  mode for a teacher from another family.
- The claim metric of a self-distillation run is the held-out task success rate
  (the same verifier, greedy decoding), not the loss. The losses of these lanes
  are JSD, KL, or policy terms, so `loss_plausibility` SKIPs for them.
- `trl_ssd` trains on unverified samples. Compare it with the verified loop (004
  machinery) before you trust a gain.
- A `trl_gkd` → `trl_distill` path changes more than the trainer. The `beta`
  default is 0.5 on GKD and 1.0 on distill. Distill also applies `temperature`
  to the loss. To keep one variable, pin `trainer.args.beta` to the GKD value.
  Map `max_new_tokens` to `max_completion_length`. Set `temperature: 1.0`. This
  path holds only when the GKD run itself uses `lmbda: 1.0` and
  `temperature: 1.0`. A 003-style run (`lmbda: 0.5`, `temperature: 0.9`) needs a
  new GKD baseline first. Re-run GKD at `lmbda: 1.0` and `temperature: 1.0`,
  then compare it with `trl_distill`.
- `trl_distill`, `trl_grpo`, `trl_rloo`, and `trl_grpo_env` do not truncate
  prompts. Drop or trim over-long prompt rows (for example, long agent traces)
  in data prep. Only `max_completion_length` bounds the sequence.
- `trl_gkd` is experimental, and a stable trainer (`trl_distill`) covers it. TRL
  can remove it in any release, patch releases included, without a deprecation.
  Plan a new on-policy run on `trl_distill`.

## Data builders that already exist

- `data.self_distill.build_prompt_dataset`
  (`configs/data/arithmetic_prompts.yaml`) — verifiable `prompt` rows with
  `answer` and `privileged_context` columns. It feeds every lane above except
  `trl_gkd` and `trl_gold`.
- `data.self_distill.arithmetic_reward` — the deterministic reward for
  `trl_grpo`, `trl_rloo`, and `trl_sdpo`
  (`trainer.reward_funcs: [data.self_distill:arithmetic_reward]`).
- `intern.traces.to_sft_messages` / `to_prompt_completion` — agent traces to the
  off-policy and verified-self datasets.
