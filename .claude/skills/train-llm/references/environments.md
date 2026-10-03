# Environments — multi-turn tool RL on the GRPO lanes

Read this file before you train a model inside an environment (`trl_grpo_env`,
or `trl_grpo` / `trl_async_grpo` with `trainer.environment_factory`,
`trainer.tools`, or `trainer.env_spec`). The contract is TRL's
`GRPOTrainer(environment_factory=...)`: an experimental feature of the stable
trainer. TRL warns about it; `run_grpo` sets `TRL_EXPERIMENTAL_SILENCE=1`.

## How one rollout runs

1. TRL builds one environment instance per concurrent rollout and reuses it.
2. TRL calls `reset(**row)` with every dataset column as a kwarg. A string that
   `reset` returns is appended to the last user message. `None` adds nothing.
3. The model generates. TRL parses each tool call and runs the matching public
   method of the instance. TRL feeds the result back as a `tool` message.
4. The loop stops at the end of the answer, at `max_tool_calling_iterations`, or
   when the exchange fills `max_completion_length`.
5. TRL calls `get_reward()` once per rollout. The value joins the reward
   functions (weight 1, logged as `rewards/<EnvClass>/mean` on `trl_grpo`; as
   `rewards/<EnvClass>` on `trl_async_grpo`).

Tool-result tokens are masked out of the loss. Only the model's own tokens
train.

## The environment contract

- Define `reset(self, **kwargs)`. Accept `**_` so that extra columns never crash
  it.
- Every public method except `reset` and `get_reward` becomes a tool. Give each
  tool a Google-style docstring with an `Args:` section and type hints:
  `transformers.utils.get_json_schema` builds the tool schema from them.
- Prefix helpers with `_`, or TRL exposes them as tools.
- Make `get_reward()` return a float, or `None` to skip the rollout. It may be
  `async` (for example an LLM judge).
- Never raise from a tool on bad model input. Return an error string. TRL turns
  an exception into `{"error": ...}`, but a clear string teaches better.
- TRL never calls `close()`. Release external resources in `reset` or `__del__`.

The worked example is `src/training/envs/guess_number.py` with
`configs/trainer/trl_grpo_env.yaml` and `configs/data/guess_number.yaml`. Its
tests in `tests/test_envs.py` show how to check the contract offline.

## The model must parse tool calls

`GRPOTrainer` refuses a chat template that cannot render a tool call. TRL also
needs a response parser for the template. TRL 1.14 parses tool calls for these
families: Qwen 2.5 / 3 / 3.5 / 3.6 / 3.8, Llama 3.1 / 3.2, Gemma 4 (since TRL
1.9.1), GPT-OSS, GLM-4-MoE, Nemotron 3 / 3.5, and LFM 2.5. TRL parses the
DeepSeek-R1-Distill template for reasoning only and refuses it for tools. The
SmolLM2 and the Gemma 1-3 templates fail. Use `model=qwen3_0_6b` as the small
default. (Checked on 2026-09-27 with
`trl.chat_template_utils.supports_tool_calling` against TRL's bundled
templates.)

## Rewards

- The environment can own the reward (`get_reward`). Then `trainer.reward_funcs`
  may stay empty.
- A reward function also receives `environments=[...]` (one instance per
  completion). Use it to combine the final answer text with the environment
  state.
- You cannot weight the environment reward: TRL appends it with weight 1.
- Check `reward_std` in the first steps. A group where every rollout scores the
  same teaches nothing. On `trl_grpo` and `trl_rloo`, also check
  `frac_reward_zero_std`. On `trl_async_grpo`, watch `reward_std` alone; the
  async trainer does not log `frac_reward_zero_std`. The verify gate fails a run
  whose `reward_std` stays 0 for the whole run.

## Packaged environments (`trainer.env_spec`)

A spec object carries the dataset, the environment factory, and the reward
functions together. Set `data.train: null` when you use one.

`data.train: null` with a plain `trainer.environment_factory` (environment-owned
data) works only when all of these are true:

- one factory, not a dict of factories and not an `env_spec`;
- its `reset()` returns the prompt text (TRL builds empty placeholder prompts);
- `trainer.args.max_steps` > 0 (TRL raises otherwise; the presets ship -1).

`GuessNumberEnv` is dataset-driven: keep `data=guess_number` with it.

| Spec                                         | What it wraps                                  | Needs                                                  |
| -------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| `trl.experimental.harbor.HarborSpec`         | Harbor task directories; one `bash` tool       | `harbor~=0.23.0` (Python ≥ 3.12) and Docker on the box |
| `trl.experimental.openreward.OpenRewardSpec` | an OpenReward catalog environment (ORS server) | `openreward~=0.1.158` and `OPENREWARD_API_KEY` in .env |

Neither package is in the lockfile. Install it on the training box with
`uv run --with "<package pin from the table>" ...`, or add it as a dependency
through the normal dependency procedure (AGENTS.md "Add dependencies"). The pins
were the newest releases ≥ 1 week old on 2026-09-27, and TRL 1.14 imports both.

Example config fragment:

```yaml
data:
  train: null
trainer:
  env_spec:
    _target_: trl.experimental.harbor.HarborSpec
    dataset: <harbor dataset>
    agent: bash
    num_tasks: 64
```

## Loop-owning agents (opencode, codex, Claude Code)

`trl.experimental.async_grpo.openenv_harness` trains a model under an agent that
owns its own tool loop. It needs the `openenv` package, a vLLM server, and CUDA.
Run `uv sync --group async` for `openenv~=0.7.0` (not `openenv-core`, which
stopped at 0.3.0). The adapter does not wire it yet: build a
`HarnessRolloutWorker` in the experiment script and pass it to
`AsyncGRPOTrainer(rollout_worker=...)`. Read `docs/011-harness-rl.md` first: it
lists the six gaps, the session contract, and the three hooks
(`rollout_reward_fn`, `train_turn_fn`, `agent_turn_fn`). trl 1.14.1's worker
re-tokenizes each prompt; the exact-token trainer is on TRL `main` (PR #6947)
until the next release. TRL's worked examples live at
`examples/async_grpo_opencode/` in the TRL repo (moved there in TRL 1.11);
`opencode_hf_sandbox.py` runs each rollout in a remote HF sandbox, so the box
needs no Docker.

## Smoke and verify

- `smoke_test=true` works on CPU for `trl_grpo_env`. A random or untrained model
  seldom calls a tool, so `tools/call_frequency` near 0 in a smoke run is
  normal.
- In the real run, check `tools/call_frequency` and `tools/failure_frequency` in
  the first logged steps. A call frequency near 0 means that the model does not
  use the tools. Fix the prompt or the model before you spend the budget.
- The policy is a generative LM, so the verify gate requires
  `logs/samples.jsonl`.
