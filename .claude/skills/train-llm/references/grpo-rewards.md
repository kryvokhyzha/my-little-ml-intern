# GRPO reward functions

GRPO optimizes exactly what the reward functions measure. The reward design IS
the task definition. Read this file before you write any reward function for a
`trainer=trl_grpo` path.

## Lane contract

- `trainer.reward_funcs` is a list of dotted import paths
  (`package.module:function`). The lane resolves each one with importlib at run
  time. An empty list is a launch-blocking error ("GRPO needs at least one
  reward function").
- For a parametrized reward, define a module-level `functools.partial` and
  reference it by dotted path. Two partials of the same function share one
  metric name, and TRL merges their `rewards/<name>` series. If you need a
  separate series per variant, define a separate module-level `def` for each
  variant.
- TRL calls each function with the batch: the prompts, the completions, and the
  extra dataset columns as kwargs. TRL expects one float per completion.
- **Floor-reward rule:** every reward function must tolerate a malformed
  completion — unparsable JSON, an empty string, or truncated output. The
  function returns the floor reward (its minimum score). The function never
  raises an exception. One raised exception stops the whole run in the middle of
  a group.

## The verifiable-reward ladder — cheapest first

Use the lowest rung that the task requires. Each higher rung costs more latency.
Each higher rung also gives the policy more opportunity to hack the reward:

1. **Exact string / numeric checks** — answer equality, numeric tolerance
   windows.
2. **Unit tests** — run the generated code against fixed test cases.
3. **Schema / tool-call validity** — the output parses as JSON, and it validates
   against the schema. The tool names and the argument types match the declared
   tools.
4. **Bounded rubric judge** — an LLM judge scores a fixed rubric onto a bounded
   range. This rung is the most expensive rung. The policy also hacks this rung
   most easily. Use it last.

## Log each reward component

Log each reward component as its own metric. Never log only the sum. A single
scalar is enough for the optimization, but it is useless when you debug. A flat
total can hide one component that saturates while another component collapses.

## Reward-hacking blacklist

Never reward the items in this list. Each item invites a known exploit:

- **response length alone** — length exploitation: the policy pads the response
  and does not solve the task.
- **formatting without task success** — format token stuffing: the policy puts a
  wrapper that looks valid around an empty answer.
- **judge prompts that reveal the answer** — judge sycophancy: the policy learns
  to repeat the judge. It does not learn to solve the task.
- **environment state unavailable at deployment time** — degenerate-but-valid
  outputs: the policy reaches these scores only while the training harness leaks
  state.
- **training on held-out eval tasks** — test-case memorization: the eval numbers
  rise, but the capability does not.

## GRPO smoke checklist

The standard smoke gate (`smoke_test=true`) still applies. Confirm each line
below before any long run:

- [ ] The generated completions parse. Read a few yourself. Check the format
      first, and the quality later.
- [ ] Each reward function returns the floor reward for a garbage string. It
      raises no exception.
- [ ] The reward variance across the group is nonzero. Identical rewards give
      zero advantage and zero gradient.
- [ ] `num_generations` × `max_completion_length` fits in memory. The trainer
      generates and scores that many completions per prompt.
- [ ] The smoke run logs the reward metrics (`reward`, `reward_std`,
      components).

The `reward_variance` verify check FAILS a run whose reward never varies. It
reports "no reward variance — the run optimized nothing". A constant reward is
not a neutral outcome. A constant reward is a failed run.

## Learning rate

The GRPO learning rate is far below the SFT learning rate. The `trl_grpo` lane
default is 1e-6. Read `references/hyperparameter-priors.md` for the band. Read
the RL column in `references/lora.md` when the path uses adapters (r=1–32, LR
10× the full-FT optimum).
