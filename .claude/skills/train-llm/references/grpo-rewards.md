# GRPO reward functions

GRPO optimizes exactly what the reward functions measure — reward design IS the
task definition. Read this before writing any reward function for a
`trainer=trl_grpo` path.

## Lane contract

- `trainer.reward_funcs` is a list of dotted import paths
  (`package.module:function`), resolved via importlib at run time. An empty list
  is a launch-blocking error ("GRPO needs at least one reward function").
- TRL calls each function with the batch (prompts, completions, plus extra
  dataset columns as kwargs) and expects one float per completion.
- **Floor-reward rule:** every reward function must tolerate malformed
  completions — unparsable JSON, empty strings, truncated output — by returning
  the floor reward (its minimum score), never by raising. One raised exception
  kills the whole run mid-group.

## The verifiable-reward ladder — cheapest first

Climb only as far as the task requires; every rung up costs latency and adds a
gameable surface:

1. **Exact string / numeric checks** — answer equality, numeric tolerance
   windows.
2. **Unit tests** — run generated code against fixed test cases.
3. **Schema / tool-call validity** — output parses as JSON, validates against
   the schema, tool names and argument types match the declared tools.
4. **Bounded rubric judge** — an LLM judge scoring a fixed rubric onto a bounded
   range; the most expensive and most gameable rung, use it last.

## Component logging

Log each reward component as its own metric, never only the sum. A single scalar
is enough for optimization but useless for debugging — a flat total can hide one
component saturating while another collapses.

## Reward-hacking blacklist

Never reward these (each invites a known exploit):

- **response length alone** — length exploitation: the policy pads instead of
  solving.
- **formatting without task success** — format token stuffing: valid-looking
  wrappers around empty answers.
- **judge prompts that reveal the answer** — judge sycophancy: the policy learns
  to echo the judge, not solve the task.
- **environment state unavailable at deployment time** — degenerate-but-valid
  outputs: scores only reachable while the training harness leaks state.
- **training on held-out eval tasks** — test-case memorization: eval numbers
  rise while capability doesn't.

## GRPO smoke checklist

On top of the standard smoke gate (`smoke_test=true`), confirm before any long
run:

- [ ] Generated completions parse — eyeball a few; format first, quality later.
- [ ] Each reward function, fed garbage strings, returns the floor reward
      without raising.
- [ ] Reward variance across the group is nonzero — identical rewards mean zero
      advantage and zero gradient.
- [ ] `num_generations` × `max_completion_length` fits memory (that many
      completions are generated and scored per prompt).
- [ ] The smoke run logs reward metrics (`reward`, `reward_std`, components).

The `reward_variance` verify check will FAIL a run whose reward never varies —
"no reward variance — the run optimized nothing". Constant reward is not a
neutral outcome; it is a failed run.

## Learning rate

GRPO LR sits far below SFT — the `trl_grpo` lane default is 1e-6. See
`references/hyperparameter-priors.md` for the band, and the RL column in
`references/lora.md` when the path uses adapters (r=1–32, LR 10× the full-FT
optimum).
