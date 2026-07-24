# 009 — Frontier lanes: block-diffusion SFT and in-harness agent RL

<!-- Adoption analysis for two upstream TRL capabilities. Verdict first, then the
mechanism, the blockers, and the trigger that would make each re-evaluable.
Sources: TRL examples/scripts/sft_diffusion_gemma.py and
examples/scripts/openenv/opencode.py, read 2026-07-24 against trl 1.8.0 /
transformers 5.14.1. Claims here were adversarially re-checked against the
installed packages; where a first draft was wrong, the corrected fact is what
survived. -->

Headline: **neither is a one-file lane addition, and both are further away than
their announcement posts suggest** — but the distances differ. Agent RL splits
into a white-box half that is _close_ (three concrete changes, none exotic) and
a loop-owning half that is _blocked on unreleased code_. Block-diffusion SFT is
out of reach for real training, though a 4 M-param test checkpoint makes the
plumbing verifiable.

|                           | **Block-diffusion SFT** (DiffusionGemma)                                                                                                                  | **Agent RL, white-box** (OpenEnv + GRPO)                                                         | **Agent RL, loop-owning** (OpenEnv + AsyncGRPO)                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Upstream                  | `examples/scripts/sft_diffusion_gemma.py`                                                                                                                 | `examples/scripts/openenv/{echo,wordle,sudoku,…}.py`                                             | `examples/scripts/openenv/opencode.py`                                                                            |
| Who drives the agent loop | n/a                                                                                                                                                       | **TRL** — it parses tool calls and feeds results back                                            | **the agent itself** (opencode/codex/Claude Code)                                                                 |
| Blocker class             | upstream unreleased + hardware                                                                                                                            | **ours** — three fixable repo-side gaps                                                          | upstream unreleased                                                                                               |
| Specifically              | `supports_gradient_checkpointing = False` in transformers 5.14.1, which the script calls required; real training needs ≥2×80 GB and a 26B-only checkpoint | needs `jmespath`, a tool-calling-capable base model, and a lane that allows env-supplied rewards | `trl.experimental.async_grpo.openenv_harness` is **main-branch only** — absent from 1.8.0 and from the v1.9.0 tag |
| Verdict                   | **document only** (plumbing smoke-able on a tiny test model)                                                                                              | **the reachable one** — a real experiment after three changes                                    | **document now, adopt when released**                                                                             |

## 1. Block-diffusion SFT (DiffusionGemma)

### The mechanism

`google/diffusiongemma-26B-A4B-it` pairs a causal encoder with a bidirectional
decoder. The encoder reads clean context into a KV cache; the decoder denoises a
**canvas** of `canvas_length = 256` tokens that cross-attends to that cache.
Once a canvas is denoised it joins the cache and the next block starts — block
autoregressive, token-parallel within a block.

Training replaces the autoregressive loss entirely (TRL does it by subclassing
`SFTTrainer` and overriding `compute_loss`):

1. Pick one response block per example at random; the decoder may only see the
   prompt plus the clean blocks **before** it.
2. Corrupt the clean canvas by replacing tokens with uniform-random vocabulary
   tokens at per-example rate `t ~ U(ε, 1-ε)` — **no mask token**.
3. With p=0.5, self-condition the decoder on its own logits from a first no-grad
   pass (so each step costs two forwards).
4. Loss = flat cross-entropy over the whole canvas + an autoregressive co-loss
   on the encoder. `model_prediction_type=mean_loo` instead trains the
   leave-one-out posterior ([paper](https://huggingface.co/papers/2605.22765)),
   which improves generation but changes what sampling must do.

Reference recipe: LR 1.5e-4, betas (0.95, 0.99), wd 1e-4, 25 warmup steps then
cosine to 10%, global batch 8, seq 1024, 800 steps, LoRA r=16/α=32 on the
attention and dense-MLP linears of encoder/decoder only (MoE experts, router,
self-conditioning block and vision tower frozen). Packing is rejected outright —
the diffusion loss needs per-example response spans. Adapters must stay
**unmerged**: encoder and decoder share tied base weights, so `merge_and_unload`
would fold both deltas into the same tensors.

### Verified facts (checked against transformers 5.14.1)

- `transformers.DiffusionGemmaForBlockDiffusion` **exists**, and its `forward`
  accepts every kwarg the TRL example passes (`decoder_input_ids`,
  `decoder_attention_mask`, `decoder_position_ids`, `self_conditioning_logits`,
  `self_conditioning_mask`).
- `supports_gradient_checkpointing` is **`False`**. The upstream script states
  gradient checkpointing is required to fit activations, and that support
  ([PR 46572](https://github.com/huggingface/transformers/pull/46572)) is merged
  but unreleased. **This is the blocker for real training.**
- The production checkpoint is 25,823,778,864 params (128 experts, top-8, vocab
  262,144, softcapping 30.0). It exceeds **every** budget profile we have (200 M
  default, 2 B pretrain, 3 B sft, 12 B lora/dpo/grpo), so the budget gate denies
  it unconditionally.
- **A tiny checkpoint does exist**:
  `trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion` — 4,272,204
  params, `canvas_length: 32`, 2 layers, 4 experts, not quantized. It is TRL's
  own test fixture, and it sits far under the 200 M default ceiling. So the
  _plumbing_ (collator, loss, gate wiring) is smoke-testable locally even though
  real training is not; what it cannot tell you is whether the objective learns
  anything.
- `ForBlockDiffusion` is registered under `AutoModelForImageTextToText` and
  `AutoModelForMultimodalLM` (and the base `DiffusionGemmaModel` under
  `AutoModel`), but **not** under `AutoModelForCausalLM` — which is what every
  `configs/model/*.yaml` in this repo targets.
- `forward()` takes no `labels` and returns no loss. "Override `compute_loss`"
  understates the work: `SFTTrainer`'s data pipeline emits `input_ids`+`labels`,
  which this forward cannot consume, so a custom collator building the
  prompt/canvas/noise tensors is also required.

### What adoption would cost here

A new budget profile, a new model group on a different auto class, a custom
collator, a `run_*` entry, deepspeed (sdist-only, needs a CUDA toolchain, and
0.19.3 is younger than our 1-week floor), and a ZeRO-3 accelerate config. Two 80
GB GPUs are not even enough for full fine-tuning: bf16 weights alone are ~51.6
GB and AdamW state would add ~310 GB.

**Trigger to re-evaluate:** a transformers release containing PR 46572 **and** a
concrete experiment that needs block diffusion. Absent the second, this stays a
doc — the repo has no diffusion-LM machinery and no hypothesis that wants one.

## 2. Agent RL

### Why this matters here

This is the natural destination of the repo's own arc. 001 did off-policy SFT on
`pi-mono` coding-agent traces; the distill-traces skill turns verified traces
into data; 004 closed the generate → verify → SFT loop with a deterministic
verifier. Agent RL is that same loop with the verifier moved **inside a live
environment**: the model improves in the context it is actually used in.

### White-box — the reachable path, after three changes

The stable `GRPOTrainer` in trl 1.8.0 accepts `environment_factory` (verified by
signature inspection). In that mode TRL drives the loop itself: it parses the
model's tool calls, invokes the environment's public methods as tools, feeds
results back, and can take the reward from the environment's `get_reward()`. The
factory is duck-typed, not an OpenEnv import — any object with a callable
`reset`, public methods as tools, and an optional `get_reward()` qualifies. It
generates in-process, so **no vLLM sidecar and one GPU** is genuinely possible
(TRL falls back to `unwrapped_model.generate` when vLLM is off), and the local
`smoke_test=true` gate keeps working.

The first draft of this doc called that "almost free". It is not. Three
repo-side gaps, all verified by reading the installed trainer:

1. **`jmespath` is required and missing.** `GRPOTrainer` raises `ImportError` at
   construction when `tools` or `environment_factory` is set and jmespath is
   unavailable (it parses tool responses with it). TRL declares it only under
   its `dev` extra, so `trl ~= 1.8.0` does not pull it in. It becomes a new
   `[project].dependencies` entry — which must itself clear the 1-week age gate.
2. **No model we ship can do this.** `GRPOTrainer` hard-fails with
   `"The provided chat template does not support tool calling"`, and
   `supports_tool_calling()` returns `False` for **every** model in
   `configs/model/` — both SmolLM2 instruct sizes and the Gemma entries. The
   auto-repair path (`add_response_schema`) runs _after_ that raise and only
   knows a fixed set of templates. A tool-calling base is required; the upstream
   examples use `Qwen/Qwen3-0.6B` / `Qwen3-1.7B`.
3. **Our own lane forbids env-supplied rewards.** `grpo_reward_funcs`
   ([rewards.py](../src/training/trl/rewards.py)) raises when
   `trainer.reward_funcs` is empty, and `run_grpo` calls it unconditionally —
   but TRL explicitly supports the environment-as-reward-source case. The guard
   has to learn about `environment_factory`.

Also note `trl.experimental.openenv` exists in 1.8.0 but is **deleted in 1.9.0**
(the integration moved onto the stable trainer), so build against
`GRPOTrainer(environment_factory=…)` and not that module. And environments ship
as Docker images / HF Spaces, so standing the env up is its own task: `echo` (no
vLLM) is the plumbing proof; `wordle` as shipped at v1.8.0 sets `use_vllm=True`
and wants `trl vllm-serve`, so it is not the single-GPU starter it looks like.

### Loop-owning — blocked

This is the one the announcement describes: take opencode/codex/Claude Code, let
it own its loop, and train the model underneath it.

1. Each rollout runs the real agent CLI in a sandbox against a task.
2. An in-sandbox proxy forwards the agent's LLM calls to your vLLM server,
   injecting `logprobs=True` so it can capture per-turn `(token_ids, logprobs)`.
   (The proxy also strips them back out before returning, so the agent never
   sees them — that behavior lives in OpenEnv's `interception.py`, not in TRL's
   example.)
3. A held-out verifier scores the resulting workspace — in the example, the
   fraction of hidden tests the written `solution.py` passes.
4. `HarnessRolloutWorker` rebuilds training rows from the proxy trace and
   `AsyncGRPOTrainer` trains on them, weights pushed back to the vLLM server
   over NCCL every `weight_sync_steps`.

Three application-owned hooks carry all the judgment: `rollout_reward_fn` (the
example binarizes the verifier and adds degeneracy penalties — a rollout that
never ran `bash` scores -0.1, killing blind-write and give-up behavior),
`train_turn_fn` (reinforce only action turns, not prose), and `agent_turn_fn`
(drop the agent's own bookkeeping calls — title generation, context summaries —
which are a different task and must not be trained on).

**Blocked because** `openenv_harness.py` is absent from trl 1.8.0 **and from the
v1.9.0 tag** — there is no released TRL that can run it. Beyond that:
`AsyncGRPOTrainer` takes a model **id string** with no `eval_dataset` and no
`peft_config` (so `_run_trl` cannot host it, and it is full-parameter RL only);
it hardcodes flash-attn3 because training runs padding-free, so there is **no
CPU/MPS path and therefore no local smoke gate**; vLLM is boxed to 0.22–0.23
while the latest is 0.25.1, and has no macOS wheels; and `AsyncGRPOConfig` has
no `beta`, so the `kl` it logs is the ratio against the _rollout_ policy, not a
reference model.

**Trigger to re-evaluate:** a TRL release shipping `openenv_harness`. Reward
shaping and turn-selection judgment learned on the white-box path transfer
directly when it unblocks.

**If the white-box path gets built:** experiment 005 on the `trl_grpo` lane —
`echo` first to prove the plumbing, then a verifier-scored environment — with a
tool-calling base (Qwen3-0.6B class), single GPU. It would also be the first
experiment to actually exercise `trl_grpo`, which has shipped since v1 and has
never been run: its `reward_variance` verify check has never fired against real
data.

## 3. What this exercise exposed in our own code

Two findings independent of whether either lane is ever adopted:

- **`stderr.log` is clobbered under multi-GPU launch.** `run_with_stderr_tee`
  ([run.py:106](../src/training/trl/run.py)) is the only instrumentation
  **writer** in `_run_trl` with no rank guard at any level, and it opens the
  file in `"w"` mode. (`apply_tracking_env` and the `TRLAlertCallback`
  construction are also unguarded at the call site, but the callback self-guards
  on `state.is_world_process_zero` internally.) Under the
  `accelerate launch --num_processes N` path that
  `.claude/skills/train-llm/references/compute-lanes.md` documents, every rank
  truncates and interleaves the single file that feeds verify's `stderr_scan`.
  Fixing it needs a small design call — per-rank files with the check globbing
  all of them — rather than a bare rank guard, which would leave a non-zero
  rank's traceback captured nowhere.
- **A new `trainer.kind` silently disables two verify checks.** `_LM_TASKS` and
  `_GENERATION_TASKS` in [verify.py](../src/intern/verify.py) are allow-lists,
  so an unknown task string SKIPs `loss_plausibility` _and_ SKIPs (rather than
  FAILs) `generation_sanity` when `samples.jsonl` is missing. A future lane can
  therefore print `OVERALL: PASS` having produced no evidence about model output
  at all. Any new lane must add itself to those tuples — or the gate should fail
  closed on unknown tasks.
