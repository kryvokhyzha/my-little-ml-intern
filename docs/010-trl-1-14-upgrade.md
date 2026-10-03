# 010 — TRL 1.8 → 1.14 and the dependency refresh

<!-- Upgrade record for 2026-09-27. Sources: the GitHub release notes of TRL
v1.9.0–v1.14.0, a commit-by-commit review of v1.8.0..v1.14.0 (519 commits,
six reviewers, each finding checked against the installed source), and one
changelog-plus-git-diff audit per dependency group. Every claim marked
"verified" ran for real in this sandbox: CPU only, no Hub access, no GPU. -->

## Decisions

The table lists every package whose pin moved, plus the holds. An "A/B" means
the same seeded run on both versions; "bit-identical" means every logged metric
matched except wall-clock time.

| Package                 | Before  | After        | Decision and evidence                                                                                                                                                                                                                  |
| ----------------------- | ------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| trl                     | 1.8.0   | **1.14.1**   | user-approved exception to the 1-week floor (below)                                                                                                                                                                                    |
| transformers            | 5.14.1  | **5.17.0**   | two breaking changes handled (below); every CPU lane A/B bit-identical                                                                                                                                                                 |
| tokenizers (transitive) | 0.22.2  | 0.23.2       | forced by transformers ≥ 5.16; 74 tokenizer probes identical; one trap, unused here (below)                                                                                                                                            |
| peft                    | 0.19.1  | **0.21.0**   | LoRA and QLoRA A/B bit-identical; adapters cross-load between versions                                                                                                                                                                 |
| accelerate              | 1.14.0  | **1.15.0**   | A/B bit-identical; fixes FSDP2 tied embeddings (torch 2.13) and the FSDP2 LoRA resume state. The advisory GHSA-4j2p-28q2-5m79 still affects 1.15.0: its fix is unmerged, and no repo path calls `load_checkpoint_in_model`             |
| datasets                | 5.0.0   | **5.0.1**    | same rows in the same order; fixes a path-traversal advisory. New: it reads JSON with `id`/`source`/`model`/`system_prompt` + `messages` columns as an agent trace (dataset-formats.md has the opt-out)                                |
| lightning               | 2.6.5   | **2.6.6**    | security-only patch (`load_from_checkpoint` code execution); A/B bit-identical                                                                                                                                                         |
| huggingface_hub[hf_xet] | 1.24.0  | **1.32.0**   | only together with trackio 0.38.1: hub 1.32 breaks `import trackio` below 0.38.1                                                                                                                                                       |
| trackio                 | 0.31.5  | **0.38.1**   | same logged metrics; see the alert and static-Space bugs below                                                                                                                                                                         |
| wandb                   | 0.28.1  | **0.30.0**   | same loss offline                                                                                                                                                                                                                      |
| hydra-core              | 1.3.3   | **1.3.7**    | `instantiate` security advisory fixed; composition and the 000 smoke identical                                                                                                                                                         |
| pandas, joblib, dotenv  | 3.0.3 … | 3.0.6 …      | pandas 3.0.4 was yanked; pandas arrives through datasets, the entrypoints use dotenv to read .env, and nothing imports joblib                                                                                                          |
| pydantic, packaging     | floors  | 2.13.5, 26.3 | floor = latest eligible; gate tooling only                                                                                                                                                                                             |
| setuptools              | ≥ 75.6  | **≥ 84.0**   | the locked 81.0.0 carries an advisory (sdist builds only) fixed in 83.0.0. This deviates from the audit's hold: vLLM 0.29/0.30 cap setuptools below 81, so the GPU-box vLLM install downgrades it (async-lanes.md: `uv run --no-sync`) |
| bitsandbytes (`gpu`)    | 0.49.2  | **0.50.2**   | CPU QLoRA smoke: `TRAIN_FAIL` on 0.49.2, `TRAIN_OK` on 0.50.2; not bit-identical (5.935518 vs 5.935561)                                                                                                                                |
| kernels (new `async`)   | ad hoc  | **0.16.2**   | enters uv.lock at the pending relock; must match the transformers `kernels` extra                                                                                                                                                      |
| ruff + pre-commit revs  | mixed   | 0.16.8 …     | the hook and the lint pin agree; the locked ruff follows at the relock; codespell 2.4.3, nbstripout 0.9.1                                                                                                                              |
| hf-transfer             | 0.1.9   | removed      | huggingface_hub ≥ 1.24 ignores it; `HF_XET_HIGH_PERFORMANCE` replaces it                                                                                                                                                               |
| torch                   | 2.13.0  | 2.13.0       | **hold** 2.14 (below)                                                                                                                                                                                                                  |
| safetensors             | 0.8.0   | 0.8.0        | current                                                                                                                                                                                                                                |

**The lockfile is current** (2026-09-27). uv.lock pins trl 1.14.0, transformers
5.17.0, and bitsandbytes 0.50.2 across 227 packages, and `uv lock --check`
exits 0. The GPU box installed the environment from this lock with
`uv sync --frozen`. The results in the section below this one ran against the
target versions installed from PyPI, not from the lock; the GPU results ran from
the lock.

Left for a separate decision: prettier 3.9 (drops the shell plugin and reflows
files), the CI actions (checkout v7, setup-uv v10), and a pinned uv version.
None of them can run here, and each changes CI behavior.

### The TRL and openenv exceptions

The user approved TRL 1.14.0 two days after its release (2026-09-27), then TRL
1.14.1 and openenv 0.7.0 on 2026-10-03. Each package has two entries:

- `[tool.intern.deps.exceptions]`: `trl = "2026-10-06"` and
  `openenv = "2026-10-08"`. Each date is the day the release turns seven days
  old. `intern.py deps` reads `[project].dependencies` only, so it checks the
  trl entry and ignores the openenv entry; openenv sits in the `async` group.
- `[tool.uv] exclude-newer-package`: `trl` and `openenv`, both
  `"2026-10-03T00:00:00Z"`. A timestamp, not `false`: a release published after
  the approval date stays blocked.

Remove the trl entries after 2026-10-06 and the openenv entries after
2026-10-08.

**trl 1.14.1** changes four things (release notes, compare v1.14.0...v1.14.1).
Two fix vLLM server mode: a crash at the first weight sync on vLLM 0.20–0.25,
and one completion per sample after a tool call instead of `num_generations`.
Two fix the CLI scripts, which this repo does not use. No recorded number moved:
526 tests pass, `tests/test_trl_lanes.py` passes, and the 000 smoke gives
3.604706287384033 again.

**openenv 0.7.0** splits `openenv.core.harness` into a package and adds RFC 006
(harness interception with token-faithful traces for TRL). TRL 1.14.1 imports
`HarnessAdapter`, `HarnessRunLimits`, `ModelStepResult`, and
`ResourceSessionFactory` from that package; all four still resolve, and
`trl.experimental.async_grpo.openenv_harness` imports. openenv 0.7.0 adds 52
packages (gradio, fastmcp, mcp) to the `async` group only and moves `tomlkit`
from 0.15.1 to 0.14.0 there, because gradio caps it. The default install gains
nothing.

### Why torch stays on 2.13

Three facts block 2.14, and each is independent:

1. **No CUDA 12.9 build.** PyTorch 2.14 ships cu126, cu130, and cu132. The repo
   locks Linux torch from the `pytorch-cu129` index, so `torch ~= 2.14.0` is
   unsatisfiable on Linux (from the 2.14.0 binary build matrix).
2. **vLLM pins torch exactly.** vLLM 0.27–0.30 pin `torch==2.13.0`; no release
   pins 2.14. Installing vLLM would downgrade torch, which breaks both async
   lanes and `use_vllm` on the sync lanes.
3. **The flash-attn3 Hub kernel needs a matching build.** `kernels` accepts a
   build for the installed torch version or a stable-ABI build. Nobody has
   checked either for 2.14, and both async lanes need the kernel.

On CPU, 2.14 is harmless: the full suite, a Lightning fit, and TRL SFT, DPO,
KTO, and GRPO runs were bit-identical to 2.13. Lift the hold when a vLLM release
that pins 2.14 is a week old and the Linux index moves to cu130. AGENTS.md now
names these two checks, because `intern.py deps` reads PyPI only.

### transformers 5.17: what changed for this repo

- **transformers 5.15.0 removed `warmup_ratio` and `logging_dir`.** The
  `trl_sft`, `trl_gkd`, and 001 configs now use `warmup_steps: 0.03`. A float
  below 1 is a ratio, and it gives the same schedule on 5.14 and 5.17.
- **`gradient_checkpointing_kwargs` gained `offload` and `every_n_layers`**
  (5.17.0). The Trainer pops them, but TRL 1.14 re-enables checkpointing with
  the raw dict inside GRPO, RLOO, DPO, and KTO. Verified: GRPO then crashes
  mid-run with "Unexpected keyword arguments". `build_args` now refuses these
  keys on those four lanes before the model trains; `trl_sft` accepts them.
- **The async lanes need `kernels>=0.16,<0.17`.** The transformers 5.15+
  `kernels` extra moved; another version raises ImportError at construction.
- **Gemma 4 configs moved to `per_layer_config`** (5.15.0). A full Gemma 4
  checkpoint saved on 5.15+ does not load on 5.14. A LoRA adapter, like 001's,
  is unaffected.
- **Long-context training needs ≥ 5.17.0.** TRL's 1M-context recipe uses
  gradient-checkpointing `offload`, first released in 5.17.0.
- PR 46572 shipped in 5.15.0, so DiffusionGemma now supports gradient
  checkpointing. docs/009 records that half its trigger is met.

Verified: the full suite, a seeded A/B of every CPU TRL lane, and a Gemma 4 text
forward/backward/generate gave bit-identical numbers on 5.14.1 and 5.17.0 (+
tokenizers 0.23.2). Only wall-clock time differed.

**The tokenizers trap.** tokenizers 0.23.x loses tokens in stride-overflow
windows: 30 tokens with `max_length=16, stride=2` gave windows of 16 and 8, not
16 and 16. Nothing here uses `return_overflowing_tokens`. The pretraining
reference now says to chunk ids in Python instead.

## TRL 1.9–1.14, filtered to what this repo calls

| Version | Change                                                                                                                                                                                                                                                                | Effect here                                                                                                                                     |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.9     | Environment-owned datasets: with `environment_factory`, `reset()` supplies the prompt                                                                                                                                                                                 | `data.train: null` works only with `max_steps` > 0 and a `reset()` that returns text; environments.md says so                                   |
| 1.9     | SFT drops fully masked rows after truncation                                                                                                                                                                                                                          | fewer rows than planned; no recorded dataset had one                                                                                            |
| 1.9     | DPO (`keep_start`) and KTO drop rows whose prompt alone reaches `max_length`                                                                                                                                                                                          | a `planned_tokens` shortfall on long prompts; the preset headers say so                                                                         |
| 1.9     | GKD: EOS no longer masked when pad == eos; in-prompt eos tokens keep attention; on-policy sampling forces `top_p=1`                                                                                                                                                   | **003 is not comparable** (below)                                                                                                               |
| 1.9     | GRPO dapo/cispo/vespo normalizer excludes truncated completions, only when `mask_truncated_completions=true`                                                                                                                                                          | none: no preset sets the flag; 1.8 already excluded tool-result tokens                                                                          |
| 1.9     | GRPO/RLOO `kl` and `entropy` become global token-weighted means; RLOO `clip_ratio/low_min`/`high_max` become 0/1 flags; GRPO `clip_ratio/low_min`/`high_max` become min/max over per-sequence means; KTO `logits/chosen` and `logits/rejected` become per-token means | multi-GPU values change meaning; single-GPU `kl` does not, so `kl_ref` is unaffected; the GRPO and KTO metric changes apply on a single GPU too |
| 1.9     | `DistillationTrainer`: `lmbda` and the off-policy branch removed; loss divides by generated tokens; full-vocab JSD                                                                                                                                                    | `trl_distill` is fully on-policy; do not compare its loss with ≤ 1.8 numbers                                                                    |
| 1.9.1   | Gemma 4 tool-call parsing; dapo normalization fixed when `steps_per_generation` ≠ GAS                                                                                                                                                                                 | Gemma 4 can run `trl_grpo_env`; the presets leave spg = GAS                                                                                     |
| 1.10    | `DistillationTrainer` goes stable; `messages` data and every length cap removed; `top_p` 0.95 → 1.0, `disable_dropout` True → False                                                                                                                                   | `trl_distill` pins both; prompts are never truncated, so trim them in data prep                                                                 |
| 1.10    | GRPO and RLOO `max_completion_length` 256 → 512; GRPO `use_bias_correction_kl` → True                                                                                                                                                                                 | none: both presets pin 256; the KL term runs only when `beta` > 0 (the preset comment says so)                                                  |
| 1.10    | AsyncGRPO OpenEnv harness; jmespath needed only on transformers < 5.13                                                                                                                                                                                                | docs/009's release blocker is gone; the harness stays manual                                                                                    |
| 1.11    | SFT's chunked loss applies Gemma 4's `final_logit_softcapping` (≤ 1.10 read it from the wrong config)                                                                                                                                                                 | **001 is not comparable** (below)                                                                                                               |
| 1.11    | `AsyncDistillationTrainer`; vLLM's native server replaces TRL's; `DistillationTrainer` accepts `tools`                                                                                                                                                                | new lane `trl_async_distill`; `trl_distill` wires `trainer.tools`                                                                               |
| 1.11    | SDPO now warns that it ignores `privileged_context` unless `include_environment_feedback` is true (1.8 ignored it silently)                                                                                                                                           | `trl_sdpo` sets the flag to false explicitly                                                                                                    |
| 1.12    | A byte-for-byte duplicate of 1.11 except the version string                                                                                                                                                                                                           | none                                                                                                                                            |
| 1.13    | PPO removed; chunked-CE projection keeps the hidden dtype, softcap and log-softmax move to fp32                                                                                                                                                                       | 001 under bf16 autocast: identical GEMM, last-digit loss change                                                                                 |
| 1.13    | Async lanes need vLLM ≥ 0.22; `AsyncDistillation` stops after `num_train_epochs` of prompts                                                                                                                                                                           | vLLM 0.27.1–0.29 on torch 2.13; the preset uses a constant LR                                                                                   |
| 1.14    | Fused Triton log-prob/entropy kernel on CUDA, fp32 included, no opt-out                                                                                                                                                                                               | it computes the loss of DPO, KTO, GRPO, RLOO, SDFT, SDPO, and SSD on GPU; run each GPU smoke once                                               |
| 1.14    | AsyncGRPO takes LoRA; its tools run on a thread pool and must be module-level                                                                                                                                                                                         | `trl_async_grpo` accepts `trainer.peft`; async-lanes.md lists the constraints                                                                   |
| 1.14    | KTO (#7247) and DPO (#7243) all-reduce under plain DDP — **only with `use_liger_kernel=True`**                                                                                                                                                                        | none: no preset sets the flag, so earlier multi-GPU runs were correct                                                                           |
| 1.14    | `router_aux_loss_coef` reads the MoE model config; a non-zero value raises on any model whose config has no `output_router_logits` (dense models, and Gemma 4 26B-A4B)                                                                                                | none: no preset sets it                                                                                                                         |
| 1.14    | Removed: `GRPOWithReplayBuffer`, BCO, PRM, XPO, NashMD, GSPO-token; experimental trainers may go without deprecation                                                                                                                                                  | seven lanes are experimental (below)                                                                                                            |

The trainer signatures that `_run_trl` calls did not change, and every field in
the shipped presets exists.
`test_every_trl_preset_instantiates_against_installed_trl` re-checks the second
claim on every upgrade.

## Which recorded numbers stay comparable

| Experiment    | Lane      | Verdict            | Why                                                                                                                                                                      |
| ------------- | --------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 000, 002, 004 | `trl_sft` | comparable         | The chunked SFT loss is unchanged in fp32. The 1.9 row filter drops fully masked rows, and these datasets have none.                                                     |
| 001           | QLoRA SFT | **not comparable** | TRL ≤ 1.10 computed Gemma 4's loss on uncapped logits: the softcap sits in `text_config`, and TRL read the top-level config. 1.11+ applies it. Re-run the baseline.      |
| 003           | `trl_gkd` | **not comparable** | With pad == eos, 1.8 masked every eos-id token from attention (the prompt's turn separators too) and dropped the generated EOS from the loss. 1.9 also forces `top_p=1`. |

Both results.md files record the boundary. The first draft of this document
called 001 comparable. The commit-level review found the softcap change, which
the release notes do not mention.

From now on the question answers itself: every `run_start` carries an `env`
stamp (library versions and the git commit), verify.md prints it as an `ENV:`
line, and the journal logs it per run.

## What we adopted

On-policy learning, self-distillation, and environments were the brief. Every
lane below smokes on CPU except the two async lanes.

| Lane                | TRL class                                  | The mechanism in one line                                                                |
| ------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `trl_distill`       | `trl.DistillationTrainer` (stable)         | The student samples; a same-vocabulary teacher grades each token (generalized JSD).      |
| `trl_gold`          | `trl.experimental.gold.GOLDTrainer`        | GKD across tokenizers: ULD compares sorted distributions and merges token spans.         |
| `trl_sdft`          | `trl.experimental.sdft.SDFTTrainer`        | The model, shown a privileged context, is its own teacher — learning without forgetting. |
| `trl_sdpo`          | `trl.experimental.sdpo.SDPOTrainer`        | Rewards select successful rollouts; an EMA self re-prompted with them distills back.     |
| `trl_ssd`           | `trl.experimental.ssd.SSDTrainer`          | SFT on the model's own raw samples — no teacher, no verifier.                            |
| `trl_rloo`          | `trl.RLOOTrainer` (stable)                 | Online RL with the leave-one-out baseline; the GRPO reward contract.                     |
| `trl_grpo_env`      | `trl.GRPOTrainer(environment_factory=...)` | Multi-turn tool rollouts; the environment owns the reward through `get_reward()`.        |
| `trl_async_grpo`    | `trl.experimental.async_grpo`              | A vLLM server generates while the trainer trains (GPU only).                             |
| `trl_async_distill` | `trl.experimental.async_distillation`      | On-policy distillation with the student and teacher(s) as vLLM servers (GPU only).       |

Three pieces of plumbing make them usable rather than merely importable:

- **Environments.** `trainer.environment_factory`, `trainer.tools`, and
  `trainer.env_spec` (Harbor / OpenReward) on the GRPO lanes. The reward guard
  accepts an empty `reward_funcs` when an environment supplies the reward.
  `env_spec` normalizes the single reward callable that the real specs return.
- **Verifiable data.** `data.self_distill.build_prompt_dataset` and
  `arithmetic_reward` put 004's task pool behind every new lane, with the same
  train/eval split.
- **On-distribution probes.** `generation_sanity` samples from the run's own
  data: a conversational `prompt` goes through the chat template, and a
  `messages` row is cut before its last assistant turn. A raw "Once upon a time"
  probe put chat models off-distribution and produced garbage samples.

## Guardrails added with the upgrade

- `generation_sanity` fails closed: only `trl_dpo`, `trl_kto`, `lightning`, and
  `axolotl` may skip samples.
- `completion_termination` fails a run whose final `completions/clipped_ratio`
  is ≥ 0.95: the policy stopped emitting EOS.
- `training_signal` fails a run that trained on nothing. Verified: an SSD run
  with the default `filter_empty` dropped every short arithmetic answer, logged
  a loss of 0 for 60 steps, and passed the old gate. It now fails.
- `reward_variance` reads SDPO's `self_distillation/reward_std`.
- `data_consumption` reads TRL's `num_tokens` on the rollout lanes and skips the
  lanes that count nothing, instead of a false FAIL.
- A setup failure (model load, column check, trainer construction) now prints
  `VERDICT: TRAIN_FAIL` and reaches the journal.
- LoRA initialization follows `cfg.seed` (TRL builds the adapter before the
  Trainer seeds), and the Lightning lane now seeds too.
- `generation_sanity` reads the new `completion` field of samples.jsonl (the
  output without the prompt). It fails an empty completion and runs the
  repetition proxies on a completion of 10+ tokens. With chat probes, `text` is
  mostly prompt: RLOO and SDPO runs that collapsed to an immediate EOS passed
  the old check. A re-run SDPO collapse now fails.
- `build_args` refuses the `gradient_checkpointing_kwargs` keys `offload` and
  `every_n_layers` on GRPO, RLOO, DPO, and KTO, before the model trains.
- `tests/test_trl_lanes.py` trains every CPU lane for one step on a random
  2-layer Qwen3, and checks the async plumbing with a fake trainer.

Two tracking bugs predate the upgrade; the hub/tracking audit found both:

- **trackio alerts were silently lost.** `fire_alert` passed a string level.
  trackio swallowed the resulting AttributeError, so the alert never reached the
  store and the loguru fallback never ran. A test now fires an alert through the
  real trackio library and reads it back. wandb alerts had the same defect in a
  milder form: they always arrived as INFO.
- **A Hub push could publish the metrics.** With `tracking=trackio`,
  transformers syncs the metrics to a static Space on every push, public unless
  `hub_private_repo=True`. `build_args` now sets `trackio_static_space_id=False`
  unless the config sets it.

## What ran, and what it proves

This sandbox has no GPU and cannot reach the Hub, so no run used a real
checkpoint. The runs below use a 0.86M-parameter Qwen3 with a digit-level
tokenizer and the Qwen3 chat template. The 000 entrypoint trained it from random
init on two-digit addition: 1500 steps reached **100% exact match** on the 60
held-out tasks. Its checkpoints at step 300 (5%) and step 950 (15%) serve as the
weak and mid bases. One held-out task is 1.7 points, so a move of one or two
tasks is noise.

**Test suites.** 526 tests pass on the old package set and on the new one. Every
CPU TRL lane trains one step in the suite.

**Cross-version A/B.** The package audits ran seeded runs on both sides of each
bump: every CPU TRL lane (transformers, peft, accelerate, datasets), a Lightning
fit and SFT/DPO/KTO/GRPO (torch 2.14), and the SFT and Lightning smokes under
each tracking backend. All were bit-identical, except bitsandbytes 0.50 (above).

**The real entrypoints.** 000, 002, 003, and 004 ran end to end: smoke
(`VERDICT: TRAIN_OK`), `budget can-launch`, the run, `verify`, the ledger, and
the journal. The gate failed all four tiny runs, and each FAIL was correct:

- 000 and 002: a loss below 1.0 on an LM task, the leakage red flag that real
  vocabularies need. 000 also wrote samples shorter than 50 characters.
- 003: samples shorter than 50 characters.
- 004: an eval–train gap of 4.4. The path ran 100 steps on 37 accepted rows,
  about 43 epochs. The retry went through `budget can-retry`, and the journal
  recorded the decision.

**Every CPU lane, many steps**, through the `run_*` entry points and the gate
CLI:

| Lane           | Base | Steps | Held-out exact match | Gate | Note                                                               |
| -------------- | ---- | ----- | -------------------- | ---- | ------------------------------------------------------------------ |
| `trl_distill`  | weak | 120   | 0.05 → 0.017         | PASS | teacher = the 100% model                                           |
| `trl_gkd`      | weak | 120   | 0.05 → 0.00          | PASS | samples answer with numbers after the probe fix                    |
| `trl_gold`     | weak | 120   | 0.05 → 0.067         | PASS |                                                                    |
| `trl_grpo`     | mid  | 60    | 0.15 → 0.10          | PASS |                                                                    |
| `trl_rloo`     | mid  | 60    | 0.15 → 0.10          | PASS |                                                                    |
| `trl_sdft`     | mid  | 60    | 0.15 → 0.05          | PASS |                                                                    |
| `trl_sdpo`     | mid  | 60    | 0.15 → 0.017         | PASS | the teacher signal died mid-run (below)                            |
| `trl_ssd`      | mid  | 60    | 0.15 → 0.15          | FAIL | `filter_empty` dropped every sample; `training_signal` fails it    |
| `trl_ssd`      | mid  | 60    | 0.15 → 0.15          | PASS | `filter_empty: false`                                              |
| `trl_grpo_env` | tool | 30    | —                    | PASS | real `guess` tool calls: 2.1 → 1.1 per rollout; reward 0.44 → 0.40 |

A 300-step sweep at higher learning rates collapsed five runs. GRPO at 5e-4 and
2e-3 looped to the length cap, and `completion_termination` fails both. RLOO at
2e-3, SDPO at 5e-4, and GRPO at 2e-4 (temperature 0.7) emitted EOS at once. The
old gate passed the RLOO and SDPO collapses; re-run under the new gate, both
fail on empty completions. SDFT (5e-4), distill, and GKD (2e-3) did not
collapse: they still answer with numbers, although SDFT fell from 0.15 to 0.00
on the held-out set.

**Does the RL loop optimize at all?** A sparse 0/1 reward gives this model
almost no signal: most groups have zero reward spread, so the gradient is zero,
and Adam's early updates are mostly noise. With a dense reward on the same task
(closeness of the predicted sum), the loop raises the reward it sees:

| Run           | Reward, first 60 → last 60 steps | Held-out exact match |
| ------------- | -------------------------------- | -------------------- |
| GRPO, lr 1e-4 | 0.76 → **0.90**                  | 0.15 → 0.10          |
| GRPO, lr 5e-5 | 0.85 → 0.89                      | 0.15 → 0.083         |
| RLOO, lr 5e-5 | 0.87 → 0.87                      | 0.15 → 0.067         |

The rise is far above the sampling noise (about ±0.005 per window). The policy
satisfied the proxy without exact answers, which is the expected failure of a
proxy reward, not of the plumbing.

**What is proven:** the lanes train, log, sample, and pass or fail the gate as
designed; the RL loop optimizes its reward; SFT learns the task from scratch;
the version bumps change no CPU number.

**What is not proven:**

- A held-out gain from any on-policy, RL, or self-distillation lane. At this
  scale none moved exact match beyond noise. That needs a real base model.
- Real checkpoints (SmolLM2, Qwen3-0.6B, Gemma 4): huggingface.co is blocked.
- vLLM, both async lanes, and FSDP2. flash-attn3 itself now works (GPU run
  below); vLLM is the remaining async blocker.
- Multi-GPU anything. The GPU run used one L4.

## The GPU run (2026-09-27)

One NVIDIA L4 (compute capability 8.9, driver 580, `gcc` present) in
us-central1-a, installed from uv.lock with `uv sync --frozen`. europe-west4
stocked out in all three zones.

- **flash-attn3 loads and runs on Ada.** `get_kernel` served a
  `torch-stable-abi29-cu128-x86_64-linux` build, and a SmolLM2-135M forward pass
  through it returned finite bf16 logits. The kernel is not Hopper-only.
- **QLoRA trains on CUDA.** `TRAIN_OK`, loss 3.9596, with 4-bit nf4 + LoRA.
  bitsandbytes 0.50.2 prints "No prebuilt binary for CUDA 12.9, loading CUDA
  12.8 instead" and then trains, as the decision table predicted.
- **Every GPU lane smokes.** 14 lanes reached `VERDICT: TRAIN_OK`: `trl_sft`,
  `trl_sft_lora`, `trl_sft_qlora`, `trl_dpo`, `trl_kto`, `trl_gkd`, `trl_gold`,
  `trl_distill`, `trl_sdft`, `trl_sdpo`, `trl_ssd`, `trl_grpo`, `trl_rloo`, and
  `trl_grpo_env`. The 1.14 Triton kernel compiled on first use. `trl_sft` gave
  1.5570449829101562 against 1.5570447444915771 on CPU — the float difference of
  the device, not a change of behavior.
- **`trl_grpo_env` needs a tool-calling chat template.** Every SmolLM2 model
  fails it with "The provided chat template does not support tool calling". Use
  `model=qwen3_0_6b`, which passes.
- **One real path passed the gate.** 60 steps of `trl_grpo` on the arithmetic
  pool with `arithmetic_reward`: `TRAIN_OK`, loss 0.1936, and `verify` exit 0 (6
  passed, 0 failed, 5 skipped). reward 0.1375 → 0.35 over 12 points, reward_std
  0.2852 → 0.4698, `completions/clipped_ratio` 0.025 → 0.0. The three written
  samples answered 48+91, 32+84, and 68+50 correctly. The reward moves inside
  its own noise band, so this proves the plumbing, not a learned gain.
- **Lightning is not GPU-smoked.** Its `module` and `datamodule` default to
  null, so the lane carries no runnable default.

## What we left out, and why

- `trl.experimental.iw_opd`, `minillm`, and `server_distillation` —
  `trl_distill` and `trl_async_distill` cover the method family. Add one when a
  plan names a hypothesis that needs it.
- The OpenEnv loop-owning harness — it needs `openenv`, vLLM, and CUDA, and each
  agent needs its own hooks. The environments reference documents the wiring.
- vLLM as a dependency — no macOS wheels, an exact torch pin, and CUDA 13 PyPI
  wheels against this repo's cu129 torch. The async reference installs the
  `+cu129` wheel on the GPU box.
- `use_liger_kernel` — since 1.14 it applies Liger's model kernels through
  transformers and TRL's own chunked log-prob loss in DPO, KTO, and GRPO. It
  needs `liger-kernel ≥ 0.8.2` (Linux only), and the async lanes reject it. Try
  it as a one-variable path when a GPU run hits memory.

## Follow-ups

- DONE 2026-09-27: the lock is regenerated and `uv lock --check` exits 0. 526
  tests pass, and the 000 smoke still gives 3.604706287384033 — the same value
  as before the upgrade.
- DONE 2026-09-27: the flash-attn3 check, the QLoRA smoke, and one smoke per GPU
  lane all passed. See "The GPU run".
- After 2026-10-06: remove the two trl exception entries in pyproject.toml.
- After 2026-10-08: remove the two openenv exception entries in pyproject.toml.
- Keep a C compiler on any GPU box: the 1.14 Triton kernel compiles on first
  use. A CUDA runtime-only image fails with `Failed to find C compiler`. The
  `common-cu129-ubuntu-2204-nvidia-580` image ships `gcc`.
- Re-run the 001 and 003 baselines before comparing any new run with them.
- Seven lanes (`trl_gkd`, `trl_gold`, `trl_sdft`, `trl_sdpo`, `trl_ssd`, both
  async lanes) import from `trl.experimental`, which TRL can delete in any
  release, patch releases included. Run `tests/test_trl_lanes.py` after every
  relock that moves trl, patch bumps too. Move 003-style work from `trl_gkd` to
  the stable `trl_distill`.
- torch 2.14: see "Why torch stays on 2.13".
- docs/009's multi-GPU `stderr.log` clobbering is still open.
- DONE 2026-09-27 for `trl_grpo` on SmolLM2-135M-Instruct: see "The GPU run".
  Still open for `trl_distill`, and for any base model large enough to move the
  held-out metric. Do this before anyone claims the new lanes work.
