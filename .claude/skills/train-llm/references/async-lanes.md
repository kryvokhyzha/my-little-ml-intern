# Async lanes — `trl_async_grpo` and `trl_async_distill`

Read this file before you launch an asynchronous lane. In these lanes a vLLM
server generates rollouts while the trainer trains. New weights go to the server
over NCCL every `weight_sync_steps`. Generation and training overlap, so a
GPU-hour buys more steps than on `trl_grpo` / `trl_distill`.

## When to use an async lane

- Use it when generation dominates the wall-clock time (long completions,
  multi-turn tool rollouts, many generations per prompt).
- Use the synchronous lane (`trl_grpo`, `trl_distill`) for a first run, a small
  model, or any run that must smoke on CPU.
- The async lanes accept slightly stale rollouts. `max_staleness` bounds the age
  in weight versions. Staleness is a new variable: do not compare an async run
  with a sync run as a one-variable path.
- Both lanes are `trl.experimental`. TRL can remove an experimental trainer in
  any release, patch releases included, without a deprecation cycle. Run
  `tests/test_trl_lanes.py` after every relock that moves `trl`, patch bumps
  too. The test checks the kwargs plumbing on every upgrade. It cannot run the
  trainers on CPU.

## Hard requirements

- CUDA and FSDP2. There is no CPU or MPS path, so `smoke_test=true` runs on the
  GPU box, not on the laptop.
- The trainer loads the model by name (`model.main._args_[0]`) with the
  `kernels-community/flash-attn3` attention. Run `uv sync --group async` for the
  `kernels` package. The `async` group pins the range that the pinned
  transformers accepts (5.17: `>=0.16,<0.17`). Another version makes the trainer
  raise ImportError at construction (verified on 2026-09-27).
- Check the attention kernel first. Run
  `uv run --no-sync python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3', version=1)"`.
  If it fails, the Hub has no build for this torch version or GPU architecture.
  Stop, because both async lanes need it. This check passed on an NVIDIA L4
  (compute capability 8.9) with torch 2.13.0+cu129 on 2026-09-27: the Hub served
  a `torch-stable-abi29-cu128-x86_64-linux` build, and a SmolLM2-135M forward
  pass through it returned finite bf16 logits. So the kernel is not limited to
  Hopper. vLLM remains the untested part of both async lanes.
- The `vllm` package. Every vLLM release pins torch exactly: 0.22–0.26 need
  torch 2.11, and 0.27.1–0.30 need torch 2.13 (this repo's pin). The async NCCL
  weight sync needs vLLM ≥ 0.22. So with torch 2.13 use vLLM 0.27.1–0.29.0
  (0.30.0 clears the 1-week floor on 2026-09-29). A wider range lets the
  installer replace torch. vLLM is not in the lockfile and has no macOS wheels.
- The CUDA build of vLLM must match the CUDA build of torch. The PyPI vLLM
  wheels (0.26 and later) are CUDA 13.0 builds, and they load `libcudart.so.13`.
  This repo locks the Linux torch from the `cu129` index. The PyPI wheel on that
  torch mixes two CUDA runtimes, and nobody has tested that mix here. So install
  the `+cu129` wheel from the vLLM GitHub release page. Check the exact file
  name on that page first. Install it on the GPU box only, inside the synced
  environment. First install torchvision from the cu129 index, because the vLLM
  wheel otherwise pulls the PyPI `+cu130` torchvision:
  `uv pip install --index-url https://download.pytorch.org/whl/cu129 --no-deps torchvision==0.28.0`.
  Then install vLLM:
  `uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"`.
- vLLM 0.28 and later need huggingface_hub ≥ 1.27, and 0.29 needs ≥ 1.28. The
  repo pins 1.32, so this holds only after `uv lock` picked up that pin.
- After the vLLM install, start every command with `uv run --no-sync`. A plain
  `uv run` syncs the environment back to the lock and changes the packages that
  vLLM installed (numpy, setuptools, huggingface_hub).
- Run `vllm serve` from that same environment, so the trainer and the server
  share one vLLM version (the weight-sync protocol changed in vLLM 0.28).
- A running vLLM server before `train()` starts. The trainer polls `/health` and
  fails after `vllm_server_timeout` seconds.
- No `eval_dataset` (set `data.eval: null`) and no reference model. The `kl`
  metric compares the policy with the rollout policy, not with a reference.
- `trl_async_distill` takes no `trainer.peft`. `trl_async_grpo` takes LoRA since
  TRL 1.14 (see "LoRA" below).
- Do not set `trainer.args.use_liger_kernel`. The async trainers raise
  NotImplementedError for it.
- Tools on `trl_async_grpo` must be module-level functions (TRL registers them
  by `__name__`). Sync tools and environment methods run concurrently on a
  thread pool of `max_inflight_tasks` workers, so shared state must be
  thread-safe. Async tools work since TRL 1.14.
- Both async lanes force `attn_implementation="kernels-community/flash-attn3"`.
  FlashAttention-3 caps the head dimension at 256. Gemma 4's global-attention
  layers use head_dim 512, so a Gemma 4 model on an async lane is unverified and
  likely fails at the first forward pass. See `gemma.md`.

## Precision

- TRL's default `dtype` is `float32`: the precision at which TRL measured the
  training-inference mismatch. The presets keep it.
- `bfloat16` halves the memory but widens that mismatch. If you choose it, set
  `trainer.args.dtype: bfloat16` AND `vllm serve --dtype bfloat16`. The values
  must match; TRL logs a warning when they differ.

## Start the servers

Student / policy server (both lanes), on port 8000. `VLLM_SERVER_DEV_MODE=1`
exposes the weight-load and pause endpoints, so bind to localhost when the
trainer runs on the same box, and never expose the port outside a private
network:

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve <model repo> --host 127.0.0.1 --dtype float32 \
  --weight-transfer-config '{"backend":"nccl"}' \
  --logprobs-mode processed_logprobs --max-logprobs -1
```

Teacher server (`trl_async_distill` only), on port 8001:

```bash
vllm serve <teacher repo> --host 127.0.0.1 --port 8001 --logprobs-mode processed_logprobs --max-logprobs -1
```

- The teacher must share the student tokenizer. For several teachers
  (multi-teacher on-policy distillation), list them in
  `trainer.args.teacher_server_urls` and add a `teacher_id` column that routes
  each row.
- `trl vllm-serve` still works, but it is a deprecated wrapper around the same
  command.

## LoRA (`trl_async_grpo`)

1. Set `trainer.peft` (a `peft.LoraConfig` node, as in `trl_sft_lora`).
2. Start the server with
   `--enable-lora --max-lora-rank <R> --max-loras <max_staleness + 2>` and
   `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`. R must be one of 1, 8, 16, 32, 64, 128,
   256, 320, 512, and at least the adapter rank.
3. The trainer writes each adapter version to
   `${experiment_dir}/ckpts/.vllm_lora`, and the server reads it from disk. When
   the server runs on another host, both hosts need that directory on a shared
   filesystem.
4. Keep `save_strategy` on for a real run. `.vllm_lora` is a pruned serving
   cache. Only a checkpoint keeps the adapter, and `output_dir` must outlive the
   job.

LoRA serving generates completions about 1.33-1.39x slower than merged sync (TRL
docs). Prefer merged sync when generation dominates the wall-clock time.

## Schedules and epochs

- With `max_steps: -1`, `trl_async_distill` stops after `num_train_epochs` ×
  `len(data.train)` prompts (TRL ≥ 1.13). The step count is then only a ceiling,
  so the presets use a constant LR. Set `max_steps` > 0 for a decaying schedule.
- Multi-turn rows can fork when the re-tokenized history drifts by more than
  `fork_threshold_tokens` (default 1024). Templates that drop past reasoning
  (Qwen3) fork more often.
- A weight sync that takes longer than `weight_sync_timeout` (1800 s) fails the
  run with `VERDICT: TRAIN_FAIL`. Raise it for very large models.

## Launch

1. Start the server(s). Record the exact commands in run.md under `## Setup`.
2. Run the smoke test on the box: `... smoke_test=true`. It must print
   `VERDICT: TRAIN_OK`.
3. Run the budget gate, then launch as in the train-llm workflow.

## Monitor

- Learning: `reward`, `reward_std`, `kl`, `entropy`, `clip_ratio/*` (GRPO);
  `jsd`, `teacher_jsd/<id>`, `teacher_token_frac/<id>` (distillation).
- Generation-bound: `perf/rollout_wait_s` > 0 and growing. Add server GPUs or
  lower `max_completion_length`.
- Trainer-bound: `rollout/backpressure_s` > 0 with a full
  `sample/rollout_queue_size`.
- Staleness: `sample/staleness_max` and `sample/dropped_stale_total` against
  `max_staleness`.
- With `tracking=trackio`, `trl_async_grpo` logs full rollout traces (prompts
  and completions) as `rollouts`. Keep the Space private, or use `tracking=none`
  for sensitive data.
- The adapter writes post-run samples through the trained model when it can.
  When that fails, `generation_sanity` fails the gate. Load the saved checkpoint
  and call `training.sampling.write_samples` on it — real generations from the
  trained weights. Then read them before you write a JUDGMENT line.
- `data_consumption` SKIPs on both async lanes: their trainers report no token
  count.

## Sync lanes with vLLM

`trl_grpo`, `trl_rloo`, and `trl_distill` can also generate with vLLM
(`trainer.args.use_vllm: true`; `vllm_mode: colocate` on one GPU, or `server`
with the command above). TRL 1.14.0's server client needs vLLM ≥ 0.26, which the
torch-2.13 range above already satisfies. Keep it off for CPU smokes. Set
`--dtype` to `model.main.dtype` for these sync lanes, or omit `--dtype`. The
command above pins `--dtype float32` for the async trainers' float32 default;
copying it verbatim serves a `bfloat16` model (for example Gemma 4) in fp32. For
a 4-bit (QLoRA) model with `use_vllm` on vLLM ≥ 0.28, also install
`vllm-bnb-plugin~=0.0.3`: vLLM 0.28 moved bitsandbytes support out of its core
package.
