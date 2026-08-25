# Compute lanes

Each `configs/compute/*.yaml` file defines one lane. The experiment config's
`compute:` group selects the lane, and `cfg.compute.kind` names the active one.

Status: **local** and **ssh** are v1-complete lanes. **hf_jobs**, **modal**, and
**vast** are config stubs — the YAML holds placeholders, and no adapter connects
them to a launch yet. This file documents what a manual launch needs, so that
you can still use those lanes deliberately.

Every lane shares the same artifact contract: checkpoints in
`experiments/NNN-<slug>/ckpts/`, `logs/train.log`, `logs/stderr.log`,
`logs/samples.jsonl`, and metrics in `metrics.jsonl`. If a lane cannot write
there directly, plan how the artifacts come back before you launch (preflight.md
line "Artifacts").

## local (`configs/compute/local.yaml`) — v1

Always probe the hardware first:

```bash
scripts/bash/gpu_probe.sh
```

The output is `key=value`: `cuda=yes|no|unknown`, `mps=yes|no|unknown`, plus
`gpu_count=`, `gpu_name=`, `vram_gb=`. Decide from that output, not from an
assumption:

- `cuda=yes` → real runs are OK. Size the batch and the precision from
  `hardware.md`.
- `mps=yes`, `cuda=no` (Apple Silicon) → **smoke runs and tiny runs only**. Use
  fp32 for the smoke run (`hardware.md` gives the MPS caveats).
- `cuda=no mps=no` (CPU only) → smoke runs only.
- `unknown` → the probe failed. Treat the machine as CPU-only until you prove
  otherwise.

Launch a long run detached. The command sends the combined output to the
experiment's log:

```bash
mkdir -p experiments/NNN-<slug>/logs
nohup uv run python scripts/python/NNN-<slug>.py \
  > experiments/NNN-<slug>/logs/train.log 2>&1 &
```

Do not shell-redirect into `logs/stderr.log`. The adapter owns that file through
its own per-run truncating tee.

## ssh (`configs/compute/ssh.yaml`) — v1

The config ships `host: null`, `user: null`, and `remote_dir: null`. Set all
three before anything else. Override them in the experiment config, or per run
with `compute.host=... compute.user=... compute.remote_dir=...`. The sequence:

1. **Sync the code** — exactly the `compute.sync_paths` list (by default `src`,
   `configs`, `scripts`, `pyproject.toml`, and `uv.lock`):

   ```bash
   rsync -az --delete src configs scripts pyproject.toml uv.lock <user>@<host>:<remote_dir>/
   ```

2. **Sync the env**: `ssh <user>@<host> "cd <remote_dir> && uv sync"`. Use
   `uv sync --group gpu` instead when the run uses QLoRA or bitsandbytes. The
   `gpu` dependency group is CUDA-only, and you never install it locally.
3. **Probe**:
   `ssh <user>@<host> "cd <remote_dir> && bash scripts/bash/gpu_probe.sh"`.
4. **Run the smoke test on the remote** (`smoke_test=true`). The local smoke run
   does not cover the remote CUDA and precision code.
5. **Launch detached**, so that the run survives the SSH session:

   ```bash
   ssh <user>@<host> "cd <remote_dir> && mkdir -p experiments/NNN-<slug>/logs && \
     nohup uv run python scripts/python/NNN-<slug>.py \
     > experiments/NNN-<slug>/logs/train.log 2>&1 &"
   ```

   (The same rule as the local lane: `logs/stderr.log` is the adapter's per-run
   tee, not a shell redirect target.)

6. **Read the log with `tail`** — never run `cat`, and never stream the log
   continuously:
   `ssh <user>@<host> "tail -n 20 <remote_dir>/experiments/NNN-<slug>/logs/train.log"`,
   `grep -c 'loss'` for progress, `tail -n 50 .../stderr.log` when you suspect a
   failure.
7. **Copy the artifacts back** after the run. The verify gate runs against the
   local experiment directory:

   ```bash
   rsync -az <user>@<host>:<remote_dir>/experiments/NNN-<slug>/ experiments/NNN-<slug>/
   ```

## Security defaults (ssh and every ssh-derived lane)

- **Key-only SSH.** `configs/compute/ssh.yaml` carries `port`, `identity_file`,
  and `ssh_opts` (with `-o IdentitiesOnly=yes` and
  `-o StrictHostKeyChecking=accept-new`). These settings remove the password
  prompt. They also remove the interactive host-key question that hangs a
  headless run. After you set them, the ssh and rsync sequence becomes:

  ```bash
  SSH="ssh -p <compute.port> -i <compute.identity_file> <compute.ssh_opts>"
  rsync -az --delete -e "$SSH" src configs scripts pyproject.toml uv.lock <user>@<host>:<remote_dir>/
  $SSH <user>@<host> "cd <remote_dir> && uv sync"
  ```

- **Non-root remote user.** Set `compute.user` to a regular account. A root
  remote user turns any rsync `--delete` typo or leaked key into a machine-wide
  incident.
- **vast.ai ssh flavors:** an instance offers direct SSH (the instance's own
  IP:port) and proxied SSH (through vast's shared gateway). Prefer direct SSH
  when the offer exposes it. The ssh.yaml key settings apply unchanged in both
  cases.
- **Cloud boxes:** restrict the security group or firewall to inbound SSH from
  your current IP only. Scanners reach a GPU box with port 22 open to the world
  within minutes.
- **Never echo a token.** Export `HF_TOKEN` and the other token variables from
  the remote's environment or from `.env`. A token inline on an ssh command line
  lands in the shell history and in the teed train.log.

## hf_jobs (`configs/compute/hf_jobs.yaml`) — config stub

Placeholders: `flavor: a10g-small`, `timeout: 3h`, `secrets: [HF_TOKEN]`. No
adapter submits from `cfg.compute` yet. You launch manually with the `hf` CLI
and a self-contained uv script (PEP 723 inline deps). The job filesystem is
ephemeral, and it cannot see this repo. The script must therefore be
inline-complete, or the job must fetch it by URL. The job MUST push the results
to the Hub, or the results are lost:

```bash
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN <script-url-or-path>
```

- Put the flags before the script argument. The flag is `--secrets` (plural).
- The script must set `push_to_hub=True` and `hub_model_id`. It MUST also set
  `hub_private_repo=True`. A Trainer Hub push is public by default, and a
  job-side push bypasses the `intern.py publish` gate completely. The flag is
  therefore mandatory, not advisory.
- Size `--timeout` to the run, plus a 30% buffer. The 3h default is a
  placeholder, not a decision.
- After you submit the job, report the job id and the URL. Record the launch.
  **Do not poll.** Check `hf jobs logs <job-id>` or `hf jobs inspect <job-id>`
  when the user asks, or when a milestone is due. Copy the final metrics by hand
  into `metrics.jsonl` and the ledger, so that the local gates still work.

## modal (`configs/compute/modal.yaml`) — config stub

Placeholders: `gpu: A10G`, `timeout_s: 10800`. A real launch needs four parts:

- a Modal app file that mounts or copies the repo, runs `uv sync`, and invokes
  the experiment entrypoint on `cfg.compute.gpu`
- `modal` in the dev deps
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`
- a volume or a Hub push that returns the artifacts

None of that exists yet. If you choose this lane, treat the launcher as the
first task to build. Otherwise use the ssh lane or the local lane.

## vast (`configs/compute/vast.yaml`) — config stub

Placeholders: `gpu_name: RTX_4090`, `num_gpus: 1`, `max_price_per_hour: 0.8`. A
real launch needs the `vastai` CLI and `VAST_API_KEY` in `.env`. Search for the
offers that match the config
(`vastai search offers 'gpu_name=RTX_4090 num_gpus=1 dph<0.8'`). Create an
instance. **The lane then becomes ssh**: reuse the ssh sequence above against
the instance's host and port. Destroy the instance when the run finishes. An
idle instance still bills, so count that spend against `compute_cap_gpu_h` even
though the gate does not. No adapter automates this lane yet.

## Multi-GPU (TRL lane, single node)

```bash
uv run accelerate launch --num_processes <N> scripts/python/NNN-<slug>.py trainer.args.bf16=true
```

- The Hydra overrides pass through unchanged after the script argument. Never
  combine `accelerate launch` with the hydra `-m` multirun.
- The instrumentation is rank-zero-safe. The TRL adapter restricts metrics.jsonl
  (run_start + meta), samples.jsonl, and the final VERDICT line to the main
  process. The alert callback acts only on world process zero. Expect one clean
  artifact set, not N interleaved copies.
- Keep the per-path gpu_min accounting correct. A path on N GPUs for T minutes
  costs N × T gpu-minutes in the ledger and in `record-gpu-h`. The wall-clock
  time alone understates the spend by a factor of N.

## Axolotl multi-GPU / multi-node notes

The axolotl lane renders the recipe locally, and a remote box runs it (usually
through the ssh sequence above). Multi-node training is deferred. The ssh lane
is single-host. Rendezvous orchestration across boxes is out of scope. Any path
that needs multi-node must first clear the `scale_ceiling_params` justification
in budget.md. The multi-GPU (single-node) specifics:

- **A DeepSpeed or ZeRO config is a standalone JSON file that the rendered YAML
  references by file path** (`deepspeed: configs/compute/zero2.json`). Axolotl
  does not inline it. Keep the JSON next to `configs/compute/`, so that the
  default `compute.sync_paths` rsync copies it too. A rendered YAML that points
  at a JSON file that never reached the remote box fails at launch, not at
  render.
- **Known-good Gemma-class perf block** for `trainer.overrides` (verified
  against the lapa-llm Gemma-3-12B production configs — liger plugin, flash
  attention, packing):

  ```yaml
  plugins:
    - axolotl.integrations.liger.LigerPlugin
  liger_rope: true
  liger_rms_norm: true
  liger_glu_activation: true
  liger_layer_norm: true
  liger_fused_linear_cross_entropy: true
  flash_attention: true
  sample_packing: true
  pad_to_sequence_len: true
  ```

- **Warning:** express dataset upsampling through the weight or config options.
  Never duplicate a `datasets:` block. (The duplication is an observed
  anti-pattern — the same dataset pasted 4–5×.) Duplication hides the real
  mixture from review. It multiplies the preprocessing. It also changes the
  epoch accounting silently.
