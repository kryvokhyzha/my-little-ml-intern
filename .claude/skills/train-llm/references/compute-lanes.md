# Compute lanes

One lane per `configs/compute/*.yaml`, selected by the experiment config's
`compute:` group; the active lane is `cfg.compute.kind`.

Honest status: **local** and **ssh** are v1-complete lanes. **hf_jobs**,
**modal**, and **vast** are config stubs — the YAML holds placeholders and no
adapter wires them into a launch yet; this file documents what a manual launch
takes so you can still use them deliberately.

Every lane shares the same artifact contract: checkpoints in
`experiments/NNN-<slug>/ckpts/`, `logs/train.log`, `logs/stderr.log`,
`logs/samples.jsonl`, metrics in `metrics.jsonl`. If a lane can't write there
directly, plan how the artifacts get back before launching (preflight.md line
"Artifacts").

## local (`configs/compute/local.yaml`) — v1

Probe hardware first, always:

```bash
scripts/bash/gpu_probe.sh
```

Output is `key=value`: `cuda=yes|no|unknown`, `mps=yes|no|unknown`, plus
`gpu_count=`, `gpu_name=`, `vram_gb=`. Decide from it, not from assumptions:

- `cuda=yes` → real runs OK; size batch/precision from `hardware.md`.
- `mps=yes`, `cuda=no` (Apple Silicon) → **smoke runs and tiny runs only**; fp32
  for smoke (MPS caveats in `hardware.md`).
- `cuda=no mps=no` (CPU only) → smoke runs only.
- `unknown` → the probe failed; treat as CPU-only until proven otherwise.

Launch (long runs detached, combined output into the experiment's log):

```bash
mkdir -p experiments/NNN-<slug>/logs
nohup uv run python scripts/python/NNN-<slug>.py \
  > experiments/NNN-<slug>/logs/train.log 2>&1 &
```

Do not shell-redirect into `logs/stderr.log` — the adapter owns it via its own
per-run truncating tee.

## ssh (`configs/compute/ssh.yaml`) — v1

Config ships `host: null`, `user: null`, `remote_dir: null` — set them (override
in the experiment config or per-run
`compute.host=... compute.user=... compute.remote_dir=...`) before anything
else. Sequence:

1. **Sync the code** — exactly the `compute.sync_paths` list (`src`, `configs`,
   `scripts`, `pyproject.toml`, `uv.lock` by default):

   ```bash
   rsync -az --delete src configs scripts pyproject.toml uv.lock <user>@<host>:<remote_dir>/
   ```

2. **Sync the env**: `ssh <user>@<host> "cd <remote_dir> && uv sync"` — use
   `uv sync --group gpu` instead when the run uses QLoRA/bitsandbytes (the `gpu`
   dependency group is CUDA-only and never installed locally).
3. **Probe**:
   `ssh <user>@<host> "cd <remote_dir> && bash scripts/bash/gpu_probe.sh"`.
4. **Smoke on the remote** (`smoke_test=true`) — local smoke does not cover
   remote CUDA/precision paths.
5. **Launch detached** so the run survives the SSH session:

   ```bash
   ssh <user>@<host> "cd <remote_dir> && mkdir -p experiments/NNN-<slug>/logs && \
     nohup uv run python scripts/python/NNN-<slug>.py \
     > experiments/NNN-<slug>/logs/train.log 2>&1 &"
   ```

   (Same rule as local: `logs/stderr.log` is the adapter's per-run tee, not a
   shell redirect target.)

6. **Tail etiquette** — never cat, never stream continuously:
   `ssh <user>@<host> "tail -n 20 <remote_dir>/experiments/NNN-<slug>/logs/train.log"`,
   `grep -c 'loss'` for progress, `tail -n 50 .../stderr.log` on suspicion.
7. **Bring artifacts home** after the run (verify runs against the local
   experiment dir):

   ```bash
   rsync -az <user>@<host>:<remote_dir>/experiments/NNN-<slug>/ experiments/NNN-<slug>/
   ```

## Security defaults (ssh and every ssh-derived lane)

- **Key-only SSH.** `configs/compute/ssh.yaml` carries `port`, `identity_file`,
  and `ssh_opts` (with `-o IdentitiesOnly=yes` and
  `-o StrictHostKeyChecking=accept-new`) — no password prompts, no interactive
  host-key question hanging a headless run. With them set, the ssh/rsync
  sequence becomes:

  ```bash
  SSH="ssh -p <compute.port> -i <compute.identity_file> <compute.ssh_opts>"
  rsync -az --delete -e "$SSH" src configs scripts pyproject.toml uv.lock <user>@<host>:<remote_dir>/
  $SSH <user>@<host> "cd <remote_dir> && uv sync"
  ```

- **Non-root remote user.** Set `compute.user` to a regular account — a root
  remote user turns any rsync `--delete` typo or leaked key into a machine-wide
  incident.
- **vast.ai ssh flavors:** instances offer direct SSH (the instance's own
  IP:port) and proxied SSH (through vast's shared gateway). Prefer direct when
  the offer exposes it; either way the ssh.yaml key settings apply unchanged.
- **Cloud boxes:** restrict the security group / firewall to inbound SSH from
  your current IP only — a GPU box with port 22 open to the world gets scanned
  within minutes.
- **Never echo tokens.** Export `HF_TOKEN` etc. from the remote's environment or
  `.env`; a token inline on an ssh command line lands in shell history and the
  teed train.log.

## hf_jobs (`configs/compute/hf_jobs.yaml`) — config stub

Placeholders: `flavor: a10g-small`, `timeout: 3h`, `secrets: [HF_TOKEN]`. No
adapter submits from `cfg.compute` yet — a launch is manual via the `hf` CLI
with a self-contained uv script (PEP 723 inline deps). The job filesystem is
ephemeral and cannot see this repo, so the script must be inline-complete or
fetched by URL, and results MUST be pushed to the Hub or they are lost:

```bash
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN <script-url-or-path>
```

- Flags before the script argument; `--secrets` (plural).
- The script must set `push_to_hub=True` + `hub_model_id` and MUST set
  `hub_private_repo=True` — Trainer Hub pushes are public by default, and a
  job-side push bypasses the `intern.py publish` gate entirely, so the flag is
  mandatory, not advisory.
- Size `--timeout` to the run (+30% buffer) — the 3h default is a placeholder,
  not a decision.
- After submitting: report the job id and URL, record the launch, and **do not
  poll** — check `hf jobs logs <job-id>` / `hf jobs inspect <job-id>` when the
  user asks or a milestone is due. Mirror final metrics back into
  `metrics.jsonl`/ledger by hand so local gates still work.

## modal (`configs/compute/modal.yaml`) — config stub

Placeholders: `gpu: A10G`, `timeout_s: 10800`. A real launch needs: a Modal app
file that mounts/copies the repo, runs `uv sync`, and invokes the experiment
entrypoint on `cfg.compute.gpu`; `modal` in dev deps;
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in `.env`; a volume or Hub push for
artifact return. None of that exists yet — treat choosing this lane as a task to
build the launcher first, or fall back to ssh/local.

## vast (`configs/compute/vast.yaml`) — config stub

Placeholders: `gpu_name: RTX_4090`, `num_gpus: 1`, `max_price_per_hour: 0.8`. A
real launch needs the `vastai` CLI (`VAST_API_KEY` in `.env`): search offers
matching the config
(`vastai search offers 'gpu_name=RTX_4090 num_gpus=1 dph<0.8'`), create an
instance, then **the lane becomes ssh** — reuse the ssh sequence above against
the instance's host/port, and destroy the instance when done (idle instances
bill against `compute_cap_gpu_h` in spirit if not in fact). No adapter automates
this yet.

## Multi-GPU (TRL lane, single node)

```bash
uv run accelerate launch --num_processes <N> scripts/python/NNN-<slug>.py trainer.args.bf16=true
```

- Hydra overrides pass through unchanged after the script path. Never combine
  `accelerate launch` with hydra `-m` multirun.
- Instrumentation is rank-zero-safe: the TRL adapter guards metrics.jsonl
  (run_start + meta), samples.jsonl, and the final VERDICT line to the main
  process, and the alert callback acts only on world process zero — expect one
  clean artifact set, not N interleaved copies.
- Keep per-path gpu_min accounting honest: a path on N GPUs for T minutes costs
  N × T gpu-minutes in the ledger and `record-gpu-h` — wall-clock alone
  understates the spend N-fold.

## Axolotl multi-GPU / multi-node notes

The axolotl lane renders locally and runs on a remote box (usually via the ssh
sequence above). Multi-node is deferred: the ssh lane is single-host, rendezvous
orchestration across boxes is out of scope, and any path that needs it must
first clear the `scale_ceiling_params` justification in budget.md. Multi-GPU
(single-node) specifics:

- **DeepSpeed/ZeRO configs are standalone JSON files referenced by path** from
  the rendered YAML (`deepspeed: configs/compute/zero2.json`) — axolotl does not
  inline them. Keep the JSON next to `configs/compute/` so it rides along with
  the default `compute.sync_paths` rsync; a rendered YAML pointing at a JSON
  that never reached the remote box fails at launch, not at render.
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

- **Warning:** express dataset upsampling via weights/config options, never by
  duplicating `datasets:` blocks (an observed anti-pattern in the wild — the
  same dataset pasted 4–5×). Duplication hides the real mixture from review,
  multiplies preprocessing, and silently changes epoch accounting.
