# Colab compute lane

Use this workflow for `compute=colab`. The config is an agent-read parameter
sheet, like the other compute configs. It does not allocate a runtime. Read
`docs/012-colab-compute.md` for the design and upstream sources.

Never write results.md or report success unless `intern.py verify` exited 0. A
failed gate means the run failed, regardless of loss.

## 1. Install and authenticate locally

Install the local CLI from the locked `colab` dependency group:

```bash
uv sync --frozen --group colab --no-install-project
uv run --frozen --group colab colab version
```

Use Python 3.12 or later for this group. The project uses Python 3.13. Pass
`--group colab` to each local CLI and helper command. Omit this group from
remote commands because the training runtime does not need the CLI.

Complete this login in a human-controlled terminal:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Pass `--auth adc` explicitly on every CLI command. Upstream documentation and
the implementation disagree about the default. `colab auth` supplies VM-side
credentials; it does not fix the local CLI login. Never start interactive
`auth`, `drivemount`, `console`, or `repl` commands from a headless agent.

## 2. Select and provision the runtime

Replace `NNN-slug` and `NNN` throughout this reference. Run these commands from
the project root. Read the experiment's journal first.

```bash
COLAB_EXPERIMENT=NNN-slug
COLAB_SESSION=intern-NNN-path-1
COLAB_STATE=trash/colab/sessions.json
mkdir -p trash/colab
uv run --frozen --group colab python scripts/python/NNN-slug.py --cfg job compute=colab
uv run --frozen --group colab python scripts/python/intern.py check --experiment NNN
uv run --frozen --group colab python scripts/python/intern.py budget --experiment NNN can-launch
```

If a gate exits nonzero, stop before allocation. Use serial paths within one
experiment. Recover its state before another path starts. Use separate sessions
for concurrent experiments. Set `compute.session` in the experiment config or
the recorded overrides. Match all CLI arguments to the composed `compute`
values. The helper does not read Hydra configs.

The GPU selectors are `T4`, `L4`, `G4`, `H100`, and `A100`. Validate the exact
selector before allocation: an unknown selector can silently select A100. Use
the default T4 unless the plan needs another GPU:

```bash
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" new -s "$COLAB_SESSION" --gpu T4
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" status -s "$COLAB_SESSION"
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" url -s "$COLAB_SESSION"
```

For `compute.gpu=null`, omit `--gpu`. Use CPU for smoke tests only. For
`compute.high_mem=true`, add `--high-mem` to `new`. Account entitlement and
capacity control allocation. L4 ignores the high-memory flag. Record the
allocated hardware, rather than the requested hardware, in run.md.

The repo has no XLA trainer lane, so TPU training is unsupported. The CLI has no
runtime-version selector. Never infer access from the browser's accelerator
list. If allocation fails, report the failure before a hardware change.

## 3. Transfer the project

Archive the current working files, including uncommitted changes. Include the
root marker and the experiment's control files. Exclude secrets and old outputs:

```bash
tar --exclude='.env*' --exclude='.git' --exclude='.venv' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  --exclude='ckpts' --exclude='logs' --exclude='traces' \
  --exclude='metrics.jsonl' --exclude='verify.md' --exclude='results.md' \
  -czf trash/colab/workspace.tar.gz \
  .project-root README.md src configs scripts pyproject.toml uv.lock \
  "experiments/$COLAB_EXPERIMENT"
tar -tzf trash/colab/workspace.tar.gz | head -n 30
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" upload -s "$COLAB_SESSION" \
  trash/colab/workspace.tar.gz /content/intern-workspace.tar.gz
```

Inspect the archive for credentials before upload. Transfer a required local
dataset separately. Keep its path consistent with the data config. Record the
local Git revision and archive checksum in run.md, because the remote workspace
has no Git metadata. Exclude local caches or large experiment data when the
experiment directory contains them.

Extract the archive in a fresh directory. Bootstrap uv only if the image lacks
it. The example pins the bootstrap tool to version 0.12.1:

```bash
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" exec -s "$COLAB_SESSION" --timeout 300 <<'PY'
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

workspace = Path('/content/my-little-ml-intern')
workspace.mkdir(exist_ok=False)
with tarfile.open('/content/intern-workspace.tar.gz') as archive:
    archive.extractall(workspace, filter='data')
if shutil.which('uv') is None:
    installer = urllib.request.urlopen('https://astral.sh/uv/0.12.1/install.sh').read()
    subprocess.run(['sh'], input=installer, check=True)
PY
```

Raw `colab exec` can exit 0 after a Python failure. Confirm setup through the
helper below. If bootstrap or extraction failed, fix that failure first. Do not
repeat extraction over an existing workspace.

For private models or tracking, upload a minimal `.env` separately to the remote
project root. Include only the credentials that this experiment needs. Never
copy the workstation's full `.env`. Never put secrets in commands, Hydra
overrides, or kernel code: Colab records execution history. The `--config` flag
isolates session metadata, not credentials or history.

## 4. Set up and probe the locked environment

Use the helper for each command whose exit code matters:

```bash
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command 'uv sync --frozen --python 3.13'
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command 'uv run --frozen bash scripts/bash/gpu_probe.sh'
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command 'uv run --frozen python -c "import torch; assert torch.cuda.is_available(); print(torch.ones(1, device=\"cuda\").sum().item())"'
```

The helper defaults to `/content/my-little-ml-intern` and a 10800-second command
timeout. Pass `--remote-dir` and `--timeout-s` when the config differs. If uv
installed into `~/.local/bin`, use `/root/.local/bin/uv` in commands when the
kernel's PATH omits that directory. Confirm the location on the runtime.

For QLoRA, add `--group gpu` to sync and every subsequent `uv run` command. Add
`--group async` only when the selected async trainer needs it. Keep `--frozen`
on all remote uv commands. Never replace the locked torch with Colab's
preinstalled torch to resolve a compatibility error.

Omit the CUDA assertion for a CPU runtime. Check the C compiler and Python
headers for Triton. Probe CUDA from this uv environment, rather than from the
notebook kernel. The Linux lockfile targets CUDA 12.9. A successful `nvidia-smi`
does not prove that torch can execute. If the CUDA probe or smoke test fails,
stop before scale.

The helper parses the command into arguments without a shell. Shell operators,
environment assignments, and variable expansion require a separate script. The
helper preserves remote exit codes through a downloaded status record. Exit 2
without a confirmed outcome requires inspection before a retry. A confirmed
subprocess timeout returns 124. A timeout does not stop the VM.

## 5. Smoke, gate, and train

Run the local smoke test when the trainer supports CPU or MPS. Run the smoke
test on the allocated runtime in every case:

```bash
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" --log-name smoke.log \
  --command "uv run --frozen python scripts/python/NNN-slug.py compute=colab compute.session=$COLAB_SESSION smoke_test=true"
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" download -s "$COLAB_SESSION" \
  "/content/my-little-ml-intern/experiments/$COLAB_EXPERIMENT/logs/smoke.log" \
  trash/colab/smoke.log
tail -n 20 trash/colab/smoke.log
```

Require both exit 0 and `VERDICT: TRAIN_OK`. Complete the normal preflight. Use
the remote workspace as the sole writer of budget.md, ledger.md, and journal.md
until recovery. Keep run.md current locally. Record the exact commands,
including preparation, setup, gates, and teardown.

Run each command below separately. If any command exits nonzero, stop scale:

```bash
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command 'uv run --frozen python scripts/python/intern.py budget --experiment NNN can-launch'
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command 'uv run --frozen python scripts/python/intern.py budget --experiment NNN record-launch'
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command 'uv run --frozen python scripts/python/intern.py ledger --experiment NNN upsert --path-id path-1 --status running --approach baseline'
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --timeout-s 10800 --log-name train.log \
  --command "uv run --frozen python scripts/python/NNN-slug.py compute=colab compute.session=$COLAB_SESSION smoke_test=false"
```

Size the timeout from the run estimate plus a 30% buffer. The example value is
not a runtime lifetime guarantee. Colab can reclaim the VM earlier. For a retry,
use `can-retry` and `record-retry` with the failed path's ID. Use the normal
notification events only when the workflow authorizes them.

Commands run synchronously. Use another terminal for a bounded status or log
check. Download train.log to `trash/colab/train.log`, then read its tail. Do not
redirect into stderr.log; the adapter owns that file. If a connection or command
times out, inspect the runtime and log before a retry. Descendant processes can
survive even a confirmed timeout with exit 124. After a transport failure, the
command itself can remain active.

## 6. Recover, verify, and stop

Recover artifacts after success and after failure. Archive the experiment
directory on the remote. Keep the current local run.md during recovery:

If training exits nonzero, mark the path as failed. Recover its artifacts for
diagnosis. Never use smoke-only metrics as evidence that the real run passed.

```bash
uv run --frozen --group colab python scripts/python/intern.py colab-exec --experiment NNN \
  --session "$COLAB_SESSION" --state-file "$COLAB_STATE" \
  --command "tar --exclude=run.md -czf /content/intern-artifacts.tar.gz experiments/$COLAB_EXPERIMENT"
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" download -s "$COLAB_SESSION" \
  /content/intern-artifacts.tar.gz trash/colab/artifacts.tar.gz
uv run --frozen --group colab python - "$COLAB_EXPERIMENT" <<'PY'
from pathlib import Path
import shutil
import sys
import tarfile

name = sys.argv[1]
stage = Path('trash/colab/recovered')
stage.mkdir(exist_ok=False)
with tarfile.open('trash/colab/artifacts.tar.gz') as archive:
    archive.extractall(stage, filter='data')
recovered = stage / 'experiments' / name
local = Path('experiments') / name
backup = Path('trash/colab/before-recovery') / name
if not recovered.is_dir() or backup.exists():
    raise ValueError('Require a recovered experiment and a fresh backup path')
shutil.copy2(local / 'run.md', recovered / 'run.md')
backup.parent.mkdir(parents=True, exist_ok=True)
local.rename(backup)
recovered.rename(local)
PY
uv run --frozen --group colab python scripts/python/intern.py verify --experiment NNN
uv run --frozen --group colab python scripts/python/intern.py check --experiment NNN
```

Require each transfer command to pass before local verification. Use fresh
recovery and backup paths for each download. Replace the local directory; never
merge old outputs into the recovered artifacts. Inspect metrics, samples,
checkpoints, and the latest journal entries before teardown. The helper's exit 0
proves only that its command passed; the verify gate decides the experiment
outcome.

The Contents API transfers whole files through memory. For large checkpoints,
split the archive into smaller files before download. Download every part.
Reassemble and verify the archive locally before extraction. If recovery fails,
keep the named session and state file for recovery. Report the failure and the
continuing compute cost. Never stop before the required artifacts return.

After artifact recovery, stop the session even when verification fails:

```bash
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" stop -s "$COLAB_SESSION"
uv run --frozen --group colab colab --auth adc --config "$COLAB_STATE" sessions
uv run --frozen --group colab python scripts/python/intern.py budget --experiment NNN record-gpu-h --hours <GPU_HOURS>
```

Confirm that the owned session is absent. Do not stop unrelated sessions. Count
the full GPU allocation interval, including setup, smoke, and idle time. Record
zero GPU-hours for CPU. Record the final ledger status through the normal
verify-run workflow. Update run.md with the confirmed teardown. Write results.md
only after the verify gate exits 0.
