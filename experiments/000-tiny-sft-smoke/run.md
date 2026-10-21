# Run — 000-tiny-sft-smoke

Exact commands executed. `local` lane — no Provision/Teardown.

## Lane

compute=local · instance: CPU (GPTNeoX/pythia diverges on MPS even in fp32; the
winning path pins CPU + fp32 — see the ledger retry chain).

## Setup

```bash
uv sync
```

## Train

```bash
uv run python scripts/python/000-tiny-sft-smoke.py smoke_test=true   # VERDICT: TRAIN_OK
uv run python scripts/python/000-tiny-sft-smoke.py                   # winner: SmolLM2-135M, CPU, 10 steps, lr 5e-5, fp32
uv run python scripts/python/intern.py verify --experiment 000-tiny-sft-smoke
```

## Benchmark

Inline only: 3 seeded generation samples (`logs/samples.jsonl`) + the smoke-scale
train loss, checked by the verify gate. No held-out eval (`data.eval: null` for
the tiny synthetic corpus).

## Colab T4 smoke — 2026-10-20

The user requests one real Colab smoke run, artifact recovery, and runtime teardown.
The baseline archive is `trash/colab/live-20261020/baseline.tar.gz`.
The run manifest is `trash/colab/live-20261020/manifest.json`.
Keep the historical CPU outputs and results unchanged.

### Local prerequisites

The existing ADC login authenticates the Colab API. No browser login is necessary.
Colab reports zero active assignments before allocation.
The isolated CLI environment uses Python 3.12 and Colab CLI 0.7.4.
The project environment remains unchanged.

```bash
UV_TOOL_DIR="$PWD/trash/colab/tools" UV_TOOL_BIN_DIR="$PWD/trash/colab/bin" UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache \
  uv tool install --exclude-newer '7 days' 'google-colab-cli==0.7.4'
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/sessions.json" sessions
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/sessions.json" usage
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py check --experiment 000
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py budget --experiment 000 can-launch
```

### First allocation

The parameter lookup reports 134515008 parameters. The budget gate permits the path.
The checked archive contains 145 members and 274799 bytes.
Its SHA256 is `1ead333415e567ba21f34a29c048a136aac31870b9129f3f9c9b4fb338262b57`.
The archive excludes credentials, old metrics, logs, checkpoints, and results.

```bash
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python trash/colab/live-20261020/pack.py
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  new -s intern-000-t4-cb347936 --gpu T4
```

Colab reports `Session READY`. Automatic approval review then rejects the archive
upload because it requires explicit approval of the payload and destination.
The project never reaches the runtime. No training command executes.
The user receives a pending request for that upload approval.

### Teardown after the blocked upload

```bash
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  stop -s intern-000-t4-cb347936
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" sessions
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  budget --experiment 000 record-gpu-h --hours 0.025849
```

Colab reports `Session terminated`. The allocation upper bound is 0.025849 GPU-hours,
from the allocation request until local teardown completion. The budget records it once.
The final session list reports `No active sessions found on server`.
The baseline results remain unchanged. No new results.md or verification report exists.

### Approved continuation

The user explicitly approves the project archive upload and smoke continuation.
Reuse the checked archive without changes. Allocate a fresh owned session:

```bash
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  new -s intern-000-t4-674b67a2 --gpu T4
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  status -s intern-000-t4-674b67a2
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  upload -s intern-000-t4-674b67a2 trash/colab/live-20261020/workspace.tar.gz /content/intern-workspace.tar.gz
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  exec -s intern-000-t4-674b67a2 --timeout 360 -f trash/colab/live-20261020/bootstrap.py
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  download -s intern-000-t4-674b67a2 /content/intern-bootstrap.json trash/colab/live-20261020/bootstrap-status.json
```

The runtime reports T4 hardware and the Standard machine shape.
The bootstrap confirms the workspace checksum and returns exit 0.
The kernel uses Python 3.13.15. The image supplies `/usr/local/bin/uv`.
The runtime has 193.31 GiB of free disk space before dependency setup.

### Bootstrap source

The executed bootstrap file contains:

```python
from pathlib import Path
from hashlib import sha256
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request

status = {"returncode": 2}
try:
    source = Path("/content/intern-workspace.tar.gz")
    expected = "1ead333415e567ba21f34a29c048a136aac31870b9129f3f9c9b4fb338262b57"
    if sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("The workspace checksum differs from the local archive")
    workspace = Path("/content/my-little-ml-intern")
    workspace.mkdir(exist_ok=False)
    with tarfile.open(source) as archive:
        archive.extractall(workspace, filter="data")
    executable = shutil.which("uv")
    if executable is None:
        with urllib.request.urlopen("https://astral.sh/uv/0.12.1/install.sh", timeout=60) as response:
            installer = response.read()
        subprocess.run(["sh"], input=installer, env={**os.environ, "UV_NO_MODIFY_PATH": "1"}, timeout=300, check=True)
        executable = str(Path.home() / ".local/bin/uv")
    subprocess.run([executable, "--version"], timeout=30, check=True)
    status = {
        "returncode": 0,
        "uv": executable,
        "kernel_python": sys.version.split()[0],
        "free_disk_gb": round(shutil.disk_usage("/content").free / 2**30, 2),
        "workspace_sha256": expected,
    }
except Exception:
    traceback.print_exc()
finally:
    Path("/content/intern-bootstrap.json").write_text(json.dumps(status), encoding="utf-8")
```

### Locked setup

The remote command installs only runtime dependencies. Keep `--no-dev` on later uv commands.

```bash
PATH="$PWD/trash/colab/bin:$PATH" UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache \
  uv run --no-sync python scripts/python/intern.py colab-exec --experiment 000 \
  --session intern-000-t4-674b67a2 --state-file trash/colab/live-20261020/sessions.json \
  --timeout-s 1200 --log-name setup.log \
  --command '/usr/local/bin/uv sync --frozen --python 3.13 --no-dev'
```

The helper confirms setup exit 0. The first invocation exposes a generated-code
SyntaxWarning. A local helper fix assigns the log path before the conditional.
Its regression test reproduces the warning before the fix and passes afterward.

### CUDA and compiler probe

Execute this code through the helper with
`/usr/local/bin/uv run --frozen --no-dev python -c <code>` and `--log-name probe.log`:

```python
import json, shutil, sysconfig
from pathlib import Path
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
device = torch.cuda.get_device_name(0)
assert "T4" in device, device
value = torch.ones(4, device="cuda").sum().item()
assert value == 4.0, value
compiler = shutil.which("gcc") or shutil.which("clang")
headers = (Path(sysconfig.get_path("include")) / "Python.h").is_file()
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": device,
            "tensor_sum": value,
            "compiler": compiler,
            "python_headers": headers,
            "architectures": torch.cuda.get_arch_list(),
        }
    ),
    flush=True,
)
assert compiler and headers, "Triton requires a C compiler and Python headers"
```

The probe exits 0. It reports `torch=2.13.0+cu129`, CUDA 12.9, and `Tesla T4`.
The tensor sum is 4.0. The image supplies `/usr/bin/gcc` and the Python headers.
The torch architecture list includes `sm_75`, which the T4 needs.

### One-step training invocation

Pass this script to `/bin/bash -c` through `intern.py colab-exec`.
Set `--timeout-s 600 --log-name smoke.log`.
The first command includes the earlier stopped allocation in the remote budget.
The remaining commands launch the new path under the existing budget.

```bash
set -eu
UV=/usr/local/bin/uv
"$UV" run --frozen --no-dev python scripts/python/intern.py budget --experiment 000 record-gpu-h --hours 0.025849
"$UV" run --frozen --no-dev python scripts/python/intern.py budget --experiment 000 can-launch --params 134515008
"$UV" run --frozen --no-dev python scripts/python/intern.py budget --experiment 000 record-launch
"$UV" run --frozen --no-dev python scripts/python/intern.py ledger --experiment 000 upsert --path-id path-7 --status running --verify pending --approach "Colab T4 one-step CUDA smoke"
"$UV" run --frozen --no-dev python scripts/python/000-tiny-sft-smoke.py compute=colab compute.session=intern-000-t4-674b67a2 smoke_test=true trainer.args.use_cpu=false
```

The command exits 0 and prints `VERDICT: TRAIN_OK`.
The run completes one optimizer step with loss 3.6047396659851074 and 544 input tokens.
The reported training duration is 4.329 seconds. The stderr log is empty.
Smoke mode saves neither a checkpoint nor generation samples.

### Artifact recovery and verification

```bash
PATH="$PWD/trash/colab/bin:$PATH" UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache \
  uv run --no-sync python scripts/python/intern.py colab-exec --experiment 000 \
  --session intern-000-t4-674b67a2 --state-file trash/colab/live-20261020/sessions.json \
  --timeout-s 120 \
  --command 'tar --exclude=run.md -czf /content/intern-artifacts.tar.gz experiments/000-tiny-sft-smoke'
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  download -s intern-000-t4-674b67a2 /content/intern-artifacts.tar.gz trash/colab/live-20261020/artifacts.tar.gz
uv run --no-sync python - <<'PY'
from pathlib import Path
import tarfile

source = Path('trash/colab/live-20261020/artifacts.tar.gz')
destination = Path('experiments/000-tiny-sft-smoke/logs/colab-20261020/artifacts')
destination.mkdir(parents=True, exist_ok=False)
with tarfile.open(source) as archive:
    archive.extractall(destination, filter='data')
PY
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  verify --experiment 000 --experiments-root experiments/000-tiny-sft-smoke/logs/colab-20261020/artifacts/experiments
```

The raw artifact archive has SHA256
`88fcf81be499455483aad741136bb673311f8167aef2888af5696c66f6c45a09`.
The recovered metrics contain one smoke run, 23 records, and only step 1.
The full verify gate exits 1 with two failures:

- `data_consumption`: 544 tokens fall below the required 3500 tokens.
- `generation_sanity`: smoke mode omits `logs/samples.jsonl`.

Loss plausibility and the stderr scan pass. Model quality remains unverified.
Path-7 has `status=failed` and `verify=fail`. No new results.md exists.
The historical root verify.md and results.md still describe the CPU baseline.
Read the separate [Colab report](logs/colab-20261020/artifacts/experiments/000-tiny-sft-smoke/verify.md)
for this smoke run.

Keep the raw archive unchanged. Copy the current run.md into the recovered
directory before the nested scaffold gate. Reconcile the recovered budget and
ledger only after a check confirms that their local upload copies remain unchanged.
Keep the root journal append-only. Preserve the raw remote journal in the recovered directory.

### Confirmed final teardown

```bash
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" \
  stop -s intern-000-t4-674b67a2
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" sessions
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/live-20261020/sessions.json" usage
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  budget --experiment 000 record-gpu-h --hours 0.111792
```

The CLI records termination at 2026-10-20 10:19:35 UTC, or 13:19:35 Europe/Kyiv.
The final API responses report zero active assignments and a 0.00/hour usage rate.
The second allocation upper bound is 0.111792 GPU-hours.
Both allocations total 0.137641 GPU-hours, or 8.25846 GPU-minutes.
The budget records each interval once. Its total spend is now 0.217641 GPU-hours.

The baseline metrics, verify.md, results.md, and generation samples match the original archive byte-for-byte.
The recovered Colab files remain separate under `logs/colab-20261020/artifacts/`.

## Full Colab T4 run — 2026-10-20

The user requests the complete 10-step recipe, full verification, artifact recovery,
and runtime teardown. The user also authorizes one additional path if necessary.
The new path is path-8. The budget profile is `tiny-sft-colab`.
Its path cap is 3. Recorded spend and all other caps remain intact.
The pre-run backup is `trash/colab/full-20261020-path8/before-full-run.tar.gz`.

### Preconditions

```bash
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py check --experiment 000
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py budget --experiment 000 can-launch
```

The budget gate permits 2 of 3 paths and reports 0.217641 of 2.0 GPU-hours spent.
The model has 134515008 parameters. The synthetic dataset preflight confirms
256 rows and the required `text` column. Keep the model, dataset, precision,
batch size, learning rate, step count, and verification thresholds from the recipe.

### Provision and locked setup

The workspace archive contains 147 members and 279828 bytes. Its SHA256 is
`57bc31ed5da6ef6a44693dc0e8a222984c5bc341be035a5139a102306b7c2401`.
Use the archive filter and bootstrap procedure above with this expected checksum.
The archive excludes credentials and prior output artifacts.

```bash
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  new -s intern-000-full-fc75ffb8 --gpu T4
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  status -s intern-000-full-fc75ffb8
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  upload -s intern-000-full-fc75ffb8 trash/colab/full-20261020-path8/workspace.tar.gz /content/intern-workspace.tar.gz
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  exec -s intern-000-full-fc75ffb8 --timeout 360 -f trash/colab/full-20261020-path8/bootstrap.py
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  download -s intern-000-full-fc75ffb8 /content/intern-bootstrap.json trash/colab/full-20261020-path8/bootstrap-status.json
PATH="$PWD/trash/colab/bin:$PATH" UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache \
  uv run --no-sync python scripts/python/intern.py colab-exec --experiment 000 \
  --session intern-000-full-fc75ffb8 --state-file trash/colab/full-20261020-path8/sessions.json \
  --timeout-s 1200 --log-name setup.log \
  --command '/usr/local/bin/uv sync --frozen --python 3.13 --no-dev'
```

Colab reports a T4 with the Standard machine shape. The bootstrap exits 0.
The image supplies Python 3.13.15, `/usr/local/bin/uv`, and 193.31 GiB of free disk space.

### Smoke and full launch

The locked setup exits 0. Reuse the CUDA/compiler probe above, then execute:

```bash
/usr/bin/env SMOKE_TEST=0 /usr/local/bin/uv run --frozen --no-dev python scripts/python/000-tiny-sft-smoke.py \
  compute=colab compute.session=intern-000-full-fc75ffb8 smoke_test=true trainer.args.use_cpu=false
```

The fresh smoke exits 0 and prints `TRAIN_OK`, with loss 3.6047396659851074.
The probe confirms a Tesla T4, CUDA 12.9, and torch 2.13.0+cu129.
The C compiler and Python headers are present.

Preflight records:

- Recipe: the existing 134515008-parameter SFT experiment on 256 synthetic text rows.
- Precision: fp32. Use the configured attention backend and no requested flash-attention kernel.
- Tracking: `none`. Metrics and logs remain local to the experiment directory.
- Budget: 2 of 3 paths launched before scale; 0.217641 of 2 GPU-hours spent.
- Timeout: 600 seconds for ten steps and the generation probe, after a fresh one-step smoke.
- Evidence: require the unchanged full verification gate and direct inspection of all three completions.
- Artifacts: recover metrics, logs, samples, and the journal before teardown.
- Checkpoints: the existing `save_strategy=no` recipe saves no checkpoint.

Pass this script through `intern.py colab-exec` with
`--timeout-s 600 --log-name train.log`:

```bash
set -eu
UV=/usr/local/bin/uv
"$UV" run --frozen --no-dev python scripts/python/intern.py budget --experiment 000 can-launch
"$UV" run --frozen --no-dev python scripts/python/intern.py budget --experiment 000 record-launch
"$UV" run --frozen --no-dev python scripts/python/intern.py ledger --experiment 000 upsert --path-id path-8 --status running --verify pending --approach "Colab T4 full 10-step SFT"
SMOKE_TEST=0 "$UV" run --frozen --no-dev python scripts/python/000-tiny-sft-smoke.py compute=colab compute.session=intern-000-full-fc75ffb8 smoke_test=false trainer.args.use_cpu=false trainer.args.max_steps=10
"$UV" run --frozen --no-dev python scripts/python/intern.py verify --experiment 000
```

### Full result and artifact recovery

The training command exits 0 and prints `TRAIN_OK`. The remote verify gate
exits 0, with four passing checks and seven skipped checks. The full run records:

- Optimizer steps: 10.
- Input tokens: 5528, above the unchanged minimum of 3500.
- Final-step loss: 1.7793203592300415.
- Run-average train loss: 2.3757517337799072.
- Trainer runtime: 4.8337 seconds, excluding setup and generation.
- Generation samples: three coherent completions for the synthetic corpus.

The first and third samples repeat scene motifs and end at the 100-token cap.
The second sample completes a sentence about a dog and the beach.
The direct language judgment passes. The run establishes no held-out quality result.

```bash
PATH="$PWD/trash/colab/bin:$PATH" UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache \
  uv run --no-sync python scripts/python/intern.py colab-exec --experiment 000 \
  --session intern-000-full-fc75ffb8 --state-file trash/colab/full-20261020-path8/sessions.json \
  --timeout-s 120 \
  --command 'tar --exclude=run.md -czf /content/intern-artifacts.tar.gz experiments/000-tiny-sft-smoke'
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  download -s intern-000-full-fc75ffb8 /content/intern-artifacts.tar.gz trash/colab/full-20261020-path8/artifacts.tar.gz
uv run --no-sync python - <<'PY'
from pathlib import Path
import tarfile

destination = Path('experiments/000-tiny-sft-smoke/logs/colab-full-20261020-path8/artifacts')
destination.mkdir(parents=True, exist_ok=False)
with tarfile.open('trash/colab/full-20261020-path8/artifacts.tar.gz') as archive:
    expected = 'experiments/000-tiny-sft-smoke'
    assert all(member.name == expected or member.name.startswith(expected + '/') for member in archive.getmembers())
    archive.extractall(destination, filter='data')
PY
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  verify --experiment 000 --experiments-root experiments/000-tiny-sft-smoke/logs/colab-full-20261020-path8/artifacts/experiments
```

The downloaded archive contains 77374 bytes. Its SHA256 is
`3b3a94078276a7000d8b67bd84d41605fcbe7ebccfa3fd68b6f5a0e2485bf6b6`.
The recovered local verify gate exits 0 with the same four passes and seven skips.
Read all three `completion` fields before the judgment annotation.

The canonical metrics retain the CPU history and append the recovered smoke
and full runs. The latest `run_start` scopes the full verification.
The journal retains its original bytes before the remote entries.
The recovered ledger and budget retain the remote launch entry.
The CPU snapshot remains under `logs/cpu-baseline-before-path-8/`.
The raw downloaded archive remains unchanged.

### Full-run teardown and budget

```bash
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" \
  stop -s intern-000-full-fc75ffb8
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" sessions
trash/colab/bin/colab --auth adc --config "$PWD/trash/colab/full-20261020-path8/sessions.json" usage
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  budget --experiment 000 record-gpu-h --hours 0.146404
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  ledger --experiment 000 upsert --path-id path-8 --status passed --verify pass \
  --final-train-loss 1.7793203592300415 --gpu-min 8.78424
```

The CLI confirms termination at 2026-10-20 15:34:59.195704 UTC, or 18:34:59 Europe/Kyiv.
The final API responses report zero active assignments and a 0.00/hour usage rate.
The interval from allocation request to termination is at most 0.146404 GPU-hours.
The budget records this interval once, including setup and idle time.
Its final tally is 3 paths, 5 historical retries, and 0.364045 GPU-hours.
The 2 GPU-hour cap remains intact.

The full report includes `JUDGMENT: generation_quality = PASS`.
Path-8 has `status=passed` and `verify=pass` before the new results.md.
The final local scaffold gate checks the canonical and recovered directories.

```bash
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  verify --experiment 000
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  check --experiment 000
UV_CACHE_DIR=/private/tmp/my-little-ml-intern-uv-cache uv run --no-sync python scripts/python/intern.py \
  check --experiment 000 --experiments-root experiments/000-tiny-sft-smoke/logs/colab-full-20261020-path8/artifacts/experiments
```
