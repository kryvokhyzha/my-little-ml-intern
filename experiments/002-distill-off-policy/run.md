# Run — 002-distill-off-policy

## Lane

compute=ssh · instance: 1x small GPU (L4 24GB is plenty; 135M student) · smoke
already executed locally on Apple-silicon MPS/CPU (TRAIN_OK).

## Provision

Mirrors `configs/compute/ssh.yaml` security: IP-locked firewall, key-only SSH.

```bash
gcloud compute instances create <VM> \
  --project=<PROJECT> --zone=<ZONE> \
  --machine-type=g2-standard-8 \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --maintenance-policy=TERMINATE --tags=<VM>
gcloud compute firewall-rules create <VM>-ssh --project=<PROJECT> \
  --allow=tcp:22 --source-ranges=<YOUR_IP>/32 --target-tags=<VM>
ssh-keygen -t ed25519 -f <KEY> -N ""
gcloud compute instances add-metadata <VM> --project=<PROJECT> --zone=<ZONE> \
  --metadata=ssh-keys="<USER>:$(cat <KEY>.pub)"
SSH="ssh -i <KEY> -o IdentitiesOnly=yes <USER>@<EXTERNAL_IP>"
```

## Setup

```bash
$SSH "git clone <REPO_URL> ml"
scp -i <KEY> .env <USER>@<EXTERNAL_IP>:ml/.env
$SSH "cd ml && uv sync && bash scripts/bash/gpu_probe.sh"
```

## Train

```bash
$SSH "cd ml && uv run python scripts/python/002-distill-off-policy.py smoke_test=true"   # VERDICT: TRAIN_OK
$SSH "cd ml && uv run python scripts/python/intern.py budget --experiment 002 can-launch"
$SSH "cd ml && uv run python scripts/python/002-distill-off-policy.py"
$SSH "cd ml && uv run python scripts/python/intern.py verify --experiment 002"
rsync -avz -e "ssh -i <KEY>" <USER>@<EXTERNAL_IP>:ml/experiments/002-distill-off-policy/ experiments/002-distill-off-policy/
uv run python scripts/python/intern.py budget --experiment 002 record-gpu-h --hours <h>
```

## Benchmark

Inline held-out eval (smoltalk test split) + generation samples; the 002-vs-003
comparison at equal steps lands in results.md.

## Teardown

```bash
gcloud compute instances delete <VM> --project=<PROJECT> --zone=<ZONE> -q
gcloud compute firewall-rules delete <VM>-ssh --project=<PROJECT> -q
rm -f <KEY> <KEY>.pub
gcloud compute disks list --project=<PROJECT>   # confirm no orphaned disk left billing
```
