# Run — 004-self-distill

## Lane

compute=ssh · instance: 1x small GPU (L4 24GB is plenty; 360M model) · smoke
already executed locally on Apple-silicon MPS/CPU (collect limit_tasks=6 →
TRAIN_OK; full collect is 240 tasks × k=4 = 960 generations — generation-bound,
run it on the VM).

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

## Prerequisite (on the VM — generation-bound)

Collect the model's own rollouts once; prints the BASELINE eval success rate —
record it here:

```bash
$SSH "cd ml && uv run python scripts/python/prep-self-distill.py"
# SELF_DISTILL_PREP | traces=... | accepted=... | baseline_success=...
```

## Train

```bash
$SSH "cd ml && uv run python scripts/python/004-self-distill.py smoke_test=true"   # VERDICT: TRAIN_OK
$SSH "cd ml && uv run python scripts/python/intern.py budget --experiment 004 can-launch"
$SSH "cd ml && uv run python scripts/python/004-self-distill.py"
$SSH "cd ml && uv run python scripts/python/intern.py verify --experiment 004"
rsync -avz -e "ssh -i <KEY>" <USER>@<EXTERNAL_IP>:ml/experiments/004-self-distill/ experiments/004-self-distill/
uv run python scripts/python/intern.py budget --experiment 004 record-gpu-h --hours <h>
```

## Benchmark (on the VM)

The claim metric — post-train success rate on the SAME held-out eval tasks with
the SAME deterministic verifier (compare against the Prerequisite baseline):

```bash
$SSH "cd ml && uv run python scripts/python/prep-self-distill.py eval_model_path=experiments/004-self-distill/ckpts/checkpoint-100"
# SELF_DISTILL_EVAL | ... | success_rate=...
```

## Teardown

```bash
gcloud compute instances delete <VM> --project=<PROJECT> --zone=<ZONE> -q
gcloud compute firewall-rules delete <VM>-ssh --project=<PROJECT> -q
rm -f <KEY> <KEY>.pub
gcloud compute disks list --project=<PROJECT>   # confirm no orphaned disk left billing
```
