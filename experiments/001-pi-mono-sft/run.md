# Run — 001-pi-mono-sft

Exact commands executed on the compute instance. Copy-pasteable; substitute
`<placeholders>`. Never inline a token — `scp` your `.env` instead.

## Lane

compute=ssh · instance: 1x NVIDIA L4 24GB (GCP `g2-standard-8`) · region:
`europe-west4-c` (L4 was stocked out in `-a`/`-b` that day — iterate zones).

## Prerequisite

Materialize the PRIVATE training dataset once, locally (needs `HF_TOKEN` +
`HF_USER` in `.env`) — the run pulls it from the Hub:

```bash
uv run python scripts/python/prep-pi-mono-sft.py   # converts pi-mono session JSONL → {prompt,completion}, pushes <HF_USER>/pi-mono-sft (private)
```

## Provision

Mirrors `configs/compute/ssh.yaml` security: IP-locked firewall, key-only SSH.

```bash
gcloud compute instances create <VM> \
  --project=<PROJECT> --zone=europe-west4-c \
  --machine-type=g2-standard-8 \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --maintenance-policy=TERMINATE --tags=<VM>

# SSH open to your IP only:
gcloud compute firewall-rules create <VM>-ssh --project=<PROJECT> \
  --allow=tcp:22 --source-ranges=<YOUR_IP>/32 --target-tags=<VM>

# ephemeral key, key-only login:
ssh-keygen -t ed25519 -f <KEY> -N ""
gcloud compute instances add-metadata <VM> --project=<PROJECT> --zone=europe-west4-c \
  --metadata=ssh-keys="<USER>:$(cat <KEY>.pub)"
SSH="ssh -i <KEY> -o IdentitiesOnly=yes <USER>@<EXTERNAL_IP>"
```

## Setup

```bash
$SSH "git clone <REPO_URL> ml"
scp -i <KEY> .env <USER>@<EXTERNAL_IP>:ml/.env   # HF_TOKEN pulls the PRIVATE dataset
$SSH "cd ml && uv sync --group gpu"              # QLoRA needs bitsandbytes (CUDA-only)
$SSH "cd ml && bash scripts/bash/gpu_probe.sh"   # expect cuda=1, vram_gb ~= 24
```

## Train

```bash
$SSH "cd ml && uv run python scripts/python/001-pi-mono-sft.py smoke_test=true"   # VERDICT: TRAIN_OK
$SSH "cd ml && uv run python scripts/python/intern.py budget --experiment 001-pi-mono-sft can-launch"
$SSH "cd ml && uv run python scripts/python/001-pi-mono-sft.py"                    # 200 steps, ~2.58 GPU-h
$SSH "cd ml && uv run python scripts/python/intern.py verify --experiment 001-pi-mono-sft"

# pull artifacts (adapter under ckpts/, logs, metrics, verify.md) back into the local repo:
rsync -avz -e "ssh -i <KEY>" <USER>@<EXTERNAL_IP>:ml/experiments/001-pi-mono-sft/ experiments/001-pi-mono-sft/
uv run python scripts/python/intern.py budget --experiment 001-pi-mono-sft record-gpu-h --hours 2.58
```

## Benchmark

No standalone benchmark harness in v1. Evaluation was **inline in the training
script**: held-out eval loss (`data.eval`, final **0.688**) plus 3 seeded
generation samples (`logs/samples.jsonl`), both produced by the run and checked
by the verify gate + a human read. The burtenshaw reference reports HumanEval
74% for the full recipe — reproducing that number needs an eval harness (repo
roadmap), not part of this run.

## Teardown

```bash
gcloud compute instances delete <VM> --project=<PROJECT> --zone=europe-west4-c -q
gcloud compute firewall-rules delete <VM>-ssh --project=<PROJECT> -q
rm -f <KEY> <KEY>.pub
gcloud compute disks list --project=<PROJECT>   # confirm no orphaned disk left billing
```
