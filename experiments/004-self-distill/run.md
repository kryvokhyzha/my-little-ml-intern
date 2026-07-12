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
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
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
scp -i <KEY> .env <USER>@<EXTERNAL_IP>:ml/.env   # set PROJECT_NAME in .env or VM cards get labeled "ml"
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

## Actuals (2026-07-12)

Same VM/session as 002/003 (`mli-distill-l4`, europe-west4-c, 1× L4, image
family `common-cu129-ubuntu-2204-nvidia-580`). Prep collect: 240 train tasks ×
k=4 → 960 traces, 726 accepted (75.6%), baseline eval success **0.867**
(~7 min). Train: 100 steps in **68 s**, completion-only train loss 0.0591.
Post-train eval on checkpoint-100: success **0.950** (+8.3pp). Verify:
eval_train_gap waived as structural format mismatch — base model scores 1.483
CE on the same gold rows (see verify.md). GPU-h recorded: 0.25.
