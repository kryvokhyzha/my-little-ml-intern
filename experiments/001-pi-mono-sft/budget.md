# Budget

max_paths: 2
max_retries_per_path: 2
# One ~1h reference run (L40S ~65 min) plus retry margin.
compute_cap_gpu_h: 2.0
# gemma-4-E2B-it is 5,123,178,051 params — above the 200M default ceiling. QLoRA
# nf4 4-bit fits it on a single 24GB GPU, so the scale is justified for this run.
scale_ceiling_params: 6000000000
token_budget: null

## Spent

paths_launched: 0
retries_used: 0
gpu_h_used: 0.0
