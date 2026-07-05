# Postmortem — path-5

**Symptom:** 4/5 active checks pass; `generation_sanity` still FAILs — one of three
sampled continuations collapses into a repeated token ("night" = 66% of output).

**Root-cause hypothesis:** intrinsic capacity limit. pythia-14m fine-tuned on a
256-row templated corpus has a real degeneracy attractor; switching greedy decoding to
seeded sampling (a legitimate adapter fix — greedy loops even on healthy models) fixed
two of three samples, but the third still collapses. The check is correctly reporting
partial distribution collapse.

**Fix:** one variable — `trainer.model_name` → `HuggingFaceTB/SmolLM2-135M` (properly
trained 135M Llama-family model that can sustain 100-token continuations);
`target_params` updated to match.
