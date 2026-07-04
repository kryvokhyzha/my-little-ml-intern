# Postmortem — path-4

**Symptom:** with the fp32 fix, no more NaN — but loss *rose* 4.48 → 7.29 over 40 steps
on MPS with grad norms ~3000, and verify failed on `generation_sanity`.

**Diagnosis:** an fp32 bisect of every adapter addition (pad token, token counting,
callback) on CPU showed all variants training identically and cleanly (4.30 → 2.32),
so the only remaining difference was the device. The identical config with
`use_cpu=true` trained perfectly (4.19 → 0.60 over 40 steps). Root cause: GPTNeoX/pythia
numerics are broken on MPS even in fp32.

Secondary finding: the healthy 40-step CPU run then *also* failed verify —
`loss_plausibility` red flag (0.60 < 1.0) — because 40 steps memorizes the tiny
5-template toy corpus. That is the check working as designed on a toy task.

**Fix:** `use_cpu: true` for this GPTNeoX example (recorded in the train-llm hardware
reference) and `max_steps: 10`, which lands the final loss ≈ 1.26, inside the
plausibility band without memorization.
