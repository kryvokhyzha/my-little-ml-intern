VERDICT: loss_plausibility = SKIP | value=n/a | threshold=n/a | completion-only SFT loss (assistant-target tokens only) is not corpus cross-entropy — band and <1.0 red-flag do not apply; rely on eval_train_gap, generation_sanity, and held-out eval
VERDICT: eval_train_gap = FAIL | value=2.021 | threshold=0.5 | final eval loss 2.08 vs final train loss 0.05913
VERDICT: data_consumption = PASS | value=97938 | threshold=70000 | consumed 98% of planned_tokens=100000
VERDICT: stderr_scan = PASS | value=0 | threshold=no Traceback/RuntimeError/CUDA OOM | stderr clean
VERDICT: param_drift = SKIP | value=n/a | threshold=0.15 | param_count or target_params meta absent
VERDICT: generation_sanity = PASS | value=0.9 | threshold=unique_ratio>=0.3, max_token_share<=0.5, chars>=50 | 3 sample(s) pass mechanical proxies — eyeball them; this is necessary, not sufficient
VERDICT: reward_margin = SKIP | value=n/a | threshold=> 0 | no DPO reward metrics logged
VERDICT: kl_ref = SKIP | value=n/a | threshold=finite and > 0 | no KL metrics logged
VERDICT: reward_variance = SKIP | value=n/a | threshold=std > 0 | no GRPO reward metrics logged
OVERALL: FAIL (3 passed, 1 failed, 5 skipped)
WAIVER: eval_train_gap = WAIVED | gap 2.02 is mostly structural: gold eval completions are bare answers ("139") while training completions are the model own rollout style ("48 + 91 = 139"); the UNTRAINED base model already scores 1.483 CE on the same gold rows (trash/gap_check.py), so 1.42 of the gap pre-exists training. The +0.6 training-induced shift is style commitment, not capability loss: the format-agnostic claim metric (deterministic verifier, greedy, 60 held-out tasks) improved 0.867 -> 0.950. Plan falsification quadrant (eval CE improves while verifier rate does not) is the opposite of what happened. Headless waiver; notify approval_required fired.
EVAL: self_distill_success_rate = 0.950 | baseline 0.867 | +8.3pp (5/60 tasks, ~2.1 sigma) | expected_delta >= 10pp NOT met (ceiling: baseline left 13.3pp headroom); falsification NOT triggered.
JUDGMENT: generation_quality = PASS | all 3 samples well-formed chat completions in the trained answer style; 1 of 3 sampled completions carries an arithmetic slip (48+91 -> 149), consistent with temperature sampling — the greedy 60-task eval (0.950) is the authoritative number.
