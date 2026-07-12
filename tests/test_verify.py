import json
import re
from pathlib import Path

from intern.metrics import MetricsLog
from intern.verify import RunVerifier, verify_run


GOOD_SAMPLES = "\n".join(
    json.dumps({"prompt": "p", "text": text})
    for text in [
        "The quick brown fox jumps over the lazy dog while a violet umbrella drifts across the quiet harbor at dawn.",
        "Seventeen sailors counted stars above the ridge,\n\ntrading maps and stories until the copper lantern faded.",
        "A gentle rain settled over the orchard as two engineers argued cheerfully about gradient clipping and tea.",
    ]
)


def build_run(
    exp: Path,
    *,
    train_loss=2.31,
    eval_loss=2.5,
    vocab_meta=32_000,
    param_count=124_000_000,
    target_params=120_000_000,
    planned_tokens=1_000_000,
    tokens_seen=900_000,
    samples=GOOD_SAMPLES,
    stderr="FutureWarning: something benign\n",
    completion_only=None,
) -> MetricsLog:
    """Write a synthetic experiment dir; pass None for any piece to omit it."""
    log = MetricsLog(exp / "metrics.jsonl")
    for key, value in [
        ("vocab_size", vocab_meta),
        ("param_count", param_count),
        ("target_params", target_params),
        ("planned_tokens", planned_tokens),
        ("completion_only", completion_only),
    ]:
        if value is not None:
            log.append_event("meta", key=key, value=value)
    if train_loss is not None:
        log.append_metric(100, "loss", train_loss, split="train")
    if eval_loss is not None:
        log.append_metric(100, "loss", eval_loss, split="eval")
    if tokens_seen is not None:
        log.append_metric(100, "num_input_tokens_seen", tokens_seen)
    logs_dir = exp / "logs"
    if samples is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "samples.jsonl").write_text(samples)
    if stderr is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "stderr.log").write_text(stderr)
    return log


def by_name(results):
    return {result.name: result for result in results}


def test_all_pass_scenario(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp)

    verifier = RunVerifier(exp)
    results = verifier.run()
    status = {r.name: r.status for r in results}

    assert status == {
        "loss_plausibility": "PASS",
        "eval_train_gap": "PASS",
        "data_consumption": "PASS",
        "stderr_scan": "PASS",
        "param_drift": "PASS",
        "generation_sanity": "PASS",
        "reward_margin": "SKIP",
        "kl_ref": "SKIP",
        "reward_variance": "SKIP",
    }
    assert verifier.passed(results)
    assert "warning line" in by_name(results)["stderr_scan"].detail
    assert verify_run(exp) == 0
    assert (exp / "verify.md").exists()


def test_low_loss_red_flag_fails(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, train_loss=0.42)

    result = by_name(RunVerifier(exp).run())["loss_plausibility"]
    assert result.status == "FAIL"
    assert "red flag" in result.detail
    assert verify_run(exp) == 1
    assert "VERDICT: loss_plausibility = FAIL" in (exp / "verify.md").read_text()


def test_completion_only_skips_loss_plausibility(tmp_path):
    exp = tmp_path / "001-demo"
    # pi-mono scenario: completion-only eval_loss ~0.55 would red-flag FAIL under corpus-CE assumptions.
    build_run(exp, train_loss=0.5, eval_loss=0.55, completion_only=True)

    verifier = RunVerifier(exp)
    results = verifier.run()
    result = by_name(results)["loss_plausibility"]
    assert result.status == "SKIP"
    assert "assistant-target tokens" in result.detail
    assert verifier.passed(results)
    assert verify_run(exp) == 0


def test_completion_only_false_meta_does_not_skip(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, train_loss=0.5, completion_only=False)  # falsey meta omitted -> normal red flag

    assert by_name(RunVerifier(exp).run())["loss_plausibility"].status == "FAIL"


def test_loss_outside_band_fails(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, train_loss=25.0, eval_loss=None)

    result = by_name(RunVerifier(exp).run())["loss_plausibility"]
    assert result.status == "FAIL"


def test_vocab_size_arg_overrides_meta(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, train_loss=5.0, vocab_meta=10)  # meta band is (0.23, 2.3) -> would FAIL

    assert by_name(RunVerifier(exp).run())["loss_plausibility"].status == "FAIL"
    assert by_name(RunVerifier(exp, vocab_size=32_000).run())["loss_plausibility"].status == "PASS"


def test_missing_samples_fails_unless_excluded(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, samples=None)

    verifier = RunVerifier(exp)
    result = by_name(verifier.run())["generation_sanity"]
    assert result.status == "FAIL"
    assert "absent" in result.detail

    excluded = [name for name in RunVerifier.CHECK_NAMES if name != "generation_sanity"]
    results = verifier.run(checks=excluded)
    assert "generation_sanity" not in by_name(results)
    assert verifier.passed(results)
    assert verify_run(exp, checks=excluded) == 0
    assert not (exp / "verify.md").exists()


def test_degenerate_samples_fail(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(
        exp,
        samples=json.dumps({"text": "yes " * 30}) + "\n" + json.dumps({"text": "too short"}),
    )

    result = by_name(RunVerifier(exp).run())["generation_sanity"]
    assert result.status == "FAIL"
    assert "unique-token ratio" in result.detail
    assert "chars" in result.detail


def test_skip_semantics_when_inputs_absent(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(
        exp,
        eval_loss=None,
        vocab_meta=None,
        param_count=None,
        target_params=None,
        planned_tokens=None,
        tokens_seen=None,
        samples=None,
        stderr=None,
    )

    status = {r.name: r.status for r in RunVerifier(exp).run()}
    assert status["loss_plausibility"] == "SKIP"  # no vocab_size meta and no override
    assert status["eval_train_gap"] == "SKIP"
    assert status["data_consumption"] == "SKIP"
    assert status["stderr_scan"] == "SKIP"
    assert status["param_drift"] == "SKIP"
    assert status["reward_margin"] == "SKIP"
    assert status["kl_ref"] == "SKIP"
    assert status["reward_variance"] == "SKIP"
    assert status["generation_sanity"] == "FAIL"  # missing samples is never a SKIP


def test_skips_alone_do_not_fail_the_gate(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, eval_loss=None, planned_tokens=None, target_params=None, stderr=None)

    checks = ["eval_train_gap", "data_consumption", "stderr_scan", "param_drift"]
    assert verify_run(exp, checks=checks) == 0
    assert not (exp / "verify.md").exists()  # scoped runs never write the report


def test_stderr_fatal_patterns_fail(tmp_path):
    for pattern in ["Traceback (most recent call last):", "RuntimeError: boom", "CUDA out of memory"]:
        exp = tmp_path / f"exp-{hash(pattern) & 0xFFFF}"
        build_run(exp, stderr=f"UserWarning: fine\n{pattern}\n")
        result = by_name(RunVerifier(exp).run())["stderr_scan"]
        assert result.status == "FAIL", pattern
        assert verify_run(exp) == 1


def test_eval_train_gap_fail(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, train_loss=2.31, eval_loss=3.5)

    result = by_name(RunVerifier(exp).run())["eval_train_gap"]
    assert result.status == "FAIL"
    assert result.value == 3.5 - 2.31


def test_data_consumption_below_threshold_fails(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, planned_tokens=1_000_000, tokens_seen=500_000)

    result = by_name(RunVerifier(exp).run())["data_consumption"]
    assert result.status == "FAIL"
    assert result.threshold == 700_000


def test_param_drift_fail(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, param_count=200_000_000, target_params=120_000_000)

    assert by_name(RunVerifier(exp).run())["param_drift"].status == "FAIL"


def test_param_drift_skips_when_quantized(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp, param_count=200_000_000, target_params=120_000_000)  # would FAIL if compared
    log.append_event("meta", key="quantized", value=True)

    result = by_name(RunVerifier(exp).run())["param_drift"]
    assert result.status == "SKIP"
    assert "4-bit" in result.detail
    assert "numel" in result.detail


def test_dpo_checks_activate_when_metrics_present(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    log.append_metric(50, "rewards/margins", 0.12)
    log.append_metric(40, "kl", 0.1)
    log.append_metric(50, "kl", 0.2)

    results = by_name(RunVerifier(exp).run())
    assert results["reward_margin"].status == "PASS"
    assert results["kl_ref"].status == "PASS"
    assert abs(results["kl_ref"].value - 0.15) < 1e-9

    log.append_metric(60, "rewards/margins", -0.05)
    assert by_name(RunVerifier(exp).run())["reward_margin"].status == "FAIL"


def test_reward_variance_all_zero_std_fails(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    for step in (10, 20, 30):
        log.append_metric(step, "reward_std", 0.0)

    result = by_name(RunVerifier(exp).run())["reward_variance"]
    assert result.status == "FAIL"
    assert "optimized nothing" in result.detail
    assert verify_run(exp) == 1


def test_reward_variance_varying_std_passes(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    for step, value in [(10, 0.0), (20, 0.4), (30, 0.2)]:
        log.append_metric(step, "reward_std", value)

    result = by_name(RunVerifier(exp).run())["reward_variance"]
    assert result.status == "PASS"
    assert result.value == 0.2  # final reward_std


def test_reward_variance_absent_skips(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp)

    result = by_name(RunVerifier(exp).run())["reward_variance"]
    assert result.status == "SKIP"
    assert "no GRPO reward metrics" in result.detail


def test_reward_variance_constant_reward_series_fails(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    for step in (10, 20, 30):
        log.append_metric(step, "reward", 1.0)

    result = by_name(RunVerifier(exp).run())["reward_variance"]
    assert result.status == "FAIL"
    assert "constant reward" in result.detail


def test_reward_variance_varying_reward_series_passes(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    log.append_metric(10, "reward", 0.0)
    log.append_metric(20, "reward", 1.0)

    result = by_name(RunVerifier(exp).run())["reward_variance"]
    assert result.status == "PASS"
    assert abs(result.value - 0.5) < 1e-9  # population std of [0, 1]


def test_reward_variance_prefers_reward_std_over_reward(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    log.append_metric(10, "reward", 1.0)  # constant reward would FAIL on its own
    log.append_metric(10, "reward_std", 0.3)

    result = by_name(RunVerifier(exp).run())["reward_variance"]
    assert result.status == "PASS"
    assert result.value == 0.3


def test_report_format_is_greppable(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp)

    verifier = RunVerifier(exp)
    results = verifier.run()
    report_path = verifier.write_report(results)

    assert report_path == exp / "verify.md"
    lines = report_path.read_text().strip().splitlines()
    assert len(lines) == len(results) + 1
    verdict_re = re.compile(r"^VERDICT: [a-z_]+ = (PASS|FAIL|SKIP) \| value=.+ \| threshold=.+ \| .+$")
    for line in lines[:-1]:
        assert verdict_re.match(line), line
    assert "VERDICT: loss_plausibility = PASS" in lines[0]
    assert re.match(r"^OVERALL: PASS \(6 passed, 0 failed, 3 skipped\)$", lines[-1])


def test_verify_run_exit_2_when_metrics_missing(tmp_path):
    exp = tmp_path / "001-empty"
    exp.mkdir()

    assert verify_run(exp) == 2
    assert not (exp / "verify.md").exists()


def test_run_start_scopes_out_previous_runs(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp)
    log.append_event("run_start", task="trl_sft")

    assert verify_run(exp) == 2  # run started, nothing logged -> pipeline bug

    log.append_event("meta", key="vocab_size", value=32_000)
    log.append_metric(5, "loss", 0.5, split="train")
    assert verify_run(exp) == 1  # scoped red flag despite healthy pre-run_start records


def test_dpo_task_skips_lm_checks(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp, train_loss=0.5, samples=None)
    log.append_event("meta", key="task", value="trl_dpo")

    results = by_name(RunVerifier(exp).run())
    assert results["loss_plausibility"].status == "SKIP"
    assert results["generation_sanity"].status == "SKIP"


def test_kto_task_skips_lm_checks_like_dpo(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp, train_loss=0.5, samples=None)
    log.append_event("meta", key="task", value="trl_kto")

    results = by_name(RunVerifier(exp).run())
    assert results["loss_plausibility"].status == "SKIP"
    assert results["generation_sanity"].status == "SKIP"


def test_gkd_task_expects_samples_but_skips_lm_loss(tmp_path):
    # GKD loss is generalized JSD (not vocab CE) -> loss_plausibility SKIPs even at 0.5;
    # but the student IS a generator -> missing samples.jsonl must FAIL, not skip.
    exp = tmp_path / "001-demo"
    log = build_run(exp, train_loss=0.5, samples=None)
    log.append_event("meta", key="task", value="trl_gkd")

    results = by_name(RunVerifier(exp).run())
    assert results["loss_plausibility"].status == "SKIP"
    assert results["generation_sanity"].status == "FAIL"


def test_red_flag_fires_without_vocab(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp, train_loss=0.5, vocab_meta=None)
    assert by_name(RunVerifier(exp).run())["loss_plausibility"].status == "FAIL"

    exp2 = tmp_path / "002-demo"
    build_run(exp2, train_loss=2.0, vocab_meta=None)
    assert by_name(RunVerifier(exp2).run())["loss_plausibility"].status == "SKIP"


def test_val_loss_counts_as_eval(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp, eval_loss=None)
    log.append_metric(100, "val_loss", 2.4, split="eval")

    result = by_name(RunVerifier(exp).run())["eval_train_gap"]
    assert result.status == "PASS"


def test_cjk_samples_use_character_tokens(tmp_path):
    exp = tmp_path / "001-demo"
    text = "从前有一只小狐狸住在森林里它每天都去河边喝水然后回到山上的家慢慢睡着了第二天又开始新的一天" * 2
    build_run(exp, samples=json.dumps({"text": text}))

    result = by_name(RunVerifier(exp).run())["generation_sanity"]
    assert result.status == "PASS"


def test_judgment_lines_survive_report_rewrite(tmp_path):
    exp = tmp_path / "001-demo"
    build_run(exp)
    assert verify_run(exp) == 0
    with (exp / "verify.md").open("a") as fh:
        fh.write("JUDGMENT: generation_quality = PASS | coherent for the training distribution\n")

    assert verify_run(exp) == 0
    text = (exp / "verify.md").read_text()
    assert text.count("JUDGMENT:") == 1
    assert "OVERALL:" in text


def test_grpo_missing_samples_fails(tmp_path):
    exp = tmp_path / "001-demo"
    log = build_run(exp, samples=None)
    log.append_event("meta", key="task", value="trl_grpo")

    assert by_name(RunVerifier(exp).run())["generation_sanity"].status == "FAIL"
