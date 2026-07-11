import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from intern.metrics import MetricsLog


Status = Literal["PASS", "FAIL", "SKIP"]

LOSS_RED_FLAG = 1.0
EVAL_GAP_MAX = 0.5
DATA_CONSUMPTION_MIN_FRACTION = 0.7
PARAM_DRIFT_MAX_FRACTION = 0.15
MIN_UNIQUE_TOKEN_RATIO = 0.3
MAX_TOKEN_SHARE = 0.5
MIN_SAMPLE_CHARS = 50

# Tasks whose train loss is vocab cross-entropy and which must produce generation samples.
_LM_TASKS = (None, "trl_sft")
# GRPO loss is not vocab cross-entropy, but its policy IS a generative LM and the
# adapter writes samples — so missing samples fail the gate for it too.
_GENERATION_TASKS = (None, "trl_sft", "trl_grpo")

_STDERR_FATAL_RE = re.compile(r"Traceback|RuntimeError|CUDA out of memory")
_STDERR_WARNING_RE = re.compile(r"warn", re.IGNORECASE)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, tuple):
        return "(" + ", ".join(_fmt(item) for item in value) + ")"
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def _snippets(lines: list[str], limit: int = 3) -> str:
    # "|" is the verify.md field separator — keep report lines greppable.
    shown = [line.strip().replace("|", "/")[:100] for line in lines[:limit]]
    suffix = f" (+{len(lines) - limit} more)" if len(lines) > limit else ""
    return "; ".join(shown) + suffix


@dataclass
class CheckResult:
    name: str
    status: Status
    value: float | str | None
    threshold: float | str | tuple[float, float] | None
    detail: str


class RunVerifier:
    """Mechanical verification gates for one experiment run.

    A low loss number is never evidence the model works — these checks only rule out
    mechanical failure modes. ``verify.md`` is written ONLY by this module.

    metrics.jsonl accumulates across paths/retries; checks are scoped to records at or
    after the LAST ``run_start`` event (whole file when no run_start exists).
    """

    CHECK_NAMES: tuple[str, ...] = (
        "loss_plausibility",
        "eval_train_gap",
        "data_consumption",
        "stderr_scan",
        "param_drift",
        "generation_sanity",
        "reward_margin",
        "kl_ref",
        "reward_variance",
    )

    def __init__(self, experiment_dir: Path, vocab_size: int | None = None) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.vocab_size = vocab_size
        records = MetricsLog(self.experiment_dir / "metrics.jsonl").read()
        start = None
        for index, record in enumerate(records):
            if isinstance(record, dict) and record.get("event") == "run_start":
                start = index
        self.had_run_start = start is not None
        self.records = [r for r in (records if start is None else records[start:]) if isinstance(r, dict)]

    def run(self, checks: list[str] | None = None) -> list[CheckResult]:
        names = list(self.CHECK_NAMES) if checks is None else list(checks)
        unknown = sorted(set(names) - set(self.CHECK_NAMES))
        if unknown:
            raise ValueError(f"Unknown checks {unknown}; known: {list(self.CHECK_NAMES)}")
        return [getattr(self, f"_check_{name}")() for name in names]

    def report_lines(self, results: list[CheckResult]) -> list[str]:
        lines = [
            f"VERDICT: {result.name} = {result.status} "
            f"| value={_fmt(result.value)} | threshold={_fmt(result.threshold)} | {result.detail}"
            for result in results
        ]
        n_pass = sum(result.status == "PASS" for result in results)
        n_fail = sum(result.status == "FAIL" for result in results)
        n_skip = sum(result.status == "SKIP" for result in results)
        overall = "PASS" if self.passed(results) else "FAIL"
        lines.append(f"OVERALL: {overall} ({n_pass} passed, {n_fail} failed, {n_skip} skipped)")
        return lines

    def write_report(self, results: list[CheckResult]) -> Path:
        lines = self.report_lines(results)
        report_path = self.experiment_dir / "verify.md"
        if report_path.exists():
            # JUDGMENT lines are appended by the agent per the verify-run skill; a re-run
            # over unchanged metrics (e.g. the publish gate) must not silently drop them.
            judgments = [
                line for line in report_path.read_text(encoding="utf-8").splitlines() if line.startswith("JUDGMENT:")
            ]
            lines.extend(judgments)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote {}", report_path)
        return report_path

    @staticmethod
    def passed(results: list[CheckResult]) -> bool:
        return all(result.status != "FAIL" for result in results)

    def metric_records(self) -> list[dict]:
        return [r for r in self.records if "event" not in r]

    def _final(self, name: str, split: str | None = None) -> float | None:
        for record in reversed(self.records):
            if "event" in record or record.get("name") != name:
                continue
            if split is not None and record.get("split") != split:
                continue
            value = record.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def _series(self, name: str) -> list[float]:
        return [
            float(record["value"])
            for record in self.records
            if "event" not in record
            and record.get("name") == name
            and isinstance(record.get("value"), (int, float))
            and not isinstance(record.get("value"), bool)
        ]

    def _meta(self, key: str) -> Any:
        for record in reversed(self.records):
            if record.get("event") == "meta" and record.get("key") == key:
                return record.get("value")
        return None

    def _has_metric(self, name: str) -> bool:
        return any("event" not in record and record.get("name") == name for record in self.records)

    def _task(self) -> str | None:
        task = self._meta("task")
        return str(task) if task is not None else None

    def _effective_vocab_size(self) -> int | None:
        if self.vocab_size is not None:
            return self.vocab_size
        meta = self._meta("vocab_size")
        return int(meta) if meta is not None else None

    def _final_train_loss(self) -> float | None:
        loss = self._final("loss", split="train")
        return loss if loss is not None else self._final("train_loss")

    def _final_eval_loss(self) -> float | None:
        for name, split in (("loss", "eval"), ("eval_loss", None), ("val_loss", None)):
            loss = self._final(name, split=split)
            if loss is not None:
                return loss
        return None

    def _check_loss_plausibility(self) -> CheckResult:
        name = "loss_plausibility"
        if self._meta("completion_only"):
            detail = (
                "completion-only SFT loss (assistant-target tokens only) is not corpus cross-entropy — "
                "band and <1.0 red-flag do not apply; rely on eval_train_gap, generation_sanity, and held-out eval"
            )
            return CheckResult(name, "SKIP", None, None, detail)
        task = self._task()
        if task not in _LM_TASKS:
            return CheckResult(name, "SKIP", None, None, f"task={task}: loss is not vocab cross-entropy")
        loss = self._final_train_loss()
        vocab = self._effective_vocab_size()
        if loss is None:
            return CheckResult(name, "SKIP", None, None, "no train loss logged")
        band = (0.1 * math.log(vocab), math.log(vocab)) if vocab is not None and vocab > 1 else None
        if loss < LOSS_RED_FLAG:
            detail = "red flag: loss < 1.0 on an LM task — suspect data leakage or broken loss, waive explicitly"
            return CheckResult(name, "FAIL", loss, band, detail)
        if band is None:
            return CheckResult(name, "SKIP", loss, None, "vocab_size unknown (no meta, no override); no red flag")
        if band[0] < loss < band[1]:
            return CheckResult(name, "PASS", loss, band, "final train loss inside ln(vocab) band")
        return CheckResult(name, "FAIL", loss, band, "final train loss outside ln(vocab) band")

    def _check_eval_train_gap(self) -> CheckResult:
        name = "eval_train_gap"
        train_loss = self._final_train_loss()
        eval_loss = self._final_eval_loss()
        if eval_loss is None:
            return CheckResult(name, "SKIP", None, EVAL_GAP_MAX, "no eval metrics logged")
        if train_loss is None:
            return CheckResult(name, "SKIP", None, EVAL_GAP_MAX, "no train loss logged")
        gap = abs(eval_loss - train_loss)
        detail = f"final eval loss {_fmt(eval_loss)} vs final train loss {_fmt(train_loss)}"
        status: Status = "PASS" if gap < EVAL_GAP_MAX else "FAIL"
        return CheckResult(name, status, gap, EVAL_GAP_MAX, detail)

    def _check_data_consumption(self) -> CheckResult:
        name = "data_consumption"
        seen = self._final("num_input_tokens_seen")
        planned = self._meta("planned_tokens")
        if planned is None:
            return CheckResult(name, "SKIP", seen, None, "planned_tokens meta absent")
        required = DATA_CONSUMPTION_MIN_FRACTION * float(planned)
        if seen is None:
            return CheckResult(name, "SKIP", None, required, "num_input_tokens_seen never logged")
        status: Status = "PASS" if seen >= required else "FAIL"
        fraction = float(seen) / float(planned) if planned else 0.0
        detail = f"consumed {fraction:.0%} of planned_tokens={planned}"
        return CheckResult(name, status, seen, required, detail)

    def _check_stderr_scan(self) -> CheckResult:
        name = "stderr_scan"
        threshold = "no Traceback/RuntimeError/CUDA OOM"
        stderr_path = self.experiment_dir / "logs" / "stderr.log"
        if not stderr_path.exists():
            return CheckResult(name, "SKIP", None, threshold, "logs/stderr.log absent")
        fatal: list[str] = []
        warnings: list[str] = []
        for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if _STDERR_FATAL_RE.search(line):
                fatal.append(line)
            elif _STDERR_WARNING_RE.search(line):
                warnings.append(line)
        if fatal:
            return CheckResult(name, "FAIL", len(fatal), threshold, f"fatal pattern hits: {_snippets(fatal)}")
        if warnings:
            return CheckResult(name, "PASS", 0, threshold, f"{len(warnings)} warning line(s): {_snippets(warnings)}")
        return CheckResult(name, "PASS", 0, threshold, "stderr clean")

    def _check_param_drift(self) -> CheckResult:
        name = "param_drift"
        if self._meta("quantized"):
            detail = "quantized run: 4-bit storage breaks numel comparability with target_params"
            return CheckResult(name, "SKIP", None, PARAM_DRIFT_MAX_FRACTION, detail)
        param_count = self._meta("param_count")
        target_params = self._meta("target_params")
        if param_count is None or target_params is None:
            detail = "param_count or target_params meta absent"
            return CheckResult(name, "SKIP", None, PARAM_DRIFT_MAX_FRACTION, detail)
        if not target_params:
            return CheckResult(name, "SKIP", None, PARAM_DRIFT_MAX_FRACTION, "target_params is zero")
        drift = abs(float(param_count) - float(target_params)) / float(target_params)
        status: Status = "PASS" if drift <= PARAM_DRIFT_MAX_FRACTION else "FAIL"
        detail = f"param_count={param_count} vs target_params={target_params}"
        return CheckResult(name, status, drift, PARAM_DRIFT_MAX_FRACTION, detail)

    @staticmethod
    def _sample_tokens(sample: str) -> list[str]:
        tokens = sample.split()
        # Non-space-delimited scripts (CJK, Thai) look like one giant "token";
        # fall back to character-level so the degeneracy proxies stay meaningful.
        if len(tokens) < 5 and len(sample) >= MIN_SAMPLE_CHARS:
            return [ch for ch in sample if not ch.isspace()]
        return tokens

    def _check_generation_sanity(self) -> CheckResult:
        name = "generation_sanity"
        threshold = (
            f"unique_ratio>={MIN_UNIQUE_TOKEN_RATIO}, max_token_share<={MAX_TOKEN_SHARE}, chars>={MIN_SAMPLE_CHARS}"
        )
        samples_path = self.experiment_dir / "logs" / "samples.jsonl"
        if not samples_path.exists():
            task = self._task()
            if task in _GENERATION_TASKS:
                detail = "logs/samples.jsonl absent — cannot verify generations"
                return CheckResult(name, "FAIL", None, threshold, detail)
            return CheckResult(name, "SKIP", None, threshold, f"task={task}: no generation samples expected")
        samples: list[str] = []
        for line in samples_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(record.get("text", "")).strip()
            if text:
                samples.append(text)
        if not samples:
            return CheckResult(name, "FAIL", None, threshold, "logs/samples.jsonl has no readable samples")
        problems: list[str] = []
        min_ratio: float | None = None
        for index, sample in enumerate(samples, start=1):
            if len(sample) < MIN_SAMPLE_CHARS:
                problems.append(f"sample {index}: only {len(sample)} chars")
            tokens = self._sample_tokens(sample)
            if not tokens:
                continue
            counts = Counter(tokens)
            ratio = len(counts) / len(tokens)
            share = max(counts.values()) / len(tokens)
            min_ratio = ratio if min_ratio is None else min(min_ratio, ratio)
            if ratio < MIN_UNIQUE_TOKEN_RATIO:
                problems.append(f"sample {index}: unique-token ratio {ratio:.2f}")
            if share > MAX_TOKEN_SHARE:
                problems.append(f"sample {index}: one token is {share:.0%} of output")
        if problems:
            return CheckResult(name, "FAIL", min_ratio, threshold, "; ".join(problems))
        detail = f"{len(samples)} sample(s) pass mechanical proxies — eyeball them; this is necessary, not sufficient"
        return CheckResult(name, "PASS", min_ratio, threshold, detail)

    def _check_reward_margin(self) -> CheckResult:
        name = "reward_margin"
        if not self._has_metric("rewards/margins"):
            return CheckResult(name, "SKIP", None, "> 0", "no DPO reward metrics logged")
        margin = self._final("rewards/margins")
        status: Status = "PASS" if margin is not None and margin > 0 else "FAIL"
        return CheckResult(name, status, margin, "> 0", "final rewards/margins")

    def _check_kl_ref(self) -> CheckResult:
        name = "kl_ref"
        if not self._has_metric("kl"):
            return CheckResult(name, "SKIP", None, "finite and > 0", "no KL metrics logged")
        values = self._series("kl")
        mean_kl = sum(values) / len(values)
        status: Status = "PASS" if math.isfinite(mean_kl) and mean_kl > 0 else "FAIL"
        return CheckResult(name, status, mean_kl, "finite and > 0", f"mean KL vs reference over {len(values)} points")

    def _check_reward_variance(self) -> CheckResult:
        name = "reward_variance"
        threshold = "std > 0"
        if self._has_metric("reward_std"):
            values = self._series("reward_std")
            if values and all(value == 0 for value in values):
                return CheckResult(name, "FAIL", 0.0, threshold, "no reward variance — the run optimized nothing")
            final = values[-1] if values else None
            return CheckResult(name, "PASS", final, threshold, f"final reward_std over {len(values)} points")
        if self._has_metric("reward"):
            values = self._series("reward")
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            if std == 0:
                detail = "constant reward series — no reward variance, the run optimized nothing"
                return CheckResult(name, "FAIL", 0.0, threshold, detail)
            return CheckResult(name, "PASS", std, threshold, f"population std of reward over {len(values)} points")
        return CheckResult(name, "SKIP", None, threshold, "no GRPO reward metrics logged")


def verify_run(
    experiment_dir: Path | str,
    vocab_size: int | None = None,
    checks: list[str] | None = None,
) -> int:
    """Run checks and return the gate exit code (0 pass, 1 fail, 2 missing artifacts).

    A full run (checks=None) writes verify.md; a scoped run only logs its report so a
    partial re-check can never overwrite the full report.
    """
    experiment_dir = Path(experiment_dir)
    metrics_path = experiment_dir / "metrics.jsonl"
    if not metrics_path.exists():
        logger.error("Missing {} — nothing to verify", metrics_path)
        return 2
    verifier = RunVerifier(experiment_dir, vocab_size=vocab_size)
    if verifier.had_run_start and not verifier.metric_records():
        logger.error("run_start present but the current run logged no metrics — training pipeline bug, not 'done'")
        return 2
    results = verifier.run(checks)
    if checks is None:
        verifier.write_report(results)
    else:
        for line in verifier.report_lines(results):
            logger.info(line)
    if verifier.passed(results):
        return 0
    failed = [result.name for result in results if result.status == "FAIL"]
    logger.error("Verification FAILED: {}", ", ".join(failed))
    return 1
