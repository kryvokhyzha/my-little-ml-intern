from pathlib import Path

import pytest

from intern.budget import Budget, BudgetGate, load_budget, load_budget_profile, save_budget
from intern.ledger import Ledger


REPO_ROOT = Path(__file__).resolve().parents[1]


CONTRACT_BUDGET = """# Budget

max_paths: 2
max_retries_per_path: 2
compute_cap_gpu_h: 2.0
scale_ceiling_params: 200000000
token_budget: null

## Spent

paths_launched: 0
retries_used: 0
gpu_h_used: 0.0
"""


def _write_budget(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "budget.md"
    save_budget(Budget(**overrides), path)
    return path


def test_load_budget_parses_contract(tmp_path: Path) -> None:
    path = tmp_path / "budget.md"
    path.write_text(CONTRACT_BUDGET)
    assert load_budget(path) == Budget(
        max_paths=2,
        max_retries_per_path=2,
        compute_cap_gpu_h=2.0,
        scale_ceiling_params=200_000_000,
        token_budget=None,
        paths_launched=0,
        retries_used=0,
        gpu_h_used=0.0,
    )


def test_save_budget_round_trips_contract_exactly(tmp_path: Path) -> None:
    source = tmp_path / "budget.md"
    source.write_text(CONTRACT_BUDGET)
    out = tmp_path / "out.md"
    save_budget(load_budget(source), out)
    assert out.read_text() == CONTRACT_BUDGET


def test_token_budget_int_round_trips(tmp_path: Path) -> None:
    path = _write_budget(tmp_path, token_budget=5_000_000)
    assert "token_budget: 5000000\n" in path.read_text()
    assert load_budget(path).token_budget == 5_000_000


def test_spent_values_round_trip(tmp_path: Path) -> None:
    path = _write_budget(tmp_path, paths_launched=1, retries_used=2, gpu_h_used=1.25)
    budget = load_budget(path)
    assert (budget.paths_launched, budget.retries_used, budget.gpu_h_used) == (1, 2, 1.25)


def test_load_budget_missing_keys_raises(tmp_path: Path) -> None:
    path = tmp_path / "budget.md"
    path.write_text("# Budget\n\nmax_paths: 2\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_budget(path)


def test_can_launch_allowed_under_caps(tmp_path: Path) -> None:
    gate = BudgetGate(_write_budget(tmp_path))
    allowed, reason = gate.can_launch_path()
    assert allowed
    assert "0/2" in reason


def test_can_launch_denied_at_max_paths(tmp_path: Path) -> None:
    gate = BudgetGate(_write_budget(tmp_path, paths_launched=2))
    allowed, reason = gate.can_launch_path()
    assert not allowed
    assert "max_paths" in reason
    assert "2" in reason


def test_can_launch_denied_at_compute_cap(tmp_path: Path) -> None:
    gate = BudgetGate(_write_budget(tmp_path, gpu_h_used=2.0))
    allowed, reason = gate.can_launch_path()
    assert not allowed
    assert "compute_cap_gpu_h" in reason


def test_record_launch_persists(tmp_path: Path) -> None:
    path = _write_budget(tmp_path)
    BudgetGate(path).record_launch()
    assert load_budget(path).paths_launched == 1


def test_record_retry_and_gpu_h_persist(tmp_path: Path) -> None:
    path = _write_budget(tmp_path)
    gate = BudgetGate(path)
    gate.record_retry()
    gate.record_gpu_h(0.5)
    gate.record_gpu_h(0.25)
    reloaded = load_budget(path)
    assert reloaded.retries_used == 1
    assert reloaded.gpu_h_used == 0.75


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-1", approach="baseline", status="failed")
    ledger.upsert("path-1r1", approach="baseline lr*0.1", status="failed", retry_of="path-1")
    ledger.upsert("path-1r2", approach="baseline lr*0.01", status="running", retry_of="path-1")
    ledger.upsert("path-2", approach="other", status="queued")
    return ledger


def test_can_retry_counts_ledger_rows(tmp_path: Path, ledger: Ledger) -> None:
    gate = BudgetGate(_write_budget(tmp_path))
    allowed, reason = gate.can_retry("path-1", ledger)
    assert not allowed
    assert "max_retries_per_path" in reason

    allowed, reason = gate.can_retry("path-2", ledger)
    assert allowed
    assert "0/2" in reason


def test_can_retry_denied_at_compute_cap(tmp_path: Path, ledger: Ledger) -> None:
    gate = BudgetGate(_write_budget(tmp_path, gpu_h_used=3.0))
    allowed, reason = gate.can_retry("path-2", ledger)
    assert not allowed
    assert "compute_cap_gpu_h" in reason


def test_can_retry_denies_chained_retries(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-1", status="failed")
    ledger.upsert("path-2", status="failed", retry_of="path-1")
    ledger.upsert("path-3", status="failed", retry_of="path-2")
    gate = BudgetGate(_write_budget(tmp_path))

    allowed, reason = gate.can_retry("path-3", ledger)
    assert not allowed
    assert "max_retries_per_path" in reason


def test_can_retry_global_backstop(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-9", status="failed")
    gate = BudgetGate(_write_budget(tmp_path, retries_used=4))

    allowed, reason = gate.can_retry("path-9", ledger)
    assert not allowed
    assert "global retry cap" in reason


def test_can_retry_survives_retry_of_cycle(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.md")
    ledger.upsert("path-a", status="failed", retry_of="path-b")
    ledger.upsert("path-b", status="failed", retry_of="path-a")
    gate = BudgetGate(_write_budget(tmp_path))

    allowed, _ = gate.can_retry("path-a", ledger)
    assert isinstance(allowed, bool)


def test_can_launch_scale_ceiling(tmp_path: Path) -> None:
    gate = BudgetGate(_write_budget(tmp_path))

    allowed, reason = gate.can_launch_path(params=200_000_001)
    assert not allowed
    assert "scale_ceiling_params" in reason
    allowed, _ = gate.can_launch_path(params=135_000_000)
    assert allowed


def _write_profile(tmp_path: Path, name: str, body: str) -> Path:
    budget_dir = tmp_path / "budget"
    budget_dir.mkdir(exist_ok=True)
    (budget_dir / f"{name}.yaml").write_text(body, encoding="utf-8")
    return tmp_path


def test_load_budget_profile_zeroes_spend(tmp_path: Path) -> None:
    configs = _write_profile(
        tmp_path,
        "lora",
        "max_paths: 2\nmax_retries_per_path: 2\ncompute_cap_gpu_h: 4.0\n"
        "scale_ceiling_params: 12000000000\ntoken_budget: null\n",
    )
    assert load_budget_profile(configs, "lora") == Budget(
        max_paths=2,
        max_retries_per_path=2,
        compute_cap_gpu_h=4.0,
        scale_ceiling_params=12_000_000_000,
        token_budget=None,
        paths_launched=0,
        retries_used=0,
        gpu_h_used=0.0,
    )


def test_load_budget_profile_token_budget_int(tmp_path: Path) -> None:
    configs = _write_profile(
        tmp_path,
        "tok",
        "max_paths: 1\nmax_retries_per_path: 1\ncompute_cap_gpu_h: 0.5\n"
        "scale_ceiling_params: 200000000\ntoken_budget: 5000000\n",
    )
    assert load_budget_profile(configs, "tok").token_budget == 5_000_000


def test_load_budget_profile_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_budget_profile(tmp_path, "nope")


def test_load_budget_profile_missing_key_raises(tmp_path: Path) -> None:
    configs = _write_profile(tmp_path, "bad", "max_paths: 2\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_budget_profile(configs, "bad")


def test_shipped_budget_profiles_all_parse() -> None:
    names = sorted(path.stem for path in (REPO_ROOT / "configs" / "budget").glob("*.yaml"))
    assert names, "budget profile catalog is empty"
    for name in names:
        profile = load_budget_profile(REPO_ROOT / "configs", name)
        assert profile.max_paths >= 1
        assert profile.max_retries_per_path >= 1
        assert profile.compute_cap_gpu_h > 0
        assert profile.scale_ceiling_params > 0
        assert (profile.paths_launched, profile.retries_used, profile.gpu_h_used) == (0, 0, 0.0)
