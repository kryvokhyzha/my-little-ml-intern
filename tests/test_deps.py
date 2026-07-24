import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from intern import deps


def _now():
    return datetime.now(timezone.utc)


def _release(days_ago, yanked=False):
    return {"upload_time_iso_8601": (_now() - timedelta(days=days_ago)).isoformat(), "yanked": yanked}


def _install_pypi(monkeypatch, payloads):
    """Replace urllib.request.urlopen with an offline fake serving canned PyPI JSON."""
    calls = []

    def fake_urlopen(url, timeout=None):
        package = url.rsplit("/pypi/", 1)[1].split("/")[0]
        calls.append(package)
        return io.BytesIO(json.dumps(payloads[package]).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def test_latest_eligible_windowing(monkeypatch):
    _install_pypi(
        monkeypatch,
        {"pkg": {"releases": {"1.0.0": [_release(30)], "1.1.0": [_release(10)], "1.2.0": [_release(2)]}}},
    )
    version, released = deps.latest_eligible("pkg", min_age_days=7)
    assert version == "1.1.0"
    assert released == (_now() - timedelta(days=10)).date()


def test_latest_eligible_skips_prereleases_and_dev(monkeypatch):
    _install_pypi(
        monkeypatch,
        {
            "pkg": {
                "releases": {
                    "1.1.0": [_release(30)],
                    "2.0.0rc1": [_release(40)],
                    "2.0.0.dev1": [_release(40)],
                }
            }
        },
    )
    assert deps.latest_eligible("pkg", min_age_days=7)[0] == "1.1.0"


def test_latest_eligible_skips_yanked_files(monkeypatch):
    _install_pypi(
        monkeypatch,
        {"pkg": {"releases": {"1.4.0": [_release(30)], "1.5.0": [_release(30, yanked=True)]}}},
    )
    assert deps.latest_eligible("pkg", min_age_days=7)[0] == "1.4.0"


def test_latest_eligible_uses_earliest_upload_of_version(monkeypatch):
    _install_pypi(
        monkeypatch,
        {"pkg": {"releases": {"1.0.0": [_release(30)], "1.6.0": [_release(3), _release(10)]}}},
    )
    version, released = deps.latest_eligible("pkg", min_age_days=7)
    assert version == "1.6.0"
    assert released == (_now() - timedelta(days=10)).date()


def test_latest_eligible_raises_when_nothing_old_enough(monkeypatch):
    _install_pypi(monkeypatch, {"pkg": {"releases": {"0.1.0": [_release(2)]}}})
    with pytest.raises(LookupError):
        deps.latest_eligible("pkg", min_age_days=7)


def test_declared_floor_extraction():
    assert deps._declared_floor(Requirement("x ~= 1.4.2")) == Version("1.4.2")
    assert deps._declared_floor(Requirement("x >= 1.2, < 2.0")) == Version("1.2")
    assert deps._declared_floor(Requirement("x == 2.0.*")) == Version("2.0")
    assert deps._declared_floor(Requirement("x")) is None
    assert deps._declared_floor(Requirement("x < 3.0")) is None


def test_dated_exception_downgrades_violation_until_expiry(monkeypatch, tmp_path):
    _install_pypi(monkeypatch, {"pkga": {"releases": {"1.2.0": [_release(2)]}}})
    # expiry is evaluated against the UTC date (inclusive) — build both sides in UTC
    future = (_now().date() + timedelta(days=3)).isoformat()
    past = (_now().date() - timedelta(days=2)).isoformat()

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f"""
[project]
name = "demo"
version = "0.0.1"
dependencies = ["pkga ~= 1.2.0"]

[tool.intern.deps]
exceptions = {{ pkga = "{future}" }}
""",
        encoding="utf-8",
    )
    lines = deps.check_project(pyproject, min_age_days=7)
    assert not [line for line in lines if not line.startswith("info:")]  # no violations
    assert any("ALLOWED by a dated exception" in line for line in lines)

    # expired exception -> violation returns, with a remove-it hint
    pyproject.write_text(pyproject.read_text().replace(future, past), encoding="utf-8")
    lines = deps.check_project(pyproject, min_age_days=7)
    violations = [line for line in lines if not line.startswith("info:")]
    assert len(violations) == 1
    assert "exception expired" in violations[0]


def test_check_project_violations_and_info(monkeypatch, tmp_path):
    calls = _install_pypi(
        monkeypatch,
        {
            "pkga": {"releases": {"1.1.0": [_release(100)], "1.2.0": [_release(2)]}},
            "pkgb": {"releases": {"2.0.0": [_release(50)], "2.1.0": [_release(30)]}},
            "pkgd": {"releases": {"3.0.0": [_release(20)]}},
        },
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "demo"
version = "0.0.1"
dependencies = [
    "pkga ~= 1.2.0",
    "pkgb >= 2.0.0, < 3.0.0",
    "pkgc",
    "pkgd == 3.0.0",
]
""",
        encoding="utf-8",
    )
    lines = deps.check_project(pyproject, min_age_days=7)
    violations = [line for line in lines if not line.startswith("info:")]
    infos = [line for line in lines if line.startswith("info:")]
    assert len(violations) == 1
    assert violations[0].startswith("pkga:")
    assert "1.2.0" in violations[0]
    assert len(infos) == 1
    assert "pkgb" in infos[0]
    assert "2.1.0" in infos[0]
    assert "pkgc" not in calls  # bare name -> no floor -> no fetch


def test_changelog_url_prefers_explicit_label_then_github_fallback():
    explicit = {"info": {"project_urls": {"Homepage": "https://github.com/o/r", "Changelog": "https://x.dev/news"}}}
    assert deps._changelog_url(explicit) == "https://x.dev/news"
    # Most ML packages declare only a GitHub homepage — point at its releases page.
    assert deps._changelog_url({"info": {"project_urls": {"Homepage": "https://github.com/o/r/"}}}) == (
        "https://github.com/o/r/releases"
    )
    # Underscored/cased labels normalize; non-GitHub homepages yield nothing to link.
    assert deps._changelog_url({"info": {"project_urls": {"Release_Notes": "https://x.dev/rn"}}}) == "https://x.dev/rn"
    assert deps._changelog_url({"info": {"project_urls": {"Homepage": "https://x.dev"}}}) is None
    assert deps._changelog_url({"info": {}}) is None


def test_check_project_info_line_carries_changelog_url(monkeypatch, tmp_path):
    _install_pypi(
        monkeypatch,
        {
            "pkgf": {
                "releases": {"1.0.0": [_release(60)], "1.1.0": [_release(30)]},
                "info": {"project_urls": {"Homepage": "https://github.com/acme/pkgf"}},
            }
        },
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "demo"
version = "0.0.1"
dependencies = ["pkgf ~= 1.0.0"]
""",
        encoding="utf-8",
    )
    (info,) = deps.check_project(pyproject, min_age_days=7)
    assert "newer eligible version 1.1.0" in info
    assert "changelog: https://github.com/acme/pkgf/releases" in info


def test_check_project_floor_missing_on_pypi(monkeypatch, tmp_path):
    _install_pypi(monkeypatch, {"pkge": {"releases": {"1.0.0": [_release(30)]}}})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["pkge >= 9.9.9"]\n', encoding="utf-8")
    lines = deps.check_project(pyproject, min_age_days=7)
    assert lines == ["info: pkge: declared floor 9.9.9 not found on PyPI"]


def test_check_project_no_lines_when_clean(monkeypatch, tmp_path):
    _install_pypi(monkeypatch, {"pkgd": {"releases": {"3.0.0": [_release(20)]}}})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["pkgd == 3.0.0"]\n', encoding="utf-8")
    assert deps.check_project(pyproject, min_age_days=7) == []
