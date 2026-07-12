"""Dependency-age gate: PyPI release-date checks for declared dependency floors."""

from __future__ import annotations

import json
import tomllib
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from loguru import logger
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
_FLOOR_OPERATORS = ("~=", ">=", "==")


def _parse_upload_time(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _fetch_release_dates(package: str) -> dict[str, datetime]:
    """Return version -> earliest upload time (UTC), ignoring yanked files."""
    with urllib.request.urlopen(PYPI_JSON_URL.format(package=package), timeout=30) as response:
        payload = json.load(response)
    releases: dict[str, datetime] = {}
    for version, files in payload.get("releases", {}).items():
        times = [
            _parse_upload_time(raw)
            for item in files
            if not item.get("yanked", False) and (raw := item.get("upload_time_iso_8601") or item.get("upload_time"))
        ]
        if times:
            releases[version] = min(times)
    return releases


def _cutoff(min_age_days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=min_age_days)


def _latest_eligible(releases: dict[str, datetime], cutoff: datetime) -> tuple[str, datetime] | None:
    best: tuple[Version, str, datetime] | None = None
    for raw, uploaded in releases.items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease or uploaded > cutoff:
            continue
        if best is None or version > best[0]:
            best = (version, raw, uploaded)
    if best is None:
        return None
    return best[1], best[2]


def latest_eligible(package: str, min_age_days: int = 7) -> tuple[str, date]:
    """Return the newest stable PyPI version of ``package`` whose earliest upload is >= ``min_age_days`` old.

    Raises:
        LookupError: When no stable release is old enough.

    """
    best = _latest_eligible(_fetch_release_dates(package), _cutoff(min_age_days))
    if best is None:
        raise LookupError(f"{package}: no stable release at least {min_age_days} days old")
    return best[0], best[1].date()


def _declared_floor(requirement: Requirement) -> Version | None:
    """Extract the declared floor version from ``~=``/``>=``/``==`` specifiers, if any."""
    floors = []
    for spec in requirement.specifier:
        if spec.operator not in _FLOOR_OPERATORS:
            continue
        try:
            floors.append(Version(spec.version.removesuffix(".*")))
        except InvalidVersion:
            continue
    return max(floors) if floors else None


def _upload_time_for(releases: dict[str, datetime], target: Version) -> datetime | None:
    for raw, uploaded in releases.items():
        try:
            if Version(raw) == target:
                return uploaded
        except InvalidVersion:
            continue
    return None


def _exception_expiry(exceptions: dict, package: str) -> date | None:
    raw = exceptions.get(package)
    if raw is None:
        return None
    try:
        return raw if isinstance(raw, date) else date.fromisoformat(str(raw))
    except ValueError:
        logger.warning("Unparsable deps exception date for {}: {!r} — ignored", package, raw)
        return None


def check_project(pyproject_path: Path | str, min_age_days: int = 7) -> list[str]:
    """Check ``[project].dependencies`` floors against the PyPI dependency-age gate.

    A deliberate one-time override lives in pyproject as a DATED, SELF-EXPIRING entry —
    ``[tool.intern.deps.exceptions]`` with ``package = "YYYY-MM-DD"`` — which downgrades
    that package's young-floor violation to a visible ``info:`` line until the date
    passes. The exception is in the diff (human-reviewed) and disarms itself.

    Returns:
        Violation strings for floors younger than ``min_age_days``, plus ``info:``-prefixed
        lines when a newer eligible version exists (or metadata could not be resolved).

    """
    data = tomllib.loads(Path(pyproject_path).read_text(encoding="utf-8"))
    exceptions = data.get("tool", {}).get("intern", {}).get("deps", {}).get("exceptions", {})
    cutoff = _cutoff(min_age_days)
    lines: list[str] = []
    unreachable = 0
    for raw_dep in data.get("project", {}).get("dependencies", []):
        try:
            requirement = Requirement(raw_dep)
        except InvalidRequirement:
            lines.append(f"info: unparsable dependency {raw_dep!r}")
            continue
        floor = _declared_floor(requirement)
        if floor is None:
            continue
        try:
            releases = _fetch_release_dates(requirement.name)
        except OSError as exc:
            lines.append(f"info: {requirement.name}: PyPI metadata unavailable ({exc})")
            unreachable += 1
            continue
        uploaded = _upload_time_for(releases, floor)
        if uploaded is None:
            lines.append(f"info: {requirement.name}: declared floor {floor} not found on PyPI")
        elif uploaded > cutoff:
            age_days = (datetime.now(timezone.utc) - uploaded).days
            expiry = _exception_expiry(exceptions, requirement.name)
            if expiry is not None and datetime.now(timezone.utc).date() <= expiry:
                lines.append(
                    f"info: {requirement.name}: floor {floor} is {age_days} days old but ALLOWED by a"
                    f" dated exception until {expiry.isoformat()} ([tool.intern.deps.exceptions])"
                )
            else:
                suffix = f"; exception expired {expiry.isoformat()} — remove it" if expiry is not None else ""
                lines.append(
                    f"{requirement.name}: declared floor {floor} is {age_days} days old"
                    f" (< {min_age_days} days, released {uploaded.date().isoformat()}){suffix}"
                )
        best = _latest_eligible(releases, cutoff)
        if best is not None and Version(best[0]) > floor:
            lines.append(f"info: {requirement.name}: newer eligible version {best[0]} (declared floor {floor})")
    if unreachable:
        logger.warning("deps check incomplete: {} package(s) unreachable", unreachable)
    return lines
