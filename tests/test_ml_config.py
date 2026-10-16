"""Tests for ml/config.py — the hardcoded 2026 grid.

These are data-integrity tests: GRID_2026 is duplicated in several places
(the agent system prompt, the predictor, the precompute flow) and a malformed
entry silently breaks predictions rather than raising.
"""
import pytest

from ml.config import GRID_2026, get_2026_team


def test_grid_has_22_drivers():
    """Cadillac's addition takes the grid from 20 to 22 for 2026."""
    assert len(GRID_2026) == 22


def test_driver_ids_are_unique():
    ids = [d["driver_id"] for d in GRID_2026]
    assert len(ids) == len(set(ids)), "duplicate driver_id in GRID_2026"


def test_all_entries_have_required_keys():
    required = {"driver_id", "name", "team"}
    for entry in GRID_2026:
        assert required.issubset(entry.keys()), f"missing keys in {entry}"
        assert all(isinstance(v, str) and v.strip() for v in entry.values()), entry


def test_driver_ids_are_three_letter_codes():
    """driver_id must match the fastf1 Abbreviation format used as the DB key."""
    for entry in GRID_2026:
        did = entry["driver_id"]
        assert len(did) == 3, f"{did} is not a 3-letter code"
        assert did.isalpha() and did.isupper(), f"{did} must be uppercase alpha"


def test_eleven_teams_for_2026():
    teams = {d["team"] for d in GRID_2026}
    assert len(teams) == 11, f"expected 11 teams, got {sorted(teams)}"


def test_two_drivers_per_team():
    counts = {}
    for d in GRID_2026:
        counts[d["team"]] = counts.get(d["team"], 0) + 1
    bad = {t: c for t, c in counts.items() if c != 2}
    assert not bad, f"teams without exactly 2 drivers: {bad}"


def test_get_2026_team_lookup():
    assert get_2026_team("VER") == "Red Bull Racing"
    assert get_2026_team("ZZZ") == "Unknown"
