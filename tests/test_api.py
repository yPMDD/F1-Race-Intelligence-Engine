"""API-level tests using FastAPI's TestClient.

`apps.api.main` runs `Base.metadata.create_all(bind=engine)` at import time.
Because conftest points DATABASE_URL at SQLite before any project import, that
happens against the throwaway test DB rather than the real Postgres.
"""
import pytest
from fastapi.testclient import TestClient

from storage.postgres.database import SessionLocal
from apps.api.main import app
from tests.conftest import seed_race, SAMPLE_RESULTS_2026


@pytest.fixture()
def client(db):
    """TestClient wired to the per-test SQLite session."""
    from storage.postgres import database as database_mod

    database_mod.SessionLocal = lambda *a, **k: db
    app.dependency_overrides = {}
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_list_races_empty(client):
    r = client.get("/races/")
    assert r.status_code == 200
    assert r.json() == []


def test_years_returns_sorted_desc_unique(client, db):
    seed_race(db, 2024, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    seed_race(db, 2025, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)

    r = client.get("/races/years")
    assert r.status_code == 200
    assert r.json() == [2026, 2025, 2024]


def test_read_races_filtered_by_year(client, db):
    seed_race(db, 2024, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    seed_race(db, 2026, 1, "Australian Grand Prix", SAMPLE_RESULTS_2026)

    r = client.get("/races/", params={"year": 2026})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["name"] == "Australian Grand Prix"
    assert body[0]["year"] == 2026


def test_read_race_by_id_and_404(client, db):
    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    r = client.get(f"/races/{race.id}")
    assert r.status_code == 200
    assert r.json()["name"] == "Bahrain Grand Prix"

    assert client.get("/races/999999").status_code == 404


def test_read_race_laps(client, db):
    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    r = client.get(f"/races/{race.id}/laps")
    assert r.status_code == 200
    laps = r.json()
    assert len(laps) == 3
    assert all(l["driver_id"] == "VER" for l in laps)
    assert all(l["lap_time_ms"] is not None for l in laps)


def test_by_year_route(client, db):
    seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    seed_race(db, 2026, 2, "Saudi Arabian Grand Prix", SAMPLE_RESULTS_2026)
    r = client.get("/races/by-year/2026")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_race_serializer_includes_results_and_laps(client, db):
    """The dashboard reads selectedRace.results for the classification table."""
    seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    r = client.get("/races/by-year/2026")
    body = r.json()
    assert len(body[0]["results"]) == len(SAMPLE_RESULTS_2026)
    assert len(body[0]["laps"]) == 3
