"""Tests for race result submission and prediction invalidation.

Defects covered:
  * `db.query(RacePrediction).delete()` removed EVERY cached prediction, not
    just the future ones the comment described. Replaying race 1 of the season
    wiped the whole forecast table.
  * F1Predictor's class-level stats cache was never invalidated here, so the
    background re-simulation predicted from pre-race momentum.
  * The semantic cache kept serving pre-race answers at 0.92 similarity.
  * Submitting results twice duplicated rows (there was a scratch script whose
    only job was cleaning them up).
"""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from storage.postgres.database import SessionLocal  # noqa: F401  (import order)
from storage.postgres.models import RacePrediction
from apps.api.main import app
from tests.conftest import seed_race, SAMPLE_RESULTS_2026


@pytest.fixture()
def client(db, monkeypatch):
    from storage.postgres.database import get_db

    started = []

    # Override the FastAPI dependency rather than monkeypatching
    # SessionLocal: get_db() closes the session in its `finally`, which would
    # detach every object the test had loaded.
    def _override():
        yield db

    app.dependency_overrides[get_db] = _override

    class _RecordingThread:
        def __init__(self, *a, **kw):
            self._k = kw

        def start(self):
            started.append(self._k.get("target"))

    import apps.api.routers.race as race_mod
    monkeypatch.setattr(race_mod.threading, "Thread", _RecordingThread)

    with TestClient(app) as c:
        c.started_threads = started
        yield c
    app.dependency_overrides = {}


def _results(position_shift: int = 0):
    return [
        {
            "driver_id": d,
            "position": p + position_shift,
            "points": pts,
            "status": "Finished",
        }
        for d, p, pts in SAMPLE_RESULTS_2026
    ]


def _add_prediction(db, race, text="old forecast"):
    pred = RacePrediction(race_id=race.id, prediction_text=text, model_version="v2_hybrid")
    db.add(pred)
    db.commit()
    return pred


# --- storage ---------------------------------------------------------------

def test_results_are_persisted(client, db):
    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    r = client.post(f"/races/{race.id}/results", json=_results(position_shift=-1))
    assert r.status_code == 200
    assert "re-simulation" in r.json()["status"]


def test_resubmitting_updates_instead_of_duplicating(client, db):
    from storage.postgres.models import ResultModel

    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    client.post(f"/races/{race.id}/results", json=_results())
    client.post(f"/races/{race.id}/results", json=_results(position_shift=-2))

    rows = (
        db.query(ResultModel)
        .filter(ResultModel.race_id == race.id, ResultModel.driver_id == "VER")
        .all()
    )
    assert len(rows) == 1, "result row was duplicated on resubmission"
    assert rows[0].position == SAMPLE_RESULTS_2026[0][1] - 2


def test_background_resimulation_is_triggered(client, db):
    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    client.post(f"/races/{race.id}/results", json=_results())
    assert len(client.started_threads) == 1
    assert client.started_threads[0].__name__ == "_resimulate"


# --- invalidation ----------------------------------------------------------

def test_only_future_predictions_are_invalidated(client, db):
    past = seed_race(
        db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026,
        date=datetime.now() - timedelta(days=10),
    )
    future = seed_race(
        db, 2026, 20, "Las Vegas Grand Prix", SAMPLE_RESULTS_2026,
        date=datetime.now() + timedelta(days=60),
    )
    past_pred = _add_prediction(db, past, "PAST FORECAST")
    future_pred = _add_prediction(db, future, "FUTURE FORECAST")
    # Capture ids now: the future row is deleted (and expired) by the endpoint.
    past_id, future_id, past_race_id = past_pred.id, future_pred.id, past.id
    db.expunge_all()

    client.post(f"/races/{past_race_id}/results", json=_results())

    remaining = {p.id for p in db.query(RacePrediction).all()}
    assert past_id in remaining, "completed race's forecast must survive"
    assert future_id not in remaining, "future forecast must be invalidated"


def test_predictor_stats_cache_is_invalidated(client, db):
    from ml.predictor import F1Predictor

    F1Predictor._stats_cache = {"VER": {"momentum": 1.0, "season_points": 0.0}}
    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)

    client.post(f"/races/{race.id}/results", json=_results())

    assert F1Predictor._stats_cache is None


def test_semantic_cache_is_invalidated(client, db):
    """Cached chat answers go stale the moment a result is recorded."""
    import nlp.cache.semantic_cache as sc_mod

    cache = sc_mod.cache_instance
    cache.set("who wins at Bahrain", "VER will win")
    assert cache.get("who wins at Bahrain") == "VER will win"

    race = seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    client.post(f"/races/{race.id}/results", json=_results())

    assert cache.count() == 0, "semantic cache still holds pre-race answers"


# --- prediction endpoint ---------------------------------------------------

def test_predict_uses_cached_prediction_when_present(client, db, monkeypatch):
    import nlp.agents.graph as graph_mod

    monkeypatch.setattr(
        graph_mod, "run_f1_agent",
        lambda p: pytest.fail("must not call the agent when a forecast exists"),
    )
    race = seed_race(db, 2026, 20, "Las Vegas Grand Prix", SAMPLE_RESULTS_2026)
    _add_prediction(db, race, "CACHED FORECAST")

    r = client.post(f"/races/{race.id}/predict")
    assert r.status_code == 200
    assert r.json() == {"prediction": "CACHED FORECAST", "source": "cache"}


def test_predict_failure_is_reported_not_guessed(client, db, monkeypatch):
    import nlp.agents.graph as graph_mod

    def boom(prompt):
        raise RuntimeError("groq unavailable")

    monkeypatch.setattr(graph_mod, "run_f1_agent", boom)
    race = seed_race(db, 2026, 20, "Las Vegas Grand Prix", SAMPLE_RESULTS_2026)

    r = client.post(f"/races/{race.id}/predict")
    assert r.status_code == 200
    body = r.json()
    # Must not be presented as a real forecast to the dashboard.
    assert body["source"] == "error"
    assert "groq unavailable" in body["prediction"]


def test_predict_persists_a_successful_forecast(client, db, monkeypatch):
    import nlp.agents.graph as graph_mod

    monkeypatch.setattr(graph_mod, "run_f1_agent", lambda p: "| Pos | Driver |")
    race = seed_race(db, 2026, 20, "Las Vegas Grand Prix", SAMPLE_RESULTS_2026)

    r = client.post(f"/races/{race.id}/predict")
    assert r.json()["source"] == "ai"
    assert db.query(RacePrediction).filter_by(race_id=race.id).count() == 1


def test_predict_404_for_unknown_race(client):
    assert client.post("/races/424242/predict").status_code == 404
