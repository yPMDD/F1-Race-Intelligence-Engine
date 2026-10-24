import logging
import time
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
from apps.api.schemas.race import Race, Driver, Lap
from storage.postgres.models import RaceModel, DriverModel, LapModel
from storage.postgres.database import get_db

router = APIRouter(
    prefix="/races",
    tags=["races"],
)


@router.get("/", response_model=List[Race])
def read_races(year: int = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    query = db.query(RaceModel)
    if year:
        query = query.filter(RaceModel.year == year)
    races = query.offset(skip).limit(limit).all()
    return races


@router.get("/years", response_model=List[int])
def get_available_years(db: Session = Depends(get_db)):
    try:
        years_tuples = db.query(RaceModel.year).distinct().all()
        years = [int(y[0]) for y in years_tuples if y[0] is not None]
        return sorted(years, reverse=True)
    except Exception as e:
        logger.error(f"Error in get_available_years: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-year/{year}", response_model=List[Race])
def get_races_by_year(year: int, db: Session = Depends(get_db)):
    races = db.query(RaceModel).filter(RaceModel.year == year).all()
    return races


@router.get("/{race_id}", response_model=Race)
def read_race(race_id: int, db: Session = Depends(get_db)):
    race = db.query(RaceModel).filter(RaceModel.id == race_id).first()
    if race is None:
        raise HTTPException(status_code=404, detail="Race not found")
    return race


@router.get("/{race_id}/laps", response_model=List[Lap])
def read_race_laps(race_id: int, db: Session = Depends(get_db)):
    laps = db.query(LapModel).filter(LapModel.race_id == race_id).all()
    return laps


@router.post("/{race_id}/predict")
def predict_race_result(race_id: int, db: Session = Depends(get_db)):
    from nlp.agents.graph import run_f1_agent
    from storage.postgres.models import RacePrediction

    race = db.query(RaceModel).filter(RaceModel.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    cached = db.query(RacePrediction).filter(RacePrediction.race_id == race_id).first()
    if cached:
        time.sleep(1.2)  # Simulate AI thinking for UI feedback
        return {"prediction": cached.prediction_text, "source": "cache"}

    prompt = f"Predict the 2026 {race.name} Grand Prix finishing order."
    try:
        prediction = run_f1_agent(prompt)
    except Exception as e:
        # Previously this swallowed the failure and returned a "[AI] Unable to
        # generate prediction" string with source="ai" -- indistinguishable
        # from a real forecast to the dashboard, which renders it as one.
        logger.exception("Prediction failed for race %s", race_id)
        return {
            "prediction": f"[AI] Unable to generate prediction at this time ({e}).",
            "source": "error",
        }

    if not prediction:
        logger.error("Empty prediction for race %s", race_id)
        return {"prediction": "[AI] Model returned an empty response.", "source": "error"}

    db.add(
        RacePrediction(
            race_id=race_id,
            prediction_text=prediction,
            model_version="v2_hybrid",
        )
    )
    db.commit()
    return {"prediction": prediction, "source": "ai"}


def _is_future(race: RaceModel) -> bool:
    """True if the race has not been run yet."""
    if race.date is None:
        return False
    date = race.date
    # fastf1 hands back tz-aware datetimes; the column is naive. Normalise
    # before comparing or the subtraction blows up.
    if date.tzinfo is not None:
        date = date.astimezone(timezone.utc).replace(tzinfo=None)
    return date > datetime.utcnow()


@router.post("/{race_id}/results")
def submit_race_results(race_id: int, results: List[dict], db: Session = Depends(get_db)):
    """
    Store final results for a race and re-simulate the remaining season.
    """
    from storage.postgres.models import ResultModel, RacePrediction

    race = db.query(RaceModel).filter(RaceModel.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    # 1. Upsert results. The previous version always INSERTed, so replaying the
    #    same race duplicated every row (there is even a scratch script whose
    #    only job is de-duplicating them afterwards).
    for res in results:
        existing = (
            db.query(ResultModel)
            .filter(
                ResultModel.race_id == race_id,
                ResultModel.driver_id == res["driver_id"],
            )
            .first()
        )
        if existing:
            existing.position = res["position"]
            existing.points = res.get("points", 0.0)
            existing.status = res.get("status", "Finished")
        else:
            db.add(
                ResultModel(
                    race_id=race_id,
                    driver_id=res["driver_id"],
                    position=res["position"],
                    points=res.get("points", 0.0),
                    status=res.get("status", "Finished"),
                )
            )
    db.commit()
    logger.info("Results recorded for race %s (%s)", race_id, race.name)

    # 2. Invalidate only forecasts for races that have NOT been run.
    #    This used to be `db.query(RacePrediction).delete()`, which wiped every
    #    stored forecast including completed races, despite the comment saying
    #    "future predictions".
    invalidated = 0
    for other in db.query(RaceModel).all():
        if other.id == race_id or not _is_future(other):
            continue
        deleted = (
            db.query(RacePrediction)
            .filter(RacePrediction.race_id == other.id)
            .delete(synchronize_session=False)
        )
        invalidated += deleted or 0
    db.commit()
    if invalidated:
        logger.info("Invalidated %d future prediction(s)", invalidated)

    # 3. Cached derived state is now stale:
    #    - the predictor's class-level driver stats (used momentum/points)
    #    - semantic-cache answers about the state of the season
    try:
        from ml.predictor import F1Predictor
        F1Predictor.invalidate_cache()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not invalidate predictor cache: %s", e)

    try:
        from nlp.cache.semantic_cache import cache_instance
        cache_instance.invalidate_all()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not invalidate semantic cache: %s", e)

    # 4. Re-simulate the rest of the season off the request thread.
    def _resimulate():
        try:
            from ml.precompute_all import precompute_2026_season
            precompute_2026_season()
        except Exception:
            logger.exception("Season re-simulation failed")

    thread = threading.Thread(target=_resimulate, daemon=True)
    thread.start()

    return {"status": "Results recorded. Season re-simulation started in background."}
