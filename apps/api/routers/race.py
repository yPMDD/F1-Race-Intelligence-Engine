import logging
import time
from nlp.agents.graph import run_f1_agent

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
        time.sleep(1.2) # Simulate AI thinking for UI feedback
        return {"prediction": cached.prediction_text, "source": "cache"}

    try:
        prompt = f"Predict the 2026 {race.name} Grand Prix finishing order."
        prediction = run_f1_agent(prompt)
        
        new_pred = RacePrediction(race_id=race_id, prediction_text=prediction, model_version="v2_hybrid")
        db.add(new_pred)
        db.commit()
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        prediction = "[AI] Unable to generate prediction at this time."
    return {"prediction": prediction, "source": "ai"}

@router.post("/{race_id}/results")
async def submit_race_results(race_id: int, results: List[dict], db: Session = Depends(get_db)):
    """
    Submits final results for a race and automatically triggers a season re-simulation.
    """
    from storage.postgres.models import ResultModel, RacePrediction
    from ml.precompute_all import precompute_2026_season
    import threading

    # 1. Store results in DB
    for res in results:
        new_result = ResultModel(
            race_id=race_id,
            driver_id=res['driver_id'],
            position=res['position'],
            points=res['points'],
            status=res.get('status', 'Finished')
        )
        db.add(new_result)
    
    db.commit()
    logger.info(f"Results recorded for race {race_id}")

    # 2. Invalidate all FUTURE predictions (because momentum/points have changed)
    db.query(RacePrediction).delete() 
    db.commit()
    logger.info("Future predictions invalidated. Starting re-simulation...")

    # 3. Trigger re-simulation in a background thread so the API remains fast
    thread = threading.Thread(target=precompute_2026_season)
    thread.start()

    return {"status": "Results recorded. Season re-simulation started in background."}
