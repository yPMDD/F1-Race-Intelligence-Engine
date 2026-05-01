import logging
from nlp.agents.graph import run_f1_agent

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
    years = db.query(RaceModel.year).distinct().all()
    return [y[0] for y in years]

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
    
    race = db.query(RaceModel).filter(RaceModel.id == race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
        
    prompt = (
        f"Perform a detailed prediction for the {race.year} {race.name}. "
        f"Step 1: Check the historical winners at this track for 2024 and 2025. "
        f"Step 2: Check 2026 Technical Regulations for aero/weight changes. "
        f"Step 3: Analyze performance trends from recent 2026 races. "
        f"Provide a predicted Top 10 with a brief reasoning for each driver."
    )
    
    try:
        prediction = run_f1_agent(prompt)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        prediction = "[AI] Unable to generate prediction at this time. Please try again later."
    return {"prediction": prediction}
