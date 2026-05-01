from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from apps.api.schemas.race import Race, Driver, Lap
from storage.postgres.models import RaceModel, DriverModel, LapModel
from storage.postgres.database import get_db

router = APIRouter(
    prefix="/races",
    tags=["races"],
)

@router.get("/", response_model=List[Race])
def read_races(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    races = db.query(RaceModel).offset(skip).limit(limit).all()
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

@router.get("/drivers/", response_model=List[Driver])
def read_drivers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    drivers = db.query(DriverModel).offset(skip).limit(limit).all()
    return drivers
