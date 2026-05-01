from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

class DriverBase(BaseModel):
    driver_id: str
    number: int
    abbreviation: str
    name: str
    team_name: str

class DriverCreate(DriverBase):
    pass

class Driver(DriverBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class LapBase(BaseModel):
    driver_id: str
    lap_number: int
    lap_time_ms: Optional[int]
    sector1_ms: Optional[int]
    sector2_ms: Optional[int]
    sector3_ms: Optional[int]
    compound: Optional[str]

class LapCreate(LapBase):
    pass

class Lap(LapBase):
    id: int
    race_id: int
    model_config = ConfigDict(from_attributes=True)

class RaceBase(BaseModel):
    year: int
    round_num: int
    name: str
    date: datetime

class RaceCreate(RaceBase):
    pass

class Race(RaceBase):
    id: int
    laps: List[Lap] = []
    model_config = ConfigDict(from_attributes=True)
