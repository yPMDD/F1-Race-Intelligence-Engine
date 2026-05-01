from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, BigInteger
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class RaceModel(Base):
    __tablename__ = "races"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, index=True)
    round_num = Column(Integer)
    name = Column(String)
    date = Column(DateTime)

    laps = relationship("LapModel", back_populates="race", cascade="all, delete-orphan")

class DriverModel(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(String, unique=True, index=True) # e.g. "max_verstappen"
    number = Column(Integer)
    abbreviation = Column(String)
    name = Column(String)
    team_name = Column(String)

class LapModel(Base):
    __tablename__ = "laps"

    id = Column(Integer, primary_key=True, index=True)
    race_id = Column(Integer, ForeignKey("races.id"))
    driver_id = Column(String, ForeignKey("drivers.driver_id"))
    lap_number = Column(Integer)
    lap_time_ms = Column(BigInteger, nullable=True)
    sector1_ms = Column(BigInteger, nullable=True)
    sector2_ms = Column(BigInteger, nullable=True)
    sector3_ms = Column(BigInteger, nullable=True)
    compound = Column(String, nullable=True)

    race = relationship("RaceModel", back_populates="laps")
