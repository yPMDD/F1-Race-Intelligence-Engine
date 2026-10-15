"""
Shared pytest fixtures.

Critically, DATABASE_URL is redirected to SQLite *before* any project module is
imported. `storage.postgres.database` reads the env var at import time and
`load_dotenv()` does not override already-set variables, so this wins over the
real Postgres URL in .env. That keeps the unit suite hermetic and runnable with
no external services.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./tests/.pytest_tmp.db")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

import pytest
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from storage.postgres.models import (
    Base,
    RaceModel,
    DriverModel,
    LapModel,
    ResultModel,
    RacePrediction,
)


@pytest.fixture(scope="session")
def engine():
    """Session-wide engine.

    The file is deliberately NOT deleted between runs: on Windows a lingering
    connection (apps.api.main holds its own engine at import time) would lock
    it. `Base.metadata.drop_all` in the per-test `db` fixture gives isolation
    without needing to remove the file.
    """
    url = os.environ["DATABASE_URL"]
    eng = sqlalchemy.create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng


@pytest.fixture()
def db(engine):
    """Fresh, empty tables for every test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def session_factory(engine):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


# --- sample data helpers -------------------------------------------------

# A compact but realistic 2026-style grid, mirroring ml/config.GRID_2026 so
# that DB joins in the predictor behave like production.
SAMPLE_RESULTS_2024 = [
    # driver_id, position, points
    ("VER", 1, 25.0), ("HAM", 2, 18.0), ("LEC", 3, 15.0),
    ("NOR", 4, 12.0), ("RUS", 5, 10.0), ("PIA", 6, 8.0),
]
SAMPLE_RESULTS_2025 = [
    ("VER", 2, 18.0), ("HAM", 1, 25.0), ("LEC", 3, 15.0),
    ("NOR", 4, 12.0), ("RUS", 5, 10.0), ("PIA", 6, 8.0),
]
SAMPLE_RESULTS_2026 = [
    ("VER", 1, 25.0), ("HAM", 2, 18.0), ("NOR", 3, 15.0),
    ("LEC", 4, 12.0), ("RUS", 5, 10.0), ("PIA", 6, 8.0),
]


def seed_race(session, year, round_num, name, results, date=None):
    from datetime import datetime
    race = RaceModel(
        year=year,
        round_num=round_num,
        name=name,
        date=date or datetime(year, 3, 1),
    )
    session.add(race)
    session.flush()

    for idx, (driver_id, position, points) in enumerate(results):
        d = session.query(DriverModel).filter_by(driver_id=driver_id).first()
        if not d:
            d = DriverModel(
                driver_id=driver_id,
                number=idx + 1,
                abbreviation=driver_id,
                name=driver_id,
                team_name="Test Team",
            )
            session.add(d)
        session.add(
            ResultModel(
                race_id=race.id,
                driver_id=driver_id,
                position=position,
                points=points,
                status="Finished",
            )
        )

    # add a handful of laps for VER
    for lap_no in range(1, 4):
        session.add(
            LapModel(
                race_id=race.id,
                driver_id="VER",
                lap_number=lap_no,
                lap_time_ms=90000 + lap_no * 250,
                sector1_ms=30000,
                sector2_ms=31000,
                sector3_ms=29000,
                compound="SOFT",
            )
        )
    session.commit()
    return race


@pytest.fixture()
def seeded_db(db):
    """Three seasons of data covering both past and 'future' 2026 rounds."""
    seed_race(db, 2024, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2024)
    seed_race(db, 2025, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2025)
    seed_race(db, 2026, 1, "Bahrain Grand Prix", SAMPLE_RESULTS_2026)
    # deliberately created as a past date so "future race" logic can distinguish
    from datetime import datetime, timedelta
    seed_race(
        db, 2026, 20, "Las Vegas Grand Prix", SAMPLE_RESULTS_2026,
        date=datetime.now() + timedelta(days=60),
    )
    return db
