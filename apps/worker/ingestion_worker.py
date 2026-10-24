import os
import sys

# Add project root to python path to allow imports from top-level packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import fastf1
import requests
import time
from datetime import datetime, timezone
import psycopg2
from storage.postgres.database import engine, SessionLocal
from storage.postgres.models import Base, RaceModel, DriverModel, LapModel, ResultModel
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure data directory exists for caching fastf1
os.makedirs('data/cache', exist_ok=True)
fastf1.Cache.enable_cache('data/cache')

def init_db():
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)

def fetch_and_store_structured_data(year: int, round_num: int):
    """Ingest one race's classification and (optionally) lap times.

    Laps and results degrade independently. `Session.load()` derives lap data
    from timing data, which fastf1 fetches from a fallback API that is
    frequently rate-limited; when that fails `session.laps` raises
    DataNotLoadedError while `session.results` is still perfectly usable.

    The previous version read `laps_df = race_session.laps` BEFORE processing
    results, so that single failure aborted the function and discarded a
    complete, valid classification for every affected race. Across a full
    three-season backfill this stored zero results at all.
    """
    logger.info(f"Fetching structured data for Year: {year}, Round: {round_num}")

    session = SessionLocal()
    try:
        race_session = fastf1.get_session(year, round_num, "R")
        race_session.load(telemetry=False, weather=False)

        # Insert or fetch the race row.
        race = session.query(RaceModel).filter_by(year=year, round_num=round_num).first()
        if not race:
            race = RaceModel(
                year=year,
                round_num=round_num,
                name=race_session.event["EventName"],
                date=race_session.date,
            )
            session.add(race)
            session.commit()
            session.refresh(race)

        # --- Results (classification) first: the most valuable data, and the
        # --- one that survives a timing-data failure.
        results_stored = 0
        try:
            results_df = race_session.results
        except Exception as exc:
            logger.warning("%s R%s: no classification available (%s)",
                           year, round_num, exc)
            results_df = None

        if results_df is not None:
            for _, res in results_df.iterrows():
                driver_name = res["Abbreviation"]
                driver_number = str(res["DriverNumber"])

                driver = session.query(DriverModel).filter_by(driver_id=driver_name).first()
                if not driver:
                    driver = DriverModel(
                        driver_id=driver_name,
                        number=int(driver_number) if driver_number.isdigit() else 0,
                        abbreviation=driver_name,
                        name=res["FullName"],
                        team_name=res["TeamName"],
                    )
                    session.add(driver)
                    session.flush()

                result_record = session.query(ResultModel).filter_by(
                    race_id=race.id, driver_id=driver_name
                ).first()

                position = int(res["Position"]) if pd.notnull(res["Position"]) else 0
                points = float(res["Points"]) if pd.notnull(res["Points"]) else 0.0
                status = str(res["Status"])

                if result_record:
                    result_record.position = position
                    result_record.points = points
                    result_record.status = status
                else:
                    session.add(ResultModel(
                        race_id=race.id,
                        driver_id=driver_name,
                        position=position,
                        points=points,
                        status=status,
                    ))
                    results_stored += 1

        # --- Laps: best-effort. Never let this block the classification.
        laps_stored = 0
        try:
            laps_df = race_session.laps
        except Exception as exc:
            logger.info("%s R%s: lap timing unavailable (%s) - classification only.",
                        year, round_num, type(exc).__name__)
            laps_df = None

        if laps_df is not None:
            # Re-running the worker must not duplicate laps.
            existing_lap_keys = {
                (l.driver_id, l.lap_number)
                for l in session.query(LapModel).filter(LapModel.race_id == race.id).all()
            }

            def to_ms(td):
                return int(td.total_seconds() * 1000) if pd.notnull(td) else None

            for _, lap in laps_df.iterrows():
                driver_name = lap["Driver"]
                lap_number = int(lap["LapNumber"])
                if (driver_name, lap_number) in existing_lap_keys:
                    continue
                session.add(LapModel(
                    race_id=race.id,
                    driver_id=driver_name,
                    lap_number=lap_number,
                    lap_time_ms=to_ms(lap["LapTime"]),
                    sector1_ms=to_ms(lap["Sector1Time"]),
                    sector2_ms=to_ms(lap["Sector2Time"]),
                    sector3_ms=to_ms(lap["Sector3Time"]),
                    compound=str(lap["Compound"]) if pd.notnull(lap["Compound"]) else None,
                ))
                existing_lap_keys.add((driver_name, lap_number))
                laps_stored += 1

        session.commit()
        logger.info("  %s: %d results, %d laps stored",
                    race.name, results_stored, laps_stored)
        return results_stored, laps_stored

    except Exception as e:
        session.rollback()
        logger.error("Error during ingestion for %s R%s: %s", year, round_num, e)
        return 0, 0
    finally:
        session.close()

def fetch_unstructured_document():
    logger.info("Fetching unstructured sample data (FIA document stub)...")
    url = "https://www.fia.com/sites/default/files/decision-document/2023%20Bahrain%20Grand%20Prix%20-%20Race%20Lap%20Analysis.pdf"
    output_path = "data/raw/2023_Bahrain_Race_Lap_Analysis.pdf"
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Document saved to {output_path}")
        else:
            logger.warning(f"Failed to download document. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to fetch unstructured doc: {e}")

def _is_past(dt) -> bool:
    """True if the given datetime has already happened.

    fastf1 hands back tz-aware timestamps; `datetime.now()` is naive, so the
    previous `race.date < datetime.now()` comparison raised TypeError.
    """
    if dt is None:
        return False
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt <= datetime.utcnow()


def ingest_seasons(years, ingest_results=True, delay_seconds: float = 1.5):
    """Sync calendars and (optionally) deep data for each season."""
    totals = {"races": 0, "results": 0, "laps": 0, "failed": 0}

    for year in years:
        logger.info("--- SYNCING CALENDAR FOR %s SEASON ---", year)
        db = SessionLocal()
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            # fastf1 enforces a 500 calls/hour budget; a blown quota used to
            # abort the entire remaining season.
            logger.error("Could not fetch %s schedule: %s", year, e)
            db.close()
            continue
        finally:
            try:
                db.close()
            except Exception:
                pass

        for _, event in schedule.iterrows():
            if event["EventFormat"] == "testing":
                continue
            round_num = int(event["RoundNumber"])
            if round_num == 0:
                continue

            db = SessionLocal()
            try:
                race = db.query(RaceModel).filter_by(year=year, round_num=round_num).first()
                if not race:
                    event_date = (
                        event["Sessions"][-1]["Date"]
                        if "Sessions" in event and event["Sessions"]
                        else event["EventDate"]
                    )
                    if hasattr(event_date, "tzinfo") and event_date.tzinfo is not None:
                        event_date = event_date.astimezone(timezone.utc).replace(tzinfo=None)
                    race = RaceModel(
                        year=year,
                        round_num=round_num,
                        name=event["EventName"],
                        date=event_date,
                    )
                    db.add(race)
                    db.commit()
                    logger.info("Race synced: %s %s", year, race.name)
                totals["races"] += 1
                race_date, race_id = race.date, race.id
            finally:
                db.close()

            if ingest_results and _is_past(race_date):
                # Be gentle with the upstream quota.
                time.sleep(delay_seconds)
                r, l = fetch_and_store_structured_data(year, round_num)
                totals["results"] += r
                totals["laps"] += l
                if r == 0 and l == 0:
                    totals["failed"] += 1

    logger.info(
        "Ingestion summary: %d races seen, %d results, %d laps, %d sessions with no data",
        totals["races"], totals["results"], totals["laps"], totals["failed"],
    )
    return totals


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="F1 telemetry ingestion worker")
    parser.add_argument("--years", default="2024,2025,2026",
                        help="comma-separated seasons")
    parser.add_argument("--calendar-only", action="store_true",
                        help="sync race metadata only, no lap/result ingestion")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="seconds between sessions (upstream is rate-limited)")
    args = parser.parse_args()

    init_db()
    years = [int(y) for y in args.years.split(",") if y.strip()]
    ingest_seasons(
        years,
        ingest_results=not args.calendar_only,
        delay_seconds=args.delay,
    )
    logger.info("Calendar sync and history ingestion complete!")
