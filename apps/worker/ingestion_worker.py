import os
import sys

# Add project root to python path to allow imports from top-level packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import fastf1
import requests
from datetime import datetime
import psycopg2
from storage.postgres.database import engine, SessionLocal
from storage.postgres.models import Base, RaceModel, DriverModel, LapModel
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure data directory exists for caching fastf1
os.makedirs('data/cache', exist_ok=True)
fastf1.Cache.enable_cache('data/cache')

def init_db():
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)

def fetch_and_store_structured_data(year: int, round_num: int):
    logger.info(f"Fetching structured data for Year: {year}, Round: {round_num}")
    
    session = SessionLocal()
    try:
        # Load the session/race
        race_session = fastf1.get_session(year, round_num, 'R')
        race_session.load(telemetry=False, weather=False) # Load laps and drivers

        # Insert Race
        race = session.query(RaceModel).filter_by(year=year, round_num=round_num).first()
        if not race:
            race = RaceModel(
                year=year,
                round_num=round_num,
                name=race_session.event['EventName'],
                date=race_session.date
            )
            session.add(race)
            session.commit()
            session.refresh(race)

        logger.info(f"Race ID {race.id} loaded.")

        # Process Laps and Drivers
        laps_df = race_session.laps
        
        for index, lap in laps_df.iterrows():
            driver_number = str(lap['DriverNumber'])
            driver_name = lap['Driver']
            
            # Upsert Driver
            driver = session.query(DriverModel).filter_by(driver_id=driver_name).first()
            if not driver:
                driver = DriverModel(
                    driver_id=driver_name,
                    number=int(driver_number) if driver_number.isdigit() else 0,
                    abbreviation=driver_name,
                    name=driver_name, # We might need a better mapping later
                    team_name=lap['Team']
                )
                session.add(driver)
                session.commit()
            
            # Convert timedelta to ms safely
            def to_ms(td):
                return int(td.total_seconds() * 1000) if pd.notnull(td) else None

            # Insert Lap
            lap_record = LapModel(
                race_id=race.id,
                driver_id=driver_name,
                lap_number=int(lap['LapNumber']),
                lap_time_ms=to_ms(lap['LapTime']),
                sector1_ms=to_ms(lap['Sector1Time']),
                sector2_ms=to_ms(lap['Sector2Time']),
                sector3_ms=to_ms(lap['Sector3Time']),
                compound=str(lap['Compound']) if pd.notnull(lap['Compound']) else None
            )
            session.add(lap_record)
        
        session.commit()
        logger.info("Structured data ingested successfully.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error during ingestion: {e}")
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

if __name__ == "__main__":
    init_db()
    # Ingest 2023 Bahrain Grand Prix
    fetch_and_store_structured_data(2023, 1)
    # Download sample document
    fetch_unstructured_document()
