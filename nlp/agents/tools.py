import os
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sqlalchemy.orm import Session
from storage.postgres.database import SessionLocal
from storage.postgres.models import LapModel, DriverModel, RaceModel
import logging

logger = logging.getLogger(__name__)

# Constants for RAG
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@tool
def query_f1_regulations(query: str, year: int = None) -> str:
    """
    Search the FIA Formula 1 Technical and Sporting Regulations for specific rules.
    Use this to clarify technical specifications, weight limits, or sporting procedures.
    If a specific year (e.g., 2024, 2025, 2026) is provided, it will prioritize that year.
    """
    logger.info(f"Agent tool: querying regulations for '{query}' (Year: {year})")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        kwargs = {"k": 5}
        if year:
            kwargs["filter"] = {"year": year}
            
        docs = vectorstore.similarity_search(query, **kwargs)
        
        context = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])
        return context
    except Exception as e:
        return f"Error querying regulations: {str(e)}"

@tool
def get_track_history(track_name: str) -> str:
    """
    Retrieve historical podiums and winners for a specific track across all ingested seasons.
    Use this to identify which drivers/teams historically perform well at this circuit.
    """
    logger.info(f"Agent tool: getting track history for {track_name}")
    db: Session = SessionLocal()
    try:
        # Find all races at this track
        races = db.query(RaceModel).filter(RaceModel.name.ilike(f"%{track_name}%")).all()
        if not races:
            return f"No historical data found for track '{track_name}'."
            
        summary = f"--- Historical Performance: {track_name} ---\n"
        from storage.postgres.models import ResultModel
        
        for race in sorted(races, key=lambda x: x.year, reverse=True):
            if race.year >= 2026: continue # Skip future/current
            
            top_3 = db.query(ResultModel).filter(ResultModel.race_id == race.id).order_by(ResultModel.position).limit(3).all()
            if top_3:
                summary += f"\nSeason {race.year}:\n"
                for res in top_3:
                    summary += f"- P{res.position}: {res.driver_id} (Status: {res.status})\n"
                    
        return summary
    except Exception as e:
        return f"Error retrieving track history: {str(e)}"
    finally:
        db.close()

@tool
def get_race_telemetry_summary(year: int, race_name: str, driver_name: str = None) -> str:
    """
    Retrieve race results and telemetry summaries for a specific year and race.
    If driver_name is provided, it focus on that driver's laps.
    Otherwise, it returns the top classification (podium) and general race info.
    """
    logger.info(f"Agent tool: getting telemetry for {year} {race_name} (Driver: {driver_name})")
    db: Session = SessionLocal()
    try:
        # Find Race
        race = db.query(RaceModel).filter(
            RaceModel.year == year,
            RaceModel.name.ilike(f"%{race_name}%")
        ).first()
        
        if not race:
            return f"Race '{race_name}' in {year} not found in database."
        
        # Get Podium
        from storage.postgres.models import ResultModel
        results = db.query(ResultModel).filter(ResultModel.race_id == race.id).order_by(ResultModel.position).limit(3).all()
        
        summary = f"--- {race.year} {race.name} Summary ---\n"
        summary += "Podium:\n"
        for res in results:
            summary += f"- P{res.position}: {res.driver_id} ({res.points} pts) - Status: {res.status}\n"
            
        if driver_name:
            # Get driver's laps
            driver = db.query(DriverModel).filter(DriverModel.driver_id.ilike(f"%{driver_name}%")).first()
            if driver:
                laps = db.query(LapModel).filter(
                    LapModel.race_id == race.id, 
                    LapModel.driver_id == driver.driver_id
                ).order_by(LapModel.lap_number).all()
                
                if laps:
                    times = [l.lap_time_ms for l in laps if l.lap_time_ms]
                    avg_lap = sum(times)/len(times) if times else 0
                    summary += f"\nDriver {driver.name} Analysis:\n"
                    summary += f"- Total Laps: {len(laps)}\n"
                    summary += f"- Average Lap: {avg_lap/1000.0:.3f}s\n"
                    summary += f"- Best Lap: {min(times)/1000.0:.3f}s\n"
            
        return summary
    except Exception as e:
        return f"Error retrieving telemetry: {str(e)}"
    finally:
        db.close()
