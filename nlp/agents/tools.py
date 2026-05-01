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
def query_2026_regulations(query: str) -> str:
    """
    Search the official 2026 FIA F1 Technical and Sporting Regulations for answers 
    to questions about rules, technical specifications, and penalties.
    """
    logger.info(f"Agent tool: querying 2026 regulations for '{query}'")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        docs = vectorstore.similarity_search(query, k=3)
        
        context = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])
        return context
    except Exception as e:
        return f"Error querying regulations: {str(e)}"

@tool
def get_race_telemetry_summary(driver_name: str) -> str:
    """
    Retrieve the latest race telemetry and timing summary for a specific driver 
    from the SQL database. Includes lap times and team information.
    """
    logger.info(f"Agent tool: getting telemetry for {driver_name}")
    db: Session = SessionLocal()
    try:
        # Simple fuzzy search for driver
        driver = db.query(DriverModel).filter(DriverModel.name.ilike(f"%{driver_name}%")).first()
        if not driver:
            return f"Driver '{driver_name}' not found in database."
        
        # Get latest laps for this driver
        laps = db.query(LapModel).filter(LapModel.driver_id == driver.driver_id).order_by(LapModel.lap_number.desc()).limit(5).all()
        
        if not laps:
            return f"No telemetry data found for {driver.name} (Team: {driver.team_name})."
            
        summary = f"Driver: {driver.name}\nTeam: {driver.team_name}\nLatest Laps:\n"
        for lap in laps:
            summary += f"- Lap {lap.lap_number}: {lap.lap_time_ms/1000.0 if lap.lap_time_ms else 'N/A'}s (S1: {lap.sector1_ms/1000.0 if lap.sector1_ms else 'N/A'}s, S2: {lap.sector2_ms/1000.0 if lap.sector2_ms else 'N/A'}s, S3: {lap.sector3_ms/1000.0 if lap.sector3_ms else 'N/A'}s)\n"
            
        return summary
    except Exception as e:
        return f"Error retrieving telemetry: {str(e)}"
    finally:
        db.close()
