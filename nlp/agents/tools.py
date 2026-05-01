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

# Shared embeddings instance (avoid reloading on every call)
_embeddings = None
_vectorstore = None

def _get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
    return _vectorstore, _embeddings

def _hybrid_search(query: str, year: int = 2026, k: int = 6) -> list:
    """
    Hybrid search: combines semantic similarity + BM25-style keyword scoring.
    Returns a deduplicated, re-ranked list of document chunks.
    """
    vectorstore, embeddings = _get_vectorstore()

    # 1. Semantic search — retrieve top candidates
    sem_filter = {"year": year} if year else None
    sem_docs = vectorstore.similarity_search(query, k=k * 2, filter=sem_filter)

    # 2. BM25-style keyword boosting: boost docs that contain exact query keywords
    keywords = set(query.lower().split())
    stop_words = {"the", "a", "is", "what", "how", "of", "in", "for", "and", "to"}
    keywords -= stop_words

    scored_docs = []
    seen_content = set()
    for doc in sem_docs:
        content_lower = doc.page_content.lower()
        # Count keyword hits for BM25 boost
        keyword_hits = sum(1 for kw in keywords if kw in content_lower)
        score = keyword_hits * 2  # BM25 boost weight
        
        # Deduplicate by content hash
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            scored_docs.append((score, doc))

    # 3. Sort: keyword-boosted docs first, then pure semantic
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:k]]

@tool
def query_f1_regulations(query: str, year: int = 2026) -> str:
    """
    Search the FIA Formula 1 Technical and Sporting Regulations for specific rules.
    Use this to clarify technical specifications, weight limits, or sporting procedures.
    Always defaults to the 2026 regulation year for accuracy.
    """
    logger.info(f"Agent tool: hybrid regulation search for '{query}' (Year: {year})")
    try:
        docs = _hybrid_search(query, year=year, k=5)
        if not docs:
            return f"No regulation data found for '{query}' in {year}. The documents may not have been ingested."
        context = "\n\n---\n\n".join([
            f"[Source: {d.metadata.get('source', 'Unknown')} | Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        ])
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

@tool
def get_ml_prediction(track_name: str) -> str:
    """
    Execute the custom XGBoost ML model to predict the full finishing order for a specific track.
    Use this as the mathematical baseline for your final classification.
    """
    logger.info(f"Agent tool: getting ML prediction for {track_name}")
    try:
        from ml.predictor import F1Predictor
        predictor = F1Predictor()
        results = predictor.predict_race_order(track_name)
        
        summary = f"ML_FORECAST|{track_name}\n"
        for res in results:
            summary += f"P{res['final_pos']}|{res['driver_id']}|{res['team']}|SCORE:{res['score']:.2f}\n"
            
        return summary
    except Exception as e:
        return f"Error running ML model: {str(e)}"
