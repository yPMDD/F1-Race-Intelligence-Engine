import os
import re
from collections import Counter
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

# Weight given to the semantic channel when blending with the lexical one.
# > 0.5 means "trust the embedding", which is what we want: the lexical pass
# is there to break ties and rescue exact regulation terms ("768 kg",
# "C6.1") that the embedding blurs, not to outvote it.
HYBRID_SEMANTIC_WEIGHT = 0.7

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "what", "which",
    "how", "of", "in", "on", "at", "to", "for", "and", "or", "do", "does", "did",
    "it", "its", "as", "by", "that", "this", "these", "those", "there", "their",
    "s", "t",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# The FIA PDFs write magnitudes glued to their unit -- "726kg", "190kW",
# "1900mm". A user asking about "726 kg" should match that, so we also emit
# the numeric part as its own token. Internal '.'/'-' are deliberately kept
# (article refs like C6.2 must survive intact), which is why this is a
# post-process on the token rather than part of the regex.
_NUM_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+)$")

# Shared embeddings instance (avoid reloading on every call)
_embeddings = None
_vectorstore = None

def _get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
    return _vectorstore, _embeddings


def _tokenize(text: str) -> list:
    """Lowercase, split on non-alphanumerics, drop stop words.

    The previous implementation used `query.lower().split()`, which left
    punctuation attached -- "What is the 2026 car weight?" yielded the token
    '2026?', a string that can never occur in the document body.
    """
    tokens = []
    for tok in _WORD_RE.findall(text.lower()):
        if tok in STOP_WORDS:
            continue
        tokens.append(tok)
        m = _NUM_UNIT_RE.match(tok)
        if m:
            # emit both halves too, so "726kg" matches a query for "726 kg"
            tokens.append(m.group(1))
            tokens.append(m.group(2))
    return tokens


def _lexical_score(terms: list, text: str, k1: float = 1.2, b: float = 0.75) -> float:
    """Saturating BM25-style term-frequency score, normalised to roughly [0, 1].

    Uses word-boundary tokens (not raw substring counts) and the standard
    saturating denominator, so a term repeated 50 times cannot dominate a
    document the way a plain count * 2 did. IDF is omitted deliberately: we
    rank within a small over-fetched candidate pool rather than a corpus, so
    per-term IDF over that pool is unstable and not worth the complexity.
    """
    if not terms:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    tf = Counter(tokens)
    doc_len = len(tokens)
    total = 0.0
    for term in terms:
        f = tf.get(term, 0)
        if not f:
            continue
        total += (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / 1500.0))
    return total / len(terms)


def _hybrid_search(query: str, year: int = 2026, k: int = 6) -> list:
    """
    Hybrid search: blends vector similarity with a BM25-style keyword score.

    Returns a deduplicated, re-ranked list of document chunks.

    Previously this called `similarity_search`, which returns Documents with
    *no* score attached, then ranked purely on `keyword_hits * 2`. The semantic
    ordering was therefore discarded outright and a single incidental keyword
    match outranked the best embedding match. We now ask Chroma for relevance
    scores and blend both channels.
    """
    vectorstore, embeddings = _get_vectorstore()

    # 1. Semantic search with scores — over-fetch so reranking has room to work.
    sem_filter = {"year": year} if year else None
    fetch_k = max(k * 4, 20)
    pairs = vectorstore.similarity_search_with_relevance_scores(
        query, k=fetch_k, filter=sem_filter
    )
    if not pairs:
        logger.warning("Hybrid search returned no candidates for %r", query)
        return []

    terms = _tokenize(query)
    alpha = HYBRID_SEMANTIC_WEIGHT

    # 2. Blend both channels, dedupe, rank.
    #
    # The relevance score is used as-is rather than min-max normalised across
    # the candidate pool. Chroma guarantees it lies in [0, 1] with 1 = closest,
    # and this collection is stored with space='l2', which compresses values
    # into a narrow band (~0.24-0.32 across the 2026 regulations). That
    # compression is *meaningful*: if the top candidates really are that
    # similar, they should rank close together and let the lexical channel
    # break the tie. Min-max rescaling would stretch a 0.01 spread into the
    # full 0..1 range and amplify meaningless noise into a decisive ranking.
    scored_docs = []
    seen_content = set()
    for doc, relevance in pairs:
        content_hash = hash(doc.page_content[:200])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        semantic = max(0.0, min(1.0, relevance))
        lexical = _lexical_score(terms, doc.page_content)
        scored_docs.append((alpha * semantic + (1 - alpha) * lexical, doc))

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
