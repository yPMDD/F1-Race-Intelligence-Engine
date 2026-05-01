import logging
from storage.postgres.database import SessionLocal
from storage.postgres.models import RaceModel, RacePrediction
from nlp.agents.graph import run_f1_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Precompute")

def precompute_2026_season():
    from ml.predictor import F1Predictor
    F1Predictor._stats_cache = None # Force refresh stats from DB
    
    db = SessionLocal()
    try:
        # 1. Get all 2026 races
        races = db.query(RaceModel).filter(RaceModel.year == 2026).order_by(RaceModel.round_num).all()
        logger.info(f"Found {len(races)} races in the 2026 season.")

        for race in races:
            # 2. Check if already predicted
            exists = db.query(RacePrediction).filter(RacePrediction.race_id == race.id).first()
            if exists:
                logger.info(f"Skipping {race.name} (Already cached)")
                continue

            # 3. Run AI Simulation
            logger.info(f"Simulating Strategy for: {race.name} Grand Prix...")
            try:
                prompt = f"Predict the 2026 {race.name} Grand Prix finishing order."
                prediction = run_f1_agent(prompt)
                
                new_pred = RacePrediction(
                    race_id=race.id,
                    prediction_text=prediction,
                    model_version="v2_hybrid"
                )
                db.add(new_pred)
                db.commit()
                logger.info(f"Success: {race.name} prediction stored.")
            except Exception as e:
                logger.error(f"Failed to simulate {race.name}: {e}")
                db.rollback()

    finally:
        db.close()

if __name__ == "__main__":
    precompute_2026_season()
