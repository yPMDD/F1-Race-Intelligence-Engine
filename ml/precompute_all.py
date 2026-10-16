import logging
import time

from storage.postgres.database import SessionLocal
from storage.postgres.models import RaceModel, RacePrediction
from nlp.agents.graph import run_f1_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Precompute")


def precompute_2026_season(force: bool = False):
    """Re-simulate the remainder of the 2026 season, in round order.

    Sequential by design: each race's forecast depends on the momentum and
    points accumulated by every race before it, so round N+1 must be predicted
    after round N is folded in.

    Idempotent -- races that already have a stored forecast are skipped unless
    `force` is set.
    """
    from ml.predictor import F1Predictor

    # Driver form changed, so drop the memoised stats before the first call.
    F1Predictor.invalidate_cache()

    db = SessionLocal()
    try:
        races = db.query(RaceModel).filter(RaceModel.year == 2026).order_by(RaceModel.round_num).all()
        logger.info("Found %d races in the 2026 season.", len(races))

        simulated = 0
        for race in races:
            exists = db.query(RacePrediction).filter(RacePrediction.race_id == race.id).first()
            if exists and not force:
                logger.info("Skipping %s (already cached)", race.name)
                continue

            logger.info("Simulating strategy for %s...", race.name)
            started = time.monotonic()
            try:
                prediction = run_f1_agent(f"Predict the 2026 {race.name} Grand Prix finishing order.")

                if not prediction:
                    raise RuntimeError("agent returned an empty forecast")

                if force and exists:
                    exists.prediction_text = prediction
                    exists.model_version = "v2_hybrid"
                else:
                    db.add(RacePrediction(
                        race_id=race.id,
                        prediction_text=prediction,
                        model_version="v2_hybrid",
                    ))
                db.commit()
                simulated += 1
                logger.info(
                    "Stored %s forecast in %.1fs",
                    race.name, time.monotonic() - started,
                )
            except Exception:
                # run_f1_agent now re-raises instead of returning a
                # "Streaming error. Please retry." placeholder, so an
                # upstream failure no longer poisons the predictions table.
                logger.exception("Failed to simulate %s", race.name)
                db.rollback()

        logger.info("Season simulation complete: %d forecast(s) written.", simulated)
        return simulated
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    precompute_2026_season(force="--force" in sys.argv)
