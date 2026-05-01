import joblib
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

class F1Predictor:
    def __init__(self):
        base_path = "storage/models"
        self.model_path = os.path.join(base_path, "f1_predictor.joblib")
        self.mapping_path = os.path.join(base_path, "mappings.json")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("ML Model not found. Run train_model.py first.")
            
        self.model = joblib.load(self.model_path)
        with open(self.mapping_path, "r") as f:
            self.mappings = json.load(f)
        
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://f1user:f1password@localhost:5433/f1db")
        self.engine = create_engine(DATABASE_URL)

    _stats_cache = None

    def predict_race_order(self, track_name: str):
        """
        Runs optimized batch prediction for the official 2026 grid.
        """
        from ml.config import GRID_2026
        
        # 1. Get/Cache latest driver stats
        if F1Predictor._stats_cache is None:
            query = """
            WITH driver_stats AS (
                SELECT 
                    res.driver_id,
                    res.position,
                    res.points,
                    r.year,
                    ROW_NUMBER() OVER(PARTITION BY res.driver_id ORDER BY r.year DESC, r.round_num DESC) as recency
                FROM results res
                JOIN races r ON res.race_id = r.id
            )
            SELECT 
                driver_id,
                AVG(position) FILTER (WHERE recency <= 5) as momentum,
                SUM(points) FILTER (WHERE year = 2025) as season_points
            FROM driver_stats
            GROUP BY driver_id
            """
            stats_df = pd.read_sql(query, self.engine)
            F1Predictor._stats_cache = stats_df.set_index('driver_id').to_dict('index')
        
        stats_dict = F1Predictor._stats_cache
        
        # 2. Map track name to index
        track_idx = self.mappings["tracks"].get(track_name, 0)
        
        # 3. Prepare Batch Data
        batch_features = []
        for entry in GRID_2026:
            driver_id = entry["driver_id"]
            d_idx = self.mappings["drivers"].get(driver_id)
            d_stats = stats_dict.get(driver_id, {"momentum": 12.0, "season_points": 0.0})
            
            batch_features.append({
                'driver_momentum': d_stats['momentum'] if d_stats['momentum'] else 12.0,
                'track_history': 10.0, 
                'season_points': d_stats['season_points'] if d_stats['season_points'] else 0.0,
                'driver_idx': d_idx if d_idx is not None else -1,
                'track_idx': track_idx
            })
        
        # 4. Batch Prediction
        features_df = pd.DataFrame(batch_features)
        scores = self.model.predict(features_df)
        
        # 5. Assemble Results
        predictions = []
        for i, entry in enumerate(GRID_2026):
            predictions.append({
                "driver_id": entry["driver_id"],
                "name": entry["name"],
                "team": entry["team"],
                "score": float(scores[i])
            })
            
        # 6. Sort and Rank
        ranked = sorted(predictions, key=lambda x: x['score'])
        for i, item in enumerate(ranked):
            item['final_pos'] = i + 1
            
        return ranked

if __name__ == "__main__":
    predictor = F1Predictor()
    print("Testing ML Prediction for Zandvoort...")
    results = predictor.predict_race_order("Dutch Grand Prix")
    for res in results[:10]:
        print(f"P{res['final_pos']}: {res['driver_id']} (Score: {res['score']:.2f})")
