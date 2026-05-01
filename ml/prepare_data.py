import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# Database setup
# Note: Use localhost for running from the host machine to the docker container
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://f1user:f1password@localhost:5433/f1db")
engine = create_engine(DATABASE_URL)

def fetch_training_data():
    """
    Fetches raw results and joins with race info to create a base dataset.
    """
    query = """
    SELECT 
        r.year, 
        r.name as race_name,
        res.driver_id,
        res.position,
        res.points,
        res.status
    FROM results res
    JOIN races r ON res.race_id = r.id
    WHERE r.year IN (2024, 2025)
    ORDER BY r.year, r.round_num
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("Warning: No data found in database. Ensure ingestion has been run.")
        return df

    # --- Feature Engineering ---
    
    # 1. Driver Momentum: Average position in the last 5 races
    df['driver_momentum'] = df.groupby('driver_id')['position'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    
    # 2. Track Specialist: Average position at this specific track
    df['track_history'] = df.groupby(['driver_id', 'race_name'])['position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # 3. Season Performance: Total points accumulated in the season so far
    df['season_points'] = df.groupby(['driver_id', 'year'])['points'].transform(
        lambda x: x.cumsum().shift(1)
    )
    
    # Fill NaNs (for early season or new drivers)
    df['driver_momentum'] = df['driver_momentum'].fillna(10.0)
    df['track_history'] = df['track_history'].fillna(10.0)
    df['season_points'] = df['season_points'].fillna(0.0)
    
    # 4. Encodings
    # We use categorical codes for the model
    df['driver_idx'] = df['driver_id'].astype('category').cat.codes
    df['track_idx'] = df['race_name'].astype('category').cat.codes
    
    # Target: The position we want to predict
    # Note: For XGBoost, we'll try to predict the position directly (Regression)
    # or we could do Classification. Let's start with Regression.
    
    return df

if __name__ == "__main__":
    print("Extracting F1 data from PostgreSQL...")
    dataset = fetch_training_data()
    
    if not dataset.empty:
        os.makedirs("data/ml", exist_ok=True)
        dataset.to_csv("data/ml/training_data.csv", index=False)
        print("Success! Prepared " + str(len(dataset)) + " rows of features.")
        print("Features created: [driver_momentum, track_history, season_points, driver_idx, track_idx]")
    else:
        print("Failed: Dataset is empty.")
