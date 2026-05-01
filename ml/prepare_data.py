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
    WHERE r.year IN (2024, 2025, 2026)
    ORDER BY r.year, r.round_num
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("Warning: No data found in database.")
        return df

    # --- 2026 Regulation Weighting ---
    df['sample_weight'] = df['year'].apply(lambda x: 3.0 if x == 2026 else (1.5 if x == 2025 else 1.0))

    # --- Feature Engineering ---
    df['driver_momentum'] = df.groupby('driver_id')['position'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['track_history'] = df.groupby(['driver_id', 'race_name'])['position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['season_points'] = df.groupby(['driver_id', 'year'])['points'].transform(
        lambda x: x.cumsum().shift(1)
    )

    # 4. 2026 Regulation Impact Factor
    df['reg_impact'] = df['year'].apply(lambda x: 1.0 if x == 2026 else 0.0)
    
    # Fill NaNs
    df['driver_momentum'] = df['driver_momentum'].fillna(12.0)
    df['track_history'] = df['track_history'].fillna(12.0)
    df['season_points'] = df['season_points'].fillna(0.0)
    
    # Encodings
    df['driver_idx'] = df['driver_id'].astype('category').cat.codes
    df['track_idx'] = df['race_name'].astype('category').cat.codes
    
    return df

if __name__ == "__main__":
    print("Extracting Weighted F1 data (2024-2026)...")
    dataset = fetch_training_data()
    
    if not dataset.empty:
        os.makedirs("data/ml", exist_ok=True)
        dataset.to_csv("data/ml/training_data.csv", index=False)
        print(f"Success! Prepared {len(dataset)} rows with 2026 priority.")
        print("Features: [driver_momentum, track_history, season_points, reg_impact, sample_weight]")
    else:
        print("Failed: Dataset is empty.")
