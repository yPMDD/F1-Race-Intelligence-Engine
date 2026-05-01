import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os

def train():
    if not os.path.exists("data/ml/training_data.csv"):
        print("Error: training_data.csv not found. Run prepare_data.py first.")
        return

    # Load data
    df = pd.read_csv("data/ml/training_data.csv")
    
    # Define features and target
    features = ['driver_momentum', 'track_history', 'season_points', 'driver_idx', 'track_idx', 'reg_impact']
    X = df[features]
    y = df['position']
    weights = df['sample_weight']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    # Initialize and train XGBoost
    model = XGBRegressor(
        n_estimators=300, 
        learning_rate=0.03, 
        max_depth=7, 
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Training Complete.")
    print(f"Mean Absolute Error: {mae:.2f} positions")
    
    # Save Model
    os.makedirs("storage/models", exist_ok=True)
    joblib.dump(model, "storage/models/f1_predictor.joblib")
    
    # Create unique mappings for inference
    # We take the latest mapping from the dataset
    driver_map = df.groupby('driver_id')['driver_idx'].first().to_dict()
    track_map = df.groupby('race_name')['track_idx'].first().to_dict()
    
    mappings = {
        "drivers": driver_map,
        "tracks": track_map
    }
    
    with open("storage/models/mappings.json", "w") as f:
        json.dump(mappings, f, indent=4)
    
    print("Model and mappings saved to storage/models/")

if __name__ == "__main__":
    train()
