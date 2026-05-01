import sys
import os
import traceback

# Add project root to path
sys.path.append(os.getcwd())

try:
    from storage.postgres.database import SessionLocal
    from storage.postgres.models import RaceModel
    
    print("Testing database connection...")
    db = SessionLocal()
    
    print("Querying years...")
    years = db.query(RaceModel.year).distinct().all()
    print(f"Success! Found: {years}")
    
    db.close()
except Exception as e:
    print("\n--- ERROR DETECTED ---")
    traceback.print_exc()
    sys.exit(1)
