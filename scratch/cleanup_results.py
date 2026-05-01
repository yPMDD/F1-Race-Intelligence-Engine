import os
import sys
sys.path.append(os.getcwd())
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from storage.postgres.models import ResultModel, Base
from storage.postgres.database import DATABASE_URL

def cleanup():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Find duplicates
        duplicates = session.query(
            ResultModel.race_id, 
            ResultModel.driver_id, 
            func.count(ResultModel.id)
        ).group_by(ResultModel.race_id, ResultModel.driver_id).having(func.count(ResultModel.id) > 1).all()
        
        print(f"Found {len(duplicates)} duplicate driver-race pairs.")
        
        for race_id, driver_id, count in duplicates:
            # Keep one, delete the rest
            all_records = session.query(ResultModel).filter_by(race_id=race_id, driver_id=driver_id).all()
            for record in all_records[1:]:
                session.delete(record)
        
        session.commit()
        print("Cleanup successful.")
    except Exception as e:
        print(f"Cleanup failed: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    cleanup()
