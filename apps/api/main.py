from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.api.routers import race, rag, agent
from storage.postgres.database import engine
from storage.postgres.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Race Intelligence Engine API",
    description="API for accessing F1 telemetry and intelligence data",
    version="0.1.0",
)

# Create DB tables
Base.metadata.create_all(bind=engine)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(race.router)
app.include_router(rag.router)
app.include_router(agent.router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
