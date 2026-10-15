from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import traceback
import logging

load_dotenv()

from apps.api.routers import race, rag, agent
from storage.postgres.database import engine
from storage.postgres.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Race Intelligence Engine API",
    version="0.2.0",
)

# Create DB tables
Base.metadata.create_all(bind=engine)


def _allowed_origins():
    """Resolve the CORS allow-list.

    The previous configuration was `allow_origins=["*"]` together with
    `allow_credentials=True`. That combination is rejected by browsers (a
    wildcard origin cannot be used with credentials) and, for an API with no
    authentication at all, it exposes every endpoint to any site on the
    internet -- including /agent/chat, which spends Groq tokens per call.

    Set CORS_ORIGINS to a comma-separated list in the environment. Default to
    the local Vite dev server; deployment sets its own origin.
    """
    raw = os.getenv("CORS_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    # No cookies/auth are used, so credentials stay off.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a generic 500 without leaking internals.

    The previous handler returned `traceback.format_exc()` in the response
    body, disclosing file paths, SQL, library versions and -- because the
    agents surface tool errors as strings -- potentially database contents to
    any caller. The full traceback is still logged server-side.
    """
    logger.error("Unhandled error on %s %s", request.method, request.url.path)
    logger.error("%s", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )


app.include_router(race.router)
app.include_router(rag.router)
app.include_router(agent.router)


@app.get("/health")
def health_check():
    return {"status": "healthy"}
