# Must match requires-python in pyproject.toml (>=3.11,<3.13).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Build deps for psycopg2 / numpy / xgboost wheels, then drop them again.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Dependencies first so the layer caches across code edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY . /app

# NOTE: `uv sync` alone is not enough -- the app imports the training artifact
# (storage/models/f1_predictor.joblib) and the vector store (data/chroma), and
# neither is a Python package.
RUN uv sync --no-dev

# data/chroma/ is gitignored (it is a ~90 MB regenerable HNSW index), so a
# fresh `git clone` -- e.g. the Hugging Face Spaces build -- arrives without a
# vector store and every regulation query would return "no data found".
# Rebuild it from the tracked PDFs only when it is actually missing.
ARG BUILD_VECTORSTORE=true
RUN if [ "$BUILD_VECTORSTORE" = "true" ] && [ ! -f data/chroma/chroma.sqlite3 ]; then \
        echo ">> data/chroma missing - ingesting data/raw into ChromaDB..." \
        && uv run python pipelines/etl/document_processor.py \
    else \
        echo ">> data/chroma present - skipping ingestion." ; \
    fi

# Hugging Face Spaces serves on port 7860.
EXPOSE 7860

# Provide a default so the container boots even before a real key is injected.
# On Spaces, set GROQ_API_KEY / DATABASE_URL / CORS_ORIGINS as Space secrets.
ENV GROQ_MODEL=qwen/qwen3.8-27b

CMD ["uv", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
