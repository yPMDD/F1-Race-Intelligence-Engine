# Must match requires-python in pyproject.toml (>=3.11,<3.13).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/.cache/huggingface \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Build deps for psycopg2 / numpy / xgboost wheels, then drop them again.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Dependencies first so the layer caches across code edits.
COPY pyproject.toml uv.lock ./
COPY scripts/docker_install_deps.sh /usr/local/bin/docker_install_deps.sh
RUN chmod +x /usr/local/bin/docker_install_deps.sh

# CPU-only torch: 3.59 GB -> 766 MB. The script documents why each naive
# alternative (UV_TORCH_BACKEND, --no-install-package, [tool.uv.sources])
# silently fails.
RUN docker_install_deps.sh --no-dev --no-install-project

COPY . /app

# `uv sync` alone is not enough -- the app imports the training artifact
# (storage/models/f1_predictor.joblib) and the vector store (data/chroma), and
# neither is a Python package.
RUN docker_install_deps.sh --no-dev

# data/chroma/ is gitignored (it is a ~90 MB regenerable HNSW index), so a
# fresh `git clone` -- e.g. the Hugging Face Spaces build -- arrives without a
# vector store and every regulation query would return "no data found".
# Rebuild it from the tracked PDFs only when it is actually missing.
ARG BUILD_VECTORSTORE=true
RUN if [ "$BUILD_VECTORSTORE" = "true" ] && [ ! -f data/chroma/chroma.sqlite3 ]; then \
        echo ">> data/chroma missing - ingesting data/raw into ChromaDB..." \
        && /app/.venv/bin/python pipelines/etl/document_processor.py \
    else \
        echo ">> data/chroma present - skipping ingestion." ; \
    fi

# Hugging Face Spaces serves on port 7860.
EXPOSE 7860

# Provide safe defaults so a minimal Space still boots. Real values come from
# Space "Variables and secrets" (or container env): GROQ_API_KEY, DATABASE_URL
# (Neon pooled endpoint) and CORS_ORIGINS.
ENV GROQ_MODEL=qwen/qwen3.8-27b \
    GROQ_MAX_TOKENS=800 \
    PORT=7860

# The entrypoint waits for the database, runs `alembic upgrade head`, then
# serves. It refuses to start if the schema cannot be brought up to date.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
