#!/usr/bin/env bash
# Container entrypoint for the FastAPI backend.
#
# Runs schema migrations before serving, so the Space is self-healing if a
# deploy's migration job was skipped or failed. CI already migrates Neon
# *before* redeploying the Space; this is the second line of defence, and the
# ordering it guarantees is the safe one -- "old image, new schema" degrades
# gracefully, whereas "new image, old schema" breaks at query time.
set -euo pipefail

echo "==> F1 Race Intelligence Engine backend"

# --- required configuration -------------------------------------------------
missing=()
[ -z "${GROQ_API_KEY:-}" ] && missing+=("GROQ_API_KEY")
[ -z "${DATABASE_URL:-}" ] && missing+=("DATABASE_URL")
if [ "${#missing[@]}" -gt 0 ]; then
  echo "FATAL: missing required environment variables: ${missing[*]}" >&2
  echo "Set these as Space settings (Variables and secrets) or container env." >&2
  exit 1
fi

# Local defaults so a minimal Space still boots.
export GROQ_MODEL="${GROQ_MODEL:-qwen/qwen3.8-27b}"
export GROQ_MAX_TOKENS="${GROQ_MAX_TOKENS:-800}"
export HF_HOME="${HF_HOME:-/app/.cache/huggingface}"
mkdir -p "$HF_HOME" /app/data/cache

# --- wait for the database --------------------------------------------------
# Neon may still be resuming the compute, or a cold start may briefly fail DNS.
echo "==> waiting for database"
for i in $(seq 1 30); do
  if uv run python -c "
from sqlalchemy import create_engine, text
import os
e = create_engine(os.environ['DATABASE_URL'], pool_pre_ping=True)
with e.connect() as c:
    c.execute(text('SELECT 1'))
" 2>/dev/null; then
    echo "    database reachable (attempt $i)"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "FATAL: database unreachable after 30 attempts" >&2
    exit 1
  fi
  sleep 5
done

# --- migrations -------------------------------------------------------------
echo "==> applying migrations"
if uv run alembic upgrade head; then
  echo "    schema up to date"
else
  echo "FATAL: migrations failed. Refusing to start against an unknown schema." >&2
  exit 1
fi

# --- serve ------------------------------------------------------------------
PORT="${PORT:-7860}"
echo "==> serving on 0.0.0.0:${PORT}"
exec uv run uvicorn apps.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --timeout-keep-alive 75
