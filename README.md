---
title: F1 Race Intelligence Engine API
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
# Cold start downloads the MiniLM embedding model and may rebuild the Chroma
# index from data/raw if it is absent. The 30 min default is tight for that.
startup_duration_timeout: 20m
---

# 🏎️ F1 Race Intelligence Engine (2026 Season)

A production-grade, real-time AI strategist and predictive engine for Formula 1. This system ingests live telemetry, 2026 FIA regulations, and historical race data to provide real-time strategic insights via a LangGraph multi-agent pipeline and an XGBoost ML forecasting model.

## 🚀 Tech Stack
- **Frontend:** React 19, Vite, TailwindCSS v4, Server-Sent Events (SSE)
- **Backend API:** FastAPI, Uvicorn, Python 3.11
- **AI & Orchestration:** LangGraph, LangChain, Groq Cloud API (`qwen/qwen3.8-27b`)
- **Databases:** PostgreSQL (Relational), ChromaDB (Vector), SQLite (Semantic Cache)
- **Machine Learning:** XGBoost (Trained on 3 seasons of telemetry, 2026 weighted)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`

> **Model note:** the agent runs on `qwen/qwen3.8-27b`. The previous
> `llama-3.1-8b-instant` is still listed by Groq but is gated to Enterprise
> ("Contact Sales") rate limits and is not visible to Developer-plan API keys.
> Override with `GROQ_MODEL`.

---

## 🏗️ Architecture & Workflow

![F1 Architecture Workflow](Gemini_Generated_Image_f5p7w0f5p7w0f5p7.png)

The project is divided into three core operational phases:

### 1. Ingestion & MLOps Phase
Raw F1 telemetry and official FIA regulation PDFs are ingested through a Hybrid ETL Pipeline. The clean data is loaded into **PostgreSQL** (structured) and **ChromaDB** (vector embeddings of the regulation text). `data/chroma/` is gitignored as a regenerable artifact — `pipelines/etl/document_processor.py` rebuilds it from the tracked PDFs in `data/raw/`.

### 2. ML Forecasting Phase
An **XGBoost Regressor** is trained on 3 full seasons of historical F1 telemetry. For the 2026 season, sample weights are scaled (3.0× for 2026, 1.5× for 2025) to account for the new regulation changes. At inference the model resolves the circuit, pulls per-driver recent form and per-circuit history from Postgres, and generates a full 22-driver grid prediction in a single batched pass.

### 3. RAG & Agentic Reasoning Core
User queries from the React dashboard hit a **Semantic Cache (SQLite)** first. On a hit, the response streams back instantly. On a miss, the query is routed through a **LangGraph Multi-Agent Router**:
- **Factual Path:** short-circuits straight to a hybrid retriever (vector similarity blended with a BM25-style lexical score) over the 2026 regulations.
- **Strategic Path:** a full agent loop powered by **Qwen3.8-27B via Groq LPU**. The agent selects tools to query telemetry, run ML predictions, or read the rulebook, streaming the answer back over true Server-Sent Events.

---

## 🛠️ How to Run Locally

### Prerequisites
- Python 3.11 or 3.12 (use [uv](https://docs.astral.sh/uv/))
- Node.js & npm
- PostgreSQL running locally
- A free [Groq API key](https://console.groq.com/keys) as `GROQ_API_KEY`

### Startup

```powershell
# one-time
python -m pip install uv
uv sync
copy .env.example .env      # then fill in GROQ_API_KEY
npm --prefix apps/dashboard install
uv run alembic upgrade head  # create the schema (not automatic any more)

# every time
.\start.ps1
```

1. **Dashboard:** `http://localhost:5173`
2. **Backend API Docs:** `http://localhost:8000/docs`

<details>
<summary>Manual start (instead of <code>start.ps1</code>)</summary>

```powershell
# terminal 1
uv run uvicorn apps.api.main:app --reload --port 8000

# terminal 2
npm --prefix apps/dashboard run dev
```
</details>

<details>
<summary>Using Docker for Postgres</summary>

```powershell
docker compose up -d      # serves on localhost:5433
```
Then set `DATABASE_URL=postgresql://f1user:f1password@localhost:5433/f1db`.
</details>

---

## 🧪 Tests

```powershell
uv run pytest                        # backend, 111 tests, hermetic (SQLite, no network)
npm --prefix apps/dashboard test     # frontend SSE parser, 14 tests
```

The backend suite redirects `DATABASE_URL` to SQLite and installs a socket
guard that fails any test attempting a real network call, so it runs offline
in a few seconds.

---

## 🚢 Deployment

`main` is production; `dev` is where changes land. CI runs on every push and PR,
and merging to `main` auto-deploys to **Neon** (Postgres), **Hugging Face Spaces**
(the FastAPI backend) and **Vercel** (the dashboard).

```
dev ──▶ PR ──▶ CI (tests · docker build · secret scan)
              └─ merge ──▶ main ──▶ Neon migrations ──▶ HF Space redeploy
                                                  └─▶ Vercel (own git integration)
```

Schema changes go through **Alembic** — `create_all()` never alters existing
tables, so it could not detect them:

```powershell
uv run alembic revision --autogenerate -m "add compound index"
uv run alembic upgrade head
```

**Full setup, secret names and troubleshooting: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).**

---

## 📂 Project Structure
- `apps/api/` - FastAPI backend: SSE streaming, router logic, health endpoint.
- `apps/dashboard/` - React frontend: chat pane, telemetry charts, prediction tables.
- `nlp/agents/` - LangGraph state, agent orchestration, tool bindings.
- `nlp/cache/` - SQLite semantic response cache.
- `ml/` - XGBoost training, feature prep, and the runtime predictor.
- `pipelines/etl/` - PDF ingestion into ChromaDB.
- `storage/` - SQLAlchemy models and the trained model artifact.
- `alembic/` - Database migrations.
- `tests/` - 111 backend tests.
