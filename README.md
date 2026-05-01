# F1 Race Intelligence Engine

**A modular data ingestion, ML forecasting, and NLP retrieval platform designed for real-time Formula 1 analytics.**

This system normalizes structured session telemetry and unstructured technical documents into a canonical data schema. It serves as the backbone for predictive modeling, retrieval-augmented generation (RAG), and agentic workflows, exposing all data through highly decoupled API boundaries.

## Architecture

The system is compartmentalized into discrete microservices interacting via REST interfaces, shared databases, and cache layers.

- **Ingestion Layer:** A hybrid pipeline combining structured batch processing (via FastF1 API) for telemetry/timing data and unstructured scraping for FIA regulations and stewards' reports.
- **RAG Subsystem:** Processes and embeds unstructured racing documents into a vector store. Implements chunking, metadata attachment, and reranking to ground LLM-generated strategy answers.
- **Agentic Layer:** Utilizes LangGraph to orchestrate specialized agents (Strategy, Analytics, Rules). Agents maintain state and execute controlled tool handoffs to reason over race data and document indices.
- **Prediction Engine:** Computes engineered features (e.g., pace advantage, tire degradation slope) to output deterministic forecasts like projected finishing positions and pit window optimizations.
- **Serving Layer:** A FastAPI-driven backend serving low-latency JSON endpoints to an F1-themed React/Vite dashboard.

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, SQLAlchemy
- **Data & Storage:** PostgreSQL, Redis, Parquet, Vector DB (e.g., pgvector/Qdrant)
- **ML & Orchestration:** Scikit-learn, LangGraph, DVC
- **Frontend:** React, Vite
- **Infrastructure:** Docker, Docker Compose, `uv`

## Quickstart

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- `uv` package manager
- Node.js & npm

### Deployment

1. **Spin up Infrastructure (PostgreSQL & Redis):**
   ```bash
   docker-compose up -d
   ```

2. **Initialize Environment and Dependencies:**
   ```bash
   uv sync
   ```

3. **Execute Data Ingestion Pipeline:**
   Populate the database with canonical telemetry and fetch unstructured document stubs.
   ```bash
   uv run apps/worker/ingestion_worker.py
   ```

4. **Start the API Backend:**
   ```bash
   uv run uvicorn apps.api.main:app --reload
   ```

5. **Start the Dashboard:**
   ```bash
   cd apps/dashboard
   npm install
   npm run dev
   ```

The API will be available at `http://localhost:8000/docs` and the Dashboard at `http://localhost:5173`.

## Design Principles

- **Data Reproducibility:** Data transformations and ML artifacts are strictly tracked using Data Version Control (DVC). Features are computed deterministically.
- **Service Decoupling:** Data layers (PostgreSQL/Parquet) and computation layers communicate solely through explicit contracts (Pydantic schemas) to prevent tight coupling.
- **Latency Optimization:** Heavy queries and repeated inference endpoints are cached natively via Redis. Background worker processes handle blocking I/O (e.g., embedding generation and ingestion operations).
