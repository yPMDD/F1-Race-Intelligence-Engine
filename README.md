# 🏎️ F1 Race Intelligence Engine (2026 Season)

A production-grade, real-time AI strategist and predictive engine for Formula 1. This system ingests live telemetry, 2026 FIA regulations, and historical race data to provide real-time strategic insights via a LangGraph multi-agent pipeline and an XGBoost ML forecasting model.

## 🚀 Tech Stack
- **Frontend:** React, Vite, TailwindCSS, Server-Sent Events (SSE)
- **Backend API:** FastAPI, Uvicorn, Python
- **AI & Orchestration:** LangGraph, LangChain, Groq Cloud API (Llama 3.1 8B)
- **Data Engineering & MLOps:** Hybrid ETL Pipeline, DVC (Data Version Control)
- **Databases:** PostgreSQL (Relational), ChromaDB (Vector), SQLite (Semantic Cache)
- **Machine Learning:** XGBoost (Trained on 3 seasons of telemetry, 2026 weighted)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`

---

## 🏗️ Architecture & Workflow

![F1 Architecture Workflow](Gemini_Generated_Image_f5p7w0f5p7w0f5p7.png)

The project is divided into three core operational phases:

### 1. Ingestion & MLOps Phase
Raw F1 telemetry, lap times, and official FIA regulation PDFs are ingested through a Hybrid ETL Pipeline. This pipeline automatically triggers post-race. **DVC (Data Version Control)** is used to snapshot the dataset state, ensuring perfect ML reproducibility and auditability. The clean data is loaded into **PostgreSQL** (for structured data) and **ChromaDB** (for vector embeddings of unstructured text).

### 2. ML Forecasting Phase
An **XGBoost Regressor** is trained on 3 full seasons of historical F1 telemetry. For the 2026 season, the model weights are adjusted to account for the new regulation changes (e.g., -30kg weight, active aero). During race weekends, this model instantly generates a full 22-driver grid prediction order at sub-100ms latency.

### 3. RAG & Agentic Reasoning Core
User queries from the React dashboard hit a **Semantic Cache (SQLite)** first. If the query is a cache hit, the response streams back instantly (<1.0s). 
If it's a cache miss, the query is routed through a **LangGraph Multi-Agent Router**:
- **Factual Path:** Short-circuits directly to the hybrid retriever to pull static 2026 regulations without wasting agentic loops.
- **Strategic Path:** Enters a full agent loop powered by **Llama 3.1 8B (via Groq LPU)**. The agent intelligently selects tools to query telemetry, run ML predictions, or read rulebooks before streaming the final strategic analysis back to the dashboard via true Server-Sent Events (SSE).

---

## 🛠️ How to Run Locally

### Prerequisites
- Python 3.11+ (using `uv` package manager)
- Node.js & npm
- PostgreSQL running locally
- Groq API Key (added to your `.env` file as `GROQ_API_KEY`)

### Startup
We have unified the startup sequence into a single PowerShell script that boots the FastAPI backend and the React dashboard simultaneously.

```powershell
.\start.ps1
```

1. **Dashboard:** `http://localhost:5173` (or `5174/5175` depending on port availability)
2. **Backend API Docs:** `http://localhost:8000/docs`

*Note: The local Ollama LLM requirement has been fully deprecated in favor of the Groq Cloud API to enable zero-GPU, 100% free cloud deployment.*

---

## 📂 Project Structure
- `/apps/api/` - FastAPI backend, SSE streaming, router logic, and health polling.
- `/apps/dashboard/` - React frontend UI, Chat components, prediction layout.
- `/nlp/agents/` - LangGraph state definitions, agent orchestration, and custom tool binding.
- `/ml/` - XGBoost prediction scripts, data prep, and DVC ML artifact management.
- `/storage/` - Vector DB setup and SQL schemas.
