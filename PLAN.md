# Earnings Call RAG — Project Plan

## Context
RAG pipeline to query and analyze public earnings call transcripts. Started as a learning project, now evolving into a production-ready service with API and UI.

## Tech Stack
- **Python 3.11+** | **uv** (package manager)
- **LangChain** — RAG orchestration
- **ChromaDB** — vector store
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **Groq API** (Llama 3.3 70B) — LLM
- **FastAPI** — REST API (Phase 3)
- **Streamlit** — UI (Phase 4)

## Project Structure (target)
```
├── src/
│   ├── config.py          # Centralized settings
│   ├── ingestion.py       # Download & parse transcripts
│   ├── embeddings.py      # ChromaDB storage & management
│   ├── retrieval.py       # Semantic, self-query, temporal, hybrid
│   └── rag.py             # RAG chain (prompt + LLM)
├── scripts/
│   ├── ingest.py          # CLI to ingest new transcripts
│   └── evaluate.py        # Run evaluation test suite
├── api/
│   └── main.py            # FastAPI app
├── app/
│   └── streamlit_app.py   # Streamlit UI
├── notebooks/             # Reference/exploration
├── data/processed/        # Transcript JSONs (gitignored)
├── chroma_db/             # Vector store (gitignored)
└── pyproject.toml
```

---

## PHASE 1: "Hello RAG" [COMPLETED]
**Goal:** Basic RAG pipeline with a single transcript.

- [x] Step 1.1 — Project setup (uv, dependencies, Groq API key)
- [x] Step 1.2 — Data ingestion (download, parse, save as JSON)
- [x] Step 1.3 — Chunking + Embeddings (speaker turns → ChromaDB)
- [x] Step 1.4 — Retrieval (semantic search + metadata filtering)
- [x] Step 1.5 — Full RAG (LangChain + Groq end-to-end)

---

## PHASE 2: "Multi-document RAG" [COMPLETED]
**Goal:** Multiple transcripts, advanced retrieval, evaluation.

- [x] Step 2.1 — Multi-document ingestion (multiple quarters, reusable module)
- [x] Step 2.2 — SelfQueryRetriever (LLM auto-extracts filters)
- [x] Step 2.3 — Temporal comparison (per-quarter retrieval)
- [x] Step 2.4 — Hybrid search (BM25 + semantic) + basic evaluation

---

## PHASE 3: "Productionize the backend"
**Goal:** Move from notebooks to a production-ready Python backend with API.

### [x] Step 3.1 — Refactor: notebooks → Python modules
- `src/config.py` — centralized configuration (model names, paths, collection name)
- `src/embeddings.py` — ChromaDB connection, document storage, collection management
- `src/retrieval.py` — all 4 retrieval strategies
- `src/rag.py` — RAG chain (prompt + LLM + output parsing)
- `src/ingestion.py` — already exists, minor updates if needed

### [x] Step 3.2 — Better chunking
- Split long speaker turns into sub-chunks with overlap
- Preserve speaker/role metadata on each sub-chunk
- Improves embedding quality for long monologues

### [ ] Step 3.3 — Multi-company support
- Config file mapping company tickers to transcript URLs
- Ingestion script that processes all configured companies
- Easy to add new companies

### [ ] Step 3.4 — Auto-ingestion script
- CLI script: `python scripts/ingest.py`
- Reads config, downloads new transcripts, indexes into ChromaDB
- Can be run manually or scheduled (cron)

### [ ] Step 3.5 — FastAPI backend
- `POST /ask` — RAG query (question + optional filters)
- `POST /ask/temporal` — temporal comparison query
- `POST /ingest` — trigger ingestion of a new transcript
- `GET /collections` — list available companies/quarters
- `GET /health` — health check

### [ ] Step 3.6 — Evaluation script
- Move eval logic from notebook 04 to `scripts/evaluate.py`
- Run independently to benchmark after changes

---

## PHASE 4: "UI"
**Goal:** User interface on top of the API.

### [ ] Step 4.1 — Streamlit app (MVP)
- Text input for questions
- Dropdown filters for company/quarter
- Results display with citations

### [ ] Step 4.2 — (Optional) Frontend upgrade
- If Streamlit feels limiting, upgrade to React or similar
