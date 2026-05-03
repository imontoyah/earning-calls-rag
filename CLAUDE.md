# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv sync                  # install dependencies
cp .env.example .env     # add GROQ_API_KEY from console.groq.com
uv run jupyter notebook  # launch notebooks
```

Environment variable required: `GROQ_API_KEY`

## Development Commands

```bash
uv run jupyter notebook                          # explore notebooks interactively
uv run python scripts/ingest.py                  # ingest new transcripts only
uv run python scripts/ingest.py --reset          # wipe ChromaDB and re-index all
uv run python scripts/ingest.py --dry-run        # preview what would be processed
uv run python scripts/ingest.py --company AAPL   # limit to one ticker
uv run fastapi dev api/main.py                   # run API in dev mode (auto-reload)
uv run python -c "from src.rag import ask; ..."  # test RAG chain directly
```

API docs available at `http://127.0.0.1:8000/docs` once running.

## Architecture

**Data flow:**
```
data/companies.json → load_companies() → process_transcript() → save_transcript()
→ index_transcript() → ChromaDB (all-MiniLM-L6-v2 embeddings)
→ retrieval strategy → Groq Llama 3.3 70B → answer
```

**`src/config.py`** — single source of truth for all paths, model names, and tuning constants (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `DEFAULT_N_RESULTS`, `HYBRID_WEIGHTS`).

**`src/ingestion.py`** — scrapes Motley Fool HTML, parses into speaker turns, persists JSON. Multi-company entry point: `load_companies()` reads `data/companies.json` (schema: `{companies: [{ticker, transcripts: [{url, quarter, date}]}]}`).

**`src/embeddings.py`** — ChromaDB management and indexing.
- Two embedding wrappers: native chromadb (`_make_chroma_embedding_fn`) and LangChain-compatible (`_make_langchain_embedding_fn`) — the latter is required for `SelfQueryRetriever`.
- `index_from_config()` — one-shot: downloads, parses, saves, and indexes all companies from `companies.json`.
- `index_all()` — re-indexes from existing `data/processed/` JSONs (no download).
- Long speaker turns are split into overlapping word-count sub-chunks via `chunk_text()` before indexing. Chunk IDs: `<company>_<quarter>_<turn_i>_<chunk_j>`.

**`src/retrieval.py`** — four strategies:
1. `semantic_search()` — direct ChromaDB cosine similarity with optional `where` filter (ChromaDB syntax).
2. `make_self_query_retriever()` — LangChain `SelfQueryRetriever`; LLM auto-extracts metadata filters. Requires the LangChain `Chroma` vectorstore, not the raw chromadb client.
3. `retrieve_per_quarter()` / `build_temporal_context()` — per-quarter semantic search, formats context with `=== Q#-YYYY ===` headers.
4. `make_hybrid_retriever()` — `EnsembleRetriever` (BM25 + semantic). BM25 reads from `data/processed/` JSONs directly, so that directory must be populated first.

**`src/rag.py`** — LangChain chains.
- `ask(collection, question, where=...)` — standard semantic RAG.
- `ask_temporal(collection, question, quarters, company=...)` — temporal comparison RAG.
- Both return plain strings. LLM is created lazily if not passed in.

**`api/`** — FastAPI backend (Step 3.5).
- `main.py` — routes: `GET /health`, `GET /collections`, `POST /ask`, `POST /ask/temporal`, `POST /ingest`.
- `deps.py` — `@lru_cache`'d singletons for the ChromaDB collection and the Groq LLM client (avoids reconnecting per request).
- `schemas.py` — Pydantic request/response models with `Field` validation. `IngestRequest` validates `quarter` and `date` formats via regex.
- `/ingest` returns 201 on success, 409 if the transcript JSON already exists on disk (no upsert support).

**ChromaDB:** `chroma_db/` (gitignored). Rebuild by calling `index_from_config()` or `index_all()`.

## Project Phases

- **Phase 1 & 2** — Complete. Notebooks 01–04 cover single-doc and multi-doc RAG.
- **Phase 3** — Steps 3.1–3.5 done. Step 3.6 (eval script) remaining.
- **Phase 4** — Planned. Streamlit UI.

See `PLAN.md` for full step-by-step roadmap.

## Data

- `data/companies.json` — registry of companies and transcript URLs to ingest (gitignored).
- `data/processed/` — transcript JSONs, one per company/quarter (gitignored).
- `chroma_db/` — ChromaDB vector store (gitignored).
