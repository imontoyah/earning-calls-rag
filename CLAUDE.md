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
uv run jupyter notebook          # run all notebooks
uv run python -c "from src.ingestion import process_transcript; ..."  # test ingestion module
```

## Architecture

**Data flow:**
```
Motley Fool URL → download_transcript() → extract_article_text() → parse_speaker_turns()
→ structured JSON (company, quarter, date, turns[]) → ChromaDB (embedded with all-MiniLM-L6-v2)
→ LangChain retrieval → Groq Llama 3.3 70B → answer
```

**Core module** — `src/ingestion.py`:
- `process_transcript(url, company, quarter, date)` — full pipeline returning structured dict
- `save_transcript(transcript, output_dir)` — persists to `data/processed/<company>_<quarter>.json`
- Each "chunk" = one speaker turn, with metadata: `company`, `quarter`, `speaker`, `role`

**ChromaDB persistence:** `chroma_db/` (gitignored). Re-run notebook 02 to rebuild from `data/processed/`.

**Retrieval strategies (implemented in notebooks, to be refactored into `src/retrieval.py` in Phase 3):**
1. Semantic search with manual metadata filters
2. `SelfQueryRetriever` — LLM auto-extracts filters from natural language
3. Per-quarter retrieval for temporal comparisons
4. Hybrid search — BM25 + semantic (EnsembleRetriever)

## Project Phases

- **Phase 1 & 2** — Complete. Notebooks 01–04 cover single-doc and multi-doc RAG.
- **Phase 3** — In progress. Refactor notebooks into `src/` modules (`config.py`, `embeddings.py`, `retrieval.py`, `rag.py`), add FastAPI backend.
- **Phase 4** — Planned. Streamlit UI on top of the API.

The `PLAN.md` file has the full roadmap with step-by-step tasks for Phases 3 and 4.

## Data

- `data/processed/` — transcript JSONs, one per company/quarter, gitignored
- `chroma_db/` — ChromaDB vector store, gitignored
- Transcripts sourced from Motley Fool earnings call pages
