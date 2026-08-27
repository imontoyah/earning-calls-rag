# Earnings Call RAG

A production-ready RAG service that turns public company earnings call transcripts into a
question-answering API. It scrapes and parses transcripts by speaker, indexes them in ChromaDB
with rich metadata (company, quarter, speaker, role), and answers natural language questions
with Groq's Llama 3.3 70B — grounded in the retrieved passages, including comparisons across
quarters. Exposed as a secured FastAPI service and deployed on Fly.io.

Ask questions like:
- *"What did the CFO say about gross margins in Q4 2025?"*
- *"How has revenue changed across the last 3 quarters?"*
- *"Compare Apple's services business with Microsoft's cloud business"*

Live API: https://earning-calls-rag.fly.dev — docs at `/docs`.

## How it works

```
Transcript (HTML) → Parse by speaker → Embeddings → ChromaDB → Retrieval → LLM → Answer
```

1. **Ingestion**: Downloads transcripts from Motley Fool, parses them into speaker turns (who said what), and stores them as structured JSON.
2. **Embeddings**: Each speaker turn becomes a chunk with metadata (company, quarter, speaker, role), embedded with `all-MiniLM-L6-v2` and stored in ChromaDB.
3. **Retrieval**: Four strategies — semantic search with metadata filters, `SelfQueryRetriever` (LLM extracts the filters from the question), per-quarter retrieval for temporal comparisons, and hybrid BM25 + semantic search.
4. **Generation**: Groq (Llama 3.3 70B) answers grounded in the retrieved context.

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Groq API (Llama 3.3 70B) |
| API | FastAPI + slowapi (rate limiting) |
| Deployment | Docker + Fly.io |
| Package Manager | uv |

## API

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Health check |
| GET | `/collections` | — | Collection stats |
| POST | `/ask` | `X-API-Key` | Ask a question over the indexed transcripts |
| POST | `/ask/temporal` | `X-API-Key` | Compare an answer across quarters |
| POST | `/ingest` | `X-API-Key` | Ingest and index a new transcript URL |

POST endpoints require an `X-API-Key` header and are rate limited per IP.

## Project Structure

```
├── api/                # FastAPI app (routes, deps, schemas)
├── src/                # config, ingestion, embeddings, retrieval, rag
├── scripts/            # ingest.py, evaluate.py
├── evals/              # eval cases for retrieval + LLM-as-judge scoring
├── tests/              # pytest suite (no network, <1s)
├── notebooks/          # 01–04, the exploratory walkthrough
├── data/               # transcripts, raw & processed (gitignored)
├── chroma_db/          # vector store persistence (gitignored)
├── Dockerfile, fly.toml
└── PLAN.md
```

## Setup

```bash
git clone git@github.com:imontoyah/earning-calls-rag.git
cd earning-calls-rag

uv sync

cp .env.example .env
# GROQ_API_KEY — free at https://console.groq.com
# API_KEY      — python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Usage

```bash
uv run python scripts/ingest.py            # ingest new transcripts
uv run python scripts/ingest.py --reset    # wipe ChromaDB and re-index all
uv run fastapi dev api/main.py             # API at http://127.0.0.1:8000/docs
uv run pytest tests/                       # test suite
uv run python scripts/evaluate.py          # retrieval + answer-quality eval
uv run jupyter notebook                    # explore the notebooks
```

## Roadmap

- [x] **Phase 1 & 2**: Single- and multi-document RAG, self-query, temporal comparisons
- [x] **Phase 3**: Production code (modules, FastAPI, tests, evals)
- [x] **Phase 4**: Docker + Fly.io deployment
- [ ] **Phase 5**: Security (auth, rate limiting, SSRF guard done; per-role keys pending)
- [ ] **Phase 6**: UX (Streamlit UI, DELETE endpoint, more companies)
- [ ] **Phase 7**: Reliability (CI, logging, Sentry, monitoring)

See `PLAN.md` for the detailed roadmap.
