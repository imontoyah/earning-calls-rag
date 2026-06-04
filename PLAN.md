# Earnings Call RAG — Project Plan

## Context
RAG pipeline to query and analyze public earnings call transcripts. Started as a learning project, evolved into a production-ready service deployed to Fly.io with an API. Next phases focus on hardening the live service and adding a UI.

## Tech Stack
- **Python 3.11+** | **uv** (package manager)
- **LangChain** — RAG orchestration
- **ChromaDB** — vector store
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **Groq API** (Llama 3.3 70B) — LLM
- **FastAPI** — REST API
- **Docker** — containerization
- **Fly.io** — deployment (microVMs + persistent volumes)
- **pytest** — automated tests
- **Streamlit** — UI (Phase 6, planned)

## Project Structure
```
├── src/
│   ├── config.py          # Centralized settings (STATE_DIR env-var aware)
│   ├── ingestion.py       # Download & parse transcripts
│   ├── embeddings.py      # ChromaDB storage & management
│   ├── retrieval.py       # Semantic, self-query, temporal, hybrid
│   └── rag.py             # RAG chain (prompt + LLM)
├── api/
│   ├── main.py            # FastAPI app + routes
│   ├── deps.py            # Cached dependencies (collection, LLM)
│   └── schemas.py         # Pydantic request/response models
├── scripts/
│   ├── ingest.py          # CLI to ingest new transcripts
│   └── evaluate.py        # Run evaluation test suite
├── tests/                 # pytest suite (API, parsing, filters)
├── evals/cases.json       # Eval test cases
├── app/                   # (planned) Streamlit UI
├── notebooks/             # Reference / exploration
├── data/                  # Transcripts + companies.json (gitignored)
├── chroma_db/             # Vector store (gitignored)
├── Dockerfile             # Container build recipe
├── .dockerignore          # Build context exclusions
├── fly.toml               # Fly.io deployment config
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

## PHASE 3: "Productionize the backend" [COMPLETED]
**Goal:** Move from notebooks to a production-ready Python backend with API.

- [x] Step 3.1 — Refactor: notebooks → Python modules (`config`, `ingestion`, `embeddings`, `retrieval`, `rag`)
- [x] Step 3.2 — Better chunking (sub-chunks with overlap for long turns)
- [x] Step 3.3 — Multi-company support (`data/companies.json`)
- [x] Step 3.4 — Auto-ingestion CLI script (`scripts/ingest.py`)
- [x] Step 3.5 — FastAPI backend (`/health`, `/collections`, `/ask`, `/ask/temporal`, `/ingest`)
- [x] Step 3.6 — Evaluation script (`scripts/evaluate.py`) + pytest suite

---

## PHASE 4: "Containerize & Deploy" [COMPLETED]
**Goal:** Take the local app to a publicly-accessible URL.

- [x] Step 4.1 — Dockerfile (non-root user, uv from official image, pre-downloaded embedding model)
- [x] Step 4.2 — `.dockerignore` (exclude `.venv`, data, tests, notebooks)
- [x] Step 4.3 — `STATE_DIR` env-var indirection in `src/config.py` (single-volume compatible)
- [x] Step 4.4 — Fly.io deploy (1GB RAM, single persistent volume, auto-stop, HTTPS)
- [x] Step 4.5 — `GROQ_API_KEY` managed via `fly secrets`

Live at: https://earning-calls-rag.fly.dev

---

## PHASE 5: "Security"
**Goal:** Prevent abuse of the public endpoints — auth + rate limiting.

### [x] Step 5.1 — API key auth on mutating + query endpoints
- `X-API-Key` header required for `POST /ingest`, `POST /ask`, `POST /ask/temporal`
- Validated against `API_KEY` env var (`fly secrets set API_KEY=...` in production)
- Constant-time comparison via `hmac.compare_digest`
- Wired via `Depends(verify_api_key)` in `api/deps.py`
- 6 new tests cover missing key, wrong key, and verify GETs stay public

### [ ] Step 5.2 — Per-role keys (optional refinement)
- Split into `API_ADMIN_KEY` (write: `/ingest`) and `API_READ_KEY` (read: `/ask`)
- Useful when sharing access with someone who shouldn't mutate
- Skip until there's a real need — adds complexity without immediate payoff

### [x] Step 5.3 — Rate limiting
- `slowapi` with per-IP key (`get_remote_address`)
- Limits: `/ask` 10/min, `/ask/temporal` 5/min, `/ingest` 5/hour
- Limiter on `app.state.limiter`; `RateLimitExceeded` returns 429
- Disabled in tests by default; `enabled_limiter` fixture flips it on for the threshold test
- In-memory storage (resets on container restart — fine for single-machine Fly)

### [x] Step 5.4 — Hardening review
- `validate_ingest_url()` in `api/deps.py`: https-only + hostname allowlist (`ALLOWED_INGEST_HOSTS = {fool.com, www.fool.com}`). Closes SSRF on `/ingest` with the minimum code surface; add hostnames here if new sources are needed
- `verify_api_key`: missing `API_KEY` now returns 503 (was 500 with the literal misconfig message) — logged server-side
- CORS: left at FastAPI default (no headers) — restrictive, fine until Phase 6.1 Streamlit UI needs it
- Tests cover unsafe URLs: non-https, non-allowlisted hosts, suffix-spoofing (`fool.com.evil.com`), private IPs, AWS metadata

---

## PHASE 6: "UX"
**Goal:** Make the system usable by humans, not just curl.

### [ ] Step 6.1 — Streamlit UI (MVP)
- Text input for questions
- Dropdown filters for company / quarter (from `/collections`)
- Results display with citations (speaker + quarter)
- Single file at `app/streamlit_app.py`, calls the deployed API

### [ ] Step 6.2 — Complete CRUD: `DELETE /transcripts/{ticker}/{quarter}`
- Delete chunks from ChromaDB by metadata filter
- Delete the JSON file from `data/processed/`
- Returns 204 No Content; idempotent (404 if nothing to delete)
- Requires the admin auth from Step 5.1

### [ ] Step 6.3 — Multi-company data (real diverse coverage)
- Ingest at least 3 companies (e.g. AAPL, MSFT, NVDA) across multiple quarters
- Showcase cross-company comparison queries
- Update `evals/cases.json` with multi-company cases

### [ ] Step 6.4 — Source citations in `/ask` response
- Return retrieved chunks alongside the answer (small refactor to `src/rag.py`)
- UI displays them as "evidence"
- Adds significant value for non-technical users (they can verify the answer)

### [ ] Step 6.5 — (Optional) Frontend upgrade
- If Streamlit feels limiting, swap to a SPA (Next.js / React)
- Deployed as a separate static site (Fly static, Vercel, etc.)

---

## PHASE 7: "Reliability"
**Goal:** Operate the service confidently — automate tests, see failures early.

### [ ] Step 7.1 — GitHub Actions CI
- On every PR: run `pytest`, build the Docker image
- On push to `develop`: optional auto-deploy to Fly (`fly deploy --remote-only`)
- Required check before merge

### [ ] Step 7.2 — Structured logging
- Replace uvicorn defaults with JSON logs (timestamp, level, request_id, etc.)
- Include request_id in every log line so traces are correlatable
- Compatible with log aggregators (Better Stack, Logtail, Datadog)

### [ ] Step 7.3 — Error tracking (Sentry)
- Wire Sentry SDK into FastAPI
- Capture unhandled exceptions automatically, with stack traces
- Use the free tier (5K events/month is plenty for this scale)

### [ ] Step 7.4 — Uptime / health monitoring
- External pinger hitting `/health` every minute (UptimeRobot, Better Stack, free tiers)
- Email/Slack alert on downtime
- Public status page (optional)

### [ ] Step 7.5 — Backups of the volume
- Fly already does daily snapshots (5-day retention) — verify and document
- Add a `scripts/restore.sh` for recovery drill
- Test the recovery path once (delete + restore from snapshot)

### [ ] Step 7.6 — Performance baseline
- Run the eval script periodically, track scores over time
- Decide on a "fail" threshold (e.g. avg faithfulness < 4.0)
- Optionally automate via CI
