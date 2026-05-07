# Earnings Call RAG

RAG pipeline to query and analyze public earnings call transcripts using LangChain, ChromaDB, and Groq.

Ask natural language questions like:
- *"What did the CFO say about gross margins in Q4 2025?"*
- *"How has revenue changed across the last 3 quarters?"*
- *"Compare Apple's services business with Microsoft's cloud business"*

## How it works

```
Transcript (HTML) → Parse by speaker → Embeddings → ChromaDB → Retrieval → LLM → Answer
```

1. **Ingestion**: Downloads earnings call transcripts from Motley Fool, parses them into speaker turns (who said what), and stores as structured JSON.
2. **Embeddings**: Each speaker turn becomes a chunk with metadata (company, quarter, speaker, role). Embedded with `all-MiniLM-L6-v2` and stored in ChromaDB.
3. **Retrieval**: Three strategies available:
   - Semantic search with metadata filtering
   - SelfQueryRetriever (LLM auto-extracts filters from natural language)
   - Per-quarter retrieval for temporal comparisons
4. **Generation**: Groq (Llama 3.3 70B) generates answers grounded in the retrieved context.

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Groq API (Llama 3.3 70B) |
| Package Manager | uv |
| Interface | Jupyter Notebooks |

## Project Structure

```
├── notebooks/
│   ├── 01_data_ingestion.ipynb          # Download & parse transcripts
│   ├── 02_chunking_embeddings.ipynb     # Embeddings & ChromaDB storage
│   ├── 03_retrieval.ipynb               # Semantic search & metadata filtering
│   ├── 04_rag_pipeline.ipynb            # Full RAG with Groq LLM
│   ├── 05_multi_document_ingestion.ipynb # Multi-company, multi-quarter ingestion
│   ├── 06_self_query_retriever.ipynb    # LLM-powered automatic filtering
│   └── 07_temporal_comparison.ipynb     # Cross-quarter trend analysis
├── src/
│   └── ingestion.py                     # Reusable transcript parsing module
├── data/                                # Downloaded & processed transcripts (gitignored)
├── chroma_db/                           # Vector store persistence (gitignored)
├── pyproject.toml
├── .env.example
└── PLAN.md
```

## Setup

```bash
# Clone
git clone git@github.com:imontoyah/earning-calls-rag.git
cd earning-calls-rag

# Install dependencies
uv sync

# Configure API key
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)

# Run notebooks
uv run jupyter notebook
```

## Notebooks

The notebooks are designed to be followed in order. Each one builds on the previous.

| # | Notebook | What you learn |
|---|---|---|
| 01 | Data Ingestion | Web scraping, regex parsing, semi-structured text |
| 02 | Chunking & Embeddings | What embeddings are, cosine similarity, ChromaDB |
| 03 | Retrieval | Semantic search, metadata filtering, compound filters |
| 04 | RAG Pipeline | Prompt engineering, grounding, RAG vs. LLM-only |
| 05 | Multi-Document | Reusable modules, batch indexing, cross-document queries |
| 06 | Self-Query Retriever | LLM auto-extracts metadata filters from questions |
| 07 | Temporal Comparison | Per-quarter retrieval, trend analysis prompts |

## Roadmap

- [x] **Phase 1**: Single-document RAG pipeline
- [x] **Phase 2**: Multi-document, self-query, temporal comparisons
- [ ] **Phase 3**: Migration to AWS (S3, Bedrock, OpenSearch)
- [ ] **Phase 4**: Streamlit UI, more companies, evaluation metrics


## Endpoints usage
http://localhost:8080/docs
