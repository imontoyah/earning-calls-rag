"""Centralized configuration for the earnings call RAG pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths. STATE_DIR holds runtime state (vector store + transcripts) and can
# be redirected via env var so a single mounted volume is enough in
# environments like Fly.io that allow only one volume per machine.
BASE_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(BASE_DIR)))
DATA_DIR = STATE_DIR / "data" / "processed"
CHROMA_DIR = STATE_DIR / "chroma_db"
COMPANIES_FILE = BASE_DIR / "data" / "companies.json"

# ChromaDB
COLLECTION_NAME = "earnings_calls"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# Retrieval defaults
DEFAULT_N_RESULTS = 5
N_RESULTS_PER_QUARTER = 2
HYBRID_WEIGHTS = [0.5, 0.5]  # [bm25_weight, semantic_weight]

# Chunking — applied when indexing speaker turns into ChromaDB
CHUNK_SIZE = 200     # max words per sub-chunk
CHUNK_OVERLAP = 30   # words shared between consecutive sub-chunks
