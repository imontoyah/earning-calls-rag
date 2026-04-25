"""Centralized configuration for the earnings call RAG pipeline."""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"

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
