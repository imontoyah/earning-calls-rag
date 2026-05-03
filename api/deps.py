"""Cached singletons exposed as FastAPI dependencies."""

from functools import lru_cache

import chromadb
from langchain_groq import ChatGroq

from src.embeddings import get_client, get_or_create_collection
from src.rag import make_llm


@lru_cache
def get_collection() -> chromadb.Collection:
    """Single ChromaDB collection reused across requests.

    Uses get_or_create so the API boots even before any ingestion has run.
    """
    return get_or_create_collection(get_client(), reset=False)


@lru_cache
def get_llm() -> ChatGroq:
    """Single Groq client reused across requests."""
    return make_llm()
