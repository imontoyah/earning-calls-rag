"""Cached singletons and shared dependencies exposed to FastAPI."""

import hmac
import os
from functools import lru_cache

import chromadb
from fastapi import Header, HTTPException, status
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


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject the request unless the X-API-Key header matches the configured secret.

    Uses hmac.compare_digest for constant-time comparison so timing analysis
    can't be used to discover the key character by character.
    """
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: API_KEY not set",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
