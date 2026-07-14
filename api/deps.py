"""Cached singletons and shared dependencies exposed to FastAPI."""

import hmac
import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

import chromadb
from fastapi import Header, HTTPException, status
from langchain_groq import ChatGroq

from src.embeddings import get_client, get_or_create_collection
from src.rag import make_llm

log = logging.getLogger(__name__)

ALLOWED_INGEST_HOSTS = {"www.fool.com", "fool.com"}


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
        # Don't leak misconfiguration details to the caller — log it server-side
        # and return a generic 503 so external scanners can't fingerprint us.
        log.error("API_KEY env var is not set; rejecting authenticated request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def validate_ingest_url(url: str) -> None:
    """Reject URLs that could trigger SSRF when fetched server-side.

    Policy: https only, hostname must be in ALLOWED_INGEST_HOSTS (fool.com).
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must use https://",
        )
    if parsed.hostname not in ALLOWED_INGEST_HOSTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL host not allowed",
        )
