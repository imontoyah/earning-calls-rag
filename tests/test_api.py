"""Endpoint tests for the FastAPI app."""

from pathlib import Path

import pytest

from api.main import limiter
from tests.conftest import AUTH_HEADERS


def test_health_returns_ok_with_collection_count(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "collection_count": 3}


def test_collections_groups_quarters_per_ticker(client) -> None:
    response = client.get("/collections")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["companies"] == [
        {"ticker": "AAPL", "quarters": ["Q3-2025", "Q4-2025"]},
    ]


def test_ask_rejects_empty_question(client, monkeypatch) -> None:
    # Stub ask() so we don't hit Groq even if validation passes.
    monkeypatch.setattr("api.main.ask", lambda *a, **k: "stub")
    response = client.post("/ask", headers=AUTH_HEADERS, json={"question": ""})
    assert response.status_code == 422


def test_ask_returns_answer_from_rag(client, monkeypatch) -> None:
    monkeypatch.setattr("api.main.ask", lambda *a, **k: "stubbed answer")
    response = client.post(
        "/ask", headers=AUTH_HEADERS, json={"question": "What is revenue?"}
    )
    assert response.status_code == 200
    assert response.json() == {"answer": "stubbed answer"}


def test_ask_temporal_rejects_empty_quarters(client, monkeypatch) -> None:
    monkeypatch.setattr("api.main.ask_temporal", lambda *a, **k: "stub")
    response = client.post(
        "/ask/temporal",
        headers=AUTH_HEADERS,
        json={"question": "trend?", "quarters": []},
    )
    assert response.status_code == 422


def test_ingest_rejects_bad_quarter_format(client) -> None:
    response = client.post(
        "/ingest",
        headers=AUTH_HEADERS,
        json={
            "url": "https://example.com",
            "ticker": "AAPL",
            "quarter": "Q5-2025",  # Q5 doesn't exist
            "date": "2025-08-01",
        },
    )
    assert response.status_code == 422


def test_ingest_rejects_bad_date_format(client) -> None:
    response = client.post(
        "/ingest",
        headers=AUTH_HEADERS,
        json={
            "url": "https://example.com",
            "ticker": "AAPL",
            "quarter": "Q3-2025",
            "date": "August 1st",  # not YYYY-MM-DD
        },
    )
    assert response.status_code == 422


def test_ingest_returns_409_when_transcript_already_saved(
    client, tmp_path: Path, monkeypatch
) -> None:
    # Point DATA_DIR at an empty tmp dir, then create the "already-ingested" file.
    monkeypatch.setattr("api.main.DATA_DIR", tmp_path)
    (tmp_path / "AAPL_Q3_2025.json").write_text("{}")

    response = client.post(
        "/ingest",
        headers=AUTH_HEADERS,
        json={
            "url": "https://www.fool.com/x",
            "ticker": "AAPL",
            "quarter": "Q3-2025",
            "date": "2025-08-01",
        },
    )
    assert response.status_code == 409
    assert "already ingested" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Auth tests — protected endpoints must reject requests without a valid key.
# ---------------------------------------------------------------------------


def test_ask_rejects_missing_api_key(client) -> None:
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 401


def test_ask_rejects_wrong_api_key(client) -> None:
    response = client.post(
        "/ask",
        headers={"X-API-Key": "wrong-key"},
        json={"question": "anything"},
    )
    assert response.status_code == 401


def test_ask_temporal_rejects_missing_api_key(client) -> None:
    response = client.post(
        "/ask/temporal",
        json={"question": "trend?", "quarters": ["Q3-2025"]},
    )
    assert response.status_code == 401


def test_ingest_rejects_missing_api_key(client) -> None:
    response = client.post(
        "/ingest",
        json={
            "url": "https://example.com",
            "ticker": "AAPL",
            "quarter": "Q3-2025",
            "date": "2025-08-01",
        },
    )
    assert response.status_code == 401


def test_health_does_not_require_api_key(client) -> None:
    # Confirms GETs stay public.
    response = client.get("/health")
    assert response.status_code == 200


def test_collections_does_not_require_api_key(client) -> None:
    response = client.get("/collections")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting tests — verify the per-IP cap kicks in and returns 429.
# ---------------------------------------------------------------------------


@pytest.fixture
def enabled_limiter():
    """Turn the rate limiter on for one test and reset its counter."""
    limiter.reset()
    limiter.enabled = True
    yield limiter
    limiter.enabled = False
    limiter.reset()


def test_ask_rate_limit_returns_429_after_threshold(
    client, monkeypatch, enabled_limiter
) -> None:
    monkeypatch.setattr("api.main.ask", lambda *a, **k: "stub")

    # /ask is capped at 10/minute; the 11th call from the same IP should 429.
    for _ in range(10):
        ok = client.post("/ask", headers=AUTH_HEADERS, json={"question": "q"})
        assert ok.status_code == 200

    blocked = client.post("/ask", headers=AUTH_HEADERS, json={"question": "q"})
    assert blocked.status_code == 429


# ---------------------------------------------------------------------------
# SSRF tests — validate_ingest_url must reject non-https and private IPs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://www.fool.com/transcript",             # http (must be https)
        "ftp://www.fool.com/transcript",              # non-http(s)
        "https://evil.com/transcript",                # host not in allowlist
        "https://fool.com.evil.com/transcript",       # suffix-spoofing attempt
        "https://127.0.0.1/transcript",               # loopback
        "https://169.254.169.254/latest/meta-data/",  # AWS metadata
    ],
)
def test_ingest_rejects_unsafe_urls(client, url) -> None:
    response = client.post(
        "/ingest",
        headers=AUTH_HEADERS,
        json={
            "url": url,
            "ticker": "AAPL",
            "quarter": "Q3-2025",
            "date": "2025-08-01",
        },
    )
    assert response.status_code == 400, f"expected 400 for {url}, got {response.status_code}"
