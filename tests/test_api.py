"""Endpoint tests for the FastAPI app."""

from pathlib import Path


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
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_returns_answer_from_rag(client, monkeypatch) -> None:
    monkeypatch.setattr("api.main.ask", lambda *a, **k: "stubbed answer")
    response = client.post("/ask", json={"question": "What is revenue?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "stubbed answer"}


def test_ask_temporal_rejects_empty_quarters(client, monkeypatch) -> None:
    monkeypatch.setattr("api.main.ask_temporal", lambda *a, **k: "stub")
    response = client.post("/ask/temporal", json={"question": "trend?", "quarters": []})
    assert response.status_code == 422


def test_ingest_rejects_bad_quarter_format(client) -> None:
    response = client.post(
        "/ingest",
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
        json={
            "url": "https://example.com",
            "ticker": "AAPL",
            "quarter": "Q3-2025",
            "date": "2025-08-01",
        },
    )
    assert response.status_code == 409
    assert "already ingested" in response.json()["detail"]
