"""Shared pytest fixtures.

The pattern: build a fake ChromaDB collection that supports the methods
our handlers actually call, then use FastAPI's dependency_overrides to
swap the real one out. Tests run without touching the real vector store.
"""

import pytest
from fastapi.testclient import TestClient

from api.deps import get_collection
from api.main import app


class FakeCollection:
    """In-memory stand-in for chromadb.Collection.

    Only implements the methods our endpoints touch. Adding more is fine —
    just keep it lean so tests stay readable.
    """

    def __init__(
        self,
        count: int = 0,
        metadatas: list[dict] | None = None,
    ) -> None:
        self._count = count
        self._metadatas = metadatas or []

    def count(self) -> int:
        return self._count

    def get(self, include=None) -> dict:
        return {"metadatas": self._metadatas}


@pytest.fixture
def fake_collection() -> FakeCollection:
    """A small fixture collection: 2 quarters across 1 ticker."""
    return FakeCollection(
        count=3,
        metadatas=[
            {"company": "AAPL", "quarter": "Q3-2025"},
            {"company": "AAPL", "quarter": "Q4-2025"},
            {"company": "AAPL", "quarter": "Q4-2025"},
        ],
    )


@pytest.fixture
def client(fake_collection: FakeCollection):
    """TestClient with the collection dependency replaced by a fake.

    Cleanup runs after each test so overrides don't leak between tests.
    """
    app.dependency_overrides[get_collection] = lambda: fake_collection
    yield TestClient(app)
    app.dependency_overrides.clear()
