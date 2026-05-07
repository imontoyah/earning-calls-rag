"""Tests for the ChromaDB filter builder.

_build_where has four branches — easy to verify exhaustively.
"""

from api.main import _build_where


def test_no_filters_returns_none() -> None:
    assert _build_where(None, None) is None


def test_company_only_returns_flat_dict() -> None:
    assert _build_where("AAPL", None) == {"company": "AAPL"}


def test_quarter_only_returns_flat_dict() -> None:
    assert _build_where(None, "Q3-2025") == {"quarter": "Q3-2025"}


def test_both_filters_use_chromadb_and_syntax() -> None:
    # ChromaDB requires the explicit $and operator for compound filters.
    assert _build_where("AAPL", "Q3-2025") == {
        "$and": [{"company": "AAPL"}, {"quarter": "Q3-2025"}],
    }
