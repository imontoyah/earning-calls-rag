"""Unit tests for transcript parsing and chunking."""

from src.embeddings import chunk_text
from src.ingestion import parse_speaker_turns


def test_parse_speaker_turns_extracts_role_from_header() -> None:
    text = (
        "CALL PARTICIPANTS\n"
        "Chief Executive Officer — Timothy D. Cook\n"
        "Chief Financial Officer — Kevan Parekh\n"
        "Full Conference Call Transcript\n"
        "Timothy D. Cook:\n"
        "Thank you, everyone. Good afternoon and welcome to the call.\n"
        "Kevan Parekh:\n"
        "Thanks Tim. Revenue was up 10 percent year over year this quarter.\n"
    )
    turns = parse_speaker_turns(text)

    assert len(turns) == 2
    assert turns[0]["speaker"] == "Timothy D. Cook"
    assert turns[0]["role"] == "Chief Executive Officer"
    assert "Good afternoon" in turns[0]["text"]
    assert turns[1]["speaker"] == "Kevan Parekh"
    assert turns[1]["role"] == "Chief Financial Officer"


def test_parse_speaker_turns_returns_empty_for_no_speakers() -> None:
    assert parse_speaker_turns("just some text with no speaker turns") == []


def test_parse_speaker_turns_skips_short_turns() -> None:
    # Turns with text shorter than ~10 chars are filtered out.
    text = (
        "Full Conference Call Transcript\n"
        "Timothy D. Cook:\n"
        "Hi.\n"
        "Kevan Parekh:\n"
        "This turn has enough text to be kept by the parser.\n"
    )
    turns = parse_speaker_turns(text)
    assert len(turns) == 1
    assert turns[0]["speaker"] == "Kevan Parekh"


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    short = " ".join(["word"] * 50)  # well below CHUNK_SIZE (200)
    assert chunk_text(short) == [short]


def test_chunk_text_splits_long_text_with_overlap() -> None:
    # 500 distinct words: w0 w1 ... w499
    words = [f"w{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    assert len(chunks) > 1
    # Each chunk should be at most chunk_size words long.
    for chunk in chunks:
        assert len(chunk.split()) <= 200
    # Overlap: the last 30 words of chunk 0 should appear at the start of chunk 1.
    chunk0_tail = chunks[0].split()[-30:]
    chunk1_head = chunks[1].split()[:30]
    assert chunk0_tail == chunk1_head
