#!/usr/bin/env python
"""CLI to ingest earnings call transcripts into ChromaDB.

Usage:
    uv run python scripts/ingest.py               # process only new transcripts
    uv run python scripts/ingest.py --reset       # wipe ChromaDB and re-index all
    uv run python scripts/ingest.py --dry-run     # preview what would be processed
    uv run python scripts/ingest.py --company AAPL  # limit to one ticker
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.embeddings import get_client, get_or_create_collection, index_transcript
from src.ingestion import load_companies, process_transcript, save_transcript


def _is_saved(ticker: str, quarter: str) -> bool:
    return (DATA_DIR / f"{ticker}_{quarter.replace('-', '_')}.json").exists()


def _ingest_one(collection, entry: dict) -> int:
    transcript = process_transcript(
        url=entry["url"],
        company=entry["ticker"],
        quarter=entry["quarter"],
        date=entry["date"],
    )
    save_transcript(transcript)
    return index_transcript(collection, transcript)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest earnings call transcripts into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Wipe ChromaDB and re-index all transcripts.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it.")
    parser.add_argument("--company", metavar="TICKER", help="Only process transcripts for this ticker.")
    args = parser.parse_args()

    all_entries = load_companies()
    scoped = [e for e in all_entries if not args.company or e["ticker"].upper() == args.company.upper()]

    if args.company and not scoped:
        print(f"No entries found for ticker: {args.company}", file=sys.stderr)
        sys.exit(1)

    to_process = scoped if args.reset else [e for e in scoped if not _is_saved(e["ticker"], e["quarter"])]

    print(f"Entries in scope  : {len(scoped)}")
    print(f"Already saved     : {len(scoped) - len(to_process)}")
    print(f"To process        : {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\nDry run — would process:")
        for e in to_process:
            print(f"  {e['ticker']}  {e['quarter']}  ({e['date']})")
        return

    collection = get_or_create_collection(get_client(), reset=args.reset)
    errors: dict[str, str] = {}
    total_chunks = 0

    for entry in to_process:
        key = f"{entry['ticker']} {entry['quarter']}"
        print(f"\nProcessing {key}...")
        try:
            n = _ingest_one(collection, entry)
            total_chunks += n
            print(f"  → {n} chunks indexed")
        except Exception as exc:
            errors[key] = str(exc)
            print(f"  ERROR: {exc}", file=sys.stderr)

    successes = len(to_process) - len(errors)
    print(f"\nTotal in collection : {collection.count()}")
    print(f"Indexed             : {total_chunks} chunks across {successes} transcript(s).")

    if errors:
        print(f"\nFailed ({len(errors)}):", file=sys.stderr)
        for key, msg in errors.items():
            print(f"  {key}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
