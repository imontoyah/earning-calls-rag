"""ChromaDB connection, document storage, and collection management."""

import json

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)
from src.ingestion import load_companies, process_transcript, save_transcript

BATCH_SIZE = 50


def _make_chroma_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def _make_langchain_embedding_fn():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Return an existing collection (raises if it doesn't exist)."""
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_make_chroma_embedding_fn(),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    reset: bool = False,
) -> chromadb.Collection:
    """Get the collection, optionally wiping it first."""
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_make_chroma_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def get_langchain_vectorstore(persist_directory: str | None = None) -> Chroma:
    """Return a LangChain Chroma wrapper (needed for SelfQueryRetriever)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_make_langchain_embedding_fn(),
        persist_directory=persist_directory or str(CHROMA_DIR),
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping sub-chunks by word count.

    Returns a list of strings. If the text is shorter than chunk_size,
    returns a single-element list with the original text.
    """
    words = text.split()
    if len(words)<= chunk_size:
        return [text]
    else:
        chunks = []
        start = 0
        while start < len(words):
            batch = words[start: start+chunk_size]
            full_text = " ".join(batch)
            chunks.append(full_text)

            start += chunk_size-overlap
    
    return chunks


def index_transcript(
    collection: chromadb.Collection,
    transcript: dict,
) -> int:
    """
    Add all speaker turns from a transcript dict into the ChromaDB collection.

    Long turns are split into overlapping sub-chunks via chunk_text().
    Each sub-chunk becomes one document, preserving speaker/role metadata.

    ID format: "<company>_<quarter>_<turn_index>_<chunk_index>"

    Returns the total number of chunks indexed.
    """
    documents = []
    metadatas = []
    ids = []

    metadata_base = {
        "company": transcript["company"],
        "quarter": transcript["quarter"],
    }

    for turn_i, turn in enumerate(transcript["turns"]):
        chunks = chunk_text(turn["text"])
        for chunk_j, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                **metadata_base,
                "speaker": turn["speaker"],
                "role": turn["role"],
            })
            ids.append(f"{transcript['company']}_{transcript['quarter']}_{turn_i}_{chunk_j}")

    for i in range(0, len(documents), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )

    return len(documents)


def index_all(reset: bool = True) -> dict:
    """
    Load every JSON file from DATA_DIR and index it into ChromaDB.

    Returns a summary dict: {filename: n_chunks_indexed}.
    """
    client = get_client()
    collection = get_or_create_collection(client, reset=reset)

    summary = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        n = index_transcript(collection, transcript)
        summary[path.name] = n
        print(f"  Indexed {n} chunks from {path.name}")

    print(f"\nTotal documents in collection: {collection.count()}")
    return summary


def index_from_config(reset: bool = True) -> dict:
    """
    Download, parse, save, and index every transcript declared in companies.json.

    This is the main entry point for multi-company ingestion. It:
      1. Reads companies.json via load_companies()
      2. Downloads and parses each transcript via process_transcript()
      3. Saves the JSON to data/processed/ via save_transcript()
      4. Indexes all chunks into ChromaDB via index_transcript()

    Returns a summary dict: {"TICKER Q#-YYYY": n_chunks_indexed}.
    """
    client = get_client()
    collection = get_or_create_collection(client, reset=reset)

    entries = load_companies()
    summary = {}

    for entry in entries:
        key = f"{entry['ticker']} {entry['quarter']}"
        print(f"  Processing {key}...")
        transcript = process_transcript(
            url=entry["url"],
            company=entry["ticker"],
            quarter=entry["quarter"],
            date=entry["date"],
        )
        save_transcript(transcript)
        n = index_transcript(collection, transcript)
        summary[key] = n
        print(f"    → {n} chunks indexed")


    print(f"\nTotal documents in collection: {collection.count()}")
    return summary


def get_summary(collection: chromadb.Collection) -> dict:
    """Return a breakdown of documents per quarter."""
    all_docs = collection.get(include=["metadatas"])
    quarters: dict[str, int] = {}
    for m in all_docs["metadatas"]:
        q = m["quarter"]
        quarters[q] = quarters.get(q, 0) + 1
    return {"total": collection.count(), "by_quarter": dict(sorted(quarters.items()))}
