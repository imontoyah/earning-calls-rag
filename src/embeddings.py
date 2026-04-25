"""ChromaDB connection, document storage, and collection management."""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)

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


def index_transcript(
    collection: chromadb.Collection,
    transcript: dict,
) -> int:
    """
    Add all speaker turns from a transcript dict into the ChromaDB collection.

    Each speaker turn becomes one document with:
      - document  : turn["text"]
      - metadata  : company, quarter, speaker, role
      - id        : "<company>_<quarter>_<turn_index>"

    Returns the number of turns indexed.
    """
    documents = []
    metadatas = []
    ids = []

    for i,turn in enumerate(transcript["turns"]):
        single_id = f"{transcript['company']}_{transcript['quarter']}_{i}"
        metadata_info = {"company": transcript["company"], "quarter": transcript["quarter"],
                         "speaker": turn["speaker"], "role": turn["role"]}
        documents.append(turn["text"])
        metadatas.append(metadata_info)
        ids.append(single_id)
    
    for i in range(0, len(documents), BATCH_SIZE):
        end = min(i+BATCH_SIZE, len(documents))

        collection.add(
          documents=documents[i:end],
          metadatas=metadatas[i:end],
          ids=ids[i:end],
        )


    return len(documents)


def index_all(reset: bool = True) -> dict:
    """
    Load every JSON file from DATA_DIR and index it into ChromaDB.

    Returns a summary dict: {filename: n_turns_indexed}.
    """
    client = get_client()
    collection = get_or_create_collection(client, reset=reset)

    summary = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        n = index_transcript(collection, transcript)
        summary[path.name] = n
        print(f"  Indexed {n} turns from {path.name}")

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
