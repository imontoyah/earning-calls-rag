"""Four retrieval strategies: semantic, self-query, temporal, hybrid."""

import json

import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.config import (
    DATA_DIR,
    DEFAULT_N_RESULTS,
    HYBRID_WEIGHTS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    N_RESULTS_PER_QUARTER,
)

try:
    from langchain_classic.chains.query_constructor.base import AttributeInfo
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.chains.query_constructor.base import AttributeInfo
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.retrievers import EnsembleRetriever


# ---------------------------------------------------------------------------
# 1. Semantic search
# ---------------------------------------------------------------------------

def semantic_search(
    collection: chromadb.Collection,
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    where: dict | None = None,
) -> list[dict]:
    """
    Query ChromaDB by semantic similarity, with optional metadata filters.

    Returns a list of dicts with keys: text, speaker, role, quarter, company, distance.

    The `where` parameter accepts ChromaDB filter syntax, e.g.:
        {"quarter": "Q4-2025"}
        {"$and": [{"role": "Chief Financial Officer"}, {"quarter": "Q3-2025"}]}
    """
    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    return [
        {
            "text": results["documents"][0][i],
            "speaker": results["metadatas"][0][i].get("speaker", ""),
            "role": results["metadatas"][0][i].get("role", ""),
            "quarter": results["metadatas"][0][i].get("quarter", ""),
            "company": results["metadatas"][0][i].get("company", ""),
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["documents"][0]))
    ]


# ---------------------------------------------------------------------------
# 2. SelfQueryRetriever
# ---------------------------------------------------------------------------

def make_self_query_retriever(
    vectorstore: Chroma,
    quarters: list[str],
    llm: ChatGroq | None = None,
) -> "SelfQueryRetriever":
    """
    Build a LangChain SelfQueryRetriever that auto-extracts metadata filters
    from a natural language question.
    """
    if llm is None:
        llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    metadata_field_info = [
        AttributeInfo(
            name="company",
            type="string",
            description="Stock ticker symbol. Values: AAPL (Apple)",
        ),
        AttributeInfo(
            name="quarter",
            type="string",
            description=f"Fiscal quarter in Q#-YYYY format. Values: {', '.join(quarters)}",
        ),
        AttributeInfo(
            name="speaker",
            type="string",
            description="Full name of the speaker. Examples: Timothy D. Cook, Kevan Parekh",
        ),
        AttributeInfo(
            name="role",
            type="string",
            description=(
                "Job title of the speaker. "
                "Examples: Chief Executive Officer, Chief Financial Officer. "
                "Empty string for analysts."
            ),
        ),
    ]

    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="Transcripts of quarterly earnings call presentations and Q&A sessions",
        metadata_field_info=metadata_field_info,
    )


# ---------------------------------------------------------------------------
# 3. Temporal comparison
# ---------------------------------------------------------------------------

def retrieve_per_quarter(
    collection: chromadb.Collection,
    query: str,
    quarters: list[str],
    n_per_quarter: int = N_RESULTS_PER_QUARTER,
    company: str | None = None,
) -> dict[str, list[dict]]:
    """
    Retrieve the top chunks for each quarter independently.

    Returns a dict keyed by quarter, each value being a list of dicts
    with keys: text, speaker, role, quarter, company, distance.
    """
    dict_quarter = {}
    for quarter in quarters:
        if company is not None:
            where_filter = {"$and": [{"quarter": quarter}, {"company": company}]}
        else:
            where_filter = {"quarter": f"{quarter}"}
        
        result = semantic_search(collection, query, n_per_quarter, where_filter)
        dict_quarter[quarter] = result

    return dict_quarter


def build_temporal_context(results_by_quarter: dict[str, list[dict]]) -> str:
    """Format per-quarter results into a context string with === Q#-YYYY === headers."""
    sections = []
    for quarter in sorted(results_by_quarter):       # sorted() gives ordered keys
        chunks = results_by_quarter[quarter]          # list[dict]
        chunk_lines = []
        for chunk in chunks:
            speaker_label = chunk["speaker"]
            if chunk["role"]:
                speaker_label += f" ({chunk['role']})"
            chunk_lines.append(f"[{speaker_label}]: {chunk['text']}")
        sections.append(f"=== {quarter} ===\n" + "\n\n".join(chunk_lines))
    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# 4. Hybrid search (BM25 + semantic)
# ---------------------------------------------------------------------------

def _load_all_documents() -> list[Document]:
    """Load every transcript turn from DATA_DIR as LangChain Documents."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        for turn in transcript["turns"]:
            docs.append(
                Document(
                    page_content=turn["text"],
                    metadata={
                        "company": transcript["company"],
                        "quarter": transcript["quarter"],
                        "speaker": turn["speaker"],
                        "role": turn["role"],
                    },
                )
            )
    return docs


def make_hybrid_retriever(
    vectorstore: Chroma,
    k: int = DEFAULT_N_RESULTS,
    weights: list[float] | None = None,
) -> "EnsembleRetriever":
    """
    Build an EnsembleRetriever that combines BM25 (keyword) and semantic search.

    weights[0] = BM25 weight, weights[1] = semantic weight. Must sum to 1.
    """
    if weights is None:
        weights = HYBRID_WEIGHTS

    documents = _load_all_documents()
    bm25 = BM25Retriever.from_documents(documents, k=k)
    semantic = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=weights,
    )
