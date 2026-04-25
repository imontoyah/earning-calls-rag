"""RAG chain: prompt + LLM + output parsing."""

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import DEFAULT_N_RESULTS, LLM_MODEL, LLM_TEMPERATURE, N_RESULTS_PER_QUARTER
from src.retrieval import (
    build_temporal_context,
    retrieve_per_quarter,
    semantic_search,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst assistant answering questions about earnings calls.
Use ONLY the information in the context below. If the answer is not present, say so explicitly — do not speculate or use prior knowledge.
When relevant, cite the speaker's name, role, and quarter.

Context:
{context}

Question: {question}
"""
)

TEMPORAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst assistant.
The context below contains earnings call excerpts organized by quarter.
Analyze the data across ALL quarters to identify trends and changes.
Be specific with numbers when they appear in the context.
If information is missing for a quarter, say so explicitly.

Context:
{context}

Question: {question}
"""
)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def make_llm(model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: list[dict]) -> str:
    """Convert a list of chunk dicts into a single context string."""
    parts = []
    for c in chunks:
        speaker = c.get("speaker", "?")
        role = c.get("role", "")
        quarter = c.get("quarter", "?")
        company = c.get("company", "?")
        label = f"{company} {quarter} — {speaker}" + (f" ({role})" if role else "")
        parts.append(f"[{label}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    collection: chromadb.Collection,
    question: str,
    llm: ChatGroq | None = None,
    n_results: int = DEFAULT_N_RESULTS,
    where: dict | None = None,
) -> str:
    """Standard RAG: semantic retrieval → prompt → LLM → answer."""
    if llm is None:
        llm = make_llm()

    chain = RAG_PROMPT | llm | StrOutputParser()

    chunks = semantic_search(collection, question, n_results=n_results, where=where)
    context = _format_context(chunks)

    return chain.invoke({"context": context, "question": question})


def ask_temporal(
    collection: chromadb.Collection,
    question: str,
    quarters: list[str],
    llm: ChatGroq | None = None,
    company: str | None = None,
    n_per_quarter: int = N_RESULTS_PER_QUARTER,
) -> str:
    """Temporal RAG: per-quarter retrieval → combined context → LLM → answer."""
    if llm is None:
        llm = make_llm()

    chain = TEMPORAL_PROMPT | llm | StrOutputParser()

    results_by_quarter = retrieve_per_quarter(
        collection, question, quarters, n_per_quarter=n_per_quarter, company=company
    )
    context = build_temporal_context(results_by_quarter)

    return chain.invoke({"context": context, "question": question})
