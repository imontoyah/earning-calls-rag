"""FastAPI backend for the earnings call RAG service."""

import chromadb
from fastapi import Depends, FastAPI, HTTPException, status
from langchain_groq import ChatGroq

from api.deps import get_collection, get_llm, verify_api_key
from api.schemas import (
    AskRequest,
    AskResponse,
    CollectionsResponse,
    CompanySummary,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    TemporalAskRequest,
    TemporalAskResponse,
)
from src.config import DATA_DIR
from src.embeddings import index_transcript
from src.ingestion import process_transcript, save_transcript
from src.rag import ask, ask_temporal

app = FastAPI(
    title="Earnings Call RAG",
    description="Query and analyze public earnings call transcripts.",
    version="0.1.0",
)


def _build_where(company: str | None, quarter: str | None) -> dict | None:
    """Compose a ChromaDB filter from optional company/quarter fields.

    ChromaDB requires {"$and": [...]} for compound filters but a flat dict
    for a single filter — this helper hides that asymmetry.
    """
    filters = []
    if company:
        filters.append({"company": company})
    if quarter:
        filters.append({"quarter": quarter})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]  
    return {"$and": filters}


@app.get("/health", response_model=HealthResponse)
def health(collection: chromadb.Collection = Depends(get_collection)) -> HealthResponse:
    return HealthResponse(status="ok", collection_count=collection.count())


@app.get("/collections", response_model=CollectionsResponse)
def collections(
    collection: chromadb.Collection = Depends(get_collection),
) -> CollectionsResponse:
    all_docs = collection.get(include=["metadatas"])
    by_company: dict[str, set[str]] = {}
    for m in all_docs["metadatas"]:
        by_company.setdefault(m["company"], set()).add(m["quarter"])
    return CollectionsResponse(
        total=collection.count(),
        companies=[
            CompanySummary(ticker=ticker, quarters=sorted(quarters))
            for ticker, quarters in sorted(by_company.items())
        ],
    )


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask_endpoint(
    req: AskRequest,
    collection: chromadb.Collection = Depends(get_collection),
    llm: ChatGroq = Depends(get_llm),
) -> AskResponse:
    answer = ask(
        collection,
        req.question,
        llm=llm,
        n_results=req.n_results,
        where=_build_where(req.company, req.quarter),
    )
    return AskResponse(answer=answer)


@app.post("/ask/temporal", response_model=TemporalAskResponse, dependencies=[Depends(verify_api_key)])
def ask_temporal_endpoint(
    req: TemporalAskRequest,
    collection: chromadb.Collection = Depends(get_collection),
    llm: ChatGroq = Depends(get_llm),
) -> TemporalAskResponse:
    answer = ask_temporal(
        collection,
        req.question,
        req.quarters,
        llm=llm,
        company=req.company,
        n_per_quarter=req.n_per_quarter,
    )
    return TemporalAskResponse(answer=answer)

@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
)
def ingest_endpoint(
    req: IngestRequest,
    collection: chromadb.Collection = Depends(get_collection),
) -> IngestResponse:
    """Download, parse, save, and index a single earnings call transcript.

    Returns 201 on success, 409 if the transcript is already on disk.
    """
    saved_path = DATA_DIR / f"{req.ticker}_{req.quarter.replace('-', '_')}.json"
    if saved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Transcript {req.ticker} {req.quarter} is already ingested. "
                   f"Delete {saved_path.name} or run 'scripts/ingest.py --reset' to re-ingest.",
        )

    transcript = process_transcript(
        url=req.url,
        company=req.ticker,
        quarter=req.quarter,
        date=req.date,
    )
    save_transcript(transcript)
    n_chunks = index_transcript(collection, transcript)

    return IngestResponse(
        ticker=req.ticker,
        quarter=req.quarter,
        total_turns=transcript["total_turns"],
        chunks_indexed=n_chunks,
    )
