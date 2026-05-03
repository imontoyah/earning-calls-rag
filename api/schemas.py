"""Pydantic request and response models for the API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    company: str | None = None
    quarter: str | None = None
    n_results: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str


class TemporalAskRequest(BaseModel):
    question: str = Field(min_length=1)
    quarters: list[str] = Field(min_length=1)
    company: str | None = None
    n_per_quarter: int = Field(default=2, ge=1, le=10)


class TemporalAskResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    collection_count: int


class CompanySummary(BaseModel):
    ticker: str
    quarters: list[str]


class CollectionsResponse(BaseModel):
    total: int
    companies: list[CompanySummary]


class IngestRequest(BaseModel):
    url: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    quarter: str = Field(pattern=r"^Q[1-4]-\d{4}$")  # e.g. "Q3-2025"
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")  # e.g. "2025-08-01"


class IngestResponse(BaseModel):
    ticker: str
    quarter: str
    total_turns: int
    chunks_indexed: int
