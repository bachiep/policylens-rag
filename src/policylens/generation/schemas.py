"""Pydantic v2 request and grounded response schemas for PolicyLens API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryFilters(BaseModel):
    """Optional metadata filters scoped to domain, language, or document ID."""

    domain: str | None = Field(default=None, max_length=50)
    language: Literal["vi", "en"] | None = None
    doc_id: str | None = Field(default=None, max_length=50)


class QueryRequest(BaseModel):
    """Incoming RAG query payload."""

    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    retrieval_mode: Literal["dense_only", "hybrid_rrf", "hybrid_rrf_rerank"] = (
        "hybrid_rrf_rerank"
    )
    filters: QueryFilters | None = None

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Query must contain at least 2 non-whitespace characters.")
        return cleaned


class CitationSpan(BaseModel):
    """Verified source citation bound to a retrieved chunk and character span."""

    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    quote: str
    score: float
    char_start: int
    char_end: int


class TokenUsage(BaseModel):
    """Token counts and estimated cost for a query request."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)


class GroundedAnswer(BaseModel):
    """Structured RAG response enforcing verified citations and abstention flags."""

    answer: str
    citations: list[CitationSpan]
    is_abstained: bool
    abstention_reason: str | None = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_mode: str
    prompt_version: str
    model_name: str
    latency_ms: float = Field(ge=0.0)
    usage: TokenUsage


class DocumentSummary(BaseModel):
    """Metadata summary for an indexed corporate document."""

    doc_id: str
    title: str
    domain: str
    language: str
    effective_date: str
    chunk_count: int
