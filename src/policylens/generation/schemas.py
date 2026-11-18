"""Pydantic v2 request and grounded response schemas for PolicyLens API."""

import re
from typing import Any, Literal

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
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    char_start: int
    char_end: int


class RetrievedChunkScore(BaseModel):
    """Per-stage retrieval score breakdown for visual score inspection."""

    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    char_start: int
    char_end: int
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    cited: bool = False


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
    retrieved_chunks: list[RetrievedChunkScore] = Field(default_factory=list)
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


class DocumentUploadRequest(BaseModel):
    """Incoming Markdown policy/SOP payload for live NFC normalization and indexing."""

    doc_id: str = Field(..., min_length=2, max_length=64)
    title: str = Field(..., min_length=2, max_length=200)
    domain: str = Field(default="general", min_length=1, max_length=50)
    language: Literal["vi", "en"] = "en"
    content: str = Field(..., min_length=10, max_length=200_000)
    effective_date: str = Field(default="2026-11-18", max_length=32)

    @field_validator("doc_id")
    @classmethod
    def _normalize_doc_id(cls, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-").upper()
        if len(cleaned) < 2:
            raise ValueError("doc_id must contain at least 2 alphanumeric characters.")
        return cleaned

    @field_validator("title", "domain", "content")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


class DocumentUploadResponse(BaseModel):
    """Result returned after idempotent document upload and index sync."""

    doc_id: str
    chunks_indexed: int
    content_sha256: str
    status: Literal["indexed", "updated", "skipped"]
    total_doc_chunks: int


class ChunkSpanDetail(BaseModel):
    """Indexed chunk metadata and verbatim text with character spans."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    section_path: str
    section_title: str
    char_start: int
    char_end: int
    text: str


class DocumentChunksResponse(BaseModel):
    """Full document text and ordered chunk spans for source highlighting."""

    doc_id: str
    title: str
    domain: str
    language: str
    effective_date: str
    chunk_count: int
    document_text: str
    chunks: list[ChunkSpanDetail]


class LatestEvalsResponse(BaseModel):
    """Latest benchmark summary and per-query diagnostics."""

    modes: list[dict[str, Any]]
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
