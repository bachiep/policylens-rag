"""Pydantic v2 request and grounded response schemas for PolicyLens API."""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RetrievalStrategy = Literal["dense_only", "hybrid_rrf", "hybrid_rrf_rerank"]


class QueryFilters(BaseModel):
    """Optional metadata filters scoped to domain, language, or document ID."""

    domain: str | None = Field(default=None, max_length=50)
    language: Literal["vi", "en"] | None = None
    doc_id: str | None = Field(default=None, max_length=50)


class QueryRequest(BaseModel):
    """Incoming RAG query payload."""

    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    retrieval_mode: RetrievalStrategy = "hybrid_rrf_rerank"
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


class QueryCompareRequest(BaseModel):
    """Side-by-side retrieval mode A/B comparison payload."""

    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    left_mode: RetrievalStrategy = "dense_only"
    right_mode: RetrievalStrategy = "hybrid_rrf_rerank"
    domain: str | None = Field(default=None, max_length=50)
    language: Literal["vi", "en"] | None = None
    doc_id: str | None = Field(default=None, max_length=50)
    filters: QueryFilters | None = None

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Query must contain at least 2 non-whitespace characters.")
        return cleaned

    def resolved_filters(self) -> QueryFilters | None:
        """Merge top-level domain/language/doc_id convenience fields with optional filters."""
        base_domain = self.domain if self.domain is not None else (
            self.filters.domain if self.filters else None
        )
        if base_domain is not None:
            base_domain = base_domain.strip().lower()
            if not base_domain or base_domain == "all":
                base_domain = None

        base_language = self.language if self.language is not None else (
            self.filters.language if self.filters else None
        )
        base_doc_id = self.doc_id if self.doc_id is not None else (
            self.filters.doc_id if self.filters else None
        )
        if base_doc_id is not None:
            base_doc_id = base_doc_id.strip() or None

        if base_domain is None and base_language is None and base_doc_id is None:
            return None
        return QueryFilters(domain=base_domain, language=base_language, doc_id=base_doc_id)


class ComparedChunkScore(BaseModel):
    """Ranked retrieved chunk with per-stage scores and A/B rank movement metadata."""

    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    quote: str
    char_start: int
    char_end: int
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    cited: bool = False
    rank_delta: int | None = None
    newly_surfaced: bool = False
    dropped_in_right: bool = False


class ComparisonSideResult(BaseModel):
    """Single-side execution result inside a side-by-side retrieval A/B comparison."""

    retrieval_mode: str
    is_abstained: bool
    abstention_reason: str | None = None
    top_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    answer: str
    answer_preview: str
    citations: list[CitationSpan] = Field(default_factory=list)
    retrieved_chunks: list[ComparedChunkScore] = Field(default_factory=list)
    usage: TokenUsage


class QueryCompareResponse(BaseModel):
    """Side-by-side A/B retrieval comparison with rank delta highlights."""

    query: str
    top_k: int
    filters: QueryFilters | None = None
    left: ComparisonSideResult
    right: ComparisonSideResult
    abstain_resolved: bool
    top_chunk_changed: bool
    top_score_delta: float
    newly_surfaced_chunk_ids: list[str] = Field(default_factory=list)
    promoted_chunk_ids: list[str] = Field(default_factory=list)
    total_latency_ms: float = Field(ge=0.0)
