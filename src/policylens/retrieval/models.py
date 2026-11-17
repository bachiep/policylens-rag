"""Data structures for retrieval filters and ranked chunk candidates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional metadata constraints applied across dense and sparse retrieval."""

    domain: str | None = None
    language: str | None = None
    doc_id: str | None = None

    def is_empty(self) -> bool:
        return self.domain is None and self.language is None and self.doc_id is None


@dataclass
class RetrievedChunk:
    """Retrieved document chunk carrying per-stage ranking signals."""

    chunk_id: str
    doc_id: str
    title: str
    domain: str
    language: str
    effective_date: str
    section_path: str
    section_title: str
    char_start: int
    char_end: int
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
