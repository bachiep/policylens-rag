"""Data models for normalized documents, chunks, and ingestion reports."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentMetadata:
    """Structured metadata extracted from a policy or SOP document."""

    doc_id: str
    title: str
    domain: str
    language: str
    effective_date: str
    owner: str
    source_path: str


@dataclass(frozen=True)
class NormalizedDocument:
    """Unicode-normalized Markdown document with content SHA-256 hash."""

    metadata: DocumentMetadata
    body: str
    doc_hash: str


@dataclass(frozen=True)
class DocumentChunk:
    """Single Markdown-aware text chunk preserving exact character spans."""

    chunk_id: str
    doc_id: str
    title: str
    domain: str
    language: str
    effective_date: str
    section_path: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int
    content_hash: str
    doc_hash: str
    text: str

    def retrieval_text(self) -> str:
        """Return text enriched with document and section breadcrumb for indexing."""
        return f"[{self.doc_id}] {self.title} | {self.section_path}\n{self.text}"


@dataclass(frozen=True)
class IngestionReport:
    """Summary metrics produced by an idempotent ingestion run."""

    total_docs: int
    indexed_docs: int
    updated_docs: int
    skipped_docs: int
    total_chunks: int
    duration_ms: float
    doc_statuses: dict[str, str] = field(default_factory=dict)
