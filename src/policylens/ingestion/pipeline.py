"""Idempotent corpus loading and chunk ingestion pipeline."""

import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from policylens.ingestion.chunker import MarkdownRecursiveChunker
from policylens.ingestion.models import DocumentChunk, IngestionReport, NormalizedDocument
from policylens.ingestion.normalizer import parse_markdown_document


@runtime_checkable
class ChunkStoreProtocol(Protocol):
    """Storage interface required by the idempotent ingestion pipeline."""

    def get_doc_hashes(self) -> dict[str, str]:
        """Return mapping of doc_id -> stored SHA-256 doc_hash."""

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all stored chunks belonging to doc_id."""

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Persist new or updated chunks into the store."""


class InMemoryChunkStore:
    """Lightweight in-memory chunk store implementing ChunkStoreProtocol."""

    def __init__(self) -> None:
        self.chunks_by_id: dict[str, DocumentChunk] = {}

    def get_doc_hashes(self) -> dict[str, str]:
        return {chunk.doc_id: chunk.doc_hash for chunk in self.chunks_by_id.values()}

    def delete_by_doc_id(self, doc_id: str) -> None:
        to_delete = [cid for cid, c in self.chunks_by_id.items() if c.doc_id == doc_id]
        for cid in to_delete:
            del self.chunks_by_id[cid]

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self.chunks_by_id[chunk.chunk_id] = chunk

    def list_chunks(self) -> list[DocumentChunk]:
        return list(self.chunks_by_id.values())


def load_corpus_documents(corpus_dir: Path) -> list[NormalizedDocument]:
    """Load and normalize all Markdown documents from a directory."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    docs: list[NormalizedDocument] = []
    for md_path in sorted(corpus_dir.glob("*.md")):
        raw_text = md_path.read_text(encoding="utf-8")
        docs.append(parse_markdown_document(raw_text, source_path=str(md_path)))
    return docs


class IdempotentIngestionPipeline:
    """SHA-256 hash-verified ingestion pipeline preventing duplicate indexing."""

    def __init__(
        self,
        store: ChunkStoreProtocol,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.store = store
        self.chunker = MarkdownRecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_documents(
        self,
        docs: list[NormalizedDocument],
        *,
        force: bool = False,
    ) -> IngestionReport:
        """Ingest normalized documents, skipping unchanged docs unless force=True."""
        started = time.perf_counter()
        existing_hashes = self.store.get_doc_hashes()

        indexed_docs = 0
        updated_docs = 0
        skipped_docs = 0
        total_chunks_written = 0
        doc_statuses: dict[str, str] = {}

        for doc in docs:
            doc_id = doc.metadata.doc_id
            prev_hash = existing_hashes.get(doc_id)

            if not force and prev_hash == doc.doc_hash:
                skipped_docs += 1
                doc_statuses[doc_id] = "skipped"
                continue

            chunks = self.chunker.chunk_document(doc)
            if prev_hash is not None:
                self.store.delete_by_doc_id(doc_id)
                updated_docs += 1
                doc_statuses[doc_id] = "updated"
            else:
                indexed_docs += 1
                doc_statuses[doc_id] = "indexed"

            if chunks:
                self.store.upsert_chunks(chunks)
                total_chunks_written += len(chunks)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return IngestionReport(
            total_docs=len(docs),
            indexed_docs=indexed_docs,
            updated_docs=updated_docs,
            skipped_docs=skipped_docs,
            total_chunks=total_chunks_written,
            duration_ms=duration_ms,
            doc_statuses=doc_statuses,
        )

    def ingest_directory(self, corpus_dir: Path, *, force: bool = False) -> IngestionReport:
        """Load all Markdown files in corpus_dir and run idempotent ingestion."""
        docs = load_corpus_documents(corpus_dir)
        return self.ingest_documents(docs, force=force)
