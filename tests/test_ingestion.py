"""Unit tests for Unicode NFC normalization, Markdown chunking, and idempotency."""

import unicodedata
from pathlib import Path

from policylens.ingestion.chunker import MarkdownRecursiveChunker
from policylens.ingestion.normalizer import (
    normalize_unicode_and_whitespace,
    parse_markdown_document,
)
from policylens.ingestion.pipeline import (
    IdempotentIngestionPipeline,
    InMemoryChunkStore,
    load_corpus_documents,
)


def test_nfc_unicode_normalization_on_vietnamese_diacritics() -> None:
    # Decomposed NFD string ("Quy định bảo mật")
    nfd_text = unicodedata.normalize("NFD", "Quy định bảo mật công ty\r\n\r\n\r\nChi tiết.")
    normalized = normalize_unicode_and_whitespace(nfd_text)

    assert unicodedata.is_normalized("NFC", normalized)
    assert normalized == "Quy định bảo mật công ty\n\nChi tiết."


def test_sample_corpus_contains_12_valid_bilingual_documents() -> None:
    docs = load_corpus_documents(Path("data/sample"))
    assert len(docs) == 12

    doc_ids = {doc.metadata.doc_id for doc in docs}
    assert {"SEC-POL-04", "FIN-POL-03", "OPS-SLA-02"}.issubset(doc_ids)

    languages = {doc.metadata.language for doc in docs}
    assert languages == {"vi", "en"}


def test_chunker_preserves_section_paths_and_exact_char_spans() -> None:
    docs = load_corpus_documents(Path("data/sample"))
    chunker = MarkdownRecursiveChunker(chunk_size=650, chunk_overlap=100)

    total_chunks = 0
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2, f"Expected multiple chunks for {doc.metadata.doc_id}"
        for chunk in chunks:
            total_chunks += 1
            assert chunk.doc_id == doc.metadata.doc_id
            assert chunk.section_path
            assert chunk.section_title
            assert len(chunk.text) <= 650
            # Verify char_start..char_end matches normalized doc.body verbatim.
            assert doc.body[chunk.char_start : chunk.char_end] == chunk.text

    assert total_chunks >= 35


def test_chunker_splits_oversized_single_paragraph_with_overlap() -> None:
    long_sentence = " ".join(f"token{i}" for i in range(180))
    raw = f"---\ndoc_id: TEST-01\ntitle: Long Doc\n---\n# Long Doc\n\n## Section A\n{long_sentence}"
    doc = parse_markdown_document(raw)
    chunker = MarkdownRecursiveChunker(chunk_size=250, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 4
    for chunk in chunks:
        assert len(chunk.text) <= 250
        assert doc.body[chunk.char_start : chunk.char_end] == chunk.text
        assert "Section A" in chunk.section_path


def test_idempotent_ingestion_skips_unchanged_and_updates_modified() -> None:
    store = InMemoryChunkStore()
    pipeline = IdempotentIngestionPipeline(store=store, chunk_size=700, chunk_overlap=100)

    doc_v1 = parse_markdown_document(
        "---\ndoc_id: POL-01\ntitle: Policy One\n---\n# Policy One\n\n## Rule\nOriginal rule text.",
        source_path="pol_01.md",
    )
    first_report = pipeline.ingest_documents([doc_v1])
    assert first_report.indexed_docs == 1
    assert first_report.skipped_docs == 0
    assert first_report.total_chunks >= 1

    # Re-ingesting identical document skips without writing duplicate chunks.
    second_report = pipeline.ingest_documents([doc_v1])
    assert second_report.indexed_docs == 0
    assert second_report.skipped_docs == 1
    assert second_report.total_chunks == 0

    # Modifying content updates the document cleanly.
    doc_v2 = parse_markdown_document(
        (
            "---\ndoc_id: POL-01\ntitle: Policy One\n---\n"
            "# Policy One\n\n## Rule\nUpdated rule text 2026."
        ),
        source_path="pol_01.md",
    )
    third_report = pipeline.ingest_documents([doc_v2])
    assert third_report.updated_docs == 1
    assert third_report.skipped_docs == 0
    assert any("Updated rule text 2026" in c.text for c in store.list_chunks())
