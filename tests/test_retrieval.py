"""Unit and integration tests for embeddings, BM25, Qdrant store, and RRF hybrid search."""

from pathlib import Path

import numpy as np

from policylens.config import Settings
from policylens.ingestion.pipeline import IdempotentIngestionPipeline
from policylens.retrieval.bm25 import BM25Index, tokenize_for_bm25
from policylens.retrieval.embeddings import LocalHashEmbeddingProvider
from policylens.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from policylens.retrieval.models import RetrievalFilters, RetrievedChunk
from policylens.retrieval.vector_store import QdrantVectorStore


def test_bm25_tokenizer_preserves_policy_codes_numbers_and_vietnamese() -> None:
    tokens = tokenize_for_bm25(
        "Quy định SEC-POL-04 yêu cầu MFA và hoàn ứng 350,000 VND với SLA 99.95%."
    )
    assert "sec-pol-04" in tokens
    assert "350000" in tokens
    assert "99.95%" in tokens
    assert "nghỉ" not in tokens
    assert "hoàn" in tokens and "ứng" in tokens
    assert "quy" in tokens and "định" in tokens


async def test_local_embedding_is_deterministic_and_normalized() -> None:
    embedder = LocalHashEmbeddingProvider(dimension=384)
    v1 = await embedder.embed_query("Quy định nghỉ phép năm và làm việc từ xa")
    v2 = await embedder.embed_query("Quy định nghỉ phép năm và làm việc từ xa")
    v3 = await embedder.embed_query("Annual leave and hybrid remote work policy")

    assert len(v1) == 384
    assert v1 == v2
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
    # Cross-lingual concept overlap yields positive cosine similarity.
    cosine_cross = float(np.dot(v1, v3))
    assert cosine_cross > 0.25


def test_reciprocal_rank_fusion_combines_dense_and_sparse_ranks() -> None:
    def _make_hit(
        chunk_id: str,
        dense_score: float = 0.0,
        sparse_score: float = 0.0,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id="DOC-1",
            title="Doc",
            domain="security",
            language="en",
            effective_date="2026-01-01",
            section_path="Sec",
            section_title="Sec",
            char_start=0,
            char_end=10,
            text="sample",
            dense_score=dense_score,
            sparse_score=sparse_score,
        )

    dense_hits = [_make_hit("A", dense_score=0.9), _make_hit("B", dense_score=0.8)]
    sparse_hits = [_make_hit("B", sparse_score=9.5), _make_hit("C", sparse_score=7.0)]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, rrf_k=60)
    # Chunk B appears in both rankings (rank 2 dense + rank 1 sparse) so it must win RRF.
    assert fused[0].chunk_id == "B"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    expected_b = (1.0 / 62) + (1.0 / 61)
    assert abs(fused[0].rrf_score - expected_b) < 1e-9


async def test_qdrant_and_hybrid_retriever_with_metadata_filters() -> None:
    settings = Settings(_env_file=None, qdrant_path=":memory:", embedding_dim=384)
    embedder = LocalHashEmbeddingProvider(dimension=384)
    store = QdrantVectorStore(settings=settings, embedder=embedder)

    try:
        pipeline = IdempotentIngestionPipeline(
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        first_report = pipeline.ingest_directory(Path("data/sample"))
        assert first_report.indexed_docs == 12
        assert store.count_chunks() == first_report.total_chunks

        # Re-running ingestion against Qdrant skips all 12 unchanged docs.
        second_report = pipeline.ingest_directory(Path("data/sample"))
        assert second_report.skipped_docs == 12

        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            bm25_index=BM25Index(),
        )
        synced = retriever.sync_bm25_from_store()
        assert synced == first_report.total_chunks

        # Exact policy code lookup should surface FIN-POL-03 at rank 1 in hybrid_rrf.
        hits = await retriever.retrieve(
            "FIN-POL-03 350,000 VND per-diem",
            mode="hybrid_rrf",
            top_k=3,
        )
        assert hits
        assert hits[0].doc_id == "FIN-POL-03"

        # Metadata filter restricts results strictly to domain="security" and language="en".
        filtered = await retriever.retrieve(
            "encryption key rotation KMS 90 days",
            mode="hybrid_rrf",
            top_k=5,
            filters=RetrievalFilters(domain="security", language="en"),
        )
        assert filtered
        assert all(item.domain == "security" and item.language == "en" for item in filtered)
        assert any(item.doc_id == "SEC-POL-07" for item in filtered)
    finally:
        store.close()
