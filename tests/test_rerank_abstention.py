"""Unit and integration tests for pluggable reranking and abstention guardrails."""

import json
from pathlib import Path

import httpx

from policylens.config import Settings
from policylens.ingestion.pipeline import IdempotentIngestionPipeline
from policylens.retrieval.abstention import AbstentionGuard, build_abstention_message
from policylens.retrieval.bm25 import BM25Index
from policylens.retrieval.embeddings import LocalHashEmbeddingProvider
from policylens.retrieval.hybrid import HybridRetriever
from policylens.retrieval.models import RetrievedChunk
from policylens.retrieval.reranker import HttpApiReranker, LocalCrossScorerReranker
from policylens.retrieval.vector_store import QdrantVectorStore


async def test_local_reranker_promotes_exact_policy_and_heading_match() -> None:
    reranker = LocalCrossScorerReranker()
    distractor = RetrievedChunk(
        chunk_id="GEN#c00",
        doc_id="GEN-01",
        title="General Office Overview",
        domain="hr",
        language="en",
        effective_date="2026-01-01",
        section_path="General > Overview",
        section_title="Overview",
        char_start=0,
        char_end=60,
        text="Internal policy overview for corporate staff and standard access.",
        dense_score=0.65,
        rrf_score=0.025,
    )
    exact_match = RetrievedChunk(
        chunk_id="SEC-POL-04#c02",
        doc_id="SEC-POL-04",
        title="Quy định Quản lý Truy cập & Xác thực Đa yếu tố",
        domain="security",
        language="vi",
        effective_date="2026-01-15",
        section_path="2.2. Thời gian hết hạn phiên (Session Timeout)",
        section_title="2.2. Thời gian hết hạn phiên (Session Timeout)",
        char_start=0,
        char_end=90,
        text="Phiên đăng nhập tại cổng quản trị tự động hết hạn sau 15 phút không có thao tác.",
        dense_score=0.45,
        rrf_score=0.020,
    )

    reranked = await reranker.rerank(
        "SEC-POL-04 session timeout cổng quản trị bao nhiêu phút?",
        [distractor, exact_match],
        top_k=2,
    )
    assert reranked[0].chunk_id == "SEC-POL-04#c02"
    assert reranked[0].rerank_score > reranked[1].rerank_score


async def test_http_api_reranker_parses_mock_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["query"] == "KMS key rotation"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.24},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    reranker = HttpApiReranker(
        provider="jina",
        api_key="test-key",
        model="jina-reranker-v2-base-multilingual",
        transport=transport,
    )
    c0 = RetrievedChunk(
        chunk_id="A#c00",
        doc_id="A",
        title="Doc A",
        domain="hr",
        language="en",
        effective_date="2026-01-01",
        section_path="A",
        section_title="A",
        char_start=0,
        char_end=10,
        text="Annual leave policy",
    )
    c1 = RetrievedChunk(
        chunk_id="SEC-POL-07#c01",
        doc_id="SEC-POL-07",
        title="Data Classification",
        domain="security",
        language="en",
        effective_date="2026-02-01",
        section_path="Encryption",
        section_title="Encryption",
        char_start=0,
        char_end=30,
        text="KMS master keys rotate every 90 days.",
    )

    reranked = await reranker.rerank("KMS key rotation", [c0, c1], top_k=2)
    assert [c.chunk_id for c in reranked] == ["SEC-POL-07#c01", "A#c00"]
    assert reranked[0].rerank_score == 0.91


async def test_abstention_guard_accepts_in_scope_and_blocks_traps() -> None:
    settings = Settings(_env_file=None, qdrant_path=":memory:", embedding_dim=384)
    embedder = LocalHashEmbeddingProvider(dimension=384)
    store = QdrantVectorStore(settings=settings, embedder=embedder)

    try:
        pipeline = IdempotentIngestionPipeline(
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        pipeline.ingest_directory(Path("data/sample"))

        retriever = HybridRetriever(embedder=embedder, vector_store=store, bm25_index=BM25Index())
        retriever.sync_bm25_from_store()
        reranker = LocalCrossScorerReranker()
        guard = AbstentionGuard(
            score_threshold=settings.abstention_threshold,
            min_content_coverage=0.25,
        )

        # In-scope Vietnamese & English queries should pass abstention guard.
        for valid_query in (
            "Phụ cấp ăn uống công tác nội địa theo FIN-POL-03 là bao nhiêu?",
            "What is the Tier-0 monthly availability SLO and on-call ack SLA?",
        ):
            fused = await retriever.retrieve(valid_query, mode="hybrid_rrf", top_k=10)
            top = await reranker.rerank(valid_query, fused, top_k=4)
            decision = guard.evaluate(valid_query, top)
            assert not decision.should_abstain, (
                f"Unexpected abstention for: {valid_query} ({decision})"
            )

        # Out-of-scope traps and fake policy codes must trigger abstention.
        for trap_query in (
            "What is the company cryptocurrency Bitcoin treasury investment policy?",
            "Chính sách trợ cấp chăm sóc thú cưng tại văn phòng là gì?",
            "What is the Istio mTLS timeout threshold in OPS-NET-99?",
        ):
            fused = await retriever.retrieve(trap_query, mode="hybrid_rrf", top_k=10)
            top = await reranker.rerank(trap_query, fused, top_k=4)
            decision = guard.evaluate(trap_query, top)
            assert decision.should_abstain, (
                f"Expected abstention for trap: {trap_query} ({decision})"
            )
            msg = build_abstention_message(trap_query, decision.reason)
            assert msg
    finally:
        store.close()
