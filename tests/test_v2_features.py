"""Unit and integration tests for PolicyLens v2.0 features."""

import json
import os
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from policylens.api.main import create_app
from policylens.config import Settings
from policylens.generation.llm_client import LocalExtractiveLLMClient
from policylens.ingestion.pipeline import IdempotentIngestionPipeline
from policylens.retrieval.embeddings import (
    DEFAULT_FASTEMBED_MODEL,
    FastEmbedMultilingualProvider,
    HashEmbeddingProvider,
    build_embedding_provider,
)
from policylens.retrieval.models import RetrievedChunk
from policylens.retrieval.vector_store import QdrantVectorStore


@pytest.fixture()
def studio_client() -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,
        qdrant_path=":memory:",
        embedding_provider="hash",
        embedding_dim=384,
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        service = app.state.rag_service
        pipeline = IdempotentIngestionPipeline(
            store=service.retriever.vector_store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        pipeline.ingest_directory(Path("data/sample"))
        service.retriever.sync_bm25_from_store()
        yield client


async def test_local_extractive_llm_cleans_unmatched_bold_and_bullets() -> None:
    llm = LocalExtractiveLLMClient()
    chunks = [
        RetrievedChunk(
            chunk_id="FIN-POL-03#c02",
            doc_id="FIN-POL-03",
            title="Công tác phí",
            domain="finance",
            language="vi",
            effective_date="2026-01-05",
            section_path="Định mức",
            section_title="Định mức",
            char_start=0,
            char_end=220,
            text=(
                "## 2. Định mức Lưu trú & Công tác phí\n"
                "- **Phụ cấp ăn uống (Meal Per-Diem):** Áp dụng mức khoán **350,000 VND/ngày** "
                "cho công tác nội địa và **$60/day** đối với công tác quốc tế.\n"
                "Lưu trú khách sạn:** Hạn mức phòng tối đa 1,500,000 VND/đêm."
            ),
            rerank_score=0.95,
        )
    ]

    result = await llm.generate(
        "FIN-POL-03 mức phụ cấp ăn uống công tác nội địa là bao nhiêu?",
        chunks,
    )

    assert "Phụ cấp ăn uống:**" not in result.answer or "**Phụ cấp ăn uống" in result.answer
    assert "Lưu trú khách sạn:**" not in result.answer
    for line in result.answer.splitlines():
        assert line.count("**") % 2 == 0, f"Dangling bold marker in line: {line}"
    assert "##" not in result.answer
    assert "350,000 VND/ngày" in result.answer


async def test_fastembed_provider_lazy_init_async_and_mocked() -> None:
    init_calls: list[dict[str, object]] = []

    class MockTextEmbedding:
        def __init__(self, model_name: str, cache_dir=None, threads=None) -> None:
            init_calls.append({"model_name": model_name, "cache_dir": cache_dir})

        def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.full(384, 2.0, dtype=np.float32) for _ in texts]

        def query_embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.full(384, 3.0, dtype=np.float32) for _ in texts]

    provider = FastEmbedMultilingualProvider(
        model_name=DEFAULT_FASTEMBED_MODEL,
        embedding_cls=MockTextEmbedding,
    )
    # Lazy initialization: underlying model must NOT load in __init__.
    assert init_calls == []
    assert provider.dimension == 384

    doc_vecs = await provider.embed_documents(["Chính sách bảo mật", "Security policy"])
    query_vec = await provider.embed_query("Security policy")

    assert len(init_calls) == 1
    assert len(doc_vecs) == 2
    assert len(query_vec) == 384
    assert abs(float(np.linalg.norm(doc_vecs[0])) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(query_vec)) - 1.0) < 1e-5

    # build_embedding_provider routes 'fastembed' cleanly and guards OpenAI placeholder models.
    settings = Settings(
        _env_file=None,
        embedding_provider="fastembed",
        embedding_model="text-embedding-3-small",
    )
    built = build_embedding_provider(settings)
    assert isinstance(built, FastEmbedMultilingualProvider)
    assert built.model_name == DEFAULT_FASTEMBED_MODEL


@pytest.mark.skipif(
    os.getenv("EMBEDDING_PROVIDER") == "hash",
    reason="Skip live ONNX model download/smoke check when CI enforces EMBEDDING_PROVIDER=hash",
)
async def test_fastembed_multilingual_smoke_real_onnx() -> None:
    provider = FastEmbedMultilingualProvider(model_name=DEFAULT_FASTEMBED_MODEL)
    q_vec = await provider.embed_query("How many annual leave days can be carried over?")
    p_vec = (
        await provider.embed_documents(
            ["Tối đa 5 ngày phép năm chưa sử dụng được chuyển sang năm kế tiếp trước 31/03."]
        )
    )[0]

    assert len(q_vec) == 384
    assert abs(float(np.linalg.norm(q_vec)) - 1.0) < 1e-5
    assert float(np.dot(q_vec, p_vec)) > 0.25


def test_qdrant_recreates_collection_on_dimension_mismatch(tmp_path: Path) -> None:
    storage_dir = tmp_path / "qdrant_dim_test"
    settings_64 = Settings(_env_file=None, qdrant_path=str(storage_dir), embedding_dim=64)
    store_64 = QdrantVectorStore(settings=settings_64, embedder=HashEmbeddingProvider(dimension=64))
    assert store_64._get_existing_vector_size() == 64
    store_64.close()

    settings_384 = Settings(_env_file=None, qdrant_path=str(storage_dir), embedding_dim=384)
    store_384 = QdrantVectorStore(
        settings=settings_384,
        embedder=HashEmbeddingProvider(dimension=384),
    )
    try:
        assert store_384._get_existing_vector_size() == 384
        assert store_384.count_chunks() == 0
    finally:
        store_384.close()


def test_upload_chunks_query_and_stage_scores(studio_client: TestClient) -> None:
    nfd_content = unicodedata.normalize(
        "NFD",
        (
            "# OPS-SOP-99: Redis Failover Guardrail\n\n"
            "## 1. Giới hạn Bộ nhớ\n"
            "- **Ngưỡng cảnh báo:** Cụm Redis production kích hoạt cảnh báo ở **75%** bộ nhớ.\n"
            "- **Thời gian Failover SLA:** Chuyển mạch Sentinel hoàn tất trong **15 giây**.\n"
        ),
    )
    upload_payload = {
        "doc_id": "OPS-SOP-99",
        "title": "Redis Failover Guardrail",
        "domain": "engineering",
        "language": "vi",
        "content": nfd_content,
    }

    first = studio_client.post("/v1/documents/upload", json=upload_payload)
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["doc_id"] == "OPS-SOP-99"
    assert first_data["status"] == "indexed"
    assert first_data["chunks_indexed"] >= 1
    assert len(first_data["content_sha256"]) == 64

    # Re-uploading identical content is idempotent.
    second = studio_client.post("/v1/documents/upload", json=upload_payload)
    assert second.status_code == 200
    assert second.json()["status"] == "skipped"
    assert second.json()["chunks_indexed"] == 0

    # Chunk endpoint returns NFC-normalized document_text matching exact char offsets.
    chunks_resp = studio_client.get("/v1/documents/OPS-SOP-99/chunks")
    assert chunks_resp.status_code == 200
    chunks_data = chunks_resp.json()
    assert chunks_data["doc_id"] == "OPS-SOP-99"
    assert chunks_data["chunk_count"] >= 1
    assert unicodedata.is_normalized("NFC", chunks_data["document_text"])
    for chunk in chunks_data["chunks"]:
        start, end = chunk["char_start"], chunk["char_end"]
        assert chunks_data["document_text"][start:end] == chunk["text"]

    # Query newly indexed document immediately and inspect per-stage scores.
    query_resp = studio_client.post(
        "/v1/query",
        json={
            "query": "OPS-SOP-99 thời gian Failover SLA Redis là bao nhiêu giây?",
            "top_k": 4,
            "retrieval_mode": "hybrid_rrf_rerank",
        },
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["is_abstained"] is False
    assert any(c["doc_id"] == "OPS-SOP-99" for c in q_data["citations"])
    top_cite = q_data["citations"][0]
    for field in ("dense_score", "bm25_score", "rrf_score", "rerank_score"):
        assert field in top_cite
        assert isinstance(top_cite[field], float)
    assert top_cite["rerank_score"] > 0.0
    assert len(q_data["retrieved_chunks"]) >= 1

    missing = studio_client.get("/v1/documents/DOES-NOT-EXIST/chunks")
    assert missing.status_code == 404


def test_evals_latest_endpoint_and_missing_dir(
    studio_client: TestClient,
    tmp_path: Path,
) -> None:
    resp = studio_client.get("/v1/evals/latest")
    assert resp.status_code == 200
    payload = resp.json()
    assert [m["mode"] for m in payload["modes"]] == [
        "dense_only",
        "hybrid_rrf",
        "hybrid_rrf_rerank",
    ]
    assert len(payload["per_query"]) == 105
    assert payload["metadata"]["embedding_dim"] == 384

    studio_client.app.state.evals_results_dir = tmp_path
    missing_resp = studio_client.get("/v1/evals/latest")
    assert missing_resp.status_code == 404


def test_studio_html_console_and_colab_notebook_valid(studio_client: TestClient) -> None:
    resp = studio_client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "PolicyLens RAG Studio" in html
    assert "/v1/query/stream" in html
    assert "/v1/documents/upload" in html
    assert "/v1/evals/latest" in html
    assert "Fira Sans" in html and "Fira Code" in html

    nb_path = Path("notebooks/colab_finetune_reranker_gpu.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 6
