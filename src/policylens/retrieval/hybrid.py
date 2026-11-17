"""Reciprocal Rank Fusion (RRF) and unified Dense/Sparse/Hybrid retriever."""

from copy import deepcopy
from typing import Literal

from policylens.retrieval.bm25 import BM25Index
from policylens.retrieval.embeddings import EmbeddingProvider
from policylens.retrieval.models import RetrievalFilters, RetrievedChunk
from policylens.retrieval.vector_store import QdrantVectorStore

RetrievalMode = Literal["dense_only", "sparse_only", "hybrid_rrf"]


def reciprocal_rank_fusion(
    dense_hits: list[RetrievedChunk],
    sparse_hits: list[RetrievedChunk],
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fuse dense and sparse candidate rankings using Reciprocal Rank Fusion."""
    merged: dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        entry = deepcopy(hit)
        entry.dense_rank = rank
        entry.rrf_score += dense_weight / (rrf_k + rank)
        merged[entry.chunk_id] = entry

    for rank, hit in enumerate(sparse_hits, start=1):
        contrib = sparse_weight / (rrf_k + rank)
        if hit.chunk_id in merged:
            existing = merged[hit.chunk_id]
            existing.sparse_score = hit.sparse_score
            existing.sparse_rank = rank
            existing.rrf_score += contrib
        else:
            entry = deepcopy(hit)
            entry.sparse_rank = rank
            entry.rrf_score += contrib
            merged[entry.chunk_id] = entry

    fused = list(merged.values())
    for item in fused:
        item.final_score = item.rrf_score

    fused.sort(
        key=lambda item: (item.rrf_score, item.dense_score, item.sparse_score),
        reverse=True,
    )
    return fused


class HybridRetriever:
    """Coordinates Qdrant dense search, BM25 sparse search, and RRF fusion."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index | None = None,
        *,
        rrf_k: int = 60,
        top_k_dense: int = 15,
        top_k_sparse: int = 15,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

    def sync_bm25_from_store(self) -> int:
        """Rebuild the BM25 index from all chunks persisted in Qdrant."""
        chunks = self.vector_store.load_all_chunks()
        self.bm25_index.build(chunks)
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid_rrf",
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Execute dense_only, sparse_only, or hybrid_rrf retrieval."""
        if mode == "sparse_only":
            return self.bm25_index.search(query, top_k=top_k, filters=filters)

        query_vector = await self.embedder.embed_query(query)
        dense_limit = top_k if mode == "dense_only" else max(self.top_k_dense, top_k)
        dense_hits = self.vector_store.search_dense(
            query_vector=query_vector,
            top_k=dense_limit,
            filters=filters,
        )

        if mode == "dense_only":
            return dense_hits[:top_k]

        sparse_limit = max(self.top_k_sparse, top_k)
        sparse_hits = self.bm25_index.search(
            query=query,
            top_k=sparse_limit,
            filters=filters,
        )
        fused = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            rrf_k=self.rrf_k,
        )
        return fused[:top_k]
