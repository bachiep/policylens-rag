"""Qdrant vector store adapter supporting :memory:, local disk path, and Docker URL."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from policylens.config import Settings
from policylens.ingestion.models import DocumentChunk
from policylens.retrieval.embeddings import EmbeddingProvider
from policylens.retrieval.models import RetrievalFilters, RetrievedChunk

logger = logging.getLogger(__name__)


def _point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _build_qdrant_filter(filters: RetrievalFilters | None) -> models.Filter | None:
    if filters is None or filters.is_empty():
        return None
    must_conditions: list[models.Condition] = []
    if filters.domain:
        must_conditions.append(
            models.FieldCondition(
                key="domain",
                match=models.MatchValue(value=filters.domain.lower()),
            )
        )
    if filters.language:
        must_conditions.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=filters.language.lower()),
            )
        )
    if filters.doc_id:
        must_conditions.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=filters.doc_id.upper()),
            )
        )
    return models.Filter(must=must_conditions)


class QdrantVectorStore:
    """Qdrant vector storage implementing ChunkStoreProtocol and dense similarity search."""

    def __init__(self, settings: Settings, embedder: EmbeddingProvider) -> None:
        self.settings = settings
        self.embedder = embedder
        self.collection_name = settings.qdrant_collection

        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        elif settings.qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=settings.qdrant_path)

        self._ensure_collection()

    def _get_existing_vector_size(self) -> int | None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, models.VectorParams):
            return int(vectors.size)
        if isinstance(vectors, dict):
            if "" in vectors:
                return int(vectors[""].size)
            if vectors:
                first = next(iter(vectors.values()))
                return int(first.size)
        return None

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self._create_collection()
            return
        existing_size = self._get_existing_vector_size()
        if existing_size is not None and existing_size != self.embedder.dimension:
            logger.warning(
                "Qdrant collection '%s' dimension mismatch (stored=%s, active=%s); recreating.",
                self.collection_name,
                existing_size,
                self.embedder.dimension,
            )
            self.client.delete_collection(self.collection_name)
            self._create_collection()

    def close(self) -> None:
        """Close underlying Qdrant client to release local file locks cleanly."""
        self.client.close()

    def count_chunks(self) -> int:
        info = self.client.count(collection_name=self.collection_name, exact=True)
        return int(info.count)

    def get_doc_hashes(self) -> dict[str, str]:
        """Return mapping of doc_id -> doc_hash stored in Qdrant payloads."""
        hashes: dict[str, str] = {}
        next_offset: Any = None
        active_embedder = self.embedder.embedder_id
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=["doc_id", "doc_hash", "embedder_id"],
                with_vectors=False,
            )
            for pt in points:
                payload = pt.payload or {}
                doc_id = payload.get("doc_id")
                doc_hash = payload.get("doc_hash")
                stored_embedder = payload.get("embedder_id")
                if isinstance(doc_id, str) and isinstance(doc_hash, str):
                    if stored_embedder == active_embedder:
                        hashes[doc_id] = doc_hash
                    else:
                        hashes[doc_id] = f"__stale__:{doc_hash}"
            if next_offset is None:
                break
        return hashes

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to doc_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id.upper()),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def _build_points(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> list[models.PointStruct]:
        embedder_id = self.embedder.embedder_id
        return [
            models.PointStruct(
                id=_point_uuid(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id.upper(),
                    "title": chunk.title,
                    "domain": chunk.domain.lower(),
                    "language": chunk.language.lower(),
                    "effective_date": chunk.effective_date,
                    "section_path": chunk.section_path,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "content_hash": chunk.content_hash,
                    "doc_hash": chunk.doc_hash,
                    "embedder_id": embedder_id,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert document chunks into Qdrant."""
        if not chunks:
            return
        texts = [chunk.retrieval_text() for chunk in chunks]
        vectors = self.embedder.embed_documents_sync(texts)
        points = self._build_points(chunks, vectors)
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    async def upsert_chunks_async(self, chunks: list[DocumentChunk]) -> None:
        """Asynchronously embed chunks without blocking the event loop and upsert."""
        if not chunks:
            return
        texts = [chunk.retrieval_text() for chunk in chunks]
        vectors = await self.embedder.embed_documents(texts)
        points = self._build_points(chunks, vectors)
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    @staticmethod
    def _payload_to_chunk(payload: dict[str, Any]) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            title=str(payload["title"]),
            domain=str(payload["domain"]),
            language=str(payload["language"]),
            effective_date=str(payload["effective_date"]),
            section_path=str(payload["section_path"]),
            section_title=str(payload["section_title"]),
            chunk_index=int(payload["chunk_index"]),
            char_start=int(payload["char_start"]),
            char_end=int(payload["char_end"]),
            content_hash=str(payload["content_hash"]),
            doc_hash=str(payload["doc_hash"]),
            text=str(payload["text"]),
        )

    def get_chunks_by_doc_id(self, doc_id: str) -> list[DocumentChunk]:
        """Load all stored chunks belonging to a single doc_id sorted by chunk_index."""
        loaded: list[DocumentChunk] = []
        next_offset: Any = None
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id.upper()),
                )
            ]
        )
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                loaded.append(self._payload_to_chunk(pt.payload or {}))
            if next_offset is None:
                break
        loaded.sort(key=lambda item: item.chunk_index)
        return loaded

    def load_all_chunks(self) -> list[DocumentChunk]:
        """Load all stored chunks from Qdrant payloads for BM25 sync and inspection."""
        loaded: list[DocumentChunk] = []
        next_offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                loaded.append(self._payload_to_chunk(pt.payload or {}))
            if next_offset is None:
                break
        loaded.sort(key=lambda item: (item.doc_id, item.chunk_index))
        return loaded

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 15,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Run dense cosine similarity search against Qdrant with optional payload filters."""
        qdrant_filter = _build_qdrant_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        hits: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            p = point.payload or {}
            score = float(point.score)
            hits.append(
                RetrievedChunk(
                    chunk_id=str(p["chunk_id"]),
                    doc_id=str(p["doc_id"]),
                    title=str(p["title"]),
                    domain=str(p["domain"]),
                    language=str(p["language"]),
                    effective_date=str(p["effective_date"]),
                    section_path=str(p["section_path"]),
                    section_title=str(p["section_title"]),
                    char_start=int(p["char_start"]),
                    char_end=int(p["char_end"]),
                    text=str(p["text"]),
                    dense_score=score,
                    dense_rank=rank,
                    final_score=score,
                )
            )
        return hits
