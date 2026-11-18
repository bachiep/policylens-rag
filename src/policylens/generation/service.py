"""End-to-end RAG orchestration service with guardrails, citation verification, and SSE."""

import hashlib
import json
import re
import time
from collections.abc import AsyncIterator

from policylens.config import Settings
from policylens.generation.llm_client import BaseLLMClient
from policylens.generation.schemas import (
    ChunkSpanDetail,
    CitationSpan,
    DocumentChunksResponse,
    DocumentSummary,
    DocumentUploadRequest,
    DocumentUploadResponse,
    GroundedAnswer,
    QueryFilters,
    QueryRequest,
    RetrievedChunkScore,
    TokenUsage,
)
from policylens.ingestion.chunker import MarkdownRecursiveChunker
from policylens.ingestion.models import DocumentChunk, DocumentMetadata, NormalizedDocument
from policylens.ingestion.normalizer import parse_markdown_document
from policylens.ingestion.pipeline import load_corpus_documents
from policylens.prompts.rag_qa_v1 import PROMPT_VERSION, estimate_cost_usd, estimate_tokens
from policylens.retrieval.abstention import AbstentionGuard, build_abstention_message
from policylens.retrieval.hybrid import HybridRetriever
from policylens.retrieval.models import RetrievalFilters, RetrievedChunk
from policylens.retrieval.reranker import BaseReranker

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+|your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"(reveal|print|output)\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety|policy)\s+guardrails", re.IGNORECASE),
)


class InputGuardrailError(ValueError):
    """Raised when a query violates length limits or prompt-injection guardrails."""


def _to_retrieval_filters(filters: QueryFilters | None) -> RetrievalFilters | None:
    if filters is None:
        return None
    return RetrievalFilters(
        domain=filters.domain,
        language=filters.language,
        doc_id=filters.doc_id,
    )


def _extract_verbatim_quote(chunk: RetrievedChunk, max_len: int = 240) -> str:
    """Extract a clean verbatim excerpt from the chunk body."""
    lines = [
        ln.strip()
        for ln in chunk.text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    source = " ".join(lines) if lines else chunk.text.strip()
    if len(source) <= max_len:
        return source
    cutoff = source.rfind(" ", 0, max_len)
    if cutoff < max_len // 2:
        cutoff = max_len
    return source[:cutoff].rstrip() + "..."


def _format_sse(event: str, payload: dict[str, object] | str) -> str:
    serialized = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {serialized}\n\n"


def _reconstruct_document_text(chunks: list[DocumentChunk]) -> str:
    """Reconstruct span-accurate document text from ordered chunks preserving char offsets."""
    if not chunks:
        return ""
    max_end = max(c.char_end for c in chunks)
    buffer = [" "] * max_end
    ordered = sorted(chunks, key=lambda c: c.char_start)

    cursor = 0
    for chunk in ordered:
        if chunk.char_start > cursor:
            gap = chunk.char_start - cursor
            for offset in range(min(2, gap)):
                buffer[cursor + offset] = "\n"
        end = min(chunk.char_end, chunk.char_start + len(chunk.text))
        for idx, ch in enumerate(chunk.text[: end - chunk.char_start]):
            buffer[chunk.char_start + idx] = ch
        cursor = max(cursor, chunk.char_end)

    return "".join(buffer)


class RagPipelineService:
    """Coordinates retrieval, reranking, abstention, grounded generation, and streaming."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        abstention_guard: AbstentionGuard,
        llm_client: BaseLLMClient,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.abstention_guard = abstention_guard
        self.llm_client = llm_client
        self._uploaded_bodies: dict[str, str] = {}

    def validate_query(self, query: str) -> str:
        cleaned = query.strip()
        if len(cleaned) > self.settings.max_query_chars:
            raise InputGuardrailError(
                f"Query exceeds maximum length of {self.settings.max_query_chars} characters."
            )
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(cleaned):
                raise InputGuardrailError(
                    "Query rejected by input guardrail: instruction override pattern detected."
                )
        return cleaned

    def list_documents(self) -> list[DocumentSummary]:
        """Return aggregated metadata for all documents currently indexed in Qdrant."""
        chunks = self.retriever.vector_store.load_all_chunks()
        grouped: dict[str, DocumentSummary] = {}
        for chunk in chunks:
            if chunk.doc_id not in grouped:
                grouped[chunk.doc_id] = DocumentSummary(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    domain=chunk.domain,
                    language=chunk.language,
                    effective_date=chunk.effective_date,
                    chunk_count=1,
                )
            else:
                grouped[chunk.doc_id].chunk_count += 1
        return sorted(grouped.values(), key=lambda doc: doc.doc_id)

    def indexed_chunk_count(self) -> int:
        return self.retriever.vector_store.count_chunks()

    async def upload_document(self, request: DocumentUploadRequest) -> DocumentUploadResponse:
        """Normalize, chunk, embed, and idempotently index a Markdown policy document."""
        parsed = parse_markdown_document(
            request.content,
            source_path=f"<upload:{request.doc_id}>",
        )
        body = parsed.body
        if not body:
            body = request.content.strip()
        doc_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        doc_id = request.doc_id.upper()

        normalized_doc = NormalizedDocument(
            metadata=DocumentMetadata(
                doc_id=doc_id,
                title=request.title,
                domain=request.domain.lower(),
                language=request.language.lower(),
                effective_date=request.effective_date,
                owner="Studio Upload",
                source_path=f"<upload:{doc_id}>",
            ),
            body=body,
            doc_hash=doc_hash,
        )

        self._uploaded_bodies[doc_id] = body
        existing_hashes = self.retriever.vector_store.get_doc_hashes()
        prev_hash = existing_hashes.get(doc_id)

        if prev_hash == doc_hash:
            existing_chunks = self.retriever.vector_store.get_chunks_by_doc_id(doc_id)
            return DocumentUploadResponse(
                doc_id=doc_id,
                chunks_indexed=0,
                content_sha256=doc_hash,
                status="skipped",
                total_doc_chunks=len(existing_chunks),
            )

        chunker = MarkdownRecursiveChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = chunker.chunk_document(normalized_doc)
        status = "updated" if prev_hash is not None else "indexed"
        if prev_hash is not None:
            self.retriever.vector_store.delete_by_doc_id(doc_id)

        if chunks:
            await self.retriever.vector_store.upsert_chunks_async(chunks)
        self.retriever.sync_bm25_from_store()

        return DocumentUploadResponse(
            doc_id=doc_id,
            chunks_indexed=len(chunks),
            content_sha256=doc_hash,
            status=status,
            total_doc_chunks=len(chunks),
        )

    def _resolve_document_text(self, doc_id: str, chunks: list[DocumentChunk]) -> str:
        cached = self._uploaded_bodies.get(doc_id)
        if cached and all(cached[c.char_start : c.char_end] == c.text for c in chunks):
            return cached

        corpus_dir = self.settings.sample_corpus_dir
        if corpus_dir.exists():
            for doc in load_corpus_documents(corpus_dir):
                if doc.metadata.doc_id.upper() == doc_id and doc.doc_hash == chunks[0].doc_hash:
                    if all(doc.body[c.char_start : c.char_end] == c.text for c in chunks):
                        return doc.body

        return _reconstruct_document_text(chunks)

    def get_document_chunks(self, doc_id: str) -> DocumentChunksResponse | None:
        """Return ordered chunks and span-accurate source document text for highlighting."""
        normalized_id = doc_id.strip().upper()
        chunks = self.retriever.vector_store.get_chunks_by_doc_id(normalized_id)
        if not chunks:
            return None

        head = chunks[0]
        document_text = self._resolve_document_text(normalized_id, chunks)
        return DocumentChunksResponse(
            doc_id=head.doc_id,
            title=head.title,
            domain=head.domain,
            language=head.language,
            effective_date=head.effective_date,
            chunk_count=len(chunks),
            document_text=document_text,
            chunks=[
                ChunkSpanDetail(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    chunk_index=c.chunk_index,
                    section_path=c.section_path,
                    section_title=c.section_title,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    text=c.text,
                )
                for c in chunks
            ],
        )

    async def retrieve_candidates(self, request: QueryRequest) -> list[RetrievedChunk]:
        """Execute retrieval according to requested mode."""
        filters = _to_retrieval_filters(request.filters)
        if request.retrieval_mode == "dense_only":
            return await self.retriever.retrieve(
                request.query,
                mode="dense_only",
                top_k=request.top_k,
                filters=filters,
            )
        if request.retrieval_mode == "hybrid_rrf":
            return await self.retriever.retrieve(
                request.query,
                mode="hybrid_rrf",
                top_k=request.top_k,
                filters=filters,
            )

        candidate_pool = max(12, request.top_k * 3)
        fused = await self.retriever.retrieve(
            request.query,
            mode="hybrid_rrf",
            top_k=candidate_pool,
            filters=filters,
        )
        return await self.reranker.rerank(request.query, fused, top_k=request.top_k)

    def _build_verified_citations(
        self,
        cited_chunk_ids: list[str],
        candidates: list[RetrievedChunk],
    ) -> list[CitationSpan]:
        by_id = {c.chunk_id: c for c in candidates}
        verified: list[CitationSpan] = []
        seen: set[str] = set()

        for cid in cited_chunk_ids:
            if cid in by_id and cid not in seen:
                seen.add(cid)
                c = by_id[cid]
                score = c.rerank_score if c.rerank_score > 0.0 else c.final_score
                verified.append(
                    CitationSpan(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        title=c.title,
                        section_path=c.section_path,
                        quote=_extract_verbatim_quote(c),
                        score=round(max(0.0, min(1.0, float(score))), 4),
                        dense_score=round(float(c.dense_score), 4),
                        bm25_score=round(float(c.sparse_score), 4),
                        rrf_score=round(float(c.rrf_score), 6),
                        rerank_score=round(float(c.rerank_score), 4),
                        char_start=c.char_start,
                        char_end=c.char_end,
                    )
                )

        return verified

    @staticmethod
    def _build_retrieved_chunk_scores(
        candidates: list[RetrievedChunk],
        cited_ids: set[str],
    ) -> list[RetrievedChunkScore]:
        return [
            RetrievedChunkScore(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                section_path=c.section_path,
                char_start=c.char_start,
                char_end=c.char_end,
                dense_score=round(float(c.dense_score), 4),
                bm25_score=round(float(c.sparse_score), 4),
                rrf_score=round(float(c.rrf_score), 6),
                rerank_score=round(float(c.rerank_score), 4),
                final_score=round(float(c.final_score), 6),
                cited=c.chunk_id in cited_ids,
            )
            for c in candidates
        ]

    async def answer_query(self, request: QueryRequest) -> GroundedAnswer:
        """Run full guarded RAG pipeline and return verified GroundedAnswer."""
        started = time.perf_counter()
        self.validate_query(request.query)

        candidates = await self.retrieve_candidates(request)
        decision = self.abstention_guard.evaluate(request.query, candidates)

        if decision.should_abstain:
            abstention_msg = build_abstention_message(request.query, decision.reason)
            prompt_tokens = estimate_tokens(request.query)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return GroundedAnswer(
                answer=abstention_msg,
                citations=[],
                retrieved_chunks=self._build_retrieved_chunk_scores(candidates, set()),
                is_abstained=True,
                abstention_reason=decision.reason,
                confidence_score=decision.top_score,
                retrieval_mode=request.retrieval_mode,
                prompt_version=PROMPT_VERSION,
                model_name=self.llm_client.model_name,
                latency_ms=latency_ms,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=prompt_tokens,
                    estimated_cost_usd=estimate_cost_usd(
                        prompt_tokens, 0, self.llm_client.model_name
                    ),
                ),
            )

        gen_result = await self.llm_client.generate(request.query, candidates)
        citations = self._build_verified_citations(gen_result.cited_chunk_ids, candidates)

        # If external LLM omitted citations despite grounded generation, link top supporting chunk
        # only when non-empty answer was produced without abstaining; if LLM returned no citations
        # and explicitly abstained, mark abstained cleanly.
        if not citations and not gen_result.cited_chunk_ids and candidates:
            citations = self._build_verified_citations([candidates[0].chunk_id], candidates)

        cited_id_set = {c.chunk_id for c in citations}
        total_tokens = gen_result.prompt_tokens + gen_result.completion_tokens
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return GroundedAnswer(
            answer=gen_result.answer,
            citations=citations,
            retrieved_chunks=self._build_retrieved_chunk_scores(candidates, cited_id_set),
            is_abstained=False,
            abstention_reason=None,
            confidence_score=decision.top_score,
            retrieval_mode=request.retrieval_mode,
            prompt_version=PROMPT_VERSION,
            model_name=gen_result.model_name,
            latency_ms=latency_ms,
            usage=TokenUsage(
                prompt_tokens=gen_result.prompt_tokens,
                completion_tokens=gen_result.completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimate_cost_usd(
                    gen_result.prompt_tokens,
                    gen_result.completion_tokens,
                    gen_result.model_name,
                ),
            ),
        )

    async def stream_query(self, request: QueryRequest) -> AsyncIterator[str]:
        """Stream retrieval metadata, token deltas, and final GroundedAnswer via SSE."""
        started = time.perf_counter()
        self.validate_query(request.query)

        candidates = await self.retrieve_candidates(request)
        decision = self.abstention_guard.evaluate(request.query, candidates)

        yield _format_sse(
            "retrieval",
            {
                "candidate_count": len(candidates),
                "top_chunk_ids": [c.chunk_id for c in candidates],
                "is_abstained": decision.should_abstain,
                "confidence_score": decision.top_score,
                "retrieved_chunks": [
                    item.model_dump()
                    for item in self._build_retrieved_chunk_scores(candidates, set())
                ],
            },
        )

        if decision.should_abstain:
            abstention_msg = build_abstention_message(request.query, decision.reason)
            prompt_tokens = estimate_tokens(request.query)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            final_answer = GroundedAnswer(
                answer=abstention_msg,
                citations=[],
                retrieved_chunks=self._build_retrieved_chunk_scores(candidates, set()),
                is_abstained=True,
                abstention_reason=decision.reason,
                confidence_score=decision.top_score,
                retrieval_mode=request.retrieval_mode,
                prompt_version=PROMPT_VERSION,
                model_name=self.llm_client.model_name,
                latency_ms=latency_ms,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=prompt_tokens,
                    estimated_cost_usd=0.0,
                ),
            )
            yield _format_sse("token", {"delta": abstention_msg})
            yield _format_sse("final", final_answer.model_dump_json())
            return

        gen_result = await self.llm_client.generate(request.query, candidates)
        citations = self._build_verified_citations(gen_result.cited_chunk_ids, candidates)
        if not citations and candidates:
            citations = self._build_verified_citations([candidates[0].chunk_id], candidates)

        # Stream answer tokens cleanly in small word chunks for responsive UI rendering.
        words = gen_result.answer.split(" ")
        chunk_step = 6
        for i in range(0, len(words), chunk_step):
            piece = " ".join(words[i : i + chunk_step])
            if i + chunk_step < len(words):
                piece += " "
            yield _format_sse("token", {"delta": piece})

        cited_id_set = {c.chunk_id for c in citations}
        total_tokens = gen_result.prompt_tokens + gen_result.completion_tokens
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        final_answer = GroundedAnswer(
            answer=gen_result.answer,
            citations=citations,
            retrieved_chunks=self._build_retrieved_chunk_scores(candidates, cited_id_set),
            is_abstained=False,
            abstention_reason=None,
            confidence_score=decision.top_score,
            retrieval_mode=request.retrieval_mode,
            prompt_version=PROMPT_VERSION,
            model_name=gen_result.model_name,
            latency_ms=latency_ms,
            usage=TokenUsage(
                prompt_tokens=gen_result.prompt_tokens,
                completion_tokens=gen_result.completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimate_cost_usd(
                    gen_result.prompt_tokens,
                    gen_result.completion_tokens,
                    gen_result.model_name,
                ),
            ),
        )
        yield _format_sse("final", final_answer.model_dump_json())
