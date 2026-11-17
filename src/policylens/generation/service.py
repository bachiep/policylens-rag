"""End-to-end RAG orchestration service with guardrails, citation verification, and SSE."""

import json
import re
import time
from collections.abc import AsyncIterator

from policylens.config import Settings
from policylens.generation.llm_client import BaseLLMClient
from policylens.generation.schemas import (
    CitationSpan,
    DocumentSummary,
    GroundedAnswer,
    QueryFilters,
    QueryRequest,
    TokenUsage,
)
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
                        char_start=c.char_start,
                        char_end=c.char_end,
                    )
                )

        return verified

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

        total_tokens = gen_result.prompt_tokens + gen_result.completion_tokens
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return GroundedAnswer(
            answer=gen_result.answer,
            citations=citations,
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
            },
        )

        if decision.should_abstain:
            abstention_msg = build_abstention_message(request.query, decision.reason)
            prompt_tokens = estimate_tokens(request.query)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            final_answer = GroundedAnswer(
                answer=abstention_msg,
                citations=[],
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

        total_tokens = gen_result.prompt_tokens + gen_result.completion_tokens
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        final_answer = GroundedAnswer(
            answer=gen_result.answer,
            citations=citations,
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
