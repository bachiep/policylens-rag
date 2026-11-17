"""Pluggable reranker interfaces (local lexical-semantic cross-scorer & HTTP API)."""

import asyncio
import re
import unicodedata
from abc import ABC, abstractmethod
from copy import deepcopy

import httpx

from policylens.config import Settings
from policylens.retrieval.bm25 import tokenize_for_bm25
from policylens.retrieval.embeddings import POLICY_CODE_RE, extract_concept_ids, strip_accents
from policylens.retrieval.models import RetrievedChunk

_GENERIC_CORPUS_TERMS = frozenset(
    {
        "policy",
        "policies",
        "rule",
        "rules",
        "standard",
        "standards",
        "sop",
        "guideline",
        "guidelines",
        "company",
        "corporate",
        "internal",
        "employee",
        "employees",
        "staff",
        "team",
        "system",
        "service",
        "threshold",
        "limit",
        "maximum",
        "minimum",
        "required",
        "requirement",
        "requirements",
        "what",
        "how",
        "many",
        "much",
        "quy",
        "định",
        "chính",
        "sách",
        "trình",
        "công",
        "ty",
        "nhân",
        "viên",
        "nội",
        "bộ",
        "hệ",
        "thống",
        "ngưỡng",
        "giới",
        "hạn",
        "tối",
        "đa",
        "thiểu",
        "yêu",
        "cầu",
        "bắt",
        "buộc",
        "thời",
        "nào",
        "như",
        "thế",
        "cần",
        "phải",
    }
)


def extract_distinctive_terms(text: str) -> list[str]:
    """Extract non-generic content tokens from a query for coverage scoring."""
    tokens = tokenize_for_bm25(text)
    distinctive: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _GENERIC_CORPUS_TERMS or len(tok) < 2:
            continue
        if tok not in seen:
            seen.add(tok)
            distinctive.append(tok)
    return distinctive


def compute_content_coverage(query: str, chunk: RetrievedChunk) -> float:
    """Measure fraction of distinctive query terms/concepts grounded in the chunk."""
    distinctive = extract_distinctive_terms(query)
    if not distinctive:
        distinctive = tokenize_for_bm25(query)
    if not distinctive:
        return 0.0

    doc_full = unicodedata.normalize(
        "NFC",
        f"{chunk.doc_id} {chunk.title} {chunk.section_path} {chunk.text}",
    ).lower()
    doc_fold = strip_accents(doc_full)
    doc_tokens = set(tokenize_for_bm25(doc_full))
    doc_fold_tokens = set(tokenize_for_bm25(doc_fold))
    doc_concepts = extract_concept_ids(doc_full)

    matched = 0.0
    for term in distinctive:
        term_fold = strip_accents(term)
        if term in doc_tokens or term in doc_full:
            matched += 1.0
        elif len(term_fold) >= 4 and (term_fold in doc_fold_tokens or term_fold in doc_fold):
            matched += 0.9
        else:
            term_concepts = extract_concept_ids(term)
            if term_concepts and (term_concepts & doc_concepts):
                matched += 0.85

    return min(1.0, matched / len(distinctive))


class BaseReranker(ABC):
    """Abstract interface for second-stage candidate reranking."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        """Score and sort candidates in descending relevance order."""


class LocalCrossScorerReranker(BaseReranker):
    """Deterministic local cross-scorer combining lexical, concept, heading, and RRF cues."""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        query_nfc = unicodedata.normalize("NFC", query).lower()
        query_codes = [c.lower() for c in POLICY_CODE_RE.findall(query_nfc)]
        distinctive = extract_distinctive_terms(query)
        bigrams = [
            f"{distinctive[i]} {distinctive[i + 1]}"
            for i in range(max(0, len(distinctive) - 1))
        ]

        reranked: list[RetrievedChunk] = []
        for candidate in candidates:
            item = deepcopy(candidate)
            coverage = compute_content_coverage(query, item)

            header_text = unicodedata.normalize(
                "NFC", f"{item.doc_id} {item.title} {item.section_path}"
            ).lower()
            header_concepts = extract_concept_ids(header_text)
            if distinctive:
                header_hits = sum(
                    1.0
                    for t in distinctive
                    if t in header_text or (extract_concept_ids(t) & header_concepts)
                )
                title_align = min(1.0, header_hits / max(1, min(len(distinctive), 3)))
            else:
                title_align = 0.0

            body_lower = unicodedata.normalize("NFC", f"{item.section_path} {item.text}").lower()
            if bigrams:
                bigram_hits = sum(1 for bg in bigrams if bg in body_lower)
                bigram_score = min(1.0, bigram_hits / len(bigrams))
            else:
                bigram_score = 0.0

            # Normalize RRF (max ~0.0328 at k=60, rank 1+1) or dense cosine fallback.
            prior_signal = max(min(1.0, item.rrf_score / 0.0328), item.dense_score * 0.6)

            score = (
                0.48 * coverage
                + 0.18 * title_align
                + 0.14 * bigram_score
                + 0.20 * prior_signal
            )

            if query_codes:
                doc_id_lower = item.doc_id.lower()
                if any(code == doc_id_lower or code in body_lower for code in query_codes):
                    score += 0.15
                else:
                    score *= 0.35

            item.rerank_score = round(max(0.0, min(1.0, score)), 4)
            item.final_score = item.rerank_score
            reranked.append(item)

        reranked.sort(
            key=lambda c: (c.rerank_score, c.rrf_score, c.dense_score),
            reverse=True,
        )
        return reranked[:top_k]


class HttpApiReranker(BaseReranker):
    """HTTP reranker supporting Jina and Cohere `/rerank` endpoints."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport

    def _endpoint(self) -> str:
        if self.provider == "cohere":
            return "https://api.cohere.com/v1/rerank"
        return "https://api.jina.ai/v1/rerank"

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        documents = [f"{c.title} | {c.section_path}\n{c.text}" for c in candidates]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(candidates)),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(self._endpoint(), headers=headers, json=payload)
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    reranked: list[RetrievedChunk] = []
                    for entry in results:
                        idx = int(entry["index"])
                        score = float(entry["relevance_score"])
                        item = deepcopy(candidates[idx])
                        item.rerank_score = round(max(0.0, min(1.0, score)), 4)
                        item.final_score = item.rerank_score
                        reranked.append(item)
                    reranked.sort(key=lambda c: c.rerank_score, reverse=True)
                    return reranked[:top_k]
                except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"Reranker API request failed after retries: {last_error}")


def build_reranker(settings: Settings) -> BaseReranker:
    """Instantiate configured reranker with deterministic local fallback."""
    if settings.reranker_provider in {"jina", "cohere"} and settings.reranker_api_key:
        return HttpApiReranker(
            provider=settings.reranker_provider,
            api_key=settings.reranker_api_key,
            model=settings.reranker_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return LocalCrossScorerReranker()


_VI_CHAR_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.I,
)


def detect_query_language(query: str) -> str:
    """Detect whether a query is predominantly Vietnamese or English."""
    if _VI_CHAR_RE.search(query):
        return "vi"
    return "en"
