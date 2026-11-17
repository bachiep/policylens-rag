"""Policy-code and Vietnamese-aware tokenizer paired with BM25Okapi sparse retrieval."""

import re
import unicodedata

from rank_bm25 import BM25Okapi

from policylens.ingestion.models import DocumentChunk
from policylens.retrieval.models import RetrievalFilters, RetrievedChunk

_STOPWORDS = frozenset(
    {
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "with",
        "by",
        "from",
        "as",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "must",
        "have",
        "has",
        "có",
        "không",
        "thể",
        "được",
        "và",
        "của",
        "là",
        "các",
        "những",
        "cho",
        "trong",
        "với",
        "tại",
        "theo",
        "này",
        "từ",
        "đến",
        "khi",
        "nào",
        "gì",
        "bao",
        "nhiêu",
    }
)

# Capture hyphenated codes (SEC-POL-04), numeric values with decimals/%, and Unicode words.
_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+"  # codes like SEC-POL-04, AES-256-GCM, T-5, Net-30
    r"|\d+(?:[.,]\d+)*%?"              # numbers like 350,000, 99.95%, 5,000,000
    r"|[a-zA-ZÀ-ỹ0-9_]+",              # words preserving Vietnamese diacritics
    re.UNICODE,
)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize bilingual text while keeping exact policy codes, numbers, and diacritics."""
    normalized = unicodedata.normalize("NFC", text).lower()
    tokens: list[str] = []

    for match in _TOKEN_RE.findall(normalized):
        if match in _STOPWORDS:
            continue

        # Normalize formatted thousand-separated integers (350,000 -> 350000) while preserving
        # decimal floats like 99.95% or 1.0%.
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", match):
            tokens.append(match.replace(",", ""))
            continue

        tokens.append(match)
        if match.endswith("%") and len(match) > 1:
            tokens.append(match[:-1])

    return tokens


class BM25Index:
    """In-memory BM25Okapi index synchronized from stored document chunks."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Rebuild the BM25 index from a list of DocumentChunk objects."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        corpus_tokens = [tokenize_for_bm25(chunk.retrieval_text()) for chunk in self._chunks]
        self._bm25 = BM25Okapi(corpus_tokens)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 15,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Score query tokens against indexed chunks and return top-K matches."""
        if not self._bm25 or not self._chunks:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        scored_indices: list[tuple[int, float]] = []
        for idx, raw_score in enumerate(scores):
            score = float(raw_score)
            if score <= 0.0:
                continue
            chunk = self._chunks[idx]
            if filters and not _matches_filters(chunk, filters):
                continue
            scored_indices.append((idx, score))

        scored_indices.sort(key=lambda item: item[1], reverse=True)
        hits: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(scored_indices[:top_k], start=1):
            c = self._chunks[idx]
            hits.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    domain=c.domain,
                    language=c.language,
                    effective_date=c.effective_date,
                    section_path=c.section_path,
                    section_title=c.section_title,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    text=c.text,
                    sparse_score=score,
                    sparse_rank=rank,
                    final_score=score,
                )
            )
        return hits


def _matches_filters(chunk: DocumentChunk, filters: RetrievalFilters) -> bool:
    if filters.domain and chunk.domain.lower() != filters.domain.lower():
        return False
    if filters.language and chunk.language.lower() != filters.language.lower():
        return False
    if filters.doc_id and chunk.doc_id.upper() != filters.doc_id.upper():
        return False
    return True
