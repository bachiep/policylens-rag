"""LLM client abstractions with local extractive fallback and async HTTP adapters."""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from policylens.config import Settings
from policylens.prompts.rag_qa_v1 import build_rag_messages, estimate_tokens
from policylens.retrieval.bm25 import tokenize_for_bm25
from policylens.retrieval.embeddings import extract_concept_ids
from policylens.retrieval.models import RetrievedChunk
from policylens.retrieval.reranker import detect_query_language, extract_distinctive_terms

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9*`\"'(\[])")
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^\s*>\s*")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")


def _sanitize_markdown_balance(text: str) -> str:
    """Remove unmatched Markdown bold/code markers so answers never leak dangling `**`."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    # Validate `**` pairs using CommonMark flanking whitespace rules.
    matches = list(re.finditer(r"\*\*", cleaned))
    if matches:
        stray_spans: list[tuple[int, int]] = []
        open_match: re.Match[str] | None = None
        for m in matches:
            start, end = m.span()
            prev_ch = cleaned[start - 1] if start > 0 else " "
            next_ch = cleaned[end] if end < len(cleaned) else " "
            can_open = not next_ch.isspace() and next_ch != "*"
            can_close = not prev_ch.isspace() and prev_ch != "*"

            if open_match is None:
                if can_open:
                    open_match = m
                else:
                    stray_spans.append((start, end))
            elif can_close:
                open_match = None
            else:
                stray_spans.append(open_match.span())
                if can_open:
                    open_match = m
                else:
                    stray_spans.append((start, end))
                    open_match = None

        if open_match is not None:
            stray_spans.append(open_match.span())

        if stray_spans:
            parts: list[str] = []
            cursor = 0
            for s_start, s_end in sorted(stray_spans):
                parts.append(cleaned[cursor:s_start])
                cursor = s_end
            parts.append(cleaned[cursor:])
            cleaned = "".join(parts)

    # Strip unmatched inline code ticks or stray single asterisks.
    if cleaned.count("`") % 2 == 1:
        cleaned = cleaned.replace("`", "")

    without_bold = cleaned.replace("**", "")
    if without_bold.count("*") % 2 == 1:
        cleaned = re.sub(r"(?<!\*)\*(?!\*)", "", cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_clean_sentences(chunk_text: str) -> list[str]:
    """Extract clean, Markdown-balanced sentences from a chunk without heading/bullet noise."""
    sentences: list[str] = []
    for raw_line in chunk_text.splitlines():
        stripped = raw_line.strip()
        if (
            not stripped
            or _HEADING_LINE_RE.match(stripped)
            or _TABLE_SEP_RE.match(stripped)
            or _HR_RE.match(stripped)
        ):
            continue
        body_line = _BLOCKQUOTE_PREFIX_RE.sub("", stripped)
        body_line = _LIST_PREFIX_RE.sub("", body_line).strip()
        if not body_line or _HEADING_LINE_RE.match(body_line):
            continue

        for raw_sent in _SENTENCE_SPLIT_RE.split(body_line):
            clean_sent = _sanitize_markdown_balance(raw_sent)
            if len(clean_sent) >= 20:
                sentences.append(clean_sent)
    return sentences


@dataclass(frozen=True)
class LLMGenerationResult:
    """Raw structured generation result before citation span verification."""

    answer: str
    cited_chunk_ids: list[str]
    prompt_tokens: int
    completion_tokens: int
    model_name: str


class BaseLLMClient(ABC):
    """Abstract interface for grounded answer generation."""

    model_name: str

    @abstractmethod
    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> LLMGenerationResult:
        """Generate a grounded answer citing chunk IDs from chunks."""


def _score_sentence(sentence: str, query: str, chunk: RetrievedChunk) -> float:
    distinctive = extract_distinctive_terms(query)
    if not distinctive:
        distinctive = tokenize_for_bm25(query)
    if not distinctive:
        return 0.0

    sent_lower = sentence.lower()
    sent_tokens = set(tokenize_for_bm25(sent_lower))
    sent_concepts = extract_concept_ids(sent_lower)

    hits = 0.0
    for term in distinctive:
        if term in sent_tokens or term in sent_lower:
            hits += 1.0
        else:
            t_concepts = extract_concept_ids(term)
            if t_concepts and (t_concepts & sent_concepts):
                hits += 0.8

    # Prefer sentences carrying concrete numbers or policy specifics over pure headings.
    specificity_bonus = 0.15 if re.search(r"\d", sentence) else 0.0
    chunk_bonus = max(chunk.rerank_score, chunk.dense_score) * 0.25
    return (hits / len(distinctive)) + specificity_bonus + chunk_bonus


class LocalExtractiveLLMClient(BaseLLMClient):
    """Deterministic $0 extractive synthesizer for offline mode, CI, and fallback."""

    def __init__(self, model_name: str = "local-extractive-v1") -> None:
        self.model_name = model_name

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> LLMGenerationResult:
        messages = build_rag_messages(query, chunks)
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        candidates: list[tuple[float, str, RetrievedChunk]] = []
        for rank_idx, chunk in enumerate(chunks):
            for sentence in _extract_clean_sentences(chunk.text):
                score = _score_sentence(sentence, query, chunk) - (rank_idx * 0.04)
                candidates.append((score, sentence, chunk))

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected_lines: list[str] = []
        cited_ids: list[str] = []
        seen_text: set[str] = set()

        for _, sentence, chunk in candidates:
            if sentence in seen_text:
                continue
            seen_text.add(sentence)
            selected_lines.append(f"- {sentence} [{chunk.chunk_id}]")
            if chunk.chunk_id not in cited_ids:
                cited_ids.append(chunk.chunk_id)
            if len(selected_lines) >= 3:
                break

        if not selected_lines and chunks:
            top = chunks[0]
            fallback_lines = [
                _sanitize_markdown_balance(_LIST_PREFIX_RE.sub("", ln.strip()))
                for ln in top.text.strip().splitlines()
                if ln.strip() and not _HEADING_LINE_RE.match(ln.strip())
            ]
            snippet = next((ln for ln in fallback_lines if ln), "")
            if not snippet:
                snippet = _sanitize_markdown_balance(
                    _HEADING_LINE_RE.sub("", top.text.strip().splitlines()[-1].strip())
                )
            selected_lines.append(f"- {snippet} [{top.chunk_id}]")
            cited_ids.append(top.chunk_id)

        lang = detect_query_language(query)
        lead = (
            "Theo quy định và tài liệu nội bộ liên quan:"
            if lang == "vi"
            else "Based on the indexed internal policies:"
        )
        answer = f"{lead}\n" + "\n".join(selected_lines)
        completion_tokens = estimate_tokens(answer)

        return LLMGenerationResult(
            answer=answer,
            cited_chunk_ids=cited_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=self.model_name,
        )


def _parse_json_answer(
    raw_content: str,
    fallback_chunks: list[RetrievedChunk],
) -> tuple[str, list[str]]:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(cleaned)
        answer = str(parsed.get("answer", "")).strip()
        raw_ids = parsed.get("cited_chunk_ids", [])
        cited = [str(cid).strip() for cid in raw_ids if isinstance(cid, str) and cid.strip()]
        if answer:
            return answer, cited
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Fallback: extract bracketed chunk IDs directly from plain text response.
    valid_ids = {c.chunk_id for c in fallback_chunks}
    found = [cid for cid in re.findall(r"\[([^\[\]]+)\]", cleaned) if cid in valid_ids]
    return cleaned, list(dict.fromkeys(found))


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Async OpenAI-compatible chat completion client (OpenAI / Groq / Ollama)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model_name: str,
        timeout: float = 20.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> LLMGenerationResult:
        messages = build_rag_messages(query, chunks)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    answer, cited_ids = _parse_json_answer(raw_text, chunks)
                    prompt_tokens = int(
                        usage.get("prompt_tokens")
                        or sum(estimate_tokens(m["content"]) for m in messages)
                    )
                    completion_tokens = int(
                        usage.get("completion_tokens") or estimate_tokens(answer)
                    )
                    return LLMGenerationResult(
                        answer=answer,
                        cited_chunk_ids=cited_ids,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model_name=self.model_name,
                    )
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"LLM generation failed after retries: {last_error}")


class GeminiLLMClient(BaseLLMClient):
    """Async Gemini API client with structured JSON output and retries."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        timeout: float = 20.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> LLMGenerationResult:
        messages = build_rag_messages(query, chunks)
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_msg}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    meta = data.get("usageMetadata", {})
                    answer, cited_ids = _parse_json_answer(raw_text, chunks)
                    prompt_tokens = int(
                        meta.get("promptTokenCount")
                        or sum(estimate_tokens(m["content"]) for m in messages)
                    )
                    completion_tokens = int(
                        meta.get("candidatesTokenCount") or estimate_tokens(answer)
                    )
                    return LLMGenerationResult(
                        answer=answer,
                        cited_chunk_ids=cited_ids,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model_name=self.model_name,
                    )
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"Gemini generation failed after retries: {last_error}")


def build_llm_client(settings: Settings) -> BaseLLMClient:
    """Instantiate configured LLM client with zero-cost local fallback."""
    if settings.llm_provider == "openai" and settings.llm_api_key:
        base_url = settings.llm_base_url or "https://api.openai.com/v1"
        return OpenAICompatibleLLMClient(
            base_url=base_url,
            api_key=settings.llm_api_key,
            model_name=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider == "ollama" and settings.llm_base_url:
        return OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model_name=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return GeminiLLMClient(
            api_key=settings.gemini_api_key,
            model_name=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return LocalExtractiveLLMClient()
