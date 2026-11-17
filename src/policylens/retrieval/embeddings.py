"""Embedding provider interfaces, deterministic local embedder, and API adapters."""

import asyncio
import hashlib
import math
import re
import unicodedata
from abc import ABC, abstractmethod

import httpx
import numpy as np

from policylens.config import Settings

# Bilingual EN<->VI domain concept bridges for zero-cost cross-lingual dense matching.
_CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("password", "mật khẩu", "khẩu"),
    ("mfa", "totp", "fido2", "webauthn", "xác thực", "đa yếu tố"),
    ("session", "timeout", "phiên", "hết hạn"),
    ("privilege", "privileged", "pam", "break-glass", "đặc quyền", "khẩn cấp"),
    ("encrypt", "encryption", "aes", "tls", "kms", "mã hóa", "khóa"),
    ("secret", "secrets", "credential", "key", "bí mật"),
    ("incident", "breach", "containment", "forensic", "post-mortem", "sự cố", "cô lập"),
    ("leave", "vacation", "annual", "nghỉ phép", "phép năm", "thâm niên"),
    ("sick", "medical", "doctor", "nghỉ ốm", "y tế", "bệnh"),
    ("remote", "wfh", "hybrid", "từ xa", "trực tuyến"),
    ("probation", "thử việc"),
    ("onboarding", "offboarding", "laptop", "revocation", "tiếp nhận", "nghỉ việc", "thu hồi"),
    ("expense", "reimbursement", "per-diem", "travel", "công tác phí", "hoàn ứng", "lưu trú"),
    ("meal", "hotel", "ăn uống", "khách sạn"),
    ("invoice", "vat", "receipt", "hóa đơn", "chứng từ"),
    ("approval", "approve", "director", "cfo", "manager", "phê duyệt", "duyệt"),
    ("vendor", "procurement", "saas", "quote", "soc", "nhà cung cấp", "mua sắm"),
    ("availability", "uptime", "downtime", "slo", "sla", "khả dụng", "gián đoạn"),
    ("latency", "p95", "độ trễ"),
    ("on-call", "escalation", "pager", "trực ca"),
    ("deploy", "deployment", "canary", "rollback", "release", "triển khai", "quay lui"),
    ("coverage", "review", "branch", "bao phủ", "kiểm thử"),
    ("database", "migration", "schema", "ddl", "lock", "cơ sở dữ liệu", "chuyển đổi"),
    ("chunk", "chunking", "overlap", "nfc", "cắt đoạn", "chồng lấn"),
    ("hallucination", "abstain", "abstention", "citation", "ảo giác", "từ chối", "trích dẫn"),
    ("prompt", "injection", "guardrail", "tiêm lệnh", "lớp chắn"),
    ("pii", "masking", "redaction", "che dấu"),
    ("working", "day", "business", "ngày làm việc"),
    ("error", "budget", "freeze", "ngân sách lỗi", "đóng băng"),
    ("post-mortem", "rca", "root cause", "nguyên nhân gốc"),
    ("certificate", "medical", "xác nhận", "cơ sở y tế"),
    ("day", "days", "ngày"),
    ("carry", "chuyển"),
    ("unused", "chưa sử dụng"),
    ("regular", "chính thức"),
)

CONCEPT_LOOKUP: dict[str, int] = {}
for _cid, _terms in enumerate(_CONCEPT_GROUPS):
    for _term in _terms:
        CONCEPT_LOOKUP[_term] = _cid

POLICY_CODE_RE = re.compile(r"\b[a-z]{2,5}-[a-z]{2,5}-\d{2,4}\b", re.IGNORECASE)
_ALPHA_WORD_RE = re.compile(r"[a-zA-ZÀ-ỹ]+", re.UNICODE)


def strip_accents(text: str) -> str:
    """Fold Vietnamese and Latin diacritics for accent-tolerant subword matching."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def extract_concept_ids(text: str) -> set[int]:
    """Return bilingual domain concept IDs mentioned in text."""
    nfc = unicodedata.normalize("NFC", text).lower()
    return {cid for term, cid in CONCEPT_LOOKUP.items() if term in nfc}


def _signed_bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    bucket = val % dim
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    return bucket, sign


class EmbeddingProvider(ABC):
    """Abstract interface for dense text embedding providers."""

    dimension: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passage texts."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous wrapper used by ingestion scripts."""
        return asyncio.run(self.embed_documents(texts))


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic zero-cost semantic-subword hashed embedding provider."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _encode_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        nfc = unicodedata.normalize("NFC", text).lower()

        # Remove rigid hyphenated policy codes from dense hash features so dense vectors
        # represent topical semantics while BM25 handles exact policy codes cleanly.
        prose = POLICY_CODE_RE.sub(" ", nfc)
        fold = strip_accents(prose)

        feature_counts: dict[str, float] = {}

        # 1. Bilingual domain concept projections
        for concept_id in extract_concept_ids(prose):
            key = f"concept:{concept_id}"
            feature_counts[key] = feature_counts.get(key, 0.0) + 2.5

        # 2. Folded words & character 3/4-grams on meaningful alphabetic words
        for word in _ALPHA_WORD_RE.findall(fold):
            if len(word) < 3:
                continue
            w_key = f"w:{word}"
            feature_counts[w_key] = feature_counts.get(w_key, 0.0) + 1.2
            padded = f"<{word}>"
            for n in (3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        gram = f"g{n}:{padded[i : i + n]}"
                        feature_counts[gram] = feature_counts.get(gram, 0.0) + 0.45

        for feat, weight in feature_counts.items():
            bucket, sign = _signed_bucket(feat, self.dimension)
            vec[bucket] += sign * math.log1p(weight)

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.astype(float).tolist()

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents_sync(texts)

    async def embed_query(self, text: str) -> list[float]:
        return self._encode_one(text)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Async HTTP embedding adapter supporting OpenAI, Groq, and Ollama `/embeddings`."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dimension: int,
        timeout: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "input": texts}
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()["data"]
                    sorted_items = sorted(data, key=lambda item: item["index"])
                    return [item["embedding"] for item in sorted_items]
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Embedding API request failed after retries: {last_error}")

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Async HTTP adapter for Gemini text embedding API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )
        results: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                payload = {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self.dimension,
                }
                last_error: Exception | None = None
                for attempt in range(self.max_retries + 1):
                    try:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        results.append(resp.json()["embedding"]["values"])
                        last_error = None
                        break
                    except (httpx.HTTPError, KeyError, ValueError) as exc:
                        last_error = exc
                        if attempt < self.max_retries:
                            await asyncio.sleep(0.5 * (2**attempt))
                if last_error is not None:
                    raise RuntimeError(f"Gemini embedding request failed: {last_error}")
        return results

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Instantiate configured embedding provider with safe local fallback."""
    if settings.embedding_provider == "openai" and settings.llm_api_key:
        base_url = settings.llm_base_url or "https://api.openai.com/v1"
        return OpenAICompatibleEmbeddingProvider(
            base_url=base_url,
            api_key=settings.llm_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.embedding_provider == "ollama" and settings.llm_base_url:
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.embedding_provider == "gemini" and settings.gemini_api_key:
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return LocalHashEmbeddingProvider(dimension=settings.embedding_dim)
