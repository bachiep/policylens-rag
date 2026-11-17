"""Unicode NFC normalization and Markdown frontmatter parsing."""

import hashlib
import re
import unicodedata
from pathlib import Path

from policylens.ingestion.models import DocumentMetadata, NormalizedDocument

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalize text to NFC Unicode and clean line breaks/trailing spaces."""
    nfc = unicodedata.normalize("NFC", text)
    unified = nfc.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [line.rstrip() for line in unified.split("\n")]
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
    return collapsed.strip()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw_header = match.group(1)
    body = text[match.end() :]
    meta: dict[str, str] = {}
    for line in raw_header.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, body


def _infer_language(text: str) -> str:
    vi_markers = ("quy định", "chính sách", "nhân viên", "tối đa", "phê duyệt", "bắt buộc")
    lower = text.lower()
    return "vi" if any(marker in lower for marker in vi_markers) else "en"


def parse_markdown_document(raw_text: str, source_path: str = "<memory>") -> NormalizedDocument:
    """Parse raw Markdown content into a NormalizedDocument with SHA-256 hash."""
    normalized_raw = normalize_unicode_and_whitespace(raw_text)
    header, body_raw = _parse_frontmatter(normalized_raw)
    body = normalize_unicode_and_whitespace(body_raw)

    stem = Path(source_path).stem if source_path != "<memory>" else "DOC-UNKNOWN"
    h1_match = _H1_RE.search(body)
    inferred_title = h1_match.group(1).strip() if h1_match else stem

    doc_id = header.get("doc_id") or stem.upper()
    title = header.get("title") or inferred_title
    domain = header.get("domain") or "general"
    language = header.get("language") or _infer_language(body)
    effective_date = header.get("effective_date") or "unknown"
    owner = header.get("owner") or "unknown"

    doc_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    metadata = DocumentMetadata(
        doc_id=doc_id,
        title=title,
        domain=domain,
        language=language,
        effective_date=effective_date,
        owner=owner,
        source_path=source_path,
    )
    return NormalizedDocument(metadata=metadata, body=body, doc_hash=doc_hash)
