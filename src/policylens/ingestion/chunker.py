"""Markdown heading-aware recursive chunker preserving parent titles and char spans."""

import hashlib
import re
from dataclasses import dataclass

from policylens.ingestion.models import DocumentChunk, NormalizedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class _SectionSpan:
    section_path: str
    section_title: str
    start: int
    end: int


@dataclass(frozen=True)
class _TextSpan:
    start: int
    end: int


class MarkdownRecursiveChunker:
    """Split normalized Markdown into heading-scoped overlapping chunks."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: NormalizedDocument) -> list[DocumentChunk]:
        """Split a NormalizedDocument into deterministic DocumentChunk objects."""
        body = doc.body
        if not body.strip():
            return []

        sections = self._extract_sections(body, fallback_title=doc.metadata.title)
        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        for section in sections:
            raw_section = body[section.start : section.end]
            # Trim leading/trailing whitespace while preserving exact offsets in doc.body.
            left_trim = len(raw_section) - len(raw_section.lstrip())
            right_trim = len(raw_section) - len(raw_section.rstrip())
            sec_start = section.start + left_trim
            sec_end = section.end - right_trim
            if sec_start >= sec_end:
                continue

            sub_spans = self._split_span_recursively(body, sec_start, sec_end)
            for span in sub_spans:
                span_text = body[span.start : span.end]
                if not span_text.strip():
                    continue
                chunk_id = f"{doc.metadata.doc_id}#c{chunk_idx:02d}"
                content_hash = hashlib.sha256(span_text.encode("utf-8")).hexdigest()
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.metadata.doc_id,
                        title=doc.metadata.title,
                        domain=doc.metadata.domain,
                        language=doc.metadata.language,
                        effective_date=doc.metadata.effective_date,
                        section_path=section.section_path,
                        section_title=section.section_title,
                        chunk_index=chunk_idx,
                        char_start=span.start,
                        char_end=span.end,
                        content_hash=content_hash,
                        doc_hash=doc.doc_hash,
                        text=span_text,
                    )
                )
                chunk_idx += 1

        return chunks

    def _extract_sections(self, body: str, fallback_title: str) -> list[_SectionSpan]:
        """Group document lines under their nearest Markdown heading hierarchy."""
        heading_stack: list[tuple[int, str]] = []
        sections: list[_SectionSpan] = []

        current_start: int | None = None
        current_has_body = False
        current_path = fallback_title
        current_title = fallback_title

        offset = 0
        lines = body.splitlines(keepends=True)
        for line in lines:
            line_len = len(line)
            stripped = line.rstrip("\r\n")
            heading_match = _HEADING_RE.match(stripped)

            if heading_match:
                if current_start is not None and current_has_body:
                    sections.append(
                        _SectionSpan(
                            section_path=current_path,
                            section_title=current_title,
                            start=current_start,
                            end=offset,
                        )
                    )
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

                current_path = " > ".join(item[1] for item in heading_stack)
                current_title = title
                current_start = offset
                current_has_body = False
            else:
                if current_start is None:
                    current_start = offset
                if stripped.strip():
                    current_has_body = True

            offset += line_len

        if current_start is not None and current_has_body:
            sections.append(
                _SectionSpan(
                    section_path=current_path,
                    section_title=current_title,
                    start=current_start,
                    end=len(body),
                )
            )
        elif not sections and body.strip():
            sections.append(
                _SectionSpan(
                    section_path=fallback_title,
                    section_title=fallback_title,
                    start=0,
                    end=len(body),
                )
            )

        return sections

    def _split_span_recursively(self, body: str, start: int, end: int) -> list[_TextSpan]:
        """Recursively pack paragraphs/lines/sentences into bounded overlapping spans."""
        if end - start <= self.chunk_size:
            return [_TextSpan(start=start, end=end)]

        atomic_units = self._split_into_atomic_units(body, start, end)
        return self._pack_units_with_overlap(body, atomic_units)

    def _split_into_atomic_units(self, body: str, start: int, end: int) -> list[_TextSpan]:
        segment = body[start:end]
        # Prefer paragraph boundaries first.
        units = self._split_by_pattern(segment, start, r"\n\s*\n+")
        if len(units) == 1:
            units = self._split_by_pattern(segment, start, r"\n+")

        refined: list[_TextSpan] = []
        for unit in units:
            if unit.end - unit.start <= self.chunk_size:
                refined.append(unit)
                continue
            sub_segment = body[unit.start : unit.end]
            sent_units = self._split_by_pattern(sub_segment, unit.start, r"(?<=[.!?])\s+")
            for s_unit in sent_units:
                if s_unit.end - s_unit.start <= self.chunk_size:
                    refined.append(s_unit)
                else:
                    refined.extend(self._window_split(body, s_unit.start, s_unit.end))
        return refined

    @staticmethod
    def _split_by_pattern(segment: str, base_offset: int, pattern: str) -> list[_TextSpan]:
        spans: list[_TextSpan] = []
        cursor = 0
        for match in re.finditer(pattern, segment):
            part_start = cursor
            part_end = match.start()
            sub = segment[part_start:part_end]
            if sub.strip():
                l_trim = len(sub) - len(sub.lstrip())
                r_trim = len(sub) - len(sub.rstrip())
                spans.append(
                    _TextSpan(
                        start=base_offset + part_start + l_trim,
                        end=base_offset + part_end - r_trim,
                    )
                )
            cursor = match.end()

        tail = segment[cursor:]
        if tail.strip():
            l_trim = len(tail) - len(tail.lstrip())
            r_trim = len(tail) - len(tail.rstrip())
            spans.append(
                _TextSpan(
                    start=base_offset + cursor + l_trim,
                    end=base_offset + len(segment) - r_trim,
                )
            )
        return spans or [_TextSpan(start=base_offset, end=base_offset + len(segment))]

    def _window_split(self, body: str, start: int, end: int) -> list[_TextSpan]:
        """Fallback word-aware sliding window split when a single sentence exceeds chunk_size."""
        spans: list[_TextSpan] = []
        step_back = min(self.chunk_overlap, self.chunk_size // 2)
        cursor = start

        while cursor < end:
            window_end = min(cursor + self.chunk_size, end)
            if window_end < end:
                space_idx = body.rfind(" ", cursor + self.chunk_size // 2, window_end)
                if space_idx > cursor:
                    window_end = space_idx

            sub = body[cursor:window_end]
            l_trim = len(sub) - len(sub.lstrip())
            r_trim = len(sub) - len(sub.rstrip())
            if cursor + l_trim < window_end - r_trim:
                spans.append(_TextSpan(start=cursor + l_trim, end=window_end - r_trim))

            if window_end >= end:
                break
            next_cursor = max(cursor + 1, window_end - step_back)
            # Snap forward to next non-space start.
            while next_cursor < end and body[next_cursor].isspace():
                next_cursor += 1
            cursor = next_cursor

        return spans

    def _pack_units_with_overlap(self, body: str, units: list[_TextSpan]) -> list[_TextSpan]:
        if not units:
            return []

        packed: list[_TextSpan] = []
        idx = 0
        n = len(units)

        while idx < n:
            chunk_start = units[idx].start
            end_idx = idx
            while end_idx + 1 < n and (units[end_idx + 1].end - chunk_start) <= self.chunk_size:
                end_idx += 1

            chunk_end = units[end_idx].end
            packed.append(_TextSpan(start=chunk_start, end=chunk_end))

            if end_idx >= n - 1:
                break

            # Advance with overlap by keeping trailing units within chunk_overlap of chunk_end.
            next_idx = end_idx + 1
            if self.chunk_overlap > 0:
                candidate = end_idx
                while (
                    candidate > idx
                    and (chunk_end - units[candidate].start) <= self.chunk_overlap
                ):
                    next_idx = candidate
                    candidate -= 1
            idx = max(idx + 1, next_idx)

        return packed
