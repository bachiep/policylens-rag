"""Unit tests for Streamlit Citation Inspector HTTP and SSE parser helpers."""

import json

from policylens.ui.app import DEFAULT_SAMPLE_QUERIES, parse_sse_line_events


def test_default_sample_queries_cover_bilingual_and_trap_cases() -> None:
    assert len(DEFAULT_SAMPLE_QUERIES) == 5
    queries = [q for _, q in DEFAULT_SAMPLE_QUERIES]
    assert any("FIN-POL-03" in q for q in queries)
    assert any("Bitcoin" in q for q in queries)


def test_parse_sse_line_events_parses_retrieval_token_and_final() -> None:
    raw_lines = iter(
        [
            "event: retrieval",
            'data: {"candidate_count": 4, "confidence_score": 0.88}',
            "",
            "event: token",
            'data: {"delta": "Hello "}',
            "",
            "event: final",
            "data: " + json.dumps({"answer": "Hello world", "is_abstained": False}),
            "",
        ]
    )
    events = list(parse_sse_line_events(raw_lines))
    assert events[0] == ("retrieval", {"candidate_count": 4, "confidence_score": 0.88})
    assert events[1] == ("token", {"delta": "Hello "})
    assert events[2] == ("final", {"answer": "Hello world", "is_abstained": False})
