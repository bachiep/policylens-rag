"""Pure HTTP client helpers and Streamlit Citation Inspector UI (port 8501)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_SAMPLE_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "FIN-POL-03 (VI Exact Code + Per-Diem)",
        "FIN-POL-03 quy định mức phụ cấp ăn uống công tác nội địa và quốc tế là bao nhiêu?",
    ),
    (
        "SEC-POL-04 (VI IAM Session Timeout)",
        "SEC-POL-04 quy định session timeout cho cổng quản trị và khóa tài khoản là bao lâu?",
    ),
    (
        "OPS-SLA-02 (EN Production SLOs)",
        "What are the Tier-0 monthly availability SLO and on-call ack SLA in OPS-SLA-02?",
    ),
    (
        "HR-POL-01 (Cross-Lingual Annual Leave)",
        "How many days of annual leave do regular employees receive and carry over past March 31?",
    ),
    (
        "Out-of-Scope Trap (Abstention Guard)",
        "What is the company cryptocurrency Bitcoin treasury investment policy?",
    ),
)


def parse_sse_line_events(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse SSE event/data pairs from a streaming text iterator."""
    current_event = "message"
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            raw_data = line.removeprefix("data:").strip()
            try:
                yield current_event, json.loads(raw_data)
            except json.JSONDecodeError:
                yield current_event, {"raw": raw_data}


def fetch_health(api_base_url: str) -> dict[str, Any]:
    """Fetch GET /health from FastAPI service."""
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{api_base_url.rstrip('/')}/health")
        resp.raise_for_status()
        return dict(resp.json())


def fetch_documents(api_base_url: str) -> list[dict[str, Any]]:
    """Fetch GET /v1/documents metadata list."""
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{api_base_url.rstrip('/')}/v1/documents")
        resp.raise_for_status()
        return list(resp.json())


def query_rest(
    api_base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute synchronous POST /v1/query request."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{api_base_url.rstrip('/')}/v1/query", json=payload)
        resp.raise_for_status()
        return dict(resp.json())


def run_streamlit_app() -> None:
    """Render the Streamlit Citation Inspector UI."""
    import streamlit as st

    st.set_page_config(
        page_title="PolicyLens RAG — Citation Inspector",
        page_icon="🔍",
        layout="wide",
    )

    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8301")

    st.title("PolicyLens RAG — Bilingual Citation Inspector")
    st.caption(
        "Hybrid RAG (Qdrant Dense + BM25 RRF + Reranker) with Abstention Guardrails "
        "and Verbatim Source Span Verification"
    )

    with st.sidebar:
        st.header("Pipeline Controls")
        api_base_url = st.text_input("FastAPI Base URL", value=api_base_url)

        try:
            health = fetch_health(api_base_url)
            st.success(
                f"API Online • {health['indexed_chunks']} chunks • "
                f"LLM: {health['llm_provider']}"
            )
        except httpx.HTTPError as exc:
            st.error(f"API unreachable at {api_base_url}: {exc}")

        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=["hybrid_rrf_rerank", "hybrid_rrf", "dense_only"],
            index=0,
        )
        top_k = st.slider("Top-K Context Chunks", min_value=1, max_value=8, value=4)
        domain = st.selectbox(
            "Domain Filter",
            options=["all", "security", "hr", "finance", "engineering"],
            index=0,
        )
        language = st.selectbox("Language Filter", options=["all", "vi", "en"], index=0)
        use_streaming = st.checkbox("Stream via SSE (/v1/query/stream)", value=True)

        st.divider()
        st.subheader("Sample Queries")
        selected_preset = st.selectbox(
            "Load preset",
            options=[label for label, _ in DEFAULT_SAMPLE_QUERIES],
            index=0,
        )
        if st.button("Apply Preset Query", use_container_width=True):
            preset_map = dict(DEFAULT_SAMPLE_QUERIES)
            st.session_state["query_input"] = preset_map[selected_preset]

    default_q = st.session_state.get("query_input", DEFAULT_SAMPLE_QUERIES[0][1])
    query_text = st.text_area("Ask a policy or engineering SOP question (VI / EN)", value=default_q)

    if st.button("Run Grounded Query", type="primary"):
        filters: dict[str, str] = {}
        if domain != "all":
            filters["domain"] = domain
        if language != "all":
            filters["language"] = language

        payload: dict[str, Any] = {
            "query": query_text,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "filters": filters or None,
        }

        try:
            final_data: dict[str, Any] | None = None
            if use_streaming:
                status_box = st.empty()
                stream_box = st.empty()
                collected_tokens: list[str] = []
                with httpx.Client(timeout=30.0) as client:
                    with client.stream(
                        "POST",
                        f"{api_base_url.rstrip('/')}/v1/query/stream",
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        for event_name, event_payload in parse_sse_line_events(resp.iter_lines()):
                            if event_name == "retrieval":
                                status_box.info(
                                    f"Retrieved {event_payload.get('candidate_count', 0)} chunks • "
                                    f"Confidence: {event_payload.get('confidence_score', 0.0):.2f}"
                                )
                            elif event_name == "token":
                                collected_tokens.append(str(event_payload.get("delta", "")))
                                stream_box.markdown("".join(collected_tokens))
                            elif event_name == "final":
                                final_data = event_payload
                status_box.empty()
                stream_box.empty()
            else:
                with st.spinner("Retrieving and verifying citations..."):
                    final_data = query_rest(api_base_url, payload)

            if final_data:
                _render_grounded_answer(final_data)
        except httpx.HTTPError as exc:
            st.error(f"Query failed: {exc}")


def _render_grounded_answer(data: dict[str, Any]) -> None:
    import streamlit as st

    if data.get("is_abstained"):
        st.warning(
            f"**Abstention Guard Triggered** — {data.get('abstention_reason') or 'Low confidence'}"
        )
    else:
        st.success("Grounded Answer Verified")

    st.markdown(data.get("answer", ""))

    usage = data.get("usage", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{data.get('confidence_score', 0.0):.2f}")
    c2.metric("Latency", f"{data.get('latency_ms', 0.0):.1f} ms")
    c3.metric("Total Tokens", str(usage.get("total_tokens", 0)))
    c4.metric("Est. Cost (USD)", f"${usage.get('estimated_cost_usd', 0.0):.6f}")

    citations = data.get("citations", [])
    st.subheader(f"Citation Inspector ({len(citations)})")
    if not citations:
        st.info("No citations attached (abstained response).")
        return

    for idx, cite in enumerate(citations, start=1):
        header = (
            f"[{idx}] {cite['chunk_id']} — {cite['title']} "
            f"(score={cite['score']:.3f}, chars={cite['char_start']}..{cite['char_end']})"
        )
        with st.expander(header, expanded=idx == 1):
            st.caption(f"Section Path: `{cite['section_path']}`")
            st.markdown(f"> {cite['quote']}")


if __name__ == "__main__":
    run_streamlit_app()
