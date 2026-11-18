"""HTTP routes for grounded policy querying, SSE streaming, ingestion, and eval inspection."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from policylens.generation.schemas import (
    DocumentChunksResponse,
    DocumentSummary,
    DocumentUploadRequest,
    DocumentUploadResponse,
    GroundedAnswer,
    LatestEvalsResponse,
    QueryRequest,
)
from policylens.generation.service import InputGuardrailError, RagPipelineService

router = APIRouter(prefix="/v1", tags=["rag"])
_DEFAULT_EVALS_DIR = Path(__file__).resolve().parents[3] / "evals" / "results"


def _get_service(request: Request) -> RagPipelineService:
    service: RagPipelineService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is not initialized.")
    return service


@router.get("/documents", response_model=list[DocumentSummary])
def list_indexed_documents(request: Request) -> list[DocumentSummary]:
    """List metadata for all documents indexed in the vector store."""
    service = _get_service(request)
    return service.list_documents()


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    payload: DocumentUploadRequest,
    request: Request,
) -> DocumentUploadResponse:
    """Normalize, chunk, embed, and idempotently index a Markdown policy document."""
    service = _get_service(request)
    return await service.upload_document(payload)


@router.get("/documents/{doc_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(doc_id: str, request: Request) -> DocumentChunksResponse:
    """Return ordered chunks and span-aligned document text for source highlighting."""
    service = _get_service(request)
    result = service.get_document_chunks(doc_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return result


@router.get("/evals/latest", response_model=LatestEvalsResponse)
def get_latest_evals(request: Request) -> LatestEvalsResponse:
    """Return latest retrieval/abstention benchmark summary and per-query diagnostics."""
    results_dir = Path(getattr(request.app.state, "evals_results_dir", _DEFAULT_EVALS_DIR))
    summary_path = results_dir / "benchmark_summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Latest benchmark summary not found. Run `python -m evals.run_eval` first.",
        )

    raw_summary: Any = json.loads(summary_path.read_text(encoding="utf-8"))
    modes: list[dict[str, Any]]
    metadata: dict[str, Any] = {}
    if isinstance(raw_summary, list):
        modes = [dict(item) for item in raw_summary]
    elif isinstance(raw_summary, dict):
        modes = [dict(item) for item in raw_summary.get("modes", [])]
        metadata = dict(raw_summary.get("metadata", {}))
    else:
        modes = []

    meta_path = results_dir / "eval_metadata.json"
    if meta_path.exists():
        metadata.update(json.loads(meta_path.read_text(encoding="utf-8")))

    per_query_path = results_dir / "per_query_results.json"
    per_query: list[dict[str, Any]] = []
    if per_query_path.exists():
        loaded = json.loads(per_query_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            per_query = [dict(item) for item in loaded]

    return LatestEvalsResponse(modes=modes, per_query=per_query, metadata=metadata)


@router.post("/query", response_model=GroundedAnswer)
async def query_policies(payload: QueryRequest, request: Request) -> GroundedAnswer:
    """Answer a policy question with verified citations and abstention guardrails."""
    service = _get_service(request)
    try:
        return await service.answer_query(payload)
    except InputGuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query/stream")
async def stream_policy_query(payload: QueryRequest, request: Request) -> StreamingResponse:
    """Stream retrieval signals, answer deltas, and final GroundedAnswer over SSE."""
    service = _get_service(request)
    try:
        service.validate_query(payload.query)
    except InputGuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        service.stream_query(payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
