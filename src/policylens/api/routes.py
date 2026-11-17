"""HTTP routes for grounded policy querying, SSE streaming, and corpus inspection."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from policylens.generation.schemas import DocumentSummary, GroundedAnswer, QueryRequest
from policylens.generation.service import InputGuardrailError, RagPipelineService

router = APIRouter(prefix="/v1", tags=["rag"])


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
