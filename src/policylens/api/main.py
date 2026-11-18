"""FastAPI application entrypoint, lifespan wiring, and health endpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from policylens import __version__
from policylens.api.routes import router as v1_router
from policylens.config import Settings, get_settings
from policylens.generation.llm_client import build_llm_client
from policylens.generation.service import RagPipelineService
from policylens.retrieval.abstention import AbstentionGuard
from policylens.retrieval.bm25 import BM25Index
from policylens.retrieval.embeddings import build_embedding_provider
from policylens.retrieval.hybrid import HybridRetriever
from policylens.retrieval.reranker import build_reranker
from policylens.retrieval.vector_store import QdrantVectorStore

_STUDIO_HTML_PATH = Path(__file__).resolve().parents[1] / "ui" / "static" / "index.html"


class HealthResponse(BaseModel):
    """Health check payload returned by GET /health."""

    status: str
    service: str
    version: str
    qdrant_mode: str
    indexed_chunks: int
    embedding_provider: str
    llm_provider: str


def build_rag_service(settings: Settings) -> tuple[RagPipelineService, QdrantVectorStore]:
    """Construct wired RAG pipeline components and return (service, vector_store)."""
    embedder = build_embedding_provider(settings)
    vector_store = QdrantVectorStore(settings=settings, embedder=embedder)
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=BM25Index(),
        rrf_k=settings.rrf_k,
        top_k_dense=settings.top_k_dense,
        top_k_sparse=settings.top_k_sparse,
    )
    retriever.sync_bm25_from_store()
    reranker = build_reranker(settings)
    abstention_guard = AbstentionGuard(score_threshold=settings.abstention_threshold)
    llm_client = build_llm_client(settings)

    service = RagPipelineService(
        settings=settings,
        retriever=retriever,
        reranker=reranker,
        abstention_guard=abstention_guard,
        llm_client=llm_client,
    )
    return service, vector_store


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance with lifecycle management."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service, store = build_rag_service(resolved_settings)
        app.state.rag_service = service
        app.state.vector_store = store
        try:
            yield
        finally:
            store.close()

    app = FastAPI(
        title="PolicyLens RAG Service",
        version=__version__,
        description=(
            "Bilingual Hybrid RAG & Citation Service with Qdrant, BM25 RRF, "
            "Reranking, and Abstention Guardrails."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.include_router(v1_router)

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def studio_console() -> HTMLResponse:
        """Serve the Swiss Enterprise RAG Studio web console."""
        html = _STUDIO_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html, status_code=200)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        qdrant_mode = (
            f"remote:{resolved_settings.qdrant_url}"
            if resolved_settings.qdrant_url
            else f"local:{resolved_settings.qdrant_path}"
        )
        service: RagPipelineService | None = getattr(request.app.state, "rag_service", None)
        indexed_chunks = service.indexed_chunk_count() if service is not None else 0
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            version=__version__,
            qdrant_mode=qdrant_mode,
            indexed_chunks=indexed_chunks,
            embedding_provider=resolved_settings.embedding_provider,
            llm_provider=resolved_settings.llm_provider,
        )

    return app


app = create_app()
