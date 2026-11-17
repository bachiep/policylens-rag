"""FastAPI application entrypoint and health endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel

from policylens import __version__
from policylens.config import Settings, get_settings


class HealthResponse(BaseModel):
    """Health check payload returned by GET /health."""

    status: str
    service: str
    version: str
    qdrant_mode: str
    embedding_provider: str
    llm_provider: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="PolicyLens RAG Service",
        version=__version__,
        description=(
            "Bilingual Hybrid RAG & Citation Service with Qdrant, BM25 RRF, "
            "Reranking, and Abstention Guardrails."
        ),
    )
    app.state.settings = resolved_settings

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        qdrant_mode = (
            f"remote:{resolved_settings.qdrant_url}"
            if resolved_settings.qdrant_url
            else f"local:{resolved_settings.qdrant_path}"
        )
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            version=__version__,
            qdrant_mode=qdrant_mode,
            embedding_provider=resolved_settings.embedding_provider,
            llm_provider=resolved_settings.llm_provider,
        )

    return app


app = create_app()
