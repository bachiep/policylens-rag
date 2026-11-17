"""Application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for PolicyLens RAG API, storage, retrieval, and UI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application & Ports
    app_env: str = "development"
    app_name: str = "policylens-rag"
    api_host: str = "0.0.0.0"
    api_port: int = 8301
    api_base_url: str = "http://localhost:8301"
    streamlit_port: int = 8501

    # Corpus & Vector Storage
    sample_corpus_dir: Path = Path("data/sample")
    qdrant_path: str = "./qdrant_storage"
    qdrant_url: str | None = None
    qdrant_collection: str = "policylens_chunks"

    # Providers
    embedding_provider: Literal["local", "openai", "gemini", "ollama"] = "local"
    embedding_dim: int = Field(default=384, ge=64, le=4096)
    embedding_model: str = "text-embedding-3-small"

    llm_provider: Literal["local", "openai", "gemini", "ollama"] = "local"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=20.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    reranker_provider: Literal["local", "jina", "cohere"] = "local"
    reranker_api_key: str | None = None
    reranker_model: str = "jina-reranker-v2-base-multilingual"

    # Ingestion & Retrieval Hyperparameters
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    top_k_dense: int = Field(default=15, ge=1, le=50)
    top_k_sparse: int = Field(default=15, ge=1, le=50)
    top_k_final: int = Field(default=4, ge=1, le=15)
    rrf_k: int = Field(default=60, ge=1, le=200)
    abstention_threshold: float = Field(default=0.34, ge=0.0, le=1.0)
    max_query_chars: int = Field(default=1000, ge=50, le=8000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
