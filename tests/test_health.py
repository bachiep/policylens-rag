"""Smoke tests for application settings, health endpoint, and studio root HTML."""

import pytest
from fastapi.testclient import TestClient

from policylens.api.main import create_app
from policylens.config import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.api_port == 8301
    assert settings.streamlit_port == 8501
    assert settings.embedding_provider == "fastembed"
    assert settings.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.llm_provider == "local"
    assert settings.reranker_provider == "local"


def test_health_endpoint_and_root_console() -> None:
    settings = Settings(_env_file=None, qdrant_path=":memory:", embedding_provider="hash")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["service"] == "policylens-rag"
        assert payload["qdrant_mode"] == "local::memory:"

        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert "text/html" in root_resp.headers["content-type"]
        assert "PolicyLens RAG Studio" in root_resp.text
