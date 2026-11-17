"""Smoke tests for application settings and health endpoint."""

from fastapi.testclient import TestClient

from policylens.api.main import create_app
from policylens.config import Settings


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_port == 8301
    assert settings.streamlit_port == 8501
    assert settings.embedding_provider == "local"
    assert settings.llm_provider == "local"
    assert settings.reranker_provider == "local"


def test_health_endpoint() -> None:
    settings = Settings(_env_file=None, qdrant_path=":memory:")
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "policylens-rag"
    assert payload["qdrant_mode"] == "local::memory:"
