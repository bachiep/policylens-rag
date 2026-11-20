FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts
COPY evals ./evals

RUN pip install --upgrade pip && pip install --no-cache-dir ".[ui]" \
    && groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser \
    && mkdir -p /app/qdrant_storage \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8301 8501

CMD ["sh", "-c", "if [ \"${INGEST_SAMPLE_ON_BOOT:-false}\" = \"true\" ]; then python -m scripts.ingest_sample; fi && uvicorn policylens.api.main:app --host 0.0.0.0 --port ${PORT:-8301}"]
