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

RUN pip install --upgrade pip && pip install ".[ui]"

EXPOSE 8301 8501

CMD ["uvicorn", "policylens.api.main:app", "--host", "0.0.0.0", "--port", "8301"]
