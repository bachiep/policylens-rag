# Contributing

Thank you for your interest in contributing to **PolicyLens RAG**.

## Development Setup

```bash
# Prerequisites: Python 3.13+, Docker (for Qdrant integration tests)
git clone https://github.com/bachiep/policylens-rag.git
cd policylens-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
EMBEDDING_PROVIDER=hash pytest -q   # run test suite with deterministic embeddings
```

## Workflow

1. **Open an Issue** describing the bug or feature.
2. **Create a branch** from `main`: `feat/issue-<n>-<slug>` or `fix/issue-<n>-<slug>`.
3. **Write tests first** — especially for retrieval logic and guardrails.
4. **Follow Conventional Commits**: `feat(retrieval): add cross-encoder reranking (#10)`.
5. **Ensure CI passes**: `ruff check . && EMBEDDING_PROVIDER=hash pytest -q`.
6. **Open a Pull Request** referencing the issue: `Closes #<n>`.

## Code Standards

- Python 3.13, async FastAPI, Pydantic v2 strict models.
- Ruff linter — zero warnings.
- `EMBEDDING_PROVIDER=hash` for fast deterministic CI; real embeddings via ONNX `fastembed`.
- Two-layer abstention guardrails: relevance score threshold + hallucination check.

## Docker Smoke Tests

```bash
docker compose up -d --build qdrant api
# Wait for health, then test grounding + abstention + A/B compare
docker compose down -v
```

## Evaluation

```bash
EMBEDDING_PROVIDER=hash python -m evals.run_eval   # local deterministic baseline
LLM_PROVIDER=gemini python -m evals.run_eval        # live Cloud LLM (requires API key)
```