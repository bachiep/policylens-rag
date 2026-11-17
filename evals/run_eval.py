"""Automated evaluation harness comparing dense_only, hybrid_rrf, and hybrid_rrf_rerank."""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from policylens.config import Settings
from policylens.generation.llm_client import LocalExtractiveLLMClient
from policylens.generation.schemas import QueryRequest
from policylens.generation.service import RagPipelineService
from policylens.ingestion.pipeline import IdempotentIngestionPipeline
from policylens.prompts.rag_qa_v1 import estimate_cost_usd
from policylens.retrieval.abstention import AbstentionGuard
from policylens.retrieval.bm25 import BM25Index
from policylens.retrieval.embeddings import LocalHashEmbeddingProvider
from policylens.retrieval.hybrid import HybridRetriever
from policylens.retrieval.reranker import LocalCrossScorerReranker
from policylens.retrieval.vector_store import QdrantVectorStore

EvalMode = Literal["dense_only", "hybrid_rrf", "hybrid_rrf_rerank"]


@dataclass(frozen=True)
class EvalSample:
    """Single golden benchmark query loaded from evals/dataset.jsonl."""

    id: str
    category: str
    language: str
    query: str
    is_unanswerable: bool
    expected_doc_ids: list[str]
    expected_chunk_ids: list[str]
    expected_keywords: list[str]


@dataclass(frozen=True)
class ModeBenchmarkMetrics:
    """Aggregated retrieval, faithfulness, safety, latency, and cost metrics for one mode."""

    mode: str
    total_queries: int
    answerable_queries: int
    trap_queries: int
    hit_rate_at_5: float
    recall_at_5: float
    mrr_at_10: float
    citation_precision: float
    keyword_recall: float
    abstention_accuracy: float
    trap_abstention_recall: float
    false_abstention_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_total_tokens: float
    est_cloud_cost_per_1k_usd: float


def load_eval_dataset(path: Path = Path("evals/dataset.jsonl")) -> list[EvalSample]:
    """Load JSONL evaluation samples from disk."""
    samples: list[EvalSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        samples.append(
            EvalSample(
                id=str(item["id"]),
                category=str(item["category"]),
                language=str(item["language"]),
                query=str(item["query"]),
                is_unanswerable=bool(item["is_unanswerable"]),
                expected_doc_ids=list(item.get("expected_doc_ids", [])),
                expected_chunk_ids=list(item.get("expected_chunk_ids", [])),
                expected_keywords=list(item.get("expected_keywords", [])),
            )
        )
    return samples


async def _retrieve_top_10(
    service: RagPipelineService,
    query: str,
    mode: EvalMode,
) -> list[str]:
    if mode == "dense_only":
        hits = await service.retriever.retrieve(query, mode="dense_only", top_k=10)
    elif mode == "hybrid_rrf":
        hits = await service.retriever.retrieve(query, mode="hybrid_rrf", top_k=10)
    else:
        fused = await service.retriever.retrieve(query, mode="hybrid_rrf", top_k=15)
        hits = await service.reranker.rerank(query, fused, top_k=10)
    return [c.chunk_id for c in hits]


async def _evaluate_mode(
    service: RagPipelineService,
    samples: list[EvalSample],
    mode: EvalMode,
) -> ModeBenchmarkMetrics:
    hits_at_5: list[float] = []
    recalls_at_5: list[float] = []
    mrrs_at_10: list[float] = []
    citation_precisions: list[float] = []
    keyword_recalls: list[float] = []
    abstention_correct = 0
    trap_correct = 0
    false_abstentions = 0
    latencies_ms: list[float] = []
    total_tokens_list: list[int] = []
    cloud_costs: list[float] = []

    answerable_samples = [s for s in samples if not s.is_unanswerable]
    trap_samples = [s for s in samples if s.is_unanswerable]

    for sample in samples:
        started = time.perf_counter()
        top_10_ids = await _retrieve_top_10(service, sample.query, mode)
        answer = await service.answer_query(
            QueryRequest(query=sample.query, top_k=4, retrieval_mode=mode)
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)
        total_tokens_list.append(answer.usage.total_tokens)
        cloud_costs.append(
            estimate_cost_usd(
                answer.usage.prompt_tokens,
                answer.usage.completion_tokens,
                model_name="gpt-4o-mini",
            )
        )

        if sample.is_unanswerable:
            if answer.is_abstained:
                abstention_correct += 1
                trap_correct += 1
            continue

        # Answerable query retrieval metrics
        expected_set = set(sample.expected_chunk_ids)
        top_5_ids = top_10_ids[:5]
        matched_5 = len(expected_set & set(top_5_ids))
        hits_at_5.append(1.0 if matched_5 > 0 else 0.0)
        recalls_at_5.append(matched_5 / max(1, len(expected_set)))

        rr = 0.0
        for rank, cid in enumerate(top_10_ids, start=1):
            if cid in expected_set:
                rr = 1.0 / rank
                break
        mrrs_at_10.append(rr)

        if answer.is_abstained:
            false_abstentions += 1
            citation_precisions.append(0.0)
            keyword_recalls.append(0.0)
        else:
            abstention_correct += 1
            if answer.citations:
                valid_cites = sum(
                    1
                    for cite in answer.citations
                    if cite.chunk_id in expected_set or cite.doc_id in sample.expected_doc_ids
                )
                citation_precisions.append(valid_cites / len(answer.citations))
            else:
                citation_precisions.append(0.0)

            combined_evidence = (
                answer.answer + " " + " ".join(c.quote for c in answer.citations)
            ).lower()
            if sample.expected_keywords:
                kw_hits = sum(
                    1 for kw in sample.expected_keywords if kw.lower() in combined_evidence
                )
                keyword_recalls.append(kw_hits / len(sample.expected_keywords))
            else:
                keyword_recalls.append(1.0)

    n_ans = max(1, len(answerable_samples))
    n_trap = max(1, len(trap_samples))
    n_total = max(1, len(samples))

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    avg_cost_per_1k = float(np.mean(cloud_costs)) * 1000.0

    return ModeBenchmarkMetrics(
        mode=mode,
        total_queries=len(samples),
        answerable_queries=len(answerable_samples),
        trap_queries=len(trap_samples),
        hit_rate_at_5=round(float(np.mean(hits_at_5)), 4),
        recall_at_5=round(float(np.mean(recalls_at_5)), 4),
        mrr_at_10=round(float(np.mean(mrrs_at_10)), 4),
        citation_precision=round(float(np.mean(citation_precisions)), 4),
        keyword_recall=round(float(np.mean(keyword_recalls)), 4),
        abstention_accuracy=round(abstention_correct / n_total, 4),
        trap_abstention_recall=round(trap_correct / n_trap, 4),
        false_abstention_rate=round(false_abstentions / n_ans, 4),
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        avg_total_tokens=round(float(np.mean(total_tokens_list)), 1),
        est_cloud_cost_per_1k_usd=round(avg_cost_per_1k, 4),
    )


async def run_benchmark(
    dataset_path: Path = Path("evals/dataset.jsonl"),
    corpus_dir: Path = Path("data/sample"),
    output_dir: Path | None = Path("evals/results"),
) -> list[ModeBenchmarkMetrics]:
    """Run all 3 retrieval configurations against the benchmark dataset in memory."""
    samples = load_eval_dataset(dataset_path)
    settings = Settings(_env_file=None, qdrant_path=":memory:", embedding_dim=384)
    embedder = LocalHashEmbeddingProvider(dimension=384)
    store = QdrantVectorStore(settings=settings, embedder=embedder)

    try:
        pipeline = IdempotentIngestionPipeline(
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        pipeline.ingest_directory(corpus_dir)

        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            bm25_index=BM25Index(),
            rrf_k=settings.rrf_k,
            top_k_dense=settings.top_k_dense,
            top_k_sparse=settings.top_k_sparse,
        )
        retriever.sync_bm25_from_store()

        service = RagPipelineService(
            settings=settings,
            retriever=retriever,
            reranker=LocalCrossScorerReranker(),
            abstention_guard=AbstentionGuard(score_threshold=settings.abstention_threshold),
            llm_client=LocalExtractiveLLMClient(),
        )

        modes: tuple[EvalMode, ...] = ("dense_only", "hybrid_rrf", "hybrid_rrf_rerank")
        results: list[ModeBenchmarkMetrics] = []
        for mode in modes:
            metrics = await _evaluate_mode(service, samples, mode)
            results.append(metrics)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / "benchmark_summary.json"
            md_path = output_dir / "benchmark_report.md"

            json_path.write_text(
                json.dumps([asdict(r) for r in results], indent=2),
                encoding="utf-8",
            )
            md_path.write_text(format_markdown_report(results), encoding="utf-8")

        return results
    finally:
        store.close()


def format_markdown_report(results: list[ModeBenchmarkMetrics]) -> str:
    """Format benchmark results into a reproducible Markdown comparison table."""
    lines = [
        "# PolicyLens RAG Evaluation Report",
        "",
        (
            f"- **Dataset:** `evals/dataset.jsonl` ({results[0].total_queries} queries: "
            f"{results[0].answerable_queries} bilingual in-scope + "
            f"{results[0].trap_queries} out-of-scope abstention traps)"
        ),
        (
            "- **Mode:** Offline deterministic local embeddings + extractive synthesizer "
            "($0.00 API cost)"
        ),
        "",
        (
            "| Configuration | HitRate@5 | Recall@5 | MRR@10 | CitationPrecision | "
            "AbstentionAcc | p50 (ms) | p95 (ms) | Est. Cloud $/1k req |"
        ),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.mode}` | {r.hit_rate_at_5:.4f} | {r.recall_at_5:.4f} | "
            f"{r.mrr_at_10:.4f} | {r.citation_precision:.4f} | "
            f"{r.abstention_accuracy:.4f} | {r.p50_latency_ms:.2f} | "
            f"{r.p95_latency_ms:.2f} | ${r.est_cloud_cost_per_1k_usd:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = asyncio.run(run_benchmark())
    print(format_markdown_report(results))


if __name__ == "__main__":
    main()
