"""Tests validating the 35-query evaluation dataset and automated benchmark runner."""

from pathlib import Path

from evals.run_eval import load_eval_dataset, run_benchmark

from policylens.config import Settings
from policylens.ingestion.chunker import MarkdownRecursiveChunker
from policylens.ingestion.pipeline import load_corpus_documents


def test_eval_dataset_schema_and_ground_truth_ids() -> None:
    samples = load_eval_dataset(Path("evals/dataset.jsonl"))
    assert len(samples) == 35

    traps = [s for s in samples if s.is_unanswerable]
    answerable = [s for s in samples if not s.is_unanswerable]
    assert len(traps) == 7
    assert len(answerable) == 28

    settings = Settings(_env_file=None)
    chunker = MarkdownRecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    all_chunk_ids = {
        c.chunk_id
        for doc in load_corpus_documents(Path("data/sample"))
        for c in chunker.chunk_document(doc)
    }

    for sample in answerable:
        assert sample.expected_chunk_ids, f"{sample.id} must declare expected_chunk_ids"
        for cid in sample.expected_chunk_ids:
            assert cid in all_chunk_ids, f"Unknown ground truth chunk {cid} in {sample.id}"


async def test_eval_harness_meets_release_gates_and_beats_dense_baseline() -> None:
    results = await run_benchmark(
        dataset_path=Path("evals/dataset.jsonl"),
        corpus_dir=Path("data/sample"),
        output_dir=None,
    )
    by_mode = {item.mode: item for item in results}

    dense = by_mode["dense_only"]
    hybrid = by_mode["hybrid_rrf"]
    reranked = by_mode["hybrid_rrf_rerank"]

    assert hybrid.hit_rate_at_5 >= dense.hit_rate_at_5
    assert reranked.mrr_at_10 >= hybrid.mrr_at_10
    assert reranked.hit_rate_at_5 >= 0.85
    assert reranked.abstention_accuracy >= 0.90
    assert reranked.trap_abstention_recall == 1.0
