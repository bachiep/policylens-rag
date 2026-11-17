"""CLI script to ingest the sample bilingual policy corpus into Qdrant."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from policylens.config import get_settings
from policylens.ingestion.pipeline import IdempotentIngestionPipeline
from policylens.retrieval.embeddings import build_embedding_provider
from policylens.retrieval.vector_store import QdrantVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample policy corpus into Qdrant.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Path to Markdown corpus directory (defaults to settings.sample_corpus_dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if SHA-256 document hashes match.",
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = args.corpus_dir or settings.sample_corpus_dir
    embedder = build_embedding_provider(settings)
    store = QdrantVectorStore(settings=settings, embedder=embedder)

    try:
        pipeline = IdempotentIngestionPipeline(
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        report = pipeline.ingest_directory(corpus_dir, force=args.force)
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
