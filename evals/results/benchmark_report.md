# PolicyLens RAG Evaluation Report

- **Dataset:** `evals/dataset.jsonl` (35 queries: 28 bilingual in-scope + 7 out-of-scope abstention traps)
- **Mode:** Offline deterministic local embeddings + extractive synthesizer ($0.00 API cost)

| Configuration | HitRate@5 | Recall@5 | MRR@10 | CitationPrecision | AbstentionAcc | p50 (ms) | p95 (ms) | Est. Cloud $/1k req |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense_only` | 1.0000 | 0.9643 | 0.9345 | 0.8393 | 0.9143 | 4.81 | 6.70 | $0.1577 |
| `hybrid_rrf` | 1.0000 | 1.0000 | 0.9583 | 0.9107 | 0.9714 | 6.41 | 9.01 | $0.1731 |
| `hybrid_rrf_rerank` | 1.0000 | 1.0000 | 0.9821 | 0.9583 | 1.0000 | 27.70 | 35.93 | $0.1778 |
