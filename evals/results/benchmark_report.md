# PolicyLens RAG Evaluation Report

- **Dataset:** `evals/dataset.jsonl` (35 queries: 28 bilingual in-scope + 7 out-of-scope abstention traps)
- **Mode:** Quantized CPU ONNX / local embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) + extractive synthesizer ($0.00 API cost)

| Configuration | HitRate@5 | Recall@5 | MRR@10 | CitationPrecision | AbstentionAcc | p50 (ms) | p95 (ms) | Est. Cloud $/1k req |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense_only` | 0.8214 | 0.8036 | 0.6009 | 0.5536 | 0.6857 | 60.02 | 115.92 | $0.1051 |
| `hybrid_rrf` | 0.9643 | 0.9643 | 0.8152 | 0.8036 | 0.8857 | 93.03 | 170.41 | $0.1509 |
| `hybrid_rrf_rerank` | 1.0000 | 1.0000 | 0.9821 | 0.9583 | 1.0000 | 88.72 | 163.43 | $0.1825 |
