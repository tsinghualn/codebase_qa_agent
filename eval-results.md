# Eval results

_Generated 2026-11-19 00:30 UTC_
_Corpus pin: `02f2c2e6c992ffccce9e9129781fdd04f015c6fa` · golden cases: answerable subset · top_k=5_

| Mode | Context P@5 | Context R@5 | Top-3 file hit rate |
| --- | --- | --- | --- |
| vector | 0.400 | 0.601 | 85.7% |
| hybrid | 0.400 | 0.601 | 100.0% |
| hybrid+rerank | 0.343 | 0.506 | 71.4% |

## Notes

- **Production default:** `hybrid` (BM25 + vectors + RRF), rerank off unless `RERANK_ENABLED=true`.
- On this code-heavy corpus, MS MARCO cross-encoder rerank often **hurts** top-3 file hit rate vs hybrid alone (see table).