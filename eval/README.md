# Retrieval eval harness

A small tool for measuring how changes to the search pipeline affect result
quality, so "swap the embedder" or "add a reranker" decisions are data-driven.

## Prerequisites

- `DATABASE_URL` set to the **public** Postgres URL of a populated DB
- Python deps from the app's `requirements.txt` installed (`sentence-transformers`,
  `asyncpg`, `pgvector`, etc.)
- The DB must have been indexed with the current embedder — re-run
  `scripts/index_outdoors.py` after changing the embedder or the index scheme

## Run

```bash
export DATABASE_URL='postgresql://postgres:...@<host>:<port>/railway'
python eval/run_eval.py
```

Runs all four pipelines (vector only / +rerank / hybrid / hybrid+rerank) against
the 25 queries in `queries.json` and prints a side-by-side comparison:

```
Pipeline              Recall@10   Prec@10     MRR   Mean ms   P95 ms
----------------------------------------------------------------
vector                   0.920     0.412   0.712       142      310
vector_rerank            0.960     0.496   0.861       378      620
hybrid                   0.960     0.456   0.793       168      350
hybrid_rerank            1.000     0.512   0.904       402      680
```

To run one pipeline only:

```bash
python eval/run_eval.py --pipeline hybrid_rerank
```

## Metrics

- **Recall@10** — fraction of queries where at least one relevant result
  surfaced in the top 10. This is the "did the user see anything useful?"
  metric.
- **Precision@10** — average fraction of the top 10 that is relevant. Higher
  means less noise in the results list.
- **MRR** — mean reciprocal rank of the first relevant result. Directly
  correlates with UX: if MRR = 1, the top hit is always relevant; if 0.5, the
  first relevant hit is at position 2 on average; etc.

## The query set

`queries.json` has 25 hand-crafted queries plus `must_match_any` keyword
patterns. A result counts as relevant if its title OR body contains any of the
listed tokens (case-insensitive substring). This is weaker than true manual
labeling of `(query, relevant_doc_id)` pairs, but scales for free across
re-indexings and model swaps.

To add queries:

```json
{
  "query": "your question here",
  "must_match_any": ["keyword1", "keyword2", "phrase"],
  "must_not_match": []
}
```

Keep `must_match_any` broad enough to catch paraphrases of the expected answer
topic. Too narrow and you'll under-count real wins; too broad and you'll
over-count irrelevant surface-level matches.

## Workflow

1. Establish a baseline — run the current pipeline and save the numbers.
2. Make a change (new embedder, tune `k_retrieve`, different reranker).
3. Re-index if the change affects embeddings: `python scripts/index_outdoors.py`.
4. Re-run the eval. Keep the change if MRR or Recall@10 improved without
   P95 latency blowing up.

## What this tool deliberately does NOT do

- **No online A/B testing.** Use click logs and dwell time for that; this
  tool is for offline regression.
- **No nDCG.** Needs graded relevance labels (perfect / good / okay). We only
  have binary. Add it if you start labeling by hand.
- **No per-query automated assertion.** Tests in `tests/test_eval.py` only
  cover the metric math, not individual query outcomes — retrieval quality
  is an aggregate measurement, not a unit test.
