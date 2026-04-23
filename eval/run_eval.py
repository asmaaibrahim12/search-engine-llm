"""Retrieval quality eval harness.

Runs a labeled query set against one of the search pipelines and reports
recall@10, precision@10, and MRR. The eval uses keyword-based relevance
judgements (a hit is "relevant" if its title OR body contains any token
from the query's `must_match_any` list) which is weaker than hand-labeled
doc IDs but scales to re-indexing and model swaps without re-labeling.

Usage:

    # Run all four pipelines (vector / hybrid / +rerank) side by side
    python eval/run_eval.py

    # Only one pipeline
    python eval/run_eval.py --pipeline hybrid_rerank

    # Different query set (must be same JSON shape)
    python eval/run_eval.py --queries eval/my_queries.json

DATABASE_URL must point at a populated Railway Postgres (public URL).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, rerank, search  # noqa: E402
from app.db import create_pool, ensure_schema  # noqa: E402
from eval.metrics import QueryMetrics, score_one, summarize  # noqa: E402

load_dotenv()

K = 10


# -----------------------------------------------------------------------------
# Pipelines under test
# -----------------------------------------------------------------------------


async def pipeline_vector(pool, query: str) -> list[dict]:
    vec = embeddings.embed_query(query)
    return await search.search_by_vector(pool, vec, k=K)


async def pipeline_vector_rerank(pool, query: str) -> list[dict]:
    vec = embeddings.embed_query(query)
    candidates = await search.search_by_vector(pool, vec, k=50)
    return rerank.rerank(query, candidates, top_k=K)


async def pipeline_hybrid(pool, query: str) -> list[dict]:
    vec = embeddings.embed_query(query)
    return await search.hybrid_search(pool, query, vec, k_retrieve=50, k_final=K)


async def pipeline_hybrid_rerank(pool, query: str) -> list[dict]:
    vec = embeddings.embed_query(query)
    candidates = await search.hybrid_search(
        pool, query, vec, k_retrieve=50, k_final=50
    )
    return rerank.rerank(query, candidates, top_k=K)


PIPELINES: dict[str, Callable[[Any, str], Awaitable[list[dict]]]] = {
    "vector": pipeline_vector,
    "vector_rerank": pipeline_vector_rerank,
    "hybrid": pipeline_hybrid,
    "hybrid_rerank": pipeline_hybrid_rerank,
}


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


async def run_pipeline(
    name: str,
    pipeline: Callable[[Any, str], Awaitable[list[dict]]],
    queries: list[dict],
    pool,
) -> list[QueryMetrics]:
    out: list[QueryMetrics] = []
    for q in queries:
        t0 = time.perf_counter()
        results = await pipeline(pool, q["query"])
        latency = int((time.perf_counter() - t0) * 1000)
        m = score_one(
            results,
            q.get("must_match_any", []),
            q.get("must_match_ids"),
        )
        m.query = q["query"]
        m.latency_ms = latency
        out.append(m)
    return out


async def load_feedback_queries(
    pool, min_thumbs_up: int = 1
) -> list[dict]:
    """Build eval queries from production feedback.

    Aggregates thumbs_up events from search_events into one query per
    unique (case-normalized) query string, with the thumbed-up result ids
    as must_match_ids. The min_thumbs_up threshold filters out single-vote
    noise; bump it up once there's enough traffic.

    Skips queries with no thumbs_up hits entirely — they'd be unscorable
    without labels.
    """
    sql = """
    SELECT lower(trim(query)) AS query_norm,
           result_id,
           COUNT(*) AS votes
    FROM search_events
    WHERE event_type = 'thumb_up'
      AND result_id IS NOT NULL
      AND query IS NOT NULL
      AND length(trim(query)) > 0
    GROUP BY 1, 2
    HAVING COUNT(*) >= $1
    ORDER BY 1, 3 DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, min_thumbs_up)
    by_query: dict[str, list[int]] = {}
    for row in rows:
        by_query.setdefault(row["query_norm"], []).append(int(row["result_id"]))
    return [
        {"query": q, "must_match_any": [], "must_match_ids": ids}
        for q, ids in by_query.items()
    ]


def print_report(results_by_pipeline: dict[str, list[QueryMetrics]]) -> None:
    print()
    print(
        f"{'Pipeline':<20} {'Recall@10':>10} {'Prec@10':>9} "
        f"{'MRR':>7} {'Mean ms':>9} {'P95 ms':>8}"
    )
    print("-" * 68)
    for name, metrics in results_by_pipeline.items():
        s = summarize(metrics)
        print(
            f"{name:<20} {s['recall_at_k']:>10.3f} {s['precision_at_k']:>9.3f} "
            f"{s['mrr']:>7.3f} {s['mean_latency_ms']:>9.0f} {s['p95_latency_ms']:>8.0f}"
        )

    # Per-query breakdown of the best pipeline so the user can see weaknesses
    print()
    best = max(
        results_by_pipeline.items(),
        key=lambda kv: summarize(kv[1]).get("mrr", 0.0),
    )
    print(f"Per-query breakdown for best pipeline ({best[0]}):")
    print(f"{'rr':>6}  {'p@10':>6}  {'ms':>5}  query")
    for m in sorted(best[1], key=lambda m: m.rr):
        print(f"{m.rr:>6.3f}  {m.precision_at_k:>6.3f}  {m.latency_ms:>5}  {m.query}")


async def main_async(args) -> int:
    queries: list[dict] = []
    if not args.feedback_only:
        with open(args.queries) as fh:
            queries = json.load(fh)
        print(f"Loaded {len(queries)} queries from {args.queries}", flush=True)

    # Warm the models once
    embeddings.get_model()
    if any("rerank" in p for p in (args.pipeline or PIPELINES)):
        rerank.get_reranker()

    pool = await create_pool()
    await ensure_schema(pool)
    try:
        if args.augment_from_feedback or args.feedback_only:
            fb = await load_feedback_queries(pool, min_thumbs_up=args.min_thumbs_up)
            print(
                f"Loaded {len(fb)} queries from search_events "
                f"(>= {args.min_thumbs_up} thumbs_up)",
                flush=True,
            )
            queries.extend(fb)
        if not queries:
            print("No queries to run — exiting.", flush=True)
            return 0
        pipelines_to_run = (
            [args.pipeline] if args.pipeline else list(PIPELINES.keys())
        )
        out: dict[str, list[QueryMetrics]] = {}
        for name in pipelines_to_run:
            print(f"Running pipeline: {name}", flush=True)
            out[name] = await run_pipeline(name, PIPELINES[name], queries, pool)
        print_report(out)
    finally:
        await pool.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).parent / "queries.json"),
    )
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINES.keys()),
        default=None,
        help="Run only this pipeline. Default: all four.",
    )
    parser.add_argument(
        "--augment-from-feedback",
        action="store_true",
        help="Also pull queries + positive labels from search_events (thumbs_up).",
    )
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Ignore queries.json; evaluate ONLY on queries derived from "
             "production thumbs_up events.",
    )
    parser.add_argument(
        "--min-thumbs-up",
        type=int,
        default=1,
        help="Minimum thumbs_up count per (query, result_id) to count as a "
             "positive label. Raise this as traffic grows.",
    )
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        sys.exit("DATABASE_URL is not set")
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
