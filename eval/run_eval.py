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
        m = score_one(results, q["must_match_any"])
        m.query = q["query"]
        m.latency_ms = latency
        out.append(m)
    return out


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
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        sys.exit("DATABASE_URL is not set")
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
