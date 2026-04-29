"""Background eval runner — drives the offline retrieval-quality eval from
the API instead of the CLI, and persists results to Postgres.

The CLI (eval/run_eval.py) prints a one-shot table to stdout. This module
does the same work but:
  * Persists each run + its per-query rows to eval_runs / eval_run_results
  * Accepts a RankingConfig so admin tweaks land in the swept config
  * Lets a tasked-out request return immediately with a run_id, then poll

Heavy: a full run is 4 pipelines × 25 queries × hybrid+rerank latency,
roughly 30–60 s. Always schedule via BackgroundTasks; never await inline.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import asyncpg

from app import embeddings, rerank, search
from app.logging_setup import get_logger
from app.search import RankingConfig
from eval.metrics import QueryMetrics, score_one, summarize

log = get_logger()

# Resolve queries.json relative to the repo root so the API can find it
# regardless of CWD (uvicorn vs pytest vs Docker).
_QUERIES_PATH = Path(__file__).resolve().parents[1] / "eval" / "queries.json"

K = 10  # results-per-query the metrics are computed against


# -----------------------------------------------------------------------------
# Pipelines (mirror eval/run_eval.py but config-aware)
# -----------------------------------------------------------------------------


def _make_pipelines(config: RankingConfig):
    async def vector(pool, query: str):
        vec = embeddings.embed_query(query)
        return await search.search_by_vector(pool, vec, k=K)

    async def vector_rerank(pool, query: str):
        vec = embeddings.embed_query(query)
        candidates = await search.search_by_vector(pool, vec, k=50)
        return rerank.rerank(query, candidates, top_k=K)

    async def hybrid(pool, query: str):
        vec = embeddings.embed_query(query)
        return await search.hybrid_search(
            pool, query, vec, k_retrieve=50, k_final=K, config=config,
        )

    async def hybrid_rerank(pool, query: str):
        vec = embeddings.embed_query(query)
        candidates = await search.hybrid_search(
            pool, query, vec, k_retrieve=50, k_final=50, config=config,
        )
        return rerank.rerank(query, candidates, top_k=K)

    return {
        "vector":         vector,
        "vector_rerank":  vector_rerank,
        "hybrid":         hybrid,
        "hybrid_rerank":  hybrid_rerank,
    }


def load_curated_queries() -> list[dict]:
    """Load the hand-labeled eval set from eval/queries.json."""
    with open(_QUERIES_PATH) as fh:
        return json.load(fh)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


async def create_run(
    pool: asyncpg.Pool,
    *,
    pipelines: list[str],
    config: RankingConfig,
    n_queries: int,
) -> int:
    """Insert a fresh eval_runs row in 'running' state. Returns the run id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO eval_runs (status, pipelines, n_queries, config)
            VALUES ('running', $1, $2, $3::jsonb)
            RETURNING id
            """,
            pipelines, n_queries, json.dumps(asdict(config)),
        )
    return int(row["id"])


async def _record_results(
    pool: asyncpg.Pool, run_id: int, pipeline: str, metrics: list[QueryMetrics]
) -> None:
    rows = [
        (run_id, pipeline, m.query, m.n_relevant, m.precision_at_k,
         m.recall_at_k, m.rr, m.latency_ms)
        for m in metrics
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO eval_run_results
                (run_id, pipeline, query, n_relevant, precision_at_k,
                 recall_at_k, rr, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            rows,
        )


async def _finalize_run(
    pool: asyncpg.Pool,
    run_id: int,
    *,
    summary_by_pipeline: dict[str, dict[str, float]],
    error: str | None = None,
) -> None:
    status = "error" if error else "done"
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE eval_runs SET
                status      = $2,
                finished_at = NOW(),
                summary     = $3::jsonb,
                error       = $4
            WHERE id = $1
            """,
            run_id, status, json.dumps(summary_by_pipeline), error,
        )


# -----------------------------------------------------------------------------
# Top-level: run all selected pipelines, persist as we go
# -----------------------------------------------------------------------------


async def run_eval(
    pool: asyncpg.Pool,
    *,
    pipelines: list[str],
    queries: list[dict],
    config: RankingConfig,
    run_id: int,
) -> None:
    """Execute the eval and persist results. Catches exceptions so a single
    bad run lands in the DB as status='error' rather than crashing the
    background task and disappearing from the dashboard."""
    summary_by_pipeline: dict[str, dict[str, float]] = {}
    try:
        all_pipelines = _make_pipelines(config)
        for name in pipelines:
            if name not in all_pipelines:
                continue
            metrics: list[QueryMetrics] = []
            for q in queries:
                t0 = time.perf_counter()
                try:
                    results = await all_pipelines[name](pool, q["query"])
                except Exception as exc:
                    log.warning(
                        "eval pipeline failed on a query",
                        extra={"pipeline": name, "query": q["query"], "err": repr(exc)},
                    )
                    results = []
                latency = int((time.perf_counter() - t0) * 1000)
                m = score_one(
                    results,
                    q.get("must_match_any", []),
                    q.get("must_match_ids"),
                )
                m.query = q["query"]
                m.latency_ms = latency
                metrics.append(m)
            await _record_results(pool, run_id, name, metrics)
            summary_by_pipeline[name] = summarize(metrics)
        await _finalize_run(pool, run_id, summary_by_pipeline=summary_by_pipeline)
        log.info(
            "eval run finished",
            extra={"run_id": run_id, "pipelines": pipelines, "n_queries": len(queries)},
        )
    except asyncio.CancelledError:
        await _finalize_run(
            pool, run_id, summary_by_pipeline=summary_by_pipeline,
            error="cancelled",
        )
        raise
    except Exception as exc:
        log.warning(
            "eval run failed",
            extra={"run_id": run_id, "err": repr(exc)},
        )
        await _finalize_run(
            pool, run_id, summary_by_pipeline=summary_by_pipeline,
            error=repr(exc),
        )
