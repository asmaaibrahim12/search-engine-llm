"""Dep-free helpers for the eval harness.

Kept separate from run_eval.py so unit tests can import these without
triggering imports of sentence-transformers, asyncpg, dotenv, etc.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class QueryMetrics:
    query: str
    n_relevant: int        # # of relevant hits in top-K
    precision_at_k: float  # n_relevant / K
    recall_at_k: float     # 1.0 if any relevant hit in top-K, else 0.0
    rr: float              # reciprocal rank of first relevant; 0 if none
    latency_ms: int


def is_relevant(result: dict, must_match_any: list[str]) -> bool:
    """A hit counts as relevant if its title or body contains any of the
    required tokens (case-insensitive substring match)."""
    if not must_match_any:
        return False
    haystack = (
        (result.get("title") or "") + " " + (result.get("body") or "")
    ).lower()
    return any(tok.lower() in haystack for tok in must_match_any)


def score_one(results: list[dict], must_match_any: list[str]) -> QueryMetrics:
    flags = [is_relevant(r, must_match_any) for r in results]
    n_relevant = sum(flags)
    first_rel = next((i + 1 for i, f in enumerate(flags) if f), None)
    return QueryMetrics(
        query="",
        n_relevant=n_relevant,
        precision_at_k=n_relevant / len(results) if results else 0.0,
        # Recall@K with "at least one relevant exists" assumption — the right
        # proxy for a question-answering search UX where the user needs ONE
        # good result, not all of them.
        recall_at_k=1.0 if n_relevant > 0 else 0.0,
        rr=(1.0 / first_rel) if first_rel else 0.0,
        latency_ms=0,
    )


def summarize(metrics: list[QueryMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    return {
        "recall_at_k": statistics.mean(m.recall_at_k for m in metrics),
        "precision_at_k": statistics.mean(m.precision_at_k for m in metrics),
        "mrr": statistics.mean(m.rr for m in metrics),
        "mean_latency_ms": statistics.mean(m.latency_ms for m in metrics),
        "p95_latency_ms": (
            statistics.quantiles([m.latency_ms for m in metrics], n=20)[18]
            if len(metrics) >= 20
            else max(m.latency_ms for m in metrics)
        ),
    }
