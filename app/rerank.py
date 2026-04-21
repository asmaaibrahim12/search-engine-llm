"""Cross-encoder reranker.

A bi-encoder (the BGE retriever) independently embeds query and document, so
it never sees them together. A cross-encoder scores (query, document) as a
single input and is strictly better at relevance judgement — at the cost of
needing one forward pass per candidate.

Pattern: retrieve 50 via the retriever, rerank to 10 via this module.
"""
from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


def _pair_text(candidate: dict[str, Any]) -> str:
    """Format a candidate row as the text the reranker sees alongside the query."""
    title = candidate.get("title", "") or ""
    body = candidate.get("body", "") or ""
    # Truncate: cross-encoders are token-bound and long bodies hurt throughput
    # without helping relevance at the top of the list.
    return f"{title}\n{body[:800]}".strip()


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Re-sort `candidates` by cross-encoder relevance to `query`. Returns
    the top `top_k`, each with a `rerank_score` field added.

    Safe to call with an empty candidate list (returns [])."""
    if not candidates:
        return []
    pairs = [(query, _pair_text(c)) for c in candidates]
    scores = get_reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
