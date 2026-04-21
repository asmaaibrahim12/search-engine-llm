from __future__ import annotations

import pytest

from app import rerank
from tests.conftest import needs_model


def test_rerank_empty_list_returns_empty():
    assert rerank.rerank("anything", [], top_k=10) == []


def test_rerank_returns_at_most_top_k():
    candidates = [
        {"title": f"doc {i}", "body": "sample body"} for i in range(20)
    ]

    class _FakeModel:
        # Deterministic fake scores: later docs score higher
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    rerank._reranker = _FakeModel()  # type: ignore[assignment]
    try:
        out = rerank.rerank("query", candidates, top_k=5)
        assert len(out) == 5
        # Top result should have the highest score (doc 19, score 19.0)
        assert out[0]["title"] == "doc 19"
        assert out[0]["rerank_score"] == 19.0
        # Sorted in descending order
        scores = [c["rerank_score"] for c in out]
        assert scores == sorted(scores, reverse=True)
    finally:
        rerank._reranker = None


def test_rerank_preserves_original_fields():
    candidates = [
        {"id": 1, "title": "A", "body": "a", "score": 0.9},
        {"id": 2, "title": "B", "body": "b", "score": 0.8},
    ]

    class _FakeModel:
        def predict(self, pairs):
            # Reverse the order: second candidate ranks higher
            return [0.1, 0.9]

    rerank._reranker = _FakeModel()  # type: ignore[assignment]
    try:
        out = rerank.rerank("q", candidates, top_k=2)
        # Original fields carried through
        assert out[0]["id"] == 2
        assert out[0]["score"] == 0.8  # original retrieval score preserved
        assert "rerank_score" in out[0]
    finally:
        rerank._reranker = None


def test_pair_text_handles_missing_body():
    text = rerank._pair_text({"title": "T", "body": None})
    assert "T" in text
    # No crash, body falls back to empty string


def test_pair_text_truncates_long_body():
    long_body = "x" * 5000
    text = rerank._pair_text({"title": "T", "body": long_body})
    # We cap body at 800 chars — the whole output is title + "\n" + body[:800]
    assert len(text) < 900


@needs_model
def test_rerank_gives_sensible_order_on_real_model():
    """End-to-end sanity with the actual cross-encoder loaded.

    Gate on RUN_MODEL_TESTS=1 — loading the reranker takes ~10s."""
    candidates = [
        {"title": "How to change a kayak paddle grip", "body": "..."},
        {"title": "Best trail running shoes for rocky terrain", "body": "..."},
        {"title": "Tent guy-line tensioning in high wind", "body": "..."},
    ]
    out = rerank.rerank("hiking footwear for uneven trails", candidates, top_k=3)
    # The trail running shoes title should be ranked #1 by any sensible model.
    assert "shoes" in out[0]["title"].lower() or "running" in out[0]["title"].lower()