"""Tests for the eval module's pure-logic helpers.

The metric math and the relevance judge don't need a database or model —
they're just functions. We cover them here. The end-to-end pipelines
(running against Postgres + the real encoder) are exercised by
`python eval/run_eval.py` against a seeded Railway DB; we don't
automate that in CI because it needs public network + model weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import QueryMetrics, is_relevant, score_one, summarize  # noqa: E402


# -----------------------------------------------------------------------------
# is_relevant
# -----------------------------------------------------------------------------


def test_is_relevant_matches_title_token():
    r = {"title": "Best hiking boots for wide feet", "body": None}
    assert is_relevant(r, ["boot", "sleeping bag"])


def test_is_relevant_matches_body_token():
    r = {"title": "unrelated", "body": "instructions for how to boil water"}
    assert is_relevant(r, ["boil"])


def test_is_relevant_case_insensitive():
    r = {"title": "HIKING BOOTS", "body": ""}
    assert is_relevant(r, ["boot"])


def test_is_relevant_handles_none_body():
    r = {"title": "boots", "body": None}
    assert is_relevant(r, ["boot"])


def test_is_relevant_returns_false_for_no_overlap():
    r = {"title": "snow tires", "body": "chains and studs"}
    assert not is_relevant(r, ["hammer", "nail"])


def test_is_relevant_empty_must_match_returns_false():
    r = {"title": "anything", "body": "anything"}
    assert not is_relevant(r, [])


# -----------------------------------------------------------------------------
# score_one — single-query metric builder
# -----------------------------------------------------------------------------


def _row(title: str) -> dict:
    return {"title": title, "body": None}


def test_score_one_all_relevant():
    results = [_row("boot A"), _row("boot B"), _row("boot C")]
    m = score_one(results, ["boot"])
    assert m.n_relevant == 3
    assert m.precision_at_k == 1.0
    assert m.recall_at_k == 1.0
    assert m.rr == 1.0  # first result relevant


def test_score_one_none_relevant():
    results = [_row("x"), _row("y"), _row("z")]
    m = score_one(results, ["boot"])
    assert m.n_relevant == 0
    assert m.precision_at_k == 0.0
    assert m.recall_at_k == 0.0
    assert m.rr == 0.0


def test_score_one_rr_is_reciprocal_of_first_relevant_rank():
    # First relevant at position 3 -> rr = 1/3
    results = [_row("junk"), _row("junk"), _row("boots"), _row("more junk")]
    m = score_one(results, ["boot"])
    assert abs(m.rr - 1 / 3) < 1e-9
    assert m.n_relevant == 1


def test_score_one_precision_partial():
    results = [_row("boots"), _row("junk"), _row("boots"), _row("junk")]
    m = score_one(results, ["boot"])
    assert m.n_relevant == 2
    assert m.precision_at_k == 0.5
    assert m.recall_at_k == 1.0  # at-least-one semantics
    assert m.rr == 1.0  # first result hit


def test_score_one_empty_results():
    m = score_one([], ["anything"])
    assert m.n_relevant == 0
    assert m.precision_at_k == 0.0
    assert m.recall_at_k == 0.0
    assert m.rr == 0.0


# -----------------------------------------------------------------------------
# summarize — aggregates across queries
# -----------------------------------------------------------------------------


def _metric(rr: float, prec: float, recall: float, ms: int) -> QueryMetrics:
    return QueryMetrics(
        query="q",
        n_relevant=0,
        precision_at_k=prec,
        recall_at_k=recall,
        rr=rr,
        latency_ms=ms,
    )


def test_summarize_empty_returns_empty_dict():
    assert summarize([]) == {}


def test_summarize_computes_means():
    ms = [
        _metric(rr=1.0, prec=0.5, recall=1.0, ms=100),
        _metric(rr=0.5, prec=0.2, recall=1.0, ms=200),
        _metric(rr=0.0, prec=0.0, recall=0.0, ms=150),
    ]
    s = summarize(ms)
    assert abs(s["mrr"] - 0.5) < 1e-9
    assert abs(s["precision_at_k"] - 0.7 / 3) < 1e-9
    assert abs(s["recall_at_k"] - 2 / 3) < 1e-9
    assert abs(s["mean_latency_ms"] - 150.0) < 1e-9


def test_summarize_p95_for_small_sample_is_max():
    ms = [_metric(1.0, 0.5, 1.0, ms=v) for v in [10, 20, 30]]
    s = summarize(ms)
    assert s["p95_latency_ms"] == 30


# -----------------------------------------------------------------------------
# queries.json — sanity-check the shipped eval set
# -----------------------------------------------------------------------------


import json as _json


def test_queries_json_is_wellformed():
    path = Path(__file__).resolve().parents[1] / "eval" / "queries.json"
    data = _json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) >= 20  # we shipped at least 20
    for q in data:
        assert isinstance(q["query"], str) and q["query"].strip()
        assert isinstance(q["must_match_any"], list) and q["must_match_any"]
        assert isinstance(q.get("must_not_match", []), list)