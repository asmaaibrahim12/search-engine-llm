"""Pure-Python tests for the RankingConfig dataclass + SQL rendering.

No DB or model — just validates the dataclass contract and that the SQL
builder substitutes each field into the template as expected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The search module imports asyncpg / numpy at the top, neither of which
# is available in the lightweight pure-Python test env. Stub them before
# importing app.search.
import types
for _mod in ("asyncpg", "numpy"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
_emb = types.ModuleType("app.embeddings")
_emb.embed_query = lambda q: [0.0] * 768  # type: ignore[attr-defined]
sys.modules.setdefault("app.embeddings", _emb)

from app.search import RankingConfig, DEFAULT_CONFIG, _build_hybrid_sql  # noqa: E402


def test_default_config_has_documented_values():
    c = DEFAULT_CONFIG
    assert c.rrf_k == 60
    assert c.accepted_bump == 0.005
    assert c.upvote_coeff == 0.002
    assert c.ctr_coeff == 0.010
    assert c.ctr_shrinkage_floor == 20
    assert c.thumb_coeff == 0.003


def test_config_rejects_negative_values():
    with pytest.raises(ValueError):
        RankingConfig(rrf_k=-1)
    with pytest.raises(ValueError):
        RankingConfig(ctr_coeff=-0.001)


def test_config_rejects_non_numeric():
    with pytest.raises(TypeError):
        RankingConfig(rrf_k="60")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RankingConfig(accepted_bump=True)  # type: ignore[arg-type]


def test_config_is_hashable_and_frozen():
    c = RankingConfig()
    with pytest.raises(Exception):
        c.rrf_k = 40  # type: ignore[misc]
    # Dataclass frozen=True means instances hash; different instances with
    # the same fields compare equal and share a hash.
    assert hash(RankingConfig()) == hash(RankingConfig())


def test_build_hybrid_sql_substitutes_default_constants():
    sql = _build_hybrid_sql(DEFAULT_CONFIG, filtered=False)
    assert "1.0 / (60 + v.rnk)" in sql
    assert "THEN 0.005 ELSE 0" in sql
    assert "0.002 * ln(1 + GREATEST(o.score, 0))" in sql
    assert "0.01 * LEAST" in sql  # python prints 0.010 as 0.01
    assert "0), 20)" in sql
    assert "0.003 * (" in sql


def test_build_hybrid_sql_substitutes_custom_constants():
    cfg = RankingConfig(
        rrf_k=40, accepted_bump=0.01, upvote_coeff=0.003, ctr_coeff=0.05,
        ctr_shrinkage_floor=10, thumb_coeff=0.005,
    )
    sql = _build_hybrid_sql(cfg, filtered=False)
    assert "1.0 / (40 + v.rnk)" in sql
    assert "THEN 0.01 ELSE 0" in sql
    assert "0.003 * ln" in sql
    assert "0.05 * LEAST" in sql
    assert ", 10)" in sql
    assert "0.005 * (" in sql


def test_build_hybrid_sql_caches_per_config_and_filter():
    """LRU cache keyed on (config, filtered) — same args = same object."""
    sql_a = _build_hybrid_sql(DEFAULT_CONFIG, filtered=False)
    sql_b = _build_hybrid_sql(DEFAULT_CONFIG, filtered=False)
    assert sql_a is sql_b
    # Different filter flag → different string (must contain the filtered
    # CTE reference that the unfiltered path doesn't).
    filtered = _build_hybrid_sql(DEFAULT_CONFIG, filtered=True)
    assert filtered is not sql_a
    assert "WITH filtered AS" in filtered
    assert "WITH filtered AS" not in sql_a
