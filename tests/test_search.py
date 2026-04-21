from __future__ import annotations

import numpy as np
import pytest

from app import search
from app.db import create_pool, ensure_schema
from app.embeddings import EMBEDDING_DIM
from tests.conftest import TEST_DATABASE_URL, needs_db, needs_model

pytestmark = [pytest.mark.asyncio, needs_db, needs_model]


@pytest.fixture
async def pool():
    p = await create_pool(TEST_DATABASE_URL)
    await ensure_schema(p)
    async with p.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
    yield p
    await p.close()


async def _seed(pool, rows: list[dict]):
    from app.embeddings import embed_batch

    titles = [r["title"] for r in rows]
    embeddings = embed_batch(titles)
    records = [
        (r["id"], r["title"], r.get("body", ""), np.array(embeddings[i], dtype=np.float32))
        for i, r in enumerate(rows)
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO outdoors (id, title, body, content_embedding) VALUES ($1, $2, $3, $4)",
            records,
        )


async def test_semantic_search_empty_db(pool):
    results = await search.semantic_search(pool, "anything", k=10)
    assert results == []


async def test_semantic_search_returns_k(pool):
    await _seed(pool, [
        {"id": 1, "title": "minimalist shoes"},
        {"id": 2, "title": "winter hiking boots"},
        {"id": 3, "title": "climbing ropes"},
        {"id": 4, "title": "tent waterproofing"},
        {"id": 5, "title": "water purification"},
    ])
    results = await search.semantic_search(pool, "footwear", k=3)
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_semantic_search_matches_known_concept(pool):
    await _seed(pool, [
        {"id": 1, "title": "What are minimalist shoes?", "body": "thin soles..."},
        {"id": 2, "title": "How to purify water", "body": "boiling..."},
        {"id": 3, "title": "Tent pole repairs", "body": "field fixes..."},
    ])
    results = await search.semantic_search(pool, "minimal shoes for running", k=3)
    assert results[0]["id"] == 1


async def test_semantic_search_score_in_valid_range(pool):
    await _seed(pool, [{"id": 1, "title": "test query"}])
    results = await search.semantic_search(pool, "completely different topic", k=1)
    assert -1.0 <= results[0]["score"] <= 1.0


async def test_semantic_search_returns_expected_fields(pool):
    await _seed(pool, [{"id": 42, "title": "sample", "body": "sample body"}])
    results = await search.semantic_search(pool, "sample", k=1)
    assert set(results[0].keys()) == {"id", "title", "body", "score"}
    assert results[0]["id"] == 42
    assert results[0]["title"] == "sample"
    assert results[0]["body"] == "sample body"


# -----------------------------------------------------------------------------
# Hybrid (vector + BM25 + RRF)
# -----------------------------------------------------------------------------


async def test_hybrid_search_empty_db(pool):
    from app.embeddings import embed_query

    vec = embed_query("anything")
    results = await search.hybrid_search(pool, "anything", vec, k_retrieve=10, k_final=10)
    assert results == []


async def test_hybrid_search_finds_exact_keyword_match(pool):
    """Exact rare token in the query should surface even when the embedding
    wouldn't rank it first. This is the whole point of the BM25 half."""
    from app.embeddings import embed_query

    await _seed(pool, [
        {"id": 1, "title": "Salomon X Ultra 4 GTX review", "body": "trail shoe"},
        {"id": 2, "title": "General trail shoe guide", "body": "various brands"},
        {"id": 3, "title": "Hiking boots 101", "body": "basics"},
    ])
    vec = embed_query("Salomon X Ultra 4")
    results = await search.hybrid_search(
        pool, "Salomon X Ultra 4", vec, k_retrieve=10, k_final=10
    )
    assert results[0]["id"] == 1
    # The top hit should be caught by BOTH retrieval halves (or at least
    # by the keyword half, which catches the exact model name)
    assert results[0]["keyword_rank"] is not None


async def test_hybrid_search_rrf_score_shape(pool):
    """RRF score should be in (0, 2/61] roughly — each half contributes
    at most 1/(60+1) ≈ 0.0164."""
    from app.embeddings import embed_query

    await _seed(pool, [{"id": 1, "title": "water purification tablets"}])
    vec = embed_query("water purification")
    results = await search.hybrid_search(
        pool, "water purification", vec, k_retrieve=10, k_final=10
    )
    assert len(results) == 1
    # Score is bounded above by 2/(60+1) when a doc is rank 1 in both halves
    assert 0 < results[0]["score"] <= 2 / 61 + 1e-9


async def test_hybrid_search_returns_vector_only_hits(pool):
    """If the keyword side finds nothing (no exact token overlap), the
    vector side should still populate results via the LEFT JOIN."""
    from app.embeddings import embed_query

    await _seed(pool, [
        {"id": 1, "title": "keeping feet dry in cold weather", "body": "socks guide"},
    ])
    # Query has zero token overlap with the title, pure semantic match.
    q = "avoiding frostbite on extremities"
    vec = embed_query(q)
    results = await search.hybrid_search(pool, q, vec, k_retrieve=10, k_final=10)
    assert len(results) == 1
    assert results[0]["vector_rank"] is not None
    # keyword_rank may be None — that's fine


async def test_hybrid_search_includes_rank_fields(pool):
    """The hybrid function must expose both vector_rank and keyword_rank in
    its output so the trace panel can display the split."""
    from app.embeddings import embed_query

    await _seed(pool, [{"id": 1, "title": "tent stakes in rocky ground"}])
    vec = embed_query("tent stakes in rocky ground")
    results = await search.hybrid_search(pool, "tent stakes", vec, k_retrieve=10, k_final=10)
    assert "vector_rank" in results[0]
    assert "keyword_rank" in results[0]
