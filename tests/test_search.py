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
            "INSERT INTO outdoors (id, title, body, title_embedding) VALUES ($1, $2, $3, $4)",
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
