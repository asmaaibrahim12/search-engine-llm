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
    expected = {
        "id", "title", "body", "score",
        "item_type", "is_accepted", "parent_id", "tags", "upvotes",
    }
    assert set(results[0].keys()) >= expected
    assert results[0]["id"] == 42
    assert results[0]["title"] == "sample"
    assert results[0]["body"] == "sample body"
    assert results[0]["item_type"] == "question"  # default from schema


async def test_accepted_answer_outranks_plain_answer_on_ties(pool):
    """Two identical-body answers; only one marked accepted. Accepted wins."""
    from app.embeddings import embed_query

    # Same body text -> near-identical vector and keyword scores. The
    # +0.005 accepted boost is what breaks the tie.
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
    await _seed(pool, [
        {"id": 1, "title": "How to lace hiking boots"},
    ])
    # Manually insert two near-identical answers; flip accepted on one.
    from app.embeddings import embed_batch
    bodies = ["Use surgeon's knot at the ankle hooks for a snug fit.",
              "Use surgeon's knot at the ankle hooks for a snug fit."]
    vecs = embed_batch(bodies)
    records = [
        (2, 1, "answer", None, bodies[0], 5, False, [], np.array(vecs[0], dtype=np.float32)),
        (3, 1, "answer", None, bodies[1], 5, True,  [], np.array(vecs[1], dtype=np.float32)),
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO outdoors (id, parent_id, item_type, title, body, "
            "score, is_accepted, tags, content_embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            records,
        )

    vec = embed_query("lace hiking boots tight")
    results = await search.hybrid_search(
        pool, "lace hiking boots tight", vec, k_retrieve=10, k_final=10
    )
    # Accepted answer should come above the non-accepted one
    positions = {r["id"]: i for i, r in enumerate(results)}
    assert positions[3] < positions[2], (
        f"accepted answer should rank above plain answer, got {results}"
    )


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


# -----------------------------------------------------------------------------
# Filter parameters
# -----------------------------------------------------------------------------


async def _seed_with_metadata(pool, rows: list[dict]):
    """Insert rows including item_type, is_accepted, tags, score."""
    from app.embeddings import embed_batch

    texts = [f"{r.get('title') or ''} {r.get('body') or ''}" for r in rows]
    vecs = embed_batch(texts)
    records = []
    for i, r in enumerate(rows):
        records.append((
            r["id"],
            r.get("parent_id"),
            r.get("item_type", "question"),
            r.get("title"),
            r.get("body", ""),
            r.get("score", 0),
            r.get("is_accepted", False),
            r.get("tags", []),
            np.array(vecs[i], dtype=np.float32),
        ))
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
        await conn.executemany(
            "INSERT INTO outdoors (id, parent_id, item_type, title, body, "
            "score, is_accepted, tags, content_embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            records,
        )


async def test_hybrid_search_filter_by_tag(pool):
    from app.embeddings import embed_query

    await _seed_with_metadata(pool, [
        {"id": 1, "title": "waterproof tent", "tags": ["camping", "tent"]},
        {"id": 2, "title": "waterproof jacket", "tags": ["apparel", "rain"]},
        {"id": 3, "title": "waterproof backpack", "tags": ["gear"]},
    ])
    vec = embed_query("waterproof gear")
    results = await search.hybrid_search(
        pool, "waterproof gear", vec,
        k_retrieve=10, k_final=10,
        tags=["camping"],
    )
    assert len(results) == 1
    assert results[0]["id"] == 1


async def test_hybrid_search_filter_by_item_type(pool):
    from app.embeddings import embed_query

    await _seed_with_metadata(pool, [
        {"id": 1, "title": "How to pitch tent", "item_type": "question"},
        {"id": 2, "title": None, "body": "use the rainfly first", "item_type": "answer", "parent_id": 1},
        {"id": 3, "title": None, "body": "stake corners tightly", "item_type": "answer", "parent_id": 1},
    ])
    vec = embed_query("pitch tent technique")

    qs_only = await search.hybrid_search(
        pool, "pitch tent technique", vec,
        k_retrieve=10, k_final=10, item_types=["question"],
    )
    assert len(qs_only) == 1
    assert all(r["item_type"] == "question" for r in qs_only)

    as_only = await search.hybrid_search(
        pool, "pitch tent technique", vec,
        k_retrieve=10, k_final=10, item_types=["answer"],
    )
    assert len(as_only) == 2
    assert all(r["item_type"] == "answer" for r in as_only)


async def test_hybrid_search_accepted_only(pool):
    from app.embeddings import embed_query

    await _seed_with_metadata(pool, [
        {"id": 1, "title": None, "body": "answer one", "item_type": "answer",
         "is_accepted": True},
        {"id": 2, "title": None, "body": "answer two", "item_type": "answer",
         "is_accepted": False},
    ])
    vec = embed_query("answer")
    results = await search.hybrid_search(
        pool, "answer", vec, k_retrieve=10, k_final=10, accepted_only=True,
    )
    assert len(results) == 1
    assert results[0]["is_accepted"] is True


async def test_hybrid_search_min_score(pool):
    from app.embeddings import embed_query

    await _seed_with_metadata(pool, [
        {"id": 1, "title": "popular question", "score": 50},
        {"id": 2, "title": "unpopular question", "score": 1},
    ])
    vec = embed_query("question")
    results = await search.hybrid_search(
        pool, "question", vec, k_retrieve=10, k_final=10, min_score=10,
    )
    assert len(results) == 1
    assert results[0]["id"] == 1


async def test_hybrid_search_combined_filters(pool):
    """Tag AND item_type AND accepted_only AND min_score all apply together."""
    from app.embeddings import embed_query

    await _seed_with_metadata(pool, [
        # Matches all filters
        {"id": 1, "title": None, "body": "pitch answer", "item_type": "answer",
         "is_accepted": True, "tags": ["camping"], "score": 20},
        # Wrong tag
        {"id": 2, "title": None, "body": "pitch answer", "item_type": "answer",
         "is_accepted": True, "tags": ["climbing"], "score": 20},
        # Not accepted
        {"id": 3, "title": None, "body": "pitch answer", "item_type": "answer",
         "is_accepted": False, "tags": ["camping"], "score": 20},
        # Too low score
        {"id": 4, "title": None, "body": "pitch answer", "item_type": "answer",
         "is_accepted": True, "tags": ["camping"], "score": 2},
    ])
    vec = embed_query("pitch")
    results = await search.hybrid_search(
        pool, "pitch", vec, k_retrieve=10, k_final=10,
        tags=["camping"], item_types=["answer"],
        accepted_only=True, min_score=10,
    )
    assert len(results) == 1
    assert results[0]["id"] == 1


async def test_hybrid_search_filters_default_noop(pool):
    """Passing None/defaults should behave like no filters at all."""
    from app.embeddings import embed_query

    await _seed_with_metadata(pool, [
        {"id": 1, "title": "a"}, {"id": 2, "title": "b"}, {"id": 3, "title": "c"},
    ])
    vec = embed_query("anything")
    results = await search.hybrid_search(
        pool, "anything", vec, k_retrieve=10, k_final=10,
        tags=None, item_types=None, accepted_only=False, min_score=None,
    )
    assert len(results) == 3


# -----------------------------------------------------------------------------
# top_tags helper
# -----------------------------------------------------------------------------


async def test_top_tags_returns_most_common(pool):
    await _seed_with_metadata(pool, [
        {"id": 1, "title": "a", "tags": ["hiking", "boots"]},
        {"id": 2, "title": "b", "tags": ["hiking", "tents"]},
        {"id": 3, "title": "c", "tags": ["hiking"]},
        {"id": 4, "title": "d", "tags": ["boots"]},
        {"id": 5, "title": "e", "tags": ["rare"]},
    ])
    tags = await search.top_tags(pool, limit=3)
    assert tags[0] == "hiking"  # 3 occurrences
    assert tags[1] == "boots"   # 2
    assert len(tags) == 3


async def test_top_tags_empty_db(pool):
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
    tags = await search.top_tags(pool)
    assert tags == []


# -----------------------------------------------------------------------------
# result_ctr feedback-loop bump
# -----------------------------------------------------------------------------


async def _insert_event(pool, *, result_id: int, event_type: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO search_events (session_id, query, event_type, result_id) "
            "VALUES ('t', 'q', $1, $2)",
            event_type, result_id,
        )


async def _refresh_ctr(pool) -> None:
    from app.db import refresh_result_ctr
    await refresh_result_ctr(pool)


async def test_hybrid_search_click_bump_breaks_ties(pool):
    """Two near-identical answers; one has prior clicks. Clicked one wins."""
    from app.embeddings import embed_batch, embed_query

    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
        await conn.execute("TRUNCATE search_events RESTART IDENTITY")
    await _seed(pool, [
        {"id": 1, "title": "How to lace hiking boots"},
    ])
    bodies = ["Use a surgeon's knot at the ankle hooks.",
              "Use a surgeon's knot at the ankle hooks."]
    vecs = embed_batch(bodies)
    records = [
        (10, 1, "answer", None, bodies[0], 5, False, [], np.array(vecs[0], dtype=np.float32)),
        (11, 1, "answer", None, bodies[1], 5, False, [], np.array(vecs[1], dtype=np.float32)),
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO outdoors (id, parent_id, item_type, title, body, "
            "score, is_accepted, tags, content_embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            records,
        )
    # Log a handful of clicks on id=11; none on id=10.
    for _ in range(5):
        await _insert_event(pool, result_id=11, event_type="click")
    await _refresh_ctr(pool)

    vec = embed_query("lace hiking boots")
    results = await search.hybrid_search(
        pool, "lace hiking boots", vec, k_retrieve=10, k_final=10,
    )
    positions = {r["id"]: i for i, r in enumerate(results)}
    assert positions[11] < positions[10], (
        f"clicked answer should rank above unclicked on ties, got {results}"
    )


async def test_hybrid_search_thumb_down_demotes(pool):
    """Two near-identical rows; one gets thumbs_down. The clean one wins."""
    from app.embeddings import embed_batch, embed_query

    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
        await conn.execute("TRUNCATE search_events RESTART IDENTITY")
    await _seed(pool, [
        {"id": 20, "title": "best rain jacket for spring hikes"},
        {"id": 21, "title": "best rain jacket for spring hikes"},
    ])
    for _ in range(4):
        await _insert_event(pool, result_id=20, event_type="thumb_down")
    await _refresh_ctr(pool)

    vec = embed_query("rain jacket spring")
    results = await search.hybrid_search(
        pool, "rain jacket spring", vec, k_retrieve=10, k_final=10,
    )
    positions = {r["id"]: i for i, r in enumerate(results)}
    assert positions[21] < positions[20], (
        f"thumb_down'd row should rank below the clean one, got {results}"
    )


async def _insert_search_impression(pool, *, result_ids: list[int]) -> None:
    import json as _json
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO search_events (session_id, query, event_type, metadata) "
            "VALUES ('t', 'q', 'search', $1::jsonb)",
            _json.dumps({"result_ids": result_ids}),
        )


async def test_hybrid_search_ctr_rate_beats_raw_clicks(pool):
    """Two docs with identical 5 clicks each: the one shown far fewer times
    (higher CTR) should outrank the one shown many times (low CTR). This
    is the whole point of using impressions as a denominator, not raw
    click counts."""
    from app.embeddings import embed_batch, embed_query

    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
        await conn.execute("TRUNCATE search_events RESTART IDENTITY")
    await _seed(pool, [
        {"id": 30, "title": "best backpack for thru-hiking"},
        {"id": 31, "title": "best backpack for thru-hiking"},
    ])
    # Both get 5 clicks, same thumb state.
    for _ in range(5):
        await _insert_event(pool, result_id=30, event_type="click")
        await _insert_event(pool, result_id=31, event_type="click")
    # id=30 shown 10 times (CTR 0.5); id=31 shown 200 times (CTR 0.025).
    for _ in range(10):
        await _insert_search_impression(pool, result_ids=[30])
    for _ in range(200):
        await _insert_search_impression(pool, result_ids=[31])
    await _refresh_ctr(pool)

    vec = embed_query("backpack thru-hiking")
    results = await search.hybrid_search(
        pool, "backpack thru-hiking", vec, k_retrieve=10, k_final=10,
    )
    positions = {r["id"]: i for i, r in enumerate(results)}
    assert positions[30] < positions[31], (
        f"higher-CTR doc (5/10) should outrank lower-CTR doc (5/200), "
        f"got {results}"
    )


async def test_hybrid_search_empty_ctr_is_noop(pool):
    """With no events logged, ranking should match the pre-feedback behavior:
    RRF score bounded by 2/61 when a doc is rank 1 in both halves."""
    from app.embeddings import embed_query

    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
        await conn.execute("TRUNCATE search_events RESTART IDENTITY")
    await _seed(pool, [{"id": 1, "title": "water purification tablets"}])
    await _refresh_ctr(pool)
    vec = embed_query("water purification")
    results = await search.hybrid_search(
        pool, "water purification", vec, k_retrieve=10, k_final=10,
    )
    assert len(results) == 1
    assert 0 < results[0]["score"] <= 2 / 61 + 1e-9
