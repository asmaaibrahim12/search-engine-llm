from __future__ import annotations

from typing import Any, Sequence

import asyncpg
import numpy as np

from app.embeddings import embed_query

# -----------------------------------------------------------------------------
# Vector-only search (fallback / testing)
# -----------------------------------------------------------------------------

VECTOR_SQL = """
SELECT id,
       title,
       body,
       item_type,
       is_accepted,
       parent_id,
       tags,
       score AS upvotes,
       1 - (content_embedding <=> $1) AS score
FROM outdoors
ORDER BY content_embedding <=> $1
LIMIT $2
"""


def _row_to_dict(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "score": float(row["score"]),
        "item_type": row.get("item_type") if isinstance(row, dict) else row["item_type"],
        "is_accepted": row["is_accepted"],
        "parent_id": row["parent_id"],
        "tags": list(row["tags"] or []),
        "upvotes": int(row["upvotes"] or 0),
    }


async def search_by_vector(
    pool: asyncpg.Pool, vector: Sequence[float], k: int = 10
) -> list[dict[str, Any]]:
    """Pure cosine-similarity search over content_embedding."""
    arr = np.array(vector, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(VECTOR_SQL, arr, k)
    return [_row_to_dict(r) for r in rows]


# -----------------------------------------------------------------------------
# Hybrid search: vector + BM25 with Reciprocal Rank Fusion
# -----------------------------------------------------------------------------
#
# Each side independently returns its top N by its own score. We then fuse by
# rank (not score) using RRF: score = 1/(k + rank). The constant k=60 is from
# the original Cormack et al. paper and is the robust default; no tuning
# required until you have a labeled eval set to tune against.
#
# The LEFT JOIN means a row qualifying in only ONE of the two halves still
# gets included — important so that pure-keyword queries (e.g. exact product
# names) aren't dropped just because the vector side didn't surface them, and
# vice versa.

# Hybrid: vector cosine + BM25, fused by Reciprocal Rank Fusion (k=60),
# then small additive bumps for authority signals so high-quality answers
# outrank equally-ranked low-quality ones:
#   +0.005  if is_accepted          (about 1/4 of a full rank, deliberate
#                                    bump without dominating retrieval)
#   +0.002 * log1p(upvotes)         (caps out around +0.01 at 150 upvotes)
HYBRID_SQL = """
WITH vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY content_embedding <=> $1) AS rnk
    FROM outdoors
    ORDER BY content_embedding <=> $1
    LIMIT $2
),
keyword_hits AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', $3)) DESC
    ) AS rnk
    FROM outdoors
    WHERE content_tsv @@ websearch_to_tsquery('english', $3)
    LIMIT $2
)
SELECT o.id,
       o.title,
       o.body,
       o.item_type,
       o.is_accepted,
       o.parent_id,
       o.tags,
       o.score AS upvotes,
       (
           COALESCE(1.0 / (60 + v.rnk), 0)
         + COALESCE(1.0 / (60 + k.rnk), 0)
         + CASE WHEN o.is_accepted THEN 0.005 ELSE 0 END
         + 0.002 * ln(1 + GREATEST(o.score, 0))
       ) AS score,
       v.rnk AS vector_rank,
       k.rnk AS keyword_rank
FROM outdoors o
LEFT JOIN vector_hits  v ON v.id = o.id
LEFT JOIN keyword_hits k ON k.id = o.id
WHERE v.rnk IS NOT NULL OR k.rnk IS NOT NULL
ORDER BY score DESC
LIMIT $4
"""


# Filter CTE — applied to both halves of the hybrid query so candidates
# that would have been filtered out don't waste a slot in the top-50.
_HYBRID_FILTERED_SQL = """
WITH filtered AS (
    SELECT *
    FROM outdoors
    WHERE ($5::text[] IS NULL OR tags && $5::text[])
      AND ($6::text[] IS NULL OR item_type = ANY($6::text[]))
      AND ($7::boolean = FALSE OR is_accepted = TRUE)
      AND ($8::int IS NULL OR score >= $8::int)
),
vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY content_embedding <=> $1) AS rnk
    FROM filtered
    ORDER BY content_embedding <=> $1
    LIMIT $2
),
keyword_hits AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', $3)) DESC
    ) AS rnk
    FROM filtered
    WHERE content_tsv @@ websearch_to_tsquery('english', $3)
    LIMIT $2
)
SELECT o.id,
       o.title,
       o.body,
       o.item_type,
       o.is_accepted,
       o.parent_id,
       o.tags,
       o.score AS upvotes,
       (
           COALESCE(1.0 / (60 + v.rnk), 0)
         + COALESCE(1.0 / (60 + k.rnk), 0)
         + CASE WHEN o.is_accepted THEN 0.005 ELSE 0 END
         + 0.002 * ln(1 + GREATEST(o.score, 0))
       ) AS score,
       v.rnk AS vector_rank,
       k.rnk AS keyword_rank
FROM outdoors o
LEFT JOIN vector_hits  v ON v.id = o.id
LEFT JOIN keyword_hits k ON k.id = o.id
WHERE v.rnk IS NOT NULL OR k.rnk IS NOT NULL
ORDER BY score DESC
LIMIT $4
"""


async def hybrid_search(
    pool: asyncpg.Pool,
    query: str,
    vector: Sequence[float],
    k_retrieve: int = 50,
    k_final: int = 50,
    tags: list[str] | None = None,
    item_types: list[str] | None = None,
    accepted_only: bool = False,
    min_score: int | None = None,
) -> list[dict[str, Any]]:
    """Hybrid vector + BM25 with optional filters.

    Filters are applied BEFORE retrieval (inside the CTE), not after,
    so candidates filtered away don't waste a slot in the top-k.
    """
    arr = np.array(vector, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            _HYBRID_FILTERED_SQL,
            arr, k_retrieve, query, k_final,
            tags or None,
            item_types or None,
            bool(accepted_only),
            int(min_score) if min_score is not None else None,
        )
    results = []
    for row in rows:
        d = _row_to_dict(row)
        d["vector_rank"] = row["vector_rank"]
        d["keyword_rank"] = row["keyword_rank"]
        results.append(d)
    return results


async def top_tags(pool: asyncpg.Pool, limit: int = 30) -> list[str]:
    """Return the most common tags, for the filter UI's picker list.

    Cached at app startup — the tag distribution doesn't change between
    re-indexings. Empty list if the DB hasn't been seeded yet."""
    sql = """
    SELECT tag, COUNT(*) AS freq
    FROM outdoors, unnest(tags) AS tag
    WHERE tag IS NOT NULL AND tag <> ''
    GROUP BY tag
    ORDER BY freq DESC
    LIMIT $1
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, limit)
    return [r["tag"] for r in rows]


async def semantic_search(
    pool: asyncpg.Pool, query: str, k: int = 10
) -> list[dict[str, Any]]:
    """Convenience wrapper: embed then vector-search. Used by tests and as
    a simple fallback when hybrid isn't desired."""
    return await search_by_vector(pool, embed_query(query), k)
