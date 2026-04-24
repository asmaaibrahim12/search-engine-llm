from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Sequence

import asyncpg
import numpy as np

from app.embeddings import embed_query

# -----------------------------------------------------------------------------
# Ranking configuration
# -----------------------------------------------------------------------------
#
# Every magic number the hybrid ranker uses lives on this dataclass so the
# eval harness can sweep them without patching SQL strings. Values are
# substituted into the SQL template at build time (numbers only, never
# user input, so str.format is safe).


@dataclass(frozen=True)
class RankingConfig:
    # Reciprocal Rank Fusion constant. 60 is the Cormack et al. default.
    rrf_k: int = 60
    # Additive bump when a result is an accepted answer.
    accepted_bump: float = 0.005
    # Coefficient on log1p(upvotes).
    upvote_coeff: float = 0.002
    # Coefficient on the (clamped) click rate from result_ctr.
    ctr_coeff: float = 0.010
    # Minimum denominator for the CTR rate — smaller values than this are
    # treated as if we had this many impressions. Acts as a Bayesian
    # shrinkage prior so 1/1 doesn't look like 100% CTR.
    ctr_shrinkage_floor: int = 20
    # Coefficient on log1p(thumbs_up) − log1p(thumbs_down).
    thumb_coeff: float = 0.003

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise TypeError(f"{f.name} must be numeric, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"{f.name} must be >= 0, got {v}")


DEFAULT_CONFIG = RankingConfig()


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
    d = {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "score": float(row["score"]),
        "item_type": row["item_type"],
        "is_accepted": row["is_accepted"],
        "parent_id": row["parent_id"],
        "tags": list(row["tags"] or []),
        "upvotes": int(row["upvotes"] or 0),
    }
    # Engagement fields are only present when the query JOINs result_ctr.
    # Copy them in when available so the UI can surface them without a
    # second round-trip.
    for key in ("clicks", "thumbs_up", "thumbs_down", "impressions"):
        try:
            d[key] = int(row[key] or 0)
        except (KeyError, IndexError):
            pass
    return d


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
#
# Small additive authority bumps, in decreasing order of magnitude:
#   +accepted_bump                        if is_accepted
#   +upvote_coeff * log1p(upvotes)
#   +ctr_coeff    * LEAST(clicks / max(impressions, floor), 1.0)
#   +thumb_coeff  * (log1p(thumbs_up) - log1p(thumbs_down))

_HYBRID_TEMPLATE = """
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
           COALESCE(1.0 / ({rrf_k} + v.rnk), 0)
         + COALESCE(1.0 / ({rrf_k} + k.rnk), 0)
         + CASE WHEN o.is_accepted THEN {accepted_bump} ELSE 0 END
         + {upvote_coeff} * ln(1 + GREATEST(o.score, 0))
         + {ctr_coeff} * LEAST(
               COALESCE(f.clicks, 0)::float
             / GREATEST(COALESCE(f.impressions, 0), {ctr_shrinkage_floor}),
               1.0
           )
         + {thumb_coeff} * (
               ln(1 + COALESCE(f.thumbs_up, 0))
             - ln(1 + COALESCE(f.thumbs_down, 0))
           )
       ) AS score,
       v.rnk AS vector_rank,
       k.rnk AS keyword_rank,
       COALESCE(f.clicks, 0)      AS clicks,
       COALESCE(f.thumbs_up, 0)   AS thumbs_up,
       COALESCE(f.thumbs_down, 0) AS thumbs_down,
       COALESCE(f.impressions, 0) AS impressions
FROM outdoors o
LEFT JOIN vector_hits  v ON v.id = o.id
LEFT JOIN keyword_hits k ON k.id = o.id
LEFT JOIN result_ctr   f ON f.result_id = o.id
WHERE v.rnk IS NOT NULL OR k.rnk IS NOT NULL
ORDER BY score DESC
LIMIT $4
"""


# Filter CTE — applied to both halves of the hybrid query so candidates
# that would have been filtered out don't waste a slot in the top-50.
_HYBRID_FILTERED_TEMPLATE = """
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
           COALESCE(1.0 / ({rrf_k} + v.rnk), 0)
         + COALESCE(1.0 / ({rrf_k} + k.rnk), 0)
         + CASE WHEN o.is_accepted THEN {accepted_bump} ELSE 0 END
         + {upvote_coeff} * ln(1 + GREATEST(o.score, 0))
         + {ctr_coeff} * LEAST(
               COALESCE(f.clicks, 0)::float
             / GREATEST(COALESCE(f.impressions, 0), {ctr_shrinkage_floor}),
               1.0
           )
         + {thumb_coeff} * (
               ln(1 + COALESCE(f.thumbs_up, 0))
             - ln(1 + COALESCE(f.thumbs_down, 0))
           )
       ) AS score,
       v.rnk AS vector_rank,
       k.rnk AS keyword_rank,
       COALESCE(f.clicks, 0)      AS clicks,
       COALESCE(f.thumbs_up, 0)   AS thumbs_up,
       COALESCE(f.thumbs_down, 0) AS thumbs_down,
       COALESCE(f.impressions, 0) AS impressions
FROM outdoors o
LEFT JOIN vector_hits  v ON v.id = o.id
LEFT JOIN keyword_hits k ON k.id = o.id
LEFT JOIN result_ctr   f ON f.result_id = o.id
WHERE v.rnk IS NOT NULL OR k.rnk IS NOT NULL
ORDER BY score DESC
LIMIT $4
"""


@lru_cache(maxsize=32)
def _build_hybrid_sql(config: RankingConfig, filtered: bool) -> str:
    """Render the SQL template with the config's numeric constants.

    Cached by (config, filtered) so sweeping in the eval harness doesn't
    re-render on every query. Values are all numeric (validated in
    RankingConfig.__post_init__), so str.format is safe against injection.
    """
    template = _HYBRID_FILTERED_TEMPLATE if filtered else _HYBRID_TEMPLATE
    return template.format(
        rrf_k=config.rrf_k,
        accepted_bump=config.accepted_bump,
        upvote_coeff=config.upvote_coeff,
        ctr_coeff=config.ctr_coeff,
        ctr_shrinkage_floor=config.ctr_shrinkage_floor,
        thumb_coeff=config.thumb_coeff,
    )


# Exposed for tests and anything external that reads the default SQL.
HYBRID_SQL = _build_hybrid_sql(DEFAULT_CONFIG, filtered=False)
_HYBRID_FILTERED_SQL = _build_hybrid_sql(DEFAULT_CONFIG, filtered=True)


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
    config: RankingConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    """Hybrid vector + BM25 with optional filters.

    Filters are applied BEFORE retrieval (inside the CTE), not after,
    so candidates filtered away don't waste a slot in the top-k.

    Pass `config=RankingConfig(...)` to override individual ranking
    constants — useful for eval sweeps. The SQL is cached per config.
    """
    sql = _build_hybrid_sql(config, filtered=True)
    arr = np.array(vector, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql,
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
