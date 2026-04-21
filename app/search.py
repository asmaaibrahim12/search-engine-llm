from __future__ import annotations

from typing import Any, Sequence

import asyncpg
import numpy as np

from app.embeddings import embed_query

SEARCH_SQL = """
SELECT id,
       title,
       body,
       1 - (title_embedding <=> $1) AS score
FROM outdoors
ORDER BY title_embedding <=> $1
LIMIT $2
"""


async def search_by_vector(
    pool: asyncpg.Pool, vector: Sequence[float], k: int = 10
) -> list[dict[str, Any]]:
    """Run the pgvector similarity query with an already-computed query vector.

    Split from semantic_search so callers that need per-stage telemetry
    (embed vs. search) can time each independently and pass the vector
    through.
    """
    arr = np.array(vector, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(SEARCH_SQL, arr, k)
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "score": float(row["score"]),
        }
        for row in rows
    ]


async def semantic_search(
    pool: asyncpg.Pool, query: str, k: int = 10
) -> list[dict[str, Any]]:
    """Convenience wrapper: embed then search in one call."""
    return await search_by_vector(pool, embed_query(query), k)
