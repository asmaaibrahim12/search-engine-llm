from __future__ import annotations

from typing import Any, Sequence

import asyncpg
import numpy as np

from app.embeddings import embed_query

VECTOR_SQL = """
SELECT id,
       title,
       body,
       1 - (content_embedding <=> $1) AS score
FROM outdoors
ORDER BY content_embedding <=> $1
LIMIT $2
"""


async def search_by_vector(
    pool: asyncpg.Pool, vector: Sequence[float], k: int = 10
) -> list[dict[str, Any]]:
    """Run pgvector cosine-similarity search against the content_embedding
    column (title + body concatenated)."""
    arr = np.array(vector, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(VECTOR_SQL, arr, k)
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
    """Convenience wrapper: embed then search."""
    return await search_by_vector(pool, embed_query(query), k)
