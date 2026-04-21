from __future__ import annotations

from typing import Any

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


async def semantic_search(
    pool: asyncpg.Pool, query: str, k: int = 10
) -> list[dict[str, Any]]:
    vector = np.array(embed_query(query), dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(SEARCH_SQL, vector, k)
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "score": float(row["score"]),
        }
        for row in rows
    ]
