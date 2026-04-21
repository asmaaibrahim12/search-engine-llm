from __future__ import annotations

import os

import asyncpg
from pgvector.asyncpg import register_vector

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS outdoors (
    id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT,
    title_embedding vector(768)
);

CREATE INDEX IF NOT EXISTS outdoors_embedding_idx
    ON outdoors USING hnsw (title_embedding vector_cosine_ops);
"""


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_pool(dsn: str | None = None) -> asyncpg.Pool:
    dsn = dsn or os.environ["DATABASE_URL"]
    return await asyncpg.create_pool(dsn, min_size=1, max_size=5, init=_init_connection)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
