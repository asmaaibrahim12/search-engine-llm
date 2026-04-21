from __future__ import annotations

import asyncio
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


async def _bootstrap_extension(dsn: str, attempts: int = 6) -> None:
    """Ensure the `vector` extension exists before the pool opens.

    Retries with exponential backoff because Postgres may be mid-restart
    when the app deploys — especially on the first deploy of a project
    where search-app and pgvector come up in parallel.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            conn = await asyncpg.connect(dsn, timeout=10)
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                return
            finally:
                await conn.close()
        except (asyncio.TimeoutError, OSError, asyncpg.PostgresError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = min(2 ** attempt, 15)
            print(
                f"Bootstrap connect failed ({exc!r}); retrying in {delay}s "
                f"(attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(
        f"Could not connect to Postgres to create the vector extension after "
        f"{attempts} attempts. Last error: {last_exc!r}"
    )


async def create_pool(dsn: str | None = None) -> asyncpg.Pool:
    dsn = dsn or os.environ["DATABASE_URL"]
    # Must run before pool creation — the pool's init hook calls
    # register_vector(), which requires the extension to already exist.
    await _bootstrap_extension(dsn)
    return await asyncpg.create_pool(dsn, min_size=1, max_size=5, init=_init_connection)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
