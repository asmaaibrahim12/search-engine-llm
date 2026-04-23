from __future__ import annotations

import asyncio
import os

import asyncpg
from pgvector.asyncpg import register_vector

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS outdoors (
    id BIGINT PRIMARY KEY,
    title TEXT,
    body TEXT,
    content_embedding vector(768),
    content_tsv tsvector
);

-- Drop legacy title-only column from earlier schemas, if it's still around.
ALTER TABLE outdoors DROP COLUMN IF EXISTS title_embedding;

-- Title is NULL for answers. Drop the NOT NULL constraint if it's still on
-- the column from an earlier schema.
ALTER TABLE outdoors ALTER COLUMN title DROP NOT NULL;

-- Existing databases created under a pre-hybrid schema lack content_tsv
-- and/or content_embedding. CREATE TABLE IF NOT EXISTS is a no-op when
-- the table exists, so we ADD COLUMN explicitly here before any UPDATE
-- or index that references them.
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS content_embedding vector(768);
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- New metadata columns (added idempotently so re-deploying is safe).
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS parent_id BIGINT;
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS item_type TEXT
    NOT NULL DEFAULT 'question'
    CHECK (item_type IN ('question', 'answer'));
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS score INT NOT NULL DEFAULT 0;
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS is_accepted BOOLEAN
    NOT NULL DEFAULT FALSE;
ALTER TABLE outdoors ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}';

-- NOTE: intentionally no big backfill UPDATE here. Historically we ran
--   UPDATE outdoors SET content_tsv = to_tsvector(...) WHERE content_tsv IS NULL
-- but on Postgres containers with the default 64 MB /dev/shm (Railway's
-- pgvector template) that statement exceeds the shared-memory segment
-- and crashes startup with DiskFullError. Existing rows just keep NULL
-- content_tsv until re-indexed — the indexer's upsert trips the
-- outdoors_tsv_update trigger and populates the column per row.

-- HNSW for vector cosine search.
CREATE INDEX IF NOT EXISTS outdoors_content_embedding_idx
    ON outdoors USING hnsw (content_embedding vector_cosine_ops);

-- GIN for BM25-style full-text search over title + body.
CREATE INDEX IF NOT EXISTS outdoors_content_tsv_idx
    ON outdoors USING GIN (content_tsv);

-- Secondary indexes for filters + parent lookups.
CREATE INDEX IF NOT EXISTS outdoors_parent_id_idx ON outdoors (parent_id);
CREATE INDEX IF NOT EXISTS outdoors_tags_idx ON outdoors USING GIN (tags);
CREATE INDEX IF NOT EXISTS outdoors_item_type_idx ON outdoors (item_type);

-- Keep content_tsv in sync with title/body via trigger.
CREATE OR REPLACE FUNCTION outdoors_tsv_refresh() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector(
        'english',
        coalesce(NEW.title, '') || ' ' || coalesce(NEW.body, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS outdoors_tsv_update ON outdoors;
CREATE TRIGGER outdoors_tsv_update
    BEFORE INSERT OR UPDATE OF title, body ON outdoors
    FOR EACH ROW EXECUTE FUNCTION outdoors_tsv_refresh();


-- ---------------------------------------------------------------------------
-- Feedback / analytics
-- ---------------------------------------------------------------------------
--
-- One row per user action: search submissions, result clicks, thumbs up/down.
-- Anonymous (session cookie, no login). Enables offline analysis of what
-- queries do well, which results get clicked/thumbed, and how each pipeline
-- performs in production alongside the offline eval harness.

CREATE TABLE IF NOT EXISTS search_events (
    id BIGSERIAL PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    event_type TEXT NOT NULL
        CHECK (event_type IN ('search', 'click', 'thumb_up', 'thumb_down')),
    result_id BIGINT,          -- NULL for 'search' rows
    result_position INT,       -- 1-indexed; NULL for 'search' rows
    pipeline TEXT,             -- which retrieval config produced the result
    latency_ms INT,            -- for 'search' events only
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS search_events_occurred_at_idx
    ON search_events (occurred_at DESC);
CREATE INDEX IF NOT EXISTS search_events_session_idx
    ON search_events (session_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS search_events_query_idx
    ON search_events (query);


-- ---------------------------------------------------------------------------
-- result_ctr: per-result engagement aggregate used as a tiny ranking signal
-- ---------------------------------------------------------------------------
--
-- Populated from search_events. The hybrid retriever LEFT JOINs this view
-- and adds a small rate-based click bump plus a log-shrunk net-thumb bump,
-- letting the pipeline close the feedback loop without touching the hot
-- path.
--
-- It's a MATERIALIZED VIEW (not a regular view) so the retrieval query
-- doesn't aggregate over search_events on every call. Refresh on a
-- schedule — see refresh_result_ctr() + the lifespan refresh loop in
-- main.py — not inline on write.
--
-- Impressions are derived from the `result_ids` array we now stash in the
-- metadata of each 'search' event. A FULL OUTER JOIN against the click /
-- thumb aggregate means a row can appear even if it was only impressed
-- (impressions > 0, clicks = 0) OR only engaged with (e.g. thumbs without
-- a paired search row, as can happen for older rows pre-dating the
-- metadata change).
--
-- Materialized view definitions can't be altered in place (Postgres won't
-- let you swap the SELECT), so we DROP + CREATE unconditionally. The MV
-- is a pure derivation of search_events — rebuilding it is cheap and the
-- lifespan refreshes it right after ensure_schema() runs.

DROP MATERIALIZED VIEW IF EXISTS result_ctr;
CREATE MATERIALIZED VIEW result_ctr AS
WITH deduped_engagements AS (
    -- Collapse repeated engagement events from the same session into one
    -- per kind. Rationale:
    --   * A user double-clicks a result → 2 rows, 1 click's worth of signal.
    --   * A user toggles 👍 → 👎 → 👍 → 3 rows, only the LATEST counts.
    -- Raw log rows stay in search_events for audit / future models; the
    -- ranking signal here is the deduped view of intent.
    SELECT DISTINCT ON (
        session_id, query, result_id,
        CASE WHEN event_type = 'click' THEN 'click' ELSE 'thumb' END
    )
        result_id, event_type
    FROM search_events
    WHERE result_id IS NOT NULL
      AND event_type IN ('click', 'thumb_up', 'thumb_down')
    ORDER BY
        session_id, query, result_id,
        CASE WHEN event_type = 'click' THEN 'click' ELSE 'thumb' END,
        occurred_at DESC
),
engagements AS (
    SELECT
        result_id,
        COUNT(*) FILTER (WHERE event_type = 'click')      AS clicks,
        COUNT(*) FILTER (WHERE event_type = 'thumb_up')   AS thumbs_up,
        COUNT(*) FILTER (WHERE event_type = 'thumb_down') AS thumbs_down
    FROM deduped_engagements
    GROUP BY result_id
),
impressions AS (
    SELECT
        (v.value)::text::bigint AS result_id,
        COUNT(*) AS impressions
    FROM search_events e,
         jsonb_array_elements(e.metadata -> 'result_ids') AS v
    WHERE e.event_type = 'search'
      AND jsonb_typeof(e.metadata -> 'result_ids') = 'array'
    GROUP BY 1
)
SELECT
    COALESCE(e.result_id, i.result_id) AS result_id,
    COALESCE(e.clicks, 0)              AS clicks,
    COALESCE(e.thumbs_up, 0)           AS thumbs_up,
    COALESCE(e.thumbs_down, 0)         AS thumbs_down,
    COALESCE(i.impressions, 0)         AS impressions
FROM engagements e
FULL OUTER JOIN impressions i ON i.result_id = e.result_id;

-- Unique index is required for REFRESH ... CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS result_ctr_result_id_idx
    ON result_ctr (result_id);
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
    await _bootstrap_extension(dsn)
    return await asyncpg.create_pool(dsn, min_size=1, max_size=5, init=_init_connection)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)


async def refresh_result_ctr(pool: asyncpg.Pool) -> None:
    """Recompute the result_ctr materialized view from search_events.

    CONCURRENTLY avoids blocking readers; it's safe because the unique
    index on result_id lets Postgres diff old vs. new rows.

    Best-effort: swallows errors so a refresh failure never crashes the
    app. Falls back to a non-concurrent refresh on first call (when the
    MV has never been populated, CONCURRENTLY raises).
    """
    try:
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "REFRESH MATERIALIZED VIEW CONCURRENTLY result_ctr"
                )
            except asyncpg.PostgresError:
                await conn.execute("REFRESH MATERIALIZED VIEW result_ctr")
    except Exception as exc:
        print(f"refresh_result_ctr failed: {exc!r}", flush=True)


async def result_ctr_refresh_loop(
    pool: asyncpg.Pool, interval_s: int
) -> None:
    """Periodically refresh result_ctr in the background.

    Meant to be scheduled from main.lifespan via asyncio.create_task.
    Sleeps first so an initial refresh (done explicitly at startup) isn't
    immediately re-run. Cancellation during sleep/refresh is the normal
    shutdown path; anything else we log and continue so a single bad
    refresh doesn't kill the loop for the rest of the process lifetime.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
            await refresh_result_ctr(pool)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"result_ctr refresh loop: {exc!r}", flush=True)
