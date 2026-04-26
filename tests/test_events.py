"""Tests for the feedback/analytics layer.

Pure-Python tests exercise the session cookie + rate limiter.
DB-backed tests cover log_event actually writing rows into search_events.
"""
from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request
from starlette.responses import Response

from app import events
from app.db import create_pool, ensure_schema
from tests.conftest import TEST_DATABASE_URL, needs_db


# -----------------------------------------------------------------------------
# Pure-Python helpers
# -----------------------------------------------------------------------------


def _make_request(cookies: dict | None = None) -> Request:
    """Minimal ASGI scope just good enough for cookie handling."""
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"cookie", "; ".join(f"{k}={v}" for k, v in (cookies or {}).items()).encode())]
                   if cookies else [],
    }
    return Request(scope)


def test_session_is_minted_when_cookie_absent():
    req = _make_request()
    resp = Response()
    sid = events.get_or_create_session(req, resp)
    assert events._is_uuid(sid)
    assert "sid=" in resp.headers["set-cookie"]


def test_session_is_reused_when_cookie_is_valid():
    existing = "11111111-2222-3333-4444-555555555555"
    req = _make_request({"sid": existing})
    resp = Response()
    sid = events.get_or_create_session(req, resp)
    assert sid == existing
    # No Set-Cookie header when we didn't mint
    assert "set-cookie" not in resp.headers


def test_session_is_reminted_when_cookie_is_not_a_uuid():
    """Guards against someone smuggling a junk/crafted sid via cookie."""
    req = _make_request({"sid": "not-a-uuid"})
    resp = Response()
    sid = events.get_or_create_session(req, resp)
    assert events._is_uuid(sid)
    assert sid != "not-a-uuid"


def test_session_is_reminted_when_cookie_wrong_length():
    req = _make_request({"sid": "short"})
    resp = Response()
    sid = events.get_or_create_session(req, resp)
    assert events._is_uuid(sid)


def test_cookie_secure_defaults_to_false(monkeypatch):
    """Local dev (HTTP) must NOT set Secure or browsers drop the cookie."""
    monkeypatch.delenv("COOKIE_SECURE", raising=False)
    req = _make_request()
    resp = Response()
    events.get_or_create_session(req, resp)
    assert "secure" not in resp.headers["set-cookie"].lower()


def test_cookie_secure_set_when_env_enabled(monkeypatch):
    monkeypatch.setenv("COOKIE_SECURE", "1")
    req = _make_request()
    resp = Response()
    events.get_or_create_session(req, resp)
    assert "secure" in resp.headers["set-cookie"].lower()


def test_is_uuid_truthy_and_falsy():
    assert events._is_uuid("11111111-2222-3333-4444-555555555555")
    assert not events._is_uuid("")
    assert not events._is_uuid("garbage")
    assert not events._is_uuid(None)


# -----------------------------------------------------------------------------
# DB-backed tests for log_event
# -----------------------------------------------------------------------------


pytestmark_db = [pytest.mark.asyncio, needs_db]


@pytest.fixture
async def pool():
    p = await create_pool(TEST_DATABASE_URL)
    await ensure_schema(p)
    async with p.acquire() as conn:
        await conn.execute("TRUNCATE search_events RESTART IDENTITY")
    yield p
    await p.close()


@pytest.mark.asyncio
@needs_db
async def test_log_event_inserts_search_row(pool):
    await events.log_event(
        pool,
        session_id="s-1", query="hiking",
        event_type="search",
        pipeline="hybrid_rerank", latency_ms=123,
        metadata={"tags": ["hiking"]},
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM search_events WHERE session_id = 's-1'")
    assert row is not None
    assert row["event_type"] == "search"
    assert row["query"] == "hiking"
    assert row["pipeline"] == "hybrid_rerank"
    assert row["latency_ms"] == 123
    assert row["result_id"] is None
    import json as _json
    assert _json.loads(row["metadata"])["tags"] == ["hiking"]


@pytest.mark.asyncio
@needs_db
async def test_log_event_inserts_click_row(pool):
    await events.log_event(
        pool,
        session_id="s-click", query="tents",
        event_type="click",
        result_id=42, result_position=3,
        pipeline="hybrid_rerank",
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM search_events WHERE session_id = 's-click'")
    assert row["event_type"] == "click"
    assert row["result_id"] == 42
    assert row["result_position"] == 3


@pytest.mark.asyncio
@needs_db
async def test_log_event_records_thumb_variants(pool):
    await events.log_event(pool, session_id="s-t", query="q", event_type="thumb_up",
                            result_id=1, result_position=1)
    await events.log_event(pool, session_id="s-t", query="q", event_type="thumb_down",
                            result_id=2, result_position=2)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT event_type FROM search_events WHERE session_id = 's-t' "
            "ORDER BY occurred_at"
        )
    assert [r["event_type"] for r in rows] == ["thumb_up", "thumb_down"]


@pytest.mark.asyncio
@needs_db
async def test_log_event_truncates_absurdly_long_queries(pool):
    long_q = "x" * 10_000
    await events.log_event(pool, session_id="s-long", query=long_q, event_type="search")
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT query FROM search_events WHERE session_id = 's-long'")
    assert len(row["query"]) <= 500


@pytest.mark.asyncio
@needs_db
async def test_log_event_swallows_db_errors(pool):
    """Logging failures must NOT raise — analytics is best-effort."""
    # Pass a pool that's been closed; log_event should return without raising.
    await pool.close()
    await events.log_event(
        pool, session_id="s", query="q", event_type="search",
    )
    # If we got here without exception, the test passes.


@pytest.mark.asyncio
@needs_db
async def test_search_events_rejects_invalid_event_type(pool):
    """CHECK constraint on event_type keeps the schema honest."""
    with pytest.raises(Exception):
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO search_events (session_id, query, event_type) "
                "VALUES ('s', 'q', 'bogus')"
            )


# -----------------------------------------------------------------------------
# DB-backed rate limiter
# -----------------------------------------------------------------------------


@pytest.fixture
async def rl_pool():
    p = await create_pool(TEST_DATABASE_URL)
    await ensure_schema(p)
    async with p.acquire() as conn:
        await conn.execute("TRUNCATE rate_limit_buckets")
    yield p
    await p.close()


@pytest.mark.asyncio
@needs_db
async def test_rate_limit_allows_up_to_quota(rl_pool):
    sid = "rl-allow"
    for i in range(events.RATE_LIMIT_PER_WINDOW):
        ok = await events.rate_limit_ok(rl_pool, sid)
        assert ok, f"failed at event #{i}"


@pytest.mark.asyncio
@needs_db
async def test_rate_limit_blocks_over_quota(rl_pool):
    sid = "rl-block"
    for _ in range(events.RATE_LIMIT_PER_WINDOW):
        await events.rate_limit_ok(rl_pool, sid)
    assert not await events.rate_limit_ok(rl_pool, sid)


@pytest.mark.asyncio
@needs_db
async def test_rate_limit_is_per_session(rl_pool):
    sid_a, sid_b = "rl-a", "rl-b"
    for _ in range(events.RATE_LIMIT_PER_WINDOW):
        await events.rate_limit_ok(rl_pool, sid_a)
    assert not await events.rate_limit_ok(rl_pool, sid_a)
    assert await events.rate_limit_ok(rl_pool, sid_b)


@pytest.mark.asyncio
@needs_db
async def test_rate_limit_window_resets_after_expiry(rl_pool):
    """Backdate the window_start so the CASE WHEN reset branch fires."""
    sid = "rl-expire"
    # Prime at the quota.
    async with rl_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO rate_limit_buckets (session_id, window_start, count) "
            "VALUES ($1, NOW() - INTERVAL '5 minutes', $2)",
            sid, events.RATE_LIMIT_PER_WINDOW,
        )
    # Fresh call after window expiry resets: count becomes 1 → allowed.
    assert await events.rate_limit_ok(rl_pool, sid)
    async with rl_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count FROM rate_limit_buckets WHERE session_id = $1", sid
        )
    assert count == 1


@pytest.mark.asyncio
@needs_db
async def test_rate_limit_fails_open_on_db_error(rl_pool):
    """A DB outage must not drop user events — we prefer slight over-count."""
    await rl_pool.close()
    ok = await events.rate_limit_ok(rl_pool, "anything")
    assert ok is True


@pytest.mark.asyncio
@needs_db
async def test_refresh_result_ctr_resets_session_settings():
    """SET (not SET LOCAL) is required because REFRESH CONCURRENTLY can't
    run inside a transaction. We must RESET in finally so the connection
    returns to the pool with stock settings — otherwise the next checkout
    inherits parallel=0 / work_mem=4MB and queries get slow silently."""
    from app.db import refresh_result_ctr
    p = await create_pool(TEST_DATABASE_URL)
    try:
        await ensure_schema(p)
        await refresh_result_ctr(p)
        # Drain other connections, force a fresh acquire that should
        # see stock settings (or at least not our overrides).
        async with p.acquire() as conn:
            workers = await conn.fetchval("SHOW max_parallel_workers_per_gather")
            wm = await conn.fetchval("SHOW work_mem")
        # Stock Postgres default is 2 (or whatever the server config has);
        # whatever it is, our refresh forced 0 — assert we didn't leak it.
        assert workers != "0", f"max_parallel_workers_per_gather leaked: {workers}"
        assert wm != "4MB", f"work_mem leaked: {wm}"
    finally:
        await p.close()


@pytest.mark.asyncio
@needs_db
async def test_prune_rate_limit_buckets_drops_old_rows():
    from app.db import prune_rate_limit_buckets
    p = await create_pool(TEST_DATABASE_URL)
    try:
        await ensure_schema(p)
        async with p.acquire() as conn:
            await conn.execute("TRUNCATE rate_limit_buckets")
            await conn.execute(
                "INSERT INTO rate_limit_buckets (session_id, window_start, count) "
                "VALUES ('old', NOW() - INTERVAL '2 days', 5), "
                "       ('new', NOW(), 5)"
            )
        deleted = await prune_rate_limit_buckets(p, older_than_s=86400)
        assert deleted == 1
        async with p.acquire() as conn:
            remaining = await conn.fetch(
                "SELECT session_id FROM rate_limit_buckets ORDER BY session_id"
            )
        assert [r["session_id"] for r in remaining] == ["new"]
    finally:
        await p.close()
