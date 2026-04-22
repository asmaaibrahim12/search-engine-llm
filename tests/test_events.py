"""Tests for the feedback/analytics layer.

Pure-Python tests exercise the session cookie + rate limiter.
DB-backed tests cover log_event actually writing rows into search_events.
"""
from __future__ import annotations

import asyncio
import time

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


def test_rate_limit_allows_up_to_quota():
    events._buckets.clear()
    sid = "test-session-allow"
    for i in range(events.RATE_LIMIT_PER_WINDOW):
        assert events.rate_limit_ok(sid), f"failed at event #{i}"


def test_rate_limit_blocks_over_quota():
    events._buckets.clear()
    sid = "test-session-block"
    for _ in range(events.RATE_LIMIT_PER_WINDOW):
        events.rate_limit_ok(sid)
    # One more should trip the limit
    assert not events.rate_limit_ok(sid)


def test_rate_limit_is_per_session():
    events._buckets.clear()
    sid_a, sid_b = "session-a", "session-b"
    for _ in range(events.RATE_LIMIT_PER_WINDOW):
        events.rate_limit_ok(sid_a)
    # A is saturated, B is fresh
    assert not events.rate_limit_ok(sid_a)
    assert events.rate_limit_ok(sid_b)


def test_rate_limit_window_expires(monkeypatch):
    events._buckets.clear()
    sid = "expiring-session"

    t = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: t["now"])

    for _ in range(events.RATE_LIMIT_PER_WINDOW):
        events.rate_limit_ok(sid)
    assert not events.rate_limit_ok(sid)

    # Jump forward past the window; bucket should drain
    t["now"] += events.RATE_LIMIT_WINDOW_S + 1
    assert events.rate_limit_ok(sid)


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
