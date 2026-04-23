"""Anonymous feedback + analytics logging.

Responsibilities:
- Mint a per-visitor session cookie (no login) so thumbs and clicks can
  be aggregated per session.
- Insert rows into `search_events`. Used by /search (type='search'),
  /events/click, and /events/thumb.
- Rate-limit per session with an in-memory token bucket to stop abuse /
  runaway JS.

Intentionally lightweight: no queue, no batcher. If Postgres is
unreachable, log-and-drop — events are best-effort, never on the hot
critical path of a user's search response.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Any

import asyncpg
from fastapi import Request, Response

SESSION_COOKIE = "sid"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _cookie_secure() -> bool:
    """Set the Secure flag on the session cookie when explicitly requested.

    On Railway + other HTTPS-only hosts, set COOKIE_SECURE=1 so the cookie
    is never sent over plaintext. Defaults to False for local dev (HTTP).
    """
    return os.environ.get("COOKIE_SECURE", "").lower() in ("1", "true", "yes")


# Token bucket per session: max 120 events per 60s window. Plenty for a
# real user (10 results * a few searches/minute), catches runaway JS.
RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_PER_WINDOW = 120

_buckets: dict[str, deque[float]] = defaultdict(deque)


def get_or_create_session(request: Request, response: Response) -> str:
    """Return the session id from the cookie, minting one if absent.

    Writes the cookie on `response` when newly minted. SameSite=Lax /
    HttpOnly / Secure-when-enabled — strictly for analytics on this origin.
    """
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and len(sid) == 36 and _is_uuid(sid):
        return sid
    sid = str(uuid.uuid4())
    response.set_cookie(
        SESSION_COOKIE,
        sid,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
    )
    return sid


def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False


def rate_limit_ok(session_id: str) -> bool:
    """Best-effort in-memory rate limit. Returns False if session is over
    its quota in the last RATE_LIMIT_WINDOW_S seconds."""
    now = time.monotonic()
    bucket = _buckets[session_id]
    cutoff = now - RATE_LIMIT_WINDOW_S
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_WINDOW:
        return False
    bucket.append(now)
    return True


async def log_event(
    pool: asyncpg.Pool,
    *,
    session_id: str,
    query: str,
    event_type: str,
    result_id: int | None = None,
    result_position: int | None = None,
    pipeline: str | None = None,
    latency_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert one event. Never raises — swallows DB errors so analytics
    failures don't break user-facing requests."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO search_events
                    (session_id, query, event_type, result_id, result_position,
                     pipeline, latency_ms, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                """,
                session_id,
                query[:500],  # cap to avoid unbounded logs
                event_type,
                result_id,
                result_position,
                pipeline,
                latency_ms,
                json.dumps(metadata or {}),
            )
    except Exception as exc:
        print(f"log_event failed: {exc!r}", flush=True)
