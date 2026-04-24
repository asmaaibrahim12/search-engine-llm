from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks, FastAPI, Form, HTTPException, Query, Request, Response,
)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from app import embeddings, events, rag, rerank, search
from app.db import (
    create_pool, ensure_schema, refresh_result_ctr, result_ctr_refresh_loop,
)
from app.logging_setup import configure as configure_logging
from app.text import strip_html

load_dotenv()


VALID_ITEM_TYPES = {"question", "answer"}


def _clean_tags(tags: Optional[List[str]]) -> Optional[list[str]]:
    """Filter form input: drop blanks, keep at most 10, lowercase."""
    if not tags:
        return None
    cleaned = [t.strip().lower() for t in tags if t and t.strip()]
    return cleaned[:10] or None


def _clean_item_types(types: Optional[List[str]]) -> Optional[list[str]]:
    if not types:
        return None
    cleaned = [t for t in types if t in VALID_ITEM_TYPES]
    return cleaned or None

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Expose strip_html to templates so result bodies don't render raw HTML
# markup (Stack Exchange stores bodies as HTML).
templates.env.filters["strip_html"] = strip_html

# Pipeline constants
RETRIEVE_K = 50   # candidates fetched by the retriever (vector / hybrid)
SEARCH_TOP_K = 10  # final results shown to the user
PROMPT_TOP_K = 5   # top results passed into the LLM prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure structured logging before anything else so the rest of
    # startup (pool bootstrap, MV refresh) emits structured records.
    configure_logging()
    # Warm both models so the first request doesn't pay the load cost.
    embeddings.get_model()
    rerank.get_reranker()
    app.state.pool = await create_pool()
    await ensure_schema(app.state.pool)
    # Precompute the filter picker list so /search doesn't hit the DB for
    # it on every request. Tag distribution only changes at re-index time.
    try:
        app.state.top_tags = await search.top_tags(app.state.pool, limit=30)
    except Exception:
        app.state.top_tags = []
    # Build result_ctr once at boot (ensure_schema above drops+recreates
    # the MV empty) and then keep it fresh on a timer. Interval is tunable
    # via env; the default is a gentle 5 minutes. Set to 0 to disable.
    await refresh_result_ctr(app.state.pool)
    refresh_interval_s = int(os.environ.get("RESULT_CTR_REFRESH_SEC", "300"))
    refresh_task: asyncio.Task | None = None
    if refresh_interval_s > 0:
        refresh_task = asyncio.create_task(
            result_ctr_refresh_loop(app.state.pool, refresh_interval_s)
        )
    try:
        yield
    finally:
        if refresh_task is not None:
            refresh_task.cancel()
            try:
                await refresh_task
            except (asyncio.CancelledError, Exception):
                pass
        await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


async def run_search_pipeline(
    pool, query: str, top_k: int = SEARCH_TOP_K,
    tags: list[str] | None = None,
    item_types: list[str] | None = None,
    accepted_only: bool = False,
    min_score: int | None = None,
) -> list[dict]:
    """Embed, retrieve (hybrid + optional filters), rerank. Returns top_k."""
    vector = embeddings.embed_query(query)
    candidates = await search.hybrid_search(
        pool, query, vector,
        k_retrieve=RETRIEVE_K, k_final=RETRIEVE_K,
        tags=tags, item_types=item_types,
        accepted_only=accepted_only, min_score=min_score,
    )
    return rerank.rerank(query, candidates, top_k=top_k)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "index.html",
        {"top_tags": getattr(request.app.state, "top_tags", [])},
    )


@app.post("/search", response_class=HTMLResponse)
async def search_endpoint(
    request: Request,
    response: Response,
    background: BackgroundTasks,
    query: str = Form(...),
    tags: Optional[List[str]] = Form(default=None),
    item_types: Optional[List[str]] = Form(default=None),
    accepted_only: bool = Form(default=False),
) -> HTMLResponse:
    tags = _clean_tags(tags)
    item_types = _clean_item_types(item_types)

    # Mint or read the session cookie so click/thumb events from this
    # rendered partial can be attributed to the same session.
    session_id = events.get_or_create_session(request, response)

    t0 = time.perf_counter()
    results = await run_search_pipeline(
        request.app.state.pool, query,
        tags=tags, item_types=item_types, accepted_only=accepted_only,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Log the search event in the background so it never blocks the user.
    background.add_task(
        events.log_event,
        request.app.state.pool,
        session_id=session_id,
        query=query,
        event_type="search",
        pipeline="hybrid_rerank",
        latency_ms=latency_ms,
        metadata={
            "tags": tags or [],
            "item_types": item_types or [],
            "accepted_only": bool(accepted_only),
            "n_results": len(results),
            # Impression set: needed so click/thumb events can be
            # joined back to what the user actually saw, and so CTR
            # can be computed offline (or feed result_ctr MV).
            "result_ids": [r["id"] for r in results],
        },
    )
    # Build the SSE URL for /summary with the same filters so the streamed
    # summary is grounded in the same retrieval the user sees.
    summary_qs = {"query": query}
    if tags:         summary_qs["tag"] = tags
    if item_types:   summary_qs["item_type"] = item_types
    if accepted_only: summary_qs["accepted_only"] = "1"
    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "query": query,
            "results": results,
            "applied_filters": {
                "tags": tags or [],
                "item_types": item_types or [],
                "accepted_only": accepted_only,
            },
            "summary_qs": summary_qs,
        },
    )


def _stage(name: str, status: str, **extra: object) -> dict[str, str]:
    payload: dict[str, object] = {"name": name, "status": status, **extra}
    return {"event": "stage", "data": json.dumps(payload)}


@app.get("/summary")
async def summary_endpoint(
    request: Request,
    query: str,
    tag: Optional[List[str]] = Query(default=None),
    item_type: Optional[List[str]] = Query(default=None),
    accepted_only: bool = Query(default=False),
) -> EventSourceResponse:
    pool = request.app.state.pool
    tags = _clean_tags(tag)
    item_types = _clean_item_types(item_type)

    async def event_stream():
        t_start = time.perf_counter()

        # --- Stage 1: embed query ------------------------------------------
        yield _stage("embed", "active")
        t0 = time.perf_counter()
        vector = embeddings.embed_query(query)
        embed_ms = int((time.perf_counter() - t0) * 1000)
        yield _stage(
            "embed",
            "done",
            ms=embed_ms,
            dim=len(vector),
            preview=[round(float(x), 3) for x in vector[:8]],
        )

        # --- Stage 2: hybrid retrieval (vector + BM25 + RRF) --------------
        yield _stage("search", "active",
                     tags=tags or [], item_types=item_types or [],
                     accepted_only=accepted_only)
        t0 = time.perf_counter()
        candidates = await search.hybrid_search(
            pool, query, vector,
            k_retrieve=RETRIEVE_K, k_final=RETRIEVE_K,
            tags=tags, item_types=item_types,
            accepted_only=accepted_only,
        )
        search_ms = int((time.perf_counter() - t0) * 1000)
        vector_hits = sum(1 for c in candidates if c.get("vector_rank") is not None)
        keyword_hits = sum(1 for c in candidates if c.get("keyword_rank") is not None)
        overlap = sum(
            1 for c in candidates
            if c.get("vector_rank") is not None and c.get("keyword_rank") is not None
        )
        yield _stage(
            "search",
            "done",
            ms=search_ms,
            k_retrieved=len(candidates),
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            overlap=overlap,
            top_score=round(candidates[0]["score"], 3) if candidates else None,
        )

        # --- Stage 3: rerank ----------------------------------------------
        yield _stage("rerank", "active", model=rerank.MODEL_NAME)
        t0 = time.perf_counter()
        results = rerank.rerank(query, candidates, top_k=SEARCH_TOP_K)
        rerank_ms = int((time.perf_counter() - t0) * 1000)
        yield _stage(
            "rerank",
            "done",
            ms=rerank_ms,
            model=rerank.MODEL_NAME,
            candidates_in=len(candidates),
            candidates_out=len(results),
            hits=[
                {
                    "title": r["title"],
                    "score": round(r["rerank_score"], 3),
                }
                for r in results
            ],
            k_used=PROMPT_TOP_K,
        )

        # --- Stage 4: prompt build -----------------------------------------
        yield _stage("prompt", "active")
        prompt = rag.build_prompt(query, results, k=PROMPT_TOP_K)
        yield _stage(
            "prompt",
            "done",
            chars=len(prompt),
            approx_tokens=len(prompt) // 4,
        )

        # --- Stage 5: LLM stream -------------------------------------------
        yield _stage("llm", "active", model=rag.MODEL)
        t0 = time.perf_counter()
        token_count = 0
        first_token_ms: int | None = None
        char_count = 0
        async for chunk in rag.stream_summary(prompt):
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
            token_count += 1
            char_count += len(chunk)
            yield {"event": "token", "data": chunk}

        total_ms = int((time.perf_counter() - t_start) * 1000)
        yield _stage(
            "llm",
            "done",
            model=rag.MODEL,
            chunks=token_count,
            chars=char_count,
            ttfb_ms=first_token_ms or 0,
            total_ms=total_ms,
        )
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())


# -----------------------------------------------------------------------------
# Feedback endpoints — anonymous, session-cookie only
# -----------------------------------------------------------------------------


def _assert_rate_limit(session_id: str) -> None:
    if not events.rate_limit_ok(session_id):
        raise HTTPException(
            status_code=429,
            detail="Too many events from this session; try again in a minute.",
        )


@app.post("/events/click", response_class=PlainTextResponse)
async def log_click(
    request: Request,
    response: Response,
    background: BackgroundTasks,
    query: str = Form(...),
    result_id: int = Form(...),
    position: int = Form(...),
) -> str:
    session_id = events.get_or_create_session(request, response)
    _assert_rate_limit(session_id)
    background.add_task(
        events.log_event,
        request.app.state.pool,
        session_id=session_id, query=query,
        event_type="click",
        result_id=result_id, result_position=position,
        pipeline="hybrid_rerank",
    )
    return ""


@app.post("/events/thumb", response_class=PlainTextResponse)
async def log_thumb(
    request: Request,
    response: Response,
    background: BackgroundTasks,
    query: str = Form(...),
    result_id: int = Form(...),
    position: int = Form(...),
    vote: str = Form(...),
) -> str:
    if vote not in ("up", "down"):
        raise HTTPException(status_code=400, detail="vote must be 'up' or 'down'")
    session_id = events.get_or_create_session(request, response)
    _assert_rate_limit(session_id)
    background.add_task(
        events.log_event,
        request.app.state.pool,
        session_id=session_id, query=query,
        event_type=f"thumb_{vote}",
        result_id=result_id, result_position=position,
        pipeline="hybrid_rerank",
    )
    return ""


# -----------------------------------------------------------------------------
# Admin endpoints — shared-secret auth, opt-in via ADMIN_TOKEN env var
# -----------------------------------------------------------------------------
#
# Deliberately not behind /api or mounted under a separate app — the scope
# is small (refresh the CTR MV, inspect per-result stats) and these paths
# don't contribute to user latency. When ADMIN_TOKEN is unset the endpoints
# 503 so a misconfigured deploy can't leak the data.


def _require_admin(request: Request) -> None:
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="admin endpoints disabled; set ADMIN_TOKEN to enable.",
        )
    provided = request.headers.get("X-Admin-Token", "")
    if not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="bad or missing X-Admin-Token")


@app.post("/admin/refresh_ctr")
async def admin_refresh_ctr(request: Request) -> JSONResponse:
    """Force a synchronous refresh of the result_ctr MV.

    Useful after a backfill, or to pick up new events before the periodic
    loop's next tick. Safe to call repeatedly — REFRESH CONCURRENTLY takes
    a light lock and the fallback non-concurrent path runs at most once,
    at first-ever refresh.
    """
    _require_admin(request)
    t0 = time.perf_counter()
    await refresh_result_ctr(request.app.state.pool)
    return JSONResponse({
        "status": "ok",
        "ms": int((time.perf_counter() - t0) * 1000),
    })


@app.get("/admin/metrics")
async def admin_metrics(request: Request) -> JSONResponse:
    """Feedback-loop health at a glance.

    Returns:
      - event_counts: total + last-24h count per event_type
      - ctr_rows: how many result_ids the MV has coverage for
      - recent_search_latency_ms: mean + p95 over the last 24h of
        'search' events (only rows where latency_ms is non-null)
    Cheap enough for a dashboard ping every few seconds; both queries hit
    the existing search_events indexes.
    """
    _require_admin(request)
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        event_rows = await conn.fetch(
            """
            SELECT event_type,
                   COUNT(*) FILTER (WHERE occurred_at >= NOW() - INTERVAL '24 hours')
                       AS last_24h,
                   COUNT(*) AS total
            FROM search_events
            GROUP BY event_type
            ORDER BY event_type
            """
        )
        ctr_rows = await conn.fetchval("SELECT COUNT(*) FROM result_ctr")
        latency = await conn.fetchrow(
            """
            SELECT AVG(latency_ms)::int AS mean_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::int
                       AS p95_ms,
                   COUNT(*) AS n
            FROM search_events
            WHERE event_type = 'search'
              AND latency_ms IS NOT NULL
              AND occurred_at >= NOW() - INTERVAL '24 hours'
            """
        )
    return JSONResponse({
        "event_counts": [
            {
                "event_type": r["event_type"],
                "last_24h": int(r["last_24h"]),
                "total": int(r["total"]),
            }
            for r in event_rows
        ],
        "ctr_rows": int(ctr_rows or 0),
        "recent_search_latency_ms": {
            "n": int(latency["n"] or 0),
            "mean": int(latency["mean_ms"] or 0) if latency["n"] else None,
            "p95":  int(latency["p95_ms"]  or 0) if latency["n"] else None,
        },
    })


@app.get("/admin/queries")
async def admin_top_queries(
    request: Request,
    since_hours: int = Query(24, ge=1, le=24 * 30),
    limit: int = Query(50, ge=1, le=500),
) -> JSONResponse:
    """Top queries by volume over the last `since_hours` hours.

    Handy for content ops: which queries drive traffic, how many clicks
    vs. thumbs they earn, median latency. The aggregate is keyed on a
    normalized (lowercased, trimmed) query string so minor capitalization
    differences collapse together.
    """
    _require_admin(request)
    sql = """
    WITH window AS (
        SELECT *
        FROM search_events
        WHERE occurred_at >= NOW() - ($1 || ' hours')::interval
    ),
    searches AS (
        SELECT lower(trim(query)) AS q,
               COUNT(*)           AS searches,
               AVG(latency_ms)::int AS mean_latency_ms
        FROM window
        WHERE event_type = 'search'
        GROUP BY 1
    ),
    engagements AS (
        SELECT lower(trim(query)) AS q,
               COUNT(*) FILTER (WHERE event_type = 'click')       AS clicks,
               COUNT(*) FILTER (WHERE event_type = 'thumb_up')    AS thumbs_up,
               COUNT(*) FILTER (WHERE event_type = 'thumb_down')  AS thumbs_down
        FROM window
        WHERE event_type IN ('click', 'thumb_up', 'thumb_down')
        GROUP BY 1
    )
    SELECT COALESCE(s.q, e.q) AS query_norm,
           COALESCE(s.searches, 0)           AS searches,
           COALESCE(e.clicks, 0)             AS clicks,
           COALESCE(e.thumbs_up, 0)          AS thumbs_up,
           COALESCE(e.thumbs_down, 0)        AS thumbs_down,
           s.mean_latency_ms                 AS mean_latency_ms
    FROM searches s
    FULL OUTER JOIN engagements e ON e.q = s.q
    WHERE COALESCE(s.q, e.q) IS NOT NULL AND COALESCE(s.q, e.q) <> ''
    ORDER BY COALESCE(s.searches, 0) DESC, COALESCE(e.clicks, 0) DESC
    LIMIT $2
    """
    async with request.app.state.pool.acquire() as conn:
        rows = await conn.fetch(sql, str(since_hours), limit)
    return JSONResponse({
        "since_hours": since_hours,
        "queries": [
            {
                "query": r["query_norm"],
                "searches": int(r["searches"]),
                "clicks": int(r["clicks"]),
                "thumbs_up": int(r["thumbs_up"]),
                "thumbs_down": int(r["thumbs_down"]),
                "mean_latency_ms": (
                    int(r["mean_latency_ms"]) if r["mean_latency_ms"] is not None else None
                ),
            }
            for r in rows
        ],
    })


@app.get("/admin/stats/{result_id}")
async def admin_stats(request: Request, result_id: int) -> JSONResponse:
    """Inspect the CTR MV row for one result_id.

    Returns impressions, clicks, thumbs, and derived CTR (with the same
    shrinkage denominator the ranker uses, so the number shown here is
    the exact value feeding the bump). `found: false` for results that
    have no events yet.
    """
    _require_admin(request)
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT result_id, impressions, clicks, thumbs_up, thumbs_down "
            "FROM result_ctr WHERE result_id = $1",
            result_id,
        )
    if row is None:
        return JSONResponse({"result_id": result_id, "found": False})
    clicks = int(row["clicks"])
    impressions = int(row["impressions"])
    thumbs_up = int(row["thumbs_up"])
    thumbs_down = int(row["thumbs_down"])
    # Match the ranker's shrinkage floor of 20 impressions.
    denom = max(impressions, 20)
    return JSONResponse({
        "result_id": int(row["result_id"]),
        "found": True,
        "impressions": impressions,
        "clicks": clicks,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "ctr_shrunk": round(clicks / denom, 6),
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
