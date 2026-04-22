from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks, FastAPI, Form, HTTPException, Query, Request, Response,
)
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from app import embeddings, events, rag, rerank, search
from app.db import create_pool, ensure_schema
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
    try:
        yield
    finally:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
