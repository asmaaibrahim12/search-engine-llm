from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from app import embeddings, rag, search
from app.db import create_pool, ensure_schema

load_dotenv()

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

PROMPT_TOP_K = 5
SEARCH_TOP_K = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings.get_model()
    app.state.pool = await create_pool()
    await ensure_schema(app.state.pool)
    try:
        yield
    finally:
        await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/search", response_class=HTMLResponse)
async def search_endpoint(request: Request, query: str = Form(...)) -> HTMLResponse:
    results = await search.semantic_search(
        request.app.state.pool, query, k=SEARCH_TOP_K
    )
    return templates.TemplateResponse(
        request,
        "results.html",
        {"query": query, "results": results},
    )


def _stage(name: str, status: str, **extra: object) -> dict[str, str]:
    """Build an SSE stage event. Payload is JSON-encoded so the browser can
    parse a single 'data:' line."""
    payload: dict[str, object] = {"name": name, "status": status, **extra}
    return {"event": "stage", "data": json.dumps(payload)}


@app.get("/summary")
async def summary_endpoint(request: Request, query: str) -> EventSourceResponse:
    pool = request.app.state.pool

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

        # --- Stage 2: vector search ----------------------------------------
        yield _stage("search", "active")
        t0 = time.perf_counter()
        results = await search.search_by_vector(pool, vector, k=SEARCH_TOP_K)
        search_ms = int((time.perf_counter() - t0) * 1000)
        yield _stage(
            "search",
            "done",
            ms=search_ms,
            hits=[
                {"title": r["title"], "score": round(r["score"], 3)} for r in results
            ],
            k_retrieved=len(results),
            k_used=PROMPT_TOP_K,
        )

        # --- Stage 3: prompt build -----------------------------------------
        yield _stage("prompt", "active")
        prompt = rag.build_prompt(query, results, k=PROMPT_TOP_K)
        yield _stage(
            "prompt",
            "done",
            chars=len(prompt),
            approx_tokens=len(prompt) // 4,  # rough heuristic; 1 token ≈ 4 chars
        )

        # --- Stage 4: LLM stream -------------------------------------------
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
