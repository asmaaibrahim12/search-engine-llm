from __future__ import annotations

import os
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the embedding model so the first real query isn't slow.
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
    results = await search.semantic_search(request.app.state.pool, query, k=10)
    return templates.TemplateResponse(
        request,
        "results.html",
        {"query": query, "results": results},
    )


@app.get("/summary")
async def summary_endpoint(request: Request, query: str) -> EventSourceResponse:
    results = await search.semantic_search(request.app.state.pool, query, k=10)
    prompt = rag.build_prompt(query, results, k=5)

    async def event_stream():
        async for token in rag.stream_summary(prompt):
            # SSE splits on newlines; send raw token as data. Client appends.
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
