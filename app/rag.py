from __future__ import annotations

import os
from typing import AsyncIterator, Iterable, Mapping

import anthropic

MODEL = "claude-opus-4-7"
MAX_TOKENS = 600  # ~3-5 sentences. Hard cap to prevent multi-section essays.

PROMPT_TEMPLATE = """Answer the user's query in 2-4 sentences, grounded only in the search results below.

Rules:
- Cite claims inline with [1], [2], etc., matching the numbered results.
- Plain prose only. No headings, no bullet lists, no "Overview" or "Conclusion" sections.
- If the results don't actually answer the query, say so in one sentence.
- Be direct. No preamble like "Based on the search results…".

## Query
{query}

## Search Results
{context}

## Answer"""


def build_prompt(
    query: str,
    results: Iterable[Mapping[str, object]],
    k: int = 5,
) -> str:
    top = list(results)[:k]
    context = "\n".join(
        f'[{i + 1}] {r["title"]}: {r.get("body") or ""}\n' for i, r in enumerate(top)
    )
    return PROMPT_TEMPLATE.format(query=query, context=context).strip()


def _client() -> anthropic.AsyncAnthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Configure it in your environment "
            "or in the Railway dashboard."
        )
    return anthropic.AsyncAnthropic(api_key=key)


async def stream_summary(prompt: str) -> AsyncIterator[str]:
    client = _client()
    async with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
