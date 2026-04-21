from __future__ import annotations

import os
from typing import AsyncIterator, Iterable, Mapping

import anthropic

MODEL = "claude-opus-4-7"
MAX_TOKENS = 16000

PROMPT_TEMPLATE = """# Instructions
For the given user query and search results, create a helpful summary of the results relevant to the query.

## User Query: {query}

## Search Results:
{context}

## Summary Generation :
- Generate a comprehensive summary of the user's query topic using the provided search results.
- Use the reference tags (e.g., [1], [2]) to cite specific information from the search results in the summary.
- Ensure all information is cross-referenced for consistency. Avoid including contradictory statements.
- Prioritize factual accuracy, grounding the summary in the content of the provided search results.
- Structure the summary with an introductory overview, detailed exploration of key points, and a concluding statement.

Please create a summary following these guidelines to ensure consistency and accuracy.

ANSWER:"""


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
