from __future__ import annotations

import os
from typing import AsyncIterator, Iterable, Mapping

import anthropic

from app.text import strip_html

MODEL = "claude-opus-4-7"
MAX_TOKENS = 400  # ~2-3 sentences. Hard cap so the UI can't ever render a
                  # multi-section essay, regardless of prompt adherence.

PROMPT_TEMPLATE = """Answer the user's query in at most two short sentences. Write one plain paragraph.

Hard rules — violating any is a failure:
- NO headings. Do not emit '#', '##', '###', or labels like 'Overview', 'Summary', 'Conclusion', 'Key Points'.
- NO bullet points or numbered lists.
- NO preamble like 'Based on the search results…'. Start with the answer.
- Cite inline using the bracketed tags: [1], [2], etc.
- Prefer results tagged 'accepted answer' or 'answer' over 'question' when they conflict.
- If the results do not address the query, say so in one sentence and stop.

## Query
{query}

## Search Results
{context}

## Answer"""


def _citation_header(idx: int, r: Mapping[str, object]) -> str:
    """Build the '[1 — accepted answer]' style marker for each reference.

    The header tells Claude the citation's provenance: question (someone
    asked this), answer (someone replied), or accepted answer (the
    community confirmed it solved the question). Claude's prompt says to
    prefer accepted answers when summarizing; this is how it can tell."""
    kind = r.get("item_type") or "item"
    if r.get("is_accepted"):
        label = "accepted answer"
    elif kind == "answer":
        label = "answer"
    else:
        label = "question"
    return f"[{idx} — {label}]"


def build_prompt(
    query: str,
    results: Iterable[Mapping[str, object]],
    k: int = 5,
) -> str:
    top = list(results)[:k]
    parts = []
    for i, r in enumerate(top):
        header = _citation_header(i + 1, r)
        title = (r.get("title") or "").strip()
        # Strip HTML from body before the LLM sees it — Stack Exchange
        # bodies are stored as HTML, which is wasted tokens and noise.
        body = strip_html(r.get("body"))[:600]
        if title:
            parts.append(f"{header} {title}: {body}\n")
        else:
            parts.append(f"{header} {body}\n")
    context = "\n".join(parts)
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
