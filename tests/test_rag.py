from __future__ import annotations

import pytest

from app import rag


SEARCH_RESULTS = [
    {"title": "Trail shoes vs hiking boots", "body": "When to pick which."},
    {"title": "How to lace hiking boots", "body": "Technique walkthrough."},
    {"title": "Waterproof hiking shoes", "body": "Membrane choices."},
    {"title": "Wide-foot hiking boots", "body": "Fit guide for EE."},
    {"title": "Breaking in new boots", "body": "Tips to avoid blisters."},
    {"title": "Post-hike shoe care", "body": "Drying and storage."},
    {"title": "Sock liners", "body": "Friction reduction."},
]


def test_build_prompt_includes_citation_tags():
    prompt = rag.build_prompt("best hiking shoes", SEARCH_RESULTS, k=5)
    for i in range(1, 6):
        assert f"[{i}]" in prompt
    assert "[6]" not in prompt


def test_build_prompt_respects_k():
    prompt = rag.build_prompt("q", SEARCH_RESULTS, k=3)
    for i in range(1, 4):
        assert f"[{i}]" in prompt
    assert "[4]" not in prompt


def test_build_prompt_interpolates_query_and_body():
    prompt = rag.build_prompt("my unique query 123", SEARCH_RESULTS, k=2)
    assert "my unique query 123" in prompt
    assert "Trail shoes vs hiking boots" in prompt
    assert "When to pick which." in prompt


def test_build_prompt_handles_missing_body():
    results = [{"title": "Only a title", "body": None}]
    prompt = rag.build_prompt("q", results, k=1)
    assert "[1] Only a title:" in prompt


@pytest.mark.asyncio
async def test_stream_summary_yields_tokens(fake_claude):
    fake_claude._tokens = ["Hel", "lo", " world"]
    chunks = []
    async for token in rag.stream_summary("anything"):
        chunks.append(token)
    assert "".join(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_stream_summary_uses_opus_4_7_and_max_tokens(fake_claude):
    async for _ in rag.stream_summary("prompt"):
        break
    kwargs = fake_claude._captured
    assert kwargs["model"] == "claude-opus-4-7"
    # max_tokens is deliberately small to keep summaries to a few sentences,
    # not a multi-section essay. If this goes up, the UI text will explode.
    assert kwargs["max_tokens"] == rag.MAX_TOKENS
    assert kwargs["max_tokens"] <= 1000


@pytest.mark.asyncio
async def test_stream_summary_no_sampling_params(fake_claude):
    async for _ in rag.stream_summary("prompt"):
        break
    kwargs = fake_claude._captured
    # Opus 4.7 400s on these — guard against accidental reintroduction
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


@pytest.mark.asyncio
async def test_stream_summary_sends_user_message(fake_claude):
    async for _ in rag.stream_summary("the actual prompt"):
        break
    messages = fake_claude._captured["messages"]
    assert messages == [{"role": "user", "content": "the actual prompt"}]


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        rag._client()
