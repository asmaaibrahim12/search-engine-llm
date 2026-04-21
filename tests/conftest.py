from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

needs_db = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="Set TEST_DATABASE_URL (or DATABASE_URL) to a Postgres+pgvector instance",
)

# Heavy sentence-transformers / torch import is gated behind an env var so the
# pure-Python unit tests run fast in CI and on laptops without the model.
needs_model = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 to run tests that load sentence-transformers",
)


class FakeAsyncStream:
    """Mimics anthropic.AsyncAnthropic.messages.stream() as an async context
    manager yielding an async iterable of text tokens."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self) -> AsyncIterator[str]:
        tokens = self._tokens

        class _AIter:
            def __aiter__(self_inner):
                return self_inner

            _idx = 0

            async def __anext__(self_inner):
                if self_inner._idx >= len(tokens):
                    raise StopAsyncIteration
                token = tokens[self_inner._idx]
                self_inner._idx += 1
                return token

        return _AIter()


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch rag._client() to return a mock whose messages.stream records calls
    and yields fake tokens."""
    from app import rag

    mock_client = MagicMock()
    captured: dict[str, Any] = {}

    def stream(**kwargs):
        captured.update(kwargs)
        tokens = getattr(mock_client, "_tokens", ["Hel", "lo", " world"])
        return FakeAsyncStream(tokens)

    mock_client.messages.stream.side_effect = stream
    mock_client._tokens = ["Hel", "lo", " world"]
    mock_client._captured = captured

    monkeypatch.setattr(rag, "_client", lambda: mock_client)
    return mock_client
