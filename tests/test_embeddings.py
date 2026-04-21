from __future__ import annotations

import math

import numpy as np

from app.embeddings import EMBEDDING_DIM, normalize
from tests.conftest import needs_model


def test_normalize_unit_vector():
    v = np.array([3.0, 4.0])
    result = normalize(v)
    assert math.isclose(math.sqrt(sum(x * x for x in result)), 1.0, rel_tol=1e-6)


def test_normalize_zero_vector_returns_zero():
    v = np.array([0.0, 0.0, 0.0])
    result = normalize(v)
    assert result == [0.0, 0.0, 0.0]


def test_normalize_returns_floats():
    v = np.array([1.0, 2.0, 3.0])
    result = normalize(v)
    assert all(isinstance(x, float) for x in result)


@needs_model
def test_embed_query_shape():
    from app.embeddings import embed_query

    result = embed_query("hello world")
    assert len(result) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in result)


@needs_model
def test_embed_query_deterministic():
    from app.embeddings import embed_query

    a = embed_query("trail running shoes")
    b = embed_query("trail running shoes")
    assert a == b


@needs_model
def test_embed_query_is_normalized():
    from app.embeddings import embed_query

    v = embed_query("any text")
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)
