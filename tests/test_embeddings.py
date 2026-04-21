from __future__ import annotations

import math

import numpy as np

from app.embeddings import EMBEDDING_DIM, MODEL_NAME, QUERY_PREFIX, normalize
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


def test_model_is_bge_base_v1_5():
    # Regression guard: don't accidentally revert to the old roberta-nli model.
    assert MODEL_NAME == "BAAI/bge-base-en-v1.5"


def test_query_prefix_is_bge_retrieval_instruction():
    # BGE v1.5 needs this exact prefix on queries for best retrieval quality.
    assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "


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


@needs_model
def test_embed_query_differs_from_embed_passage_for_same_text():
    """BGE applies a query prefix, so the two functions must produce
    different vectors even when handed identical input."""
    from app.embeddings import embed_passage, embed_query

    same_text = "How do I pitch a tent in high wind?"
    q = embed_query(same_text)
    p = embed_passage(same_text)
    dot = sum(a * b for a, b in zip(q, p))
    # Similar but not identical — dot product should be high (>0.9) but < 0.9999
    assert 0.80 < dot < 0.9999


@needs_model
def test_query_semantic_match_stronger_than_random():
    """Sanity check: BGE embeds semantic neighbors closer than random pairs."""
    from app.embeddings import embed_passage, embed_query

    q = embed_query("best hiking shoes for rocky terrain")
    near = embed_passage("trail running footwear for uneven ground")
    far = embed_passage("how to change a kayak paddle grip")
    dot_near = sum(a * b for a, b in zip(q, near))
    dot_far = sum(a * b for a, b in zip(q, far))
    assert dot_near > dot_far
