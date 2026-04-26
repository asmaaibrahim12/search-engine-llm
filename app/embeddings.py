from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# sentence_transformers (and its torch dependency) is a ~1.5 GB install.
# The CI's "unit" tier intentionally doesn't ship it, so we defer the
# import until get_model() is actually called. Module-level constants
# and pure-Python helpers like normalize() stay importable without it.
if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

# BAAI/bge-base-en-v1.5 — retrieval-tuned sentence encoder, 768-dim.
# Strictly better than the older roberta-base-nli-stsb-mean-tokens on
# MTEB retrieval benchmarks, same output dimensionality so no schema change.
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# BGE v1.5 was trained with an asymmetric query/passage setup: queries get
# a short instruction prefix, passages do not. Omitting the prefix on
# queries costs 2–3 nDCG points in practice.
# See: https://huggingface.co/BAAI/bge-base-en-v1.5
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def normalize(vector: np.ndarray) -> list[float]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return [float(x) for x in vector]
    return [float(x) for x in np.divide(vector, norm)]


def embed_query(query: str) -> list[float]:
    """Encode a user query for retrieval. Applies the BGE query prefix."""
    text = QUERY_PREFIX + query
    return normalize(get_model().encode(text))


def embed_passage(text: str) -> list[float]:
    """Encode a single document passage. No prefix."""
    return normalize(get_model().encode(text))


def embed_batch_passages(texts: list[str]) -> list[list[float]]:
    """Batch-encode document passages. No prefix."""
    raw = get_model().encode(texts, show_progress_bar=False, batch_size=32)
    return [normalize(v) for v in raw]


# Kept for backward compatibility with callers that imported embed_batch.
embed_batch = embed_batch_passages
