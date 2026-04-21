from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "roberta-base-nli-stsb-mean-tokens"
EMBEDDING_DIM = 768

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def normalize(vector: np.ndarray) -> list[float]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return [float(x) for x in vector]
    return [float(x) for x in np.divide(vector, norm)]


def embed_query(query: str) -> list[float]:
    return normalize(get_model().encode(query))


def embed_batch(texts: list[str]) -> list[list[float]]:
    raw = get_model().encode(texts, show_progress_bar=False, batch_size=32)
    return [normalize(v) for v in raw]
