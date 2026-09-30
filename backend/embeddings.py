"""Embedding service backed by FastEmbed (ONNX, low memory).

We use FastEmbed + BAAI/bge-small-en-v1.5 instead of sentence-transformers/
PyTorch: ONNX runtime keeps RAM usage suitable for a 512 MB free-tier host.
The model is loaded once and reused (module-level lazy singleton).
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_model() -> "TextEmbedding":
    with _lock:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=MODEL_NAME)


def embed_one(text: str) -> list[float]:
    """Return the embedding vector for a single text."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of texts (single model load)."""
    model = _get_model()
    return [vector.tolist() for vector in model.embed(texts)]
