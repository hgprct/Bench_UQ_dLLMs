"""Text embedding backends for kernel-based MMD computation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class TextEmbedder(Protocol):
    """Protocol for text embedding backends."""
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class HashingTextEmbedder:
    """Deterministic bag-of-words hashing embedder for tests and smoke runs."""
    dim: int = 256
    normalize: bool = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), int(self.dim)), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = str(text).lower().split() or [str(text).lower()]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                index = value % int(self.dim)
                sign = 1.0 if (value // int(self.dim)) % 2 == 0 else -1.0
                matrix[row, index] += sign
        return normalize_rows(matrix) if self.normalize else matrix


class SentenceTransformerEmbedder:
    """Sentence-transformers embedding backend (lazy import)."""

    def __init__(self, model_name: str, *, batch_size: int, device: str, normalize: bool) -> None:
        from sentence_transformers import SentenceTransformer
        kwargs = {} if device == "auto" else {"device": device}
        self.model = SentenceTransformer(model_name, **kwargs)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def build_embedder(
    model_name: str,
    *,
    batch_size: int = 64,
    device: str = "auto",
    normalize: bool = True,
) -> TextEmbedder:
    """Build an embedder by model name. 'hashing' or 'hashing:N' for the deterministic stub."""
    normalized = str(model_name).strip().lower()
    if normalized.startswith("hashing"):
        dim = 256
        if ":" in normalized:
            try:
                dim = int(normalized.split(":", 1)[1])
            except ValueError:
                dim = 256
        return HashingTextEmbedder(dim=dim, normalize=normalize)
    return SentenceTransformerEmbedder(
        model_name, batch_size=batch_size, device=device, normalize=normalize,
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a matrix."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
