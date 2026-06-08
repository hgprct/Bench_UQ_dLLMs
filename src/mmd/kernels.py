"""Kernel functions for MMD computation: embedding-based, KLE graph, and NLI graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.mmd.embeddings import TextEmbedder, build_embedder, normalize_rows
from src.mmd.mmd import KernelBlocks


class TextKernel(Protocol):
    """Protocol for text kernel functions that produce kernel matrix blocks."""
    name: str
    def blocks(self, x_texts: list[str], y_texts: list[str]) -> KernelBlocks: ...


def _linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.matmul(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).T)


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * np.matmul(x, y.T), 0.0).astype(np.float32, copy=False)


def _resolve_rbf_sigma_sq(embeddings: np.ndarray, bandwidth: str | float) -> float:
    try:
        sigma = float(bandwidth)
    except (TypeError, ValueError):
        sigma = float("nan")
    if math.isfinite(sigma):
        if sigma <= 0:
            raise ValueError(f"RBF bandwidth must be positive, got {bandwidth}")
        return sigma * sigma
    if str(bandwidth).strip().lower() != "median":
        raise ValueError(f"Unsupported RBF bandwidth '{bandwidth}'. Use median or a positive number.")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] < 2:
        return 1.0
    distances = _pairwise_sq_dists(embeddings, embeddings)
    positive = distances[distances > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def _rbf(x: np.ndarray, y: np.ndarray, sigma_sq: float) -> np.ndarray:
    return np.exp(-_pairwise_sq_dists(x, y) / (2.0 * float(sigma_sq))).astype(np.float32, copy=False)


@dataclass
class EmbeddingKernel:
    """Classic text kernel computed after embedding answers (RBF, linear, cosine)."""
    name: str
    embedding_model: str = "hashing:256"
    rbf_bandwidth: str = "median"
    embedding_batch_size: int = 64
    embedding_device: str = "auto"
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        self.embedder: TextEmbedder = build_embedder(
            self.embedding_model,
            batch_size=self.embedding_batch_size,
            device=self.embedding_device,
            normalize=self.normalize_embeddings,
        )

    def blocks(self, x_texts: list[str], y_texts: list[str]) -> KernelBlocks:
        embeddings = self.embedder.encode([*x_texts, *y_texts])
        x = embeddings[:len(x_texts)]
        y = embeddings[len(x_texts):]
        return self._kernel_blocks(x, y, embeddings)

    def blocks_from_embeddings(self, x_emb: np.ndarray, y_emb: np.ndarray) -> KernelBlocks:
        combined = np.vstack([x_emb, y_emb])
        return self._kernel_blocks(
            np.asarray(x_emb, dtype=np.float32),
            np.asarray(y_emb, dtype=np.float32),
            combined,
        )

    def resolve_sigma_sq(self, reference_embeddings: np.ndarray) -> float | None:
        kind = self.name.removeprefix("embedding-")
        if kind != "rbf":
            return None
        return _resolve_rbf_sigma_sq(
            np.asarray(reference_embeddings, dtype=np.float32),
            self.rbf_bandwidth,
        )

    def cross_kernel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        sigma_sq: float | None = None,
    ) -> np.ndarray:
        kind = self.name.removeprefix("embedding-")
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if kind == "linear":
            return _linear(x, y)
        if kind == "cosine":
            return _linear(normalize_rows(x), normalize_rows(y))
        if kind == "rbf":
            if sigma_sq is None:
                sigma_sq = _resolve_rbf_sigma_sq(np.vstack([x, y]), self.rbf_bandwidth)
            return _rbf(x, y, sigma_sq)
        raise ValueError(f"Unsupported embedding kernel '{self.name}'")

    def _kernel_blocks(self, x: np.ndarray, y: np.ndarray, all_embeddings: np.ndarray) -> KernelBlocks:
        kind = self.name.removeprefix("embedding-")
        if kind == "linear":
            return KernelBlocks(_linear(x, x), _linear(y, y), _linear(x, y))
        if kind == "cosine":
            x_norm = normalize_rows(x)
            y_norm = normalize_rows(y)
            return KernelBlocks(_linear(x_norm, x_norm), _linear(y_norm, y_norm), _linear(x_norm, y_norm))
        if kind == "rbf":
            sigma_sq = _resolve_rbf_sigma_sq(all_embeddings, self.rbf_bandwidth)
            return KernelBlocks(_rbf(x, x, sigma_sq), _rbf(y, y, sigma_sq), _rbf(x, y, sigma_sq))
        raise ValueError(f"Unsupported embedding kernel '{self.name}'")


@dataclass
class KLEGraphKernel:
    """Heat or Matern graph kernel from a Laplacian over answer text embeddings."""
    name: str
    embedding_model: str = "hashing:256"
    embedding_batch_size: int = 64
    embedding_device: str = "auto"
    normalize_embeddings: bool = True
    heat_t: float = 0.3
    matern_kappa: float = 1.0
    matern_nu: float = 1.0
    normalized_laplacian: bool = False

    def __post_init__(self) -> None:
        self.embedder: TextEmbedder = build_embedder(
            self.embedding_model,
            batch_size=self.embedding_batch_size,
            device=self.embedding_device,
            normalize=self.normalize_embeddings,
        )

    def blocks(self, x_texts: list[str], y_texts: list[str]) -> KernelBlocks:
        embeddings = normalize_rows(self.embedder.encode([*x_texts, *y_texts]))
        kernel = self._kernel_from_embeddings(embeddings)
        n_x = len(x_texts)
        return KernelBlocks(xx=kernel[:n_x, :n_x], yy=kernel[n_x:, n_x:], xy=kernel[:n_x, n_x:])

    def _kernel_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        weights = np.maximum(np.matmul(embeddings, embeddings.T), 0.0).astype(np.float64, copy=False)
        np.fill_diagonal(weights, 0.0)
        laplacian = _laplacian(weights, self.normalized_laplacian)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if self.name == "kle-heat":
            factors = np.exp(-float(self.heat_t) * eigenvalues)
        elif self.name == "kle-matern":
            shift = 2.0 * float(self.matern_nu) / float(self.matern_kappa) ** 2
            factors = np.power(np.maximum(shift + eigenvalues, 1e-12), -float(self.matern_nu))
        else:
            raise ValueError(f"Unsupported KLE kernel '{self.name}'")
        kernel = (eigenvectors * factors) @ eigenvectors.T
        return 0.5 * (kernel + kernel.T)


def _laplacian(weights: np.ndarray, normalized: bool) -> np.ndarray:
    """Compute the (optionally normalized) graph Laplacian."""
    degree = weights.sum(axis=1)
    if not normalized:
        return np.diag(degree) - weights
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degree > 1e-12, 1.0 / np.sqrt(degree), 0.0)
    return np.eye(weights.shape[0], dtype=np.float64) - (inv_sqrt[:, None] * weights * inv_sqrt[None, :])


def build_text_kernel(
    name: str,
    *,
    embedding_model: str = "hashing:256",
    rbf_bandwidth: str = "median",
    embedding_batch_size: int = 64,
    embedding_device: str = "auto",
    normalize_embeddings: bool = True,
    heat_t: float = 0.3,
    matern_kappa: float = 1.0,
    matern_nu: float = 1.0,
    normalized_laplacian: bool = False,
) -> TextKernel:
    """Factory for text kernel instances."""
    normalized = str(name).strip().lower().replace("_", "-")
    if normalized in {"rbf", "linear", "cosine"}:
        normalized = f"embedding-{normalized}"
    if normalized in {"embedding-rbf", "embedding-linear", "embedding-cosine"}:
        return EmbeddingKernel(
            name=normalized, embedding_model=embedding_model,
            rbf_bandwidth=rbf_bandwidth, embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device, normalize_embeddings=normalize_embeddings,
        )
    if normalized in {"kle-heat", "kle-matern"}:
        for value, label in ((heat_t, "heat_t"), (matern_kappa, "matern_kappa"), (matern_nu, "matern_nu")):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label} must be positive, got {value}")
        return KLEGraphKernel(
            name=normalized, embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size, embedding_device=embedding_device,
            normalize_embeddings=normalize_embeddings,
            heat_t=heat_t, matern_kappa=matern_kappa, matern_nu=matern_nu,
            normalized_laplacian=normalized_laplacian,
        )
    raise ValueError(
        f"Unsupported kernel '{name}'. Use embedding-rbf, embedding-linear, "
        "embedding-cosine, kle-heat, or kle-matern."
    )
