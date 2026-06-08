"""Tests for the selection module: discretization, baselines, kernel matrices."""

from __future__ import annotations

import numpy as np
import pytest

from selection.evaluate import (
    binarized_weight_vector,
    discretize,
    last_k_selection,
    random_selection,
    randomized_rounding,
    top_k_selection,
    uniform_selection,
)
from selection.train import compute_delta_U, compute_kernel_matrices, verify_psd


class TestDiscretization:
    def test_top_k_selection(self):
        w = np.array([0.0, 0.5, 0.1, 0.3, 0.1])
        result = top_k_selection(w, 2)
        assert len(result) == 2
        assert 1 in result
        assert 3 in result
        assert list(result) == sorted(result)

    def test_uniform_selection(self):
        result = uniform_selection(10, 3)
        assert len(result) <= 3
        assert all(0 <= idx < 9 for idx in result)

    def test_last_k_selection(self):
        result = last_k_selection(10, 3)
        np.testing.assert_array_equal(result, [6, 7, 8])

    def test_random_selection_deterministic(self):
        r1 = random_selection(20, 5, seed=42)
        r2 = random_selection(20, 5, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_randomized_rounding_deterministic(self):
        w = np.array([0.0, 0.5, 0.1, 0.3, 0.1])
        r1 = randomized_rounding(w, 2, seed=123)
        r2 = randomized_rounding(w, 2, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_discretize_baseline_methods(self):
        w = np.zeros(10)
        for method in ("uniform", "random", "last_k"):
            result = discretize(w, 3, method, seed=0)
            assert len(result) <= 3
            assert all(idx < 10 for idx in result)

    def test_discretize_learned_methods(self):
        w = np.array([0.0, 0.5, 0.1, 0.3, 0.1])
        result = discretize(w, 2, "top_k")
        assert len(result) == 2

    def test_discretize_unknown_method(self):
        w = np.zeros(5)
        with pytest.raises(ValueError, match="Unknown"):
            discretize(w, 2, "nonexistent")


class TestBinarizedWeightVector:
    def test_basic(self):
        steps = np.array([1, 3])
        w = binarized_weight_vector(steps, 5, 2)
        assert w[1] == pytest.approx(0.5)
        assert w[3] == pytest.approx(0.5)
        assert w[0] == 0.0
        assert w[2] == 0.0
        assert w[4] == 0.0


class TestKernelMatrices:
    def test_shapes_and_symmetry(self):
        from mmd.kernels import EmbeddingKernel

        rng = np.random.default_rng(42)
        N, T, D = 3, 4, 8
        emb = rng.standard_normal((N, T, D)).astype(np.float32)
        kernel = EmbeddingKernel(name="embedding-linear", embedding_model="hashing:8")

        A_V, A_U, C = compute_kernel_matrices(emb, kernel)

        assert A_V.shape == (T, T)
        assert A_U.shape == (T, T)
        assert C.shape == (T, T)
        np.testing.assert_allclose(A_V, A_V.T, atol=1e-10)
        np.testing.assert_allclose(A_U, A_U.T, atol=1e-10)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_degenerate_single_trajectory(self):
        from mmd.kernels import EmbeddingKernel

        emb = np.random.default_rng(0).standard_normal((1, 3, 4)).astype(np.float32)
        kernel = EmbeddingKernel(name="embedding-linear", embedding_model="hashing:4")
        A_V, A_U, C = compute_kernel_matrices(emb, kernel)
        np.testing.assert_array_equal(A_V, np.zeros((3, 3)))


class TestDeltaU:
    def test_basic(self):
        C = np.array([[2.0, 1.0], [1.0, 3.0]])
        A_U = np.array([[1.0, 0.5], [0.5, 2.0]])
        delta = compute_delta_U(C, A_U)
        np.testing.assert_allclose(delta, C - A_U)


class TestVerifyPSD:
    def test_identity_is_psd(self):
        result = verify_psd(np.eye(3), "identity")
        assert result["is_psd"] is True
        assert result["min_eigenvalue"] > 0

    def test_negative_definite(self):
        result = verify_psd(-np.eye(3), "neg_identity")
        assert result["is_psd"] is False
