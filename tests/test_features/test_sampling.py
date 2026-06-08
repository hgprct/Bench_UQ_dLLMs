"""Tests for src.features.sampling — NLI-based UQ features."""

import math

import numpy as np
import pytest

from features.sampling import (
    CachedEntailmentModel,
    compute_sampling_features,
    _greedy_semantic_clusters,
    _von_neumann_entropy,
)


class MockNLIModel:
    """Deterministic NLI model returning entailment when texts match."""

    entailment_id = 2

    def __init__(self, entailment_matrix=None):
        self._matrix = entailment_matrix

    def batch_probabilities(self, premises, hypotheses, *, batch_size=8):
        n = len(premises)
        if n == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)
        probs = np.zeros((n, 3), dtype=np.float32)
        classes = np.zeros(n, dtype=np.int64)
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            if self._matrix is not None:
                pass
            elif p == h:
                probs[i] = [0.05, 0.05, 0.9]
                classes[i] = 2
            else:
                probs[i] = [0.7, 0.2, 0.1]
                classes[i] = 0
        if self._matrix is not None:
            probs[:, 2] = self._matrix.ravel()[:n]
            probs[:, 0] = 1.0 - probs[:, 2]
            classes = (probs[:, 2] > 0.5).astype(np.int64) * 2
        return probs, classes


class TestGreedySemanticClusters:
    def test_all_identical(self):
        mat = np.ones((3, 3))
        classes = _greedy_semantic_clusters(mat, threshold=0.5)
        assert classes == [0, 0, 0]

    def test_all_different(self):
        mat = np.eye(3)
        classes = _greedy_semantic_clusters(mat, threshold=0.5)
        assert classes == [0, 1, 2]

    def test_two_clusters(self):
        mat = np.array([
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        classes = _greedy_semantic_clusters(mat, threshold=0.5)
        assert classes[0] == classes[1]
        assert classes[2] != classes[0]

    def test_empty(self):
        mat = np.empty((0, 0))
        assert _greedy_semantic_clusters(mat) == []


class TestVonNeumannEntropy:
    def test_uniform_factors(self):
        factors = np.array([1.0, 1.0, 1.0, 1.0])
        result = _von_neumann_entropy(factors)
        assert result == pytest.approx(math.log(4), rel=1e-6)

    def test_single_factor(self):
        assert _von_neumann_entropy(np.array([5.0])) == pytest.approx(0.0)

    def test_zero_factors_ignored(self):
        factors = np.array([0.0, 0.0, 1.0])
        assert _von_neumann_entropy(factors) == pytest.approx(0.0)

    def test_empty(self):
        assert _von_neumann_entropy(np.array([])) == 0.0


class TestComputeSamplingFeatures:
    def test_too_few_texts(self):
        result = compute_sampling_features(["hello"], MockNLIModel())
        assert all(v is None for v in result.values())
        assert set(result.keys()) == {"se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern"}

    def test_identical_texts(self):
        texts = ["The answer is 42"] * 5
        result = compute_sampling_features(texts, MockNLIModel())
        assert result["se-conditional"] == pytest.approx(1.0)
        assert result["se-marginal"] == pytest.approx(0.0)
        assert result["ecc"] is not None
        assert result["eigval"] is not None
        assert result["kle-heat"] is not None
        assert result["kle-matern"] is not None

    def test_all_different_texts(self):
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        result = compute_sampling_features(texts, MockNLIModel())
        assert result["se-conditional"] == pytest.approx(0.0)
        assert result["se-marginal"] > 0.0
        assert result["ecc"] is not None

    def test_returns_all_keys(self):
        texts = ["a", "b", "c"]
        result = compute_sampling_features(texts, MockNLIModel())
        expected_keys = {"se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern"}
        assert set(result.keys()) == expected_keys

    def test_se_conditional_bounds(self):
        texts = ["same"] * 10
        result = compute_sampling_features(texts, MockNLIModel())
        assert 0.0 <= result["se-conditional"] <= 1.0

    def test_se_marginal_nonnegative(self):
        texts = ["a", "b", "c", "d"]
        result = compute_sampling_features(texts, MockNLIModel())
        assert result["se-marginal"] >= 0.0

    def test_subsampling(self):
        texts = [f"text_{i}" for i in range(20)]
        result = compute_sampling_features(texts, MockNLIModel(), max_items=5)
        assert set(result.keys()) == {"se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern"}


class TestCachedEntailmentModel:
    def test_caches_results(self):
        call_count = 0
        base = MockNLIModel()
        original_fn = base.batch_probabilities

        def counting_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)

        base.batch_probabilities = counting_fn
        cached = CachedEntailmentModel(base)
        cached.batch_probabilities(["a"], ["b"])
        cached.batch_probabilities(["a"], ["b"])
        assert call_count == 1

    def test_delegates_attributes(self):
        base = MockNLIModel()
        cached = CachedEntailmentModel(base)
        assert cached.entailment_id == 2

    def test_empty_input(self):
        cached = CachedEntailmentModel(MockNLIModel())
        probs, classes = cached.batch_probabilities([], [])
        assert probs.shape[0] == 0
