"""Tests for src.features.trajectory — averaged dissimilarity (AD)."""

import numpy as np
import pytest

from features.trajectory import (
    compute_averaged_dissimilarity,
    compute_step_dissimilarities,
)


class MockNLIModel:
    """Deterministic NLI model: entailment when texts match, contradiction otherwise."""

    entailment_id = 2

    def batch_probabilities(self, premises, hypotheses, *, batch_size=8):
        n = len(premises)
        if n == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)
        probs = np.zeros((n, 3), dtype=np.float32)
        classes = np.zeros(n, dtype=np.int64)
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            if p == h:
                probs[i] = [0.05, 0.05, 0.9]
                classes[i] = 2
            else:
                probs[i] = [0.7, 0.2, 0.1]
                classes[i] = 0
        return probs, classes


class TestComputeStepDissimilarities:
    def test_all_identical_to_final(self):
        steps = ("answer", "answer", "answer")
        dissim = compute_step_dissimilarities(steps, "answer", MockNLIModel())
        np.testing.assert_array_equal(dissim, [0.0, 0.0, 0.0])

    def test_all_different_from_final(self):
        steps = ("wrong1", "wrong2", "wrong3")
        dissim = compute_step_dissimilarities(steps, "answer", MockNLIModel())
        assert dissim.shape == (3,)
        for d in dissim:
            assert d == pytest.approx(1.0 - 0.1, rel=1e-5)

    def test_mixed_steps(self):
        steps = ("answer", "wrong", "answer")
        dissim = compute_step_dissimilarities(steps, "answer", MockNLIModel())
        assert dissim[0] == pytest.approx(0.0)
        assert dissim[1] == pytest.approx(1.0 - 0.1, rel=1e-5)
        assert dissim[2] == pytest.approx(0.0)

    def test_invalid_final_answer(self):
        steps = ("a", "b", "c")
        dissim = compute_step_dissimilarities(steps, "", MockNLIModel())
        np.testing.assert_array_equal(dissim, [0.0, 0.0, 0.0])

    def test_invalid_step_answers(self):
        steps = ("", "None", "valid")
        dissim = compute_step_dissimilarities(steps, "answer", MockNLIModel())
        assert dissim[0] == 0.0
        assert dissim[1] == 0.0
        assert dissim[2] > 0.0

    def test_empty_steps(self):
        dissim = compute_step_dissimilarities((), "answer", MockNLIModel())
        assert dissim.shape == (0,)

    def test_symmetrization(self):
        """The mock gives identical forward/backward (both texts are different),
        so symmetrization should match either direction."""
        steps = ("wrong",)
        dissim = compute_step_dissimilarities(steps, "answer", MockNLIModel())
        assert dissim[0] == pytest.approx(1.0 - 0.1, rel=1e-5)


class TestComputeAveragedDissimilarity:
    def test_uniform_average(self):
        steps = ("wrong1", "wrong2", "answer")
        result = compute_averaged_dissimilarity(steps, "answer", MockNLIModel())
        assert result is not None
        expected_dissim = 1.0 - 0.1
        assert result == pytest.approx((expected_dissim + expected_dissim + 0.0) / 3.0, rel=1e-5)

    def test_weighted_average(self):
        steps = ("wrong1", "wrong2", "answer")
        weights = np.array([0.5, 0.3, 0.2])
        result = compute_averaged_dissimilarity(
            steps, "answer", MockNLIModel(), weights=weights,
        )
        assert result is not None
        d = 1.0 - 0.1
        expected = 0.5 * d + 0.3 * d + 0.2 * 0.0
        assert result == pytest.approx(expected, rel=1e-5)

    def test_none_for_invalid_final(self):
        result = compute_averaged_dissimilarity(("a", "b"), "", MockNLIModel())
        assert result is None

    def test_none_for_empty_steps(self):
        result = compute_averaged_dissimilarity((), "answer", MockNLIModel())
        assert result is None

    def test_weight_length_mismatch(self):
        steps = ("a", "b")
        weights = np.array([0.5, 0.3, 0.2])
        result = compute_averaged_dissimilarity(
            steps, "answer", MockNLIModel(), weights=weights,
        )
        assert result is None

    def test_all_identical_returns_zero(self):
        steps = ("answer", "answer", "answer")
        result = compute_averaged_dissimilarity(steps, "answer", MockNLIModel())
        assert result == pytest.approx(0.0)

    def test_weighted_all_identical_returns_zero(self):
        steps = ("answer", "answer", "answer")
        weights = np.array([0.4, 0.3, 0.3])
        result = compute_averaged_dissimilarity(
            steps, "answer", MockNLIModel(), weights=weights,
        )
        assert result == pytest.approx(0.0)
