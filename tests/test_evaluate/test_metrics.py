"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from evaluate.metrics import (
    brier_score,
    evaluate_score,
    expected_calibration_error,
    kendall_tau,
    minmax_normalize,
    spearman_rho,
)


class TestMinmaxNormalize:
    def test_basic(self):
        result = minmax_normalize([0.0, 5.0, 10.0])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_constant_returns_half(self):
        result = minmax_normalize([3.0, 3.0, 3.0])
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])


class TestECE:
    def test_perfect_calibration(self):
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        scores_01 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
        ece = expected_calibration_error(labels, scores_01, n_bins=2)
        assert ece is not None
        assert ece == pytest.approx(0.2)

    def test_too_few_samples(self):
        assert expected_calibration_error([1], np.array([0.5])) is None


class TestBrierScore:
    def test_perfect_predictions(self):
        labels = [0, 0, 1, 1]
        scores_01 = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(labels, scores_01) == 0.0

    def test_worst_predictions(self):
        labels = [0, 0, 1, 1]
        scores_01 = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(labels, scores_01) == 1.0


class TestSpearmanRho:
    def test_perfect_positive(self):
        labels = [0, 0, 1, 1]
        scores = [1.0, 2.0, 3.0, 4.0]
        rho = spearman_rho(labels, scores)
        assert rho is not None
        assert rho > 0.85

    def test_insufficient_data(self):
        assert spearman_rho([1, 1], [0.5, 0.6]) is None


class TestKendallTau:
    def test_perfect_positive(self):
        labels = [0, 0, 1, 1]
        scores = [1.0, 2.0, 3.0, 4.0]
        tau = kendall_tau(labels, scores)
        assert tau is not None
        assert tau > 0.8

    def test_insufficient_data(self):
        assert kendall_tau([1, 1], [0.5, 0.6]) is None


class TestEvaluateScore:
    def test_returns_all_metrics(self):
        labels = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6, 0.35, 0.65]
        result = evaluate_score(labels, scores)
        for key in ("auroc", "auprc", "prr", "ece", "brier", "spearman_rho", "kendall_tau"):
            assert key in result
            assert result[key] is not None

    def test_single_class_returns_none(self):
        labels = [0, 0, 0]
        scores = [0.1, 0.2, 0.3]
        result = evaluate_score(labels, scores)
        assert result["auroc"] is None
        assert result["ece"] is None
