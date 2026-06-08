"""Tests for src.mmd.mmd -- MMD^2 estimators."""

import numpy as np

from mmd.mmd import KernelBlocks, estimate_mmd2, finite_mean_std, mmd2_biased, mmd2_unbiased


class TestMmd2Biased:
    def test_identical_distributions(self):
        k = np.ones((3, 3), dtype=np.float64)
        blocks = KernelBlocks(xx=k, yy=k, xy=k)
        assert mmd2_biased(blocks) == 0.0

    def test_different_distributions(self):
        xx = np.eye(3, dtype=np.float64)
        yy = np.eye(3, dtype=np.float64)
        xy = np.zeros((3, 3), dtype=np.float64)
        blocks = KernelBlocks(xx=xx, yy=yy, xy=xy)
        result = mmd2_biased(blocks)
        assert result > 0

    def test_empty_returns_nan(self):
        blocks = KernelBlocks(
            xx=np.array([]).reshape(0, 0),
            yy=np.ones((2, 2)),
            xy=np.array([]).reshape(0, 2),
        )
        assert np.isnan(mmd2_biased(blocks))


class TestMmd2Unbiased:
    def test_requires_multiple_trajectories(self):
        k = np.ones((1, 1), dtype=np.float64)
        blocks = KernelBlocks(xx=k, yy=k, xy=k)
        result = mmd2_unbiased(blocks, np.array([0]), np.array([0]))
        assert np.isnan(result)


class TestEstimateMmd2:
    def test_dispatch_biased(self):
        k = np.ones((2, 2), dtype=np.float64)
        blocks = KernelBlocks(xx=k, yy=k, xy=k)
        assert estimate_mmd2(blocks, "biased") == 0.0

    def test_invalid_estimator(self):
        k = np.ones((2, 2))
        blocks = KernelBlocks(xx=k, yy=k, xy=k)
        try:
            estimate_mmd2(blocks, "invalid")
            assert False
        except ValueError:
            pass


class TestFiniteMeanStd:
    def test_basic(self):
        mean, std, count = finite_mean_std([1.0, 2.0, 3.0])
        assert abs(mean - 2.0) < 1e-10
        assert count == 3

    def test_filters_nan(self):
        mean, std, count = finite_mean_std([1.0, float("nan"), 3.0])
        assert abs(mean - 2.0) < 1e-10
        assert count == 2

    def test_all_nan(self):
        mean, std, count = finite_mean_std([float("nan")])
        assert np.isnan(mean)
        assert count == 0
