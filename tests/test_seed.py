"""Tests for src.seed -- deterministic seeding."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seed import seed_everything


class TestSeedEverything:
    def test_numpy_deterministic(self):
        seed_everything(42)
        a = np.random.rand(10)
        seed_everything(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_stdlib_deterministic(self):
        seed_everything(42)
        a = [random.random() for _ in range(10)]
        seed_everything(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_different_seeds_differ(self):
        seed_everything(42)
        a = np.random.rand(10)
        seed_everything(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_sets_pythonhashseed(self):
        import os
        seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"
