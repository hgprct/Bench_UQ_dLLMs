"""Tests for src.mmd.scopes -- step scope selection."""

from mmd.scopes import StepScope, build_scopes, select_steps


class TestBuildScopes:
    def test_default(self):
        scopes = build_scopes(k=8, num_ranges=5)
        assert len(scopes) == 7  # full + random + 5 ranges
        assert scopes[0].mode == "full"
        assert scopes[1].mode == "random"
        assert scopes[1].k == 8
        assert all(s.mode == "range" for s in scopes[2:])

    def test_names(self):
        scopes = build_scopes(k=4, num_ranges=3)
        names = [s.name for s in scopes]
        assert names == ["full", "random:4", "range-0", "range-1", "range-2"]


class TestSelectSteps:
    def test_full(self):
        steps = select_steps(10, "full")
        assert steps == list(range(10))

    def test_random_deterministic(self):
        import random
        rng = random.Random(42)
        a = select_steps(20, "random", k=5, rng=rng)
        rng = random.Random(42)
        b = select_steps(20, "random", k=5, rng=rng)
        assert a == b
        assert len(a) == 5
        assert a == sorted(a)

    def test_range_first(self):
        steps = select_steps(20, "range", k=5, range_index=0, num_ranges=3)
        assert steps == [0, 1, 2, 3, 4]

    def test_range_last(self):
        steps = select_steps(20, "range", k=5, range_index=2, num_ranges=3)
        assert steps == [15, 16, 17, 18, 19]

    def test_k_clamped(self):
        steps = select_steps(3, "random", k=10)
        assert len(steps) == 3

    def test_invalid_mode(self):
        try:
            select_steps(10, "invalid", k=5)
            assert False
        except ValueError:
            pass

    def test_missing_k(self):
        try:
            select_steps(10, "random")
            assert False
        except ValueError:
            pass
