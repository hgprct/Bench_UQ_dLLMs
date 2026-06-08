"""Tests for src.timing.clock."""

import time

import numpy as np
import pytest

from timing.clock import Timer, autodetect_nli_batch


class TestTimer:
    def test_records_positive_elapsed(self):
        with Timer(device=None, sync=False) as t:
            time.sleep(0.01)
        assert t.elapsed is not None
        assert t.elapsed >= 0.01

    def test_no_sync_when_cpu(self):
        with Timer(device="cpu", sync=True) as t:
            pass
        assert t.elapsed is not None
        assert t.elapsed >= 0.0

    def test_exception_still_sets_elapsed(self):
        t = Timer(device=None, sync=False)
        try:
            with t:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert t.elapsed is not None


class FakeNLI:
    """Records max batch ever passed; raises OOM above a configurable cap."""

    def __init__(self, oom_above: int):
        self.oom_above = oom_above
        self.max_seen = 0

    def batch_probabilities(self, premises, hypotheses, *, batch_size):
        self.max_seen = max(self.max_seen, batch_size)
        if batch_size > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        return np.zeros((len(premises), 3), dtype=np.float32), np.zeros(len(premises), dtype=np.int64)


class TestAutodetect:
    def test_locks_largest_successful(self):
        nli = FakeNLI(oom_above=80)
        bs = autodetect_nli_batch(nli, device=None, candidates=(32, 64, 128, 256))
        assert bs == 64

    def test_raises_when_all_fail(self):
        nli = FakeNLI(oom_above=16)
        with pytest.raises(RuntimeError, match="OOMs at smallest candidate"):
            autodetect_nli_batch(nli, device=None, candidates=(32, 64))

    def test_picks_max_when_all_succeed(self):
        nli = FakeNLI(oom_above=10_000)
        bs = autodetect_nli_batch(nli, device=None, candidates=(32, 64, 128))
        assert bs == 128
