"""Tests for src.timing.runner: aggregation, decoding, weight loading."""

import numpy as np
import pytest

from timing.runner import (
    GenTimings, PromptRecord, aggregate, decode_x0_steps, load_qp_weights,
)


class FakeTokenizer:
    def decode(self, token_ids, *, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in token_ids)


class TestDecodeX0Steps:
    def test_decodes_each_step(self):
        arr = np.array([[[1, 2, 3], [4, 5, 6]]])
        steps = decode_x0_steps(arr, FakeTokenizer(), sample_idx=0)
        assert steps == ["1 2 3", "4 5 6"]

    def test_returns_empty_on_bad_shape(self):
        arr = np.array([1, 2, 3])
        assert decode_x0_steps(arr, FakeTokenizer()) == []

    def test_out_of_range_sample(self):
        arr = np.array([[[1]]])
        assert decode_x0_steps(arr, FakeTokenizer(), sample_idx=5) == []


class TestLoadQpWeights:
    def test_returns_none_when_dir_missing(self, tmp_path):
        assert load_qp_weights(None, budget_k=8) is None
        assert load_qp_weights(tmp_path / "nope", budget_k=8) is None

    def test_returns_none_when_budget_none(self, tmp_path):
        np.savez(tmp_path / "trained_weights.npz", w_star=np.ones(10))
        assert load_qp_weights(tmp_path, budget_k=None) is None

    def test_picks_top_k_active_indices(self, tmp_path):
        w = np.array([0.0, 0.5, 0.1, 0.3, 0.05, 0.0])
        np.savez(tmp_path / "trained_weights.npz", w_star=w)
        # active = w[:-1] = [0.0, 0.5, 0.1, 0.3, 0.05]; top-3 = {1, 2, 3}
        selected = load_qp_weights(tmp_path, budget_k=3)
        assert selected == [1, 2, 3]

    def test_caps_k_at_active_size(self, tmp_path):
        w = np.array([0.5, 0.5, 0.0])
        np.savez(tmp_path / "trained_weights.npz", w_star=w)
        selected = load_qp_weights(tmp_path, budget_k=100)
        assert selected == [0, 1]


class TestAggregate:
    def test_single_feature_two_prompts(self):
        recs = [
            PromptRecord(0, "msp", "token", 0, 0, 0.0, 0.0, 0.01),
            PromptRecord(1, "msp", "token", 0, 0, 0.0, 0.0, 0.03),
        ]
        s = aggregate(recs)
        row = s[("token", "msp")]
        assert row["n_prompts"] == 2
        assert row["time_math_mean"] == pytest.approx(0.02)
        # ddof=1 sample std: sqrt(sum((xi-mean)^2)/(n-1))
        assert row["time_math_std"] == pytest.approx(0.01 * 2**0.5, rel=1e-6)
        assert row["time_total_mean"] == pytest.approx(0.02)

    def test_multiple_families_grouped_separately(self):
        recs = [
            PromptRecord(0, "msp", "token", 0, 0, 0.0, 0.0, 0.01),
            PromptRecord(0, "ecc", "iid", 19, 400, 5.0, 0.5, 0.001),
        ]
        s = aggregate(recs)
        assert set(s.keys()) == {("token", "msp"), ("iid", "ecc")}
        assert s[("iid", "ecc")]["n_extra_gen"] == 19
        assert s[("iid", "ecc")]["n_nli_pairs"] == 400
        assert s[("iid", "ecc")]["time_total_mean"] == pytest.approx(5.501)

    def test_single_prompt_uses_ddof_zero(self):
        recs = [PromptRecord(0, "msp", "token", 0, 0, 0.0, 0.0, 0.05)]
        s = aggregate(recs)
        assert s[("token", "msp")]["time_math_std"] == 0.0

    def test_empty_returns_empty(self):
        assert aggregate([]) == {}


class TestGenTimings:
    def test_fields(self):
        gt = GenTimings(greedy=1.5, single_stochastic=1.6, batched_stochastic=4.2, n_samples=19)
        assert gt.greedy == 1.5
        assert gt.single_stochastic == 1.6
        assert gt.batched_stochastic == 4.2
        assert gt.n_samples == 19
