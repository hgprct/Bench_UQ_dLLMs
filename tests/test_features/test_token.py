"""Tests for src.features.token — token-level UQ features."""

import math

import numpy as np
import pytest

from features.token import (
    compute_mcnse,
    compute_msp,
    compute_mte,
    compute_perplexity,
    special_ids_from_config,
    visible_positions,
    _softmax_entropy,
)


class TestVisiblePositions:
    def test_basic(self):
        tids = [10, 20, 30, 40]
        pos = visible_positions(tids, special_ids={99}, eos_ids={50})
        assert pos == [0, 1, 2, 3]

    def test_stops_at_eos(self):
        tids = [10, 20, 50, 40]
        pos = visible_positions(tids, special_ids={99}, eos_ids={50})
        assert pos == [0, 1]

    def test_skips_special(self):
        tids = [10, 99, 20, 30]
        pos = visible_positions(tids, special_ids={99}, eos_ids={50})
        assert pos == [0, 2, 3]

    def test_eos_at_start(self):
        tids = [50, 10, 20]
        pos = visible_positions(tids, special_ids=set(), eos_ids={50})
        assert pos == []

    def test_all_special(self):
        tids = [99, 99, 99]
        pos = visible_positions(tids, special_ids={99}, eos_ids={50})
        assert pos == []

    def test_empty(self):
        pos = visible_positions([], special_ids={99}, eos_ids={50})
        assert pos == []

    def test_numpy_input(self):
        tids = np.array([10, 20, 50, 40])
        pos = visible_positions(tids, special_ids=set(), eos_ids={50})
        assert pos == [0, 1]

    def test_multiple_eos_ids(self):
        tids = [10, 20, 51, 40]
        pos = visible_positions(tids, special_ids=set(), eos_ids={50, 51})
        assert pos == [0, 1]


class TestSpecialIdsFromConfig:
    def test_basic_config(self):
        config = {"eos_token_ids": [2, 3], "mask_id": 126336, "pad_token_id": 0}
        special, eos = special_ids_from_config(config)
        assert eos == {2, 3}
        assert 126336 in special
        assert 0 in special
        assert 2 in special
        assert 3 in special

    def test_empty_config(self):
        special, eos = special_ids_from_config({})
        assert special == set()
        assert eos == set()

    def test_scalar_eos(self):
        config = {"eos_token_ids": 2}
        _, eos = special_ids_from_config(config)
        assert eos == {2}

    def test_numpy_values(self):
        config = {"eos_token_ids": np.array([2, 3])}
        _, eos = special_ids_from_config(config)
        assert eos == {2, 3}


class TestComputeMsp:
    def test_all_certain(self):
        logprobs = [0.0, 0.0, 0.0]
        result = compute_msp(logprobs, visible_positions=[0, 1, 2])
        assert result == pytest.approx(1.0)

    def test_typical_logprobs(self):
        logprobs = [math.log(0.9), math.log(0.8), math.log(0.7)]
        result = compute_msp(logprobs, visible_positions=[0, 1, 2])
        assert result == pytest.approx(0.9 * 0.8 * 0.7)

    def test_subset_visible(self):
        logprobs = [math.log(0.9), math.log(0.1), math.log(0.8)]
        result = compute_msp(logprobs, visible_positions=[0, 2])
        assert result == pytest.approx(0.9 * 0.8)

    def test_empty_visible(self):
        assert compute_msp([0.0, 0.0], visible_positions=[]) is None

    def test_inf_logprobs_filtered(self):
        logprobs = [float("-inf"), math.log(0.5)]
        result = compute_msp(logprobs, visible_positions=[0, 1])
        assert result == pytest.approx(0.5)

    def test_all_inf_returns_none(self):
        logprobs = [float("-inf"), float("-inf")]
        assert compute_msp(logprobs, visible_positions=[0, 1]) is None


class TestComputePerplexity:
    def test_certain_tokens(self):
        logprobs = [0.0, 0.0, 0.0]
        result = compute_perplexity(logprobs, visible_positions=[0, 1, 2])
        assert result == pytest.approx(1.0)

    def test_uniform_logprobs(self):
        lp = math.log(0.1)
        result = compute_perplexity([lp, lp, lp], visible_positions=[0, 1, 2])
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_empty_visible(self):
        assert compute_perplexity([0.0], visible_positions=[]) is None

    def test_all_inf_returns_none(self):
        assert compute_perplexity([float("-inf")], visible_positions=[0]) is None


class TestComputeMte:
    def test_peaked_logits(self):
        logits = np.array([[100.0, -100.0, -100.0], [100.0, -100.0, -100.0]])
        result = compute_mte(logits, visible_positions=[0, 1])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_uniform_logits(self):
        k = 4
        logits = np.zeros((3, k))
        result = compute_mte(logits, visible_positions=[0, 1, 2])
        assert result == pytest.approx(math.log(k), rel=1e-6)

    def test_empty_visible(self):
        logits = np.zeros((3, 4))
        assert compute_mte(logits, visible_positions=[]) is None

    def test_1d_logits_returns_none(self):
        assert compute_mte(np.array([1.0, 2.0]), visible_positions=[0]) is None

    def test_subset_visible(self):
        peaked = np.array([[100.0, -100.0], [0.0, 0.0], [100.0, -100.0]])
        result = compute_mte(peaked, visible_positions=[1])
        assert result == pytest.approx(math.log(2), rel=1e-6)


class TestComputeMcnse:
    def test_single_item_certain(self):
        logprobs = np.array([0.0, 0.0, 0.0])
        token_ids = np.array([10, 20, 30])
        result = compute_mcnse([logprobs], [token_ids], special_ids={99}, eos_ids={50})
        assert result == pytest.approx(0.0)

    def test_single_item_uncertain(self):
        lp = math.log(0.1)
        logprobs = np.array([lp, lp, lp])
        token_ids = np.array([10, 20, 30])
        result = compute_mcnse([logprobs], [token_ids], special_ids={99}, eos_ids={50})
        assert result == pytest.approx(-lp, rel=1e-6)

    def test_multiple_items_averaged(self):
        lp1 = np.array([math.log(0.9), math.log(0.9)])
        lp2 = np.array([math.log(0.1), math.log(0.1)])
        tids = np.array([10, 20])
        result = compute_mcnse([lp1, lp2], [tids, tids], special_ids=set(), eos_ids={50})
        expected = (-math.log(0.9) + (-math.log(0.1))) / 2
        assert result == pytest.approx(expected, rel=1e-6)

    def test_empty_items(self):
        assert compute_mcnse([], [], special_ids=set(), eos_ids=set()) is None

    def test_all_special_tokens(self):
        logprobs = np.array([0.0, 0.0])
        token_ids = np.array([99, 99])
        assert compute_mcnse([logprobs], [token_ids], special_ids={99}, eos_ids=set()) is None

    def test_eos_stops_averaging(self):
        logprobs = np.array([math.log(0.5), math.log(0.1), math.log(0.9)])
        token_ids = np.array([10, 50, 20])
        result = compute_mcnse([logprobs], [token_ids], special_ids=set(), eos_ids={50})
        assert result == pytest.approx(-math.log(0.5))

    def test_inf_logprobs_skipped(self):
        logprobs = np.array([float("-inf"), math.log(0.5)])
        token_ids = np.array([10, 20])
        result = compute_mcnse([logprobs], [token_ids], special_ids=set(), eos_ids=set())
        assert result == pytest.approx(-math.log(0.5))

    def test_direction_is_uncertainty(self):
        lp_confident = np.array([math.log(0.99)] * 4)
        lp_uncertain = np.array([math.log(0.01)] * 4)
        tids = np.array([10, 20, 30, 40])
        confident = compute_mcnse([lp_confident], [tids], special_ids=set(), eos_ids=set())
        uncertain = compute_mcnse([lp_uncertain], [tids], special_ids=set(), eos_ids=set())
        assert uncertain > confident


class TestSoftmaxEntropy:
    def test_uniform(self):
        logits = np.zeros((2, 4))
        ent = _softmax_entropy(logits)
        assert ent.shape == (2,)
        np.testing.assert_allclose(ent, math.log(4), rtol=1e-6)

    def test_peaked(self):
        logits = np.array([[1000.0, -1000.0]])
        ent = _softmax_entropy(logits)
        assert ent[0] == pytest.approx(0.0, abs=1e-6)

    def test_empty(self):
        ent = _softmax_entropy(np.array([]).reshape(0, 3))
        assert ent.size == 0
