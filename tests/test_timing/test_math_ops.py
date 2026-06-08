"""Equivalence tests: timing math kernels must match compute_sampling_features."""

import numpy as np
import pytest

from features.sampling import compute_sampling_features
from timing import math_ops


class StubNLI:
    entailment_id = 2

    def __init__(self, entail_matrix: np.ndarray):
        self.m = entail_matrix.astype(np.float64)
        self.n = self.m.shape[0]

    def batch_probabilities(self, premises, hypotheses, *, batch_size):
        n = self.n
        probs = np.zeros((n * n, 3), dtype=np.float32)
        flat = self.m.flatten()
        probs[:, 2] = flat
        probs[:, 0] = 1.0 - flat
        classes = (flat > 0.5).astype(np.int64) * 2
        return probs, classes


def _sym_entail_from_compute_path(texts):
    """Run compute_sampling_features's matrix-prep path to get sym_entail for stub NLI."""
    n = len(texts)
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=(n, n))
    entail = raw.copy()
    np.fill_diagonal(entail, 1.0)
    sym = (entail + entail.T) / 2.0
    return sym, raw


class TestNliFeatureEquivalence:

    def _matched_run(self, texts, entail_matrix):
        nli = StubNLI(entail_matrix)
        reference = compute_sampling_features(texts, nli, nli_batch_size=8)

        n = len(texts)
        sym = (entail_matrix + entail_matrix.T) / 2.0
        np.fill_diagonal(sym, 1.0)
        sym = (sym + sym.T) / 2.0

        return reference, sym

    def test_se_marginal_matches(self):
        texts = [f"t{i}" for i in range(6)]
        m = np.array([
            [1.0, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0, 0.7],
            [0.1, 0.1, 0.1, 0.1, 0.7, 1.0],
        ])
        ref, sym = self._matched_run(texts, m)
        _, val = math_ops.time_se_marginal(sym)
        assert val == pytest.approx(ref["se-marginal"], abs=1e-9)

    def test_se_conditional_matches(self):
        texts = [f"t{i}" for i in range(5)]
        m = np.full((5, 5), 0.9)
        ref, sym = self._matched_run(texts, m)
        _, val = math_ops.time_se_conditional(sym)
        assert val == pytest.approx(ref["se-conditional"], abs=1e-9)

    def test_ecc_eigval_kle_match(self):
        rng = np.random.default_rng(123)
        n = 7
        m = rng.uniform(0, 1, size=(n, n))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        texts = [f"t{i}" for i in range(n)]
        ref, sym = self._matched_run(texts, m)

        _, ecc = math_ops.time_ecc(sym)
        _, eig = math_ops.time_eigval(sym)
        _, kh = math_ops.time_kle_heat(sym)
        _, km = math_ops.time_kle_matern(sym)
        assert ecc == pytest.approx(ref["ecc"], abs=1e-6)
        assert eig == pytest.approx(ref["eigval"], abs=1e-6)
        assert kh == pytest.approx(ref["kle-heat"], abs=1e-6)
        assert km == pytest.approx(ref["kle-matern"], abs=1e-6)


class TestTokenKernels:
    def test_msp_perplexity_basic(self):
        logprobs = np.array([-0.1, -0.2, -0.3, -0.4])
        _, msp = math_ops.time_msp(logprobs, vis=[0, 1, 2, 3])
        _, ppl = math_ops.time_perplexity(logprobs, vis=[0, 1, 2, 3])
        assert msp == pytest.approx(float(np.exp(-1.0)))
        assert ppl == pytest.approx(float(np.exp(0.25)))

    def test_msp_empty_visible(self):
        _, msp = math_ops.time_msp(np.array([-0.1]), vis=[])
        assert msp is None

    def test_mte_basic(self):
        topk = np.array([[1.0, 1.0], [2.0, 0.0]])
        _, mte = math_ops.time_mte(topk, vis=[0, 1])
        assert mte is not None and mte > 0.0

    def test_mcnse_average(self):
        lps = [np.array([-0.5, -0.5]), np.array([-1.0, -1.0])]
        tids = [np.array([10, 11]), np.array([12, 13])]
        _, val = math_ops.time_mcnse(lps, tids, special_ids=set(), eos_ids=set())
        assert val == pytest.approx(0.75)
