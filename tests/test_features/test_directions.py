"""Tests for src.features.directions -- single source of truth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from features.directions import (
    BASE_FEATURES,
    FEATURE_DIRECTIONS,
    NLI_FEATURES,
    SAMPLING_FEATURES,
    TRAJECTORY_ONLY_FEATURES,
    TRAJECTORY_PREFIXES,
    all_feature_names,
    build_feature_directions,
)


class TestFeatureDirections:
    def test_msp_is_confidence(self):
        assert FEATURE_DIRECTIONS["msp"] == "confidence"

    def test_ecc_is_confidence(self):
        assert FEATURE_DIRECTIONS["ecc"] == "confidence"

    def test_se_conditional_is_confidence(self):
        assert FEATURE_DIRECTIONS["se-conditional"] == "confidence"

    def test_mcnse_is_uncertainty(self):
        assert FEATURE_DIRECTIONS["mcnse"] == "uncertainty"

    def test_se_marginal_is_uncertainty(self):
        assert FEATURE_DIRECTIONS["se-marginal"] == "uncertainty"

    def test_ad_is_uncertainty(self):
        assert FEATURE_DIRECTIONS["ad"] == "uncertainty"

    def test_all_others_are_uncertainty(self):
        confidence_features = {"msp", "ecc", "se-conditional"}
        for name, direction in FEATURE_DIRECTIONS.items():
            if name not in confidence_features:
                assert direction == "uncertainty", f"{name} should be uncertainty"

    def test_all_base_and_sampling_have_directions(self):
        for name in BASE_FEATURES + SAMPLING_FEATURES:
            assert name in FEATURE_DIRECTIONS

    def test_trajectory_only_features_have_directions(self):
        for name in TRAJECTORY_ONLY_FEATURES:
            assert name in FEATURE_DIRECTIONS

    def test_nli_features_subset_of_sampling(self):
        for name in NLI_FEATURES:
            assert name in SAMPLING_FEATURES

    def test_mcnse_not_in_nli_features(self):
        assert "mcnse" not in NLI_FEATURES
        assert "mcnse" in SAMPLING_FEATURES


class TestAllFeatureNames:
    def test_order(self):
        names = all_feature_names()
        assert names[:3] == BASE_FEATURES
        assert names[3:3 + len(SAMPLING_FEATURES)] == SAMPLING_FEATURES
        expected_total = (
            len(BASE_FEATURES)
            + len(SAMPLING_FEATURES)
            + len(TRAJECTORY_PREFIXES) * len(NLI_FEATURES)
            + len(TRAJECTORY_ONLY_FEATURES)
            + 2 * len(TRAJECTORY_ONLY_FEATURES)
        )
        assert len(names) == expected_total

    def test_ad_variants_present(self):
        names = all_feature_names()
        assert "ad" in names
        assert "full-ad" in names
        assert "selected-ad" in names


class TestBuildFeatureDirections:
    def test_base_directions(self):
        dirs = build_feature_directions()
        assert dirs["msp"] == "confidence"
        assert dirs["mcnse"] == "uncertainty"
        assert dirs["se-conditional"] == "confidence"
        assert dirs["ecc"] == "confidence"
        assert dirs["kle-heat"] == "uncertainty"

    def test_ad_directions(self):
        dirs = build_feature_directions()
        assert dirs["ad"] == "uncertainty"
        assert dirs["full-ad"] == "uncertainty"
        assert dirs["selected-ad"] == "uncertainty"
