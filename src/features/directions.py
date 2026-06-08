"""Single source of truth for UQ feature names and their orientation.

Confidence features (higher = more confident, e.g. msp) are negated by the
evaluator to orient as uncertainty. All other features are already oriented
as uncertainty (higher = more uncertain).
"""

from __future__ import annotations

BASE_FEATURES = ["msp", "perplexity", "mte"]
NLI_FEATURES = ["se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern"]
SAMPLING_FEATURES = ["mcnse"] + NLI_FEATURES
TRAJECTORY_PREFIXES = ("full", "selected", "random")
TRAJECTORY_ONLY_FEATURES = ["ad"]

FEATURE_DIRECTIONS: dict[str, str] = {
    "msp": "confidence",
    "perplexity": "uncertainty",
    "mte": "uncertainty",
    "mcnse": "uncertainty",
    "se-marginal": "uncertainty",
    "se-conditional": "confidence",
    "ecc": "confidence",
    "eigval": "uncertainty",
    "kle-heat": "uncertainty",
    "kle-matern": "uncertainty",
    "ad": "uncertainty",
}

IID_SAMPLE_FEATURES = frozenset(SAMPLING_FEATURES)


def all_feature_names() -> list[str]:
    """All features in canonical order: base + sampling + trajectory-prefixed + trajectory-only."""
    base = BASE_FEATURES + SAMPLING_FEATURES
    prefixed = [f"{p}-{f}" for p in TRAJECTORY_PREFIXES for f in NLI_FEATURES]
    traj_only = [f"{p}-{f}" for p in ("full", "selected") for f in TRAJECTORY_ONLY_FEATURES]
    return base + prefixed + TRAJECTORY_ONLY_FEATURES + traj_only


def build_feature_directions() -> dict[str, str]:
    """Build directions dict for all features including prefixed variants."""
    dirs = dict(FEATURE_DIRECTIONS)
    for prefix in TRAJECTORY_PREFIXES:
        for feat, direction in FEATURE_DIRECTIONS.items():
            dirs[f"{prefix}-{feat}"] = direction
    for prefix in ("full", "selected"):
        for feat in TRAJECTORY_ONLY_FEATURES:
            dirs[f"{prefix}-{feat}"] = FEATURE_DIRECTIONS[feat]
    return dirs
