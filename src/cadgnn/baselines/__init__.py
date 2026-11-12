"""Baselines the dual-stream model is compared against.

The CONTINUUM-style chained model lives in cadgnn.baselines.chained and is
selected with model.architecture: chained, so it trains like any variant.
"""

from cadgnn.baselines.classic import (
    BASELINES,
    FittedBaseline,
    fit_baseline,
    fit_estimator,
    rule_scores,
)
from cadgnn.baselines.features import FEATURE_NAMES, episode_features, feature_matrix

__all__ = [
    "BASELINES", "FittedBaseline", "fit_baseline", "fit_estimator", "rule_scores",
    "FEATURE_NAMES", "episode_features", "feature_matrix",
]  # fmt: skip
