"""Significance tests between the proposed model and the others
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _pairs(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("both models need a score on the same folds")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep] - b[keep]


def paired_t_p(a, b) -> float:
    """1.0 when the scores are equal on every fold; NaN when the gap never changes."""
    diff = _pairs(a, b)
    if len(diff) < 2:
        return math.nan
    if np.allclose(diff, 0):
        return 1.0
    if np.allclose(diff, diff[0]):
        return math.nan
    return float(stats.ttest_1samp(diff, 0.0).pvalue)


def wilcoxon_p(a, b) -> float:
    diff = _pairs(a, b)
    if len(diff) < 2:
        return math.nan
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff).pvalue)


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test. ``only_a`` counts episodes model a gets right and b gets wrong."""
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    a_right, b_right = a == y, b == y
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p": p}
