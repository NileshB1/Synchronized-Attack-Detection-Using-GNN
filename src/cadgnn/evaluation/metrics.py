"""Classification and alert-fatigue metrics for coordinated vs coincidental episodes.

Positive class is "coordinated" (1). A ratio whose denominator is zero, such as
precision when nothing was flagged, is reported as None rather than 0, so an
empty case is not mistaken for a bad one.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class MetricsError(ValueError):
    pass


def _check_inputs(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise MetricsError(
            f"labels and scores must be 1-D and the same length, got {y.shape} and {s.shape}"
        )
    if len(y) == 0:
        raise MetricsError("no predictions to evaluate")
    if not np.isin(y, (0, 1)).all():
        raise MetricsError("labels must be 0 or 1")
    if not np.isfinite(s).all():
        raise MetricsError("scores must be finite")
    return y.astype(int), s


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def classification_metrics(y_true, scores, threshold: float) -> dict:
    y, s = _check_inputs(y_true, scores)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if tp + fp + fn > 0 else None)
    )
    specificity = _ratio(tn, tn + fp)
    both_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "n_episodes": len(y),
        "n_coordinated": int(y.sum()),
        "n_coincidental": int(len(y) - y.sum()),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (
            (recall + specificity) / 2 if recall is not None and specificity is not None else None
        ),
        # Threshold-free ranking metrics; undefined when only one class is present.
        "roc_auc": float(roc_auc_score(y, s)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, s)) if both_classes else None,
    }


def rates_by_pattern(patterns, y_true, scores, threshold: float) -> dict:
    """Share of episodes flagged for each generator pattern.

    For a coincidental pattern this is its false alarm rate, for a coordinated
    one its detection rate. It shows which kinds of episode cause the errors.
    """
    y, s = _check_inputs(y_true, scores)
    patterns = np.asarray(patterns, dtype=object)
    if len(patterns) != len(y):
        raise MetricsError("need one pattern per episode")
    rates = {}
    for pattern in sorted(set(patterns)):
        mask = patterns == pattern
        flagged = int((s[mask] >= threshold).sum())
        rates[pattern] = {
            "label": "coordinated" if y[mask].mean() >= 0.5 else "coincidental",
            "episodes": int(mask.sum()),
            "flagged": flagged,
            "flagged_rate": flagged / int(mask.sum()),
        }
    return rates


def select_threshold(
    y_true, scores, strategy: str = "f1", max_fpr: float = 0.1, min_precision: float = 0.9
) -> float:
    """Choose a decision threshold from validation scores.

    ``f1`` takes the threshold with the best F1. ``fpr`` takes the threshold with
    the best recall among those whose false positive rate is at most ``max_fpr``.
    ``precision`` takes the best recall among those with precision of at least
    ``min_precision``, so at most 1 - min_precision of the alerts are false.
    Candidates are the midpoints between neighbouring scores. When several are
    equally good the middle one is used, so the threshold sits in the middle of
    the gap rather than right next to a validation score.
    """
    y, s = _check_inputs(y_true, scores)
    if len(np.unique(y)) < 2:
        raise MetricsError("validation set needs both classes to choose a threshold")
    if strategy not in ("f1", "fpr", "precision"):
        raise MetricsError(f"unknown threshold strategy {strategy!r}")

    levels = np.unique(s)
    candidates = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels
    values = []
    for t in candidates:
        m = classification_metrics(y, s, t)
        if strategy == "f1":
            values.append(m["f1"] or 0.0)
        elif strategy == "fpr":
            values.append(m["recall"] if m["false_positive_rate"] <= max_fpr else -1.0)
        else:
            precise = m["precision"] is not None and m["precision"] >= min_precision
            values.append(m["recall"] if precise else -1.0)
    values = np.array(values)
    if values.max() < 0:
        # Even the highest cut lets too many false alarms through, so flag
        # nothing at all.
        return float(np.nextafter(s.max(), np.inf))
    tied = candidates[values == values.max()]
    return float(tied[len(tied) // 2])
