import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from cadgnn.evaluation import (
    MetricsError,
    classification_metrics,
    rates_by_pattern,
    select_threshold,
)

Y = [1, 1, 1, 0, 0, 0, 0, 0]
SCORES = [0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.05, 0.4]


def test_counts_and_rates():
    m = classification_metrics(Y, SCORES, threshold=0.5)
    assert m["confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 4, "fn": 1}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert (m["n_coordinated"], m["n_coincidental"]) == (3, 5)


def test_ranking_metrics_match_sklearn():
    m = classification_metrics(Y, SCORES, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y, SCORES))
    assert m["pr_auc"] == pytest.approx(average_precision_score(Y, SCORES))


def test_threshold_is_inclusive():
    m = classification_metrics([1, 0], [0.5, 0.2], threshold=0.5)
    assert m["confusion_matrix"]["tp"] == 1


def test_undefined_ratios_are_none():
    nothing_flagged = classification_metrics([1, 0, 0], [0.1, 0.2, 0.3], threshold=0.9)
    assert nothing_flagged["precision"] is None
    assert nothing_flagged["recall"] == 0.0
    assert nothing_flagged["f1"] == 0.0

    only_coincidental = classification_metrics([0, 0], [0.1, 0.7], threshold=0.5)
    assert only_coincidental["recall"] is None
    assert only_coincidental["false_negative_rate"] is None
    assert only_coincidental["roc_auc"] is None and only_coincidental["pr_auc"] is None
    assert only_coincidental["false_positive_rate"] == 0.5


@pytest.mark.parametrize(
    "y, scores, message",
    [
        ([], [], "no predictions"),
        ([1, 0], [0.5], "same length"),
        ([1, 2], [0.5, 0.4], "0 or 1"),
        ([1, 0], [0.5, float("nan")], "finite"),
        ([[1, 0]], [[0.5, 0.4]], "1-D"),
    ],
)
def test_invalid_inputs(y, scores, message):
    with pytest.raises(MetricsError, match=message):
        classification_metrics(y, scores, 0.5)


def test_f1_threshold_sits_in_the_gap():
    # Perfectly separable: any cut in (0.4, 0.7) gives F1 = 1.
    y = [0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.4, 0.7, 0.9]
    threshold = select_threshold(y, scores, "f1")
    assert threshold == pytest.approx(0.55)
    assert classification_metrics(y, scores, threshold)["f1"] == 1.0


def test_f1_threshold_on_overlapping_scores():
    threshold = select_threshold(Y, SCORES, "f1")
    best = max(classification_metrics(Y, SCORES, t)["f1"] for t in np.linspace(0, 1, 201))
    assert classification_metrics(Y, SCORES, threshold)["f1"] == pytest.approx(best)


def test_fpr_strategy_respects_the_limit():
    threshold = select_threshold(Y, SCORES, "fpr", max_fpr=0.0)
    m = classification_metrics(Y, SCORES, threshold)
    assert m["false_positive_rate"] == 0.0
    assert m["recall"] == pytest.approx(2 / 3)
    loose = select_threshold(Y, SCORES, "fpr", max_fpr=0.2)
    assert classification_metrics(Y, SCORES, loose)["false_positive_rate"] <= 0.2


def test_fpr_strategy_falls_back_to_flagging_nothing():
    # The coincidental episode outscores the coordinated one, so no cut can keep FPR at 0
    # while catching anything; the threshold ends above every score.
    threshold = select_threshold([1, 0], [0.3, 0.8], "fpr", max_fpr=0.0)
    assert classification_metrics([1, 0], [0.3, 0.8], threshold)["confusion_matrix"]["fp"] == 0


def test_rates_by_pattern():
    patterns = ["fast", "fast", "stealth", "lookalike", "scattered", "scattered", "burst", "burst"]
    rates = rates_by_pattern(patterns, Y, SCORES, threshold=0.5)
    assert rates["fast"] == {"label": "coordinated", "episodes": 2, "flagged": 2,
                             "flagged_rate": 1.0}  # fmt: skip
    assert rates["stealth"]["flagged_rate"] == 0.0
    assert rates["lookalike"] == {"label": "coincidental", "episodes": 1, "flagged": 1,
                                  "flagged_rate": 1.0}  # fmt: skip
    assert rates["burst"]["flagged_rate"] == 0.0
    with pytest.raises(MetricsError, match="one pattern per episode"):
        rates_by_pattern(patterns[:3], Y, SCORES, 0.5)


def test_threshold_needs_both_classes_and_a_known_strategy():
    with pytest.raises(MetricsError, match="both classes"):
        select_threshold([1, 1], [0.3, 0.8])
    with pytest.raises(MetricsError, match="unknown threshold strategy"):
        select_threshold(Y, SCORES, "youden")
