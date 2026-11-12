import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cadgnn.evaluation.calibration import brier_score, fit_calibrator


@pytest.fixture
def overconfident():
    # Scores run high: a third of the episodes are coordinated but most score above 0.5.
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 1 / 3).astype(int)
    scores = np.clip(0.55 + 0.3 * y + rng.normal(0, 0.15, 300), 0.01, 0.99)
    return y, scores


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_improves_brier_and_keeps_the_order(overconfident, method):
    y, scores = overconfident
    calibrated = fit_calibrator(method, y, scores)(scores)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert brier_score(y, calibrated) < brier_score(y, scores)
    if method == "platt":
        assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, scores))
    assert np.all(np.diff(calibrated[np.argsort(scores)]) >= -1e-12)


def test_calibration_cuts_false_alarms_at_one_half(overconfident):
    y, scores = overconfident
    calibrated = fit_calibrator("platt", y, scores)(scores)
    false_alarms = ((scores >= 0.5) & (y == 0)).sum()
    assert ((calibrated >= 0.5) & (y == 0)).sum() < false_alarms


def test_bad_calibration_requests():
    with pytest.raises(ValueError, match="unknown calibration"):
        fit_calibrator("beta", [0, 1], [0.2, 0.8])
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrator("platt", [0, 0], [0.2, 0.8])


def test_brier_score():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
