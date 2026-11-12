"""Probability calibration of model scores, fitted on the validation part.

A model trained with class weights tends to give coordinated-looking scores to
too many episodes, so a fixed cut of 0.5 raises false alarms. Calibration maps
the scores to probabilities that match how often an episode with that score is
really coordinated.

- platt: a logistic regression on the logit of the score (Platt, 1999).
- isotonic: a step function that only goes up (Zadrozny and Elkan, 2002). It
  needs more data than Platt scaling and can overfit a small validation part.

Both keep the order of the scores, so ROC-AUC does not change; what changes is
which episodes pass a fixed threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATION_METHODS = ("platt", "isotonic")
_EPS = 1e-6


def _logit(scores) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))[:, None]


@dataclass
class Calibrator:
    method: str
    model: object

    def __call__(self, scores) -> np.ndarray:
        if self.method == "platt":
            return self.model.predict_proba(_logit(scores))[:, 1]
        return self.model.predict(np.asarray(scores, dtype=float))


def fit_calibrator(method: str, y_true, scores) -> Calibrator:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("calibration needs both classes")
    if method == "platt":
        # A large C means almost no regularisation, as in plain Platt scaling.
        return Calibrator(method, LogisticRegression(C=1e6).fit(_logit(scores), y))
    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        return Calibrator(method, model.fit(np.asarray(scores, dtype=float), y))
    raise ValueError(f"unknown calibration method {method!r}, choose from platt, isotonic")


def brier_score(y_true, scores) -> float:
    """Mean squared gap between score and label; lower means better calibrated."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(scores, dtype=float) - y) ** 2))
