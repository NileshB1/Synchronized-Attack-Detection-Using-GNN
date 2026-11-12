"""Rule-based correlation and classic machine learning baselines
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

from cadgnn.baselines.features import feature_matrix, labels, steps
from cadgnn.training.balance import FEATURE_BALANCING, balance_indices, smote

BASELINES = ("rule_based", "logistic_regression", "random_forest")
RULE_WINDOWS_HOURS = (0.25, 0.5, 1, 2, 4, 8, 12, 24, 48)
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
FOREST_GRID = {"max_depth": (None, 4, 8), "min_samples_leaf": (1, 5)}


@dataclass
class FittedBaseline:
    name: str
    score: Callable[[Sequence[HeteroData]], np.ndarray]
    settings: dict = field(default_factory=dict)
    # The chosen model options, to fit the same model again on other data.
    options: dict = field(default_factory=dict)


def make_estimator(name: str, options: dict, seed: int, balancing: str = "class_weight"):
    weight = "balanced" if balancing == "class_weight" else None
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(), LogisticRegression(class_weight=weight, max_iter=5000, **options)
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight=weight, random_state=seed, n_jobs=-1, **options
        )
    raise ValueError(f"{name} is not a scikit-learn baseline")


def fit_estimator(name: str, options: dict, x, y, seed: int, balancing: str = "class_weight"):
    """Balance the training rows as asked, then fit."""
    if balancing not in FEATURE_BALANCING:
        choices = ", ".join(FEATURE_BALANCING)
        raise ValueError(f"unknown balancing {balancing!r}, choose from {choices}")
    if balancing == "smote":
        x, y = smote(x, y, seed)
    else:
        keep = balance_indices(y, balancing, seed)
        x, y = x[keep], y[keep]
    return make_estimator(name, options, seed, balancing).fit(x, y)


def rule_scores(graphs: Sequence[HeteroData], window_hours: float) -> np.ndarray:
    scores = []
    for g in graphs:
        step = steps(g)
        linked = step.shared & (step.gaps <= window_hours)
        scores.append(float(linked.mean()) if len(linked) else 0.0)
    return np.array(scores)


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else 0.5


def _pick(candidates, val_y: np.ndarray):
    """The candidate with the best validation ROC-AUC; the first one wins a tie."""
    best, best_auc = None, -1.0
    for settings, score in candidates:
        auc = _auc(val_y, score)
        if auc > best_auc:
            best, best_auc = (settings, score), auc
    return best[0], best_auc


def fit_baseline(
    name: str,
    train: Sequence[HeteroData],
    val: Sequence[HeteroData],
    seed: int,
    balancing: str = "class_weight",
) -> FittedBaseline:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}, choose from {', '.join(BASELINES)}")
    val_y = labels(val)

    if name == "rule_based":
        candidates = [({"window_hours": w}, rule_scores(val, w)) for w in RULE_WINDOWS_HOURS]
        settings, auc = _pick(candidates, val_y)
        window = settings["window_hours"]
        return FittedBaseline(
            name, lambda gs: rule_scores(gs, window), {**settings, "val_roc_auc": auc}
        )

    x_train, y_train = feature_matrix(train), labels(train)
    x_val = feature_matrix(val)
    if name == "logistic_regression":
        options = [{"C": c} for c in LOGISTIC_C]
    else:
        options = [
            dict(zip(FOREST_GRID, values, strict=True))
            for values in product(*FOREST_GRID.values())
        ]

    fitted = {}
    candidates = []
    for i, opts in enumerate(options):
        model = fit_estimator(name, opts, x_train, y_train, seed, balancing)
        fitted[i] = model
        candidates.append(({"index": i, **opts}, model.predict_proba(x_val)[:, 1]))
    settings, auc = _pick(candidates, val_y)
    model = fitted[settings.pop("index")]

    def score(graphs: Sequence[HeteroData]) -> np.ndarray:
        return model.predict_proba(feature_matrix(graphs))[:, 1]

    return FittedBaseline(name, score, {**settings, "val_roc_auc": auc}, dict(settings))
