"""Ways to handle the class imbalance of the training part.

There are about two coincidental episodes for every coordinated one. Only the
training part is ever balanced; validation and test keep the real ratio, so
the scores on them stay honest.

- class_weight: no resampling, the loss weighs the rarer class up instead.
- oversample: copies of random coordinated episodes until both classes match.
- undersample: a random subset of the coincidental episodes, as many as the
  coordinated ones.
- smote: new coordinated samples on the line between a sample and one of its
  nearest coordinated neighbours (Chawla et al., 2002). It needs fixed-length
  feature vectors, so only the feature baselines can use it.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

RESAMPLING = ("oversample", "undersample")
FEATURE_BALANCING = ("none", "class_weight", "oversample", "undersample", "smote")


def balance_indices(y, method: str, seed: int) -> np.ndarray:
    """Positions of the samples to train on after resampling, in random order."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    everything = np.arange(len(y))
    if method in ("none", "class_weight"):
        return everything
    if method not in RESAMPLING:
        raise ValueError(f"unknown resampling method {method!r}")
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        raise ValueError("resampling needs both classes in the training part")
    minority = int(counts.argmin())
    small = np.flatnonzero(y == minority)
    large = np.flatnonzero(y != minority)
    if method == "oversample":
        extra = rng.choice(small, size=len(large) - len(small), replace=True)
        picked = np.concatenate([everything, extra])
    else:
        picked = np.concatenate([small, rng.choice(large, size=len(small), replace=False)])
    return rng.permutation(picked)


def smote(x, y, seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Add synthetic minority samples until both classes have the same count."""
    x, y = np.asarray(x, dtype=float), np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        raise ValueError("smote needs at least two samples of the smaller class")
    minority = int(counts.argmin())
    small = x[y == minority]
    need = int(counts.max() - counts.min())
    if need == 0:
        return x, y
    rng = np.random.default_rng(seed)
    k = min(k, len(small) - 1)
    # The first neighbour of a point is the point itself, so ask for one more.
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(small).kneighbors(small)[1][:, 1:]
    base = rng.integers(0, len(small), size=need)
    other = neighbours[base, rng.integers(0, k, size=need)]
    step = rng.random((need, 1))
    new = small[base] + step * (small[other] - small[base])
    return np.vstack([x, new]), np.concatenate([y, np.full(need, minority)])
