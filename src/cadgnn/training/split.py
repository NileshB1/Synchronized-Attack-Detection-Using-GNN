"""Stratified train / validation / test split over episode graphs.

Each episode is one sample, so splitting by episode keeps all of an episode's
events on one side. The same VMs can still appear in episodes on different
sides; the model never sees node identities, so that does not leak labels.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class Split:
    train: list[int]
    val: list[int]
    test: list[int]

    def episode_ids(self, graphs: Sequence[HeteroData]) -> dict[str, list[str]]:
        return {
            name: [graphs[i].episode_id for i in getattr(self, name)]
            for name in ("train", "val", "test")
        }


def stratified_split(
    graphs: Sequence[HeteroData], val_fraction: float, test_fraction: float, seed: int
) -> Split:
    """Split each label separately so all three parts keep the class ratio.

    Every part gets at least one episode of each label; a label with fewer than
    three episodes cannot be split and raises ValueError.
    """
    labels = np.array([int(g.y) for g in graphs])
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 3:
            raise ValueError(f"need at least 3 episodes with label {label}, found {len(idx)}")
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_fraction))
        n_val = max(1, round(len(idx) * val_fraction))
        if n_test + n_val >= len(idx):
            n_test = n_val = 1
        parts["test"] += idx[:n_test].tolist()
        parts["val"] += idx[n_test : n_test + n_val].tolist()
        parts["train"] += idx[n_test + n_val :].tolist()
    return Split(**{name: sorted(part) for name, part in parts.items()})


def save_split(split: Split, graphs: Sequence[HeteroData], path: Path, **extra) -> Path:
    """Write the split as positions and episode ids, so it can be rebuilt later."""
    payload = {
        **extra,
        "sizes": {name: len(getattr(split, name)) for name in ("train", "val", "test")},
        "indices": asdict(split),
        "episode_ids": split.episode_ids(graphs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def split_from_ids(graphs: Sequence[HeteroData], ids: dict[str, list[str]]) -> Split:
    """Rebuild a saved split from episode ids, e.g. the one stored in a checkpoint."""
    position = {g.episode_id: i for i, g in enumerate(graphs)}
    missing = [e for part in ids.values() for e in part if e not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} episode(s) from the saved split are not in the graphs, "
            f"e.g. {missing[:3]}; were the graphs rebuilt after training?"
        )
    return Split(
        **{name: sorted(position[e] for e in ids[name]) for name in ("train", "val", "test")}
    )
