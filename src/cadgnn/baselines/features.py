"""Fixed-length features per episode for the non-graph baselines.

They summarise what an analyst could read off an episode without a GNN: how
many networks and storage pools the VMs share, how spread out the events are in
time, and how intense the telemetry is. Like the GNN, they never use labels,
episode ids, patterns or node identities.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import HeteroData

from cadgnn.graph import EDGE_TYPES, NODE_TYPES, VM_FEATURES

FEATURE_NAMES = (
    "n_vms", "n_networks", "n_storage_pools", "n_components",
    "largest_network_share", "largest_storage_share",
    "log_span_hours", "log_mean_gap", "log_min_gap", "log_max_gap", "gap_variation",
    "shared_step_share",
    *(f"mean_{name}" for name in VM_FEATURES), "mean_anomaly_score",
)  # fmt: skip


class Steps(NamedTuple):
    gaps: np.ndarray  # hours between consecutive events
    shared: np.ndarray  # whether the two VMs share a network or a storage pool


def vm_resources(graph: HeteroData) -> tuple[np.ndarray, np.ndarray]:
    """Network and storage pool index of every VM in the episode graph."""
    n = graph["vm"].num_nodes
    network = np.full(n, -1)
    vm, net = graph["vm", "attached_to", "network"].edge_index.numpy()
    network[vm] = net
    image_of = np.full(n, -1)
    vm, img = graph["vm", "has_backup", "backup_image"].edge_index.numpy()
    image_of[vm] = img
    pool_of_image = np.full(graph["backup_image"].num_nodes, -1)
    img, pool = graph["backup_image", "stored_on", "storage"].edge_index.numpy()
    pool_of_image[img] = pool
    return network, pool_of_image[image_of]


def steps(graph: HeteroData) -> Steps:
    order = torch.argsort(graph["vm"].event_order).numpy()
    times = graph["vm"].rel_time_hours.numpy()[order]
    network, storage = (r[order] for r in vm_resources(graph))
    shared = (network[1:] == network[:-1]) | (storage[1:] == storage[:-1])
    return Steps(np.diff(times), shared)


def _components(graph: HeteroData) -> int:
    offsets, total = {}, 0
    for kind in NODE_TYPES:
        offsets[kind] = total
        total += graph[kind].num_nodes
    rows, cols = [], []
    for src, rel, dst in EDGE_TYPES:
        s, d = graph[src, rel, dst].edge_index.numpy()
        rows.append(s + offsets[src])
        cols.append(d + offsets[dst])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(total, total))
    return int(connected_components(matrix, directed=False)[0])


def _largest_share(index: np.ndarray) -> float:
    return float(np.bincount(index).max() / len(index))


def episode_features(graph: HeteroData) -> np.ndarray:
    n = graph["vm"].num_nodes
    network, storage = vm_resources(graph)
    step = steps(graph)
    gaps = step.gaps if len(step.gaps) else np.zeros(1)
    span = float(graph["vm"].rel_time_hours.max())
    x = graph["vm"].x.numpy()
    values = [
        n, graph["network"].num_nodes, graph["storage"].num_nodes, _components(graph),
        _largest_share(network), _largest_share(storage),
        np.log1p(span), np.log1p(gaps.mean()), np.log1p(gaps.min()), np.log1p(gaps.max()),
        gaps.std() / gaps.mean() if gaps.mean() > 0 else 0.0,
        float(step.shared.mean()) if len(step.shared) else 0.0,
        *x.mean(axis=0), float(graph["vm"].anomaly_score.mean()),
    ]  # fmt: skip
    return np.asarray(values, dtype=np.float64)


def feature_matrix(graphs: Sequence[HeteroData]) -> np.ndarray:
    return np.stack([episode_features(g) for g in graphs])


def labels(graphs: Sequence[HeteroData]) -> np.ndarray:
    return np.array([int(g.y) for g in graphs])
