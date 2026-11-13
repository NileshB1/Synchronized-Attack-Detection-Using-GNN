"""Episode summary: a few whole-episode numbers given to the classifier next to the streams.

Some things are easy to count but hard for message passing to work out, such as the longest
gap between two events or how many separate pieces the episode graph falls into. This is the
global attribute of a graph network: one vector per graph, computed from the same graph the
streams read. Labels, episode ids and generator patterns are not used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import scatter

from cadgnn.graph import EDGE_TYPES, NODE_TYPES
from cadgnn.models import FeatureScaler, collate
from cadgnn.temporal import event_sequences, vm_resources

SUMMARY_FEATURES = (
    "log_vms", "log_networks", "log_storage_pools", "log_components",
    "largest_network_share", "largest_storage_share",
    "log_span_hours", "log_mean_gap", "log_min_gap", "log_max_gap", "gap_variation",
    "linked_step_share",
)  # fmt: skip


def _counts(batch: Batch, kind: str) -> Tensor:
    return torch.bincount(batch[kind].batch, minlength=batch.num_graphs).float()


def _components(batch: Batch) -> Tensor:
    """Connected pieces of each episode graph."""
    offsets, total = {}, 0
    for kind in NODE_TYPES:
        offsets[kind] = total
        total += batch[kind].num_nodes
    rows, cols = [], []
    for src, rel, dst in EDGE_TYPES:
        s, d = batch[src, rel, dst].edge_index.cpu().numpy()
        rows.append(s + offsets[src])
        cols.append(d + offsets[dst])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(total, total))
    label = connected_components(matrix, directed=False)[1]
    graph = np.concatenate([batch[kind].batch.cpu().numpy() for kind in NODE_TYPES])
    pairs = np.unique(np.stack([graph, label]), axis=1)
    count = np.bincount(pairs[0], minlength=batch.num_graphs)
    return torch.as_tensor(count, dtype=torch.float, device=batch["vm"].x.device)


def _largest_share(resource: Tensor, graph: Tensor, n_nodes: int, n_vms: Tensor) -> Tensor:
    """Share of the episode's VMs on its busiest network or storage pool."""
    used = resource >= 0
    per_node = torch.bincount(resource[used], minlength=n_nodes).float()
    owner = torch.full((n_nodes,), 0, dtype=torch.long, device=graph.device)
    owner[resource[used]] = graph[used]
    most = scatter(per_node, owner, dim=0, dim_size=len(n_vms), reduce="max")
    return most / n_vms.clamp(min=1)


def episode_statistics(batch: Batch) -> Tensor:
    """[num_graphs, len(SUMMARY_FEATURES)], unscaled."""
    vm = batch["vm"]
    size = batch.num_graphs
    n_vms = _counts(batch, "vm")
    network, storage = vm_resources(batch)

    seq = event_sequences(batch, links=True)
    order = seq.order
    graph = vm.batch[order]
    times = vm.rel_time_hours[order].float()
    same_graph = graph[1:] == graph[:-1]
    gaps = (times[1:] - times[:-1])[same_graph]
    gap_graph = graph[1:][same_graph]
    mean = scatter(gaps, gap_graph, dim=0, dim_size=size, reduce="mean")
    std = (scatter((gaps - mean[gap_graph]) ** 2, gap_graph, dim=0, dim_size=size,
                   reduce="mean")).sqrt()  # fmt: skip
    has_gaps = torch.bincount(gap_graph, minlength=size) > 0
    low = scatter(gaps, gap_graph, dim=0, dim_size=size, reduce="min")
    high = scatter(gaps, gap_graph, dim=0, dim_size=size, reduce="max")
    low, high = (torch.where(has_gaps, v, torch.zeros_like(v)) for v in (low, high))
    variation = torch.where(mean > 0, std / mean.clamp(min=1e-12), torch.zeros_like(mean))

    # Steps are rows 1.. of each sequence; a step is linked if either flag is on.
    width = seq.values.size(-1)
    linked = (seq.values[:, 1:, width - 2 :] > 0).any(dim=-1) & seq.mask[:, 1:]
    n_steps = (seq.lengths - 1).clamp(min=1).float()
    linked_share = linked.sum(dim=1).float() / n_steps

    span = scatter(times, graph, dim=0, dim_size=size, reduce="max")
    values = [
        torch.log1p(n_vms), torch.log1p(_counts(batch, "network")),
        torch.log1p(_counts(batch, "storage")), torch.log1p(_components(batch)),
        _largest_share(network, vm.batch, batch["network"].num_nodes, n_vms),
        _largest_share(storage, vm.batch, batch["storage"].num_nodes, n_vms),
        torch.log1p(span), torch.log1p(mean), torch.log1p(low), torch.log1p(high),
        variation, linked_share,
    ]  # fmt: skip
    return torch.stack(values, dim=1)


class EpisodeSummary(nn.Module):
    """Scaled episode statistics through one small layer."""

    def __init__(self, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.scaler = FeatureScaler(len(SUMMARY_FEATURES))
        self.layer = nn.Sequential(
            nn.Linear(len(SUMMARY_FEATURES), out_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def fit(self, graphs: Sequence[HeteroData], chunk: int = 256) -> EpisodeSummary:
        """Fit the scaler on the training graphs."""
        stats = [
            episode_statistics(collate(list(graphs[i : i + chunk])))
            for i in range(0, len(graphs), chunk)
        ]
        x = torch.cat(stats).to(self.scaler.mean)
        std = x.std(dim=0, unbiased=False)
        std[std < 1e-8] = 1.0
        self.scaler.mean.copy_(x.mean(dim=0))
        self.scaler.std.copy_(std)
        self.scaler.fitted.fill_(True)
        return self

    def forward(self, batch: Batch) -> Tensor:
        return self.layer(self.scaler(episode_statistics(batch)))
