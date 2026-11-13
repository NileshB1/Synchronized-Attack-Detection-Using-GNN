"""Per-episode event sequences built from a batch of episode graphs.

Each step is one flagged event: its (scaled) telemetry plus two timing values, log1p of the hours since 
the episode's first event and log1p of the hours since
the previous event. 
Both only look backwards, so step t never sees later events.

With ``links`` each step also says whether its VM shares a network or a storage pool with the
previous event's VM. A campaign moves along those links, a coincidental episode mostly does not,
and neither stream sees this on its own: the GRU knows the order but not the topology, the GAT
knows the topology but not the order.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

TIME_FEATURES = ("log_hours_since_start", "log_hours_since_previous")
LINK_FEATURES = ("same_network_as_previous", "same_storage_as_previous")


class TemporalInputError(ValueError):
    pass


class EventSequences(NamedTuple):
    values: Tensor  # [num_graphs, max_len, n_features + 2 (+ 2 with links)], zero padded
    mask: Tensor  # [num_graphs, max_len], True for real events
    lengths: Tensor  # [num_graphs]
    order: Tensor  # position of each sequence step's VM in batch["vm"]


def vm_resources(batch: Batch) -> tuple[Tensor, Tensor]:
    """Network and storage pool node of every VM in the batch, -1 if it has none."""
    vm = batch["vm"]
    n, device = vm.num_nodes, vm.x.device
    network = torch.full((n,), -1, dtype=torch.long, device=device)
    v, net = batch["vm", "attached_to", "network"].edge_index
    network[v] = net
    image = torch.full((n,), -1, dtype=torch.long, device=device)
    v, img = batch["vm", "has_backup", "backup_image"].edge_index
    image[v] = img
    pool_of_image = torch.full((batch["backup_image"].num_nodes,), -1, dtype=torch.long,
                               device=device)  # fmt: skip
    img, pool = batch["backup_image", "stored_on", "storage"].edge_index
    pool_of_image[img] = pool
    storage = torch.where(image >= 0, pool_of_image[image.clamp(min=0)], image)
    return network, storage


def event_sequences(batch: Batch, links: bool = False) -> EventSequences:
    vm = batch["vm"]
    x, graph = vm.x, vm.batch
    times = vm.rel_time_hours.to(x.dtype)
    num_graphs = batch.num_graphs

    if not torch.isfinite(times).all():
        raise TemporalInputError("event times must be finite")
    if (times < 0).any():
        raise TemporalInputError("event times must not be negative")
    lengths = torch.bincount(graph, minlength=num_graphs)
    if (lengths == 0).any():
        empty = torch.nonzero(lengths == 0).flatten().tolist()
        raise TemporalInputError(f"graph(s) {empty} have no VM events")

    #sort by graph, then by event order rather than trusting the storage order
    key = graph * (int(vm.event_order.max()) + 1) + vm.event_order
    order = torch.argsort(key, stable=True)
    graph, times, x = graph[order], times[order], x[order]

    same_graph = graph[1:] == graph[:-1]
    steps = times[1:] - times[:-1]
    if (same_graph & (steps < 0)).any():
        raise TemporalInputError("event times go backwards along event_order")
    gaps = torch.zeros_like(times)
    gaps[1:] = torch.where(same_graph, steps, torch.zeros_like(steps))

    columns = [x, torch.log1p(times)[:, None], torch.log1p(gaps)[:, None]]
    if links:
        for resource in vm_resources(batch):
            resource = resource[order]
            same = torch.zeros_like(times)
            same[1:] = (same_graph & (resource[1:] == resource[:-1]) & (resource[1:] >= 0)).to(
                times.dtype
            )
            columns.append(same[:, None])
    features = torch.cat(columns, dim=1)
    values, mask = to_dense_batch(features, graph, batch_size=num_graphs)
    return EventSequences(values, mask, lengths, order)
