"""CONTINUUM-style baseline: a GAT on each snapshot, then a GRU across snapshots.

CONTINUUM (Bahar et al., 2025) builds node embeddings as RNN([GNN(G_t)] for t = 1..T)
over time snapshots of a provenance graph and takes the mean of the node
embeddings as the graph embedding. Here the snapshots of an episode are the
episode graph after each of its events, so snapshot t holds the first t
flagged VMs and their resources.

Changes from the paper, so it can be compared on the same task and split:
- trained with labels and the same weighted loss, threshold and metrics as the
  dual-stream model, instead of an autoencoder with k-nearest-neighbour
  distances to benign graphs;
- no edge features in the attention, since the episode graphs have none;
- snapshots follow events, not fixed time intervals, so the hours since the
  first event are passed to the GRU with each snapshot;
- no federated learning.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_mean_pool

from cadgnn.config import ModelConfig
from cadgnn.fusion.model import CoordinationOutput
from cadgnn.graph import NODE_TYPES, prefix_graph
from cadgnn.models import FeatureScaler, check_batch
from cadgnn.spatial import SpatialTopologyStream

SnapshotCache = dict[tuple[str, int], tuple[list[HeteroData], Tensor]]


def episode_snapshots(graph: HeteroData) -> tuple[list[HeteroData], Tensor]:
    n = graph["vm"].num_nodes
    rel = graph["vm"].rel_time_hours[torch.argsort(graph["vm"].event_order)]
    return [prefix_graph(graph, k) for k in range(1, n + 1)], rel.clone()


def snapshots(
    batch: Batch, cache: SnapshotCache | None = None
) -> tuple[Batch, Tensor, list[int]]:
    """Every episode after each of its events, the time of each snapshot and the counts.

    Building the snapshots is the slow part, so they can be kept in a cache keyed
    by episode id and number of events. A prefix of an episode has the same id
    but fewer events, so it gets its own entry.
    """
    parts, times, counts = [], [], []
    for i in range(batch.num_graphs):
        key = (batch.episode_id[i], int((batch["vm"].batch == i).sum()))
        found = cache.get(key) if cache is not None else None
        if found is None:
            found = episode_snapshots(batch.get_example(i))
            if cache is not None:
                cache[key] = found
        parts += found[0]
        times.append(found[1])
        counts.append(len(found[0]))
    return Batch.from_data_list(parts), torch.cat(times), counts


class ChainedGatGru(nn.Module):
    """GAT per snapshot, mean pooled over all nodes, then a GRU over the snapshots."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.scaler = FeatureScaler()
        # Only its node embeddings are used; the pooling head is not.
        self.gat = SpatialTopologyStream(
            config.hidden_dim, config.hidden_dim,
            num_layers=config.spatial_layers, heads=config.spatial_heads, dropout=config.dropout,
        )  # fmt: skip
        self.gru = nn.GRU(
            config.hidden_dim + 1, config.hidden_dim,
            num_layers=config.temporal_layers, batch_first=True,
            dropout=config.dropout if config.temporal_layers > 1 else 0.0,
        )  # fmt: skip
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 1))
        # Snapshots of episodes seen so far. Not saved; episode ids must be unique per dataset.
        self.cache: SnapshotCache = {}

    def fit_scaler(self, train_graphs: Sequence[HeteroData]) -> ChainedGatGru:
        self.scaler.fit(train_graphs)
        return self

    def forward(self, batch: Batch) -> CoordinationOutput:
        check_batch(batch)
        snaps, times, counts = snapshots(batch.cpu(), self.cache)
        snaps = snaps.to(batch["vm"].x.device)
        snaps["vm"].x = self.scaler(snaps["vm"].x)

        h = self.gat.node_embeddings(snaps)
        nodes = torch.cat([h[kind] for kind in NODE_TYPES])
        owner = torch.cat([snaps[kind].batch for kind in NODE_TYPES])
        pooled = global_mean_pool(nodes, owner, size=snaps.num_graphs)
        steps = torch.cat([pooled, torch.log1p(times.to(pooled))[:, None]], dim=-1)

        sequences = pad_sequence(list(steps.split(counts)), batch_first=True)
        packed = pack_padded_sequence(
            sequences, torch.tensor(counts), batch_first=True, enforce_sorted=False
        )
        _, last = self.gru(packed)
        logit = self.classifier(last[-1]).squeeze(-1)
        return CoordinationOutput(logit, None, None, None)

    @torch.no_grad()
    def score(self, batch: Batch) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(batch).logit)
        finally:
            self.train(was_training)
