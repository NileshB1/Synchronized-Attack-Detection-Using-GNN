from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from cadgnn.graph import NODE_TYPES, VM_FEATURES


class NodeEncoder(nn.Module):
    """Maps every node type to vectors of the same size.

    VMs are projected from their telemetry. 
    Backup images, storage pools and networks have no features, 
    so each of those types gets one learned vector.
    """

    def __init__(
        self,
        hidden_dim: int, node_types: Sequence[str] = NODE_TYPES,
        n_features: int = len(VM_FEATURES),  dropout: float = 0.0):
        super().__init__()
        unknown = set(node_types) - set(NODE_TYPES)
        if unknown:
            raise ValueError(f"unknown node type(s): {sorted(unknown)}")
        self.node_types = tuple(node_types)
        self.vm = nn.Linear(n_features, hidden_dim) if "vm" in node_types else None
        self.type_vectors = nn.ParameterDict({
                kind: nn.Parameter(torch.randn(hidden_dim) * 0.1)
                for kind in self.node_types
                if kind != "vm"
            })
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        out = {}
        if self.vm is not None:
            out["vm"] = self.vm(batch["vm"].x)
        for kind, vector in self.type_vectors.items():
            out[kind] = vector.expand(batch[kind].num_nodes, -1)
        return {kind: self.dropout(h) for kind, h in out.items()}
