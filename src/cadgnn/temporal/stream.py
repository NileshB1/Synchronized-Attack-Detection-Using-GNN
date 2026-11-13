"""Temporal severity stream: 
are the flagged events close in time and intense?

There are no severity labels in the data, 
so this stream is not trained on a severity target. It encodes 
telemetry intensity and event timing into one vector per episode 
and learns only through the episode label downstream.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Batch

from cadgnn.config import ModelConfig
from cadgnn.graph import VM_FEATURES
from cadgnn.temporal.sequences import LINK_FEATURES, TIME_FEATURES, event_sequences


class TemporalSeverityStream(nn.Module):
    """GRU over each episode's events in time order.

    The GRU runs forwards only, so the state at step t depends on events 0..t.
    With ``links`` each step also gets the two network and storage link flags.
    The episode vector joins the final state with the mean state over all steps.
    """

    def __init__(
        self,  hidden_dim: int,
        out_dim: int, n_features: int = len(VM_FEATURES),
        num_layers: int = 1,  dropout: float = 0.0, links: bool = False
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.links = links
        n_in = n_features + len(TIME_FEATURES) + (len(LINK_FEATURES) if links else 0)
        self.input = nn.Sequential(nn.Linear(n_in, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(
            hidden_dim,  hidden_dim, num_layers=num_layers,  batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, out_dim))

    def step_states(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """GRU state after each event, [num_graphs, max_len, hidden_dim], and the padding mask."""
        seq = event_sequences(batch, self.links)
        h = self.input(seq.values)
        packed = pack_padded_sequence(h, seq.lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
        return out, seq.mask

    def forward(self, batch: Batch) -> Tensor:
        states, mask = self.step_states(batch)
        lengths = mask.sum(dim=1)
        last = states[torch.arange(states.size(0), device=states.device), lengths - 1]
        mean = (states * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        return self.head(torch.cat([last, mean], dim=-1))


def build_temporal_stream(config: ModelConfig) -> TemporalSeverityStream:
    return TemporalSeverityStream(
        hidden_dim=config.hidden_dim,  out_dim=config.temporal_dim,  num_layers=config.temporal_layers,
        dropout=config.dropout, links=config.step_links
    )
