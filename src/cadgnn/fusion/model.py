"""Full model: both streams, fusion and a single coordination logit per episode.

The logit is for "coordinated" (graph label 1) against "coincidental" (0). sigmoid(logit) is a coordination score in 
(0, 1); it is not calibrated, so it should not be read as a probability. 
Picking a decision threshold is left to evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData

from cadgnn.config import ModelConfig
from cadgnn.fusion.layers import build_fusion
from cadgnn.models import DualStreamModel
from cadgnn.spatial import build_spatial_stream
from cadgnn.temporal import build_temporal_stream


class CoordinationOutput(NamedTuple):
    logit: Tensor  # [num_graphs]
    temporal: Tensor | None  # stream embeddings, None if the stream is off
    spatial: Tensor | None
    gate: Tensor | None  # [num_graphs] temporal share, gated fusion only


class CoordinationModel(nn.Module):
    def __init__(self, config: ModelConfig, streams: DualStreamModel, fusion: nn.Module) -> None:
        super().__init__()
        if getattr(fusion, "out_dim", None) != config.fusion_dim:
            raise ValueError(
                f"fusion out_dim is {getattr(fusion, 'out_dim', None)}, "
                f"config expects {config.fusion_dim}"
            )
        self.config=config
        self.streams=streams
        self.fusion=fusion
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.fusion_dim, 1))

    def fit_scaler(self, train_graphs: Sequence[HeteroData]) -> CoordinationModel:
        self.streams.fit_scaler(train_graphs)
        return self

    def forward(self, batch: Batch) -> CoordinationOutput:
        temporal, spatial = self.streams(batch)
        fused, gate = self.fusion(temporal, spatial)
        logit = self.classifier(fused).squeeze(-1)
        return CoordinationOutput(logit, temporal, spatial, gate)

    @torch.no_grad()
    def score(self, batch: Batch) -> Tensor:
        """Uncalibrated coordination score per episode, in eval mode"""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(batch).logit)
        finally:
            self.train(was_training)


def build_coordination_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "chained":
        # Imported here because the baseline itself builds on this module.
        from cadgnn.baselines.chained import ChainedGatGru

        return ChainedGatGru(config)
    temporal=build_temporal_stream(config) if config.use_temporal else None
    spatial=build_spatial_stream(config) if config.use_spatial else None
    streams=DualStreamModel(config, temporal, spatial)
    return CoordinationModel(config, streams, build_fusion(config))
