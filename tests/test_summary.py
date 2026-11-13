import numpy as np
import pytest
import torch

from cadgnn.baselines.features import feature_matrix
from cadgnn.config import ModelConfig
from cadgnn.fusion import build_coordination_model
from cadgnn.fusion.summary import SUMMARY_FEATURES, EpisodeSummary, episode_statistics
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.models import collate


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


def test_statistics_match_the_baseline_features(graphs):
    expected = feature_matrix(graphs)[:, : len(SUMMARY_FEATURES)].copy()
    expected[:, :4] = np.log1p(expected[:, :4])
    found = episode_statistics(collate(graphs)).double().numpy()
    assert np.allclose(found, expected, atol=1e-5)


def test_statistics_do_not_depend_on_the_batch(graphs):
    alone = episode_statistics(collate(graphs[3:4]))
    together = episode_statistics(collate(graphs[:6]))[3:4]
    assert torch.allclose(alone, together)


def test_summary_needs_fitting(graphs):
    summary = EpisodeSummary(8)
    with pytest.raises(ValueError, match="not fitted"):
        summary(collate(graphs[:2]))
    out = summary.fit(graphs).eval()(collate(graphs[:5]))
    assert out.shape == (5, 8)


def test_model_with_and_without_summary(graphs):
    small = {"hidden_dim": 8, "temporal_dim": 8, "spatial_dim": 8, "fusion_dim": 8}
    on = build_coordination_model(ModelConfig(**small)).fit_scaler(graphs)
    off = build_coordination_model(ModelConfig(use_summary=False, **small)).fit_scaler(graphs)
    assert on.classifier[1].in_features == 16 and off.classifier[1].in_features == 8
    assert off.summary is None
    logit = on(collate(graphs[:4])).logit
    assert logit.shape == (4,)
    logit.sum().backward()
    assert on.summary.layer[0].weight.grad is not None
