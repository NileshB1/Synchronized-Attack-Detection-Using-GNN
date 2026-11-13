import copy

import pytest
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from cadgnn.baselines.features import steps, vm_resources
from cadgnn.config import ConfigError, ModelConfig
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.models import DualStreamModel, NodeEncoder, collate, seed_everything
from cadgnn.temporal import (
    LINK_FEATURES,
    TIME_FEATURES,
    TemporalInputError,
    TemporalSeverityStream,
    build_temporal_stream,
    event_sequences,
)


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


def make_stream(**kwargs):
    seed_everything(0)
    stream = TemporalSeverityStream(hidden_dim=16, out_dim=8, **kwargs)
    return stream.eval()


def edited(graph, **vm_values):
    g = copy.copy(graph)
    for name, value in vm_values.items():
        g["vm"][name] = value
    return g


def test_sequence_shapes_and_mask(graphs):
    batch = collate(graphs[:5])
    seq = event_sequences(batch)
    lengths = [g["vm"].num_nodes for g in graphs[:5]]
    assert seq.values.shape == (5, max(lengths), 4 + len(TIME_FEATURES))
    assert seq.lengths.tolist() == lengths
    assert seq.mask.sum(1).tolist() == lengths
    assert (seq.values[~seq.mask] == 0).all()


def test_timing_features(graphs):
    g = graphs[0]
    seq = event_sequences(collate([g]))
    times = g["vm"].rel_time_hours
    n = len(times)
    assert torch.allclose(seq.values[0, :n, 4], torch.log1p(times))
    gaps = torch.cat([torch.zeros(1), times.diff()])
    assert torch.allclose(seq.values[0, :n, 5], torch.log1p(gaps))
    assert seq.values[0, 0, 5] == 0


def test_sequence_follows_event_order_not_storage_order(graphs):
    g = graphs[3]
    perm = torch.randperm(g["vm"].num_nodes, generator=torch.Generator().manual_seed(1))
    shuffled = edited(
        g,
        x=g["vm"].x[perm],
        rel_time_hours=g["vm"].rel_time_hours[perm],
        event_order=g["vm"].event_order[perm],
    )
    a = event_sequences(collate([g])).values
    b = event_sequences(collate([shuffled])).values
    assert torch.equal(a, b)


def test_link_flags_match_the_baseline_features(graphs):
    batch = collate(graphs[:8])
    seq = event_sequences(batch, links=True)
    width = 4 + len(TIME_FEATURES)
    assert seq.values.shape[2] == width + len(LINK_FEATURES)
    assert torch.equal(seq.values[..., :width], event_sequences(batch).values)
    for i, g in enumerate(graphs[:8]):
        n = g["vm"].num_nodes
        flags = seq.values[i, :n, width:]
        assert (flags[0] == 0).all()
        either = (flags[1:] > 0).any(dim=1).numpy()
        assert (either == steps(g).shared).all()
        network, _ = vm_resources(g)
        order = torch.argsort(g["vm"].event_order).numpy()
        same = network[order][1:] == network[order][:-1]
        assert (flags[1:, 0].numpy() == same).all()


def test_links_change_the_stream_input():
    assert make_stream(links=True).input[0].in_features == 4 + 4
    assert build_temporal_stream(ModelConfig(step_links=False)).input[0].in_features == 4 + 2
    with pytest.raises(ConfigError, match="step_links"):
        ModelConfig(step_links="yes")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda t: t.index_fill(0, torch.tensor([1]), float("nan")), "finite"),
        (lambda t: t - 1.0, "negative"),
        (lambda t: t.flip(0), "backwards"),
    ],
)
def test_invalid_times(graphs, change, message):
    g = edited(graphs[0], rel_time_hours=change(graphs[0]["vm"].rel_time_hours))
    with pytest.raises(TemporalInputError, match=message):
        event_sequences(collate([g]))


def test_graph_without_events(graphs):
    empty = copy.copy(graphs[1])
    empty["vm"].x = empty["vm"].x[:0]
    empty["vm"].rel_time_hours = empty["vm"].rel_time_hours[:0]
    empty["vm"].event_order = empty["vm"].event_order[:0]
    empty["vm"].num_nodes = 0
    with pytest.raises(TemporalInputError, match=r"graph\(s\) \[1\] have no VM events"):
        event_sequences(collate([graphs[0], empty]))


def test_output_shape(graphs):
    out = make_stream()(collate(graphs[:6]))
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()


def test_padding_does_not_change_results(graphs):
    stream = make_stream()
    short = min(graphs, key=lambda g: g["vm"].num_nodes)
    long = max(graphs, key=lambda g: g["vm"].num_nodes)
    assert short["vm"].num_nodes < long["vm"].num_nodes
    alone = stream(collate([short]))
    together = stream(collate([long, short]))[1:]
    assert torch.allclose(alone, together, atol=1e-6)


def test_states_only_depend_on_earlier_events(graphs):
    stream = make_stream()
    g = max(graphs, key=lambda g: g["vm"].num_nodes)
    x = g["vm"].x.clone()
    x[-1] += 10.0
    states, _ = stream.step_states(collate([g]))
    changed, _ = stream.step_states(collate([edited(g, x=x)]))
    n = g["vm"].num_nodes
    assert torch.allclose(states[0, : n - 1], changed[0, : n - 1])
    assert not torch.allclose(states[0, n - 1], changed[0, n - 1])


def test_timing_affects_the_output(graphs):
    stream = make_stream()
    g = graphs[0]
    spread = edited(g, rel_time_hours=g["vm"].rel_time_hours * 20)
    assert not torch.allclose(stream(collate([g])), stream(collate([spread])))


def test_gradients_reach_every_parameter(graphs):
    stream = TemporalSeverityStream(hidden_dim=16, out_dim=8, num_layers=2, dropout=0.1)
    stream(collate(graphs)).sum().backward()
    for name, param in stream.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_reproducible_with_seed(graphs):
    batch = collate(graphs[:4])
    assert torch.equal(make_stream()(batch), make_stream()(batch))


def test_build_from_config():
    config = ModelConfig(hidden_dim=32, temporal_dim=24, temporal_layers=2, dropout=0.2)
    stream = build_temporal_stream(config)
    assert stream.out_dim == 24
    assert stream.gru.num_layers == 2
    assert stream.gru.dropout == 0.2
    single = build_temporal_stream(ModelConfig(temporal_layers=1, dropout=0.2))
    # torch only applies GRU dropout between layers.
    assert single.gru.dropout == 0.0
    with pytest.raises(ConfigError, match="temporal_layers"):
        ModelConfig(temporal_layers=0)


class MeanPoolSpatial(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.encoder = NodeEncoder(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        h = self.encoder(batch)
        return self.head(global_mean_pool(h["vm"], batch["vm"].batch, size=batch.num_graphs))


def test_inside_the_dual_stream_model(graphs):
    config = ModelConfig(hidden_dim=16, temporal_dim=8, spatial_dim=4, dropout=0.0)
    model = DualStreamModel(config, build_temporal_stream(config), MeanPoolSpatial(16, 4))
    model.fit_scaler(graphs[:12])
    out = model(collate(graphs))
    assert out.temporal.shape == (len(graphs), 8)
    out.temporal.sum().backward()
    assert all(p.grad is not None for p in model.temporal.parameters())
