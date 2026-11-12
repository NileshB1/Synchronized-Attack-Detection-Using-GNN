import pytest
import torch
import torch.nn.functional as F

from cadgnn.config import ConfigError, ModelConfig
from cadgnn.fusion import (
    ConcatFusion,
    GatedFusion,
    SingleStreamProjection,
    build_coordination_model,
    build_fusion,
    coordination_loss,
    positive_weight,
)
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.models import (
    ModelInputError,
    collate,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)

SMALL = dict(hidden_dim=16, temporal_dim=8, spatial_dim=12, fusion_dim=10, dropout=0.0)


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


def make_model(**overrides):
    seed_everything(0)
    return build_coordination_model(ModelConfig(**{**SMALL, **overrides}))


def test_gated_fusion_mixes_the_streams():
    fusion = GatedFusion(temporal_dim=8, spatial_dim=12, out_dim=10)
    t, s = torch.randn(5, 8), torch.randn(5, 12)
    h, gate = fusion(t, s)
    assert h.shape == (5, 10)
    assert gate.shape == (5,)
    assert ((gate > 0) & (gate < 1)).all()
    # Same formula written out by hand.
    z = torch.sigmoid(fusion.gate(torch.cat([t, s], -1)))
    expected = z * torch.tanh(fusion.temporal(t)) + (1 - z) * torch.tanh(fusion.spatial(s))
    assert torch.allclose(h, expected)


def test_gate_follows_the_stronger_stream():
    fusion = GatedFusion(temporal_dim=2, spatial_dim=2, out_dim=2)
    with torch.no_grad():
        fusion.gate.weight.zero_()
        fusion.gate.bias.fill_(10.0)
    t, s = torch.randn(3, 2), torch.randn(3, 2)
    h, gate = fusion(t, s)
    # A gate near 1 means the result is almost all temporal.
    assert torch.allclose(gate, torch.ones(3), atol=1e-4)
    assert torch.allclose(h, torch.tanh(fusion.temporal(t)), atol=1e-4)


def test_concat_fusion():
    fusion = ConcatFusion(temporal_dim=8, spatial_dim=12, out_dim=10)
    h, gate = fusion(torch.randn(4, 8), torch.randn(4, 12))
    assert h.shape == (4, 10)
    assert gate is None


@pytest.mark.parametrize("fusion_cls", [GatedFusion, ConcatFusion])
def test_pair_fusions_reject_bad_input(fusion_cls):
    fusion = fusion_cls(8, 12, 10)
    with pytest.raises(ValueError, match="both stream"):
        fusion(torch.randn(4, 8), None)
    with pytest.raises(ValueError, match="number of episodes"):
        fusion(torch.randn(4, 8), torch.randn(3, 12))


def test_single_stream_projection():
    fusion = SingleStreamProjection(in_dim=12, out_dim=10, stream="spatial")
    h, gate = fusion(None, torch.randn(4, 12))
    assert h.shape == (4, 10) and gate is None
    with pytest.raises(ValueError, match="only the spatial"):
        fusion(torch.randn(4, 8), torch.randn(4, 12))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, GatedFusion),
        ({"fusion": "concat"}, ConcatFusion),
        ({"use_spatial": False}, SingleStreamProjection),
        ({"use_temporal": False}, SingleStreamProjection),
    ],
)
def test_build_fusion(overrides, expected):
    assert isinstance(build_fusion(ModelConfig(**{**SMALL, **overrides})), expected)


@pytest.mark.parametrize("fusion", ["gated", "concat"])
def test_full_model_forward(graphs, fusion):
    model = make_model(fusion=fusion).fit_scaler(graphs)
    out = model(collate(graphs))
    n = len(graphs)
    assert out.logit.shape == (n,)
    assert out.temporal.shape == (n, 8)
    assert out.spatial.shape == (n, 12)
    assert (out.gate is not None) == (fusion == "gated")


@pytest.mark.parametrize(
    "overrides, temporal_on, spatial_on",
    [({"use_spatial": False}, True, False), ({"use_temporal": False}, False, True)],
)
def test_single_stream_baselines(graphs, overrides, temporal_on, spatial_on):
    model = make_model(**overrides).fit_scaler(graphs)
    assert (model.streams.temporal is not None) == temporal_on
    assert (model.streams.spatial is not None) == spatial_on
    out = model(collate(graphs[:5]))
    assert out.logit.shape == (5,)
    assert (out.temporal is not None) == temporal_on
    assert (out.spatial is not None) == spatial_on
    assert out.gate is None


def test_gradients_flow_through_both_streams_and_fusion(graphs):
    model = make_model().fit_scaler(graphs)
    batch = collate(graphs)
    loss = coordination_loss(model(batch).logit, batch.y, positive_weight(graphs))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    for part in (model.streams.temporal, model.streams.spatial, model.fusion, model.classifier):
        assert any(p.grad.abs().sum() > 0 for p in part.parameters())


def test_score_is_uncalibrated_sigmoid_in_eval_mode(graphs):
    model = make_model(dropout=0.3).fit_scaler(graphs)
    model.train()
    batch = collate(graphs[:6])
    scores = model.score(batch)
    assert model.training
    model.eval()
    assert torch.allclose(scores, torch.sigmoid(model(batch).logit))
    assert ((scores > 0) & (scores < 1)).all()


def test_model_can_fit_a_small_set(graphs):
    model = make_model().fit_scaler(graphs)
    batch = collate(graphs)
    pos_weight = positive_weight(graphs)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(40):
        optimiser.zero_grad()
        loss = coordination_loss(model(batch).logit, batch.y, pos_weight)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    # Only checks that training moves the loss; no claim about held-out performance.
    assert losses[-1] < losses[0]


def test_positive_weight(graphs):
    n_pos = sum(int(g.y) for g in graphs)
    assert float(positive_weight(graphs)) == pytest.approx((len(graphs) - n_pos) / n_pos)
    only_coincidental = [g for g in graphs if int(g.y) == 0]
    with pytest.raises(ValueError, match="both classes"):
        positive_weight(only_coincidental)
    with pytest.raises(ValueError, match="both classes"):
        positive_weight([])


def test_coordination_loss_matches_torch():
    logit = torch.tensor([2.0, -1.0, 0.5])
    y = torch.tensor([1, 0, 1])
    pos_weight = torch.tensor(2.0)
    expected = F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=pos_weight)
    assert torch.allclose(coordination_loss(logit, y, pos_weight), expected)
    assert torch.allclose(
        coordination_loss(logit, y), F.binary_cross_entropy_with_logits(logit, y.float())
    )


def test_coordination_loss_rejects_bad_labels():
    with pytest.raises(ValueError, match="does not match"):
        coordination_loss(torch.zeros(3), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="0 .coincidental. or 1"):
        coordination_loss(torch.zeros(2), torch.tensor([0, 2]))


def test_checkpoint_round_trip(tmp_path, graphs):
    model = make_model().fit_scaler(graphs)
    save_checkpoint(model, tmp_path / "model.pt")
    restored = make_model()
    load_checkpoint(restored, tmp_path / "model.pt")
    batch = collate(graphs[:5])
    assert torch.equal(model.score(batch), restored.score(batch))
    with pytest.raises(ModelInputError, match="model config"):
        load_checkpoint(make_model(fusion="concat"), tmp_path / "model.pt")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"use_temporal": False, "use_spatial": False}, "at least one"),
        ({"fusion": "attention"}, "fusion must be one of"),
        ({"fusion_dim": 0}, "fusion_dim"),
        ({"use_spatial": "no"}, "true or false"),
    ],
)
def test_config_validation(overrides, message):
    with pytest.raises(ConfigError, match=message):
        ModelConfig(**{**SMALL, **overrides})


def test_false_alarm_cost_weighs_coincidental_episodes():
    logit = torch.tensor([0.3, -1.0, 2.0])
    y = torch.tensor([1, 0, 0])
    each = F.binary_cross_entropy_with_logits(logit, y.float(), reduction="none")
    expected = (each * torch.tensor([1.0, 3.0, 3.0])).mean()
    assert torch.allclose(coordination_loss(logit, y, neg_weight=3.0), expected)
    assert torch.allclose(coordination_loss(logit, y, neg_weight=1.0), each.mean())
