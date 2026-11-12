import json

import numpy as np
import pytest
import torch

from cadgnn.baselines import (
    FEATURE_NAMES,
    episode_features,
    fit_baseline,
    fit_estimator,
    rule_scores,
)
from cadgnn.baselines.chained import ChainedGatGru, snapshots
from cadgnn.baselines.features import steps, vm_resources
from cadgnn.baselines.run import baseline_results, run_baselines
from cadgnn.config import ModelConfig, parse_config
from cadgnn.fusion import build_coordination_model, coordination_loss
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE, save_episode_graphs
from cadgnn.models import collate, seed_everything
from cadgnn.training import stratified_split
from test_spatial_stream import episode

CHAINED = ModelConfig(architecture="chained", hidden_dim=8, dropout=0.0, device="cpu")


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(
        infra, clean_events(small_dataset.events, infra), patterns=small_dataset.patterns
    )


def parts(graphs, seed=1):
    split = stratified_split(graphs, 0.2, 0.2, seed)
    return [graphs[i] for i in split.train], [graphs[i] for i in split.val]


def test_resources_and_steps_of_a_hand_built_episode():
    g = episode(networks=[0, 0, 1], storages=[0, 1, 1])
    network, storage = vm_resources(g)
    assert network.tolist() == [0, 0, 1] and storage.tolist() == [0, 1, 1]
    step = steps(g)
    assert step.gaps.tolist() == [1.0, 1.0]
    # vm0 -> vm1 share network 0, vm1 -> vm2 share storage 1.
    assert step.shared.tolist() == [True, True]
    assert steps(episode(networks=[0, 1], storages=[0, 1])).shared.tolist() == [False]


def test_episode_features():
    g = episode(networks=[0, 0, 0, 1], storages=[0, 1, 2, 3])
    values = dict(zip(FEATURE_NAMES, episode_features(g), strict=True))
    assert values["n_vms"] == 4
    assert values["n_networks"] == 2
    assert values["largest_network_share"] == pytest.approx(0.75)
    assert values["n_components"] == 2
    assert values["log_span_hours"] == pytest.approx(np.log1p(3.0))
    assert values["shared_step_share"] == pytest.approx(2 / 3)
    assert values["gap_variation"] == pytest.approx(0.0)
    assert np.isfinite(list(values.values())).all()


def test_features_of_a_single_event_are_finite():
    assert np.isfinite(episode_features(episode(networks=[0], storages=[0]))).all()


def test_rule_score_depends_on_the_window():
    g = episode(networks=[0, 0, 0], storages=[0, 1, 2])
    assert rule_scores([g], window_hours=2.0).tolist() == [1.0]
    assert rule_scores([g], window_hours=0.5).tolist() == [0.0]


@pytest.mark.parametrize("name", ["rule_based", "logistic_regression", "random_forest"])
def test_fitted_baselines_score_every_episode(graphs, name):
    train, val = parts(graphs)
    fitted = fit_baseline(name, train, val, seed=0)
    scores = fitted.score(graphs)
    assert scores.shape == (len(graphs),)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert 0 <= fitted.settings["val_roc_auc"] <= 1
    json.dumps(fitted.settings)


@pytest.mark.parametrize("balancing", ["none", "oversample", "undersample", "smote"])
def test_feature_baselines_with_other_balancing(graphs, balancing):
    train, val = parts(graphs)
    fitted = fit_baseline("logistic_regression", train, val, seed=0, balancing=balancing)
    assert fitted.score(graphs).shape == (len(graphs),)
    assert set(fitted.options) == {"C"}


def test_fit_estimator_balances_only_what_it_is_given():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3))
    y = np.array([0] * 20 + [1] * 10)
    model = fit_estimator("random_forest", {"max_depth": 2}, x, y, seed=0, balancing="oversample")
    assert model.class_weight is None
    with pytest.raises(ValueError, match="unknown balancing"):
        fit_estimator("random_forest", {}, x, y, seed=0, balancing="tomek")


def test_unknown_baseline(graphs):
    with pytest.raises(ValueError, match="unknown baseline"):
        fit_baseline("svm", *parts(graphs), seed=0)


def test_baselines_use_the_training_split(tmp_path, graphs):
    config = parse_config({"seed": 3, "paths": {"output_dir": str(tmp_path)}}, base_dir=tmp_path)
    found = baseline_results(graphs, config, ["rule_based", "random_forest"])
    split = stratified_split(graphs, 0.2, 0.2, 3)
    for outcome in found.values():
        assert outcome["test_metrics"]["n_episodes"] == len(split.test)
        assert outcome["split_sizes"]["train"] == len(split.train)
        assert set(outcome["by_pattern"]) <= {"scattered", "fast"}
        assert outcome["time_to_detect"]["n_campaigns"] == outcome["test_metrics"]["n_coordinated"]
    assert found["rule_based"]["balancing"] == "n/a"
    assert found["random_forest"]["balancing"] == "class_weight"


def test_run_baselines_writes_metrics(tmp_path, graphs):
    save_episode_graphs(graphs, tmp_path / EPISODE_GRAPHS_FILE)
    config = parse_config({"paths": {"output_dir": str(tmp_path)}}, base_dir=tmp_path)
    found = run_baselines(config)
    saved = json.loads(found["path"].read_text(encoding="utf-8"))
    assert {"rule_based", "logistic_regression", "random_forest"} <= set(saved)


def test_snapshots_grow_one_event_at_a_time(graphs):
    batch = collate(graphs[:3])
    snaps, times, counts = snapshots(batch)
    assert counts == [g["vm"].num_nodes for g in graphs[:3]]
    assert snaps.num_graphs == sum(counts)
    sizes = torch.bincount(snaps["vm"].batch).tolist()
    expected = [k for n in counts for k in range(1, n + 1)]
    assert sizes == expected
    assert times[0] == 0


def test_chained_model_trains_and_scores(graphs):
    seed_everything(0)
    model = build_coordination_model(CHAINED)
    assert isinstance(model, ChainedGatGru)
    model.fit_scaler(graphs)
    batch = collate(graphs[:6])
    out = model(batch)
    assert out.logit.shape == (6,) and out.gate is None
    loss = coordination_loss(out.logit, batch.y)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gat.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gru.parameters())
    scores = model.score(batch)
    assert ((scores > 0) & (scores < 1)).all()


def test_chained_score_does_not_depend_on_batch_neighbours(graphs):
    seed_everything(0)
    model = build_coordination_model(CHAINED).fit_scaler(graphs).eval()
    with torch.no_grad():
        alone = model(collate([graphs[2]])).logit
        together = model(collate(graphs[:5])).logit[2]
    assert torch.allclose(alone, together, atol=1e-5)


def test_unknown_architecture():
    with pytest.raises(ValueError, match="architecture must be one of"):
        ModelConfig(architecture="transformer")
