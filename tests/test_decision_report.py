import json

import numpy as np
import pytest
import torch

from cadgnn.config import ModelConfig
from cadgnn.evaluation import (
    NOT_EVALUATED,
    Predictions,
    episode_decisions,
    predict,
    prefix_graph,
    render_report,
    time_to_detect,
    write_evaluation,
)
from cadgnn.evaluation.metrics import classification_metrics
from cadgnn.fusion import build_coordination_model
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.models import check_batch, collate, seed_everything
from test_spatial_stream import episode


@pytest.fixture(scope="module")
def built(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return infra, build_episode_graphs(infra, clean_events(small_dataset.events, infra))


@pytest.fixture(scope="module")
def model(built):
    seed_everything(0)
    config = ModelConfig(hidden_dim=16, temporal_dim=8, spatial_dim=8, fusion_dim=8, dropout=0.0)
    return build_coordination_model(config).fit_scaler(built[1]).eval()


def test_prefix_graph_keeps_only_the_first_events(built):
    _, graphs = built
    g = max(graphs, key=lambda g: g["vm"].num_nodes)
    p = prefix_graph(g, 2)
    assert p["vm"].num_nodes == 2
    assert p["vm"].event_order.tolist() == [0, 1]
    assert p["backup_image"].num_nodes == 2
    assert p["vm"].global_index.tolist() == g["vm"].global_index[:2].tolist()
    assert p.y.tolist() == g.y.tolist() and p.episode_id == g.episode_id
    check_batch(collate([p]))
    full = prefix_graph(g, g["vm"].num_nodes)
    assert torch.equal(full["vm"].x, g["vm"].x)
    with pytest.raises(ValueError, match="between 1 and"):
        prefix_graph(g, 0)


def test_prefix_graph_drops_unused_resources():
    g = episode(networks=[0, 1, 1, 2], storages=[0, 0, 1, 2])
    p = prefix_graph(g, 2)
    assert (p["network"].num_nodes, p["storage"].num_nodes) == (2, 1)
    assert p["network"].global_index.tolist() == [0, 1]


def test_time_to_detect_bounds(built, model):
    _, graphs = built
    campaigns = [g for g in graphs if int(g.y) == 1]
    always = time_to_detect(model, graphs, threshold=0.0, min_events=2)
    assert always["n_campaigns"] == len(campaigns)
    assert always["detection_rate"] == 1.0
    for e, g in zip(always["episodes"], campaigns, strict=True):
        assert e["events_needed"] == 2
        assert e["hours_to_detect"] == pytest.approx(float(g["vm"].rel_time_hours[1]))
    never = time_to_detect(model, graphs, threshold=1.1)
    assert never["detected"] == 0
    assert never["mean_hours_to_detect"] is None
    assert all(e["hours_to_detect"] is None for e in never["episodes"])


def test_time_to_detect_without_campaigns(built, model):
    _, graphs = built
    result = time_to_detect(model, [g for g in graphs if int(g.y) == 0], threshold=0.5)
    assert result["n_campaigns"] == 0 and result["detection_rate"] is None


def test_decisions_follow_scores_and_threshold(built, model):
    infra, graphs = built
    pred = predict(model, graphs)
    threshold = float(np.median(pred.scores))
    decisions = episode_decisions(graphs, pred, threshold, infra)
    assert [d["episode_id"] for d in decisions] == pred.episode_ids
    for d, score, y in zip(decisions, pred.scores, pred.y, strict=True):
        assert d["score"] == pytest.approx(float(score))
        assert d["decision"] == ("coordinated" if score >= threshold else "coincidental")
        assert d["true_label"] == ("coordinated" if y == 1 else "coincidental")
        assert d["correct"] == (d["decision"] == d["true_label"])
        assert (d["propagation_path"] is not None) == (d["decision"] == "coordinated")
        assert d["pattern"] == "unknown"
        assert d["time_window_hours"]["span"] >= 0
        assert 0 < d["temporal_share"] < 1


def test_vms_are_listed_in_event_order(built, model):
    infra, graphs = built
    g = graphs[0]
    d = episode_decisions([g], predict(model, [g]), 0.5, infra)[0]
    expected = [infra.node_ids["vm"][i] for i in g["vm"].global_index.tolist()]
    assert d["vms"] == expected
    assert d["time_window_hours"]["start"] == pytest.approx(float(g["vm"].timestamp_hours[0]))


def fixed_predictions(graphs, scores, gates=None):
    return Predictions(
        episode_ids=[g.episode_id for g in graphs],
        y=np.array([int(g.y) for g in graphs]),
        scores=np.array(scores),
        gates=gates,
    )


def test_shared_resources_and_propagation_path():
    g = episode(networks=[0, 0, 1], storages=[0, 1, 1])
    g.episode_id = "hand"
    d = episode_decisions([g], fixed_predictions([g], [0.9]), threshold=0.5)[0]
    # Without the infrastructure graph, nodes are named by type and index.
    assert d["vms"] == ["vm#0", "vm#1", "vm#2"]
    assert d["shared_resources"] == {
        "network": {"network#0": ["vm#0", "vm#1"]},
        "storage": {"storage#1": ["vm#1", "vm#2"]},
    }
    assert [step["via"] for step in d["propagation_path"]] == [["network#0"], ["storage#1"]]
    assert d["propagation_path"][0]["hours_between"] == pytest.approx(1.0)
    assert d["temporal_share"] is None


def test_not_flagged_episode_has_no_path():
    g = episode(networks=[0, 0], storages=[0, 1])
    d = episode_decisions([g], fixed_predictions([g], [0.2]), threshold=0.5)[0]
    assert d["decision"] == "coincidental"
    assert d["propagation_path"] is None


def results_for(decisions, preds, threshold):
    return {
        "seed": 1,
        "model_config": {"fusion": "gated", "use_temporal": True, "use_spatial": True},
        "split_sizes": {"train": 10, "val": 4, "test": len(decisions)},
        "threshold_strategy": "f1",
        "test_metrics": classification_metrics(preds.y, preds.scores, threshold),
        "time_to_detect": {
            "detected": 1,
            "n_campaigns": 1,
            "mean_hours_to_detect": 0.5,
            "median_hours_to_detect": 0.5,
        },  # fmt: skip
        "not_evaluated": NOT_EVALUATED,
    }


def test_report_and_files(tmp_path):
    graphs = [episode(networks=[0, 0], storages=[0, 1]), episode(networks=[0, 1], storages=[0, 1])]
    graphs[0].episode_id, graphs[0].y = "coordinated_0", torch.tensor([1])
    graphs[1].episode_id = "coincidental_0"
    preds = fixed_predictions(graphs, [0.8, 0.3], gates=np.array([0.4, 0.6]))
    decisions = episode_decisions(graphs, preds, 0.5)
    results = results_for(decisions, preds, 0.5)

    report = render_report(results, decisions)
    for section in ("## Held-out test set", "## Time to detect", "## Decisions",
                    "## Propagation evidence", "## Not evaluated"):  # fmt: skip
        assert section in report
    assert "coordinated_0: vm#0 -> vm#1 (shared: network#0)" in report
    assert "do not show performance on real attack campaigns" in report
    assert "## Flagged episodes by pattern" not in report

    with_patterns = {**results, "by_pattern": {
        "fast": {"label": "coordinated", "episodes": 1, "flagged": 1, "flagged_rate": 1.0}
    }}  # fmt: skip
    assert "| fast | coordinated | 1 | 1 | 1.000 |" in render_report(with_patterns, decisions)
    chained = {**results, "model_config": {"architecture": "chained"}}
    assert "CONTINUUM-style baseline" in render_report(chained, decisions)

    paths = write_evaluation(tmp_path / "evaluation", results, decisions)
    assert json.loads(paths["metrics"].read_text())["test_metrics"]["f1"] == 1.0
    assert len(json.loads(paths["decisions"].read_text())) == 2
    assert paths["report"].read_text(encoding="utf-8") == report
