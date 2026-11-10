import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from cadgnn.data.calibration import calibrate
from cadgnn.cli import main
from cadgnn.generator import save_outputs
from cadgnn.graph import (
    EDGE_TYPES,  NODE_TYPES,  VM_FEATURES,
    GraphConstructionError,  InfrastructureGraph,
    all_edge_types,  anomaly_episodes,
    build_episode_graph,  build_episode_graphs,
    clean_events, episode_summary,
    load_episode_graphs, load_episode_patterns, load_generator_output,
    save_episode_graphs
)
from cadgnn.graph.schema import reverse
from conftest import REPO_ROOT


@pytest.fixture(scope="module")
def built(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    events = clean_events(small_dataset.events, infra)
    return infra, events, build_episode_graphs(infra, events)


def test_one_graph_per_episode_with_aligned_labels(built):
    _, events, graphs = built
    episodes = anomaly_episodes(events)
    assert [g.episode_id for g in graphs] == list(episodes)
    for g in graphs:
        label = episodes[g.episode_id]["label"].iloc[0]
        assert g.y.tolist() == [1 if label == "coordinated" else 0]
    assert sum(int(g.y) for g in graphs) == 8


def test_vm_features_and_times_match_source_rows(built):
    infra, events, graphs = built
    episodes = anomaly_episodes(events)
    for g in graphs:
        ep = episodes[g.episode_id]
        vm_ids = [infra.node_ids["vm"][i] for i in g["vm"].global_index.tolist()]
        assert vm_ids == ep["vm"].tolist()
        expected = torch.tensor(ep[list(VM_FEATURES)].to_numpy(), dtype=torch.float32)
        assert torch.equal(g["vm"].x, expected)
        assert g["vm"].x.shape == (len(ep), 4)
        times = torch.tensor(ep["timestamp_hours"].to_numpy(), dtype=torch.float32)
        assert torch.allclose(g["vm"].timestamp_hours, times)
        assert g["vm"].rel_time_hours[0] == 0
        assert bool((g["vm"].rel_time_hours.diff() >= 0).all())
        assert g["vm"].event_order.tolist() == list(range(len(ep)))


def test_every_graph_has_the_same_schema(built):
    _, _, graphs = built
    for g in graphs:
        assert set(g.node_types) == set(NODE_TYPES)
        assert set(g.edge_types) == set(all_edge_types())
        for et in all_edge_types():
            assert g[et].edge_index.dtype == torch.long


def test_edges_exist_in_the_infrastructure(built):
    infra, _, graphs = built
    for g in graphs:
        for et in EDGE_TYPES:
            src, _, dst = et
            local = g[et].edge_index
            # Map local indices back to infrastructure indices and check each edge exists.
            global_pairs = {
                (int(g[src].global_index[s]), int(g[dst].global_index[d])) for s, d in local.t()
            }
            assert global_pairs <= set(map(tuple, infra.edges[et].T.tolist()))
            assert torch.equal(g[reverse(et)].edge_index, local.flip(0))


def test_each_vm_brings_its_image_storage_and_network(built):
    _, _, graphs = built
    for g in graphs:
        n_vm = g["vm"].num_nodes
        assert g["backup_image"].num_nodes == n_vm
        assert g["vm", "has_backup", "backup_image"].edge_index.size(1) == n_vm
        assert g["vm", "attached_to", "network"].edge_index.size(1) == n_vm
        assert g["backup_image", "stored_on", "storage"].edge_index.size(1) == n_vm
        assert 1 <= g["storage"].num_nodes <= n_vm
        assert 1 <= g["network"].num_nodes <= n_vm


def test_coordinated_episodes_share_a_network(built):
    # This follows from the generator: campaigns move between VMs on one fabric.
    _, _, graphs = built
    for g in graphs:
        if int(g.y) == 1:
            assert g["network"].num_nodes == 1


def test_construction_is_deterministic(built, small_dataset):
    infra, events, graphs = built
    again = build_episode_graphs(InfrastructureGraph.from_networkx(small_dataset.graph), events)
    for a, b in zip(graphs, again, strict=True):
        assert a.episode_id == b.episode_id
        for store_a, store_b in zip(a.stores, b.stores, strict=True):
            for key in store_a.keys():
                va, vb = store_a[key], store_b[key]
                assert torch.equal(va, vb) if torch.is_tensor(va) else va == vb


def test_short_episodes_are_skipped(built, caplog):
    infra, events, graphs = built
    assert len(build_episode_graphs(infra, events, min_events=4)) < len(graphs)
    assert "skipped" in caplog.text
    with pytest.raises(ValueError):
        build_episode_graphs(infra, events, min_events=0)


def test_no_episodes(built, caplog):
    infra, events, _ = built
    assert build_episode_graphs(infra, events[events["label"] == "none"]) == []
    assert "no anomaly episodes" in caplog.text


def test_benign_episode_is_rejected(built):
    infra, events, _ = built
    benign = events[events["label"] == "none"].head(3)
    with pytest.raises(GraphConstructionError, match="label"):
        build_episode_graph(infra, benign, "benign_0")


def test_batching_keeps_graphs_apart(built):
    _, _, graphs = built
    batch = next(iter(DataLoader(graphs[:4], batch_size=4)))
    assert batch.num_graphs == 4
    assert batch["vm"].x.shape[0] == sum(g["vm"].num_nodes for g in graphs[:4])
    assert batch.y.tolist() == [int(g.y) for g in graphs[:4]]
    assert batch.episode_id == [g.episode_id for g in graphs[:4]]
    edge_index = batch["vm", "attached_to", "network"].edge_index
    assert int(edge_index[1].max()) < batch["network"].num_nodes


def test_save_and_load(tmp_path, built):
    _, _, graphs = built
    path = tmp_path / "graphs" / "episode_graphs.pt"
    save_episode_graphs(graphs, path)
    loaded = load_episode_graphs(path)
    assert [g.episode_id for g in loaded] == [g.episode_id for g in graphs]
    assert torch.equal(loaded[0]["vm"].x, graphs[0]["vm"].x)
    with pytest.raises(GraphConstructionError, match="not found"):
        load_episode_graphs(tmp_path / "missing.pt")


def test_summary(built):
    _, _, graphs = built
    summary = episode_summary(graphs)
    assert len(summary) == len(graphs)
    coordinated = summary[summary["y"] == 1]
    assert (coordinated["n_components"] == 1).all()
    assert (summary["n_components"] >= 1).all()
    assert (summary["n_backup_image"] == summary["n_vm"]).all()
    assert episode_summary([]).empty


@pytest.fixture
def generator_output(tmp_path, small_dataset, fake_datasets):
    save_outputs(small_dataset, calibrate(fake_datasets), tmp_path / "out")
    return tmp_path / "out"


def test_load_generator_output(generator_output, built):
    infra, events = load_generator_output(generator_output)
    graphs = build_episode_graphs(infra, events)
    _, _, expected = built
    assert [g.episode_id for g in graphs] == [g.episode_id for g in expected]
    # Values went through a CSV round trip; the generator rounds to 4 places.
    assert np.allclose(graphs[0]["vm"].x.numpy(), expected[0]["vm"].x.numpy())


def test_patterns_reach_the_graphs(generator_output, small_dataset):
    patterns = load_episode_patterns(generator_output)
    assert patterns == small_dataset.patterns
    infra, events = load_generator_output(generator_output)
    graphs = build_episode_graphs(infra, events, patterns=patterns)
    assert all(g.pattern == patterns[g.episode_id] for g in graphs)
    assert set(episode_summary(graphs)["pattern"]) <= {"scattered", "fast"}


def test_graphs_without_patterns(tmp_path, built):
    assert load_episode_patterns(tmp_path) == {}
    _, _, graphs = built
    assert {g.pattern for g in graphs} == {"unknown"}


def test_load_generator_output_missing_files(tmp_path):
    with pytest.raises(GraphConstructionError, match="run the generator first"):
        load_generator_output(tmp_path)


def test_cli_build_graphs(generator_output, capsys):
    assert main(["--output-dir", str(generator_output), "build-graphs"]) == 0
    out = capsys.readouterr().out
    assert "18 episode graphs" in out
    assert len(load_episode_graphs(generator_output / "episode_graphs.pt")) == 18
    summary = pd.read_csv(generator_output / "episode_summary.csv")
    assert summary["y"].sum() == 8


def test_cli_build_graphs_without_generator_output(tmp_path):
    assert main(["--output-dir", str(tmp_path), "build-graphs"]) == 1


def test_run_graph_builder_script(generator_output, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_graph_builder.py"), "--output-dir",
         str(generator_output)],
        capture_output=True, text=True,
        cwd=tmp_path,  env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (generator_output / "episode_graphs.pt").is_file()
