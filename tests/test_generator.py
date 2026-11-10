import json
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cadgnn.data.calibration import FeatureProfile, calibrate
from cadgnn.config import GeneratorConfig
from cadgnn.generator import (
    EVENT_COLUMNS,  LABELS,
    NODE_KINDS, PATTERNS,
    build_infrastructure, find_attack_chain,
    generate, generate_events,
    save_outputs, vm_nodes,
)

# Simple profile with disjoint ranges so states can be told apart from the values.
PROFILE = {
    "change_rate": FeatureProfile(0.35, 0.0, 0.1, 0.5, 1.0, ""),
    "file_entropy": FeatureProfile(0.30, 0.3, 0.5, 0.6, 0.8, ""),
    "backup_size_gb": FeatureProfile(0.20, 0.5, 25.0, 0.6, 0.8, ""),
    "file_count": FeatureProfile(0.15, 200, 30000, 0.4, 1.6, ""),
}
SMALL = GeneratorConfig(
    n_vms=60, n_storage=4,
    n_network=5, n_coincidental_episodes=10,
    n_coordinated_episodes=8,
)


def neighbours_of_kind(graph, node, kind):
    return [n for n in graph.neighbors(node) if graph.nodes[n]["kind"] == kind]


def test_infrastructure_structure():
    g = build_infrastructure(30, 4, 3, np.random.default_rng(0))
    kinds = pd.Series(dict(g.nodes(data="kind"))).value_counts()
    assert kinds.to_dict() == {"vm": 30, "backup_image": 30, "storage": 4, "network": 3}
    assert g.number_of_edges() == 3 * 30
    for i in range(30):
        assert neighbours_of_kind(g, f"vm_{i}", "backup_image") == [f"image_{i}"]
        assert len(neighbours_of_kind(g, f"vm_{i}", "network")) == 1
        assert len(neighbours_of_kind(g, f"image_{i}", "storage")) == 1
    assert set(nx.get_node_attributes(g, "kind").values()) <= set(NODE_KINDS)


def test_infrastructure_is_deterministic_per_seed():
    a = build_infrastructure(50, 5, 5, np.random.default_rng(7))
    b = build_infrastructure(50, 5, 5, np.random.default_rng(7))
    c = build_infrastructure(50, 5, 5, np.random.default_rng(8))
    assert list(a.edges()) == list(b.edges())
    assert list(a.edges()) != list(c.edges())


@pytest.mark.parametrize("args", [(0, 1, 1), (1, 0, 1), (1, 1, -2), (2.0, 1, 1), (True, 1, 1)])
def test_infrastructure_rejects_bad_sizes(args):
    with pytest.raises(ValueError, match="positive integer"):
        build_infrastructure(*args, np.random.default_rng(0))


@pytest.mark.parametrize("seed", range(5))
def test_infrastructure_with_random_sizes(seed):
    rng = np.random.default_rng(seed)
    n_vms, n_storage, n_network = (int(x) for x in rng.integers(1, 200, size=3))
    g = build_infrastructure(n_vms, n_storage, n_network, rng)
    assert g.number_of_nodes() == 2 * n_vms + n_storage + n_network
    assert g.number_of_edges() == 3 * n_vms


def test_vm_nodes_are_sorted_as_strings():
    g = build_infrastructure(12, 1, 1, np.random.default_rng(0))
    assert vm_nodes(g)[:4] == ["vm_0", "vm_1", "vm_10", "vm_11"]


def test_attack_chain_moves_through_shared_networks():
    g = build_infrastructure(200, 5, 4, np.random.default_rng(3))
    chain = find_attack_chain(g, "vm_0", 8, np.random.default_rng(4))
    assert len(chain) == 8
    assert len(set(chain)) == 8
    for a, b in zip(chain, chain[1:], strict=False):
        assert neighbours_of_kind(g, a, "network") == neighbours_of_kind(g, b, "network")


def test_attack_chain_stops_when_isolated():
    g = nx.Graph()
    g.add_node("vm_0", kind="vm")
    g.add_node("network_0", kind="network")
    g.add_edge("vm_0", "network_0")
    assert find_attack_chain(g, "vm_0", 5, np.random.default_rng(0)) == ["vm_0"]
    with pytest.raises(ValueError):
        find_attack_chain(g, "vm_0", 0, np.random.default_rng(0))


def test_event_schema_and_label_semantics():
    events = generate(SMALL, PROFILE, seed=1).events
    assert list(events.columns) == EVENT_COLUMNS
    assert set(events["label"]) <= set(LABELS)
    assert events["timestamp_hours"].is_monotonic_increasing

    benign = events[events["label"] == "none"]
    assert len(benign) == SMALL.n_vms * SMALL.benign_backups_per_vm
    assert (benign["state"] == "benign").all()
    assert benign["episode_id"].isna().all()
    assert benign["change_rate"].between(0.0, 0.1).all()

    attacks = events[events["label"] != "none"]
    assert (attacks["state"] == "infected").all()
    assert attacks["change_rate"].between(0.5, 1.0).all()
    assert (attacks["episode_id"].str.split("_").str[0] == attacks["label"]).all()
    assert events["anomaly_score"].between(0, 1).all()


def test_coincidental_episodes():
    events = generate(SMALL, PROFILE, seed=2).events
    episodes = events[events["label"] == "coincidental"].groupby("episode_id")
    assert episodes.ngroups == SMALL.n_coincidental_episodes
    for _, ep in episodes:
        assert 3 <= len(ep) <= 6
        assert ep["vm"].is_unique
        assert ep["timestamp_hours"].between(0, SMALL.observation_hours).all()


def test_coordinated_episodes():
    dataset = generate(SMALL, PROFILE, seed=3)
    events, graph = dataset.events, dataset.graph
    episodes = events[events["label"] == "coordinated"].groupby("episode_id")
    assert episodes.ngroups == SMALL.n_coordinated_episodes
    for _, ep in episodes:
        assert 4 <= len(ep) <= 8
        assert ep["vm"].is_unique
        times = ep["timestamp_hours"].to_numpy()
        assert times[0] <= SMALL.observation_hours - SMALL.campaign_start_margin_hours
        # Rounding to 3 decimals can move each step by up to a millihour.
        gaps = np.diff(times) * 60
        assert ((gaps >= 5 - 0.12) & (gaps <= 45 + 0.12)).all()
        networks = {neighbours_of_kind(graph, vm, "network")[0] for vm in ep["vm"]}
        assert len(networks) == 1


def network_of(graph, vm):
    return neighbours_of_kind(graph, vm, "network")[0]


def overlap_scenario(**settings):
    return GeneratorConfig(
        n_vms=300, n_storage=10, n_network=8,
        n_coincidental_episodes=30, n_coordinated_episodes=30, **settings,
    )  # fmt: skip


def test_lookalike_episodes_share_a_network_and_a_short_window():
    config = overlap_scenario(coincidental_lookalike_fraction=1.0)
    dataset = generate(config, PROFILE, seed=4)
    events = dataset.events
    for _, ep in events[events["label"] == "coincidental"].groupby("episode_id"):
        assert len({network_of(dataset.graph, vm) for vm in ep["vm"]}) == 1
        assert np.ptp(ep["timestamp_hours"]) <= config.coincidental_window_hours[1]


def test_burst_episodes_stay_in_a_short_window():
    config = overlap_scenario(coincidental_burst_fraction=1.0)
    events = generate(config, PROFILE, seed=5).events
    for _, ep in events[events["label"] == "coincidental"].groupby("episode_id"):
        assert np.ptp(ep["timestamp_hours"]) <= config.coincidental_window_hours[1]


def test_cluster_episodes_share_a_network():
    dataset = generate(overlap_scenario(coincidental_cluster_fraction=1.0), PROFILE, seed=6)
    events = dataset.events
    for _, ep in events[events["label"] == "coincidental"].groupby("episode_id"):
        assert len({network_of(dataset.graph, vm) for vm in ep["vm"]}) == 1


def test_stealth_campaigns_use_slow_hops_and_stay_observed():
    config = overlap_scenario(stealth_campaign_fraction=1.0)
    events = generate(config, PROFILE, seed=7).events
    for _, ep in events[events["label"] == "coordinated"].groupby("episode_id"):
        assert ep["timestamp_hours"].max() <= config.observation_hours
        assert (np.diff(ep["timestamp_hours"]) * 60 >= 120 - 0.12).all()


def test_storage_pivots_leave_the_network():
    g = build_infrastructure(200, 3, 20, np.random.default_rng(3))
    chain = find_attack_chain(g, "vm_0", 8, np.random.default_rng(4), storage_pivot_probability=1.0)
    assert len(set(chain)) == len(chain) > 1
    pools = {
        neighbours_of_kind(g, neighbours_of_kind(g, vm, "backup_image")[0], "storage")[0]
        for vm in chain
    }
    assert len(pools) == 1
    assert len({network_of(g, vm) for vm in chain}) > 1


def test_missed_events_leave_at_least_two():
    config = overlap_scenario(missed_event_probability=1.0)
    events = generate(config, PROFILE, seed=8).events
    sizes = events[events["label"] != "none"].groupby("episode_id").size()
    assert (sizes == 2).all()


def test_stray_vm_joins_the_episode_with_its_label():
    plain = generate(overlap_scenario(), PROFILE, seed=9).events
    noisy = generate(overlap_scenario(stray_vm_probability=1.0), PROFILE, seed=9).events
    assert (noisy["label"] != "none").sum() > (plain["label"] != "none").sum()
    attacks = noisy[noisy["label"] != "none"]
    assert not attacks.duplicated(["episode_id", "vm"]).any()
    assert (attacks["episode_id"].str.split("_").str[0] == attacks["label"]).all()


def test_overlap_settings_do_not_disturb_the_plain_scenario():
    # Zero shares must make no random draws, or the notebook match would break.
    a = generate(SMALL, PROFILE, seed=10).events
    b = generate(replace(SMALL, coincidental_window_hours=(1.0, 2.0)), PROFILE, seed=10).events
    pd.testing.assert_frame_equal(a, b)


def test_every_episode_gets_a_pattern():
    config = overlap_scenario(
        coincidental_burst_fraction=0.3, coincidental_cluster_fraction=0.3,
        coincidental_lookalike_fraction=0.3, stealth_campaign_fraction=0.5,
    )  # fmt: skip
    dataset = generate(config, PROFILE, seed=12)
    episodes = set(dataset.events["episode_id"].dropna())
    assert set(dataset.patterns) == episodes
    for episode_id, pattern in dataset.patterns.items():
        assert pattern in PATTERNS[episode_id.split("_")[0]]
    assert len(set(dataset.patterns.values())) == 6


def test_plain_scenario_has_only_the_notebook_patterns():
    patterns = generate(SMALL, PROFILE, seed=1).patterns
    assert set(patterns.values()) == {"scattered", "fast"}


def test_generation_is_reproducible():
    a = generate(SMALL, PROFILE, seed=11)
    b = generate(SMALL, PROFILE, seed=11)
    c = generate(SMALL, PROFILE, seed=12)
    pd.testing.assert_frame_equal(a.events, b.events)
    assert not a.events.equals(c.events)


def test_empty_scenario_keeps_schema():
    config = GeneratorConfig(
        n_vms=3, benign_backups_per_vm=0, n_coincidental_episodes=0, n_coordinated_episodes=0
    )
    events = generate(config, PROFILE, seed=0).events
    assert events.empty
    assert list(events.columns) == EVENT_COLUMNS


def test_short_chains_are_reported(caplog):
    # Chains need at least 4 VMs, but the graph only has 3.
    config = GeneratorConfig(
        n_vms=3, n_network=3, n_coincidental_episodes=0, n_coordinated_episodes=5
    )
    generate(config, PROFILE, seed=0)
    assert "ended early" in caplog.text


def test_generate_events_checks_graph_size():
    g = build_infrastructure(2, 1, 1, np.random.default_rng(0))
    config = GeneratorConfig(n_vms=10)
    with pytest.raises(ValueError, match="coincidental"):
        generate_events(g, PROFILE, config, np.random.default_rng(0))


def test_save_outputs(tmp_path, fake_datasets):
    calibration = calibrate(fake_datasets)
    dataset = generate(SMALL, calibration.profile, seed=5)
    paths = save_outputs(dataset, calibration, tmp_path / "out")

    events = pd.read_csv(paths["telemetry"])
    assert list(events.columns) == EVENT_COLUMNS
    assert len(events) == len(dataset.events)
    nodes = pd.read_csv(paths["nodes"])
    assert len(nodes) == dataset.graph.number_of_nodes()
    assert len(pd.read_csv(paths["edges"])) == dataset.graph.number_of_edges()

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["random_seed"] == 5
    assert manifest["num_vms"] == SMALL.n_vms
    assert manifest["total_events"] == len(events)
    assert sum(manifest["label_counts"].values()) == len(events)
    assert manifest["generator"]["coincidental_vms"] == [3, 6]
    assert manifest["calibration"]["fallbacks"] == []
    assert manifest["features"] == list(calibration.profile)
    assert sum(manifest["pattern_counts"].values()) == len(dataset.patterns)
    saved = pd.read_csv(paths["patterns"])
    assert dict(zip(saved["episode_id"], saved["pattern"], strict=True)) == dataset.patterns
