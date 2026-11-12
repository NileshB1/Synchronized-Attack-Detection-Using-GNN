import json

import pytest

from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.training import save_split, split_from_ids, stratified_split


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


def labels(graphs, idx):
    return [int(graphs[i].y) for i in idx]


def test_parts_are_disjoint_and_cover_everything(graphs):
    split = stratified_split(graphs, 0.2, 0.2, seed=1)
    parts = [set(split.train), set(split.val), set(split.test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set.union(*parts) == set(range(len(graphs)))


def test_every_part_has_both_labels(graphs):
    split = stratified_split(graphs, 0.2, 0.2, seed=1)
    for part in (split.train, split.val, split.test):
        assert set(labels(graphs, part)) == {0, 1}


def test_class_ratio_is_kept(graphs):
    # 10 coincidental and 8 coordinated episodes in the fixture.
    split = stratified_split(graphs, 0.2, 0.2, seed=3)
    assert labels(graphs, split.test).count(0) == 2
    assert labels(graphs, split.test).count(1) == 2
    assert len(split.train) == 10


def test_same_seed_same_split(graphs):
    assert stratified_split(graphs, 0.2, 0.2, 7) == stratified_split(graphs, 0.2, 0.2, 7)
    assert stratified_split(graphs, 0.2, 0.2, 7) != stratified_split(graphs, 0.2, 0.2, 8)


def test_split_does_not_follow_storage_order(graphs):
    # Graphs are sorted by episode id, so a split that just took the first rows would
    # put every coincidental episode in training.
    split = stratified_split(graphs, 0.2, 0.2, seed=1)
    assert split.train != list(range(len(split.train)))


def test_too_few_episodes_of_a_label(graphs):
    few = [g for g in graphs if int(g.y) == 0] + [g for g in graphs if int(g.y) == 1][:2]
    with pytest.raises(ValueError, match="at least 3 episodes with label 1"):
        stratified_split(few, 0.2, 0.2, seed=0)


def test_saved_split_round_trip(graphs):
    split = stratified_split(graphs, 0.2, 0.2, seed=2)
    ids = split.episode_ids(graphs)
    assert split_from_ids(graphs, ids) == split
    # Same episodes come back even if the graphs are stored in another order.
    reordered = list(reversed(graphs))
    again = split_from_ids(reordered, ids).episode_ids(reordered)
    assert {k: sorted(v) for k, v in again.items()} == {k: sorted(v) for k, v in ids.items()}


def test_saved_split_with_missing_episodes(graphs):
    ids = stratified_split(graphs, 0.2, 0.2, seed=2).episode_ids(graphs)
    ids["test"].append("coordinated_999")
    with pytest.raises(ValueError, match="rebuilt after training"):
        split_from_ids(graphs, ids)


def test_saved_split_file(tmp_path, graphs):
    split = stratified_split(graphs, 0.2, 0.1, seed=2)
    path = save_split(split, graphs, tmp_path / "splits" / "split.json", seed=2)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["seed"] == 2
    assert saved["indices"]["test"] == split.test
    assert split_from_ids(graphs, saved["episode_ids"]) == split
    assert sum(saved["sizes"].values()) == len(graphs)
