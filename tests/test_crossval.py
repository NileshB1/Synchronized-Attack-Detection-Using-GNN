import json

import numpy as np
import pandas as pd
import pytest

from cadgnn.config import ConfigError, parse_config
from cadgnn.crossval import (
    load_crossval_spec,
    mcnemar,
    paired_t_p,
    parse_crossval_spec,
    results_frame,
    run_crossval,
    stratified_folds,
    wilcoxon_p,
    write_reports,
)
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE, save_episode_graphs
from cadgnn.training import stratified_split
from conftest import REPO_ROOT

SMALL = {
    "model": {"hidden_dim": 8, "temporal_dim": 8, "spatial_dim": 8, "fusion_dim": 8,
              "dropout": 0.0, "device": "cpu"},
    "training": {"epochs": 2, "batch_size": 8, "patience": 5, "test_fraction": 0.1},
}  # fmt: skip
SPEC = {
    "folds": 2,
    "variants": {"gated": {}, "concat": {"model": {"fusion": "concat"}}},
    "baselines": ["rule_based", "logistic_regression"],
    "balancing": ["class_weight", "oversample", "smote"],
    "tuning": {"training": {"learning_rate": [0.001, 0.01]}},
}


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(
        infra, clean_events(small_dataset.events, infra), patterns=small_dataset.patterns
    )


def config_for(tmp_path):
    return parse_config({"seed": 3, "paths": {"output_dir": str(tmp_path)}, **SMALL},
                        base_dir=tmp_path)  # fmt: skip


@pytest.fixture(scope="module")
def study(tmp_path_factory, graphs):
    out = tmp_path_factory.mktemp("crossval")
    save_episode_graphs(graphs, out / EPISODE_GRAPHS_FILE)
    config = config_for(out)
    spec = parse_crossval_spec(SPEC, config)
    result = run_crossval(config, spec, out)
    return out, spec, result


def test_folds_cover_the_training_part_once(graphs):
    split = stratified_split(graphs, 0.2, 0.1, 3)
    folds = stratified_folds(graphs, split.train, 2, 3)
    held = sorted(i for _, h in folds for i in h)
    assert held == sorted(split.train)
    for train, h in folds:
        assert not set(train) & set(h)
        assert set(train) | set(h) == set(split.train)
        assert {int(graphs[i].y) for i in h} == {0, 1}
    with pytest.raises(ValueError, match="at least 50"):
        stratified_folds(graphs, split.train, 50, 3)


def test_spec_from_the_repo_parses():
    config = parse_config({}, base_dir=REPO_ROOT)
    spec = load_crossval_spec(REPO_ROOT / "configs" / "crossval.yaml", config)
    assert spec.reference == "gated"
    assert spec.folds == 5
    assert len(spec.tuning_grid()) == 2


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"folds": 1}, "at least 2"),
        ({"balancing": ["tomek"]}, "balancing must be"),
        ({"balancing": ["smote"]}, "GNN can use"),
        ({"baselines": ["svm"]}, "baselines must be"),
        ({"tuning": {"loss": {"x": [1]}}}, "tuning must map"),
        ({"tuning": {"training": {"learning_rate": []}}}, "non-empty list"),
        ({"tuning": {"training": {"epochs": [0]}}}, "epochs"),
        ({"false_positives": {"cost": 2}}, "false_positives takes"),
        ({"alpha": 2}, "alpha"),
        ({"surprise": 1}, "unknown key"),
    ],
)
def test_bad_specs(raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_crossval_spec(raw, parse_config({}, base_dir=REPO_ROOT))


def test_tuning_grid_combines_every_value():
    spec = parse_crossval_spec(
        {"tuning": {"model": {"dropout": [0.0, 0.2]}, "training": {"learning_rate": [1, 2]}}},
        parse_config({}, base_dir=REPO_ROOT),
    )
    grid = spec.tuning_grid()
    assert len(grid) == 4
    assert {"model": {"dropout": 0.2}, "training": {"learning_rate": 2}} in grid


def test_paired_tests():
    a, b = [0.8, 0.7, 0.9, 0.85, 0.75], [0.6, 0.65, 0.7, 0.6, 0.62]
    assert paired_t_p(a, b) < 0.05
    assert paired_t_p(a, a) == 1.0
    assert np.isnan(paired_t_p([1, 2], [0, 1]))
    assert wilcoxon_p(a, b) == pytest.approx(0.0625)
    assert wilcoxon_p(a, a) == 1.0


def test_mcnemar_counts_only_disagreements():
    y = np.array([1, 1, 0, 0, 1, 0])
    ours = np.array([1, 1, 0, 0, 1, 1])
    theirs = np.array([0, 0, 0, 1, 1, 1])
    found = mcnemar(y, ours, theirs)
    assert (found["only_a"], found["only_b"]) == (3, 0)
    assert found["p"] == pytest.approx(0.25)
    assert mcnemar(y, ours, ours)["p"] == 1.0


def test_study_writes_splits_and_tables(study, graphs):
    out, spec, result = study
    holdout = json.loads((out / "splits" / "holdout.json").read_text(encoding="utf-8"))
    folds = json.loads((out / "splits" / "folds.json").read_text(encoding="utf-8"))
    assert sum(holdout["sizes"].values()) == len(graphs)
    held = [e for f in folds for e in f["held_out"]]
    assert sorted(held) == sorted(holdout["episode_ids"]["train"])
    assert not set(held) & set(holdout["episode_ids"]["test"])
    for name in ("folds", "test", "significance", "predictions", "class_counts"):
        assert result.paths[name].is_file()


def test_study_steps(study):
    _, spec, result = study
    folds = result.folds
    assert set(folds["study"]) == {"balancing", "cv", "fp"}
    balancing = folds[folds["study"] == "balancing"]
    assert "smote" not in set(balancing.loc[balancing["kind"] == "gnn", "balancing"])
    assert "smote" in set(balancing.loc[balancing["model"] == "logistic_regression", "balancing"])
    assert result.balancing in ("class_weight", "oversample")
    cv = folds[folds["study"] == "cv"]
    assert set(cv["model"]) == {"gated", "concat", "rule_based", "logistic_regression"}
    assert (cv.groupby("model").size() == spec.folds).all()
    assert set(cv.loc[cv["model"] == "rule_based", "balancing"]) == {"n/a"}
    fp = folds[folds["study"] == "fp"]
    assert len(set(fp["strategy"])) == 6
    assert result.fp_strategy in set(fp["strategy"])
    assert len(result.tuning) == 2 and result.tuning["chosen"].sum() == 1
    assert len(result.histories) == spec.folds
    assert {"train_acc", "val_acc"} <= set(result.histories[0][0])


def test_test_part_is_scored_once_per_model(study):
    _, _, result = study
    test = result.test
    assert test["model"].is_unique
    n_test = len(result.predictions)
    assert (test[["tp", "fp", "tn", "fn"]].sum(axis=1) == n_test).all()
    assert f"gated + {result.fp_strategy}" in set(test["model"])
    assert "gated + default" in set(test["model"])
    plain = test.set_index("model").loc[["gated", "gated + default"]]
    assert plain["fp"].nunique() == 1


def test_results_table(study, tmp_path):
    out, spec, result = study
    frame = results_frame(result)
    assert frame["model"].iloc[0] == "gated"
    assert pd.isna(frame["p_value_vs_ours"].iloc[0])
    others = frame[~frame["model"].str.startswith("gated")]
    sig = result.significance.set_index("model")
    for _, row in others.iterrows():
        expected = sig.loc[row["model"], "ours_f1"] > sig.loc[row["model"], "theirs_f1"] and (
            sig.loc[row["model"], "t_test_p_f1"] < spec.alpha
        )
        assert row["ours_beats_it"] == expected

    paths = write_reports(result, spec, tmp_path)
    text = paths["results_table"].read_text(encoding="utf-8")
    header = text.splitlines()[next(i for i, line in enumerate(text.splitlines())
                                    if line.startswith("| Model"))]  # fmt: skip
    assert "Beats baseline? (Y/N)" in header and "CV Acc (mean±std)" in header
    assert "**" in text
    for name in ("imbalance", "cross_validation", "false_positives", "novelty"):
        assert paths[name].read_text(encoding="utf-8").startswith("# ")
    assert len(pd.read_csv(paths["results_csv"])) == len(frame)


def test_figures(study, tmp_path):
    from cadgnn.crossval.figures import write_figures

    _, _, result = study
    paths = write_figures(result, tmp_path)
    assert set(paths) == {"class_distribution", "train_val_curves", "learning_curves",
                          "pr_curve", "confusion_matrices", "false_positive_rates"}  # fmt: skip
    for path in paths.values():
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_crossval(tmp_path, graphs, capsys):
    import yaml

    from cadgnn.cli import main

    save_episode_graphs(graphs, tmp_path / EPISODE_GRAPHS_FILE)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"seed": 3, "paths": {"output_dir": "."}, **SMALL}),
                      encoding="utf-8")  # fmt: skip
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({**SPEC, "tuning": {}, "balancing": ["class_weight"]}),
                    encoding="utf-8")  # fmt: skip
    assert main(["--config", str(config), "crossval", "--spec", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "balancing chosen on the folds: class_weight" in out
    assert (tmp_path / "reports" / "results_table.md").is_file()
    assert (tmp_path / "figures" / "pr_curve.png").is_file()
    bad = tmp_path / "bad.yaml"
    bad.write_text("folds: 1\n", encoding="utf-8")
    assert main(["--config", str(config), "crossval", "--spec", str(bad)]) == 2
