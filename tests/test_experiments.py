import math
from dataclasses import fields

import pandas as pd
import pytest
import yaml
from scipy import stats

from cadgnn.cli import main
from cadgnn.config import ConfigError, GraphConfig, ModelConfig, TrainingConfig, parse_config
from cadgnn.experiments import (
    PARAMETERS,
    load_spec,
    parameter_rows,
    parse_spec,
    render_parameter_table,
    render_summary,
    run_experiment,
    scenario_config,
    summarise,
    variant_config,
)
from cadgnn.experiments.summary import paired_p_value
from conftest import REPO_ROOT

TINY = {
    "generator": {"n_vms": 60, "n_storage": 4, "n_network": 5,
                  "n_coincidental_episodes": 10, "n_coordinated_episodes": 8},
    "model": {"hidden_dim": 8, "temporal_dim": 8, "spatial_dim": 8, "fusion_dim": 8,
              "dropout": 0.0, "device": "cpu"},
    "training": {"epochs": 2, "batch_size": 4},
}  # fmt: skip


def base_config(tmp_path, datasets=None):
    raw = {**TINY, "paths": {"output_dir": str(tmp_path / "out")}}
    if datasets is not None:
        raw["paths"]["datasets_dir"] = str(datasets)
    return parse_config(raw, base_dir=tmp_path)


def test_repository_spec_is_valid(tmp_path):
    spec = load_spec(REPO_ROOT / "configs" / "experiments.yaml", base_config(tmp_path))
    names = [v.name for v in spec.variants]
    assert {"gated", "concat", "temporal_only", "spatial_only", "continuum_style"} <= set(names)
    assert spec.reference == "gated"
    assert set(spec.baselines) == {"rule_based", "logistic_regression", "random_forest"}
    assert len(spec.seeds) >= 5


def test_defaults_when_sections_are_missing(tmp_path):
    spec = parse_spec({}, base_config(tmp_path))
    assert spec.seeds == [8304]
    assert [s.name for s in spec.scenarios] == ["default"]
    assert [v.name for v in spec.variants] == ["default"]


def test_scenario_overrides_and_randomisation(tmp_path):
    base = base_config(tmp_path)
    spec = parse_spec(
        {
            "seeds": [1, 2],
            "scenarios": {
                "slow": {"generator": {"hop_delay_minutes": [60, 120]}},
                "random": {"randomize": {"n_network": [2, 40], "observation_hours": [48, 96]}},
            },
        },
        base,
    )
    slow, random = spec.scenarios
    assert scenario_config(base.generator, slow, 1).hop_delay_minutes == (60, 120)
    a = scenario_config(base.generator, random, 1)
    assert 2 <= a.n_network <= 40 and 48 <= a.observation_hours <= 96
    assert scenario_config(base.generator, random, 1) == a
    draws = {scenario_config(base.generator, random, s).n_network for s in range(10)}
    assert len(draws) > 1
    # Settings that are not randomised keep the base value.
    assert a.n_vms == base.generator.n_vms


def test_variant_config(tmp_path):
    base = base_config(tmp_path)
    spec = parse_spec({"variants": {"t": {"model": {"use_spatial": False}}}}, base)
    config = variant_config(base, spec.variants[0], seed=9)
    assert config.seed == 9
    assert config.model.use_spatial is False
    assert config.model.hidden_dim == 8
    assert config.training == base.training


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"seeds": []}, "non-empty list"),
        ({"seeds": [1, 1]}, "must not repeat"),
        ({"seeds": [-1]}, "non-negative"),
        ({"repeats": 3}, "unknown key"),
        ({"variants": {}}, "non-empty mapping"),
        ({"variants": {"bad name": {}}}, "letters, digits"),
        ({"variants": {"v": {"optimizer": {}}}}, "unknown key"),
        ({"variants": {"v": {"model": {"fusion": "sum"}}}}, "fusion must be one of"),
        ({"variants": {"v": {"model": {"layers": 3}}}}, "variant v"),
        ({"scenarios": {"s": {"generator": {"n_vms": 0}}}}, "n_vms"),
        ({"scenarios": {"s": {"randomize": {"hop_delay_minutes": [1, 2]}}}}, "cannot be random"),
        ({"scenarios": {"s": {"randomize": {"n_network": [9, 3]}}}}, r"\[low, high\]"),
    ],
)  # fmt: skip
def test_invalid_specs_fail_before_running(tmp_path, raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_spec(raw, base_config(tmp_path))


def test_missing_spec_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_spec(tmp_path / "nope.yaml", base_config(tmp_path))


def fake_results():
    rows = []
    for seed in (1, 2):
        for variant, val_f1, test_f1 in (("a", 0.8, 0.9), ("b", 0.9, 0.7)):
            rows.append({
                "scenario": "s", "seed": seed, "variant": variant, "val_f1": val_f1,
                "best_val_loss": 0.3, "test_f1": test_f1 + 0.02 * seed, "test_precision": 1.0,
                "test_recall": 1.0, "test_fpr": 0.0, "test_fnr": 0.0, "test_roc_auc": 1.0,
                "test_pr_auc": 1.0, "detection_rate": 1.0, "mean_hours_to_detect": None,
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_summary_selects_on_validation_not_test():
    summary = summarise(fake_results())
    selected = summary.loc[summary["selected"], "variant"].tolist()
    # Variant a has the better test F1, but b has the better validation F1.
    assert selected == ["b"]
    a = summary[summary["variant"] == "a"].iloc[0]
    assert a["runs"] == 2
    assert a["test_f1_mean"] == pytest.approx(0.93)
    assert a["test_f1_std"] == pytest.approx(pd.Series([0.92, 0.94]).std())


def test_summary_report(tmp_path):
    spec = parse_spec({"seeds": [1, 2]}, base_config(tmp_path))
    text = render_summary(summarise(fake_results()), spec)
    assert "## s" in text
    assert "| b * |" in text
    assert "n/a" in text
    assert "do not measure performance on real attack campaigns" in text
    with pytest.raises(ValueError, match="no experiment results"):
        summarise(pd.DataFrame())


def test_spec_baselines_and_reference(tmp_path):
    base = base_config(tmp_path)
    spec = parse_spec(
        {"variants": {"gated": {}, "concat": {"model": {"fusion": "concat"}}},
         "baselines": ["rule_based", "random_forest"]},
        base,
    )  # fmt: skip
    assert spec.baselines == ["rule_based", "random_forest"]
    assert spec.reference == "gated"
    assert parse_spec({}, base).baselines == []


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"baselines": ["svm"]}, "unknown baseline"),
        ({"baselines": "rule_based"}, "list of names"),
        ({"baselines": ["rule_based", "rule_based"]}, "without repeats"),
        ({"variants": {"rule_based": {}}, "baselines": ["rule_based"]}, "taken by baselines"),
        ({"variants": {"v": {"model": {"architecture": "rnn"}}}}, "architecture"),
    ],
)  # fmt: skip
def test_invalid_baselines(tmp_path, raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_spec(raw, base_config(tmp_path))


def test_paired_p_value():
    a = pd.Series([0.80, 0.82, 0.85, 0.79, 0.81], index=[1, 2, 3, 4, 5])
    b = pd.Series([0.70, 0.75, 0.72, 0.71, 0.74], index=[1, 2, 3, 4, 5])
    assert paired_p_value(a, b) == pytest.approx(stats.ttest_rel(a, b).pvalue)
    assert paired_p_value(a, a) == 1.0
    assert math.isnan(paired_p_value(a, a - 0.1))
    assert math.isnan(paired_p_value(a.iloc[:1], b.iloc[:1]))
    # Only seeds both models ran on are paired.
    assert paired_p_value(a, b.iloc[:2]) == pytest.approx(stats.ttest_rel(a[:2], b[:2]).pvalue)


def test_summary_tests_against_the_reference_and_lists_patterns(tmp_path):
    results = fake_results()
    results["kind"] = "gnn"
    # Rows are seed 1 a, seed 1 b, seed 2 a, seed 2 b.
    results["rate_lookalike"] = [0.5, 0.25, 0.25, 0.25]
    summary = summarise(results, reference="a")
    a = summary[summary["variant"] == "a"].iloc[0]
    b = summary[summary["variant"] == "b"].iloc[0]
    assert math.isnan(a["p_test_f1"])
    assert math.isnan(b["p_test_f1"])  # b is always 0.2 below a: a constant difference
    assert b["p_test_roc_auc"] == 1.0
    assert a["rate_lookalike_mean"] == pytest.approx(0.375)
    spec = parse_spec({"seeds": [1, 2]}, base_config(tmp_path))
    text = render_summary(summary, spec)
    assert "paired t-test over seeds against `a`" in text
    assert "| Model | lookalike |" in text


def test_run_experiment_end_to_end(tmp_path, fake_datasets):
    base = base_config(tmp_path, fake_datasets)
    spec = parse_spec(
        {"seeds": [1], "scenarios": {"plain": {}},
         "variants": {"gated": {}, "temporal_only": {"model": {"use_spatial": False}},
                      "continuum_style": {"model": {"architecture": "chained"}}},
         "baselines": ["rule_based", "logistic_regression"]},
        base,
    )  # fmt: skip
    out = tmp_path / "exp"
    experiment = run_experiment(base, spec, out)
    results = experiment["results"]
    assert list(results["variant"]) == [
        "gated", "temporal_only", "continuum_style", "rule_based", "logistic_regression"
    ]  # fmt: skip
    assert list(results["kind"]) == ["gnn"] * 3 + ["baseline"] * 2
    assert results["test_f1"].notna().all()
    assert results["rate_scattered"].notna().all()
    # Every model is tested on the same episodes.
    assert results["n_episodes"].nunique() == 1
    assert (out / "plain" / "seed_1" / "baselines" / "metrics.json").is_file()
    for variant in ("gated", "temporal_only", "continuum_style"):
        run_dir = out / "plain" / "seed_1" / variant
        assert (run_dir / "model" / "coordination_model.pt").is_file()
        assert (run_dir / "evaluation" / "decision_report.md").is_file()
    # Both variants train on the same generated graphs.
    assert (out / "plain" / "seed_1" / "episode_graphs.pt").is_file()
    assert len(set(results["n_episodes"])) == 1
    for name in ("results.csv", "summary.csv", "summary.md"):
        assert (out / name).is_file()


def test_cli_experiment(tmp_path, fake_datasets, capsys):
    config = tmp_path / "base.yaml"
    raw = {**TINY, "paths": {"datasets_dir": str(fake_datasets), "output_dir": str(tmp_path)}}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"seeds": [1], "variants": {"gated": {}}}), encoding="utf-8")
    assert main(["--config", str(config), "experiment", "--spec", str(spec)]) == 0
    assert "selected on validation" in capsys.readouterr().out
    assert (tmp_path / "experiments" / "summary.md").is_file()

    spec.write_text(yaml.safe_dump({"seeds": "1"}), encoding="utf-8")
    assert main(["--config", str(config), "experiment", "--spec", str(spec)]) == 2


def test_every_tunable_setting_is_in_the_parameter_table():
    for section, cls in (("graph", GraphConfig), ("model", ModelConfig),
                         ("training", TrainingConfig)):  # fmt: skip
        for f in fields(cls):
            assert (section, f.name) in PARAMETERS, f"{section}.{f.name} missing"


def test_parameter_table_shows_current_values(tmp_path):
    config = base_config(tmp_path)
    rows = {r["setting"]: r for r in parameter_rows(config)}
    assert rows["model.hidden_dim"]["default"] == 64
    assert rows["model.hidden_dim"]["current"] == 8
    table = render_parameter_table(config)
    assert table.startswith("| Setting |")
    assert "`training.threshold_strategy`" in table


def test_cli_params(capsys):
    assert main(["--config", str(REPO_ROOT / "configs" / "default.yaml"), "params"]) == 0
    assert "`model.fusion`" in capsys.readouterr().out
