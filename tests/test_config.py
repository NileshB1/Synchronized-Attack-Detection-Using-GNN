from pathlib import Path

import pytest

from cadgnn.config import (
    DEFAULTS, ConfigError,
    GeneratorConfig, describe, display_path, load_config,
    parse_config, with_overrides
)
from conftest import REPO_ROOT


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_resolve_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.seed == DEFAULTS["seed"]
    assert config.log_level == "INFO"
    assert config.paths.datasets_dir == (tmp_path / "datasets").resolve()


def test_repo_default_config_points_at_repo_datasets():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.paths.datasets_dir == (REPO_ROOT / "datasets").resolve()
    assert config.paths.output_dir == (REPO_ROOT / "output").resolve()


def test_relative_paths_resolve_against_config_file(tmp_path):
    path = write(tmp_path, "paths:\n  datasets_dir: data/raw\n")
    config = load_config(path)
    assert config.paths.datasets_dir == (tmp_path / "data" / "raw").resolve()
    # Unspecified keys keep their defaults.
    assert config.paths.output_dir == (tmp_path / "output").resolve()


def test_empty_file_gives_defaults(tmp_path):
    assert load_config(write(tmp_path, "")).seed == DEFAULTS["seed"]


def test_log_level_is_case_insensitive(tmp_path):
    assert load_config(write(tmp_path, "log_level: debug\n")).log_level == "DEBUG"


@pytest.mark.parametrize(
    "text, message",
    [
        ("seed: -1\n", "non-negative"),  ("seed: 1.5\n", "integer"),
        ("seed: true\n", "integer"),
        ("seed: '42'\n", "integer"),  ("log_level: LOUD\n", "log_level"),
        ("paths: nowhere\n", "mapping"),
        ("paths:\n  datasets_dir: ''\n", "non-empty path"),
        ("paths:\n  datasets_dir: 3\n", "non-empty path"),
        ("sede: 1\n", "unknown key"), ("paths:\n  dataset_dir: x\n", "unknown key"),
        ("- a\n- b\n", "mapping"),
        ("seed: [1\n", "could not parse"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = with_overrides(
        parse_config({}, base_dir=Path("/elsewhere")), seed=7, datasets_dir="d", log_level="warning"
    )
    assert config.seed == 7
    assert config.log_level == "WARNING"
    assert config.paths.datasets_dir == (tmp_path / "d").resolve()


def test_overrides_are_validated():
    config = parse_config({}, base_dir=Path.cwd())
    with pytest.raises(ConfigError):
        with_overrides(config, seed=-5)


def test_generator_defaults_match_reference_notebook():
    g = GeneratorConfig()
    assert (g.n_vms, g.n_storage, g.n_network) == (12000, 300, 200)
    assert g.observation_hours == 72
    assert g.benign_backups_per_vm == 6
    assert (g.n_coincidental_episodes, g.coincidental_vms) == (100, (3, 6))
    assert (g.n_coordinated_episodes, g.coordinated_chain_length) == (50, (4, 8))
    assert g.hop_delay_minutes == (5, 45)
    assert g.campaign_start_margin_hours == 2


def test_generator_overlap_settings_are_off_by_default():
    g = GeneratorConfig()
    assert g.coincidental_burst_fraction == 0
    assert g.coincidental_cluster_fraction == 0
    assert g.coincidental_lookalike_fraction == 0
    assert g.stealth_campaign_fraction == 0
    assert g.storage_pivot_probability == 0
    assert g.missed_event_probability == 0
    assert g.stray_vm_probability == 0


def test_repo_default_config_turns_on_class_overlap():
    g = load_config(REPO_ROOT / "configs" / "default.yaml").generator
    assert (g.n_coincidental_episodes, g.n_coordinated_episodes) == (1000, 500)
    assert g.coincidental_lookalike_fraction > 0
    assert g.stealth_campaign_fraction > 0
    assert (g.n_vms, g.n_storage, g.n_network) == (12000, 300, 200)


def test_generator_section_overrides_and_lists_become_tuples(tmp_path):
    config = load_config(
        write(tmp_path, "generator:\n  n_vms: 50\n  coordinated_chain_length: [2, 3]\n")
    )
    assert config.generator.n_vms == 50
    assert config.generator.coordinated_chain_length == (2, 3)
    assert config.generator.n_storage == 300


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_vms": 0}, "n_vms must be at least 1"),
        ({"n_storage": 2.5}, "n_storage must be an integer"),
        ({"n_network": True}, "n_network must be an integer"),
        ({"benign_backups_per_vm": -1}, "at least 0"),
        ({"observation_hours": 0}, "positive"),
        ({"observation_hours": "72"}, "number"),
        ({"campaign_start_margin_hours": 72}, r"\[0, observation_hours\)"),
        ({"coincidental_vms": (5, 3)}, "greater than high"),
        ({"coincidental_vms": (0, 3)}, "at least 1"),
        ({"coincidental_vms": [3]}, "pair"),
        ({"coordinated_chain_length": (1.5, 3)}, "integer"),
        ({"hop_delay_minutes": (-1, 5)}, "at least 0"),
        ({"n_vms": 4, "coincidental_vms": (3, 6)}, "exceeds n_vms"),
        ({"stealth_campaign_fraction": 1.5}, r"must be in \[0, 1\]"),
        ({"missed_event_probability": -0.1}, r"must be in \[0, 1\]"),
        ({"stray_vm_probability": "high"}, "number"),
        (
            {"coincidental_burst_fraction": 0.6, "coincidental_cluster_fraction": 0.6},
            "add up to",
        ),
        ({"coincidental_window_hours": (5, 1)}, "greater than high"),
        ({"coincidental_window_hours": (1, 100)}, "inside observation_hours"),
        ({"stealth_hop_delay_minutes": (-5, 60)}, "at least 0"),
    ],
)
def test_invalid_generator_config(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        GeneratorConfig(**kwargs)


def test_small_scenario_without_coincidental_episodes_is_valid():
    # The VM-count check only applies when coincidental episodes are generated.
    assert GeneratorConfig(n_vms=2, n_coincidental_episodes=0).n_vms == 2


def test_unknown_generator_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "generator:\n  n_vm: 10\n"))


def test_graph_section(tmp_path):
    assert load_config(REPO_ROOT / "configs" / "default.yaml").graph.min_episode_events == 2
    config = load_config(write(tmp_path, "graph:\n  min_episode_events: 3\n"))
    assert config.graph.min_episode_events == 3
    with pytest.raises(ConfigError, match="min_episode_events"):
        load_config(write(tmp_path, "graph:\n  min_episode_events: 0\n"))


def test_model_section(tmp_path):
    model = load_config(REPO_ROOT / "configs" / "default.yaml").model
    assert (model.hidden_dim, model.temporal_dim, model.spatial_dim) == (64, 64, 64)
    assert model.device == "auto"
    config = load_config(write(tmp_path, "model:\n  hidden_dim: 32\n  device: cpu\n"))
    assert (config.model.hidden_dim, config.model.device) == (32, "cpu")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "model:\n  layers: 3\n"))


def test_training_section(tmp_path):
    training = load_config(REPO_ROOT / "configs" / "default.yaml").training
    assert (training.val_fraction, training.test_fraction) == (0.2, 0.1)
    assert training.threshold_strategy == "f1"
    config = load_config(write(tmp_path, "training:\n  epochs: 5\n  threshold_strategy: fpr\n"))
    assert (config.training.epochs, config.training.threshold_strategy) == (5, "fpr")


@pytest.mark.parametrize(
    "text, message",
    [
        ("training:\n  val_fraction: 0.6\n  test_fraction: 0.5\n", "leave some training"),
        ("training:\n  test_fraction: 0\n", "between 0 and 1"),
        ("training:\n  learning_rate: 0\n", "learning_rate"),
        ("training:\n  max_fpr: 1.5\n", "max_fpr"),
        ("training:\n  threshold_strategy: youden\n", "threshold_strategy"),
        ("training:\n  epochs: 0\n", "epochs"),
    ],
)
def test_invalid_training_section(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))


def test_display_path_hides_the_local_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert display_path(tmp_path / "output" / "model.pt") == str(Path("output") / "model.pt")
    outside = tmp_path.parent / "elsewhere"
    assert display_path(outside) == str(outside)
    config = parse_config({}, base_dir=tmp_path)
    assert str(tmp_path) not in describe(config)


@pytest.mark.parametrize(
    "text, message",
    [
        ("training:\n  balancing: smote\n", "balancing must be one of"),
        ("training:\n  fp_cost: 0\n", "fp_cost must be positive"),
        ("training:\n  min_precision: 1.5\n", "min_precision"),
    ],
)
def test_imbalance_and_false_alarm_settings(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))


def test_precision_threshold_strategy_is_accepted(tmp_path):
    text = "training:\n  threshold_strategy: precision\n  min_precision: 0.8\n"
    training = load_config(write(tmp_path, text)).training
    assert (training.threshold_strategy, training.min_precision) == ("precision", 0.8)
    assert (training.balancing, training.fp_cost) == ("class_weight", 1.0)
