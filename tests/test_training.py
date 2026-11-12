import json
import os
import subprocess
import sys

import pytest
import torch

from cadgnn.cli import main
from cadgnn.config import ModelConfig, TrainingConfig, parse_config
from cadgnn.data.calibration import calibrate
from cadgnn.evaluation.run import run_evaluation
from cadgnn.fusion import build_coordination_model
from cadgnn.generator import save_outputs
from cadgnn.graph import (
    InfrastructureGraph,
    build_episode_graphs,
    clean_events,
    save_episode_graphs,
)
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE
from cadgnn.models import ModelInputError, seed_everything
from cadgnn.training import CHECKPOINT_FILE, run_training, stratified_split, train_model
from conftest import REPO_ROOT

SMALL_MODEL = {"hidden_dim": 16, "temporal_dim": 8, "spatial_dim": 8, "fusion_dim": 8,
               "dropout": 0.0, "device": "cpu"}  # fmt: skip
SMALL_TRAINING = {"epochs": 4, "batch_size": 4, "patience": 10}


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


@pytest.fixture
def output_dir(tmp_path, small_dataset, graphs, fake_datasets):
    out = tmp_path / "out"
    save_outputs(small_dataset, calibrate(fake_datasets), out)
    save_episode_graphs(graphs, out / EPISODE_GRAPHS_FILE)
    return out


def make_config(output_dir, seed=5, **training):
    raw = {
        "seed": seed,
        "paths": {"output_dir": str(output_dir)},
        "model": SMALL_MODEL,
        "training": {**SMALL_TRAINING, **training},
    }
    return parse_config(raw, base_dir=output_dir)


def split_parts(graphs, seed=1):
    split = stratified_split(graphs, 0.2, 0.2, seed)
    return [graphs[i] for i in split.train], [graphs[i] for i in split.val]


def test_train_model_records_history_and_restores_best(graphs):
    seed_everything(0)
    model = build_coordination_model(ModelConfig(**SMALL_MODEL))
    train, val = split_parts(graphs)
    result = train_model(model, train, val, TrainingConfig(**SMALL_TRAINING), seed=0)
    assert [h["epoch"] for h in result.history] == [1, 2, 3, 4]
    best = min(result.history, key=lambda h: h["val_loss"])
    assert result.best_epoch == best["epoch"]
    assert result.best_val_loss == pytest.approx(best["val_loss"])


def test_scaler_and_class_weight_come_from_training_only(graphs):
    model = build_coordination_model(ModelConfig(**SMALL_MODEL))
    train, val = split_parts(graphs)
    result = train_model(model, train, val, TrainingConfig(**SMALL_TRAINING), seed=0)
    train_x = torch.cat([g["vm"].x for g in train])
    assert torch.allclose(model.streams.scaler.mean, train_x.mean(0))
    n_pos = sum(int(g.y) for g in train)
    assert result.pos_weight == pytest.approx((len(train) - n_pos) / n_pos)


def test_history_has_accuracy(graphs):
    model = build_coordination_model(ModelConfig(**SMALL_MODEL))
    train, val = split_parts(graphs)
    result = train_model(model, train, val, TrainingConfig(**SMALL_TRAINING), seed=0)
    for row in result.history:
        assert 0 <= row["train_acc"] <= 1
        assert 0 <= row["val_acc"] <= 1


@pytest.mark.parametrize("balancing", ["none", "oversample", "undersample"])
def test_resampling_drops_the_class_weight(graphs, balancing):
    model = build_coordination_model(ModelConfig(**SMALL_MODEL))
    train, val = split_parts(graphs)
    config = TrainingConfig(**SMALL_TRAINING, balancing=balancing)
    result = train_model(model, train, val, config, seed=0)
    assert result.pos_weight == 1.0
    assert len(result.history) == SMALL_TRAINING["epochs"]


def test_false_alarm_cost_changes_training(graphs):
    train, val = split_parts(graphs)
    losses = []
    for cost in (1.0, 4.0):
        seed_everything(0)
        model = build_coordination_model(ModelConfig(**SMALL_MODEL))
        config = TrainingConfig(**SMALL_TRAINING, fp_cost=cost)
        losses.append(train_model(model, train, val, config, seed=0).history[0]["train_loss"])
    assert losses[1] > losses[0]


def test_early_stopping(graphs):
    model = build_coordination_model(ModelConfig(**SMALL_MODEL))
    train, val = split_parts(graphs)
    config = TrainingConfig(epochs=200, batch_size=4, patience=1, learning_rate=0.5)
    result = train_model(model, train, val, config, seed=0)
    assert len(result.history) < 200
    assert len(result.history) - result.best_epoch == 1


def test_empty_sets_are_rejected(graphs):
    model = build_coordination_model(ModelConfig(**SMALL_MODEL))
    with pytest.raises(ValueError, match="must not be empty"):
        train_model(model, graphs, [], TrainingConfig(**SMALL_TRAINING), seed=0)


def test_training_saves_split_threshold_and_history(output_dir):
    summary = run_training(make_config(output_dir))
    checkpoint = torch.load(output_dir / CHECKPOINT_FILE, weights_only=True)
    split = checkpoint["split"]
    assert not set(split["test"]) & (set(split["train"]) | set(split["val"]))
    assert checkpoint["threshold"] == summary["threshold"]
    assert 0 <= checkpoint["threshold"] <= 1
    assert len(checkpoint["history"]) == 4
    assert checkpoint["training_config"]["threshold_strategy"] == "f1"
    assert checkpoint["val_metrics"]["n_episodes"] == len(split["val"])
    saved = json.loads(summary["split_file"].read_text(encoding="utf-8"))
    assert saved["episode_ids"] == split


def test_training_is_reproducible(output_dir):
    first = run_training(make_config(output_dir))
    a = torch.load(output_dir / CHECKPOINT_FILE, weights_only=True)
    second = run_training(make_config(output_dir))
    b = torch.load(output_dir / CHECKPOINT_FILE, weights_only=True)
    assert first["threshold"] == second["threshold"]
    assert a["split"] == b["split"]
    for key in a["state_dict"]:
        assert torch.equal(a["state_dict"][key], b["state_dict"][key]), key


def test_evaluation_scores_only_the_held_out_episodes(output_dir):
    run_training(make_config(output_dir))
    evaluation = run_evaluation(make_config(output_dir))
    results = evaluation["results"]
    checkpoint = torch.load(output_dir / CHECKPOINT_FILE, weights_only=True)
    decisions = json.loads(evaluation["paths"]["decisions"].read_text(encoding="utf-8"))
    assert sorted(d["episode_id"] for d in decisions) == sorted(checkpoint["split"]["test"])
    assert results["test_metrics"]["n_episodes"] == len(checkpoint["split"]["test"])
    assert results["test_metrics"]["threshold"] == checkpoint["threshold"]
    assert results["seed"] == 5
    assert set(results["not_evaluated"]) == {
        "structural_path_similarity",
        "severity",
        "calibration",
    }
    # Real node ids from the generator output are used in the report.
    assert all(vm.startswith("vm_") for d in decisions for vm in d["vms"])
    report = evaluation["paths"]["report"].read_text(encoding="utf-8")
    assert "## Held-out test set" in report


def test_evaluation_is_reproducible(output_dir):
    run_training(make_config(output_dir))
    a = run_evaluation(make_config(output_dir))["results"]["test_metrics"]
    b = run_evaluation(make_config(output_dir))["results"]["test_metrics"]
    assert a == b


def test_evaluation_needs_a_checkpoint(output_dir):
    with pytest.raises(ModelInputError, match="run training first"):
        run_evaluation(make_config(output_dir))


def test_evaluation_with_a_different_model_config(output_dir):
    run_training(make_config(output_dir))
    other = parse_config(
        {"paths": {"output_dir": str(output_dir)}, "model": {**SMALL_MODEL, "fusion": "concat"}},
        base_dir=output_dir,
    )
    with pytest.raises(ModelInputError, match="model config"):
        run_evaluation(other)


def test_fpr_threshold_strategy(output_dir):
    summary = run_training(make_config(output_dir, threshold_strategy="fpr", max_fpr=0.0))
    assert summary["val_metrics"]["false_positive_rate"] == 0.0


def write_yaml(tmp_path, output_dir):
    import yaml

    path = tmp_path / "small.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 5,
                "paths": {"output_dir": str(output_dir)},
                "model": SMALL_MODEL,
                "training": SMALL_TRAINING,
            }
        ),  # fmt: skip
        encoding="utf-8",
    )
    return path


def test_cli_train_and_evaluate(tmp_path, output_dir, capsys):
    config = str(write_yaml(tmp_path, output_dir))
    assert main(["--config", config, "train"]) == 0
    assert "threshold" in capsys.readouterr().out
    assert main(["--config", config, "evaluate"]) == 0
    out = capsys.readouterr().out
    assert "confusion matrix" in out
    assert (output_dir / "evaluation" / "decision_report.md").is_file()


def test_cli_evaluate_without_training(tmp_path, output_dir):
    assert main(["--config", str(write_yaml(tmp_path, output_dir)), "evaluate"]) == 1


def test_cli_train_without_graphs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--config", str(write_yaml(tmp_path, empty)), "train"]) == 1


def test_entry_scripts(tmp_path, output_dir):
    config = str(write_yaml(tmp_path, output_dir))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script in ("run_training.py", "run_evaluation.py"):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / script), "--config", config],
            capture_output=True, text=True, cwd=tmp_path, env=env,
        )  # fmt: skip
        assert result.returncode == 0, result.stderr
    assert (output_dir / "evaluation" / "metrics.json").is_file()
