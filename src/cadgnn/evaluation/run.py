"""Score the held-out test episodes with a saved checkpoint and write the report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cadgnn.config import Config, display_path
from cadgnn.evaluation.detection_time import time_to_detect
from cadgnn.evaluation.inference import predict
from cadgnn.evaluation.metrics import classification_metrics, rates_by_pattern
from cadgnn.evaluation.report import NOT_EVALUATED, episode_decisions, write_evaluation
from cadgnn.fusion import build_coordination_model
from cadgnn.generator import EDGES_FILE, NODES_FILE
from cadgnn.graph import InfrastructureGraph, load_episode_graphs
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE
from cadgnn.models import ModelInputError, load_checkpoint, resolve_device
from cadgnn.training.run import CHECKPOINT_FILE
from cadgnn.training.split import split_from_ids

EVALUATION_DIR = Path("evaluation")


def _load_infrastructure(output_dir: Path) -> InfrastructureGraph | None:
    # Only used to print real node ids in the report; without it ids fall back to indices.
    nodes, edges = output_dir / NODES_FILE, output_dir / EDGES_FILE
    if not (nodes.is_file() and edges.is_file()):
        return None
    return InfrastructureGraph.from_frames(
        pd.read_csv(nodes, dtype=str), pd.read_csv(edges, dtype=str)
    )


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def run_evaluation(
    config: Config, graphs_file: Path | None = None, run_dir: Path | None = None
) -> dict:
    """Evaluate the checkpoint in ``run_dir/model``; results go to ``run_dir/evaluation``."""
    graphs_file = graphs_file or config.paths.output_dir / EPISODE_GRAPHS_FILE
    run_dir = run_dir or config.paths.output_dir
    checkpoint_path = run_dir / CHECKPOINT_FILE
    if not checkpoint_path.is_file():
        raise ModelInputError(
            f"no trained model at {display_path(checkpoint_path)}, run training first"
        )

    graphs = load_episode_graphs(graphs_file)
    model = build_coordination_model(config.model)
    checkpoint = load_checkpoint(model, checkpoint_path)
    split = split_from_ids(graphs, checkpoint["split"])
    test = [graphs[i] for i in split.test]

    device = resolve_device(config.model.device)
    threshold = checkpoint["threshold"]
    pred = predict(model, test, device=device)
    test_metrics = classification_metrics(pred.y, pred.scores, threshold)
    patterns = [getattr(g, "pattern", "unknown") for g in test]
    by_pattern = rates_by_pattern(patterns, pred.y, pred.scores, threshold)
    detection = time_to_detect(model, test, threshold, config.graph.min_episode_events, device)
    # The generator writes its node and edge tables next to the graphs.
    infra = _load_infrastructure(graphs_file.parent)
    decisions = episode_decisions(test, pred, threshold, infra)

    gate_summary = None
    if pred.gates is not None:
        flagged = pred.scores >= threshold
        gate_summary = {
            "mean_temporal_share_flagged": _mean_or_none(pred.gates[flagged]),
            "mean_temporal_share_not_flagged": _mean_or_none(pred.gates[~flagged]),
        }

    results = {
        "seed": checkpoint["seed"],
        "checkpoint": display_path(checkpoint_path),
        "episode_graphs": display_path(graphs_file),
        "model_config": checkpoint["model_config"],
        "training_config": checkpoint["training_config"],
        "threshold_strategy": checkpoint["training_config"]["threshold_strategy"],
        "best_epoch": checkpoint["best_epoch"],
        "split_sizes": {name: len(getattr(split, name)) for name in ("train", "val", "test")},
        "val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "time_to_detect": detection,
        "by_pattern": by_pattern,
        "gate_summary": gate_summary,
        "mean_test_score": float(np.mean(pred.scores)),
        "not_evaluated": NOT_EVALUATED,
    }
    paths = write_evaluation(run_dir / EVALUATION_DIR, results, decisions)
    return {"results": results, "paths": paths}
