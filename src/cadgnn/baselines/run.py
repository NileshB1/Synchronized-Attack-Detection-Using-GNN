"""Fit and score the non-graph baselines on the same split as the GNN
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from torch_geometric.data import HeteroData

from cadgnn.baselines.classic import BASELINES, FittedBaseline, fit_baseline
from cadgnn.baselines.features import labels
from cadgnn.config import Config, TrainingConfig, display_path
from cadgnn.evaluation import (
    classification_metrics, detection_times,
    rates_by_pattern, select_threshold
)
from cadgnn.graph import load_episode_graphs
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE
from cadgnn.training.split import Split, stratified_split

BASELINES_DIR = Path("baselines")


def patterns_of(graphs: Sequence[HeteroData]) -> list[str]:
    return [getattr(g, "pattern", "unknown") for g in graphs]


def evaluate_baseline(
    fitted: FittedBaseline,
    graphs: Sequence[HeteroData],
    split: Split,   training: TrainingConfig,
    min_events: int = 2
) -> dict:
    val = [graphs[i] for i in split.val]
    test = [graphs[i] for i in split.test]
    val_scores = fitted.score(val)
    threshold = select_threshold(
        labels(val), val_scores, training.threshold_strategy, training.max_fpr,
        training.min_precision,
    )  # fmt: skip
    test_y, test_scores = labels(test), fitted.score(test)
    return {
        "baseline": fitted.name,
        "balancing": "n/a" if fitted.name == "rule_based" else training.balancing,
        "settings": fitted.settings,  "threshold": threshold,
        "val_metrics": classification_metrics(labels(val), val_scores, threshold),
        "test_metrics": classification_metrics(test_y, test_scores, threshold),
        "time_to_detect": detection_times(fitted.score, test, threshold, min_events),
        "by_pattern": rates_by_pattern(patterns_of(test), test_y, test_scores, threshold),
        "mean_test_score": float(np.mean(test_scores)),
    }


def baseline_results(
    graphs: Sequence[HeteroData],
    config: Config,
    names: Sequence[str] = BASELINES,
) -> dict[str, dict]:
    tc = config.training
    split = stratified_split(graphs, tc.val_fraction, tc.test_fraction, config.seed)
    train = [graphs[i] for i in split.train]
    val = [graphs[i] for i in split.val]
    results = {}
    for name in names:
        fitted = fit_baseline(name, train, val, config.seed, tc.balancing)
        results[name] = evaluate_baseline(
            fitted, graphs, split, tc, config.graph.min_episode_events
        )
        results[name]["split_sizes"] = {
            part: len(getattr(split, part)) for part in ("train", "val", "test")
        }
    return results


def run_baselines(
    config: Config, graphs_file: Path | None = None, run_dir: Path | None = None
) -> dict:
    """Results go to ``run_dir/baselines/metrics.json``, by default in the output directory."""
    graphs_file = graphs_file or config.paths.output_dir / EPISODE_GRAPHS_FILE
    run_dir = run_dir or config.paths.output_dir
    results = baseline_results(load_episode_graphs(graphs_file), config)
    path = run_dir / BASELINES_DIR / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": config.seed, "episode_graphs": display_path(graphs_file), **results}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"results": results, "path": path}
