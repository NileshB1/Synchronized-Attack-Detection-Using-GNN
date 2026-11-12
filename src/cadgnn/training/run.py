"""Split, train, choose the threshold on validation and save a checkpoint
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from cadgnn.config import Config
from cadgnn.evaluation import classification_metrics, predict, select_threshold
from cadgnn.fusion import build_coordination_model
from cadgnn.graph import load_episode_graphs
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE
from cadgnn.models import resolve_device, save_checkpoint, seed_everything
from cadgnn.training.split import save_split, stratified_split
from cadgnn.training.trainer import train_model

log = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("model") / "coordination_model.pt"
SPLIT_FILE = Path("splits") / "split.json"


def run_training(
    config: Config, graphs_file: Path | None = None, run_dir: Path | None = None
) -> dict:
    """Train on the episode graphs; the checkpoint goes to ``run_dir/model``
    """
    output_dir = config.paths.output_dir
    run_dir = run_dir or output_dir
    tc = config.training
    graphs = load_episode_graphs(graphs_file or output_dir / EPISODE_GRAPHS_FILE)

    seed_everything(config.seed)
    split = stratified_split(graphs, tc.val_fraction, tc.test_fraction, config.seed)
    train = [graphs[i] for i in split.train]
    val = [graphs[i] for i in split.val]
    log.info("split: %d train, %d val, %d test", len(train), len(val), len(split.test))
    save_split(split, graphs, run_dir / SPLIT_FILE, seed=config.seed)

    device = resolve_device(config.model.device)
    model = build_coordination_model(config.model)
    result = train_model(model, train, val, tc, config.seed, device)

    val_pred = predict(model, val, device=device)
    threshold = select_threshold(
        val_pred.y, val_pred.scores, tc.threshold_strategy, tc.max_fpr, tc.min_precision
    )
    val_metrics = classification_metrics(val_pred.y, val_pred.scores, threshold)

    path = run_dir / CHECKPOINT_FILE
    save_checkpoint(
        model.cpu(),
        path,  seed=config.seed,
        training_config=asdict(tc), split=split.episode_ids(graphs),
        threshold=threshold,  best_epoch=result.best_epoch,
        best_val_loss=result.best_val_loss, pos_weight=result.pos_weight,
        history=result.history, val_metrics=val_metrics
    )
    return {
        "checkpoint": path, "split_file": run_dir / SPLIT_FILE,
        "threshold": threshold,  "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,   "epochs_run": len(result.history),
        "val_metrics": val_metrics,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(split.test)}
    }
