"""Table of the tunable settings of the dual-stream GNN and its training.

The table is built from the config classes, so defaults and current values are
always the ones the code uses. A test checks that every model and training
setting has an entry here.
"""

from __future__ import annotations

from dataclasses import asdict, fields

from cadgnn.config import Config, GraphConfig, ModelConfig, TrainingConfig

# (section, setting) -> (component, what it controls, values worth trying)
PARAMETERS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("graph", "min_episode_events"): (
        "graph construction", "smallest episode turned into a graph", "2, 3"),
    ("model", "architecture"): (
        "baseline", "dual_stream, or the CONTINUUM-style chained GAT then GRU",
        "dual_stream, chained"),
    ("model", "hidden_dim"): (
        "both streams", "width of node and sequence representations", "32, 64, 128"),
    ("model", "temporal_dim"): (
        "temporal stream", "size of the temporal episode vector", "32, 64"),
    ("model", "spatial_dim"): (
        "spatial stream", "size of the spatial episode vector", "32, 64"),
    ("model", "temporal_layers"): ("temporal stream", "stacked GRU layers", "1, 2"),
    ("model", "spatial_layers"): (
        "spatial stream", "GAT layers (2 lets VMs on one network see each other)", "1, 2, 3"),
    ("model", "spatial_heads"): ("spatial stream", "attention heads per GAT layer", "1, 2, 4"),
    ("model", "use_temporal"): ("ablation", "temporal stream on or off", "true, false"),
    ("model", "use_spatial"): ("ablation", "spatial stream on or off", "true, false"),
    ("model", "fusion"): ("fusion", "how the two streams are combined", "gated, concat"),
    ("model", "fusion_dim"): ("fusion", "size of the fused vector", "32, 64"),
    ("model", "dropout"): ("all", "dropout rate", "0.0, 0.1, 0.3"),
    ("model", "device"): ("runtime", "cpu, cuda or auto", "not tuned"),
    ("training", "val_fraction"): ("split", "share of episodes for validation", "0.2"),
    ("training", "test_fraction"): ("split", "share of episodes held out for testing", "0.1, 0.2"),
    ("training", "epochs"): ("training", "maximum epochs", "100, 200"),
    ("training", "batch_size"): ("training", "episodes per batch", "8, 16, 32"),
    ("training", "learning_rate"): ("training", "Adam learning rate", "1e-3, 3e-4"),
    ("training", "weight_decay"): ("training", "L2 penalty in Adam", "0, 1e-4, 1e-3"),
    ("training", "patience"): ("training", "epochs without validation gain before stopping", "15"),
    ("training", "threshold_strategy"): (
        "decision", "best validation F1, or best recall under an FPR cap or precision floor",
        "f1, fpr, precision"),
    ("training", "max_fpr"): (
        "decision", "FPR cap for the fpr strategy (alert fatigue)", "0.05, 0.1"),
    ("training", "min_precision"): (
        "decision", "precision floor for the precision strategy", "0.8, 0.9"),
    ("training", "balancing"): (
        "imbalance", "class weights, or resampling of the training part",
        "class_weight, none, oversample, undersample"),
    ("training", "fp_cost"): (
        "false positives", "loss weight of a coincidental episode", "1, 2, 4"),
}  # fmt: skip

SECTIONS = {"graph": GraphConfig, "model": ModelConfig, "training": TrainingConfig}


def parameter_rows(config: Config) -> list[dict]:
    rows = []
    for section, cls in SECTIONS.items():
        current = asdict(getattr(config, section))
        for f in fields(cls):
            component, meaning, values = PARAMETERS[(section, f.name)]
            rows.append({
                "setting": f"{section}.{f.name}",
                "component": component,
                "controls": meaning,
                "default": f.default,
                "current": current[f.name],
                "values_to_try": values,
            })  # fmt: skip
    return rows


def render_parameter_table(config: Config) -> str:
    lines = [
        "| Setting | Component | Controls | Default | Current | Values to try |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in parameter_rows(config):
        lines.append(
            f"| `{r['setting']}` | {r['component']} | {r['controls']} | {r['default']} | "
            f"{r['current']} | {r['values_to_try']} |"
        )
    return "\n".join(lines) + "\n"
