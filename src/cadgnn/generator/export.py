"""Writing the generated scenario to CSV files and a JSON manifest."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cadgnn import __version__
from cadgnn.data.calibration import Calibration
from cadgnn.generator.events import SyntheticDataset

TELEMETRY_FILE = "synthetic_backup_telemetry.csv"
NODES_FILE = "infrastructure_graph_nodes.csv"
EDGES_FILE = "infrastructure_graph_edges.csv"
MANIFEST_FILE = "experiment_manifest.json"
PATTERNS_FILE = "episode_patterns.csv"


def build_manifest(dataset: SyntheticDataset, calibration: Calibration) -> dict:
    graph, events, config = dataset.graph, dataset.events, dataset.config
    profile = calibration.profile
    return {
        "random_seed": dataset.seed,
        "num_vms": config.n_vms,
        "num_storage": config.n_storage,  "num_network": config.n_network,
        "total_graph_nodes": graph.number_of_nodes(), "total_graph_edges": graph.number_of_edges(),
        "num_features": len(profile),
        "features": list(profile),  "feature_weights": {name: f.weight for name, f in profile.items()},
        "total_events": len(events),
        "label_counts": {k: int(v) for k, v in events["label"].value_counts().items()},
        "pattern_counts": dict(sorted(Counter(dataset.patterns.values()).items())),
        "calibration_sources": {
            "ransap": "Real ransomware storage write & Shannon entropy traces",
            "napierone": "Ransomware vs normal file size encryption ratios",
            "filesystems": (
                f"FSL Filesystems Archive ({len(calibration.fsl)} snapshot trace files found, "
                f"max file_count={calibration.fsl_file_count})"
            ),
        },
        "generator": asdict(config),
        "param_profile": {name: asdict(f) for name, f in profile.items()},
        "calibration": {
            "ransap": calibration.ransap.to_dict(orient="records"),
            "napierone_normal_avg_kb": calibration.napierone_normal_kb,
            "napierone_ransomware_avg_kb": calibration.napierone_ransomware_kb,
            "size_ratio": calibration.size_ratio, "fsl_file_count": calibration.fsl_file_count,
            "fallbacks": calibration.fallbacks,
        },
        "package_version": __version__,
    }


def save_outputs(
    dataset: SyntheticDataset, calibration: Calibration, output_dir: str | Path
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": output_dir / TELEMETRY_FILE,
        "nodes": output_dir / NODES_FILE,  "edges": output_dir / EDGES_FILE,
        "manifest": output_dir / MANIFEST_FILE,
        "patterns": output_dir / PATTERNS_FILE,
    }

    dataset.events.to_csv(paths["telemetry"], index=False)
    pd.DataFrame(
        sorted(dataset.patterns.items()), columns=["episode_id", "pattern"]
    ).to_csv(paths["patterns"], index=False)
    nodes = [{"node_id": n, "kind": d["kind"]} for n, d in dataset.graph.nodes(data=True)]
    pd.DataFrame(nodes, columns=["node_id", "kind"]).to_csv(paths["nodes"], index=False)
    edges = [{"source": u, "target": v} for u, v in dataset.graph.edges()]
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(paths["edges"], index=False)
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(build_manifest(dataset, calibration), f, indent=2)
    return paths
