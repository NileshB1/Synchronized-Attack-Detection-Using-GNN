"""Per-episode decisions and the final evaluation report.

Every field comes from the model output or the episode graph itself. There are
no severity labels in the data, so no severity is reported. The propagation
path for a flagged episode is the time-ordered list of its VMs with the shared
network or storage linking each step; it is evidence for an analyst, not a path
the model predicted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from cadgnn.evaluation.inference import Predictions
from cadgnn.graph import InfrastructureGraph

LABEL_NAMES = {0: "coincidental", 1: "coordinated"}
NOT_EVALUATED = {
    "structural_path_similarity": (
        "not computed: the ground truth only gives the order of the same VMs the model "
        "receives, so there is no separately predicted path to compare against"
    ),
    "severity": "not reported: the generated data has no severity labels",
    "calibration": "scores are sigmoid outputs and have not been calibrated as probabilities",
}


def _node_names(graph: HeteroData, kind: str, infra: InfrastructureGraph | None) -> list[str]:
    index = graph[kind].global_index.tolist()
    if infra is None:
        return [f"{kind}#{i}" for i in index]
    return [infra.node_ids[kind][i] for i in index]


def _shared_resources(graph: HeteroData, vm_names: list[str], infra) -> dict[str, dict[str, list]]:
    """Networks and storage pools used by more than one of the episode's VMs."""
    users: dict[str, dict[str, list[str]]] = {
        "network": defaultdict(list),
        "storage": defaultdict(list),
    }
    networks = _node_names(graph, "network", infra)
    for vm, net in graph["vm", "attached_to", "network"].edge_index.t().tolist():
        users["network"][networks[net]].append(vm_names[vm])

    storages = _node_names(graph, "storage", infra)
    image_owner = dict(
        (img, vm) for vm, img in graph["vm", "has_backup", "backup_image"].edge_index.t().tolist()
    )
    for img, sto in graph["backup_image", "stored_on", "storage"].edge_index.t().tolist():
        users["storage"][storages[sto]].append(vm_names[image_owner[img]])
    return {
        kind: {name: sorted(vms) for name, vms in found.items() if len(vms) > 1}
        for kind, found in users.items()
    }


def episode_decisions(
    graphs: Sequence[HeteroData],
    predictions: Predictions,
    threshold: float,
    infra: InfrastructureGraph | None = None,
) -> list[dict]:
    by_id = {g.episode_id: g for g in graphs}
    decisions = []
    for i, episode_id in enumerate(predictions.episode_ids):
        g = by_id[episode_id]
        order = torch.argsort(g["vm"].event_order)
        names = _node_names(g, "vm", infra)
        vm_order = [names[j] for j in order.tolist()]
        times = g["vm"].timestamp_hours[order].tolist()
        score = float(predictions.scores[i])
        decision = int(score >= threshold)
        shared = _shared_resources(g, names, infra)

        record = {
            "episode_id": episode_id,
            "pattern": getattr(g, "pattern", "unknown"),
            "true_label": LABEL_NAMES[int(predictions.y[i])],
            "decision": LABEL_NAMES[decision],
            "correct": decision == int(predictions.y[i]),
            "score": score,
            "threshold": threshold,
            "temporal_share": (
                float(predictions.gates[i]) if predictions.gates is not None else None
            ),
            "vms": vm_order,
            "time_window_hours": {
                "start": times[0],
                "end": times[-1],
                "span": times[-1] - times[0],
            },  # fmt: skip
            "shared_resources": shared,
            "propagation_path": None,
        }
        if decision == 1:
            record["propagation_path"] = [
                {
                    "from": a,
                    "to": b,
                    "hours_between": t_b - t_a,
                    "via": sorted(
                        name
                        for kind in ("network", "storage")
                        for name, vms in shared[kind].items()
                        if a in vms and b in vms
                    ),
                }
                for a, b, t_a, t_b in zip(vm_order, vm_order[1:], times, times[1:], strict=False)
            ]
        decisions.append(record)
    return decisions


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_pattern_table(by_pattern: dict) -> list[str]:
    """False alarms per coincidental pattern and detections per coordinated one."""
    if not by_pattern:
        return []
    lines = [
        "## Flagged episodes by pattern",
        "",
        "For coincidental patterns the rate is the false alarm rate, for coordinated ones "
        "the detection rate.",
        "",
        "| Pattern | Label | Episodes | Flagged | Rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pattern, row in by_pattern.items():
        lines.append(
            f"| {pattern} | {row['label']} | {row['episodes']} | {row['flagged']} | "
            f"{row['flagged_rate']:.3f} |"
        )
    return [*lines, ""]


def render_report(results: dict, decisions: list[dict]) -> str:
    test = results["test_metrics"]
    cm = test["confusion_matrix"]
    ttd = results["time_to_detect"]
    mc = results["model_config"]
    if mc.get("architecture") == "chained":
        model = "CONTINUUM-style baseline, a GAT on each snapshot followed by a GRU"
    else:
        model = (
            f"fusion `{mc['fusion']}`, temporal stream {'on' if mc['use_temporal'] else 'off'}, "
            f"spatial stream {'on' if mc['use_spatial'] else 'off'}"
        )
    lines = [
        "# Coordination decision report",
        "",
        f"Model: {model}. Seed {results['seed']}.",
        f"Episodes: {results['split_sizes']['train']} train, {results['split_sizes']['val']} "
        f"validation, {results['split_sizes']['test']} test (stratified by label).",
        f"Threshold {test['threshold']:.4f}, chosen on the validation set with strategy "
        f"`{results['threshold_strategy']}`.",
        "",
        "## Held-out test set",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
                "balanced_accuracy", "roc_auc", "pr_auc"):  # fmt: skip
        lines.append(f"| {key} | {_fmt(test[key])} |")
    lines += [
        "",
        "Confusion matrix (positive = coordinated):",
        "",
        "| | predicted coordinated | predicted coincidental |",
        "| --- | --- | --- |",
        f"| actual coordinated | {cm['tp']} | {cm['fn']} |",
        f"| actual coincidental | {cm['fp']} | {cm['tn']} |",
        "",
        "## Time to detect (coordinated test episodes)",
        "",
        f"Detected {ttd['detected']} of {ttd['n_campaigns']} campaigns when replayed event by "
        f"event. Mean {_fmt(ttd['mean_hours_to_detect'])} h, median "
        f"{_fmt(ttd['median_hours_to_detect'])} h after the first flagged event.",
        "",
    ]
    lines += render_pattern_table(results.get("by_pattern") or {})
    lines += [
        "## Decisions",
        "",
        "| Episode | True | Decision | Score | Temporal share | VMs | Span (h) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in decisions:
        lines.append(
            f"| {d['episode_id']} | {d['true_label']} | {d['decision']} | {d['score']:.3f} | "
            f"{_fmt(d['temporal_share'])} | {len(d['vms'])} | "
            f"{d['time_window_hours']['span']:.2f} |"
        )
    flagged = [d for d in decisions if d["propagation_path"]]
    if flagged:
        lines += ["", "## Propagation evidence for flagged episodes", ""]
        for d in flagged:
            steps = " -> ".join(d["vms"])
            via = sorted({v for step in d["propagation_path"] for v in step["via"]})
            lines.append(f"- {d['episode_id']}: {steps} (shared: {', '.join(via) or 'none'})")
    lines += ["", "## Not evaluated", ""]
    lines += [f"- {key}: {reason}" for key, reason in results["not_evaluated"].items()]
    lines += [
        "",
        "Results come from synthetic episodes produced by the random anomaly generator and "
        "do not show performance on real attack campaigns.",
        "",
    ]
    return "\n".join(lines)


def write_evaluation(output_dir: str | Path, results: dict, decisions: list[dict]) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "metrics.json",
        "decisions": output_dir / "decisions.json",
        "report": output_dir / "decision_report.md",
    }
    paths["metrics"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    paths["decisions"].write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    paths["report"].write_text(render_report(results, decisions), encoding="utf-8")
    return paths
