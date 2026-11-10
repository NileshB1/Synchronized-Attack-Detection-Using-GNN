"""Averages over seeds, paired t-tests and readable comparison tables.

The variant marked as selected for a scenario is the one with the best mean
validation F1 (lowest validation loss breaks ties). Test results are shown for
every variant but play no part in that choice.

Each variant and baseline is compared with the reference variant (the first one
in the spec) by a two-tailed paired t-test over seeds on test F1 and ROC-AUC,
pairing runs that share a seed and so the same data and split. No correction
for multiple comparisons is applied, and with few seeds the test has little
power, so a high p-value is not evidence that two models are equal.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cadgnn.experiments.spec import ExperimentSpec

METRICS = [
    "val_f1", "best_val_loss", "test_f1", "test_precision", "test_recall", "test_fpr",
    "test_fnr", "test_roc_auc", "test_pr_auc", "detection_rate", "mean_hours_to_detect",
]  # fmt: skip
TESTED = ("test_f1", "test_roc_auc")
PATTERN_ORDER = ("scattered", "burst", "cluster", "lookalike", "fast", "stealth")
SUMMARY_FILE = "summary.csv"
REPORT_FILE = "summary.md"


def paired_p_value(a: pd.Series, b: pd.Series) -> float:
    """Two-tailed paired t-test of a against b, both indexed by seed.

    1.0 when every pair is identical; NaN when there are fewer than two pairs or
    the difference is the same non-zero value for every seed.
    """
    pairs = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(pairs) < 2:
        return math.nan
    diff = (pairs.iloc[:, 0] - pairs.iloc[:, 1]).to_numpy(dtype=float)
    if np.allclose(diff, 0):
        return 1.0
    if np.allclose(diff, diff[0]):
        return math.nan
    return float(stats.ttest_rel(pairs.iloc[:, 0], pairs.iloc[:, 1]).pvalue)


def pattern_columns(results: pd.DataFrame) -> list[str]:
    found = [c for c in results.columns if c.startswith("rate_")]
    known = [f"rate_{p}" for p in PATTERN_ORDER if f"rate_{p}" in found]
    return known + sorted(set(found) - set(known))


def summarise(results: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    if results.empty:
        raise ValueError("no experiment results to summarise")
    results = results.copy()
    if "kind" not in results:
        results["kind"] = "gnn"
    reference = reference or results["variant"].iloc[0]
    metrics = METRICS + pattern_columns(results)
    results[metrics] = results[metrics].apply(pd.to_numeric, errors="coerce")

    grouped = results.groupby(["scenario", "variant"], sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["runs"] = grouped.size()
    summary["kind"] = grouped["kind"].first()
    summary = summary.reset_index()

    summary["reference"] = reference
    for metric in TESTED:
        values = []
        for _, row in summary.iterrows():
            part = results[results["scenario"] == row["scenario"]]
            ours = part[part["variant"] == row["variant"]].set_index("seed")[metric]
            ref = part[part["variant"] == reference].set_index("seed")[metric]
            values.append(math.nan if row["variant"] == reference else paired_p_value(ours, ref))
        summary[f"p_{metric}"] = values

    summary["selected"] = False
    for _, part in summary.groupby("scenario", sort=False):
        best = part.sort_values(["val_f1_mean", "best_val_loss_mean"], ascending=[False, True])
        summary.loc[best.index[0], "selected"] = True
    return summary


def _cell(mean, std) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}"


def _p(value) -> str:
    if pd.isna(value):
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def render_summary(summary: pd.DataFrame, spec: ExperimentSpec) -> str:
    reference = summary["reference"].iloc[0] if "reference" in summary else spec.reference
    lines = [
        "# Experiment summary",
        "",
        f"Seeds: {', '.join(map(str, spec.seeds))}. Values are mean ± standard deviation over "
        "seeds. The selected variant (*) per scenario has the best mean validation F1; test "
        "results did not influence the choice.",
        "",
        f"p-values: two-tailed paired t-test over seeds against `{reference}`, on test F1 and "
        "test ROC-AUC, without correction for multiple comparisons. 1.000 means identical "
        "results on every seed. With this few seeds a large p-value does not show that two "
        "models are equally good.",
        "",
    ]
    patterns = [c.removesuffix("_mean") for c in summary.columns
                if c.startswith("rate_") and c.endswith("_mean")]  # fmt: skip
    for scenario, part in summary.groupby("scenario", sort=False):
        lines += [
            f"## {scenario}",
            "",
            "| Model | Type | Val F1 | Test F1 | Precision | Recall | FPR | FNR | ROC-AUC | PR-AUC "
            "| Detected | Hours to detect | p (F1) | p (ROC-AUC) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in part.iterrows():
            name = f"{row['variant']} *" if row["selected"] else row["variant"]
            cells = [
                _cell(row[f"{m}_mean"], row[f"{m}_std"])
                for m in ("val_f1", "test_f1", "test_precision", "test_recall", "test_fpr",
                          "test_fnr", "test_roc_auc", "test_pr_auc", "detection_rate",
                          "mean_hours_to_detect")
            ]  # fmt: skip
            cells += [_p(row.get("p_test_f1")), _p(row.get("p_test_roc_auc"))]
            kind = row.get("kind", "gnn")
            lines.append(f"| {name} | {kind} | " + " | ".join(cells) + " |")
        lines.append("")

        shown = [p for p in patterns if part[f"{p}_mean"].notna().any()]
        if shown:
            lines += [
                "Share of test episodes flagged, by generator pattern. Coincidental patterns "
                "(scattered, burst, cluster, lookalike) give the false alarm rate, coordinated "
                "ones (fast, stealth) the detection rate.",
                "",
                "| Model | " + " | ".join(p.removeprefix("rate_") for p in shown) + " |",
                "| --- | " + " | ".join("---" for _ in shown) + " |",
            ]
            for _, row in part.iterrows():
                cells = [_cell(row[f"{p}_mean"], row[f"{p}_std"]) for p in shown]
                lines.append(f"| {row['variant']} | " + " | ".join(cells) + " |")
            lines.append("")
    lines.append(
        "All episodes are synthetic. These numbers compare design choices within the "
        "generator's scenarios and do not measure performance on real attack campaigns."
    )
    return "\n".join(lines) + "\n"


def write_summary(
    out_dir: Path, results: pd.DataFrame, summary: pd.DataFrame, spec: ExperimentSpec
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": out_dir / "results.csv",
        "summary": out_dir / SUMMARY_FILE,
        "report": out_dir / REPORT_FILE,
    }
    results.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["report"].write_text(render_summary(summary, spec), encoding="utf-8")
    return paths
