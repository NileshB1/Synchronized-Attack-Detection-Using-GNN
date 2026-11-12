"""Markdown and CSV tables of the cross-validation study.

Only numbers go in here
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cadgnn.crossval.runner import BEFORE, CrossvalResult, curve_summary
from cadgnn.crossval.spec import CrossvalSpec

REPORTS_DIR = Path("reports")
SUMMARY = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp",
           "false_positive_rate")
NAMES = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "fp": "FP", "false_positive_rate": "FP rate",
    "brier": "Brier",
}  # fmt: skip
# Gaps and loss levels used for the note under the training curves.
CLOSE_GAP = 0.05
HIGH_LOSS = 0.5


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}f}"


def p_value(value) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def mean_std(values) -> str:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return "n/a"
    if len(values) == 1:
        return num(float(values.iloc[0]))
    return f"{values.mean():.3f} ± {values.std():.3f}"


def markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def fold_summary(folds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in folds.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key, strict=True))
        has_brier = "brier" in part and part["brier"].notna().any()
        for metric in SUMMARY + (("brier",) if has_brier else ()):
            row[NAMES[metric]] = mean_std(part[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def render_imbalance(result: CrossvalResult) -> str:
    counts = result.class_counts.copy()
    counts["coordinated_share"] = counts["coordinated_share"].map(lambda v: f"{100 * v:.1f}%")
    counts["imbalance_ratio"] = counts["imbalance_ratio"].map(lambda v: f"{v:.2f} : 1")
    lines = [
        "# Class distribution and balancing", "",
        "Episodes per part (the target the models predict):", "", markdown(counts),
    ]  # fmt: skip
    if result.event_counts is not None:
        events = result.event_counts.copy()
        events["share"] = events["share"].map(lambda v: f"{100 * v:.2f}%")
        lines += ["Raw backup events by label:", "", markdown(events)]
    balancing = result.folds[result.folds["study"] == "balancing"]
    lines += [
        "Balancing methods on the cross-validation folds (mean ± std over folds). Resampling "
        "is applied to the training folds only; validation, held-out folds and test keep the "
        "real ratio. smote needs feature vectors, so only the feature baselines use it.",
        "",
        markdown(fold_summary(balancing, ["model", "balancing"])),
        f"Chosen: `{result.balancing}`, the method with the best mean F1 for "
        f"`{result.reference}` (lower false positive rate breaks a tie). Every model uses it "
        "from here on; the rule-based baseline has nothing to train.",
        "",
    ]
    return "\n".join(lines)


def curve_note(summary: dict) -> str:
    if not summary:
        return "No training curves were recorded."
    numbers = (
        f"At the best epoch (mean {summary['best_epoch']:.1f}) the training loss is "
        f"{summary['train_loss']:.3f} and the validation loss {summary['val_loss']:.3f} "
        f"(gap {summary['gap']:+.3f}); accuracy {summary['train_acc']:.3f} on training and "
        f"{summary['val_acc']:.3f} on validation."
    )
    if summary["train_loss"] > HIGH_LOSS and summary["val_loss"] > HIGH_LOSS:
        reading = f"Both losses stay above {HIGH_LOSS}: a sign of underfitting."
    elif summary["gap"] > CLOSE_GAP:
        reading = (f"Validation loss is more than {CLOSE_GAP} above training loss: a sign of "
                   "overfitting, cut short by early stopping at the best epoch.")  # fmt: skip
    elif summary["gap"] < -CLOSE_GAP:
        reading = ("Validation loss is below training loss, as dropout and resampling only "
                   "act during training: no sign of overfitting.")  # fmt: skip
    else:
        reading = f"The two losses are within {CLOSE_GAP}: no clear over- or underfitting."
    return f"{numbers} {reading}"


def spec_strategy(result: CrossvalResult) -> str:
    rows = result.folds[result.folds["study"] == "cv"]
    return str(rows["strategy"].iloc[0]) if len(rows) else "default"


def render_cv(result: CrossvalResult, spec: CrossvalSpec) -> str:
    cv = result.folds[result.folds["study"] == "cv"]
    lines = [
        "# Cross-validation", "",
        f"{spec.folds}-fold stratified cross-validation on the training part. In each fold the "
        "model trains on the other folds, stops early and picks its threshold on the "
        "validation part, and is scored on the held-out fold. Mean ± std over folds.", "",
        markdown(fold_summary(cv, ["model", "kind", "balancing"])),
    ]  # fmt: skip
    test = result.test.set_index("model")
    acc = []
    for model, part in cv.groupby("model", sort=False):
        acc.append({
            "Model": model, "CV accuracy": mean_std(part["accuracy"]),
            "Validation accuracy": num(test.loc[model, "val_accuracy"]),
            "Test accuracy": num(test.loc[model, "accuracy"]),
        })  # fmt: skip
    lines += ["Accuracy on the folds, on the validation part (final model) and on the test "
              "part (scored once):", "", markdown(pd.DataFrame(acc))]  # fmt: skip
    sums = cv.groupby("model", sort=False)[["tp", "fp", "tn", "fn"]].sum().reset_index()
    lines += ["Confusion matrix summed over the held-out folds:", "", markdown(sums)]
    if not result.tuning.empty:
        tuning = result.tuning.copy()
        tuning["chosen"] = tuning["chosen"].map({True: "yes", False: ""})
        for c in ("val_roc_auc", "val_f1", "best_val_loss"):
            tuning[c] = tuning[c].map(num)
        lines += [f"Settings of `{result.reference}` tried on the validation part (best "
                  "validation ROC-AUC wins, lower validation loss breaks a tie). The chosen "
                  "values are used by every GNN variant.", "", markdown(tuning)]  # fmt: skip
    lines += ["## Training curves", "", curve_note(curve_summary(result.histories)), ""]
    return "\n".join(lines)


def render_false_positives(result: CrossvalResult, spec: CrossvalSpec) -> str:
    fp = result.folds[result.folds["study"] == "fp"]
    names = {
        BEFORE: f"no fix: the model's own {spec_strategy(result)} threshold",
        "threshold_0.5": "fixed threshold 0.5",
        "threshold_precision": f"threshold for validation precision >= {spec.min_precision}",
        "platt": "Platt scaling on validation, threshold 0.5",
        "isotonic": "isotonic calibration on validation, threshold 0.5",
        "cost_sensitive": f"false alarms cost {spec.fp_cost}x in the loss, best-F1 threshold",
    }
    table = fold_summary(fp, ["strategy"])
    table.insert(1, "What it does", table["strategy"].map(names))
    test = result.test
    ours = test[test["model"].str.startswith(f"{result.reference} + ")]
    cm = ours[["strategy", "threshold", "tp", "fp", "tn", "fn", "precision", "recall",
               "false_positive_rate"]].copy()  # fmt: skip
    for c in ("threshold", "precision", "recall", "false_positive_rate"):
        cm[c] = cm[c].map(num)
    cv = result.folds[result.folds["study"] == "cv"]
    models = []
    for model, part in cv.groupby("model", sort=False):
        row = test.set_index("model").loc[model]
        models.append({
            "Model": model, "CV FP (per fold)": mean_std(part["fp"]),
            "CV FP rate": mean_std(part["false_positive_rate"]),
            "Test FP": num(row["fp"]), "Test FP rate": num(row["false_positive_rate"]),
        })  # fmt: skip
    after = ours[ours["strategy"] == result.fp_strategy].iloc[0]
    if result.fp_strategy != BEFORE:
        models.append({
            "Model": f"{result.reference} + {result.fp_strategy}",
            "CV FP (per fold)": mean_std(fp[fp["strategy"] == result.fp_strategy]["fp"]),
            "CV FP rate": mean_std(fp[fp["strategy"] == result.fp_strategy]
                                   ["false_positive_rate"]),
            "Test FP": num(after["fp"]), "Test FP rate": num(after["false_positive_rate"]),
        })  # fmt: skip
    return "\n".join([
        "# False positives", "",
        f"Fixes for `{result.reference}` compared on the cross-validation folds (mean ± std). "
        "Thresholds and calibration are fitted on the validation part only.", "",
        markdown(table),
        f"Chosen: `{result.fp_strategy}`, the lowest mean false positive rate among the fixes "
        f"whose mean recall is at most {spec.max_recall_drop} below the model without a fix "
        "(`default`).", "",
        f"Test part, `{result.reference}` before and after the fix:", "", markdown(cm),
        "False positives of every model:", "", markdown(pd.DataFrame(models)),
    ])  # fmt: skip


def render_novelty(result: CrossvalResult, spec: CrossvalSpec) -> str:
    cv = result.folds[result.folds["study"] == "cv"]
    gnn = cv[cv["kind"] == "gnn"]
    ref = gnn[gnn["model"] == result.reference]
    changes = {v.name: ", ".join(f"{s}.{k}={val}" for s in ("model", "training")
                                 for k, val in getattr(v, s).items()) or "proposed model"
               for v in spec.variants}  # fmt: skip
    rows = []
    for model, part in gnn.groupby("model", sort=False):
        row = {"Model": model, "Change": changes.get(model, "")}
        for metric in ("f1", "roc_auc", "pr_auc", "false_positive_rate"):
            mean = part[metric].astype(float).mean()
            row[NAMES[metric]] = mean_std(part[metric])
            if model != result.reference:
                delta = mean - ref[metric].astype(float).mean()
                row[f"{NAMES[metric]} change"] = f"{delta:+.3f}"
            else:
                row[f"{NAMES[metric]} change"] = "-"
        rows.append(row)
    sig = result.significance.copy()
    table = pd.DataFrame({
        "Model": sig["model"],
        f"{result.reference} CV F1": sig["ours_f1"].map(num),
        "Their CV F1": sig["theirs_f1"].map(num),  "t-test p (F1)": sig["t_test_p_f1"].map(p_value),
        "Wilcoxon p (F1)": sig["wilcoxon_p_f1"].map(p_value), "t-test p (ROC-AUC)": sig["t_test_p_roc_auc"].map(p_value),
        "McNemar: only ours right": sig["mcnemar_only_ours"],
        "McNemar: only theirs right": sig["mcnemar_only_theirs"],  "McNemar p (test)": sig["mcnemar_p"].map(p_value)
    })
    return "\n".join([
        "# Ablation and significance", "",
        f"Ablation: each variant changes one part of `{result.reference}`. Change columns are "
        "variant minus the proposed model, mean over the folds.", "",
        markdown(pd.DataFrame(rows)),
        f"Significance against `{result.reference}`: paired t-test and Wilcoxon signed-rank test "
        f"over the {spec.folds} folds, McNemar's exact test on the test decisions. No "
        f"correction for multiple comparisons. With {spec.folds} folds the smallest possible "
        f"two-sided Wilcoxon p-value is {2 / 2 ** spec.folds:.4f}.", "",
        markdown(table),
    ])  # fmt: skip


RESULT_COLUMNS = [
    "Model", "Balancing", "CV Acc (mean±std)", "Val Acc", "Test Acc", "Precision", "Recall",
    "F1", "ROC-AUC", "PR-AUC", "FP", "FP Rate", "p-value vs ours", "Beats baseline? (Y/N)",
]  # fmt: skip
HIGHER = ("CV Acc (mean±std)", "Val Acc", "Test Acc", "Precision", "Recall", "F1", "ROC-AUC",
          "PR-AUC")  # fmt: skip
LOWER = ("FP", "FP Rate")


def results_frame(result: CrossvalResult) -> pd.DataFrame:
    """One row per model with raw numbers; the CSV version of the results table."""
    folds, test = result.folds, result.test.set_index("model")
    sig = result.significance.set_index("model")
    rows = []
    models = list(dict.fromkeys(folds.loc[folds["study"] == "cv", "model"]))
    fp_name = f"{result.reference} + {result.fp_strategy}"
    extra = [fp_name] if result.fp_strategy != BEFORE else []
    for model in [*models[:1], *extra, *models[1:]]:
        if model == fp_name:
            part = folds[(folds["study"] == "fp") & (folds["strategy"] == result.fp_strategy)]
            balancing = result.balancing
        else:
            part = folds[(folds["study"] == "cv") & (folds["model"] == model)]
            balancing = part["balancing"].iloc[0]
        row = test.loc[model]
        is_ours = model == result.reference
        rows.append({
            "model": model, "balancing": balancing,
            "cv_accuracy_mean": part["accuracy"].mean(), "cv_accuracy_std": part["accuracy"].std(),
            "val_accuracy": row.get("val_accuracy"), "test_accuracy": row["accuracy"],
            "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
            "roc_auc": row["roc_auc"], "pr_auc": row["pr_auc"], "fp": row["fp"],
            "fp_rate": row["false_positive_rate"],
            "p_value_vs_ours": np.nan if is_ours else sig.loc[model, "t_test_p_f1"],
            "ours_beats_it": None if is_ours or model == fp_name
            else bool(sig.loc[model, "ours_better"]),
        })
    return pd.DataFrame(rows)


def _bold_best(cells: list[str], values: list[float], lower: bool) -> list[str]:
    finite = [v for v in values if v is not None and not np.isnan(v)]
    if not finite:
        return cells
    best = min(finite) if lower else max(finite)
    return [f"**{c}**" if v is not None and not np.isnan(v) and np.isclose(v, best) else c
            for c, v in zip(cells, values, strict=True)]  # fmt: skip


def render_results(frame: pd.DataFrame, result: CrossvalResult, spec: CrossvalSpec) -> str:
    table = pd.DataFrame({
        "Model": frame["model"], "Balancing": frame["balancing"],
        "CV Acc (mean±std)": [f"{m:.3f} ± {s:.3f}" for m, s in
                              zip(frame["cv_accuracy_mean"], frame["cv_accuracy_std"],
                                  strict=True)],
        "Val Acc": frame["val_accuracy"].map(num), "Test Acc": frame["test_accuracy"].map(num),
        "Precision": frame["precision"].map(num), "Recall": frame["recall"].map(num),
        "F1": frame["f1"].map(num), "ROC-AUC": frame["roc_auc"].map(num),
        "PR-AUC": frame["pr_auc"].map(num), "FP": frame["fp"].map(lambda v: num(int(v))),
        "FP Rate": frame["fp_rate"].map(num),
        "p-value vs ours": frame["p_value_vs_ours"].map(p_value),
        "Beats baseline? (Y/N)": frame["ours_beats_it"].map(
            lambda v: "-" if v is None or v is np.nan else ("Y" if v else "N")),
    })  # fmt: skip
    raw = {
        "CV Acc (mean±std)": frame["cv_accuracy_mean"], "Val Acc": frame["val_accuracy"],
        "Test Acc": frame["test_accuracy"], "Precision": frame["precision"],
        "Recall": frame["recall"], "F1": frame["f1"], "ROC-AUC": frame["roc_auc"],
        "PR-AUC": frame["pr_auc"], "FP": frame["fp"], "FP Rate": frame["fp_rate"],
    }  # fmt: skip
    for column, values in raw.items():
        values = pd.to_numeric(values, errors="coerce").tolist()
        table[column] = _bold_best(table[column].tolist(), values, column in LOWER)
    fix = (
        f" The `{result.reference} + {result.fp_strategy}` row is the proposed model with the "
        "false positive fix chosen on the folds."
        if result.fp_strategy != BEFORE
        else " No false positive fix beat the plain model on the folds."
    )
    return "\n".join([
        "# Results", "",
        f"Proposed model: `{result.reference}`. Balancing `{result.balancing}` for every model "
        "that trains. CV accuracy is over the folds of the training part; Val Acc is the final "
        "model on the validation part; the other columns are the test part, scored once."
        + fix, "",
        f"p-value vs ours: paired t-test on the fold F1 scores. Beats baseline? is Y when "
        f"`{result.reference}` has the higher mean CV F1 and p < {spec.alpha}; it is computed, "
        "not written by hand. Bold marks the best value in each column (lowest for FP and FP "
        "Rate).", "",
        markdown(table[RESULT_COLUMNS]),
        "All episodes are synthetic, so these numbers compare the methods on the generator's "
        "data and do not measure performance on real attack campaigns.", "",
    ])  # fmt: skip


def write_reports(result: CrossvalResult, spec: CrossvalSpec, out_dir: Path) -> dict[str, Path]:
    reports = Path(out_dir) / REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    frame = results_frame(result)
    texts = {
        "imbalance": render_imbalance(result), "cross_validation": render_cv(result, spec),
        "false_positives": render_false_positives(result, spec),
        "novelty": render_novelty(result, spec),  "results_table": render_results(frame, result, spec)
    }
    paths = {}
    for name, text in texts.items():
        paths[name] = reports / f"{name}.md"
        paths[name].write_text(text, encoding="utf-8")
    paths["results_csv"] = reports / "results_table.csv"
    frame.to_csv(paths["results_csv"], index=False)
    return paths
