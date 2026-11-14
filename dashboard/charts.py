"""Matplotlib charts for the dashboard. Each function returns a figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import BLUE, COLOURS, GREY, GROUP_NAMES, ORANGE, name
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, roc_curve

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 11,

    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,

    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,

    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,


    "legend.frameon": False,
})


def _legend(ax, groups) -> None:
    present = [g for g in GROUP_NAMES if g in set(groups)]
    ax.legend(handles=[Patch(color=COLOURS[g], label=GROUP_NAMES[g]) for g in present],
              loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=3, fontsize=8)


def model_bars(frame: pd.DataFrame, column: str, title: str, lower_is_better=False,
               percent=False, error: str | None = None):
    """Horizontal bars, one per model, best at the top. ``frame`` needs model and group."""
    frame = frame.sort_values(column, ascending=not lower_is_better)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(frame) + 1.0))
    y = np.arange(len(frame))
    colours = [COLOURS[g] for g in frame["group"]]
    ax.barh(y, frame[column], color=colours, height=0.62,
            xerr=frame[error] if error else None,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
    ax.set_yticks(y, [name(m) for m in frame["model"]])
    ax.grid(axis="y", visible=False)
    top = float((frame[column] + (frame[error] if error else 0)).max())
    for i, v in enumerate(frame[column]):
        label = f"{v:.1%}" if percent else (f"{v:.0f}" if float(v).is_integer() else f"{v:.3f}")
        ax.text(v + top * 0.01 + (frame[error].iloc[i] if error else 0), i, label,
                va="center", fontsize=8, color=INK)
    ax.set_xlim(0, top * 1.15 if top > 0 else 1)
    ax.set_title(title)
    _legend(ax, frame["group"])
    fig.tight_layout()
    return fig


def class_split(counts: pd.DataFrame):
    parts = counts[counts["part"].isin(["train", "validation", "test"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(parts))
    ax.bar(x - 0.2, parts["coincidental"], 0.38, color=GREY, label="Coincidental")
    ax.bar(x + 0.2, parts["coordinated"], 0.38, color=BLUE, label="Coordinated")
    for i, (a, b) in enumerate(zip(parts["coincidental"], parts["coordinated"], strict=True)):
        ax.text(i - 0.2, a, f"{a}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, f"{b}", ha="center", va="bottom", fontsize=8)
    labels = {"train": "Train (70%)", "validation": "Validation (20%)", "test": "Test (10%)"}

    ax.set_xticks(x, [labels[p] for p in parts["part"]])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Episodes")
    ax.set_title("Episodes per split")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def pr_and_roc(predictions: pd.DataFrame, models: list[str], colours: dict[str, str]):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    y = predictions["y"].to_numpy()
    for model in models:
        scores = predictions[f"score_{model}"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, scores)
        fpr, tpr, _ = roc_curve(y, scores)
        left.plot(recall, precision, lw=2, color=colours[model], label=name(model))
        right.plot(fpr, tpr, lw=2, color=colours[model], label=name(model))
    right.plot([0, 1], [0, 1], ls="--", lw=1, color=GREY)
    left.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.05))
    right.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1),
              ylim=(0, 1.05))
    left.set_title("Precision-recall curve (test part)")
    right.set_title("ROC curve (test part)")
    left.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def confusion(row: pd.Series):
    cells = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cells, cmap="Blues", vmin=0, vmax=cells.max() * 1.3)
    words = [["True negative", "False alarm"], ["Missed attack", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cells[i, j]}\n{words[i][j]}", ha="center", va="center", fontsize=9,
                    color=INK)

    ax.set_xticks([0, 1], ["Coincidental", "Syncronized"])
    ax.set_yticks([0, 1], ["Coincidental", "Syncronized"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.grid(False)
    ax.set_title(name(row["model"]), fontsize=9)
    fig.tight_layout()
    return fig


def ablation(frame: pd.DataFrame, reference_f1: float):
    """CV F1 of each variant against the proposed model's, with the p-value on the bar."""
    frame = frame.sort_values("theirs_f1")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(frame) + 1.2))
    y = np.arange(len(frame))
    ax.barh(y, frame["theirs_f1"], color=COLOURS["gnn"], height=0.6)
    ax.axvline(reference_f1, color=BLUE, lw=2, label=f"Proposed GNN ({reference_f1:.3f})")
    for i, (f1, p) in enumerate(zip(frame["theirs_f1"], frame["t_test_p_f1"], strict=True)):
        mark = "significant" if p < 0.05 else "could be chance"
        ax.text(max(f1, reference_f1) + 0.004, i, f"{f1:.3f}  (p = {p:.3f}, {mark})",
                va="center", fontsize=8)
    ax.set_yticks(y, [name(m) for m in frame["model"]])
    low = min(frame["theirs_f1"].min(), reference_f1)
    ax.set_xlim(max(0, low - 0.08), 1.0)

    ax.grid(axis="y", visible=False)

    ax.set_xlabel("Mean F1 over the 5 folds")
    ax.set_title("What each part of the model adds")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.18))
    fig.tight_layout()
    return fig


def strategies(frame: pd.DataFrame, chosen: str, labels: dict[str, str]):
    """Mean false positive rate and recall of each false alarm fix over the folds."""
    frame = frame.sort_values("false_positive_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(frame))
    ax.barh(y - 0.2, frame["false_positive_rate"], 0.38, color=ORANGE,
            label="False positive rate")
    ax.barh(y + 0.2, frame["recall"], 0.38, color=BLUE, label="Recall (attacks caught)")
    for i, (fpr, rec) in enumerate(zip(frame["false_positive_rate"], frame["recall"],
                                       strict=True)):
        ax.text(fpr + 0.01, i - 0.2, f"{fpr:.1%}", va="center", fontsize=8)
        ax.text(rec + 0.01, i + 0.2, f"{rec:.1%}", va="center", fontsize=8)
    ax.set_yticks(y, [labels.get(s, s) + ("  (chosen)" if s == chosen else "")
                      for s in frame["strategy"]])
    ax.set_xlim(0, 1.12)

    ax.grid(axis="y", visible=False)


    ax.set_title("False alarm fixes, mean over the folds")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2)
    fig.tight_layout()
    return fig


def patterns(predictions: pd.DataFrame, models: list[str], colours: dict[str, str]):
    """Share of each kind of episode the models flag as coordinated."""
    kinds = sorted(predictions["pattern"].dropna().unique(),
                   key=lambda p: (p in ("fast", "stealth"), p))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(kinds))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        rates = [predictions.loc[predictions["pattern"] == p, f"pred_{model}"].mean()
                 for p in kinds]
        ax.bar(x - 0.4 + width * (k + 0.5), rates, width, color=colours[model],
               label=name(model))
    coincidental = sum(p not in ("fast", "stealth") for p in kinds)
    ax.axvline(coincidental - 0.5, color=GREY, lw=1, ls="--")
    ax.text(coincidental - 0.6, 1.04, "coincidental: lower is better", ha="right", fontsize=8,
            color=MUTED)

    ax.text(coincidental - 0.4, 1.04, "coordinated: higher is better", fontsize=8, color=MUTED)

    ax.set_xticks(x, kinds)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share flagged as coordinated")
    ax.grid(axis="x", visible=False)
    ax.set_title("Which kinds of episode each model flags (test part)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0, 0.6))
    fig.tight_layout()
    return fig


def training_curves(history: list[dict]):
    epochs = [h["epoch"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))

    left.plot(epochs, [h["train_loss"] for h in history], color=BLUE, lw=2, label="Training")
    left.plot(epochs, [h["val_loss"] for h in history], color=ORANGE, lw=2, label="Validation")


    right.plot(epochs, [h["train_acc"] for h in history], color=BLUE, lw=2, label="Training")
    right.plot(epochs, [h["val_acc"] for h in history], color=ORANGE, lw=2, label="Validation")

    best = min(history, key=lambda h: h["val_loss"])["epoch"]
    for ax in (left, right):
        ax.axvline(best, color=GREY, ls="--", lw=1)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    left.text(best, left.get_ylim()[1], " best epoch", va="top", fontsize=8, color=MUTED)
    left.set_title("Loss")
    right.set_title("Accuracy")
    fig.tight_layout()
    return fig
