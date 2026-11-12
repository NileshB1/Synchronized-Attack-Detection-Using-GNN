"""Figures of the cross-validation study, written as PNG files
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_curve  # noqa: E402

from cadgnn.baselines.classic import fit_estimator  # noqa: E402
from cadgnn.baselines.features import feature_matrix, labels  # noqa: E402
from cadgnn.crossval.runner import BEFORE, CrossvalResult  # noqa: E402

FIGURES_DIR = Path("figures")
LEARNING_SIZES = (0.2, 0.4, 0.6, 0.8, 1.0)
# Categorical colours in a fixed order, checked for colour blindness.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "font.size": 9, "lines.linewidth": 2, "axes.axisbelow": True,
}  # fmt: skip


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def class_distribution(result: CrossvalResult, path: Path) -> Path:
    row = result.class_counts.set_index("part").loc["all"]
    counts = [row["coincidental"], row["coordinated"]]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["coincidental (0)", "coordinated (1)"], counts, color=[BLUE, ORANGE],
                  width=0.55, edgecolor=SURFACE, linewidth=2)  # fmt: skip
    total = sum(counts)
    for bar, n in zip(bars, counts, strict=True):
        ax.annotate(f"{int(n)} ({100 * n / total:.1f}%)", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                    color=INK)  # fmt: skip
    ax.set_ylabel("episodes")
    ax.set_title(f"Class distribution, imbalance {row['imbalance_ratio']:.2f} : 1")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(counts) * 1.15)
    return _save(fig, path)


def train_val_curves(result: CrossvalResult, path: Path) -> Path:
    histories = [h for h in result.histories if h]
    longest = max(len(h) for h in histories)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        for part, colour in (("train", BLUE), ("val", ORANGE)):
            grid = np.full((len(histories), longest), np.nan)
            for i, history in enumerate(histories):
                values = [r[f"{part}_{metric}"] for r in history]
                grid[i, : len(values)] = values
                ax.plot(range(1, len(values) + 1), values, color=colour, alpha=0.25,
                        linewidth=1)  # fmt: skip
            # The mean only covers epochs that at least half of the folds reached.
            reached = (~np.isnan(grid)).sum(axis=0) >= max(1, len(histories) / 2)
            mean = np.nanmean(grid[:, reached], axis=0)
            name = "training" if part == "train" else "validation"
            ax.plot(np.arange(1, longest + 1)[reached], mean, color=colour,
                    label=f"{name} (mean of folds)")  # fmt: skip
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{title}, {result.reference}, faint lines are single folds")
    axes[0].legend()
    return _save(fig, path)


def learning_curves(result: CrossvalResult, path: Path) -> Path | None:
    info = result.learning
    names = [n for n in ("logistic_regression", "random_forest") if n in info["options"]]
    if not names:
        return None
    x, y = feature_matrix(info["train"]), labels(info["train"])
    rng = np.random.default_rng(info["seed"])
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        train_scores, val_scores = [], []
        for share in LEARNING_SIZES:
            fold_train, fold_val = [], []
            for train, held in info["folds"]:
                train = np.asarray(train)
                if share < 1:
                    # Keep the class ratio when taking a share of the training folds.
                    train = np.concatenate([
                        rng.choice(train[y[train] == c],
                                   max(2, round(share * (y[train] == c).sum())), replace=False)
                        for c in (0, 1)
                    ])  # fmt: skip
                model = fit_estimator(name, info["options"][name], x[train], y[train],
                                      info["seed"], info["balancing"])  # fmt: skip
                fold_train.append(f1_score(y[train], model.predict(x[train])))
                fold_val.append(f1_score(y[held], model.predict(x[held])))
            train_scores.append(fold_train)
            val_scores.append(fold_val)
        sizes = [round(s * np.mean([len(t) for t, _ in info["folds"]])) for s in LEARNING_SIZES]
        for scores, colour, label in ((train_scores, BLUE, "training"),
                                      (val_scores, ORANGE, "validation (held-out fold)")):
            scores = np.asarray(scores)
            mean, std = scores.mean(axis=1), scores.std(axis=1)
            ax.fill_between(sizes, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0)
            ax.plot(sizes, mean, color=colour, marker="o", markersize=5, label=label)
        ax.set_xlabel("training episodes")
        ax.set_ylabel("F1")
        ax.set_title(f"Learning curve, {name} (band: ± std over folds)")
        ax.legend(loc="lower right")
    return _save(fig, path)


def pr_curve(result: CrossvalResult, path: Path) -> Path:
    pred = result.predictions
    test = result.test.set_index("model")
    models = [m for m in test.index if f"score_{m}" in pred and " + " not in m]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for i, model in enumerate(models):
        precision, recall, _ = precision_recall_curve(pred["y"], pred[f"score_{model}"])
        ours = model == result.reference
        ax.plot(recall, precision, color=SERIES[i % len(SERIES)], linewidth=2.5 if ours else 1.2,
                alpha=1 if ours else 0.7, label=model, zorder=3 if ours else 2)  # fmt: skip
    points = dict.fromkeys((BEFORE, result.fp_strategy))
    for strategy, marker in zip(points, ("o", "D"), strict=False):
        row = test.loc[f"{result.reference} + {strategy}"]
        ax.scatter(row["recall"], row["precision"], s=70, marker=marker, color=INK, zorder=4,
                   edgecolor=SURFACE, linewidth=1.5,
                   label=f"{result.reference}, {strategy} (t = {row['threshold']:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on the test part")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def confusion_matrices(result: CrossvalResult, path: Path) -> Path:
    test = result.test.set_index("model")
    shown = list(dict.fromkeys((BEFORE, result.fp_strategy)))
    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 3.8), squeeze=False)
    for ax, strategy in zip(axes[0], shown, strict=True):
        row = test.loc[f"{result.reference} + {strategy}"]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.sum())
        for (i, j), n in np.ndenumerate(matrix):
            dark = n > matrix.sum() * 0.45
            ax.text(j, i, str(n), ha="center", va="center", fontsize=13,
                    color=SURFACE if dark else INK)  # fmt: skip
        ax.set_xticks([0, 1], ["coincidental", "coordinated"])
        ax.set_yticks([0, 1], ["coincidental", "coordinated"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        when = "before" if strategy == BEFORE else "after"
        ax.set_title(f"{when}: {strategy}\nFP {int(row['fp'])}, "
                     f"FP rate {row['false_positive_rate']:.3f}")  # fmt: skip
    return _save(fig, path)


def false_positive_rates(result: CrossvalResult, path: Path) -> Path:
    test = result.test
    rows = test[~test["model"].str.contains(" + ", regex=False)
                | (test["model"] == f"{result.reference} + {result.fp_strategy}")]
    rows = rows.sort_values("false_positive_rate")
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(rows) + 1.2))
    colours = [BLUE if m.startswith(result.reference) else "#9a9893" for m in rows["model"]]
    bars = ax.barh(rows["model"], rows["false_positive_rate"], color=colours, height=0.6,
                   edgecolor=SURFACE, linewidth=2)  # fmt: skip
    for bar, (_, row) in zip(bars, rows.iterrows(), strict=True):
        ax.annotate(f"{row['false_positive_rate']:.3f} ({int(row['fp'])} FP)",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2), va="center",
                    xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlabel("false positive rate on the test part")
    ax.set_xlim(0, max(rows["false_positive_rate"].max() * 1.35, 0.05))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(f"False alarms per model ({result.reference} in blue)")
    return _save(fig, path)


def write_figures(result: CrossvalResult, out_dir: Path) -> dict[str, Path]:
    figures = Path(out_dir) / FIGURES_DIR
    figures.mkdir(parents=True, exist_ok=True)
    makers = {
        "class_distribution": class_distribution, "train_val_curves": train_val_curves,
        "learning_curves": learning_curves, "pr_curve": pr_curve,
        "confusion_matrices": confusion_matrices, "false_positive_rates": false_positive_rates,
    }  # fmt: skip
    paths = {}
    with plt.rc_context(STYLE):
        for name, make in makers.items():
            path = make(result, figures / f"{name}.png")
            if path is not None:
                paths[name] = path
    return paths
