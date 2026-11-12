"""Cross-validation study of the proposed model, its ablations and the baselines
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData

from cadgnn.baselines.classic import fit_baseline
from cadgnn.config import Config, TrainingConfig
from cadgnn.crossval.spec import CrossvalSpec, apply_overrides
from cadgnn.crossval.stats import mcnemar, paired_t_p, wilcoxon_p
from cadgnn.evaluation import classification_metrics, predict, select_threshold
from cadgnn.evaluation.calibration import brier_score, fit_calibrator
from cadgnn.experiments.spec import variant_config
from cadgnn.fusion import build_coordination_model
from cadgnn.generator import TELEMETRY_FILE
from cadgnn.graph import load_episode_graphs
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE
from cadgnn.models import resolve_device, seed_everything
from cadgnn.training import save_split, stratified_split, train_model

log = logging.getLogger(__name__)

CROSSVAL_DIR = Path("crossval")
SPLITS_DIR = Path("splits")
FEATURE_MODELS = ("logistic_regression", "random_forest")
# "default" is the model as it is: its own threshold strategy from the config.
BEFORE = "default"
FP_STRATEGIES = (
    BEFORE, "threshold_0.5", "threshold_precision", "platt", "isotonic", "cost_sensitive",
)  # fmt: skip
METRIC_COLUMNS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate", "tp", "fp", "tn", "fn",
)  # fmt: skip


@dataclass
class Run:
    """Scores of one fitted model on the validation part and on the part it is tested on."""

    val_y: np.ndarray
    val_scores: np.ndarray
    y: np.ndarray
    scores: np.ndarray
    history: list[dict] | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    options: dict = field(default_factory=dict)


@dataclass
class CrossvalResult:
    reference: str
    balancing: str
    fp_strategy: str
    tuned: dict
    class_counts: pd.DataFrame
    event_counts: pd.DataFrame | None
    tuning: pd.DataFrame
    folds: pd.DataFrame
    test: pd.DataFrame
    significance: pd.DataFrame
    predictions: pd.DataFrame
    histories: list[list[dict]]
    # What the learning curves of the feature baselines need.
    learning: dict
    paths: dict[str, Path] = field(default_factory=dict)


def labels_of(graphs: Sequence[HeteroData]) -> np.ndarray:
    return np.array([int(g.y) for g in graphs])


def stratified_folds(
    graphs: Sequence[HeteroData], positions: Sequence[int], k: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """k folds over ``positions``; each item is (training positions, held-out positions)."""
    positions = np.asarray(positions)
    y = labels_of([graphs[i] for i in positions])
    if np.bincount(y, minlength=2).min() < k:
        raise ValueError(f"every class needs at least {k} training episodes for {k} folds")
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        (sorted(positions[a].tolist()), sorted(positions[b].tolist()))
        for a, b in splitter.split(positions, y)
    ]


def metric_row(metrics: dict) -> dict:
    row = {name: metrics[name] for name in METRIC_COLUMNS if name in metrics}
    row.update(metrics["confusion_matrix"])
    return row


def decide(
    run: Run, training: TrainingConfig, strategy: str | None = None
) -> tuple[float, dict]:
    threshold = select_threshold(
        run.val_y, run.val_scores, strategy or training.threshold_strategy,
        training.max_fpr, training.min_precision,
    )  # fmt: skip
    return threshold, classification_metrics(run.y, run.scores, threshold)


def class_counts(graphs, split, folds) -> pd.DataFrame:
    parts = {"all": range(len(graphs)), "train": split.train, "validation": split.val,
             "test": split.test}  # fmt: skip
    for k, (_, held) in enumerate(folds, start=1):
        parts[f"fold {k}"] = held
    rows = []
    for name, idx in parts.items():
        y = labels_of([graphs[i] for i in idx])
        counts = np.bincount(y, minlength=2)
        rows.append({
            "part": name, "coincidental": int(counts[0]), "coordinated": int(counts[1]),
            "total": int(counts.sum()), "coordinated_share": counts[1] / counts.sum(),
            "imbalance_ratio": counts.max() / counts.min() if counts.min() else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def event_counts(graphs_file: Path) -> pd.DataFrame | None:
    """Label counts of the raw events, if the generator output sits next to the graphs."""
    path = graphs_file.parent / TELEMETRY_FILE
    if not path.is_file():
        return None
    counts = pd.read_csv(path, usecols=["label"])["label"].value_counts()
    frame = counts.rename_axis("label").reset_index(name="events")
    frame["share"] = frame["events"] / frame["events"].sum()
    return frame


class Study:
    def __init__(self, config: Config, spec: CrossvalSpec, graphs: Sequence[HeteroData]):
        self.config, self.spec, self.graphs = config, spec, graphs
        tc = config.training
        self.split = stratified_split(graphs, tc.val_fraction, tc.test_fraction, config.seed)
        self.folds = stratified_folds(graphs, self.split.train, spec.folds, config.seed)
        self.train = self.pick(self.split.train)
        self.val = self.pick(self.split.val)
        self.test = self.pick(self.split.test)
        self.device = resolve_device(config.model.device)
        self.rows: list[dict] = []
        self.test_rows: list[dict] = []
        self.predictions = pd.DataFrame({
            "episode_id": [g.episode_id for g in self.test], "y": labels_of(self.test),
            "pattern": [getattr(g, "pattern", "unknown") for g in self.test],
        })  # fmt: skip
        self.cv_runs: dict[str, list[Run]] = {}
        self.trainings_done = 0
        grid = spec.tuning_grid()
        gnn_methods = [m for m in spec.balancing if m != "smote"]
        self.trainings_planned = (
            (len(grid) if len(grid) > 1 else 0)
            + spec.folds * (len(gnn_methods) + len(spec.variants) - 1 + 1)
            + len(spec.variants) + 1
        )

    def pick(self, positions) -> list[HeteroData]:
        return [self.graphs[i] for i in positions]

    def fit_gnn(self, config: Config, train, scored, what: str) -> Run:
        self.trainings_done += 1
        log.info("training %d of %d: %s", self.trainings_done, self.trainings_planned, what)
        seed_everything(config.seed)
        model = build_coordination_model(config.model)
        result = train_model(model, train, self.val, config.training, config.seed, self.device)
        val_pred = predict(model, self.val, device=self.device)
        pred = predict(model, scored, device=self.device)
        return Run(val_pred.y, val_pred.scores, pred.y, pred.scores,
                   result.history, result.best_epoch, result.best_val_loss)  # fmt: skip

    def fit_baseline(self, name: str, train, scored, balancing: str) -> Run:
        fitted = fit_baseline(name, train, self.val, self.config.seed, balancing)
        return Run(labels_of(self.val), fitted.score(self.val), labels_of(scored),
                   fitted.score(scored), options=fitted.options)  # fmt: skip

    def record(self, study, model, kind, balancing, fold, run, training, strategy=None):
        threshold, metrics = decide(run, training)
        self.rows.append({
            "study": study, "model": model, "kind": kind, "balancing": balancing,
            "strategy": strategy or training.threshold_strategy, "fold": fold,
            "threshold": threshold, **metric_row(metrics),
            "best_epoch": run.best_epoch, "best_val_loss": run.best_val_loss,
        })  # fmt: skip

    def mean(self, study: str, model: str, column: str, **match) -> float:
        rows = [r for r in self.rows if r["study"] == study and r["model"] == model
                and all(r.get(k) == v for k, v in match.items())]  # fmt: skip
        values = [r[column] for r in rows if r[column] is not None]
        return float(np.mean(values)) if values else np.nan

    # Step 3
    def tune(self) -> tuple[dict, pd.DataFrame]:
        grid = self.spec.tuning_grid()
        reference = self.spec.variants[0]
        if len(grid) == 1:
            return grid[0], pd.DataFrame()
        rows = []
        for overrides in grid:
            config = variant_config(apply_overrides(self.config, overrides), reference,
                                    self.config.seed)  # fmt: skip
            run = self.fit_gnn(config, self.train, self.val, f"tuning {overrides}")
            threshold, metrics = decide(run, config.training)
            rows.append({
                "settings": json.dumps(overrides), "val_roc_auc": metrics["roc_auc"],
                "val_f1": metrics["f1"], "best_val_loss": run.best_val_loss,
                "best_epoch": run.best_epoch,
            })  # fmt: skip
        table = pd.DataFrame(rows)
        best = table.sort_values(["val_roc_auc", "best_val_loss"], ascending=[False, True])
        table["chosen"] = table.index == best.index[0]
        return grid[int(best.index[0])], table

    # Step 4
    def compare_balancing(self, base: Config) -> str:
        reference = self.spec.variants[0]
        for method in self.spec.balancing:
            if method != "smote":
                config = variant_config(base, reference, self.config.seed)
                config = replace(config, training=replace(config.training, balancing=method))
                runs = []
                for k, (train, held) in enumerate(self.folds, start=1):
                    what = f"{reference.name}, balancing {method}, fold {k}"
                    run = self.fit_gnn(config, self.pick(train), self.pick(held), what)
                    self.record("balancing", reference.name, "gnn", method, k, run,
                                config.training)  # fmt: skip
                    runs.append(run)
                self.cv_runs[f"{reference.name}/{method}"] = runs
            for name in self.spec.baselines:
                if name not in FEATURE_MODELS:
                    continue
                runs = []
                for k, (train, held) in enumerate(self.folds, start=1):
                    run = self.fit_baseline(name, self.pick(train), self.pick(held), method)
                    self.record("balancing", name, "baseline", method, k, run, base.training)
                    runs.append(run)
                self.cv_runs[f"{name}/{method}"] = runs

        candidates = [m for m in self.spec.balancing if m != "smote"]
        return min(candidates, key=lambda m: (
            -self.mean("balancing", reference.name, "f1", balancing=m),
            self.mean("balancing", reference.name, "false_positive_rate", balancing=m),
        ))  # fmt: skip

    # Step 5
    def crossvalidate(self, base: Config, balancing: str) -> None:
        for i, variant in enumerate(self.spec.variants):
            config = variant_config(base, variant, self.config.seed)
            if i == 0:
                runs = self.cv_runs[f"{variant.name}/{balancing}"]
            else:
                runs = [
                    self.fit_gnn(config, self.pick(train), self.pick(held),
                                 f"{variant.name}, fold {k}")
                    for k, (train, held) in enumerate(self.folds, start=1)
                ]  # fmt: skip
                self.cv_runs[f"{variant.name}/{balancing}"] = runs
            for k, run in enumerate(runs, start=1):
                self.record("cv", variant.name, "gnn", balancing, k, run, config.training)
        for name in self.spec.baselines:
            if name in FEATURE_MODELS:
                runs = self.cv_runs[f"{name}/{balancing}"]
            else:
                runs = [self.fit_baseline(name, self.pick(train), self.pick(held), balancing)
                        for train, held in self.folds]  # fmt: skip
            used = balancing if name in FEATURE_MODELS else "n/a"
            for k, run in enumerate(runs, start=1):
                self.record("cv", name, "baseline", used, k, run, base.training)

    def strategy_metrics(
        self, run: Run, strategy: str, training: TrainingConfig
    ) -> tuple[dict, np.ndarray]:
        """Metrics of one false positive strategy, and the scores it thresholds."""
        scores, val_scores = run.scores, run.val_scores
        if strategy in ("platt", "isotonic"):
            calibrate = fit_calibrator(strategy, run.val_y, run.val_scores)
            scores, val_scores = calibrate(scores), calibrate(val_scores)
        if strategy in ("threshold_0.5", "platt", "isotonic"):
            threshold = 0.5
        else:
            name = "precision" if strategy == "threshold_precision" else None
            threshold = select_threshold(
                run.val_y, run.val_scores, name or training.threshold_strategy,
                training.max_fpr, training.min_precision,
            )  # fmt: skip
        found = {
            "threshold": threshold,
            "val_accuracy": classification_metrics(run.val_y, val_scores, threshold)["accuracy"],
            **metric_row(classification_metrics(run.y, scores, threshold)),
            "brier": brier_score(run.y, scores),
        }
        return found, scores

    # Step 6
    def false_positives(self, base: Config, balancing: str) -> str:
        reference = self.spec.variants[0]
        config = variant_config(base, reference, self.config.seed)
        training = replace(config.training, min_precision=self.spec.min_precision)
        costly = replace(config, training=replace(training, fp_cost=self.spec.fp_cost))
        plain = self.cv_runs[f"{reference.name}/{balancing}"]
        for strategy in FP_STRATEGIES:
            for k, (train, held) in enumerate(self.folds, start=1):
                if strategy == "cost_sensitive":
                    run = self.fit_gnn(costly, self.pick(train), self.pick(held),
                                       f"{reference.name}, fp_cost {self.spec.fp_cost}, fold {k}")
                    self.cv_runs.setdefault(f"{reference.name}/cost_sensitive", []).append(run)
                else:
                    run = plain[k - 1]
                self.rows.append({
                    "study": "fp", "model": reference.name, "kind": "gnn",
                    "balancing": balancing, "strategy": strategy, "fold": k,
                    **self.strategy_metrics(run, strategy, training)[0],
                })  # fmt: skip

        def mean(strategy, column):
            return self.mean("fp", reference.name, column, strategy=strategy)

        floor = mean(BEFORE, "recall") - self.spec.max_recall_drop
        allowed = [s for s in FP_STRATEGIES if mean(s, "recall") >= floor - 1e-12]
        return min(allowed, key=lambda s: (mean(s, "false_positive_rate"), -mean(s, "f1")))

    # Step 7
    def final(self, base: Config, balancing: str, fp_strategy: str) -> list[list[dict]]:
        reference = self.spec.variants[0]
        for variant in self.spec.variants:
            config = variant_config(base, variant, self.config.seed)
            run = self.fit_gnn(config, self.train, self.test, f"{variant.name}, final")
            self.test_row(variant.name, "gnn", balancing, run, config.training)
            if variant is reference:
                training = replace(config.training, min_precision=self.spec.min_precision)
                for strategy in dict.fromkeys((BEFORE, fp_strategy)):
                    used = run
                    if strategy == "cost_sensitive":
                        costly = replace(config, training=replace(training,
                                                                  fp_cost=self.spec.fp_cost))
                        used = self.fit_gnn(costly, self.train, self.test,
                                            f"{variant.name}, fp_cost {self.spec.fp_cost}, final")
                    found, scores = self.strategy_metrics(used, strategy, training)
                    self.test_rows.append({
                        "model": f"{variant.name} + {strategy}", "kind": "gnn",
                        "balancing": balancing, "strategy": strategy, **found,
                    })  # fmt: skip
                    self.predictions[f"score_{variant.name} + {strategy}"] = scores
                    self.predictions[f"pred_{variant.name} + {strategy}"] = (
                        scores >= found["threshold"]
                    ).astype(int)
        for name in self.spec.baselines:
            run = self.fit_baseline(name, self.train, self.test, balancing)
            used = balancing if name in FEATURE_MODELS else "n/a"
            self.test_row(name, "baseline", used, run, base.training, options=run.options)
        return [run.history for run in self.cv_runs[f"{reference.name}/{balancing}"]]

    def test_row(self, model, kind, balancing, run, training, options=None) -> None:
        threshold, metrics = decide(run, training)
        val = classification_metrics(run.val_y, run.val_scores, threshold)
        self.test_rows.append({
            "model": model, "kind": kind, "balancing": balancing,
            "strategy": training.threshold_strategy, "threshold": threshold,
            "val_accuracy": val["accuracy"], **metric_row(metrics),
            "best_epoch": run.best_epoch, "options": json.dumps(options or {}),
        })  # fmt: skip
        self.predictions[f"score_{model}"] = run.scores
        self.predictions[f"pred_{model}"] = (run.scores >= threshold).astype(int)

    def significance(self, fp_strategy: str) -> pd.DataFrame:
        reference = self.spec.variants[0].name
        folds = pd.DataFrame(self.rows)
        cv = folds[folds["study"] == "cv"]
        ours = cv[cv["model"] == reference].sort_values("fold")
        y = self.predictions["y"].to_numpy()
        rows = []
        others = [(m, cv[cv["model"] == m].sort_values("fold"), f"pred_{m}")
                  for m in cv["model"].unique() if m != reference]  # fmt: skip
        if fp_strategy != BEFORE:
            fp = folds[(folds["study"] == "fp") & (folds["strategy"] == fp_strategy)]
            fp_name = f"{reference} + {fp_strategy}"
            others.append((fp_name, fp.sort_values("fold"), f"pred_{fp_name}"))
        for model, theirs, column in others:
            row = {"model": model, "reference": reference}
            for metric in ("f1", "roc_auc", "false_positive_rate"):
                a = ours[metric].to_numpy(dtype=float)
                b = theirs[metric].to_numpy(dtype=float)
                row[f"ours_{metric}"] = float(np.nanmean(a))
                row[f"theirs_{metric}"] = float(np.nanmean(b))
                row[f"t_test_p_{metric}"] = paired_t_p(a, b)
            row["wilcoxon_p_f1"] = wilcoxon_p(ours["f1"].to_numpy(float),
                                              theirs["f1"].to_numpy(float))  # fmt: skip
            test = mcnemar(y, self.predictions[f"pred_{reference}"], self.predictions[column])
            row.update({"mcnemar_only_ours": test["only_a"], "mcnemar_only_theirs": test["only_b"],
                        "mcnemar_p": test["p"]})  # fmt: skip
            row["ours_better"] = bool(
                row["ours_f1"] > row["theirs_f1"] and row["t_test_p_f1"] < self.spec.alpha
            )
            rows.append(row)
        return pd.DataFrame(rows)


def curve_summary(histories: list[list[dict]]) -> dict:
    """Losses at each fold's best epoch, averaged over folds."""
    best = [min(h, key=lambda r: r["val_loss"]) for h in histories if h]
    if not best:
        return {}
    train = float(np.mean([b["train_loss"] for b in best]))
    val = float(np.mean([b["val_loss"] for b in best]))
    return {"train_loss": train, "val_loss": val, "gap": val - train,
            "train_acc": float(np.mean([b["train_acc"] for b in best])),
            "val_acc": float(np.mean([b["val_acc"] for b in best])),
            "best_epoch": float(np.mean([b["epoch"] for b in best]))}  # fmt: skip


def run_crossval(
    config: Config, spec: CrossvalSpec, out_dir: Path, graphs_file: Path | None = None
) -> CrossvalResult:
    graphs_file = graphs_file or config.paths.output_dir / EPISODE_GRAPHS_FILE
    out_dir = Path(out_dir)
    graphs = load_episode_graphs(graphs_file)
    study = Study(config, spec, graphs)
    log.info("split: %d train, %d val, %d test; %d folds, about %d trainings",
             len(study.train), len(study.val), len(study.test), spec.folds,
             study.trainings_planned)  # fmt: skip

    paths = {
        "holdout_split": save_split(study.split, graphs, out_dir / SPLITS_DIR / "holdout.json",
                                    seed=config.seed),
        "fold_split": out_dir / SPLITS_DIR / "folds.json",
    }  # fmt: skip
    paths["fold_split"].write_text(json.dumps([
        {"fold": k, "train": [graphs[i].episode_id for i in tr],
         "held_out": [graphs[i].episode_id for i in held]}
        for k, (tr, held) in enumerate(study.folds, start=1)
    ], indent=2), encoding="utf-8")  # fmt: skip

    overrides, tuning = study.tune()
    tuned = apply_overrides(config, overrides)
    balancing = study.compare_balancing(tuned)
    log.info("balancing chosen on the folds: %s", balancing)
    base = replace(tuned, training=replace(tuned.training, balancing=balancing))
    study.crossvalidate(base, balancing)
    fp_strategy = study.false_positives(base, balancing)
    log.info("false positive fix chosen on the folds: %s", fp_strategy)
    histories = study.final(base, balancing, fp_strategy)

    position = {i: n for n, i in enumerate(study.split.train)}
    learning = {
        "train": study.train, "seed": config.seed, "balancing": balancing,
        "folds": [([position[i] for i in tr], [position[i] for i in held])
                  for tr, held in study.folds],
        "options": {r["model"]: json.loads(r["options"]) for r in study.test_rows
                    if r["model"] in FEATURE_MODELS},
    }  # fmt: skip
    result = CrossvalResult(
        reference=spec.reference, balancing=balancing, fp_strategy=fp_strategy,
        tuned=overrides, class_counts=class_counts(graphs, study.split, study.folds),
        event_counts=event_counts(graphs_file), tuning=tuning,
        folds=pd.DataFrame(study.rows), test=pd.DataFrame(study.test_rows),
        significance=study.significance(fp_strategy), predictions=study.predictions,
        histories=histories, learning=learning, paths=paths,
    )  # fmt: skip
    data_dir = out_dir / CROSSVAL_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("class_counts", "tuning", "folds", "test", "significance", "predictions"):
        path = data_dir / f"{name}.csv"
        getattr(result, name).to_csv(path, index=False)
        paths[name] = path
    paths["histories"] = data_dir / "histories.json"
    paths["histories"].write_text(json.dumps(histories), encoding="utf-8")
    return result
