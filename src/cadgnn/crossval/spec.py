"""Settings of the cross-validation study, read from configs/crossval.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from cadgnn.baselines.classic import BASELINES
from cadgnn.config import Config, ConfigError, ModelConfig, TrainingConfig
from cadgnn.experiments.spec import Variant, _named, _section, variant_config
from cadgnn.training.balance import FEATURE_BALANCING

TUNABLE = {"model": ModelConfig, "training": TrainingConfig}


@dataclass(frozen=True)
class CrossvalSpec:
    folds: int
    # The first variant is the proposed model; the rest are compared with it.
    variants: list[Variant]
    baselines: list[str] = field(default_factory=list)
    # Balancing methods compared on the folds; smote is skipped for the GNN.
    balancing: list[str] = field(default_factory=lambda: ["class_weight"])
    # Values of the proposed model tried on the validation part, per section.
    tuning: dict[str, dict[str, list]] = field(default_factory=dict)
    fp_cost: float = 2.0
    min_precision: float = 0.9
    # A false positive fix may cost at most this much mean recall on the folds.
    max_recall_drop: float = 0.05
    alpha: float = 0.05

    @property
    def reference(self) -> str:
        return self.variants[0].name

    def tuning_grid(self) -> list[dict[str, dict[str, Any]]]:
        """Every combination of the tuning values, as section -> key -> value."""
        keys = [(s, k) for s, values in self.tuning.items() for k in values]
        if not keys:
            return [{}]
        grid = []
        for combo in product(*(self.tuning[s][k] for s, k in keys)):
            overrides: dict[str, dict[str, Any]] = {}
            for (section, key), value in zip(keys, combo, strict=True):
                overrides.setdefault(section, {})[key] = value
            grid.append(overrides)
        return grid


def apply_overrides(config: Config, overrides: dict[str, dict[str, Any]]) -> Config:
    try:
        model = ModelConfig(**{**asdict(config.model), **overrides.get("model", {})})
        training = TrainingConfig(**{**asdict(config.training), **overrides.get("training", {})})
    except TypeError as exc:
        raise ConfigError(f"tuning: {exc}") from exc
    return replace(config, model=model, training=training)


def load_crossval_spec(path: str | Path, base: Config) -> CrossvalSpec:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"cross-validation spec not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    return parse_crossval_spec(raw, base)


def parse_crossval_spec(raw: dict, base: Config) -> CrossvalSpec:
    if not isinstance(raw, dict):
        raise ConfigError("cross-validation spec must be a mapping")
    allowed = {"folds", "variants", "baselines", "balancing", "tuning", "false_positives", "alpha"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in cross-validation spec: {', '.join(sorted(unknown))}")

    folds = raw.get("folds", 5)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ConfigError("folds must be an integer of at least 2")

    variants = [
        Variant(name, **_section(body, name, {"model", "training"}))
        for name, body in _named(raw.get("variants", {"proposed": {}}), "variants")
    ]
    baselines = raw.get("baselines", [])
    if not isinstance(baselines, list) or any(b not in BASELINES for b in baselines):
        raise ConfigError(f"baselines must be a list taken from {', '.join(BASELINES)}")
    clash = set(baselines) & {v.name for v in variants}
    if clash:
        raise ConfigError(f"variant name(s) {', '.join(sorted(clash))} are taken by baselines")

    balancing = raw.get("balancing", [base.training.balancing])
    if (
        not isinstance(balancing, list)
        or not balancing
        or any(b not in FEATURE_BALANCING for b in balancing)
        or len(set(balancing)) != len(balancing)
    ):
        raise ConfigError(f"balancing must be a list taken from {', '.join(FEATURE_BALANCING)}")
    if balancing == ["smote"]:
        raise ConfigError("balancing needs at least one method the GNN can use")

    tuning = raw.get("tuning", {}) or {}
    if not isinstance(tuning, dict) or set(tuning) - set(TUNABLE):
        raise ConfigError("tuning must map model and/or training to setting -> list of values")
    for section, values in tuning.items():
        if not isinstance(values, dict) or not all(
            isinstance(v, list) and v for v in values.values()
        ):
            raise ConfigError(f"tuning.{section} must map each setting to a non-empty list")

    fp = raw.get("false_positives", {}) or {}
    if not isinstance(fp, dict) or set(fp) - {"fp_cost", "min_precision", "max_recall_drop"}:
        raise ConfigError("false_positives takes fp_cost, min_precision and max_recall_drop")
    alpha = raw.get("alpha", 0.05)
    spec = CrossvalSpec(
        folds=folds, variants=variants, baselines=baselines, balancing=balancing,
        tuning=tuning, alpha=alpha, **fp,
    )  # fmt: skip
    for name in ("fp_cost", "min_precision", "max_recall_drop", "alpha"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"{name} must be a non-negative number")
    if not 0 < spec.alpha < 1:
        raise ConfigError("alpha must be between 0 and 1")

    # Build every configuration now, so a typo fails before hours of training.
    for overrides in spec.tuning_grid():
        tuned = apply_overrides(base, overrides)
        for variant in variants:
            variant_config(tuned, variant, base.seed)
    apply_overrides(base, {"training": {"fp_cost": spec.fp_cost}})
    return spec
