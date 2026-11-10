"""Experiment specification: seeds x generator scenarios x model variants and baselines.

A scenario changes the generator (fixed values under ``generator`` and/or
ranges under ``randomize`` that are sampled again for every seed). A variant
changes the model or training settings, e.g. switching a stream off or using a
different fusion method. Every variant is trained and evaluated on every
scenario and seed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cadgnn.baselines.classic import BASELINES
from cadgnn.config import Config, ConfigError, GeneratorConfig, ModelConfig, TrainingConfig

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
# Only plain numeric settings can be randomised; ranges like hop_delay_minutes are
# changed through fixed scenario values instead.
RANDOMISABLE = {
    "n_vms": int,
    "n_storage": int,
    "n_network": int,
    "benign_backups_per_vm": int,
    "n_coincidental_episodes": int,
    "n_coordinated_episodes": int,
    "observation_hours": float,
    "campaign_start_margin_hours": float,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    generator: dict[str, Any] = field(default_factory=dict)
    randomize: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    name: str
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    seeds: list[int]
    scenarios: list[Scenario]
    variants: list[Variant]
    # Non-graph baselines from cadgnn.baselines, run on the same split as the variants.
    baselines: list[str] = field(default_factory=list)

    @property
    def reference(self) -> str:
        """The variant every other result is tested against: the first one listed."""
        return self.variants[0].name


def load_spec(path: str | Path, base: Config) -> ExperimentSpec:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"experiment spec not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    return parse_spec(raw, base)


def parse_spec(raw: dict, base: Config) -> ExperimentSpec:
    if not isinstance(raw, dict):
        raise ConfigError("experiment spec must be a mapping")
    unknown = set(raw) - {"seeds", "scenarios", "variants", "baselines"}
    if unknown:
        raise ConfigError(f"unknown key(s) in experiment spec: {', '.join(sorted(unknown))}")

    seeds = raw.get("seeds", [base.seed])
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)
    ):
        raise ConfigError("seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("seeds must not repeat")

    scenarios = [
        Scenario(name, **_section(body, name, {"generator", "randomize"}))
        for name, body in _named(raw.get("scenarios", {"default": {}}), "scenarios")
    ]
    variants = [
        Variant(name, **_section(body, name, {"model", "training"}))
        for name, body in _named(raw.get("variants", {"default": {}}), "variants")
    ]
    baselines = raw.get("baselines", [])
    if not isinstance(baselines, list) or len(set(baselines)) != len(baselines):
        raise ConfigError("baselines must be a list of names without repeats")
    for name in baselines:
        if name not in BASELINES:
            raise ConfigError(f"unknown baseline {name!r}, choose from {', '.join(BASELINES)}")
    clash = set(baselines) & {v.name for v in variants}
    if clash:
        raise ConfigError(f"variant name(s) {', '.join(sorted(clash))} are taken by baselines")
    spec = ExperimentSpec(seeds, scenarios, variants, baselines)

    # Build every configuration once now, so a typo fails before hours of training.
    for scenario in scenarios:
        for seed in seeds:
            scenario_config(base.generator, scenario, seed)
    for variant in variants:
        variant_config(base, variant, seeds[0])
    return spec


def scenario_config(base: GeneratorConfig, scenario: Scenario, seed: int) -> GeneratorConfig:
    values = {**asdict(base), **scenario.generator}
    rng = np.random.default_rng([seed, _stable_hash(scenario.name)])
    for key, bounds in sorted(scenario.randomize.items()):
        if key not in RANDOMISABLE:
            raise ConfigError(
                f"scenario {scenario.name}: {key} cannot be randomised, "
                f"choose from {', '.join(RANDOMISABLE)}"
            )
        if not (isinstance(bounds, list) and len(bounds) == 2 and bounds[0] <= bounds[1]):
            raise ConfigError(f"scenario {scenario.name}: randomize.{key} must be [low, high]")
        low, high = bounds
        if RANDOMISABLE[key] is int:
            values[key] = int(rng.integers(low, high + 1))
        else:
            values[key] = float(rng.uniform(low, high))
    try:
        return GeneratorConfig(**values)
    except TypeError as exc:
        raise ConfigError(f"scenario {scenario.name}: {exc}") from exc


def variant_config(base: Config, variant: Variant, seed: int) -> Config:
    try:
        model = ModelConfig(**{**asdict(base.model), **variant.model})
        training = TrainingConfig(**{**asdict(base.training), **variant.training})
    except TypeError as exc:
        raise ConfigError(f"variant {variant.name}: {exc}") from exc
    return replace(base, seed=seed, model=model, training=training)


def _named(section, where: str):
    if not isinstance(section, dict) or not section:
        raise ConfigError(f"{where} must be a non-empty mapping of name -> settings")
    for name, body in section.items():
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise ConfigError(f"{where} name {name!r} may only use letters, digits, _ and -")
        yield name, body or {}


def _section(body, name: str, allowed: set[str]) -> dict:
    if not isinstance(body, dict):
        raise ConfigError(f"{name} must be a mapping")
    unknown = set(body) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in {name}: {', '.join(sorted(unknown))}")
    for key, value in body.items():
        if not isinstance(value, dict):
            raise ConfigError(f"{name}.{key} must be a mapping")
    return dict(body)


def _stable_hash(text: str) -> int:
    # Python's hash() changes between runs; this does not.
    return sum((i + 1) * ord(c) for i, c in enumerate(text))
