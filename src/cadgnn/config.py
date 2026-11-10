"""Run configuration loaded from YAML.

Relative paths in a config file are resolved against the file's own directory
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
FUSION_METHODS = ("gated", "concat")
ARCHITECTURES = ("dual_stream", "chained")
THRESHOLD_STRATEGIES = ("f1", "fpr")

class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PathsConfig:
    datasets_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class GeneratorConfig:
    """Scenario sizes for the random anomaly generator.
    """

    n_vms: int = 12000
    n_storage: int = 300
    n_network: int = 200
    observation_hours: float = 72.0
    benign_backups_per_vm: int = 6
    n_coincidental_episodes: int = 100
    coincidental_vms: tuple[int, int] = (3, 6)
    n_coordinated_episodes: int = 50
    coordinated_chain_length: tuple[int, int] = (4, 8)
    hop_delay_minutes: tuple[float, float] = (5.0, 45.0)
    # Campaigns start no later than observation_hours minus this margin.
    campaign_start_margin_hours: float = 2.0

    # Overlap between the two classes. All off here, which reproduces the notebook;
    # configs/default.yaml turns them on.
    # Share of coincidental episodes in a short window on unrelated networks.
    coincidental_burst_fraction: float = 0.0
    # Share on a single network but spread over the whole observation period.
    coincidental_cluster_fraction: float = 0.0
    # Share on a single network and in a short window, e.g. a patch rollout.
    coincidental_lookalike_fraction: float = 0.0
    coincidental_window_hours: tuple[float, float] = (0.5, 6.0)
    # Share of campaigns that move slowly to stay under the radar.
    stealth_campaign_fraction: float = 0.0
    stealth_hop_delay_minutes: tuple[float, float] = (120.0, 720.0)
    # Chance that a hop goes through a shared storage pool instead of the network.
    storage_pivot_probability: float = 0.0
    # Chance that the anomaly detector misses a single event.
    missed_event_probability: float = 0.0
    # Chance that one unrelated VM gets grouped into an episode by mistake.
    stray_vm_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in ("n_vms", "n_storage", "n_network"):
            _check_int(getattr(self, name), name, minimum=1)
        for name in ("benign_backups_per_vm", "n_coincidental_episodes", "n_coordinated_episodes"):
            _check_int(getattr(self, name), name, minimum=0)
        _check_number(self.observation_hours, "observation_hours")
        if self.observation_hours <= 0:
            raise ConfigError("observation_hours must be positive")
        _check_number(self.campaign_start_margin_hours, "campaign_start_margin_hours")
        if not 0 <= self.campaign_start_margin_hours < self.observation_hours:
            raise ConfigError("campaign_start_margin_hours must be in [0, observation_hours)")

        # Frozen dataclass: normalise list values from YAML to tuples via object.__setattr__.
        object.__setattr__(
            self, "coincidental_vms", _check_range(self.coincidental_vms, "coincidental_vms", 1)
        )
        object.__setattr__(
            self,
            "coordinated_chain_length",
            _check_range(self.coordinated_chain_length, "coordinated_chain_length", 1),
        )
        object.__setattr__(
            self,
            "hop_delay_minutes",
            _check_range(self.hop_delay_minutes, "hop_delay_minutes", 0, integer=False),
        )
        # Coincidental VMs are drawn without replacement.
        largest = self.coincidental_vms[1]
        if self.n_coincidental_episodes and largest > self.n_vms:
            raise ConfigError(f"coincidental_vms upper bound {largest} exceeds n_vms {self.n_vms}")

        for name in (
            "coincidental_burst_fraction", "coincidental_cluster_fraction",
            "coincidental_lookalike_fraction", "stealth_campaign_fraction",
            "storage_pivot_probability", "missed_event_probability", "stray_vm_probability",
        ):  # fmt: skip
            _check_number(getattr(self, name), name)
            if not 0 <= getattr(self, name) <= 1:
                raise ConfigError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        shares = (
            self.coincidental_burst_fraction
            + self.coincidental_cluster_fraction
            + self.coincidental_lookalike_fraction
        )
        if shares > 1:
            raise ConfigError(f"coincidental episode fractions add up to {shares}, more than 1")
        for name in ("coincidental_window_hours", "stealth_hop_delay_minutes"):
            object.__setattr__(
                self, name, _check_range(getattr(self, name), name, 0, integer=False)
            )
        if self.coincidental_window_hours[1] > self.observation_hours:
            raise ConfigError("coincidental_window_hours must fit inside observation_hours")


@dataclass(frozen=True)
class GraphConfig:
    # A single flagged event cannot be judged coordinated or coincidental.
    min_episode_events: int = 2

    def __post_init__(self) -> None:
        _check_int(self.min_episode_events, "min_episode_events", minimum=1)


@dataclass(frozen=True)
class ModelConfig:
    # dual_stream is the proposed model. chained is the CONTINUUM-style baseline,
    # a GAT on each snapshot followed by a GRU; it ignores the stream and fusion settings.
    architecture: str = "dual_stream"
    hidden_dim: int = 64
    temporal_dim: int = 64
    spatial_dim: int = 64
    temporal_layers: int = 1
    spatial_layers: int = 2
    spatial_heads: int = 2
    # Turning a stream off is how the single-stream baselines are run.
    use_temporal: bool = True
    use_spatial: bool = True
    fusion: str = "gated"
    fusion_dim: int = 64
    dropout: float = 0.1
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim", "temporal_dim", "spatial_dim",
            "temporal_layers", "spatial_layers", "spatial_heads", "fusion_dim",
        ):  # fmt: skip
            _check_int(getattr(self, name), name, minimum=1)
        for name in ("use_temporal", "use_spatial"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be true or false, got {getattr(self, name)!r}")
        if self.architecture not in ARCHITECTURES:
            choices = ", ".join(ARCHITECTURES)
            raise ConfigError(f"architecture must be one of {choices}, got {self.architecture!r}")
        if not (self.use_temporal or self.use_spatial):
            raise ConfigError("at least one of use_temporal and use_spatial must be true")
        if self.fusion not in FUSION_METHODS:
            methods = ", ".join(FUSION_METHODS)
            raise ConfigError(f"fusion must be one of {methods}, got {self.fusion!r}")
        _check_number(self.dropout, "dropout")
        if not 0 <= self.dropout < 1:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if not isinstance(self.device, str) or not (
            self.device in ("auto", "cpu", "cuda") or self.device.startswith("cuda:")
        ):
            raise ConfigError(f"device must be auto, cpu, cuda or cuda:N, got {self.device!r}")


@dataclass(frozen=True)
class TrainingConfig:
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Stop when validation loss has not improved for this many epochs.
    patience: int = 15
    # f1: best F1 on validation. fpr: best recall with validation FPR <= max_fpr.
    threshold_strategy: str = "f1"
    max_fpr: float = 0.1

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            _check_int(getattr(self, name), name, minimum=1)
        for name in ("val_fraction", "test_fraction", "learning_rate", "weight_decay", "max_fpr"):
            _check_number(getattr(self, name), name)
        if not (0 < self.val_fraction < 1 and 0 < self.test_fraction < 1):
            raise ConfigError("val_fraction and test_fraction must be between 0 and 1")
        if self.val_fraction + self.test_fraction >= 1:
            raise ConfigError("val_fraction + test_fraction must leave some training data")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ConfigError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.max_fpr <= 1:
            raise ConfigError(f"max_fpr must be in [0, 1], got {self.max_fpr}")
        if self.threshold_strategy not in THRESHOLD_STRATEGIES:
            strategies = ", ".join(THRESHOLD_STRATEGIES)
            raise ConfigError(f"threshold_strategy must be one of {strategies}")


@dataclass(frozen=True)
class Config:
    seed: int
    log_level: str
    paths: PathsConfig
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


DEFAULTS: dict[str, Any] = {
    "seed": 8304,
    "log_level": "INFO",
    "paths": {
        "datasets_dir": "datasets",
        "output_dir": "output",
    },
    "generator": {f.name: f.default for f in fields(GeneratorConfig)},
    "graph": {f.name: f.default for f in fields(GraphConfig)},
    "model": {f.name: f.default for f in fields(ModelConfig)},
    "training": {f.name: f.default for f in fields(TrainingConfig)},
}


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return parse_config({}, base_dir=Path.cwd())

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return parse_config(raw, base_dir=path.resolve().parent)


def parse_config(raw: dict[str, Any], base_dir: Path) -> Config:
    data = _merge(DEFAULTS, raw, where="config")
    paths = data["paths"]
    return Config(
        seed=_seed(data["seed"]),
        log_level=_log_level(data["log_level"]),
        paths=PathsConfig(
            datasets_dir=_path(paths["datasets_dir"], base_dir, "paths.datasets_dir"),
            output_dir=_path(paths["output_dir"], base_dir, "paths.output_dir"),
        ),
        generator=GeneratorConfig(**data["generator"]),
        graph=GraphConfig(**data["graph"]),
        model=ModelConfig(**data["model"]),
        training=TrainingConfig(**data["training"]),
    )


def with_overrides(
    config: Config,
    *,
    seed: int | None = None,  log_level: str | None = None,
    datasets_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Config:
    """Apply command-line overrides; override paths are relative to the working directory."""
    paths = config.paths
    if datasets_dir is not None:
        paths = replace(paths, datasets_dir=_path(datasets_dir, Path.cwd(), "datasets_dir"))
    if output_dir is not None:
        paths = replace(paths, output_dir=_path(output_dir, Path.cwd(), "output_dir"))
    return replace(
        config,
        seed=config.seed if seed is None else _seed(seed),
        log_level=config.log_level if log_level is None else _log_level(log_level),
        paths=paths,
    )


def _merge(defaults: dict[str, Any], raw: dict[str, Any], where: str) -> dict[str, Any]:
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(map(str, unknown))}")

    merged = {}
    for key, default in defaults.items():
        value = raw.get(key, default)
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must be a mapping")
            value = _merge(default, value, where=key)
        merged[key] = value
    return merged


def _seed(value: Any) -> int:
    # bool is a subclass of int, but `seed: true` is almost certainly a mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"seed must be an integer, got {value!r}")
    if value < 0:
        raise ConfigError(f"seed must be non-negative, got {value}")
    return value


def _check_int(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}")


def _check_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")


def _check_range(value: Any, name: str, minimum: float, integer: bool = True) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must be a [low, high] pair, got {value!r}")
    low, high = value
    for bound in (low, high):
        if integer:
            _check_int(bound, name, minimum)
        else:
            _check_number(bound, name)
            if bound < minimum:
                raise ConfigError(f"{name} values must be at least {minimum}, got {bound}")
    if low > high:
        raise ConfigError(f"{name} low bound {low} is greater than high bound {high}")
    return (low, high)


def _log_level(value: Any) -> str:
    if not isinstance(value, str) or value.upper() not in LOG_LEVELS:
        raise ConfigError(f"log_level must be one of {', '.join(LOG_LEVELS)}, got {value!r}")
    return value.upper()


def _path(value: Any, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"{name} must be a non-empty path, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def display_path(path: str | Path) -> str:
    """Path relative to the working directory when it lies inside it.

    Keeps the local folder layout (user name, drive) out of logs and saved results.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def describe(config: Config) -> str:
    return (
        f"seed={config.seed} log_level={config.log_level} "
        f"datasets_dir={display_path(config.paths.datasets_dir)} "
        f"output_dir={display_path(config.paths.output_dir)}"
    )
