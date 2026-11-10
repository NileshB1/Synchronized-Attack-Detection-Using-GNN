"""Benign, coincidental and coordinated backup events.

Port of sections 5 and 6 of notebooks/random_anomaly_generator_simple.ipynb. All
randomness comes from one numpy Generator and draws happen in the notebook's
order, so the same seed and scenario reproduce the notebook's graph and events.

The notebook's two classes never overlap. The overlap settings in GeneratorConfig
add harder cases on top of it, and make no random draws while they are off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from cadgnn.config import GeneratorConfig
from cadgnn.data.calibration import ParamProfile, weighted_score
from cadgnn.generator.infrastructure import (
    build_infrastructure,
    find_attack_chain,
    vm_nodes,
    vms_on_one_network,
)

log = logging.getLogger(__name__)

LABELS = ("none", "coincidental", "coordinated")
EVENT_COLUMNS = [
    "vm", "timestamp_hours",
    "change_rate", "file_entropy", "backup_size_gb",
    "file_count", "anomaly_score",
    "state", "episode_id",
    "label",
]


@dataclass
class SyntheticDataset:
    graph: nx.Graph
    events: pd.DataFrame
    config: GeneratorConfig
    seed: int


def make_event(
    vm: str,
    timestamp_hours: float,
    state: str,
    episode_id: str | None,
    label: str,
    profile: ParamProfile,
    rng: np.random.Generator,
) -> dict:
    values = {name: float(rng.uniform(*feature.bounds(state))) for name, feature in profile.items()}
    # The score uses unrounded values; only the stored values are rounded.
    score = weighted_score(profile, values)
    return {
        "vm": vm,  "timestamp_hours": round(timestamp_hours, 3),
        **{name: round(value, 4) for name, value in values.items()},
        "anomaly_score": round(score, 4),
        "state": state,  "episode_id": episode_id,
        "label": label,
    }


def coincidental_pattern(config: GeneratorConfig, rng: np.random.Generator) -> str:
    """How one coincidental episode is laid out. Scattered is the notebook's behaviour."""
    shares = {
        "burst": config.coincidental_burst_fraction,
        "cluster": config.coincidental_cluster_fraction,
        "lookalike": config.coincidental_lookalike_fraction,
    }
    # No draw when everything is off, so the notebook's random sequence is kept.
    if not any(shares.values()):
        return "scattered"
    draw = rng.random()
    for pattern, share in shares.items():
        if draw < share:
            return pattern
        draw -= share
    return "scattered"


def observed_events(
    episode: list[dict],
    vms: np.ndarray,
    profile: ParamProfile,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """What the anomaly detector reports for an episode: some events lost, maybe one extra."""
    if config.missed_event_probability:
        keep = rng.random(len(episode)) >= config.missed_event_probability
        # Two events are the least that can still be judged as an episode.
        if keep.sum() < 2:
            keep[:2] = True
        episode = [event for event, kept in zip(episode, keep, strict=True) if kept]

    if config.stray_vm_probability and rng.random() < config.stray_vm_probability:
        vm = str(rng.choice(vms))
        if vm not in {event["vm"] for event in episode}:
            times = [event["timestamp_hours"] for event in episode]
            t = float(rng.uniform(min(times), max(times)))
            first = episode[0]
            episode = [
                *episode,
                make_event(vm, t, "infected", first["episode_id"], first["label"], profile, rng),
            ]
    return episode


def generate_events(
    graph: nx.Graph, profile: ParamProfile, config: GeneratorConfig, rng: np.random.Generator
) -> pd.DataFrame:
    vms = vm_nodes(graph)
    if not vms:
        raise ValueError("graph has no VM nodes")
    if config.n_coincidental_episodes and config.coincidental_vms[1] > len(vms):
        raise ValueError(
            f"coincidental episodes need up to {config.coincidental_vms[1]} VMs, "
            f"graph has {len(vms)}"
        )
    vm_array = np.array(vms)
    hours = config.observation_hours
    events = []

    for vm in vms:
        for t in rng.uniform(0, hours, size=config.benign_backups_per_vm):
            events.append(make_event(vm, t, "benign", None, "none", profile, rng))

    low, high = config.coincidental_vms
    for ep in range(config.n_coincidental_episodes):
        episode_id = f"coincidental_{ep}"
        size = int(rng.integers(low, high + 1))
        pattern = coincidental_pattern(config, rng)
        if pattern in ("cluster", "lookalike"):
            chosen = vms_on_one_network(graph, vm_array, size, rng)
        else:
            chosen = rng.choice(vm_array, size=size, replace=False)
        start, span = 0.0, hours
        if pattern in ("burst", "lookalike"):
            span = float(rng.uniform(*config.coincidental_window_hours))
            start = float(rng.uniform(0, hours - span))
        episode = []
        for vm in chosen:
            t = start + float(rng.uniform(0, span))
            episode.append(
                make_event(str(vm), t, "infected", episode_id, "coincidental", profile, rng)
            )
        events += observed_events(episode, vm_array, profile, config, rng)

    low, high = config.coordinated_chain_length
    latest_start = hours - config.campaign_start_margin_hours
    short_chains = 0
    for ep in range(config.n_coordinated_episodes):
        episode_id = f"coordinated_{ep}"
        length = int(rng.integers(low, high + 1))
        start_vm = str(rng.choice(vm_array))
        chain = find_attack_chain(graph, start_vm, length, rng, config.storage_pivot_probability)
        short_chains += len(chain) < length
        stealthy = bool(config.stealth_campaign_fraction) and (
            rng.random() < config.stealth_campaign_fraction
        )
        delay_low, delay_high = config.hop_delay_minutes
        last_start = latest_start
        if stealthy:
            delay_low, delay_high = config.stealth_hop_delay_minutes
            # Leave room for the slow hops, using the average delay.
            last_start = max(0.0, latest_start - (len(chain) - 1) * (delay_low + delay_high) / 120)
        t = float(rng.uniform(0, last_start))
        episode = []
        for vm in chain:
            # A long campaign can still run past the end of the observation period.
            if stealthy and t > hours:
                break
            episode.append(make_event(vm, t, "infected", episode_id, "coordinated", profile, rng))
            # A delay is drawn after the last VM as well; the notebook does the same and
            # skipping it would shift every later draw.
            t += float(rng.uniform(delay_low, delay_high)) / 60.0
        events += observed_events(episode, vm_array, profile, config, rng)
    if short_chains:
        log.warning(
            "%d of %d coordinated chains ended early for lack of unvisited neighbours",
            short_chains,
            config.n_coordinated_episodes,
        )

    df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return df.sort_values("timestamp_hours").reset_index(drop=True)


def generate(config: GeneratorConfig, profile: ParamProfile, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    graph = build_infrastructure(config.n_vms, config.n_storage, config.n_network, rng)
    events = generate_events(graph, profile, config, rng)
    log.info(
        "generated %d events on %d nodes / %d edges (seed %d)",
        len(events),
        graph.number_of_nodes(),
        graph.number_of_edges(), seed
    )
    return SyntheticDataset(graph=graph, events=events, config=config, seed=seed)
