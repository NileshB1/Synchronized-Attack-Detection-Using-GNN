"""Synthetic enterprise infrastructure and attack-chain walks."""

from __future__ import annotations

import networkx as nx
import numpy as np

NODE_KINDS = ("vm", "backup_image", "storage", "network")


def build_infrastructure(
    n_vms: int, n_storage: int, n_network: int, rng: np.random.Generator
) -> nx.Graph:
    """Each VM has its own backup image; each image sits on one random storage pool
    and each VM on one random network fabric."""
    for name, value in (("n_vms", n_vms), ("n_storage", n_storage), ("n_network", n_network)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    g = nx.Graph()
    for i in range(n_vms):
        g.add_node(f"vm_{i}", kind="vm")
        g.add_node(f"image_{i}", kind="backup_image")
        g.add_edge(f"vm_{i}", f"image_{i}")
    for i in range(n_storage):
        g.add_node(f"storage_{i}", kind="storage")
    for i in range(n_network):
        g.add_node(f"network_{i}", kind="network")
    for i in range(n_vms):
        storage = f"storage_{rng.integers(0, n_storage)}"
        network = f"network_{rng.integers(0, n_network)}"
        g.add_edge(f"image_{i}", storage)
        g.add_edge(f"vm_{i}", network)
    return g


def vm_nodes(graph: nx.Graph) -> list[str]:
    # Sorted as strings (vm_0, vm_1, vm_10, ...), as in the notebook; the order
    # determines which random draws each VM receives.
    return sorted(n for n, d in graph.nodes(data=True) if d["kind"] == "vm")


def vms_on_one_network(
    graph: nx.Graph, vms: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Up to size VMs from the network of one randomly chosen VM."""
    first = str(rng.choice(vms))
    network = next(n for n in graph.neighbors(first) if graph.nodes[n]["kind"] == "network")
    members = sorted(n for n in graph.neighbors(network) if graph.nodes[n]["kind"] == "vm")
    return rng.choice(members, size=min(size, len(members)), replace=False)


def _vms_on_same_storage(graph: nx.Graph, vm: str) -> set[str]:
    return {
        other_vm
        for image in graph.neighbors(vm)
        if graph.nodes[image]["kind"] == "backup_image"
        for pool in graph.neighbors(image)
        if graph.nodes[pool]["kind"] == "storage"
        for other_image in graph.neighbors(pool)
        for other_vm in graph.neighbors(other_image)
        if graph.nodes[other_vm]["kind"] == "vm"
    }


def find_attack_chain(
    graph: nx.Graph,
    start_vm: str,
    n_vms: int,
    rng: np.random.Generator,
    storage_pivot_probability: float = 0.0,
) -> list[str]:
    """Random walk over unvisited VMs that are two edges apart.

    With the graph from build_infrastructure, the only VM-to-VM paths of length
    two pass through a shared network fabric. With storage_pivot_probability set,
    a hop may instead reach a VM whose backup image is on the same storage pool.
    The chain stops early when the current VM has no unvisited neighbour.
    """
    if n_vms < 1:
        raise ValueError(f"chain length must be at least 1, got {n_vms}")
    chain = [start_vm]
    visited = {start_vm}
    current = start_vm
    for _ in range(n_vms - 1):
        candidates = set()
        if storage_pivot_probability and rng.random() < storage_pivot_probability:
            candidates = _vms_on_same_storage(graph, current) - visited
        candidates = candidates or {
            n2
            for n1 in graph.neighbors(current)
            for n2 in graph.neighbors(n1)
            if graph.nodes[n2]["kind"] == "vm" and n2 not in visited
        }
        if not candidates:
            break
        current = str(rng.choice(sorted(candidates)))
        chain.append(current)
        visited.add(current)
    return chain
