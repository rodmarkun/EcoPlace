"""Confirmatory hand-written PolicyBundles (TODO 7.1).

The Kubernetes resource scorers reproduce the pinned v1.30.14 integer formulas for
CPU and memory with equal weights.  NetworkOverhead re-expresses the upstream
accumulated dependency-cost idea over ``CallGraphCtx`` and the simulator's complete
node-to-node latency matrix; it does not claim AppGroup/NetworkTopology CRD or Filter
contract equivalence.  The reusable-policy registry contains honest per-pod FirstFit;
the full-queue First-Fit Decreasing contextual assignment baseline lives in
``assignment_oracles.py`` because only that boundary can order a pod batch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from baselines.adapter import PolicyBundleAdapter
from sim.actuation import CallableProposer, PlacementScriptProposer
from sim.scheduler import CallableScorer, PlacementScriptScorer

POLICIES = Path(__file__).resolve().parent.parent / "runtime" / "policies"
HANDWRITTEN_NAMES = (
    "latency_greedy",
    "energy_greedy",
    "first_fit",
    "least_allocated",
    "most_allocated",
    "network_overhead",
)


def _resource_score(requested: int, capacity: int, *, most: bool) -> int:
    if capacity == 0:
        return 0
    if requested > capacity:
        return 100 if most else 0
    return requested * 100 // capacity if most else (capacity - requested) * 100 // capacity


def least_allocated_score(pod: dict, node: dict, ctx: dict) -> float:
    del ctx
    cpu = _resource_score(
        node["cpu_allocated"] + pod["cpu_request"], node["cpu_capacity"], most=False
    )
    mem = _resource_score(
        node["mem_allocated"] + pod["mem_request"], node["mem_capacity"], most=False
    )
    return float((cpu + mem) // 2)


def most_allocated_score(pod: dict, node: dict, ctx: dict) -> float:
    del ctx
    cpu = _resource_score(
        node["cpu_allocated"] + pod["cpu_request"], node["cpu_capacity"], most=True
    )
    mem = _resource_score(
        node["mem_allocated"] + pod["mem_request"], node["mem_capacity"], most=True
    )
    return float((cpu + mem) // 2)


def first_fit_score(pod: dict, node: dict, ctx: dict) -> float:
    del pod, ctx
    return -float(node["id"])


def latency_greedy_score(pod: dict, node: dict, ctx: dict) -> float:
    total = 0.0
    graph = ctx["graph"]
    for src, dst, rate in zip(graph["src"], graph["dst"], graph["rate"], strict=True):
        other = dst if src == pod["id"] else src if dst == pod["id"] else -1
        if other >= 0 and other != pod["id"]:
            neighbor_node = ctx["pods"][other]["current_node"]
            if neighbor_node >= 0:
                total += rate * node["latency_to_ms"][neighbor_node]
    return -total


def network_overhead_score(pod: dict, node: dict, ctx: dict) -> float:
    """Negated accumulated topology cost across already placed dependencies.

    Upstream NetworkOverhead treats lower accumulated cost as better and reverses it
    in NormalizeScore.  Our common normalizer treats larger as better, hence negation.
    Rates/payloads remain observations available to every method but are intentionally
    not weights here; the upstream scoring idea counts dependency instances.
    """
    total = 0.0
    graph = ctx["graph"]
    for src, dst in zip(graph["src"], graph["dst"], strict=True):
        other = dst if src == pod["id"] else src if dst == pod["id"] else -1
        if other >= 0 and other != pod["id"]:
            neighbor_node = ctx["pods"][other]["current_node"]
            if neighbor_node >= 0:
                total += node["latency_to_ms"][neighbor_node]
    return -total


def energy_greedy_score(pod: dict, node: dict, ctx: dict) -> float:
    del ctx
    capacity = float(max(node["cpu_capacity"], 1))
    span = max(node["power_max_w"] - node["power_idle_w"], 0.0)
    cost = span * (float(pod["cpu_request"]) / capacity)
    if node["suspended"]:
        cost += node["power_idle_w"] + 5.0
    if node["cpu_allocated"] == 0 and not node["suspended"]:
        cost += node["power_idle_w"] * 0.5
    return -cost


def _noop_proposer(state: dict, graph: dict, budget: int) -> list[dict]:
    del state, graph, budget
    return []


def _latency_proposer(state: dict, graph: dict, budget: int) -> list[dict]:
    out = []
    if budget <= 0:
        return out
    for src, dst, rate in zip(graph["src"], graph["dst"], graph["rate"], strict=True):
        if len(out) >= budget or rate <= 1.0:
            continue
        source = state["pods"][src]
        target = state["pods"][dst]
        placed = source["current_node"] >= 0 and target["current_node"] >= 0
        if (
            placed
            and source["current_node"] != target["current_node"]
            and not source["stateful"]
            and not source["has_pdb"]
            and not state["nodes"][target["current_node"]]["suspended"]
        ):
            out.append({"kind": 0, "pod": source["id"], "dest": target["current_node"]})
    return out


def _energy_proposer(state: dict, graph: dict, budget: int) -> list[dict]:
    del graph
    if budget <= 0 or len(state["nodes"]) < 2:
        return []
    drain = min(
        (
            (node["cpu_allocated"] / max(node["cpu_capacity"], 1), node["id"])
            for node in state["nodes"]
            if node["suspendable"] and not node["suspended"]
        ),
        default=None,
    )
    if drain is None or drain[0] >= 0.3:
        return []
    choices = [node for node in state["nodes"] if node["id"] != drain[1]]
    awake = [node for node in choices if not node["suspended"]]
    pool = awake or [node for node in choices if node["suspended"]]
    if not pool:
        return []
    target = min(
        pool,
        key=lambda node: (
            node["power_idle_w"] / max(node["cpu_capacity"], 1),
            node["id"],
        ),
    )
    out = []
    if target["suspended"]:
        if budget < 2:
            return []
        out.append({"kind": 1, "pod": 0, "dest": target["id"]})
    for pod in state["pods"]:
        if len(out) >= budget:
            break
        if pod["current_node"] == drain[1] and not pod["stateful"] and not pod["has_pdb"]:
            out.append({"kind": 0, "pod": pod["id"], "dest": target["id"]})
    return out


SCORERS: dict[str, Callable[[dict, dict, dict], float]] = {
    "latency_greedy": latency_greedy_score,
    "energy_greedy": energy_greedy_score,
    "first_fit": first_fit_score,
    "least_allocated": least_allocated_score,
    "most_allocated": most_allocated_score,
    "network_overhead": network_overhead_score,
}

PROPOSERS: dict[str, Callable[[dict, dict, int], list[dict]]] = {
    "latency_greedy": _latency_proposer,
    "energy_greedy": _energy_proposer,
    "first_fit": _noop_proposer,
    "least_allocated": _noop_proposer,
    "most_allocated": _noop_proposer,
    "network_overhead": _noop_proposer,
}

NATIVE_FILES = {
    "latency_greedy": "latency_greedy.ps",
    "energy_greedy": "energy_greedy.ps",
    "first_fit": "first_fit.ps",
    "least_allocated": "kube_least_allocated.ps",
    "most_allocated": "kube_most_allocated.ps",
    "network_overhead": "network_overhead.ps",
}


def _metadata(name: str, implementation: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "status": "confirmatory-reusable-policy",
        "implementation": implementation,
        "observation_schema": "placementscript-cluster-ctx-v1",
        "actuation_schema": "placementscript-move-v1",
    }
    if name == "first_fit":
        common["contract_note"] = (
            "per-pod FirstFit over stable node-id order; full-queue FFD is a separate "
            "contextual assignment baseline"
        )
    if name == "network_overhead":
        common.update(
            {
                "upstream": "kubernetes-sigs/scheduler-plugins@v0.30.12",
                "upstream_component": "NetworkOverhead accumulated dependency cost",
                "contract_mismatch": (
                    "our graph+latency observation replaces AppGroup and NetworkTopology CRDs; "
                    "max-network-cost Filter behavior is outside this scorer baseline"
                ),
            }
        )
    if name in {"least_allocated", "most_allocated"}:
        common["formula_pin"] = "kubernetes/v1.30.14 NodeResourcesFit cpu+memory weights=1"
    return common


def python_handwritten(name: str) -> PolicyBundleAdapter:
    if name not in HANDWRITTEN_NAMES:
        raise KeyError(f"unknown hand-written baseline {name!r}")
    artifact_sha256 = hashlib.sha256(
        Path(__file__).read_bytes() + b"\0" + name.encode()
    ).hexdigest()
    return PolicyBundleAdapter(
        name=name,
        version="1.0.0",
        scorer=CallableScorer(name, SCORERS[name]),
        proposer=CallableProposer(name, PROPOSERS[name]),
        representation="python-policybundle-adapter",
        artifact_sha256=artifact_sha256,
        metadata=_metadata(name, "python-emulator"),
    )


def native_handwritten(name: str) -> PolicyBundleAdapter:
    """Load the same policy through the native Rust PlacementScript interpreter."""
    if name not in HANDWRITTEN_NAMES:
        raise KeyError(f"unknown hand-written baseline {name!r}")
    import placementscript as ps

    path = POLICIES / NATIVE_FILES[name]
    source = path.read_text()
    policy = ps.Policy(source)
    if policy.name != name:
        raise ValueError(f"native policy identity mismatch: expected {name}, got {policy.name}")
    return PolicyBundleAdapter(
        name=policy.name,
        version=policy.version,
        scorer=PlacementScriptScorer(policy),
        proposer=PlacementScriptProposer(policy),
        representation="placementscript-native",
        artifact_sha256=hashlib.sha256(source.encode()).hexdigest(),
        metadata={
            **_metadata(name, "rust-native"),
            "source_path": str(path.relative_to(POLICIES.parent.parent)),
        },
    )


def all_handwritten(*, native: bool = True) -> tuple[PolicyBundleAdapter, ...]:
    factory = native_handwritten if native else python_handwritten
    return tuple(factory(name) for name in HANDWRITTEN_NAMES)
