from __future__ import annotations

import dataclasses

import pytest

from baselines.handwritten import (
    HANDWRITTEN_NAMES,
    all_handwritten,
    native_handwritten,
    python_handwritten,
)


def _ctx() -> tuple[dict, list[dict], dict]:
    nodes = [
        {
            "id": 0,
            "tier": 0,
            "cpu_capacity": 4000,
            "cpu_allocated": 1000,
            "mem_capacity": 8000,
            "mem_allocated": 2000,
            "power_idle_w": 80.0,
            "power_max_w": 200.0,
            "suspended": False,
            "suspendable": False,
            "latency_to_ms": [0.0, 20.0, 40.0],
        },
        {
            "id": 1,
            "tier": 1,
            "cpu_capacity": 2000,
            "cpu_allocated": 1500,
            "mem_capacity": 4000,
            "mem_allocated": 2500,
            "power_idle_w": 12.0,
            "power_max_w": 28.0,
            "suspended": False,
            "suspendable": True,
            "latency_to_ms": [20.0, 0.0, 5.0],
        },
        {
            "id": 2,
            "tier": 2,
            "cpu_capacity": 2000,
            "cpu_allocated": 0,
            "mem_capacity": 4000,
            "mem_allocated": 0,
            "power_idle_w": 4.0,
            "power_max_w": 10.0,
            "suspended": False,
            "suspendable": True,
            "latency_to_ms": [40.0, 5.0, 0.0],
        },
    ]
    pods = [
        {
            "id": 0,
            "service": 0,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": -1,
        },
        {
            "id": 1,
            "service": 1,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": 1,
        },
        {
            "id": 2,
            "service": 2,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": 2,
        },
    ]
    graph = {"src": [0, 0], "dst": [1, 2], "rate": [5.0, 1.0], "payload_kb": [2.0, 8.0]}
    return pods[0], nodes, {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}


@pytest.mark.parametrize("name", HANDWRITTEN_NAMES)
def test_native_and_python_handwritten_scores_match(name: str) -> None:
    pod, nodes, ctx = _ctx()
    python = python_handwritten(name)
    native = native_handwritten(name)
    assert native.scorer.score_batch(pod, nodes, ctx) == pytest.approx(
        python.scorer.score_batch(pod, nodes, ctx)
    )
    assert native.name == python.name == name
    assert native.metadata["observation_schema"] == python.metadata["observation_schema"]


@pytest.mark.parametrize("name", HANDWRITTEN_NAMES)
def test_native_and_python_handwritten_proposals_match(name: str) -> None:
    _pod, nodes, ctx = _ctx()
    pods = [dict(pod) for pod in ctx["pods"]]
    pods[0]["current_node"] = 0
    state = {"nodes": nodes, "pods": pods}
    graph = ctx["graph"]
    python = python_handwritten(name).proposer.propose_moves(state, graph, 3)
    native = native_handwritten(name).proposer.propose_moves(state, graph, 3)
    assert native == python


def test_resource_scorers_reproduce_pinned_integer_formulas() -> None:
    pod, nodes, ctx = _ctx()
    least = python_handwritten("least_allocated").scorer.score_batch(pod, nodes, ctx)
    most = python_handwritten("most_allocated").scorer.score_batch(pod, nodes, ctx)
    assert least == [70.0, 23.0, 91.0]
    assert most == [28.0, 76.0, 7.0]


def test_network_overhead_counts_dependencies_not_traffic_volume() -> None:
    pod, nodes, ctx = _ctx()
    network = python_handwritten("network_overhead").scorer.score_batch(pod, nodes, ctx)
    latency = python_handwritten("latency_greedy").scorer.score_batch(pod, nodes, ctx)
    assert network == [-60.0, -5.0, -5.0]
    assert latency == [-140.0, -5.0, -25.0]
    assert "AppGroup" in python_handwritten("network_overhead").metadata["contract_mismatch"]


def test_reusable_first_fit_is_honestly_named_and_stable() -> None:
    pod, nodes, ctx = _ctx()
    baseline = native_handwritten("first_fit")
    assert baseline.scorer.score_batch(pod, [nodes[2], nodes[0]], ctx) == [-2.0, -0.0]
    assert "full-queue FFD" in baseline.metadata["contract_note"]


def test_kubernetes_scorer_bundles_do_not_attach_custom_rebalancers() -> None:
    _pod, nodes, ctx = _ctx()
    state = {"nodes": nodes, "pods": ctx["pods"]}
    for name in ("least_allocated", "most_allocated"):
        assert python_handwritten(name).proposer.propose_moves(state, ctx["graph"], 3) == []
        assert native_handwritten(name).proposer.propose_moves(state, ctx["graph"], 3) == []


def test_all_handwritten_registry_is_complete_and_identity_is_stable() -> None:
    first = all_handwritten(native=False)
    second = all_handwritten(native=False)
    assert tuple(bundle.name for bundle in first) == HANDWRITTEN_NAMES
    assert [bundle.identity_sha256 for bundle in first] == [
        bundle.identity_sha256 for bundle in second
    ]
    assert all(len(bundle.artifact_sha256) == 64 for bundle in first)
    changed_bytes = dataclasses.replace(first[0], artifact_sha256="f" * 64)
    assert changed_bytes.identity_sha256 != first[0].identity_sha256


def test_unknown_handwritten_name_fails_closed() -> None:
    with pytest.raises(KeyError, match="unknown hand-written"):
        native_handwritten("not-a-baseline")
