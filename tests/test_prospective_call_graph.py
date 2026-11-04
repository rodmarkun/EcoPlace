"""The pod being scored must observe the call-graph edges it will carry.

Before this projection existed, ``_pod_level_graph`` expanded edges over READY replicas
only. The pod in a scheduling cycle is by construction not READY, so ``ctx.graph``
contained no edge incident on it and every *scoring* policy that reads the graph
returned a constant. ``network_overhead``, ``latency_greedy`` and ``first_fit`` scored
bit-identical energies on every fixture as a result, and the dependency-aware baseline
the plan guarantees (§6.2) could not express its own idea.

These tests pin the three properties the fix has to have:

1. the scored pod's prospective edges appear;
2. a policy that never reads ``ctx.graph`` is bit-for-bit unaffected;
3. the rebalancer's view (``include_pod=None``, no prospective set) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import simpy
from sim_helpers import cheap_energy

from sim.energy import EnergyMeter
from sim.runtime import READY, ClusterRuntime, Ledger
from sim.scheduler import SchedulerCache

POLICIES = Path(__file__).resolve().parents[1] / "runtime" / "policies"


def _chain_episode():
    """Two services wired parent -> child, three nodes, one ready child replica."""
    from sim import (
        ArrivalSpec,
        CallEdgeSpec,
        CallSpec,
        ClusterSpec,
        EpisodeConfig,
        NodeClassSpec,
        NodeSpec,
        ServiceSpec,
        TemplateSpec,
        WorkloadSpec,
        det,
    )

    cluster_spec = ClusterSpec(
        (NodeClassSpec("box", 1.0, 2.0),),
        tuple(NodeSpec(i, "box", 2, 8000, 8192) for i in range(3)),
        ((0.0, 5.0, 40.0), (5.0, 0.0, 40.0), (40.0, 40.0, 0.0)),
    )
    parent = ServiceSpec(0, "parent", 0, 100, 128, det(0.01), 100, image_mb=0)
    child = ServiceSpec(1, "child", 0, 100, 128, det(0.01), 100, image_mb=0)
    template = TemplateSpec(
        0,
        "chain",
        CallSpec(0, (CallEdgeSpec(CallSpec(1), request_kb=1.0, response_kb=1.0),)),
        ArrivalSpec("deterministic", 1.0),
    )
    config = EpisodeConfig(cluster_spec, WorkloadSpec((parent, child), (template,)), 5.0)
    return config, parent, child


def _runtime(config):
    env = simpy.Environment()
    meter = EnergyMeter(config.cluster, cheap_energy())
    cluster = ClusterRuntime(env, config.cluster, meter, Ledger())
    cache = SchedulerCache(env, cluster, informer_delay_s=0.0)
    return env, cluster, cache


def test_scored_pod_sees_its_own_prospective_edges():
    from sim.scheduler import build_ctx

    config, parent_spec, child_spec = _chain_episode()
    env, cluster, cache = _runtime(config)

    serving_child = cluster.create_pod(child_spec, "initial")
    cluster.bind(serving_child, 1, notify=True)
    serving_child.phase = READY
    env.run(until=0.5)

    pending_parent = cluster.create_pod(parent_spec, "hpa")
    ctx, uid_to_dense, _ = build_ctx(
        cluster, config.workload, cache, env.now, config.cutoff_s, include_pod=pending_parent
    )
    dense = uid_to_dense[pending_parent.uid]
    incident = [
        index
        for index, (src, dst) in enumerate(
            zip(ctx["graph"]["src"], ctx["graph"]["dst"], strict=True)
        )
        if dense in (src, dst)
    ]
    assert incident, "the pod under decision must carry its prospective call-graph edges"

    # Same instant, projection disabled: the pod is invisible to a graph-aware scorer.
    ready_only, uid_to_dense_ro, _ = build_ctx(
        cluster,
        config.workload,
        cache,
        env.now,
        config.cutoff_s,
        include_pod=pending_parent,
        prospective=(),
    )
    dense_ro = uid_to_dense_ro[pending_parent.uid]
    assert not [
        index
        for index, (src, dst) in enumerate(
            zip(ready_only["graph"]["src"], ready_only["graph"]["dst"], strict=True)
        )
        if dense_ro in (src, dst)
    ]


def test_graph_blind_policy_is_bit_identical_under_the_projection():
    """A policy that never reads ctx.graph must not move when the graph changes."""
    import placementscript as ps

    from sim.scheduler import PlacementScriptScorer, build_ctx

    config, parent_spec, child_spec = _chain_episode()
    env, cluster, cache = _runtime(config)
    serving_child = cluster.create_pod(child_spec, "initial")
    cluster.bind(serving_child, 1, notify=True)
    serving_child.phase = READY
    env.run(until=0.5)
    pending_parent = cluster.create_pod(parent_spec, "hpa")

    scorer = PlacementScriptScorer(ps.Policy((POLICIES / "least_allocated.ps").read_text()))
    scores = []
    for prospective in (None, ()):
        ctx, uid_to_dense, _ = build_ctx(
            cluster,
            config.workload,
            cache,
            env.now,
            config.cutoff_s,
            include_pod=pending_parent,
            prospective=prospective,
        )
        pod = ctx["pods"][uid_to_dense[pending_parent.uid]]
        scores.append(tuple(scorer.score_batch(pod, ctx["nodes"], ctx)))
    assert scores[0] == scores[1]


def test_graph_aware_policy_prefers_the_neighbour_site():
    """latency_greedy must now rank the low-latency neighbour node first."""
    import placementscript as ps

    from sim.scheduler import PlacementScriptScorer, build_ctx

    config, parent_spec, child_spec = _chain_episode()
    env, cluster, cache = _runtime(config)
    serving_child = cluster.create_pod(child_spec, "initial")
    cluster.bind(serving_child, 1, notify=True)
    serving_child.phase = READY
    env.run(until=0.5)
    pending_parent = cluster.create_pod(parent_spec, "hpa")

    ctx, uid_to_dense, _ = build_ctx(
        cluster, config.workload, cache, env.now, config.cutoff_s, include_pod=pending_parent
    )
    pod = ctx["pods"][uid_to_dense[pending_parent.uid]]
    scorer = PlacementScriptScorer(ps.Policy((POLICIES / "latency_greedy.ps").read_text()))
    scores = scorer.score_batch(pod, ctx["nodes"], ctx)
    # Node 2 sits 40 ms from the child's node 1; nodes 0 and 1 sit 5 ms and 0 ms away.
    assert scores[2] < scores[0]
    assert scores[1] >= scores[0]


def test_rebalancer_view_is_unchanged():
    """include_pod=None keeps the historical READY-only expansion exactly."""
    from sim.scheduler import build_ctx

    config, _parent_spec, child_spec = _chain_episode()
    env, cluster, cache = _runtime(config)
    serving_child = cluster.create_pod(child_spec, "initial")
    cluster.bind(serving_child, 1, notify=True)
    serving_child.phase = READY
    env.run(until=0.5)

    default, _, _ = build_ctx(cluster, config.workload, None, env.now, config.cutoff_s)
    explicit, _, _ = build_ctx(
        cluster, config.workload, None, env.now, config.cutoff_s, prospective=()
    )
    assert default["graph"] == explicit["graph"]


@pytest.mark.parametrize("name", ["network_overhead", "latency_greedy", "graph_headroom"])
def test_dependency_aware_baselines_load_and_read_the_graph(name):
    """Every policy the discrimination gate calls graph-reading must actually read it."""
    import placementscript as ps

    source = (POLICIES / f"{name}.ps").read_text()
    ps.Policy(source)
    assert "ctx.graph" in source
