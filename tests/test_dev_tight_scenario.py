"""Design invariants of the discrimination-first ``dev-continuum-*`` family.

These are not "does it load" tests. Each one pins a property the family exists to have,
so that a later edit which quietly restores the old degenerate landscape fails loudly
instead of silently producing numbers again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.scenarios import dev_tight

ROOT = Path(__file__).parents[1]
TASK_CONFIG = ROOT / "evolution" / "task" / "config_dev_continuum_tight.json"
CALIBRATED_TASK_CONFIG = (
    ROOT / "evolution" / "task" / "config_dev_continuum_tight_calibrated.json"
)


def test_calibrated_task_config_only_relaxes_slo_targets():
    original = json.loads(TASK_CONFIG.read_text())
    calibrated = json.loads(CALIBRATED_TASK_CONFIG.read_text())

    original_targets = original["constraints"].pop("slo_targets")
    calibrated_targets = calibrated["constraints"].pop("slo_targets")

    assert original == calibrated
    assert original_targets == {
        "ali-trace-0": 0.9,
        "ali-trace-1": 0.9,
        "ali-trace-2": 0.9,
    }
    assert calibrated_targets == {
        "ali-trace-0": 0.88,
        "ali-trace-1": 0.88,
        "ali-trace-2": 0.88,
    }


@pytest.mark.parametrize(
    "scenario_id", [dev_tight.DEV_CONTINUUM_TIGHT_ID, dev_tight.DEV_CONTINUUM_WIDE_ID]
)
def test_manifest_matches_the_tree(scenario_id):
    """The sealed manifest must rebuild byte-identically from the generator."""
    recorded = dev_tight.verify_tight_manifest(scenario_id)
    assert recorded["data_role"] == "development-only-quarantined"
    assert recorded["workload_builder"]["provenance_label"].startswith(
        "calibration-informed-then-transformed"
    ), "a transformed workload must not claim the untransformed builder's provenance"


@pytest.mark.parametrize(
    "scenario_id", [dev_tight.DEV_CONTINUUM_TIGHT_ID, dev_tight.DEV_CONTINUUM_WIDE_ID]
)
def test_horizon_outlives_the_published_fog_wake(scenario_id):
    """Suspension has to be a recoverable decision, not a one-way door.

    Every earlier fixture ran a 65 s horizon against a published 182 s fog wake latency,
    so a suspended fog node could never return and the only fog decision was "let it go".
    """
    config, _build = dev_tight.build_registered_scenario(scenario_id)
    wake_latencies = [
        cls.wake_latency_s for cls in config.cluster.classes if cls.wake_latency_s is not None
    ]
    assert wake_latencies, "the family needs at least one suspendable class"
    assert config.horizon_s >= 2 * max(wake_latencies)


@pytest.mark.parametrize(
    "scenario_id", [dev_tight.DEV_CONTINUUM_TIGHT_ID, dev_tight.DEV_CONTINUUM_WIDE_ID]
)
def test_slo_can_actually_bind(scenario_id):
    """The SLO must sit close enough to the service critical path to price the network.

    The builder's own rule is ``max(1.5, 8.0 * critical_p95)``; with a critical path near
    0.19 s the 1.5 s floor always won and no placement could threaten the constraint.
    """
    config, _build = dev_tight.build_registered_scenario(scenario_id)
    by_id = {service.id: service for service in config.workload.services}
    worst_hop = max(
        config.cluster.latency_ms[i][j]
        for i in range(len(config.cluster.nodes))
        for j in range(len(config.cluster.nodes))
    )
    for template in config.workload.templates:
        assert template.slo_latency_s is not None
        critical = dev_tight._critical_path_p95(template.root, by_id)
        headroom = template.slo_latency_s - critical
        assert headroom > 0, "an SLO below the service critical path is unsatisfiable"
        # A single worst-case hop must be able to consume a real share of the headroom;
        # otherwise placement cannot move the constraint and the fixture is back to
        # rewarding pure bin packing.
        assert worst_hop / 1000.0 > 0.2 * headroom


@pytest.mark.parametrize(
    "scenario_id", [dev_tight.DEV_CONTINUUM_TIGHT_ID, dev_tight.DEV_CONTINUUM_WIDE_ID]
)
def test_queues_are_bounded(scenario_id):
    """Unbounded queues make an overloading policy unrealistic *and* ruinously slow."""
    config, _build = dev_tight.build_registered_scenario(scenario_id)
    for service in config.workload.services:
        assert service.queue_capacity == dev_tight.QUEUE_CAPACITY_PER_REPLICA


def test_far_edge_capacity_is_actually_contended():
    """Peak *demand* must be able to exceed the far-edge tier.

    ``dev-medium-hot`` offered ~8,300 mc of peak requests against 48,000 mc of far edge,
    so "put everything on far edge" dominated unconditionally and no spill decision ever
    arose. Contention is driven by ``cpu_demand`` -- the millicores actually held while a
    replica processes -- not by the scheduling request, which is deliberately set below it
    (the classic under-set-requests deployment the fixture models).
    """
    config, _build = dev_tight.build_registered_scenario(dev_tight.DEV_CONTINUUM_TIGHT_ID)
    far_edge = sum(node.cpu_capacity for node in config.cluster.nodes if node.tier == 2)
    peak_demand = sum(
        service.cpu_demand * service.max_replicas for service in config.workload.services
    )
    peak_request = sum(
        service.cpu_request * service.max_replicas for service in config.workload.services
    )
    # Demand can oversubscribe the tier, so consolidation has to be paid for; requests
    # stay under it, so a competent policy can still admit everything somewhere.
    assert peak_demand > far_edge
    assert peak_request < far_edge


def test_the_two_scenarios_are_structurally_different():
    """The transfer target has to be a different cluster, not a reseeded twin."""
    tight, _ = dev_tight.build_registered_scenario(dev_tight.DEV_CONTINUUM_TIGHT_ID)
    wide, _ = dev_tight.build_registered_scenario(dev_tight.DEV_CONTINUUM_WIDE_ID)
    assert len(tight.cluster.nodes) != len(wide.cluster.nodes)
    tight_sites = {item[5] for item in dev_tight._TIGHT_NODES}
    wide_sites = {item[5] for item in dev_tight._WIDE_NODES}
    assert wide_sites - tight_sites, "the transfer cluster needs unseen sites"
