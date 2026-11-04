"""Topology-family labels, caps, transforms, and replay tests."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from sim.scenarios.sources import load_source_registry
from sim.scenarios.topology import (
    BA_LABEL,
    NO_EXTERNAL_VALIDITY,
    ZOO_LABEL,
    RoleBinding,
    _load_scale_policy_fixture_bytes,
    barabasi_albert_topology,
    hierarchical_topology,
    load_scale_policy,
    materialize_cluster,
    parse_undirected_graphml,
    topology_zoo_from_graphml,
)
from sim.spec import NodeClassSpec


def _bindings() -> tuple[RoleBinding, ...]:
    return tuple(
        RoleBinding(
            role=role,
            runtime_tier=tier,
            node_class=NodeClassSpec(
                name=role,
                power_idle_w=2.0 + index,
                power_max_w=5.0 + index,
            ),
            cpu_capacity=2000 + 500 * index,
            mem_capacity=4096,
        )
        for index, (role, tier) in enumerate(
            zip(
                ("cloud-core", "regional-fog", "access-edge", "device-edge"),
                (0, 1, 2, 2),
                strict=True,
            )
        )
    )


def _policy(confirmatory: int, exploratory: int):
    del exploratory
    return _load_scale_policy_fixture_bytes(
        json.dumps(
            {
                "schema": "g0-capacity-table-v1",
                "table_id": "g0-test-table",
                "ceilings": [
                    {"name": "cpu-hours", "capacity": 100.0, "projected_use": 70.0}
                ],
                "scale_trials": [
                    {
                        "nodes": confirmatory,
                        "p90_runtime_s": 7.0,
                        "runtime_budget_s": 10.0,
                    }
                ],
            },
            sort_keys=True,
        ).encode()
    )


def test_hierarchical_and_ba_are_seed_replayable_and_scale_bounded():
    policy = _policy(16, 32)
    a = hierarchical_topology((2, 3, 4, 5), 7, policy)
    b = hierarchical_topology((2, 3, 4, 5), 7, policy)
    assert a == b
    cluster = materialize_cluster(a, policy, _bindings())
    assert len(cluster.nodes) == 14
    assert tuple(node.tier for node in cluster.nodes[-5:]) == (2,) * 5

    ba = barabasi_albert_topology(12, 2, 8, policy)
    assert ba.evidence_label == BA_LABEL
    assert ba.external_validity_statement == NO_EXTERNAL_VALIDITY
    with pytest.raises(ValueError, match="exceeds G0 cap"):
        barabasi_albert_topology(17, 2, 8, policy)
    with pytest.raises(ValueError, match="digest mismatch"):
        dataclasses.replace(ba, transform="tampered").verify(policy)


def test_production_g0_fails_closed_and_fixture_status_is_recomputed():
    with pytest.raises(ValueError, match="production G0.*unavailable"):
        load_scale_policy()
    failing = json.dumps(
        {
            "schema": "g0-capacity-table-v1",
            "table_id": "g0-failing-test-table",
            "ceilings": [
                {"name": "cpu-hours", "capacity": 100.0, "projected_use": 80.0}
            ],
            "scale_trials": [
                {"nodes": 8, "p90_runtime_s": 12.0, "runtime_budget_s": 10.0}
            ],
        },
        sort_keys=True,
    ).encode()
    with pytest.raises(ValueError, match=">=30% reserve"):
        _load_scale_policy_fixture_bytes(failing)


def test_topology_zoo_uses_only_hashed_structure_and_exact_label():
    graphml = b"""<?xml version='1.0'?>
    <graphml xmlns='http://graphml.graphdrawing.org/xmlns'>
      <graph id='G' edgedefault='undirected'>
        <node id='a'/><node id='b'/><node id='c'/>
        <edge source='a' target='b'/><edge source='b' target='c'/>
      </graph>
    </graphml>"""
    digest = hashlib.sha256(graphml).hexdigest()
    policy = _policy(8, 16)
    node_ids, edges = parse_undirected_graphml(graphml)
    assert node_ids == ("a", "b", "c") and edges == ((0, 1), (1, 2))
    with pytest.raises(ValueError, match="payload.*unavailable"):
        topology_zoo_from_graphml(graphml, digest, load_source_registry(), 99, policy)
    assert ZOO_LABEL == "empirically grounded network-structure generalization"
    with pytest.raises(ValueError, match="hash mismatch"):
        topology_zoo_from_graphml(graphml + b" ", digest, load_source_registry(), 99, policy)
