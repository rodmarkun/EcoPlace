"""Construct independent synthetic scenario designs, without policy evaluation.

Independence here means new declared topology/workload construction, not empirical
independence from the researcher's knowledge of historical results. No trace fit is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from evolution.evaluation import config_for_development_bundle
from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.matched_control_20260909.real_pilot import historical_seed_inventory
from sim.energy import build_development_bundles, load_pinned_sources
from sim.spec import (
    ArrivalSpec,
    CallEdgeSpec,
    CallSpec,
    ClusterSpec,
    DistributionSpec,
    EpisodeConfig,
    HpaParams,
    NodeClassSpec,
    NodeSpec,
    PowerParams,
    RebalancerParams,
    SchedulerParams,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    config_sha256,
    det,
    episode_config_from_json,
    episode_config_to_json,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "independent_scenarios_20260909_v1"
TOPOLOGIES = {
    "distributed": {"cloud_nodes": 3, "sites": 3, "fog_per_site": 2, "edge_per_site": 3},
    "edge_heavy": {"cloud_nodes": 2, "sites": 4, "fog_per_site": 1, "edge_per_site": 3},
}
PRESSURES = (0.15, 0.30, 0.45)
PROFILES = ("steady", "two_bursts")
# Name, mean seconds, CV, CPU request, CPU demand, memory MiB.
SERVICE_ROWS = (
    ("gateway", 0.006, 0.25, 50, 500, 128),
    ("lookup", 0.018, 0.50, 500, 1000, 256),
    ("profile", 0.030, 0.75, 500, 1500, 512),
    ("inventory", 0.012, 0.25, 250, 500, 256),
    ("compute", 0.060, 0.75, 1000, 2000, 1024),
    ("commit", 0.024, 0.50, 500, 1000, 512),
)
MIX = (0.5, 0.3, 0.2)
DEADLINES = (0.3, 0.5, 0.4)
TEMPLATE_NAMES = ("interactive", "transaction", "aggregate")
CONSTRAINTS = {
    "slo_targets": dict.fromkeys(TEMPLATE_NAMES, 0.90),
    "goodput_target": 0.95,
    "drop_cap": 0.025,
    "timeout_cap": 0.025,
    "disruption_caps": {"bytes_mb": 1024, "moves": 32, "unavailability_s": 120},
    "zero_cap_scales": {},
}


def call(service: int, *children: CallSpec) -> CallSpec:
    return CallSpec(
        service, tuple(CallEdgeSpec(c, request_kb=2.0, response_kb=4.0) for c in children)
    )


def trees() -> tuple[CallSpec, ...]:
    return (
        call(0, call(1), call(2, call(3))),
        call(0, call(4, call(5))),
        call(0, call(1, call(3)), call(2, call(3)), call(4)),
    )


def counts(tree: CallSpec) -> Counter:
    out = Counter({tree.service: 1})
    for edge in tree.children:
        out.update(counts(edge.child))
    return out


def mean_cpu_work() -> tuple[float, ...]:
    """Offered processing demand in model-millicore seconds per complete call tree."""
    return tuple(
        math.fsum(n * SERVICE_ROWS[s][1] * SERVICE_ROWS[s][4] for s, n in counts(t).items())
        for t in trees()
    )


def profile_segments(profile: str) -> tuple[tuple[float, float, float], ...]:
    if profile == "steady":
        return ((30.0, 870.0, 1.0),)
    if profile == "two_bursts":
        return (
            (30.0, 270.0, 0.75),
            (270.0, 330.0, 2.5),
            (330.0, 570.0, 0.75),
            (570.0, 690.0, 1.625),
            (690.0, 870.0, 0.75),
        )
    raise ValueError("unknown demand profile")


def topology(name: str, bundles) -> tuple[ClusterSpec, list[dict], list[dict]]:
    design = TOPOLOGIES[name]
    bindings = bundles[2].node_class_bindings()
    classes = (
        NodeClassSpec("cloud-region", power_idle_w=110.0, power_max_w=340.0),
        bindings["fog-server"].spec,
        bindings["far-edge"].spec,
    )
    nodes, locations, links = [], [], []
    for _ in range(design["cloud_nodes"]):
        i = len(nodes)
        nodes.append(NodeSpec(i, "cloud-region", 0, 16000, 32768))
        locations.append({"node": i, "site": None})
        links.append({"a": f"n{i}", "b": "wan", "one_way_ms": 5.0})
    for site in range(design["sites"]):
        hub = f"site{site}"
        links.append({"a": hub, "b": "wan", "one_way_ms": 15.0 + 5.0 * site})
        for tier, number, cls, memory, delay in (
            (1, design["fog_per_site"], "fog-server", 8192, 1.0),
            (2, design["edge_per_site"], "far-edge", 4096, 2.0),
        ):
            for _ in range(number):
                i = len(nodes)
                nodes.append(NodeSpec(i, cls, tier, 4000, memory))
                locations.append({"node": i, "site": site})
                links.append({"a": f"n{i}", "b": hub, "one_way_ms": delay})
    vertices = sorted({v for link in links for v in (link["a"], link["b"])})
    distances = {(a, b): 0.0 if a == b else math.inf for a in vertices for b in vertices}
    for link in links:
        distances[link["a"], link["b"]] = link["one_way_ms"]
        distances[link["b"], link["a"]] = link["one_way_ms"]
    for k in vertices:
        for i in vertices:
            for j in vertices:
                distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
    matrix = tuple(
        tuple(distances[f"n{i}", f"n{j}"] for j in range(len(nodes))) for i in range(len(nodes))
    )
    return ClusterSpec(classes, tuple(nodes), matrix), locations, links


def episode(topology_name: str, profile: str, pressure: float, bundles) -> EpisodeConfig:
    if pressure not in PRESSURES:
        raise ValueError("pressure outside the finite grid")
    cluster, _, _ = topology(topology_name, bundles)
    edge_capacity = sum(n.cpu_capacity for n in cluster.nodes if n.tier == 2)
    work = math.fsum(a * b for a, b in zip(MIX, mean_cpu_work(), strict=True))
    rate = pressure * edge_capacity / work
    services = []
    for i, (name, mean_s, cv, request, demand, memory) in enumerate(SERVICE_ROWS):
        sigma2 = math.log1p(cv * cv)
        services.append(
            ServiceSpec(
                id=i,
                name=name,
                app=0,
                cpu_request=request,
                mem_request=memory,
                service_time=DistributionSpec(
                    "lognormal", {"mu": math.log(mean_s) - sigma2 / 2, "sigma": math.sqrt(sigma2)}
                ),
                cpu_demand=demand,
                concurrency=2,
                queue_capacity=32,
                cold_start=det(0.2),
                image_mb=20,
                initial_replicas=4 if i == 0 else 2,
                min_replicas=4 if i == 0 else 2,
                max_replicas=16 if i == 0 else 8,
                category="directly-specified-synthetic-development",
            )
        )
    segments = profile_segments(profile)
    templates = tuple(
        TemplateSpec(
            id=i,
            name=TEMPLATE_NAMES[i],
            root=tree,
            arrival=ArrivalSpec(
                "poisson",
                rate * mix * segments[0][2],
                start_s=30.0,
                rate_schedule=tuple((a, rate * mix * level) for a, _, level in segments[1:]),
            ),
            slo_latency_s=DEADLINES[i],
        )
        for i, (mix, tree) in enumerate(zip(MIX, trees(), strict=True))
    )
    return EpisodeConfig(
        cluster=cluster,
        workload=WorkloadSpec(tuple(services), templates),
        horizon_s=900.0,
        arrival_cutoff_s=870.0,
        scheduler=SchedulerParams(bind_latency=det(0.01)),
        hpa=HpaParams(
            enabled=True,
            sync_period_s=6.0,
            target_utilization=0.6,
            scale_down_stabilization_s=120.0,
        ),
        rebalancer=RebalancerParams(
            enabled=True,
            window_s=30.0,
            budget_per_window=2,
            max_total_bytes_mb=1024,
            max_total_moves=32,
            max_total_unavailability_s=120,
        ),
        power=PowerParams(suspend_after_idle_s=30.0),
        snapshot_times_s=(180.0, 300.0, 600.0, 840.0),
        snapshot_cycles=tuple(range(8)),
    )


def packing_witness(config: EpisodeConfig) -> list[dict]:
    """Construct a static reservation packing at maximum replicas, not a QoS witness."""
    remaining = {
        n.id: [n.cpu_capacity, n.mem_capacity] for n in config.cluster.nodes if n.tier == 2
    }
    assigned = []
    for s in sorted(config.workload.services, key=lambda s: (-s.cpu_request, -s.mem_request, s.id)):
        for replica in range(s.max_replicas):
            node = next(
                (n for n, (c, m) in remaining.items() if c >= s.cpu_request and m >= s.mem_request),
                None,
            )
            if node is None:
                raise ValueError("static edge reservation packing failed")
            remaining[node][0] -= s.cpu_request
            remaining[node][1] -= s.mem_request
            assigned.append({"service": s.id, "replica": replica, "node": node})
    return assigned


def seed_design() -> dict:
    inventory = historical_seed_inventory()
    excluded = set(inventory["seeds"]) | set(range(800000801, 800000821))
    plans = []
    for folder in ("workload_sensitivity_20260909", "analytical_tails_20260909"):
        path = HERE / folder / "PLAN.json"
        plan = read_sealed(path)
        if "seeds" in plan:
            excluded.update(plan["seeds"])
            excluded.add(plan["pilot_seed"])
        else:
            excluded.update(job["seed"] for job in plan["jobs"])
        plans.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    fixture = ROOT / "sim/fixtures/MANIFEST.json"
    excluded.update(json.loads(fixture.read_text())["acceptance_seed_matrix"])
    excluded.update((3246458936443790985, 9113512321037399042))
    roles = {}
    for role, count in (("calibration", 3), ("resource_pilot", 1)):
        seeds = [
            int.from_bytes(
                hashlib.sha256(
                    f"independent-synthetic-design-20260909-v1/{role}/{i}".encode()
                ).digest()[:8],
                "big",
            )
            % (2**63 - 1)
            for i in range(count)
        ]
        if len(set(seeds)) != len(seeds) or set(seeds) & excluded:
            raise ValueError("seed collision; cannot silently redraw")
        roles[role] = seeds
        excluded.update(seeds)
    return {
        "roles": roles,
        "historical_inventory": inventory,
        "active_plan_files": plans,
        "acceptance_seed_manifest_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "search_and_final_seeds": None,
        "scope": "Checked against retained main historical evaluation grids, main roots, "
        "active study grids, fixture acceptance grid and corrected initialization pilot. "
        "Not an exhaustive inventory of every historical random draw.",
    }


def build() -> tuple[dict, dict[str, str]]:
    bundles = build_development_bundles(load_pinned_sources())
    policies = read_sealed(HERE / "workload_sensitivity_20260909/PLAN.json")["policies"]
    refs = [p for p in policies if p["group"] in ("fixed-reference", "posthoc-reference")]
    if len(refs) != 9:
        raise ValueError("expected seven fixed and two disclosed posthoc references")
    for policy in refs:
        if (
            hashlib.sha256((ROOT / policy["path"]).read_bytes()).hexdigest()
            != policy["source_sha256"]
        ):
            raise ValueError("reference source changed")
    outputs, cells, topology_records = {}, [], {}
    for name in TOPOLOGIES:
        cluster, sites, links = topology(name, bundles)
        topology_records[name] = {
            "design": TOPOLOGIES[name],
            "locations": sites,
            "links": links,
            "latency_ms": cluster.latency_ms,
        }
        for profile in PROFILES:
            for pressure in PRESSURES:
                config = episode(name, profile, pressure, bundles)
                identity = f"{name}-{profile}-{pressure:.2f}"
                canonical = episode_config_to_json(config)
                if config_sha256(episode_config_from_json(canonical)) != config_sha256(config):
                    raise ValueError("base scenario round-trip identity mismatch")
                outputs[f"{identity}.json"] = canonical + "\n"
                cells.append(
                    {
                        "id": identity,
                        "topology": name,
                        "profile": profile,
                        "nominal_average_edge_cpu_pressure": pressure,
                        "file_sha256": hashlib.sha256((canonical + "\n").encode()).hexdigest(),
                        "base_config_sha256": config_sha256(config),
                        "expected_offered_requests": [
                            t.arrival.integrated_rate(config.cutoff_s)
                            for t in config.workload.templates
                        ],
                        "reservation_packing_at_max_replicas": packing_witness(config),
                        "effective_configs": [
                            {
                                "bundle_id": b.bundle_id,
                                "config_sha256": config_sha256(
                                    config_for_development_bundle(config, b)
                                ),
                            }
                            for b in bundles
                        ],
                    }
                )
    design = {
        "schema": "independent-synthetic-scenario-design-v1",
        "status": "constructed-not-calibrated-not-launch-receipt",
        "empirical_independence": False,
        "new_trace_records_used": False,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (HERE / "INDEPENDENT_SCENARIOS.md").read_bytes()
        ).hexdigest(),
        "structural_tests_sha256": hashlib.sha256(
            (ROOT / "tests/test_independent_scenarios.py").read_bytes()
        ).hexdigest(),
        "topologies": topology_records,
        "cells": cells,
        "constraints": CONSTRAINTS,
        "boundary_weights": {"device": 1, "cloud": 1, "network": 1},
        "boundary_bounds": {"max_transitions_per_node": 64, "network_bytes_bound": 20 * 1024**3},
        "seed_design": seed_design(),
        "fixed_references": refs,
        "calibration_episodes": len(cells) * len(refs) * len(bundles) * 3,
        "cpu_work_per_template": mean_cpu_work(),
        "mixed_cpu_work_per_request": math.fsum(
            a * b for a, b in zip(MIX, mean_cpu_work(), strict=True)
        ),
        "bundle_manifests": [b.manifest() for b in bundles],
        "protocol": "../INDEPENDENT_SCENARIOS.md",
    }
    return design, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    design, outputs = build()
    outputs["DESIGN.json"] = json.dumps(design, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if not args.check:
        OUT.mkdir(exist_ok=True)
    for name, value in outputs.items():
        path = OUT / name
        if args.check:
            if path.read_text() != value:
                raise ValueError(f"design mismatch: {name}")
        else:
            path.write_text(value)
    print(
        json.dumps(
            {
                "status": "checked" if args.check else "constructed",
                "cells": len(design["cells"]),
                "calibration_episodes_planned": design["calibration_episodes"],
                "simulation_episodes_executed": 0,
                "seeds": design["seed_design"]["roles"],
            }
        )
    )


if __name__ == "__main__":
    main()
