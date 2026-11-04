"""L0 throughput benchmark (TODO 2.7 acceptance: ~1 ms/candidate, measured, recorded).

Two modes:

- default: builds a frozen snapshot library from the ``small_cluster`` development
  fixture (one episode; timed + decision-point snapshots) and measures the wall-clock
  cost of one *candidate evaluation* (= ``L0Evaluator.evaluate`` over the whole
  library) for each seed policy, via the native PlacementScript batch API.
- ``--scale``: a synthetic representative-size library (``--nodes`` nodes,
  ``--snapshots`` snapshots, ``--pending`` pending pods each) for the scaling curve —
  the plan's ~1 ms budget is checked at intended library sizes, not just the tiny
  fixture (R1-16).

Run: ``uv run python sim/benchmarks/l0_bench.py --seed S [--repeats N] [--scale ...]``
``--seed`` is REQUIRED (docs/SEEDS.md: no seed defaults anywhere).
The measured numbers quoted in ``sim/README.md`` come from this script; re-run it to
reproduce them on your hardware.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import placementscript as ps

from sim.energy import BoundaryWeights
from sim.l0 import L0Evaluator
from sim.scheduler import PlacementScriptScorer
from sim.snapshot import SnapshotLibrary, export_library, initial_snapshot
from sim.spec import (
    ArrivalSpec,
    CallSpec,
    ClusterSpec,
    EpisodeConfig,
    NodeClassSpec,
    NodeSpec,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    expo,
    load_episode_config,
)

REPO = Path(__file__).resolve().parent.parent.parent
POLICIES = REPO / "runtime" / "policies"
FIXTURE = REPO / "sim" / "fixtures" / "dev" / "small_cluster.json"
SEED_POLICIES = ("least_allocated", "most_allocated", "latency_greedy", "energy_greedy")


def fixture_library(seed: int) -> SnapshotLibrary:
    config = load_episode_config(FIXTURE)
    generator = PlacementScriptScorer(ps.Policy((POLICIES / "least_allocated.ps").read_text()))
    from sim.engine import run_episode  # late import: keep CLI startup cheap

    result = run_episode(config, seed=seed, scorer=generator)
    return export_library(
        config,
        [initial_snapshot(config), *result.snapshots],
        meta={"seed": seed, "generator": generator.name, "episode": result.digest()},
    )


def synthetic_library(nodes: int, snapshots: int, pending: int) -> SnapshotLibrary:
    """Representative-size library: ``pending`` initial replicas over ``nodes`` nodes,
    snapshot replicated ``snapshots`` times (identical work per snapshot — the cost
    driver is snapshots x pending x nodes)."""
    if nodes < 1 or snapshots < 1 or pending < 1:
        raise ValueError("nodes, snapshots, and pending must all be positive")
    edge = NodeClassSpec(name="edge", power_idle_w=3.0, power_max_w=8.0)
    cloud = NodeClassSpec(name="cloud", power_idle_w=100.0, power_max_w=300.0)
    node_specs = tuple(
        NodeSpec(i, "cloud" if i % 10 == 0 else "edge", 0 if i % 10 == 0 else 2, 8000, 16384)
        for i in range(nodes)
    )
    lat = tuple(
        tuple(0.0 if i == j else float(5 + ((i + j) % 40)) for j in range(nodes))
        for i in range(nodes)
    )
    cluster = ClusterSpec(classes=(edge, cloud), nodes=node_specs, latency_ms=lat)
    n_services = max(1, pending // 2)
    replicas_per_service, extra_replicas = divmod(pending, n_services)
    services = tuple(
        ServiceSpec(
            id=s,
            name=f"svc{s}",
            app=0,
            cpu_request=250,
            mem_request=128,
            service_time=expo(0.02),
            cpu_demand=250,
            image_mb=10,
            initial_replicas=replicas_per_service + (s < extra_replicas),
            max_replicas=max(8, replicas_per_service + (s < extra_replicas)),
        )
        for s in range(n_services)
    )
    templates = tuple(
        TemplateSpec(
            id=t, name=f"t{t}", root=CallSpec(service=t), arrival=ArrivalSpec("poisson", 2.0)
        )
        for t in range(n_services)
    )
    config = EpisodeConfig(
        cluster=cluster,
        workload=WorkloadSpec(services=services, templates=templates),
        horizon_s=60.0,
    )
    snap = initial_snapshot(config)
    return export_library(config, [snap] * snapshots, meta={"synthetic": True, "nodes": nodes})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--seed", type=int, required=True, help="episode root seed (required; SEEDS.md)"
    )
    parser.add_argument("--scale", action="store_true", help="synthetic scaled library")
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--snapshots", type=int, default=20)
    parser.add_argument("--pending", type=int, default=12)
    args = parser.parse_args()

    if args.scale:
        library = synthetic_library(args.nodes, args.snapshots, args.pending)
    else:
        library = fixture_library(args.seed)
    evaluator = L0Evaluator(library, BoundaryWeights())
    n_pending = sum(len(s.pending_dense_ids) for s in library.snapshots)
    print(
        f"library: {len(library.snapshots)} snapshots, {n_pending} pending pods total, "
        f"{len(library.config.cluster.nodes)} nodes, digest {library.digest()[:12]}"
    )
    for name in SEED_POLICIES:
        scorer = PlacementScriptScorer(ps.Policy((POLICIES / f"{name}.ps").read_text()))
        out = evaluator.evaluate(scorer)  # warm-up
        samples = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = evaluator.evaluate(scorer)
            samples.append((time.perf_counter() - t0) * 1000.0)
        assert out.valid
        samples.sort()
        p50 = statistics.median(samples)
        p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
        print(
            f"{name:>16}: p50={p50:.3f} ms/candidate  p95={p95:.3f} ms  "
            f"(mean_norm_energy={out.mean_normalized_energy:.4f}, "
            f"feasible={out.feasible})"
        )


if __name__ == "__main__":
    main()
