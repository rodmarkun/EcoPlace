"""Regenerate the development and acceptance fixture files (deterministic, no RNG) and
the frozen ``MANIFEST.json`` (R1-14).

Run: ``uv run python sim/fixtures/make_fixtures.py``  (pre-P1 only; after the P1 freeze
this script must not be re-run — the manifest hashes are the frozen record).

Development fixtures (``dev/``) are used freely by the verification suite while the
simulator is under construction. Acceptance fixtures (``acceptance/``) are structurally
analogous but use *different parameters*; they are reserved for the one-shot P1-gated
DES verification run (T6) executed by ``acceptance/run_gate.py`` with the
``ACCEPTANCE_SEEDS`` matrix below — see ``acceptance/README.md``.

The manifest records: per-file sha256, the generator's own sha256, the acceptance seed
matrix, and the non-overlap rule; ``tests/test_sim_spec.py`` verifies all of it,
including that no acceptance seed appears anywhere in test or sim code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sim.spec import (
    ArrivalSpec,
    BackendSpec,
    CallEdgeSpec,
    CallSpec,
    ClusterSpec,
    EpisodeConfig,
    HpaParams,
    NodeClassSpec,
    NodeSpec,
    SchedulerParams,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    det,
    expo,
    save_episode_config,
)

HERE = Path(__file__).resolve().parent

# The frozen acceptance seed matrix (T6): reserved exclusively for the one-shot gate
# run. These values must never appear in tests/ or development code (enforced by
# TestFixtureDiscipline). Chosen as distinctive 11-digit literals so a plain grep of
# the source tree finds any leak.
# v2 (24 September 2026): twenty fresh seeds after the v1 one-shot gate failed its
# per-seed M/M/1 tolerance (experiments/scientific_gates_20260920/DES_V2_DESIGN.md).
# The five v1 seeds are retired, recorded in MANIFEST.json and never reused.
ACCEPTANCE_SEEDS = (
    51082845389,
    22257148542,
    15834129497,
    98153319815,
    65552557695,
    38857304140,
    90853140906,
    23957934471,
    93460894020,
    40967710316,
    33397376163,
    84305870083,
    42189232650,
    68915814697,
    77402427717,
    90345793472,
    91320516170,
    81387282767,
    53899113507,
    28682788941,
)



def _single_node_cluster() -> ClusterSpec:
    """One ample edge node: replica queueing is the only bottleneck (closed forms)."""
    cls = NodeClassSpec(name="edge-xl", power_idle_w=10.0, power_max_w=30.0)
    return ClusterSpec(
        classes=(cls,),
        nodes=(NodeSpec(0, "edge-xl", 2, 1_000_000, 1_000_000),),
        latency_ms=((0.0,),),
    )


def mm1(rate: float, mean_service_s: float, horizon_s: float) -> EpisodeConfig:
    """M/M/1: Poisson arrivals, one single-server exponential replica, no children.

    Closed form: sojourn W = 1 / (mu - lambda), mu = 1/mean_service_s.
    """
    service = ServiceSpec(
        id=0,
        name="station",
        app=0,
        cpu_request=100,
        mem_request=64,
        service_time=expo(mean_service_s),
        cpu_demand=0,  # no CPU contention: the replica queue is the only station
        image_mb=1,
    )
    template = TemplateSpec(
        id=0, name="mm1", root=CallSpec(service=0), arrival=ArrivalSpec("poisson", rate)
    )
    return EpisodeConfig(
        cluster=_single_node_cluster(),
        workload=WorkloadSpec(services=(service,), templates=(template,)),
        horizon_s=horizon_s,
        arrival_cutoff_s=horizon_s,
        scheduler=SchedulerParams(bind_latency=det(0.001)),
    )


def jackson(
    rate: float, mean_a_s: float, mean_b_s: float, p_ab: float, horizon_s: float
) -> EpisodeConfig:
    """Feed-forward Jackson network: A -> (prob p_ab, forward handoff) -> B, same node.

    Closed form (Jackson/Burke): W_i = 1/(mu_i - lambda_i) with lambda_A = rate,
    lambda_B = p_ab * rate; E[T] = W_A + p_ab * W_B (no network delay on one node).
    """
    svc_a = ServiceSpec(
        id=0,
        name="a",
        app=0,
        cpu_request=100,
        mem_request=64,
        service_time=expo(mean_a_s),
        cpu_demand=0,
        image_mb=1,
    )
    svc_b = ServiceSpec(
        id=1,
        name="b",
        app=0,
        cpu_request=100,
        mem_request=64,
        service_time=expo(mean_b_s),
        cpu_demand=0,
        image_mb=1,
    )
    template = TemplateSpec(
        id=0,
        name="jackson",
        root=CallSpec(
            service=0,
            mode="forward",
            children=(CallEdgeSpec(child=CallSpec(service=1), probability=p_ab),),
        ),
        arrival=ArrivalSpec("poisson", rate),
    )
    return EpisodeConfig(
        cluster=_single_node_cluster(),
        workload=WorkloadSpec(services=(svc_a, svc_b), templates=(template,)),
        horizon_s=horizon_s,
        arrival_cutoff_s=horizon_s,
        scheduler=SchedulerParams(bind_latency=det(0.001)),
    )


def small_cluster(arrival_rate: float, horizon_s: float) -> EpisodeConfig:
    """A 3-node cloud/edge cluster with fan-out, retries, a backend, HPA, and a
    suspendable node — the general-purpose development scenario."""
    classes = (
        NodeClassSpec(
            name="edge",
            power_idle_w=3.0,
            power_max_w=8.0,
            suspend_energy_j=5.0,
            suspend_latency_s=2.0,
            wake_energy_j=8.0,
            wake_latency_s=3.0,
            suspended_power_w=0.5,
        ),
        NodeClassSpec(name="cloud", power_idle_w=100.0, power_max_w=300.0),
    )
    nodes = (
        NodeSpec(0, "cloud", 0, 8000, 16384),
        NodeSpec(1, "edge", 2, 4000, 8192),
        NodeSpec(2, "edge", 2, 4000, 8192, initially_suspended=False),
    )
    lat = ((0.0, 20.0, 20.0), (20.0, 0.0, 5.0), (20.0, 5.0, 0.0))
    front = ServiceSpec(
        id=0,
        name="front",
        app=0,
        cpu_request=500,
        mem_request=256,
        service_time=expo(0.02),
        cpu_demand=500,
        initial_replicas=2,
        max_replicas=4,
        image_mb=20,
    )
    back = ServiceSpec(
        id=1,
        name="back",
        app=0,
        cpu_request=500,
        mem_request=256,
        service_time=expo(0.03),
        cpu_demand=500,
        backend="db",
        image_mb=20,
    )
    template = TemplateSpec(
        id=0,
        name="rpc",
        root=CallSpec(
            service=0,
            children=(CallEdgeSpec(child=CallSpec(service=1), timeout_s=1.0, max_retries=1),),
        ),
        arrival=ArrivalSpec("poisson", arrival_rate),
        slo_latency_s=0.5,
    )
    return EpisodeConfig(
        cluster=ClusterSpec(classes=classes, nodes=nodes, latency_ms=lat),
        workload=WorkloadSpec(
            services=(front, back),
            templates=(template,),
            backends=(BackendSpec("db", ((1.0, 0.002), (10.0, 0.02))),),
        ),
        horizon_s=horizon_s,
        arrival_cutoff_s=horizon_s * 0.8,
        hpa=HpaParams(enabled=True),
        snapshot_times_s=(horizon_s * 0.25, horizon_s * 0.5, horizon_s * 0.75),
        snapshot_cycles=(0, 1, 2, 3, 4),  # decision-point snapshots (pending pods) for L0
    )


def overload(horizon_s: float) -> EpisodeConfig:
    """Pathological overload: offered load far above capacity, bounded queues, tight
    timeouts — drops and timeouts must be accounted, never lost."""
    cls = NodeClassSpec(name="tiny", power_idle_w=2.0, power_max_w=4.0)
    cluster = ClusterSpec(
        classes=(cls,),
        nodes=(NodeSpec(0, "tiny", 2, 1000, 1024),),
        latency_ms=((0.0,),),
    )
    service = ServiceSpec(
        id=0,
        name="swamped",
        app=0,
        cpu_request=200,
        mem_request=64,
        service_time=expo(0.5),
        cpu_demand=200,
        queue_capacity=10,
        image_mb=1,
    )
    template = TemplateSpec(
        id=0,
        name="flood",
        root=CallSpec(service=0),
        arrival=ArrivalSpec("poisson", 50.0),
    )
    return EpisodeConfig(
        cluster=cluster,
        workload=WorkloadSpec(services=(service,), templates=(template,)),
        horizon_s=horizon_s,
        arrival_cutoff_s=horizon_s,
    )


def main() -> None:
    dev = HERE / "dev"
    acc = HERE / "acceptance"
    # Development set.
    save_episode_config(mm1(rate=8.0, mean_service_s=0.08, horizon_s=2500.0), dev / "mm1.json")
    save_episode_config(
        jackson(rate=5.0, mean_a_s=0.08, mean_b_s=0.1, p_ab=0.6, horizon_s=3000.0),
        dev / "jackson.json",
    )
    save_episode_config(small_cluster(8.0, 60.0), dev / "small_cluster.json")
    save_episode_config(overload(30.0), dev / "overload.json")
    # Acceptance set: structurally analogous, different parameters (T6 disjointness);
    # frozen at P1, executed once (see acceptance/README.md).
    save_episode_config(mm1(rate=6.0, mean_service_s=0.1, horizon_s=4000.0), acc / "mm1.json")
    save_episode_config(
        jackson(rate=4.0, mean_a_s=0.1, mean_b_s=0.125, p_ab=0.5, horizon_s=4500.0),
        acc / "jackson.json",
    )
    save_episode_config(small_cluster(11.0, 90.0), acc / "small_cluster.json")
    save_episode_config(overload(45.0), acc / "overload.json")

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "generated_by": "sim/fixtures/make_fixtures.py (deterministic; no RNG)",
        "generator_sha256": sha(Path(__file__)),
        "freeze_rule": (
            "frozen at P1; regeneration after the P1 freeze invalidates the gate. "
            "Acceptance fixtures + ACCEPTANCE_SEEDS are consumed only by "
            "acceptance/run_gate.py, exactly once."
        ),
        "acceptance_seed_matrix": list(ACCEPTANCE_SEEDS),
        "retired_acceptance_seeds": {
            "v1": [91731400017, 91731400129, 91731400231, 91731400347, 91731400459],
            "reason": "consumed by the failed v1 one-shot gate on 2026-09-24",
        },
        "seed_non_overlap_rule": (
            "no acceptance seed may appear anywhere in tests/ or sim/ outside this "
            "manifest and run_gate.py (enforced by TestFixtureDiscipline)"
        ),
        "files": {
            f"dev/{p.name}": sha(p) for p in sorted(dev.glob("*.json"))
        }
        | {f"acceptance/{p.name}": sha(p) for p in sorted(acc.glob("*.json"))},
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"fixtures + MANIFEST.json written under {HERE}")


if __name__ == "__main__":
    main()
