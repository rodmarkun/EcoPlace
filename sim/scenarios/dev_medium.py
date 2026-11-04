"""Quarantined **development** scenario ``dev-medium-cluster`` (Phase-5 support).

Why this file exists
--------------------
The frozen ``dev-small-cluster`` fixture (``sim/fixtures/dev/small_cluster.json``) has
three nodes on two tiers (cloud + far-edge) and three pods.  Because it carries **no
fog tier**, the three development energy bundles (``dev-1``/``dev-2``/``dev-3``) differ
only in their fog-server machine, so every bundle scores an identical energy and the
whole feasible landscape collapses onto a handful of points: the 2026-08-09 paired
development campaign (``experiments/dev_campaign_20260809/REPORT.md``) recorded 54/54
feasible LLM candidates at exactly the initial policy's energy.

``dev-medium-cluster`` is a *second, additional* development fixture built for gradient,
not a replacement.  ``sim/fixtures/**`` is frozen at P1 (its ``MANIFEST.json`` enumerates
the exact file set), so this fixture lives under ``sim/scenarios/dev/`` with its own
manifest and seal and is consumed through a **new** task config
(``evolution/task/config_dev_medium.json``); the existing task config is untouched.

Shape
-----
20 nodes on three tiers (2 cloud, 4 fog, 14 far-edge), 5 services (11 initial replicas,
scaling to ~17-21 under load) wired into a two-template call graph with retries,
timeouts, two backends and two SLOs; Poisson arrivals with a mid-episode rate step;
65 s horizon, arrivals cut off at 52 s; four evaluation seeds x three development
bundles = a twelve-episode L2 grid.

Design levers that create real gradient (all placement-driven):

* **Three tiers.**  Tier-0 cloud energy is *allocated* (bound ``cpu_request`` vCPU-hours
  x a per-vCPU-hour coefficient), tier-1/tier-2 device energy is *integrated from the
  measured power curve over actual utilization*.  Moving one pod across that boundary
  changes energy by a large, continuous amount, and the fog tier makes ``dev-1``,
  ``dev-2`` and ``dev-3`` genuinely distinct instances (their fog machines differ by
  ~6x in idle power and ~90x in capacity).  ``dev-small-cluster`` has no tier-1 node at
  all, which is precisely why its three bundles score bit-identical energies.
* **Auto-suspend.**  Fog-server classes carry published transition data, so an empty
  fog node suspends after ``suspend_after_idle_s``; consolidating policies bank the
  idle-power saving, spreading policies do not.  Far-edge boards have no published
  transition data and are non-suspendable, so the lever is tier-specific.  The horizon
  is long enough for the saving to exceed the published suspend pulse — at a 40 s
  horizon suspension is almost exactly energy-neutral and the lever does nothing.
* **A mid-episode load step plus requests below demand.**  ``cpu_request`` is
  deliberately smaller than ``cpu_demand`` (the classic under-set-requests deployment),
  so measured HPA utilization is high enough for the rate step to trigger scale-ups.
  Those new replicas are placed *by the policy under test*, at times that depend on the
  policy, which is the fixture's main continuous energy channel: it turns a one-shot
  initial-placement puzzle into a trajectory.
* **Per-node image cache.**  A pod start pulls its image only on a node that has not
  cached it, and pulled bytes are charged to the allocated network category, so the
  *spread* of a scaled-up deployment is directly priced.
* **Heterogeneous memory and a site-structured latency matrix.**  Bundle binding
  overrides tier-1/tier-2 ``cpu_capacity`` with the sealed capacity bridge, so memory
  and latency are the per-node heterogeneity this fixture owns.
* **An enabled rebalancer.**  ``propose_moves`` is half of an evolved PlacementScript;
  with the rebalancer disabled it is dead code and half the search space is flat.

Known structural flatness (a simulator property, not a fixture choice): the scheduler's
pod-level call graph is expanded over **READY** replicas only, and the pod being
scheduled is by construction not ready, so a pure *scoring* policy never observes its
own call-graph edges.  Graph-only scorers (``network_overhead``) therefore cannot
separate from ``first_fit`` on any fixture; ``latency_greedy`` separates only through
its proposer.  Five distinct hand-written baseline energies is the ceiling here.

Everything is deterministic: the builder takes no RNG, arrivals are driven by the
explicit per-episode seeds in :data:`DEV_MEDIUM_CLUSTER_SEEDS`, and the fixture bytes
plus the generator bytes are sealed in ``sim/scenarios/dev/MANIFEST.json``.

Richness evidence: ``experiments/dev_scenario_richness_20260809/RICHNESS.md``.

This is development/quarantined data.  It is never acceptance evidence, it is not part
of the public/private split machinery, and it must not be cited as confirmatory.
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
    PowerParams,
    RebalancerParams,
    SchedulerParams,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    config_sha256,
    det,
    expo,
    load_episode_config,
    save_episode_config,
)

HERE = Path(__file__).resolve().parent
DEV_DIR = HERE / "dev"
DEV_MEDIUM_CLUSTER_ID = "dev-medium-cluster"
DEV_MEDIUM_CLUSTER_PATH = DEV_DIR / "medium_cluster.json"
MANIFEST_PATH = DEV_DIR / "MANIFEST.json"
MANIFEST_SEAL_TAG = "dev-scenario-manifest-seal-v1:"
DATA_ROLE = "development-only-quarantined"

# Explicit evaluation seeds (hard rule: docs/SEEDS.md — no wall-clock or entropy
# defaults anywhere).  Four distinct primes, disjoint from the ``dev-small-cluster``
# seeds (104729, 130363) and from the frozen acceptance seed matrix.  Four seeds x the
# three development bundles is the twelve-episode L2 grid.
DEV_MEDIUM_CLUSTER_SEEDS: tuple[int, ...] = (224737, 350377, 611953, 821641)

# Node layout.  Tier-1/tier-2 ``cpu_capacity`` below is a structural placeholder: the
# sealed capacity bridge (``evolution.evaluation.config_for_development_bundle``)
# replaces it with ``NodeClassBinding.capacity_millicores`` for the bound bundle.
_CLOUD = "cloud-region"
_CLOUD_EDGEY = "cloud-zone"
_FOG = "fog"
_EDGE = "edge"

# (node_id, class, tier, cpu_capacity, mem_capacity_mib, site)
# site is a latency-topology label only; -1 marks the central cloud region.
#
# Node ids deliberately **interleave** the tiers.  If ids were grouped cloud -> fog ->
# edge, then stable node-id order would coincide exactly with latency proximity to the
# cloud, and per-pod FirstFit, latency-greedy and network-overhead scorers would all
# emit the same placement — three baselines collapsing onto one point for a reason that
# is an artefact of the fixture's numbering, not of the policies.
#
# Memory is likewise deliberately *not* monotone in node id.  Bundle binding overwrites
# every tier-1/tier-2 ``cpu_capacity`` with the sealed capacity bridge, so RAM is the
# only per-node size this fixture owns; if the lowest-id far-edge board were also the
# tightest one, a most-allocated (bin-packing) scorer and an energy-greedy scorer would
# choose the same node for the same pod and score identically for the whole episode.
_NODES: tuple[tuple[int, str, int, int, int, int], ...] = (
    (0, _EDGE, 2, 4000, 3072, 0),
    (1, _CLOUD, 0, 3200, 8192, -1),
    (2, _EDGE, 2, 4000, 1536, 1),
    (3, _FOG, 1, 16000, 24576, 0),
    (4, _EDGE, 2, 4000, 6144, 0),
    (5, _CLOUD_EDGEY, 0, 2400, 6144, -1),
    (6, _EDGE, 2, 4000, 3072, 2),
    (7, _FOG, 1, 16000, 12288, 1),
    (8, _EDGE, 2, 4000, 2048, 1),
    (9, _EDGE, 2, 4000, 4096, 0),
    (10, _FOG, 1, 16000, 6144, 2),
    (11, _EDGE, 2, 4000, 8192, 2),
    (12, _EDGE, 2, 4000, 2048, 1),
    (13, _EDGE, 2, 4000, 4096, 0),
    (14, _EDGE, 2, 4000, 1536, 2),
    (15, _EDGE, 2, 4000, 6144, 2),
    (16, _EDGE, 2, 4000, 2560, 1),
    (17, _FOG, 1, 16000, 3072, 3),
    (18, _EDGE, 2, 4000, 5120, 3),
    (19, _EDGE, 2, 4000, 1024, 3),
)

_HORIZON_S = 65.0
_ARRIVAL_CUTOFF_S = 52.0
# Offered load starts only after the initial deployment can plausibly be ready
# (bind latency + image pull + cold start).  Without this warm-up window every policy
# collects the same batch of ``dropped_no_replica`` records at t=0, which is a fixture
# artefact rather than a placement consequence.
_ARRIVAL_START_S = 4.0
# A mid-episode load step: the cluster runs comfortably, then the offered rate roughly
# doubles.  This is what makes HPA scale-ups (and therefore *policy-chosen* placement of
# newly created replicas, with their image pulls) part of the landscape instead of a
# one-shot initial-placement puzzle.
_RATE_STEP_S = 26.0
_RPC_RATE_STEP: tuple[tuple[float, float], ...] = ((_RATE_STEP_S, 14.0),)
_BATCH_RATE_STEP: tuple[tuple[float, float], ...] = ((_RATE_STEP_S, 4.0),)


def _latency_ms() -> tuple[tuple[float, ...], ...]:
    """Symmetric, zero-diagonal site-structured matrix (ms).

    Cloud is one region; each fog node anchors a site and its far-edge boards sit
    beside it.  Cross-site hops are expensive, so co-location is worth something to a
    latency-aware scorer without ever being free.
    """

    Node = tuple[int, str, int, int, int, int]

    def cost(a: Node, b: Node) -> float:
        if a[0] == b[0]:
            return 0.0
        tier_a, tier_b = a[2], b[2]
        site_a, site_b = a[5], b[5]
        tiers = (min(tier_a, tier_b), max(tier_a, tier_b))
        if tiers == (0, 0):
            return 1.0
        if tiers == (0, 1):
            return 25.0
        if tiers == (0, 2):
            return 45.0
        if tiers == (1, 1):
            return 8.0
        if tiers == (1, 2):
            return 4.0 if site_a == site_b else 18.0
        return 2.0 if site_a == site_b else 20.0

    return tuple(tuple(cost(a, b) for b in _NODES) for a in _NODES)


def _cluster() -> ClusterSpec:
    classes = (
        # Tier-0 classes carry no device energy (cloud is allocation-metered); their
        # power fields are observations available to every scorer.
        NodeClassSpec(name=_CLOUD, power_idle_w=110.0, power_max_w=340.0),
        NodeClassSpec(name=_CLOUD_EDGEY, power_idle_w=70.0, power_max_w=210.0),
        # Tier-1/tier-2 classes are replaced wholesale by the sealed bundle bridge;
        # these development placeholders keep the raw fixture a valid EpisodeConfig.
        NodeClassSpec(
            name=_FOG,
            power_idle_w=60.0,
            power_max_w=200.0,
            suspend_energy_j=1655.0,
            suspend_latency_s=15.0,
            wake_energy_j=23683.0,
            wake_latency_s=182.0,
            suspended_power_w=8.0,
        ),
        NodeClassSpec(name=_EDGE, power_idle_w=1.8, power_max_w=4.4),
    )
    nodes = tuple(
        NodeSpec(node_id, class_name, tier, cpu, mem)
        for node_id, class_name, tier, cpu, mem, _site in _NODES
    )
    return ClusterSpec(classes=classes, nodes=nodes, latency_ms=_latency_ms())


def _workload() -> WorkloadSpec:
    gateway = ServiceSpec(
        id=0,
        name="gateway",
        app=0,
        cpu_request=100,
        mem_request=256,
        service_time=expo(0.030),
        cpu_demand=320,
        image_mb=24,
        initial_replicas=3,
        min_replicas=3,
        max_replicas=10,
        cold_start=det(0.2),
    )
    auth = ServiceSpec(
        id=1,
        name="auth",
        app=0,
        cpu_request=55,
        mem_request=192,
        service_time=expo(0.018),
        cpu_demand=190,
        image_mb=12,
        initial_replicas=2,
        min_replicas=2,
        max_replicas=6,
        cold_start=det(0.2),
    )
    catalog = ServiceSpec(
        id=2,
        name="catalog",
        app=0,
        cpu_request=315,
        mem_request=384,
        service_time=expo(0.040),
        cpu_demand=430,
        backend="db",
        image_mb=32,
        initial_replicas=2,
        min_replicas=2,
        max_replicas=8,
        cold_start=det(0.3),
    )
    recommender = ServiceSpec(
        id=3,
        name="recommender",
        app=1,
        cpu_request=165,
        mem_request=640,
        service_time=expo(0.060),
        cpu_demand=680,
        image_mb=48,
        initial_replicas=2,
        min_replicas=2,
        max_replicas=5,
        cold_start=det(0.4),
    )
    ledger = ServiceSpec(
        id=4,
        name="ledger",
        app=1,
        cpu_request=130,
        mem_request=320,
        service_time=expo(0.035),
        cpu_demand=360,
        backend="kv",
        image_mb=20,
        initial_replicas=2,
        min_replicas=2,
        max_replicas=6,
        state_transfer_mb=4.0,
        cold_start=det(0.2),
    )

    # Interactive path: gateway fans out to auth and catalog; catalog sometimes writes
    # through the ledger.  Response payloads are asymmetric so co-location decisions
    # price differently per edge.
    rpc = TemplateSpec(
        id=0,
        name="rpc",
        root=CallSpec(
            service=0,
            children=(
                CallEdgeSpec(
                    child=CallSpec(service=1),
                    probability=1.0,
                    timeout_s=0.8,
                    max_retries=1,
                    request_kb=1.5,
                    response_kb=1.0,
                ),
                CallEdgeSpec(
                    child=CallSpec(
                        service=2,
                        children=(
                            CallEdgeSpec(
                                child=CallSpec(service=4),
                                probability=0.4,
                                timeout_s=0.6,
                                max_retries=1,
                                request_kb=1.0,
                                response_kb=1.0,
                            ),
                        ),
                    ),
                    probability=1.0,
                    timeout_s=1.0,
                    max_retries=1,
                    request_kb=2.0,
                    response_kb=6.0,
                ),
            ),
        ),
        arrival=ArrivalSpec(
            "poisson",
            8.0,
            start_s=_ARRIVAL_START_S,
            rate_schedule=_RPC_RATE_STEP,
        ),
        slo_latency_s=1.0,
    )
    # Analytical path: a second, heavier root with its own SLO, so a policy cannot
    # optimise one template's placement without paying somewhere else.
    batch = TemplateSpec(
        id=1,
        name="batch",
        root=CallSpec(
            service=3,
            children=(
                CallEdgeSpec(
                    child=CallSpec(service=2),
                    probability=0.6,
                    timeout_s=1.5,
                    max_retries=0,
                    request_kb=4.0,
                    response_kb=12.0,
                ),
                CallEdgeSpec(
                    child=CallSpec(service=4),
                    probability=1.0,
                    timeout_s=1.0,
                    max_retries=1,
                    request_kb=2.0,
                    response_kb=2.0,
                ),
            ),
        ),
        arrival=ArrivalSpec(
            "poisson",
            2.0,
            start_s=_ARRIVAL_START_S,
            rate_schedule=_BATCH_RATE_STEP,
        ),
        slo_latency_s=2.5,
    )
    return WorkloadSpec(
        services=(gateway, auth, catalog, recommender, ledger),
        templates=(rpc, batch),
        backends=(
            BackendSpec("db", ((1.0, 0.003), (12.0, 0.030))),
            BackendSpec("kv", ((1.0, 0.001), (20.0, 0.012))),
        ),
    )


def medium_cluster() -> EpisodeConfig:
    """Build the ``dev-medium-cluster`` episode config (deterministic, no RNG)."""
    return EpisodeConfig(
        cluster=_cluster(),
        workload=_workload(),
        horizon_s=_HORIZON_S,
        arrival_cutoff_s=_ARRIVAL_CUTOFF_S,
        scheduler=SchedulerParams(bind_latency=det(0.01)),
        hpa=HpaParams(enabled=True, sync_period_s=8.0, target_utilization=0.5),
        rebalancer=RebalancerParams(
            enabled=True,
            window_s=18.0,
            budget_per_window=3,
            unavailability_precharge_s=1.0,
        ),
        # Empty, suspendable fog nodes power down: consolidation is worth real joules
        # and spreading is not.  Far-edge boards have no published transition data and
        # stay awake by construction (NodeClassSpec suspendability rule).
        power=PowerParams(suspend_after_idle_s=8.0),
        snapshot_times_s=(15.0, 32.0, 50.0),
        snapshot_cycles=(0, 1, 2, 3, 4, 5, 6, 7),
    )


# ------------------------------------------------------------------ manifest + seal
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_payload() -> dict[str, object]:
    return {
        "schema": "evolved-continuum-dev-scenario-manifest-v1",
        "data_role": DATA_ROLE,
        "scenario_id": DEV_MEDIUM_CLUSTER_ID,
        "evaluation_seeds": list(DEV_MEDIUM_CLUSTER_SEEDS),
        "horizon_s": _HORIZON_S,
        "arrival_cutoff_s": _ARRIVAL_CUTOFF_S,
        "generated_by": "sim/scenarios/dev_medium.py (deterministic; no RNG)",
        "generator_sha256": _sha256_file(Path(__file__)),
        "files": {
            DEV_MEDIUM_CLUSTER_PATH.name: _sha256_file(DEV_MEDIUM_CLUSTER_PATH),
        },
        "config_sha256": config_sha256(load_episode_config(DEV_MEDIUM_CLUSTER_PATH)),
        "quarantine_rule": (
            "development-only: never acceptance evidence, never part of the "
            "public/private split machinery, never confirmatory"
        ),
    }


def _seal(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((MANIFEST_SEAL_TAG + canonical).encode()).hexdigest()


def verify_manifest() -> dict[str, object]:
    """Fail closed unless the fixture bytes, generator bytes and seal all agree."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing dev scenario manifest: {MANIFEST_PATH}")
    recorded = json.loads(MANIFEST_PATH.read_text())
    seal = recorded.get("seal")
    payload = {key: value for key, value in recorded.items() if key != "seal"}
    if _seal(payload) != seal:
        raise ValueError("dev scenario manifest seal mismatch — payload was edited")
    expected = _manifest_payload()
    if payload != expected:
        differing = sorted(
            key for key in set(payload) | set(expected) if payload.get(key) != expected.get(key)
        )
        raise ValueError(f"dev scenario manifest drifted from the tree: {differing}")
    return recorded


def load_dev_medium_cluster() -> EpisodeConfig:
    """Load the sealed fixture, verifying the manifest first (fail-closed)."""
    verify_manifest()
    return load_episode_config(DEV_MEDIUM_CLUSTER_PATH)


def main() -> None:
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    save_episode_config(medium_cluster(), DEV_MEDIUM_CLUSTER_PATH)
    payload = _manifest_payload()
    manifest = {**payload, "seal": _seal(payload)}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DEV_MEDIUM_CLUSTER_PATH} and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
