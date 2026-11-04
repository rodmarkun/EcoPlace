"""Quarantined **development** scenario family ``dev-continuum-*`` (discrimination-first).

Why this file exists
--------------------
``experiments/landscape_20260825/LANDSCAPE.md`` measured the existing development
fixtures and found them structurally unable to discriminate between search methods:

* **Suspend/wake was dead.** The published fog transition data has a 182 s wake
  latency; every existing fixture runs a 65 s horizon, so a fog node that suspends can
  never return inside the episode. The only fog decision was the one-way "let it go".
* **The SLO never bound.** The Alibaba builder sets
  ``slo_latency_s = max(1.5, 8.0 * critical_p95)``. With a critical-path service p95
  near 0.19 s the 1.5 s floor always won, leaving ~1.3 s of slack against a worst-case
  cross-tier network path of ~0.14 s. Nothing a placement policy did could threaten it,
  so **dependency-aware placement had no price signal at all** — the paper's headline
  capability was unpriced by its own benchmark.
* **Capacity never bound.** Twelve far-edge boards supply 48,000 mc against a peak
  requestable demand near 8,300 mc. Far edge idles at 1.8 W against a fog server's
  17.8-108 W, so "put everything on far edge" dominated unconditionally and the
  landscape collapsed onto one plateau: 72-74% of GP and random-AST offspring landed on
  a single energy value, and a random tree at generation 0 beat 42 candidates of guided
  search.

This family fixes all three **inside the same sealed evidence and the same energy
bundles**; nothing here invents data. The levers are horizon, node counts, an SLO slack
multiplier and a resource scale — every one a stated modelling choice, recorded in the
manifest, and every one applied identically to all methods.

What each lever buys
--------------------
``HORIZON_S = 420 s`` (> 2x the published 182 s fog wake latency)
    Suspending a fog node is now a *recoverable* decision, so "consolidate and suspend"
    trades against "keep capacity warm for the flash crowd" instead of being free.
``SLO_SLACK_MULTIPLIER = 2.3``, no floor
    The end-to-end SLO becomes a multiple of each template's own critical-path service
    p95 instead of a flat 1.5 s, so placement-induced latency is finally inside the
    constraint. Combined with the widened latency profile below, co-locating a call
    chain now pays and splitting it now costs. This is what makes ``ctx.graph`` worth
    reading — and it composes with the ``sim.scheduler._pod_level_graph`` prospective-edge
    fix, without which a *scoring* policy cannot observe its own chain at all.
``10 far-edge boards`` at the bundle-bound 4,000 mc, ``CPU_REQUEST_SCALE = 1.9``
    Swept against the locked node table until the *kind* of policy that survives is the
    kind the paper is about. Worst SLO satisfaction over 2 seeds x 3 bundles against the
    0.90 target, with per-episode closed-boundary energy:

    ====================  =========  ========  ==========
    policy                worst SLO  feasible  energy
    ====================  =========  ========  ==========
    ``most_allocated``      0.000    no         36 kJ
    ``energy_greedy``       0.025    no         37 kJ
    ``first_fit``           0.882    no         42 kJ
    ``least_allocated``     0.883    no        107 kJ
    ``network_overhead``    0.926    **yes**    62 kJ
    ``latency_greedy``      0.926    **yes**    62 kJ
    ``graph_headroom``      0.934    **yes**   105 kJ
    ====================  =========  ========  ==========

    **Every feasible hand-written policy reads the call graph and every infeasible one
    does not.** That is the property the old fixtures could not produce and the reason
    this family exists: dependency awareness is the price of admission rather than an
    unpriced decoration. The headroom left over is real and large — ``first_fit`` reaches
    42 kJ but misses the SLO by 0.018, so a policy that consolidates like ``first_fit``
    while holding its chains together has roughly a third of the best feasible
    baseline's energy still to take.

The bundle structure is the other reason the rebuild is worth it: on ``dev-1`` a fog
server is 357,193 mc for 108 W and waking it is excellent; on ``dev-3`` it is 4,000 mc
for 17.8 W and waking it is strictly worse than a far-edge board. **The right policy is
bundle-conditioned**, which is exactly the model-uncertainty structure §4.1 claims to
evaluate and which the old fixtures could not express.

Construction
------------
The workload comes from the *unmodified* sealed Alibaba builder
(``sim.scenarios.alibaba_workload``) so its calibration evidence and seals are reused
byte-for-byte; this module then applies two deterministic, explicitly-recorded
transforms — a resource scale and an SLO retightening computed from the built
lognormal service-time parameters. Editing the builder itself would have drifted the
three existing sealed fixtures, so the modelling choices live here instead.

Development/quarantined data. Never acceptance evidence, never part of the
public/private split, never confirmatory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.tracking import derive_seed
from sim.scenarios.alibaba_workload import (
    BUILDER_VERSION,
    CALIBRATION_SHA256,
    CALL_GRAPH_SAMPLE_SHA256,
    AlibabaWorkloadBuildConfig,
    CalibrationInformedWorkload,
    build_calibration_informed_workload,
)
from sim.scenarios.sources import load_source_registry
from sim.spec import (
    CallSpec,
    ClusterSpec,
    EpisodeConfig,
    HpaParams,
    NodeClassSpec,
    NodeSpec,
    PowerParams,
    RebalancerParams,
    SchedulerParams,
    WorkloadSpec,
    config_sha256,
    det,
    load_episode_config,
    save_episode_config,
)

HERE = Path(__file__).resolve().parent
DEV_DIR = HERE / "dev"
DATA_ROLE = "development-only-quarantined"
MANIFEST_TAG = "dev-tight-scenario-manifest-v1:"
ROOT_SEED = 20260825

# --------------------------------------------------------------------- design levers
# Longer than 2x the published fog wake latency (182 s), so suspension is a recoverable
# decision rather than a one-way door.
HORIZON_S = 420.0
ARRIVAL_START_S = 10.0
ARRIVAL_SUPPORT_S = 360.0
# Strictly after the builder's last scheduled rate change, which lands on the end of
# the arrival support; the spec refuses a cutoff that coincides with one.
ARRIVAL_CUTOFF_S = ARRIVAL_START_S + ARRIVAL_SUPPORT_S + 2.0
# End-to-end SLO as a multiple of the critical-path service p95, with **no floor**: the
# builder's own max(1.5, 8.0 * p95) rule guarantees ~8x slack and is why no fixture has
# ever priced a placement-induced latency.
SLO_SLACK_MULTIPLIER = 2.3
# cpu_request/cpu_demand scale. Far-edge capacity is bundle-bound at 4,000 mc/board and
# cannot be chosen here, so demand is the only side of the ratio this module owns.
CPU_REQUEST_SCALE = 1.9
MEM_REQUEST_SCALE = 1.0
# Long enough that keeping a fog node warm is a deliberate act, short enough that an
# unused node still gives its idle draw back within the horizon.
SUSPEND_AFTER_IDLE_S = 30.0
# Bounded per-replica FIFO. The Alibaba builder leaves queues unbounded, which makes
# an overloading policy both unrealistic (real services shed load) and *expensive*:
# `most_allocated` left 57% of requests in flight at the horizon and its episode cost
# an order of magnitude more wall-clock than a sane policy's, so pathological
# candidates dominated campaign throughput. With a bound, overload shows up as drops
# against the pre-registered drop cap instead of as an unbounded latency tail.
QUEUE_CAPACITY_PER_REPLICA = 48

DEV_CONTINUUM_TIGHT_ID = "dev-continuum-tight"
DEV_CONTINUUM_TIGHT_PATH = DEV_DIR / "continuum_tight.json"
DEV_CONTINUUM_TIGHT_SEEDS = (1000151, 1000159, 1000171, 1000183)

# Transfer target: same generator, different topology shape and site structure, so a
# champion evolved on ``dev-continuum-tight`` can be scored zero-shot on a cluster it
# never saw. Kept in this module so both share one manifest discipline.
DEV_CONTINUUM_WIDE_ID = "dev-continuum-wide"
DEV_CONTINUUM_WIDE_PATH = DEV_DIR / "continuum_wide.json"
DEV_CONTINUUM_WIDE_SEEDS = (1000201, 1000213, 1000217, 1000231)

# Drift target for the RQ2 adaptation question: the same cluster and workload as
# ``dev-continuum-tight``, plus an operational shock the policy never saw evolving --
# three nodes cordoned mid-episode and returned later, and a sustained demand step on
# top of the flash crowd. A frozen champion faces this with no adaptation budget (the
# plan's no-adaptation control, S9/U7); a re-evolved champion gets a budgeted episode.
DEV_CONTINUUM_DRIFT_ID = "dev-continuum-drift"
DEV_CONTINUUM_DRIFT_PATH = DEV_DIR / "continuum_drift.json"
# The *same* evaluation seeds as the tight registration, deliberately: the drift
# contrast is paired, so the arrival draws must be common random numbers and the node
# churn plus demand step must be the only difference between the two configs.
DEV_CONTINUUM_DRIFT_SEEDS = DEV_CONTINUUM_TIGHT_SEEDS
# Cordon after the cluster has settled and well before the flash; recover late enough
# that the policy has to survive the peak with reduced capacity.
DRIFT_CORDON_S = 150.0
DRIFT_RECOVERY_S = 330.0
DRIFT_DOWN_NODES = 3
# Swept for the regime RQ2 needs: the shock has to break a policy that was feasible
# before it, without breaking every policy. At (1.10, 3 nodes) the seed survives
# (0.951 worst SLO) while the cheapest feasible baseline, latency_greedy, falls out
# of feasibility (0.926 -> 0.879), so a no-adaptation control demonstrably fails and
# re-adaptation has something to recover. At (1.18, 3) everything fails and the
# endpoint says nothing; at (1.10, 5) cordoning *helps*, because fewer available
# nodes means less cross-site spread on this latency profile.
DRIFT_RATE_STEP_S = 210.5
DRIFT_RATE_STEP_MULTIPLIER = 1.10

_CLOUD = "cloud-region"
_CLOUD_EDGEY = "cloud-zone"
_FOG = "fog"
_EDGE = "edge"

# (node_id, class, tier, cpu_capacity, mem_capacity_mib, site)
#
# Tier-1/tier-2 ``cpu_capacity`` is a structural placeholder: the sealed capacity bridge
# (``evolution.evaluation.config_for_development_bundle``) overwrites it per bundle.
# Memory and the latency matrix are the per-node heterogeneity this fixture owns.
#
# Ids interleave the tiers and memory is not monotone in id, for the reason
# ``dev_medium`` documents: grouped ids make stable-id order coincide with latency
# proximity and collapse first-fit, latency-greedy and network-overhead onto one point
# as a numbering artefact.
_TIGHT_NODES: tuple[tuple[int, str, int, int, int, int], ...] = (
    (0, _EDGE, 2, 4000, 3072, 0),
    (1, _CLOUD, 0, 16000, 32768, -1),
    (2, _EDGE, 2, 4000, 6144, 1),
    (3, _FOG, 1, 16000, 24576, 0),
    (4, _EDGE, 2, 4000, 2048, 0),
    (5, _CLOUD_EDGEY, 0, 12000, 24576, -1),
    (6, _EDGE, 2, 4000, 4096, 2),
    (7, _FOG, 1, 16000, 12288, 1),
    (8, _EDGE, 2, 4000, 8192, 1),
    (9, _EDGE, 2, 4000, 2560, 3),
    (10, _FOG, 1, 16000, 6144, 2),
    (11, _EDGE, 2, 4000, 5120, 2),
    (12, _EDGE, 2, 4000, 1536, 3),
    (13, _EDGE, 2, 4000, 4096, 0),
    (14, _EDGE, 2, 4000, 3072, 3),
)

# Wider, flatter transfer cluster: more far-edge sites, one fog anchor, same classes.
_WIDE_NODES: tuple[tuple[int, str, int, int, int, int], ...] = (
    (0, _EDGE, 2, 4000, 4096, 0),
    (1, _EDGE, 2, 4000, 2048, 4),
    (2, _CLOUD, 0, 16000, 32768, -1),
    (3, _EDGE, 2, 4000, 6144, 2),
    (4, _FOG, 1, 16000, 12288, 0),
    (5, _EDGE, 2, 4000, 3072, 5),
    (6, _EDGE, 2, 4000, 8192, 1),
    (7, _EDGE, 2, 4000, 2560, 3),
    (8, _FOG, 1, 16000, 6144, 3),
    (9, _EDGE, 2, 4000, 5120, 0),
    (10, _EDGE, 2, 4000, 1536, 4),
    (11, _CLOUD_EDGEY, 0, 12000, 24576, -1),
    (12, _EDGE, 2, 4000, 4096, 2),
    (13, _FOG, 1, 16000, 24576, 5),
    (14, _EDGE, 2, 4000, 2048, 1),
    (15, _EDGE, 2, 4000, 6144, 5),
    (16, _EDGE, 2, 4000, 3072, 4),
)


@dataclass(frozen=True)
class TightScenarioRegistration:
    scenario_id: str
    path: Path
    manifest_path: Path
    evaluation_seeds: tuple[int, ...]
    nodes: tuple[tuple[int, str, int, int, int, int], ...]
    workload: AlibabaWorkloadBuildConfig
    flash_multiplier: float
    drifted: bool = False


def _seed(scenario_id: str, component: str) -> int:
    return derive_seed(ROOT_SEED, "sim", "scenarios", scenario_id, component)


def _workload_config(scenario_id: str, *, rate_scale: float) -> AlibabaWorkloadBuildConfig:
    """Arrivals stretched across the long horizon, with a flash crowd in its second half.

    The flash is what turns far-edge saturation into a *decision*: below it the boards
    absorb everything, above it the policy must choose cloud allocation or a fog wake.
    """
    return AlibabaWorkloadBuildConfig(
        seed=_seed(scenario_id, "workload-builder"),
        arrival_rate_scale=rate_scale,
        arrival_start_s=ARRIVAL_START_S,
        arrival_support_s=ARRIVAL_SUPPORT_S,
        flash_start_s=ARRIVAL_START_S + 0.55 * ARRIVAL_SUPPORT_S,
        flash_end_s=ARRIVAL_START_S + 0.78 * ARRIVAL_SUPPORT_S,
        flash_multiplier=1.9,
    )


REGISTRATIONS = (
    TightScenarioRegistration(
        DEV_CONTINUUM_TIGHT_ID,
        DEV_CONTINUUM_TIGHT_PATH,
        DEV_DIR / "continuum_tight.manifest.json",
        DEV_CONTINUUM_TIGHT_SEEDS,
        _TIGHT_NODES,
        _workload_config(DEV_CONTINUUM_TIGHT_ID, rate_scale=0.42),
        1.9,
    ),
    TightScenarioRegistration(
        DEV_CONTINUUM_WIDE_ID,
        DEV_CONTINUUM_WIDE_PATH,
        DEV_DIR / "continuum_wide.manifest.json",
        DEV_CONTINUUM_WIDE_SEEDS,
        _WIDE_NODES,
        _workload_config(DEV_CONTINUUM_WIDE_ID, rate_scale=0.42),
        1.9,
    ),
    TightScenarioRegistration(
        DEV_CONTINUUM_DRIFT_ID,
        DEV_CONTINUUM_DRIFT_PATH,
        DEV_DIR / "continuum_drift.manifest.json",
        DEV_CONTINUUM_DRIFT_SEEDS,
        _TIGHT_NODES,
        _workload_config(DEV_CONTINUUM_TIGHT_ID, rate_scale=0.42),
        1.9,
        drifted=True,
    ),
)
_BY_ID = {item.scenario_id: item for item in REGISTRATIONS}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workload_digest(workload: WorkloadSpec) -> str:
    """Digest of the *transformed* workload, so the manifest pins what actually runs."""
    return hashlib.sha256(
        (
            "dev-tight-post-transform-workload-v1:"
            + json.dumps(
                _workload_plain(workload), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        ).encode()
    ).hexdigest()


def _workload_plain(workload: WorkloadSpec) -> dict[str, object]:
    def call(node: CallSpec) -> dict[str, object]:
        return {
            "service": node.service,
            "mode": node.mode,
            "children": [
                {
                    "child": call(edge.child),
                    "probability": edge.probability,
                    "request_kb": edge.request_kb,
                    "response_kb": edge.response_kb,
                    "timeout_s": edge.timeout_s,
                    "max_retries": edge.max_retries,
                }
                for edge in node.children
            ],
        }

    return {
        "services": [
            {
                "id": s.id,
                "name": s.name,
                "cpu_request": s.cpu_request,
                "cpu_demand": s.cpu_demand,
                "mem_request": s.mem_request,
                "initial_replicas": s.initial_replicas,
                "min_replicas": s.min_replicas,
                "max_replicas": s.max_replicas,
                "service_time": {
                    "kind": s.service_time.kind,
                    "params": dict(s.service_time.params),
                },
            }
            for s in workload.services
        ],
        "templates": [
            {"id": t.id, "name": t.name, "slo_latency_s": t.slo_latency_s, "root": call(t.root)}
            for t in workload.templates
        ],
    }


def _latency_ms(
    nodes: tuple[tuple[int, str, int, int, int, int], ...],
) -> tuple[tuple[float, ...], ...]:
    """Symmetric, zero-diagonal, site-structured matrix (ms).

    Widened from ``dev_medium``'s profile on purpose. At that profile a co-located call
    chain saved ~70 ms against a queueing term that could reach seconds, so co-location
    was always the losing move and every dependency-aware policy was dominated: the
    scenario was contention-only, which is a bin-packing problem, not a placement one.
    Here a cross-site chain pays ~40 ms per hop (~160 ms over a depth-3 template) against
    an SLO near 0.35 s, so **neither** term dominates: packing alone dies of queueing and
    spreading alone dies of wire latency. The optimum needs both, which is the two-factor
    structure a short *program* can express and a one-line greedy cannot.
    """

    Node = tuple[int, str, int, int, int, int]

    def cost(a: Node, b: Node) -> float:
        if a[0] == b[0]:
            return 0.0
        tiers = (min(a[2], b[2]), max(a[2], b[2]))
        same_site = a[5] == b[5]
        if tiers == (0, 0):
            return 1.0
        if tiers == (0, 1):
            return 30.0
        if tiers == (0, 2):
            return 60.0
        if tiers == (1, 1):
            return 4.0 if same_site else 12.0
        if tiers == (1, 2):
            return 5.0 if same_site else 35.0
        return 2.0 if same_site else 40.0

    return tuple(tuple(cost(a, b) for b in nodes) for a in nodes)


def _cluster(nodes: tuple[tuple[int, str, int, int, int, int], ...]) -> ClusterSpec:
    classes = (
        # Tier-0 carries no device energy (cloud is allocation-metered); the power
        # fields remain observations available to every scorer.
        NodeClassSpec(name=_CLOUD, power_idle_w=110.0, power_max_w=340.0),
        NodeClassSpec(name=_CLOUD_EDGEY, power_idle_w=70.0, power_max_w=210.0),
        # Tier-1/tier-2 classes are replaced wholesale by the sealed bundle bridge.
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
    node_specs = tuple(
        NodeSpec(node_id, class_name, tier, cpu, mem)
        for node_id, class_name, tier, cpu, mem, _site in nodes
    )
    return ClusterSpec(classes=classes, nodes=node_specs, latency_ms=_latency_ms(nodes))


# ------------------------------------------------------------------ workload transforms
def _lognormal_p95(service: Any) -> float:  # noqa: ANN401 - ServiceSpec, avoid cycle
    """p95 of the built service-time distribution.

    The Alibaba builder emits ``lognormal(mu, sigma)``; ``exp(mu + 1.645 sigma)`` is its
    95th percentile. Deterministic distributions carry their own value. Anything else
    is refused rather than silently approximated, because the SLO derives from it.
    """
    dist = service.service_time
    if dist.kind == "lognormal":
        return float(math.exp(dist.params["mu"] + 1.6448536269514722 * dist.params["sigma"]))
    if dist.kind == "deterministic":
        return float(dist.params["value"])
    raise ValueError(f"cannot derive an SLO from service-time kind {dist.kind!r}")


def _critical_path_p95(call: CallSpec, by_id: dict[int, Any]) -> float:
    """Own p95 plus the worst child branch: blocking fan-in waits for the slowest child."""
    own = _lognormal_p95(by_id[call.service])
    return own + max(
        (_critical_path_p95(edge.child, by_id) for edge in call.children), default=0.0
    )


def _scaled_services(workload: WorkloadSpec) -> tuple[Any, ...]:
    """Scale requests/demand so peak offered load approaches far-edge capacity.

    ``cpu_request`` and ``cpu_demand`` scale together: scaling only the request would
    make the scheduler's view disagree with the node contention the DES actually
    simulates, and scaling only the demand would leave HPA blind to the change.
    """
    out = []
    for service in workload.services:
        out.append(
            dataclasses.replace(
                service,
                cpu_request=max(1, int(round(service.cpu_request * CPU_REQUEST_SCALE))),
                cpu_demand=max(1, int(round(service.cpu_demand * CPU_REQUEST_SCALE))),
                mem_request=max(1, int(round(service.mem_request * MEM_REQUEST_SCALE))),
                queue_capacity=QUEUE_CAPACITY_PER_REPLICA,
            )
        )
    return tuple(out)


def _retightened_workload(workload: WorkloadSpec) -> WorkloadSpec:
    """Apply the resource scale and recompute every template SLO from its own critical path."""
    services = _scaled_services(workload)
    by_id = {service.id: service for service in services}
    templates = tuple(
        dataclasses.replace(
            template,
            slo_latency_s=round(SLO_SLACK_MULTIPLIER * _critical_path_p95(template.root, by_id), 6),
        )
        for template in workload.templates
    )
    return dataclasses.replace(workload, services=services, templates=templates)


def _episode(cluster: ClusterSpec, workload: WorkloadSpec) -> EpisodeConfig:
    return EpisodeConfig(
        cluster=cluster,
        workload=workload,
        horizon_s=HORIZON_S,
        arrival_cutoff_s=ARRIVAL_CUTOFF_S,
        scheduler=SchedulerParams(bind_latency=det(0.01)),
        hpa=HpaParams(enabled=True, sync_period_s=6.0, target_utilization=0.50),
        rebalancer=RebalancerParams(
            enabled=True,
            window_s=30.0,
            budget_per_window=3,
            unavailability_precharge_s=1.0,
            max_total_bytes_mb=1536.0,
            max_total_moves=40,
            max_total_unavailability_s=180.0,
        ),
        power=PowerParams(suspend_after_idle_s=SUSPEND_AFTER_IDLE_S),
        # Four decision points spread across the horizon: before the flash, at its
        # onset, at its peak and in the recovery tail.
        snapshot_times_s=(90.0, 220.0, 300.0, 380.0),
        snapshot_cycles=(0, 1, 2, 3, 4, 5, 6, 7),
    )


def drift_plan_for(
    nodes: tuple[tuple[int, str, int, int, int, int], ...], workload: WorkloadSpec
):
    """The sealed operational shock applied to the drift registration.

    Built from the shared Phase-4 drift machinery rather than a bespoke edit, so the
    plan carries its own digest and ``assert_drift_applied`` can check that the saved
    config really contains every effect the plan names.

    Node churn is the part the shared machinery expresses; the demand step is applied
    separately by :func:`_with_sustained_demand_step` and recorded in the manifest. A
    single ``RateChange`` cannot express a *sustained* step here: the calibrated builder
    emits a schedule point every 1.6 s, so one inserted point is overridden 1.1 s later
    and measured as a no-op (first_fit moved from 0.882 to 0.907 SLO, i.e. the opposite
    direction, when it was tried that way).
    """
    from sim.scenarios.drift import churn_schedule, make_drift_plan

    del workload  # kept in the signature so callers pass the workload the step uses
    churn = churn_schedule(
        node_count=len(nodes),
        down_count=DRIFT_DOWN_NODES,
        down_s=DRIFT_CORDON_S,
        recovery_s=DRIFT_RECOVERY_S,
        seed=_seed(DEV_CONTINUUM_DRIFT_ID, "churn"),
    )
    return churn, make_drift_plan(
        seed=_seed(DEV_CONTINUUM_DRIFT_ID, "plan"),
        rate_changes=(),
        node_churn=churn,
        mix_shift_statement="none",
    )


def _with_sustained_demand_step(workload: WorkloadSpec) -> WorkloadSpec:
    """Multiply every scheduled arrival rate at or after the step time.

    This is the operating-point change the RQ2 episode is about: it persists past the
    flash crowd instead of decaying with it, so a policy tuned for the pre-drift regime
    stays wrong rather than being rescued by the workload returning to normal.
    """
    templates = []
    for template in workload.templates:
        arrival = template.arrival
        schedule = tuple(
            (at_s, rate * DRIFT_RATE_STEP_MULTIPLIER if at_s >= DRIFT_RATE_STEP_S else rate)
            for at_s, rate in arrival.rate_schedule
        )
        templates.append(
            dataclasses.replace(
                template, arrival=dataclasses.replace(arrival, rate_schedule=schedule)
            )
        )
    return dataclasses.replace(workload, templates=tuple(templates))


def build_registered_scenario(
    scenario_id: str,
) -> tuple[EpisodeConfig, CalibrationInformedWorkload]:
    """Pure deterministic builder for one module-owned scenario registration."""
    try:
        registration = _BY_ID[scenario_id]
    except KeyError as exc:
        raise KeyError(f"unknown tight development scenario {scenario_id!r}") from exc
    registry = load_source_registry()
    build = build_calibration_informed_workload(registration.workload, registry)
    workload = _retightened_workload(build.workload)
    config = _episode(_cluster(registration.nodes), workload)
    if registration.drifted:
        from sim.scenarios.drift import apply_drift, assert_drift_applied

        _churn, plan = drift_plan_for(registration.nodes, workload)
        config = apply_drift(config, plan)
        assert_drift_applied(config, plan)
        config = dataclasses.replace(
            config, workload=_with_sustained_demand_step(config.workload)
        )
    return config, build


def _manifest_payload(registration: TightScenarioRegistration) -> dict[str, object]:
    config, build = build_registered_scenario(registration.scenario_id)
    registry = load_source_registry()
    # `seal_calibration_informed_synthetic_workload` is fail-closed on
    # `config.workload != build.workload` and is therefore **not** used here: this
    # family deliberately transforms the built workload, so claiming the builder's
    # synthetic-workload seal for it would be a provenance lie. Instead the builder
    # result is verified against the sealed evidence in its untransformed form, and the
    # transform that follows is recorded explicitly with its own digest.
    build.verify(registry)
    return {
        "schema": "evolved-continuum-dev-tight-scenario-manifest-v1",
        "data_role": DATA_ROLE,
        "scenario_id": registration.scenario_id,
        "evaluation_seeds": list(registration.evaluation_seeds),
        "generator_root_seed": ROOT_SEED,
        "component_seeds": {"workload_builder": registration.workload.seed},
        "design_levers": {
            "horizon_s": HORIZON_S,
            "arrival_start_s": ARRIVAL_START_S,
            "arrival_support_s": ARRIVAL_SUPPORT_S,
            "arrival_cutoff_s": ARRIVAL_CUTOFF_S,
            "slo_slack_multiplier": SLO_SLACK_MULTIPLIER,
            "slo_floor_s": None,
            "far_edge_boards": 10,
            "cpu_request_scale": CPU_REQUEST_SCALE,
            "mem_request_scale": MEM_REQUEST_SCALE,
            "suspend_after_idle_s": SUSPEND_AFTER_IDLE_S,
            "queue_capacity_per_replica": QUEUE_CAPACITY_PER_REPLICA,
            "flash_multiplier": registration.flash_multiplier,
            "rationale": (
                "horizon > 2x the published 182 s fog wake latency; SLO derived from the "
                "critical-path service p95 with no floor so cross-tier chains are priced; "
                "demand scaled until the flash crowd saturates far-edge capacity"
            ),
        },
        "drift": (
            None
            if not registration.drifted
            else {
                "plan_digest": drift_plan_for(registration.nodes, config.workload)[1].digest,
                "churn_digest": drift_plan_for(registration.nodes, config.workload)[0].digest,
                "cordon_s": DRIFT_CORDON_S,
                "recovery_s": DRIFT_RECOVERY_S,
                "down_nodes": DRIFT_DOWN_NODES,
                "demand_step": {
                    "at_s": DRIFT_RATE_STEP_S,
                    "multiplier": DRIFT_RATE_STEP_MULTIPLIER,
                    "applied_by": "sim.scenarios.dev_tight._with_sustained_demand_step",
                    "why_not_a_RateChange": (
                        "the calibrated builder emits a schedule point every 1.6 s, so a "
                        "single inserted RateChange is overridden 1.1 s later and is a no-op"
                    ),
                },
                "role": (
                    "RQ2 adaptation target: a frozen champion meets this with no "
                    "adaptation budget (the no-adaptation control); a re-evolved "
                    "champion gets a budgeted episode from the same starting policy"
                ),
            }
        ),
        "transforms_applied_here": [
            "cpu_request/cpu_demand/mem_request scaled by the recorded factors",
            "template slo_latency_s recomputed as slack x critical-path service p95",
        ],
        "generated_by": "sim/scenarios/dev_tight.py (deterministic)",
        "generator_sha256": _sha256_file(Path(__file__)),
        "workload_builder": {
            "version": BUILDER_VERSION,
            "module_sha256": _sha256_file(HERE / "alibaba_workload.py"),
            "build_sha256": build.digest,
            "calibration_export_sha256": CALIBRATION_SHA256,
            "call_graph_export_sha256": CALL_GRAPH_SAMPLE_SHA256,
            "evidence_seals": [list(pair) for pair in build.evidence_seals],
            "selected_fit_seals": [list(pair) for pair in build.selected_fit_seals],
            "code_versions": list(build.code_versions),
            "provenance_label": (
                "calibration-informed-then-transformed: the Alibaba typed builder output "
                "verified against sealed evidence, then rescaled and SLO-retightened by "
                "sim/scenarios/dev_tight.py as recorded under design_levers"
            ),
            "post_transform_workload_sha256": _workload_digest(config.workload),
        },
        "topology": {
            "family": "hand-specified-continuum-tiers",
            "evidence_label": "synthetic development cluster; tiers/capacities/latencies stated",
            "external_validity_statement": "no placement external-validity claim",
            "node_count": len(registration.nodes),
            "tier_counts": {
                str(tier): sum(1 for item in registration.nodes if item[2] == tier)
                for tier in sorted({item[2] for item in registration.nodes})
            },
        },
        "file": {
            "path": str(registration.path.relative_to(HERE.parents[1])),
            "sha256": _sha256_file(registration.path),
        },
        "config_sha256": config_sha256(load_episode_config(registration.path)),
        "quarantine_rule": (
            "development-only: never acceptance evidence, never public/private split, "
            "never face-validity, never confirmatory"
        ),
    }


def _seal(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        (
            MANIFEST_TAG
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ).encode()
    ).hexdigest()


def verify_tight_manifest(scenario_id: str) -> dict[str, object]:
    """Rebuild the scenario and every closure; reject any manifest/tree drift."""
    registration = _BY_ID[scenario_id]
    if not registration.manifest_path.is_file():
        raise FileNotFoundError(f"missing scenario manifest: {registration.manifest_path}")
    recorded = json.loads(registration.manifest_path.read_text())
    if not isinstance(recorded, dict):
        raise ValueError("development scenario manifest must be an object")
    payload = {key: value for key, value in recorded.items() if key != "seal"}
    if recorded.get("seal") != _seal(payload):
        raise ValueError("development scenario manifest seal mismatch")
    expected = _manifest_payload(registration)
    if payload != expected:
        differing = sorted(
            key for key in set(payload) | set(expected) if payload.get(key) != expected.get(key)
        )
        raise ValueError(f"development scenario manifest drifted from tree: {differing}")
    return recorded


def load_dev_continuum_tight() -> EpisodeConfig:
    verify_tight_manifest(DEV_CONTINUUM_TIGHT_ID)
    return load_episode_config(DEV_CONTINUUM_TIGHT_PATH)


def load_dev_continuum_wide() -> EpisodeConfig:
    verify_tight_manifest(DEV_CONTINUUM_WIDE_ID)
    return load_episode_config(DEV_CONTINUUM_WIDE_PATH)


def main() -> None:
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    for registration in REGISTRATIONS:
        config, _build = build_registered_scenario(registration.scenario_id)
        save_episode_config(config, registration.path)
        payload = _manifest_payload(registration)
        registration.manifest_path.write_text(
            json.dumps({**payload, "seal": _seal(payload)}, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {registration.path} and {registration.manifest_path}")


if __name__ == "__main__":
    main()
