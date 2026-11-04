"""Shared builders for the DES verification suite (TODO 2.5).

All scenarios built here are development-fixture material; the tests must never consume
the acceptance fixture set (T6 fixture discipline — see the README in that directory).
"""

from __future__ import annotations

from pathlib import Path

from sim.spec import (
    ArrivalSpec,
    BackendSpec,
    CallEdgeSpec,
    CallSpec,
    ClusterSpec,
    EnergyCoefficients,
    EpisodeConfig,
    NodeClassSpec,
    NodeSpec,
    SchedulerParams,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    det,
    expo,
)

REPO = Path(__file__).resolve().parent.parent
DEV_FIXTURES = REPO / "sim" / "fixtures" / "dev"
POLICIES = REPO / "runtime" / "policies"

EDGE_CLASS = NodeClassSpec(
    name="edge",
    power_idle_w=3.0,
    power_max_w=8.0,
    suspend_energy_j=5.0,
    suspend_latency_s=2.0,
    wake_energy_j=8.0,
    wake_latency_s=3.0,
    suspended_power_w=0.5,
)
CLOUD_CLASS = NodeClassSpec(name="cloud", power_idle_w=100.0, power_max_w=300.0)
PLAIN_EDGE = NodeClassSpec(name="plain-edge", power_idle_w=2.0, power_max_w=6.0)


def two_edge_cluster(
    cpu: int = 4000, mem: int = 8192, latency: float = 5.0
) -> ClusterSpec:
    return ClusterSpec(
        classes=(EDGE_CLASS,),
        nodes=(NodeSpec(0, "edge", 2, cpu, mem), NodeSpec(1, "edge", 2, cpu, mem)),
        latency_ms=((0.0, latency), (latency, 0.0)),
    )


def mixed_cluster(suspended_third: bool = False) -> ClusterSpec:
    return ClusterSpec(
        classes=(EDGE_CLASS, CLOUD_CLASS),
        nodes=(
            NodeSpec(0, "cloud", 0, 8000, 16384),
            NodeSpec(1, "edge", 2, 4000, 8192),
            NodeSpec(2, "edge", 2, 4000, 8192, initially_suspended=suspended_third),
        ),
        latency_ms=((0.0, 20.0, 20.0), (20.0, 0.0, 5.0), (20.0, 5.0, 0.0)),
    )


def simple_service(
    sid: int = 0,
    name: str = "svc",
    mean_s: float = 0.02,
    replicas: int = 1,
    **kwargs,
) -> ServiceSpec:
    defaults = dict(
        id=sid,
        name=name,
        app=0,
        cpu_request=500,
        mem_request=256,
        service_time=expo(mean_s),
        cpu_demand=500,
        image_mb=10,
        initial_replicas=replicas,
        max_replicas=max(8, replicas),
    )
    defaults.update(kwargs)
    return ServiceSpec(**defaults)


def single_service_config(
    cluster: ClusterSpec,
    rate: float = 5.0,
    horizon_s: float = 30.0,
    service_kwargs: dict | None = None,
    **config_kwargs,
) -> EpisodeConfig:
    svc = simple_service(**(service_kwargs or {}))
    template = TemplateSpec(
        id=0, name="t", root=CallSpec(service=0), arrival=ArrivalSpec("poisson", rate)
    )
    return EpisodeConfig(
        cluster=cluster,
        workload=WorkloadSpec(services=(svc,), templates=(template,)),
        horizon_s=horizon_s,
        **config_kwargs,
    )


def fanout_config(horizon_s: float = 60.0, rate: float = 8.0, **kwargs) -> EpisodeConfig:
    """front -> {back (blocking, timeout+retry), side (blocking)} with a shared backend."""
    front = simple_service(0, "front", 0.02, replicas=2)
    back = simple_service(1, "back", 0.03, backend="db")
    side = simple_service(2, "side", 0.01)
    template = TemplateSpec(
        id=0,
        name="fanout",
        root=CallSpec(
            service=0,
            children=(
                CallEdgeSpec(child=CallSpec(service=1), timeout_s=0.8, max_retries=1),
                CallEdgeSpec(child=CallSpec(service=2), probability=0.5),
            ),
        ),
        arrival=ArrivalSpec("poisson", rate),
        slo_latency_s=0.5,
    )
    return EpisodeConfig(
        cluster=mixed_cluster(),
        workload=WorkloadSpec(
            services=(front, back, side),
            templates=(template,),
            backends=(BackendSpec("db", ((1.0, 0.002), (10.0, 0.02))),),
        ),
        horizon_s=horizon_s,
        arrival_cutoff_s=horizon_s * 0.8,
        **kwargs,
    )


def flat_scorer_fn(pod: dict, node: dict, ctx: dict) -> float:
    """Constant score: quantization sends everything to 100, tie-break -> lowest id."""
    del pod, node, ctx
    return 1.0


def spread_scorer_fn(pod: dict, node: dict, ctx: dict) -> float:
    """least-allocated-style spreading (pure Python baseline path)."""
    del ctx
    cpu_free = node["cpu_capacity"] - node["cpu_allocated"] - pod["cpu_request"]
    return cpu_free / max(node["cpu_capacity"], 1)


def det_bind() -> SchedulerParams:
    return SchedulerParams(bind_latency=det(0.005))


def cheap_energy() -> EnergyCoefficients:
    return EnergyCoefficients(version="test-0")


def v2021_dump_bytes(payload: str | bytes) -> bytes:
    """Rewrite a documented-schema v2021 fixture CSV into the observed dump schema.

    The published dumps carry an unnamed pandas-style leading index column, and the
    MSRTQps metric-name column is ``metric`` (singular) — both contradicting the
    pinned trace README (see the parse_v2021_* docstrings in sim/scenarios/alibaba.py).
    """
    text = payload.decode() if isinstance(payload, bytes) else payload
    lines = text.strip("\n").split("\n")
    header = "," + lines[0].replace("metrics,value", "metric,value")
    rows = [f"{index},{line}" for index, line in enumerate(lines[1:])]
    return ("\n".join([header, *rows]) + "\n").encode()
