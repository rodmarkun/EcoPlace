"""Seeded replayable Phase-4 drift construction (TODO 4.3)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, replace

from analysis.tracking import spawn_rng
from sim.spec import (
    CallEdgeSpec,
    CallSpec,
    EpisodeConfig,
    NodeChurnSpec,
    ServiceSpec,
    WorkloadSpec,
)

_DRIFT_TAG = "scenario-drift-plan-v1:"
_CHURN_TAG = "scenario-churn-schedule-v1:"
_DRIFT_AUTHORITY = object()
_CHURN_AUTHORITY = object()
_CATEGORY_PRIOR_DEFINITIONS = {
    # Frozen qualitative priors; they shape parent selection only. They are synthetic
    # scenario assumptions, not empirical Alibaba/DSB/TeaStore observations.
    "retail": "front-door-heavy-v1",
    "analytics": "balanced-fanout-v1",
    "media": "pipeline-heavy-v1",
}
CATEGORY_PRIOR_REGISTRY_SHA256 = hashlib.sha256(
    json.dumps(
        _CATEGORY_PRIOR_DEFINITIONS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _canonical(value: object) -> str:
    def jsonable(item: object) -> object:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return {
                field.name: jsonable(getattr(item, field.name))
                for field in dataclasses.fields(item)
            }
        if isinstance(item, tuple | list):
            return [jsonable(value) for value in item]
        if isinstance(item, dict):
            return {key: jsonable(value) for key, value in item.items()}
        return item

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, order=True)
class RateChange:
    at_s: float
    template_id: int
    rate_per_s: float
    cause: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.at_s) or self.at_s <= 0:
            raise ValueError("rate-change time must be finite and > 0")
        if not isinstance(self.template_id, int) or self.template_id < 0:
            raise ValueError("rate-change template id must be a nonnegative integer")
        if not math.isfinite(self.rate_per_s) or self.rate_per_s < 0:
            raise ValueError("rate-change rate must be finite and >= 0")
        if not self.cause:
            raise ValueError("rate-change cause must be non-empty")


@dataclass(frozen=True, order=True)
class OnboardingActivation:
    activation_s: float
    template_id: int
    service_ids: tuple[int, ...]
    category: str
    prior_seed: int
    prior_graph_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.activation_s) or self.activation_s <= 0:
            raise ValueError("onboarding activation must be finite and > 0")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.template_id, *self.service_ids)
        ):
            raise ValueError("onboarding service/template ids must be nonnegative integers")
        if not self.service_ids or tuple(sorted(set(self.service_ids))) != self.service_ids:
            raise ValueError("onboarding service_ids must be non-empty, sorted, and unique")
        if not isinstance(self.prior_seed, int) or isinstance(self.prior_seed, bool):
            raise TypeError("onboarding prior seed must be an explicit int")
        if not self.category:
            raise ValueError("onboarding category must be non-empty")
        if self.category not in _CATEGORY_PRIOR_DEFINITIONS:
            raise ValueError(
                f"unknown onboarding category prior {self.category!r}; "
                f"expected one of {tuple(_CATEGORY_PRIOR_DEFINITIONS)}"
            )
        if len(self.prior_graph_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.prior_graph_sha256
        ):
            raise ValueError("onboarding prior graph needs its SHA-256")


@dataclass(frozen=True)
class ChurnSchedule:
    schema: str
    seed: int
    node_count: int
    down_count: int
    down_s: float
    recovery_s: float
    events: tuple[NodeChurnSpec, ...]
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("digest", "_authority")
        }

    def verify(self) -> None:
        if self._authority is not _CHURN_AUTHORITY:
            raise ValueError("ChurnSchedule must be issued by churn_schedule()")
        if self.schema != "scenario-churn-schedule-v1":
            raise ValueError("unknown churn schedule schema")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("churn seed must be an explicit int")
        if not 0 < self.down_count < self.node_count or not 0 < self.down_s < self.recovery_s:
            raise ValueError("invalid churn schedule dimensions/times")
        if self.events != tuple(sorted(self.events, key=lambda event: (event.at_s, event.node_id))):
            raise ValueError("churn events must be in canonical order")
        if len(self.events) != 2 * self.down_count:
            raise ValueError("churn schedule event count does not match down_count")
        expected = hashlib.sha256((_CHURN_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("churn schedule digest mismatch")


@dataclass(frozen=True)
class DriftPlan:
    schema: str
    seed: int
    rate_changes: tuple[RateChange, ...]
    node_churn: tuple[NodeChurnSpec, ...]
    onboarding: tuple[OnboardingActivation, ...]
    component_seeds: tuple[tuple[str, int], ...]
    mix_shift_statement: str
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("digest", "_authority")
        }

    def verify(self) -> None:
        if self._authority is not _DRIFT_AUTHORITY:
            raise ValueError("DriftPlan must be issued by make_drift_plan()")
        if self.schema != "scenario-drift-plan-v1":
            raise ValueError("unknown drift plan schema")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("drift seed must be an explicit int")
        if self.rate_changes != tuple(sorted(self.rate_changes)):
            raise ValueError("rate changes must be in canonical order")
        identities = [(change.at_s, change.template_id) for change in self.rate_changes]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate rate change for one template at one time")
        if self.onboarding != tuple(sorted(self.onboarding)):
            raise ValueError("onboarding activations must be in canonical order")
        onboarded_services = [service for item in self.onboarding for service in item.service_ids]
        if len(onboarded_services) != len(set(onboarded_services)):
            raise ValueError("a service may be onboarded only once")
        if len({item.template_id for item in self.onboarding}) != len(self.onboarding):
            raise ValueError("an application template may be onboarded only once")
        if self.component_seeds != tuple(sorted(self.component_seeds)):
            raise ValueError("drift component seeds must be in canonical order")
        namespaces = [namespace for namespace, _ in self.component_seeds]
        if len(namespaces) != len(set(namespaces)) or any(
            not namespace for namespace in namespaces
        ):
            raise ValueError("drift component seed namespaces must be non-empty and unique")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for _, seed in self.component_seeds
        ):
            raise TypeError("drift component seeds must be explicit ints")
        expected = hashlib.sha256((_DRIFT_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("drift plan digest mismatch")


def make_drift_plan(
    *,
    seed: int,
    rate_changes: tuple[RateChange, ...] = (),
    node_churn: ChurnSchedule | tuple[NodeChurnSpec, ...] = (),
    onboarding: tuple[OnboardingActivation, ...] = (),
    mix_shift_statement: str = "none",
) -> DriftPlan:
    if isinstance(node_churn, ChurnSchedule):
        node_churn.verify()
        churn_events = node_churn.events
        component_seeds = (("churn", node_churn.seed),)
    else:
        churn_events = tuple(node_churn)
        if churn_events:
            raise ValueError("non-empty churn must use a sealed ChurnSchedule with its seed")
        component_seeds = ()
    component_seeds += tuple(
        (
            f"onboarding-prior:{activation.template_id}",
            activation.prior_seed,
        )
        for activation in onboarding
    )
    payload = {
        "schema": "scenario-drift-plan-v1",
        "seed": seed,
        "rate_changes": tuple(sorted(rate_changes)),
        "node_churn": tuple(sorted(churn_events, key=lambda event: (event.at_s, event.node_id))),
        "onboarding": tuple(sorted(onboarding)),
        "component_seeds": tuple(sorted(component_seeds)),
        "mix_shift_statement": mix_shift_statement,
    }
    digest = hashlib.sha256((_DRIFT_TAG + _canonical(payload)).encode()).hexdigest()
    plan = DriftPlan(
        **payload, digest=digest, _authority=_DRIFT_AUTHORITY  # type: ignore[arg-type]
    )
    plan.verify()
    return plan


def diurnal_changes(
    *,
    template_id: int,
    start_s: float,
    end_s: float,
    base_rate: float,
    amplitude: float,
    steps: int,
    phase_radians: float = 0.0,
) -> tuple[RateChange, ...]:
    if not 0 <= amplitude < 1 or steps < 2 or not start_s < end_s:
        raise ValueError("diurnal requires amplitude in [0,1), steps >= 2, start < end")
    if not math.isfinite(base_rate) or base_rate <= 0:
        raise ValueError("diurnal base rate must be finite and > 0")
    return tuple(
        RateChange(
            at_s=start_s + i * (end_s - start_s) / steps,
            template_id=template_id,
            rate_per_s=base_rate
            * (1.0 + amplitude * math.sin(phase_radians + 2.0 * math.pi * i / steps)),
            cause="diurnal",
        )
        for i in range(steps + 1)
    )


def _visits(call: CallSpec, visits: float, out: dict[int, float]) -> None:
    """Accumulate a conservative per-request upper bound on service attempts."""
    out[call.service] = out.get(call.service, 0.0) + visits
    for edge in call.children:
        _visits(edge.child, visits * edge.probability * (edge.max_retries + 1), out)


def _blocking_occupancy_s(
    call: CallSpec,
    config: EpisodeConfig,
    backend_latency_s: dict[str, float],
    max_oneway_s: float,
) -> float:
    """Conservative mean residence bound for one server slot at ``call``."""
    service = config.workload.services[call.service]
    own = service.service_time.mean()
    if service.backend is not None:
        own += backend_latency_s[service.backend]
    if call.mode != "blocking":
        return own
    child_wait_bound = 0.0
    for edge in call.children:
        one_attempt = (
            max_oneway_s
            + _blocking_occupancy_s(edge.child, config, backend_latency_s, max_oneway_s)
            + max_oneway_s
        )
        if edge.timeout_s is not None:
            one_attempt = min(one_attempt, edge.timeout_s)
        # Children fan out concurrently, so summing is conservative relative to max.
        child_wait_bound += (edge.max_retries + 1) * one_attempt
    return own + child_wait_bound


def _station_work(
    call: CallSpec,
    rate: float,
    config: EpisodeConfig,
    backend_latency_s: dict[str, float],
    max_oneway_s: float,
    occupancy_work: dict[int, float],
    cpu_work: dict[int, float],
) -> None:
    service = config.workload.services[call.service]
    occupancy_work[call.service] = occupancy_work.get(call.service, 0.0) + rate * (
        _blocking_occupancy_s(call, config, backend_latency_s, max_oneway_s)
    )
    cpu_work[call.service] = cpu_work.get(call.service, 0.0) + (
        rate * service.service_time.mean() * service.cpu_demand / service.cpu_request
    )
    for edge in call.children:
        _station_work(
            edge.child,
            rate * edge.probability * (edge.max_retries + 1),
            config,
            backend_latency_s,
            max_oneway_s,
            occupancy_work,
            cpu_work,
        )


def assert_hpa_feasible_peak(
    config: EpisodeConfig, template_id: int, peak_rate: float, at_s: float
) -> None:
    """Reject flash peaks that HPA cannot serve or place at its target utilization.

    This is intentionally conservative: every bounded retry is charged, traffic from
    every concurrently active template is included, and the required replica multiset
    must pass a deterministic first-fit-decreasing packing on nodes available at the
    flash time. A conservative false rejection is preferable to labeling an intrinsically
    overloaded flash crowd HPA-feasible.
    """
    if not config.hpa.enabled:
        raise ValueError("HPA-feasible flash crowds require HPA to be enabled")
    if not 0 <= template_id < len(config.workload.templates):
        raise ValueError("flash crowd references an unknown template")
    backend_latency_s: dict[str, float] = {}
    for backend in config.workload.backends:
        maximum_concurrency = sum(
            service.max_replicas * service.concurrency
            for service in config.workload.services
            if service.backend == backend.name
        )
        backend_latency_s[backend.name] = backend.latency_s(max(1, maximum_concurrency))
    max_oneway_s = max(max(row) for row in config.cluster.latency_ms) / 1000.0
    occupancy_work: dict[int, float] = {}
    cpu_work: dict[int, float] = {}
    for template in config.workload.templates:
        rate = peak_rate if template.id == template_id else template.arrival.rate_at(at_s)
        if rate <= 0:
            continue
        _station_work(
            template.root,
            rate,
            config,
            backend_latency_s,
            max_oneway_s,
            occupancy_work,
            cpu_work,
        )

    required: list[tuple[int, int, int]] = []
    for service in config.workload.services:
        if service.activation_s > at_s:
            continue
        replicas = max(
            service.min_replicas,
            math.ceil(
                occupancy_work.get(service.id, 0.0)
                / (service.concurrency * config.hpa.target_utilization)
            ),
        )
        if replicas > service.max_replicas:
            raise ValueError(
                f"flash peak is not HPA-feasible at service {service.id}: "
                f"needs {replicas} replicas, max is {service.max_replicas}"
            )
        if replicas > service.initial_replicas:
            raise ValueError(
                f"flash peak is not operationally HPA-feasible at service {service.id}: "
                f"blocking-slot occupancy needs {replicas} replicas but only "
                f"{service.initial_replicas} are pre-scaled; stochastic cold starts have "
                "no finite worst-case readiness bound"
            )
        cpu_driven_replicas = max(
            service.min_replicas,
            math.ceil(cpu_work.get(service.id, 0.0) / config.hpa.target_utilization),
        )
        if replicas > max(service.initial_replicas, min(cpu_driven_replicas, service.max_replicas)):
            raise ValueError(
                f"flash peak is not HPA-feasible at service {service.id}: blocking-slot "
                f"occupancy needs {replicas} replicas but CPU-driven HPA can reach only "
                f"{min(cpu_driven_replicas, service.max_replicas)} from "
                f"initial {service.initial_replicas}"
            )
        # Runtime enqueues every initially active replica.  Certifying only the
        # demand-minimum subset could let surplus initial pods consume capacity and
        # strand another service even though this check reported the flash feasible.
        required.extend(
            (service.cpu_request, service.mem_request, service.id)
            for _ in range(service.initial_replicas)
        )

    schedulable = [True] * len(config.cluster.nodes)
    for event in config.node_churn:
        if event.at_s > at_s:
            break
        schedulable[event.node_id] = event.schedulable
    remaining = [
        [node.cpu_capacity, node.mem_capacity]
        for node, available in zip(config.cluster.nodes, schedulable, strict=True)
        if available and not node.initially_suspended
    ]
    if not remaining:
        raise ValueError("flash peak is not HPA-feasible: no available awake nodes")
    required.sort(
        key=lambda pod: (
            -max(
                pod[0] / max(node[0] for node in remaining),
                pod[1] / max(node[1] for node in remaining),
            ),
            -pod[0],
            -pod[1],
            pod[2],
        )
    )
    for cpu, mem, service_id in required:
        candidates = [
            index
            for index, (cpu_free, mem_free) in enumerate(remaining)
            if cpu <= cpu_free and mem <= mem_free
        ]
        if not candidates:
            raise ValueError(
                f"flash peak is not HPA-feasible: replica of service {service_id} "
                "cannot be packed on the available nodes"
            )
        destination = min(
            candidates,
            key=lambda index: (
                remaining[index][0] - cpu,
                remaining[index][1] - mem,
                index,
            ),
        )
        remaining[destination][0] -= cpu
        remaining[destination][1] -= mem


def flash_crowd_changes(
    config: EpisodeConfig,
    *,
    template_id: int,
    rise_s: float,
    fall_s: float,
    peak_rate: float,
    recovery_rate: float,
) -> tuple[RateChange, RateChange]:
    if not 0 < rise_s < fall_s < config.cutoff_s:
        raise ValueError("flash window must be inside the arrival horizon")
    assert_hpa_feasible_peak(config, template_id, peak_rate, rise_s)
    return (
        RateChange(rise_s, template_id, peak_rate, "pre-scaled-hpa-feasible-flash-crowd"),
        RateChange(fall_s, template_id, recovery_rate, "flash-crowd-recovery"),
    )


def churn_schedule(
    *,
    node_count: int,
    down_count: int,
    down_s: float,
    recovery_s: float,
    seed: int,
) -> ChurnSchedule:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    if not 0 < down_count < node_count or not 0 < down_s < recovery_s:
        raise ValueError("churn needs 0 < down_count < node_count and down < recovery")
    rng = spawn_rng(seed, "scenario", "drift", "churn")
    nodes = sorted(int(node) for node in rng.choice(node_count, size=down_count, replace=False))
    events = tuple(
        sorted(
            (
                *(NodeChurnSpec(down_s, node, False, "planned-capacity-churn") for node in nodes),
                *(
                    NodeChurnSpec(recovery_s, node, True, "planned-capacity-recovery")
                    for node in nodes
                ),
            ),
            key=lambda event: (event.at_s, event.node_id),
        )
    )
    payload = {
        "schema": "scenario-churn-schedule-v1",
        "seed": seed,
        "node_count": node_count,
        "down_count": down_count,
        "down_s": down_s,
        "recovery_s": recovery_s,
        "events": events,
    }
    digest = hashlib.sha256((_CHURN_TAG + _canonical(payload)).encode()).hexdigest()
    schedule = ChurnSchedule(
        **payload, digest=digest, _authority=_CHURN_AUTHORITY  # type: ignore[arg-type]
    )
    schedule.verify()
    return schedule


def mix_shift_changes(
    *,
    at_s: float,
    total_rate: float,
    template_ids: tuple[int, ...],
    target_fractions: tuple[float, ...],
) -> tuple[RateChange, ...]:
    if len(template_ids) != len(target_fractions) or len(set(template_ids)) != len(template_ids):
        raise ValueError("mix shift needs unique template ids and matching fractions")
    if any(fraction < 0 for fraction in target_fractions) or not math.isclose(
        math.fsum(target_fractions), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("mix-shift target fractions must be nonnegative and sum to 1")
    return tuple(
        RateChange(at_s, template_id, total_rate * fraction, "gradual-mix-shift")
        for template_id, fraction in zip(template_ids, target_fractions, strict=True)
    )


def gradual_mix_shift_changes(
    *,
    start_s: float,
    end_s: float,
    total_rate: float,
    template_ids: tuple[int, ...],
    start_fractions: tuple[float, ...],
    target_fractions: tuple[float, ...],
    steps: int,
) -> tuple[RateChange, ...]:
    """Linearly change template mixture while preserving total offered rate."""
    if not 0 < start_s < end_s or steps < 2:
        raise ValueError("gradual mix shift requires 0 < start < end and steps >= 2")
    if len(start_fractions) != len(target_fractions):
        raise ValueError("mix-shift start/target fractions must align")
    for fractions in (start_fractions, target_fractions):
        if any(value < 0 for value in fractions) or not math.isclose(
            math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("mix-shift fractions must be nonnegative and sum to 1")
    changes: list[RateChange] = []
    # Emit the declared start mixture as well as every interpolation step. Without
    # step zero, callers could silently begin from unrelated pre-existing rates.
    for step in range(0, steps + 1):
        fraction = step / steps
        at_s = start_s + fraction * (end_s - start_s)
        mixture = tuple(
            before + fraction * (after - before)
            for before, after in zip(start_fractions, target_fractions, strict=True)
        )
        changes.extend(
            mix_shift_changes(
                at_s=at_s,
                total_rate=total_rate,
                template_ids=template_ids,
                target_fractions=mixture,
            )
        )
    return tuple(changes)


def category_prior_call_graph(
    service_ids: tuple[int, ...], category: str, seed: int
) -> tuple[CallSpec, str]:
    """Generate a seeded tree under one frozen, explicitly synthetic category prior."""
    if (
        not category
        or not service_ids
        or tuple(sorted(set(service_ids))) != service_ids
    ):
        raise ValueError("category prior needs a category and sorted unique service ids")
    if category not in _CATEGORY_PRIOR_DEFINITIONS:
        raise ValueError(
            f"unknown category prior {category!r}; "
            f"expected one of {tuple(_CATEGORY_PRIOR_DEFINITIONS)}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    rng = spawn_rng(seed, "scenario", "onboarding", category)
    children: dict[int, list[int]] = {service: [] for service in service_ids}
    for index, child in enumerate(service_ids[1:], start=1):
        candidates = service_ids[:index]
        if category == "retail":
            weights = [4.0 if rank == 0 else 1.0 for rank in range(index)]
        elif category == "analytics":
            weights = [1.0 / (1.0 + len(children[parent])) for parent in candidates]
        else:  # media: favor recent stages and therefore deeper pipelines
            weights = [float(rank + 1) for rank in range(index)]
        total = math.fsum(weights)
        probabilities = [weight / total for weight in weights]
        parent = candidates[int(rng.choice(index, p=probabilities))]
        children[parent].append(child)

    def build(service: int) -> CallSpec:
        return CallSpec(
            service,
            tuple(
                CallEdgeSpec(child=build(child), probability=1.0)
                for child in sorted(children[service])
            ),
        )

    graph = build(service_ids[0])
    digest = _category_graph_digest(graph, category, seed)
    return graph, digest


def _category_graph_digest(graph: CallSpec, category: str, seed: int) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "category": category,
                "prior": _CATEGORY_PRIOR_DEFINITIONS.get(category),
                "prior_registry_sha256": CATEGORY_PRIOR_REGISTRY_SHA256,
                "seed": seed,
                "tree": dataclasses.asdict(graph),
            }
        ).encode()
    ).hexdigest()


def _call_services(call: CallSpec) -> set[int]:
    services = {call.service}
    for edge in call.children:
        services.update(_call_services(edge.child))
    return services


def apply_drift(config: EpisodeConfig, plan: DriftPlan) -> EpisodeConfig:
    plan.verify()
    by_template: dict[int, list[RateChange]] = {}
    for change in plan.rate_changes:
        if not 0 <= change.template_id < len(config.workload.templates):
            raise ValueError(f"drift references unknown template {change.template_id}")
        by_template.setdefault(change.template_id, []).append(change)
    activation_by_service = {
        service_id: item for item in plan.onboarding for service_id in item.service_ids
    }
    if any(not 0 <= sid < len(config.workload.services) for sid in activation_by_service):
        raise ValueError("onboarding references an unknown service")
    for activation in plan.onboarding:
        if not 0 <= activation.template_id < len(config.workload.templates):
            raise ValueError("onboarding references an unknown template")
        template = config.workload.templates[activation.template_id]
        if template.arrival.start_s < activation.activation_s:
            raise ValueError(
                "onboarding template arrivals must not start before application activation"
            )
        template_services = tuple(sorted(_call_services(template.root)))
        if activation.service_ids != template_services:
            raise ValueError(
                f"onboarding application {activation.template_id} must activate exactly "
                f"its call-graph services {template_services}"
            )
        expected_graph, actual = category_prior_call_graph(
            template_services,
            activation.category,
            activation.prior_seed,
        )
        if expected_graph != template.root or actual != activation.prior_graph_sha256:
            raise ValueError("onboarding prior graph hash does not match its template")
    services: list[ServiceSpec] = []
    for service in config.workload.services:
        activation = activation_by_service.get(service.id)
        services.append(
            service
            if activation is None
            else replace(
                service,
                activation_s=activation.activation_s,
                category=activation.category,
            )
        )
    templates = []
    for template in config.workload.templates:
        changes = by_template.get(template.id, [])
        schedule_by_time = dict(template.arrival.rate_schedule)
        for change in changes:
            if change.at_s in schedule_by_time:
                raise ValueError(
                    f"drift change collides with existing template {template.id} schedule"
                )
            schedule_by_time[change.at_s] = change.rate_per_s
        schedule = tuple(sorted(schedule_by_time.items()))
        templates.append(
            replace(template, arrival=replace(template.arrival, rate_schedule=schedule))
        )
    return replace(
        config,
        workload=WorkloadSpec(
            services=tuple(services),
            templates=tuple(templates),
            backends=config.workload.backends,
        ),
        node_churn=plan.node_churn,
    )


def assert_drift_applied(config: EpisodeConfig, plan: DriftPlan) -> None:
    """Verify that a sealed config contains every effect named by ``plan``."""
    plan.verify()
    if config.node_churn != plan.node_churn:
        raise ValueError("scenario config node churn does not match its drift plan")
    for change in plan.rate_changes:
        template = config.workload.templates[change.template_id]
        if (change.at_s, change.rate_per_s) not in template.arrival.rate_schedule:
            raise ValueError(
                f"scenario config is missing drift change for template {change.template_id} "
                f"at {change.at_s}"
            )
    for activation in plan.onboarding:
        template = config.workload.templates[activation.template_id]
        if template.arrival.start_s < activation.activation_s:
            raise ValueError(
                "scenario onboarding arrivals start before application activation"
            )
        template_services = tuple(sorted(_call_services(template.root)))
        if activation.service_ids != template_services:
            raise ValueError("scenario onboarding application does not cover its whole template")
        for service_id in activation.service_ids:
            service = config.workload.services[service_id]
            if (
                service.activation_s != activation.activation_s
                or service.category != activation.category
            ):
                raise ValueError(
                    f"scenario config onboarding state does not match service {service_id}"
                )
        expected_graph, graph_sha256 = category_prior_call_graph(
            template_services,
            activation.category,
            activation.prior_seed,
        )
        if expected_graph != template.root or graph_sha256 != activation.prior_graph_sha256:
            raise ValueError("scenario onboarding graph does not match its drift plan")
