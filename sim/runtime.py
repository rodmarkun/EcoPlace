"""Runtime cluster state: nodes, pods, power states, and the conservation ledger.

This module owns the *truth* (API-server-equivalent) state. The scheduler emulator keeps
its own, possibly stale, view in ``sim.scheduler.SchedulerCache``; controllers
(``sim.actuation``) mutate truth and notifications propagate to the cache with the
configured informer delay.

Pod phase machine (every pod ends in exactly one of the terminal/live states counted by
the ledger)::

    PENDING -> BINDING -> PULLING -> STARTING -> READY -> TERMINATING -> TERMINATED
        ^         |  (binding failure: back to PENDING, requeued with backoff)
        +---------+

Node power states: AWAKE <-> (SUSPENDING -> SUSPENDED -> WAKING) — transitions only for
suspendable classes, with published transition energy/latency; suspension is only legal
on empty nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import simpy

from sim.energy import MIB, EnergyMeter
from sim.spec import TIER_CLOUD, ClusterSpec, NodeClassSpec, NodeSpec, ServiceSpec

if TYPE_CHECKING:
    from collections.abc import Callable

# Pod phases
PENDING = "pending"
BINDING = "binding"
PULLING = "pulling"
STARTING = "starting"
READY = "ready"
TERMINATING = "terminating"
TERMINATED = "terminated"

# Node power states
AWAKE = "awake"
SUSPENDING = "suspending"
SUSPENDED = "suspended"
WAKING = "waking"


class HostError(RuntimeError):
    """A host, adapter, data, or simulator failure that is never a policy's fault."""


class SimulationError(HostError):
    """An internal invariant of the simulator was violated (never a policy's fault)."""


class PolicyError(RuntimeError):
    """A candidate policy crashed or violated its output contract."""


@dataclass
class Ledger:
    """Conservation accounting (S6) built on **independently tracked lifecycle state**,
    not derived complements: every request, subcall, pod, and scheduler reservation has
    a live-ID entry from creation to its terminal transition, and every transition is
    validated at the moment it happens (unknown/duplicate transitions raise
    ``SimulationError`` immediately — lost work can never masquerade as live work).

    The integer counters are summaries; ``verify()`` cross-checks them against the live
    sets, so a counter that drifts from the underlying state is itself a violation.
    """

    requests_injected: int = 0
    requests_completed: int = 0
    requests_timeout: int = 0
    requests_dropped_no_replica: int = 0
    requests_dropped_queue_full: int = 0
    requests_in_flight_at_end: int = 0

    subcalls_started: int = 0
    subcalls_finished: int = 0
    subcalls_abandoned: int = 0  # waiter timed out; the subtree keeps running (orphan)
    orphans_finished: int = 0
    subcalls_in_flight_at_end: int = 0

    pods_created: int = 0
    pods_terminated: int = 0
    pods_live_at_end: int = 0

    scheduling_cycles: int = 0
    unschedulable_cycles: int = 0
    scorer_errors: int = 0
    reserves: int = 0
    binds_succeeded: int = 0
    bind_failures_injected: int = 0
    bind_failures_validation: int = 0
    unreserves: int = 0

    hpa_scale_ups: int = 0
    hpa_scale_downs: int = 0
    rebalancer_moves_executed: int = 0
    rebalancer_wakes_executed: int = 0
    rebalancer_actions_rejected: int = 0
    rebalancer_contract_violations: int = 0

    # Independent live-ID state (not serialized; the state machine behind the counters).
    _live_requests: set = field(default_factory=set, repr=False)
    _live_subcalls: set = field(default_factory=set, repr=False)
    _live_pods: set = field(default_factory=set, repr=False)
    _reservations: dict = field(default_factory=dict, repr=False)  # pod uid -> node id
    # Complete retained histories. Live sets are fast indexes; histories are the
    # independent source used to prove exact counters and terminal identities.
    _request_status: dict = field(default_factory=dict, repr=False)  # rid -> status
    _subcall_status: dict = field(default_factory=dict, repr=False)  # cid -> status
    _pod_status: dict = field(default_factory=dict, repr=False)  # uid -> live|terminated
    _abandoned_subcalls: set = field(default_factory=set, repr=False)
    _finished_orphans: set = field(default_factory=set, repr=False)
    _reservation_history: list = field(default_factory=list, repr=False)
    _active_reservation_record: dict = field(default_factory=dict, repr=False)
    _next_request_id: int = field(default=0, repr=False)
    _next_subcall_id: int = field(default=0, repr=False)

    _REQUEST_STATUSES = (
        "completed",
        "timeout",
        "dropped_no_replica",
        "dropped_queue_full",
    )

    # -- request lifecycle -----------------------------------------------------
    def request_injected(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        self._live_requests.add(rid)
        self._request_status[rid] = "in_flight"
        self.requests_injected += 1
        return rid

    def request_terminal(self, rid: int, status: str) -> None:
        if rid not in self._live_requests:
            raise SimulationError(f"request {rid} terminated twice or never injected")
        if status not in self._REQUEST_STATUSES:
            raise SimulationError(f"unknown request terminal status {status!r}")
        self._live_requests.remove(rid)
        self._request_status[rid] = status
        attr = f"requests_{status}"
        setattr(self, attr, getattr(self, attr) + 1)

    # -- subcall lifecycle -------------------------------------------------------
    def subcall_started(self) -> int:
        cid = self._next_subcall_id
        self._next_subcall_id += 1
        self._live_subcalls.add(cid)
        self._subcall_status[cid] = "in_flight"
        self.subcalls_started += 1
        return cid

    def subcall_finished(self, cid: int) -> None:
        if cid not in self._live_subcalls:
            raise SimulationError(f"subcall {cid} finished twice or never started")
        self._live_subcalls.remove(cid)
        self._subcall_status[cid] = "finished"
        self.subcalls_finished += 1

    def subcall_abandoned(self, cid: int) -> None:
        if cid not in self._live_subcalls:
            raise SimulationError(f"subcall {cid} abandoned after finishing or never started")
        if cid in self._abandoned_subcalls:
            raise SimulationError(f"subcall {cid} abandoned twice")
        self._abandoned_subcalls.add(cid)
        self.subcalls_abandoned += 1

    def orphan_finished(self, cid: int) -> None:
        if cid not in self._abandoned_subcalls:
            raise SimulationError(f"orphan {cid} finished without abandonment")
        if cid in self._finished_orphans:
            raise SimulationError(f"orphan {cid} finished twice")
        if self._subcall_status.get(cid) != "finished":
            raise SimulationError(f"orphan {cid} recorded before its subcall finished")
        self._finished_orphans.add(cid)
        self.orphans_finished += 1

    # -- pod lifecycle -------------------------------------------------------------
    def pod_created(self, uid: int) -> None:
        if uid in self._pod_status:
            raise SimulationError(f"pod uid {uid} created twice")
        self._live_pods.add(uid)
        self._pod_status[uid] = "live"
        self.pods_created += 1

    def pod_terminated(self, uid: int) -> None:
        if uid not in self._live_pods:
            raise SimulationError(f"pod {uid} terminated twice or never created")
        self._live_pods.remove(uid)
        self._pod_status[uid] = "terminated"
        self.pods_terminated += 1

    # -- scheduler reservation lifecycle (assume -> confirm | unreserve) -----------
    def reserved(self, uid: int, node_id: int) -> None:
        if uid in self._reservations:
            raise SimulationError(f"pod {uid} reserved twice without release")
        self._reservations[uid] = node_id
        rec_id = len(self._reservation_history)
        self._reservation_history.append(
            {"uid": uid, "node": node_id, "status": "reserved", "failure": None}
        )
        self._active_reservation_record[uid] = rec_id
        self.reserves += 1

    def bind_confirmed(self, uid: int) -> None:
        if uid not in self._reservations:
            raise SimulationError(f"bind confirm for pod {uid} without a reservation")
        rec_id = self._active_reservation_record.pop(uid)
        self._reservation_history[rec_id]["status"] = "bound"
        del self._reservations[uid]
        self.binds_succeeded += 1

    def unreserved(self, uid: int, *, injected_failure: bool) -> None:
        if uid not in self._reservations:
            raise SimulationError(f"unreserve of pod {uid} without a reservation")
        rec_id = self._active_reservation_record.pop(uid)
        rec = self._reservation_history[rec_id]
        rec["status"] = "unreserved"
        rec["failure"] = "injected" if injected_failure else "validation"
        del self._reservations[uid]
        self.unreserves += 1
        if injected_failure:
            self.bind_failures_injected += 1
        else:
            self.bind_failures_validation += 1

    def binds_in_flight(self) -> int:
        return len(self._reservations)

    def reservations(self) -> dict[int, int]:
        return dict(self._reservations)

    # -- closing + verification -------------------------------------------------
    def close(self) -> None:
        """Record end-of-episode in-flight counts from the live sets (independent
        state, never arithmetic complements)."""
        self.requests_in_flight_at_end = len(self._live_requests)
        self.subcalls_in_flight_at_end = len(self._live_subcalls)
        self.pods_live_at_end = len(self._live_pods)

    def verify(self, requests: Any = None, pods: Any = None) -> None:
        """Raise ``SimulationError`` on any conservation violation. Counters are checked
        against complete retained per-ID histories and the independent live indexes.
        Optional request/pod truth records make the end-of-episode reconciliation exact."""
        public_counters = [
            f.name for f in self.__dataclass_fields__.values() if not f.name.startswith("_")
        ]
        for name in public_counters:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SimulationError(f"counter {name} must be a nonnegative int, got {value!r}")
        if self._next_request_id != len(self._request_status) or set(self._request_status) != set(
            range(self._next_request_id)
        ):
            raise SimulationError("request identity history is incomplete or non-dense")
        expected_live_requests = {
            rid for rid, status in self._request_status.items() if status == "in_flight"
        }
        if self._live_requests != expected_live_requests:
            raise SimulationError("request live index drifted from retained history")
        request_counts = {
            status: sum(s == status for s in self._request_status.values())
            for status in ("completed", "timeout", "dropped_no_replica", "dropped_queue_full")
        }
        if self.requests_injected != len(self._request_status) or any(
            getattr(self, f"requests_{status}") != count for status, count in request_counts.items()
        ):
            raise SimulationError(
                "request conservation violated: counters drifted from retained identity history"
            )
        terminal = (
            self.requests_completed
            + self.requests_timeout
            + self.requests_dropped_no_replica
            + self.requests_dropped_queue_full
        )
        if terminal + len(self._live_requests) != self.requests_injected:
            raise SimulationError(
                f"request conservation violated: injected={self.requests_injected} "
                f"terminal={terminal} live={len(self._live_requests)}"
            )
        if self.requests_in_flight_at_end != len(self._live_requests):
            raise SimulationError("requests_in_flight_at_end drifted from live set")
        if self._next_subcall_id != len(self._subcall_status) or set(self._subcall_status) != set(
            range(self._next_subcall_id)
        ):
            raise SimulationError("subcall identity history is incomplete or non-dense")
        expected_live_subcalls = {
            cid for cid, status in self._subcall_status.items() if status == "in_flight"
        }
        if self._live_subcalls != expected_live_subcalls:
            raise SimulationError("subcall live index drifted from retained history")
        if self.subcalls_started != len(self._subcall_status) or self.subcalls_finished != sum(
            status == "finished" for status in self._subcall_status.values()
        ):
            raise SimulationError("subcall counters drifted from retained identity history")
        if self.subcalls_finished + len(self._live_subcalls) != self.subcalls_started:
            raise SimulationError(
                f"subcall conservation violated: started={self.subcalls_started} "
                f"finished={self.subcalls_finished} live={len(self._live_subcalls)}"
            )
        if self.subcalls_in_flight_at_end != len(self._live_subcalls):
            raise SimulationError("subcalls_in_flight_at_end drifted from live set")
        if self.subcalls_abandoned != len(self._abandoned_subcalls):
            raise SimulationError("abandonment counter drifted from subcall identities")
        if self.orphans_finished != len(self._finished_orphans) or not (
            self._finished_orphans <= self._abandoned_subcalls
        ):
            raise SimulationError("orphan counter drifted from abandoned subcall identities")
        expected_live_pods = {uid for uid, status in self._pod_status.items() if status == "live"}
        if self._live_pods != expected_live_pods:
            raise SimulationError("pod live index drifted from retained history")
        if self.pods_created != len(self._pod_status) or self.pods_terminated != sum(
            status == "terminated" for status in self._pod_status.values()
        ):
            raise SimulationError("pod counters drifted from retained identity history")
        if self.pods_terminated + len(self._live_pods) != self.pods_created:
            raise SimulationError(
                f"pod conservation violated: created={self.pods_created} "
                f"terminated={self.pods_terminated} live={len(self._live_pods)}"
            )
        if self.pods_live_at_end != len(self._live_pods):
            raise SimulationError("pods_live_at_end drifted from live set")
        if self.unreserves != self.bind_failures_injected + self.bind_failures_validation:
            raise SimulationError("every binding failure must unreserve exactly once")
        statuses = [r["status"] for r in self._reservation_history]
        failures = [r["failure"] for r in self._reservation_history]
        if self.reserves != len(statuses):
            raise SimulationError("reserve counter drifted from retained history")
        if self.binds_succeeded != statuses.count("bound"):
            raise SimulationError(
                "reserve/bind/unreserve accounting: bind counter drifted from history"
            )
        if self.unreserves != statuses.count("unreserved"):
            raise SimulationError("unreserve counter drifted from retained history")
        if self.bind_failures_injected != failures.count("injected") or (
            self.bind_failures_validation != failures.count("validation")
        ):
            raise SimulationError("binding-failure counters drifted from retained history")
        expected_active = {
            r["uid"]: r["node"] for r in self._reservation_history if r["status"] == "reserved"
        }
        if self._reservations != expected_active:
            raise SimulationError("active reservations drifted from retained history")
        if set(self._active_reservation_record) != set(expected_active):
            raise SimulationError("active reservation index drifted from retained history")
        if requests is not None:
            records = {r.rid: r.status for r in requests}
            if records != self._request_status:
                raise SimulationError("request records do not match ledger identities/statuses")
        if pods is not None:
            truth = {p.uid: ("live" if p.live else "terminated") for p in pods}
            if truth != self._pod_status:
                raise SimulationError("pod truth does not match ledger identities/statuses")

    def as_dict(self) -> dict[str, int]:
        return {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
            if not f.name.startswith("_")
        }


@dataclass
class PodRuntime:
    """One pod (service replica instance)."""

    uid: int  # dense creation sequence number, never reused
    service: ServiceSpec
    origin: str  # "initial" | "hpa" | "rebalancer"
    created_s: float
    service_ordinal: int = 0  # per-service creation index (stable RNG-keying identity)
    rng_identity: tuple[str | int, ...] = ()  # semantic, behavior-independent CRN key
    phase: str = PENDING
    node_id: int | None = None
    bound_s: float | None = None
    ready_s: float | None = None
    terminated_s: float | None = None
    termination_reason: str | None = None
    scheduling_attempts: int = 0
    binding_failures: int = 0
    outstanding: int = 0  # requests holding/waiting for this replica's server
    drained: Any = None  # simpy event, set when outstanding hits 0 while TERMINATING
    startup_image_mb: float = 0.0  # actual pull initiated for this pod (migration ledger)
    ready_event: Any = None  # exact lifecycle signals (created by ClusterRuntime)
    terminated_event: Any = None

    @property
    def live(self) -> bool:
        return self.phase != TERMINATED

    @property
    def has_pdb(self) -> bool:
        return self.service.pdb_protected


class NodeRuntime:
    """One node: capacity accounting, CPU contention container, power state, image cache."""

    def __init__(
        self,
        env: simpy.Environment,
        spec: NodeSpec,
        node_class: NodeClassSpec,
        meter: EnergyMeter,
    ) -> None:
        self.env = env
        self.spec = spec
        self.node_class = node_class
        self._meter = meter
        self.power_state = SUSPENDED if spec.initially_suspended else AWAKE
        # Planned Phase-4 capacity churn is scheduling availability, deliberately
        # orthogonal to physical power.  Cordoning does not evict or kill existing
        # replicas and therefore does not change the energy meter state.
        self.schedulable = True
        self.cpu_allocated = 0  # sum of bound pods' cpu_request
        self.mem_allocated = 0
        self.busy_millicores = 0.0  # demand held by in-service requests
        self.cpu = simpy.Container(env, capacity=spec.cpu_capacity, init=spec.cpu_capacity)
        self.image_cache: set[int] = set()
        self.pods: dict[int, PodRuntime] = {}
        self.last_emptied_s = 0.0

    @property
    def id(self) -> int:
        return self.spec.id

    @property
    def awake(self) -> bool:
        return self.power_state == AWAKE

    def hold_cpu(self, demand: int) -> Any:
        """Acquire ``demand`` millicores (FIFO; node-capacity contention). Demand is
        clamped to the node capacity — a single request cannot use more CPU than the
        machine has."""
        demand = min(demand, self.spec.cpu_capacity)
        if demand <= 0:
            return None
        return self.cpu.get(demand)

    def held(self, demand: int, delta: int) -> None:
        """Track busy millicores for the power curve (call around service intervals)."""
        demand = min(demand, self.spec.cpu_capacity)
        self.busy_millicores += delta * demand
        self._meter.set_busy_millicores(self.id, self.busy_millicores, self.env.now)

    def release_cpu(self, demand: int) -> None:
        demand = min(demand, self.spec.cpu_capacity)
        if demand > 0:
            self.cpu.put(demand)


class ClusterRuntime:
    """Truth state + mutation methods. External observers (the scheduler cache) subscribe
    via ``on_truth_change`` and receive change closures to apply after their own delay."""

    def __init__(
        self,
        env: simpy.Environment,
        spec: ClusterSpec,
        meter: EnergyMeter,
        ledger: Ledger,
    ) -> None:
        self.env = env
        self.spec = spec
        self.meter = meter
        self.ledger = ledger
        self.nodes = [NodeRuntime(env, n, spec.node_class_of(n.id), meter) for n in spec.nodes]
        self.pods: dict[int, PodRuntime] = {}  # by uid, insertion == creation order
        self._next_uid = 0
        self._origin_ordinals: dict[tuple[int, str], int] = {}
        self._observers: list[Callable[[dict[str, Any]], None]] = []

    # -- observation --------------------------------------------------------
    def subscribe(self, deliver: Callable[[dict[str, Any]], None]) -> None:
        """``deliver(change)`` is called on every externally-visible truth change with a
        change record (``{"kind": "alloc"|"power", ...}``); the subscriber decides when
        to apply it to its own view (informer staleness)."""
        self._observers.append(deliver)

    def _notify(self, change: dict[str, Any]) -> None:
        for deliver in self._observers:
            deliver(change)

    # -- pods -----------------------------------------------------------------
    def create_pod(
        self,
        service: ServiceSpec,
        origin: str,
        rng_identity: tuple[str | int, ...] | None = None,
    ) -> PodRuntime:
        """Create a pod with an origin-specific semantic RNG identity.

        Callers that own a semantic generation/action identity pass it explicitly.
        The fallback counter is origin-specific, so rebalancer activity can never shift
        HPA/initial cold-start or bind streams (CRN discipline).
        """
        key = (service.id, origin)
        ordinal = self._origin_ordinals.get(key, 0)
        self._origin_ordinals[key] = ordinal + 1
        identity = (origin, ordinal) if rng_identity is None else tuple(rng_identity)
        if not identity or any(
            isinstance(x, bool) or not isinstance(x, str | int) for x in identity
        ):
            raise SimulationError(f"invalid pod RNG identity {identity!r}")
        pod = PodRuntime(
            uid=self._next_uid,
            service=service,
            origin=origin,
            created_s=self.env.now,
            service_ordinal=ordinal,
            rng_identity=identity,
            ready_event=self.env.event(),
            terminated_event=self.env.event(),
        )
        self._next_uid += 1
        self.pods[pod.uid] = pod
        self.ledger.pod_created(pod.uid)
        return pod

    def bind(self, pod: PodRuntime, node_id: int, notify: bool) -> None:
        """Commit a binding in truth. ``notify=False`` for scheduler-initiated binds (the
        scheduler already holds the allocation as an assumed pod); ``notify=True`` for
        rebalancer direct binds (the scheduler learns via its informer).

        Truth-level capacity is unconditional: every evaluable trace must remain inside
        the energy ceiling's capacity assumptions. A stale scheduler view may produce a
        bind failure, but it may never commit overcapacity truth.
        """
        node = self.nodes[node_id]
        if not node.awake or not node.schedulable:
            raise SimulationError(f"bind to unavailable node {node_id}")
        if pod.node_id is not None:
            raise SimulationError(f"pod {pod.uid} is already bound to node {pod.node_id}")
        if (
            node.cpu_allocated + pod.service.cpu_request > node.spec.cpu_capacity
            or node.mem_allocated + pod.service.mem_request > node.spec.mem_capacity
        ):
            raise SimulationError(
                f"truth capacity overcommit on node {node_id} binding pod {pod.uid}"
            )
        pod.node_id = node_id
        pod.bound_s = self.env.now
        node.pods[pod.uid] = pod
        node.cpu_allocated += pod.service.cpu_request
        node.mem_allocated += pod.service.mem_request
        self._sync_cloud_alloc()
        if notify:
            self._notify(
                {
                    "kind": "alloc",
                    "node": node_id,
                    "cpu": pod.service.cpu_request,
                    "mem": pod.service.mem_request,
                }
            )

    def unbind(self, pod: PodRuntime, reason: str) -> None:
        if pod.node_id is None:
            raise SimulationError(f"unbind of unbound pod {pod.uid}")
        node = self.nodes[pod.node_id]
        del node.pods[pod.uid]
        node.cpu_allocated -= pod.service.cpu_request
        node.mem_allocated -= pod.service.mem_request
        if not node.pods:
            node.last_emptied_s = self.env.now
        self._sync_cloud_alloc()
        self._notify(
            {
                "kind": "alloc",
                "node": pod.node_id,
                "cpu": -pod.service.cpu_request,
                "mem": -pod.service.mem_request,
            }
        )
        pod.phase = TERMINATED
        pod.terminated_s = self.env.now
        pod.termination_reason = reason
        pod.node_id = None
        if not pod.terminated_event.triggered:
            pod.terminated_event.succeed()
        self.ledger.pod_terminated(pod.uid)

    def terminate_unbound(self, pod: PodRuntime, reason: str) -> None:
        """Terminate a pod that never bound (e.g. pending at scale-down)."""
        if pod.node_id is not None:
            raise SimulationError(f"terminate_unbound of bound pod {pod.uid}")
        pod.phase = TERMINATED
        pod.terminated_s = self.env.now
        pod.termination_reason = reason
        if not pod.terminated_event.triggered:
            pod.terminated_event.succeed()
        self.ledger.pod_terminated(pod.uid)

    def _sync_cloud_alloc(self) -> None:
        total = sum(n.cpu_allocated for n in self.nodes if n.spec.tier == TIER_CLOUD)
        self.meter.set_cloud_alloc_millicores(float(total), self.env.now)

    def set_schedulable(self, node_id: int, schedulable: bool) -> None:
        """Apply a planned cordon/uncordon without changing power or resident pods."""
        node = self.nodes[node_id]
        if node.schedulable == schedulable:
            raise SimulationError(f"node {node_id} schedulability already {schedulable}")
        node.schedulable = schedulable
        self._notify({"kind": "availability", "node": node_id, "unavailable": not schedulable})

    # -- power state machine --------------------------------------------------
    def suspend(self, node_id: int) -> Any:
        """Suspend an empty, awake, suspendable node (a simpy process)."""
        node = self.nodes[node_id]
        cls = node.node_class
        if not cls.suspendable:
            raise SimulationError(f"suspend of non-suspendable node {node_id}")
        if node.power_state != AWAKE or node.pods:
            raise SimulationError(f"suspend of non-idle node {node_id}")
        return self.env.process(self._suspend_proc(node, cls))

    def _suspend_proc(self, node: NodeRuntime, cls: NodeClassSpec) -> Any:
        node.power_state = SUSPENDING
        self.meter.set_state(node.id, "transition", self.env.now)
        self.meter.add_transition_pulse(node.id, cls.suspend_energy_j or 0.0)
        yield self.env.timeout(cls.suspend_latency_s or 0.0)
        node.power_state = SUSPENDED
        self.meter.set_state(node.id, "suspended", self.env.now)
        self._notify({"kind": "power", "node": node.id, "suspended": True})

    def wake(self, node_id: int) -> Any:
        node = self.nodes[node_id]
        cls = node.node_class
        if not cls.suspendable:
            raise SimulationError(f"wake of non-suspendable node {node_id}")
        if node.power_state != SUSPENDED:
            raise SimulationError(f"wake of non-suspended node {node_id}")
        return self.env.process(self._wake_proc(node, cls))

    def _wake_proc(self, node: NodeRuntime, cls: NodeClassSpec) -> Any:
        node.power_state = WAKING
        self.meter.set_state(node.id, "transition", self.env.now)
        self.meter.add_transition_pulse(node.id, cls.wake_energy_j or 0.0)
        yield self.env.timeout(cls.wake_latency_s or 0.0)
        node.power_state = AWAKE
        node.last_emptied_s = self.env.now
        self.meter.set_state(node.id, "awake", self.env.now)
        self._notify({"kind": "power", "node": node.id, "suspended": False})

    # -- network accounting -----------------------------------------------------
    def transfer_bytes(self, src_node: int, dst_node: int, kb: float) -> None:
        """Account cross-node payload bytes for the network-allocated category."""
        if src_node != dst_node and kb > 0:
            self.meter.add_bytes(kb * 1000.0)

    def account_image_pull(self, node_id: int, image_mb: int) -> None:
        self.meter.add_bytes(image_mb * MIB)

    def oneway_latency_s(self, src_node: int, dst_node: int) -> float:
        return self.spec.latency_ms[src_node][dst_node] / 1000.0

    # -- views ---------------------------------------------------------------
    def live_pods(self) -> list[PodRuntime]:
        return [p for p in self.pods.values() if p.live]


IN_FLIGHT = "in_flight"


@dataclass
class RequestRecord:
    """One **offered** root request (plan §5: constraints are computed over the offered
    cohort). Created at injection with a stable ``rid``; ``status`` stays ``"in_flight"``
    until the request reaches a terminal state. Requests still in flight at the horizon
    remain in the record set — they are never dropped from any denominator."""

    rid: int
    template: int
    arrival_s: float
    end_s: float | None = None
    status: str = IN_FLIGHT  # + "completed"|"timeout"|"dropped_no_replica"|"dropped_queue_full"

    @property
    def terminal(self) -> bool:
        return self.status != IN_FLIGHT

    @property
    def latency_s(self) -> float:
        if self.end_s is None:
            raise ValueError(f"request {self.rid} is still in flight; no latency")
        return self.end_s - self.arrival_s


@dataclass
class ServiceCallRecord:
    """One admitted call stage at a concrete service replica.

    ``start_s`` is the arrival at the selected replica, before its local FIFO wait;
    ``end_s`` is when the stage returns, including local service/backend time and any
    blocking descendants.  Calls still executing at the episode horizon remain
    ``in_flight``.  This is the retained simulator truth used to aggregate the Alibaba
    service-instance-by-60-second interval-mean response-time estimand.
    """

    template: int
    request_index: int
    path: tuple[tuple[int, int], ...]
    service: int
    pod_uid: int
    start_s: float
    end_s: float | None = None
    status: str = IN_FLIGHT

    @property
    def latency_s(self) -> float:
        if self.end_s is None:
            raise ValueError("service call is still in flight; no latency")
        return self.end_s - self.start_s


@dataclass
class PodEvent:
    uid: int
    service: int
    origin: str
    node_id: int | None
    created_s: float
    bound_s: float | None
    ready_s: float | None
    terminated_s: float | None
    phase: str
    termination_reason: str | None
    binding_failures: int = 0
    scheduling_attempts: int = 0


def pod_event(pod: PodRuntime) -> PodEvent:
    return PodEvent(
        uid=pod.uid,
        service=pod.service.id,
        origin=pod.origin,
        node_id=pod.node_id,
        created_s=pod.created_s,
        bound_s=pod.bound_s,
        ready_s=pod.ready_s,
        terminated_s=pod.terminated_s,
        phase=pod.phase,
        termination_reason=pod.termination_reason,
        binding_failures=pod.binding_failures,
        scheduling_attempts=pod.scheduling_attempts,
    )
