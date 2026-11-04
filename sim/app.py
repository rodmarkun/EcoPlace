"""C11 application model (TODO 2.1): open queueing network over service replicas.

Each service replica (pod) is a FIFO multi-server station (``concurrency`` servers).
A request template is a DAG of call stages; a stage queues at one replica (load-balanced),
holds node CPU (capacity contention), draws a service time, optionally touches a shared
backend (concurrency-latency curve), then invokes its children:

- ``blocking`` stages hold their server slot through the child fan-out and wait for all
  children (fan-in) — synchronous RPC semantics; per-edge timeouts + bounded retries.
- ``forward`` stages release the server first, then hand the request off — pipeline
  semantics; a chain of forward stages with exponential service and Poisson arrivals is
  exactly a (feed-forward) Jackson network, which the verification suite exploits.
  Fan-out children run concurrently in both modes; the stage completes at the fan-in
  (``AllOf``), i.e. the makespan of its children.

Timeout semantics (documented): a timed-out child call is *abandoned, not cancelled* —
the child subtree keeps executing (orphaned work; its resource usage stays real) and is
counted in the ledger (``subcalls_abandoned`` / ``orphans_finished``). Retries issue a
fresh child call. Cross-node calls pay one-way network latency each direction (blocking)
or on the handoff only (forward), and account payload bytes for the network-allocated
energy category; same-node calls are free (no loopback modeling).

**CRN discipline (C13, docs/SEEDS.md):** every environmental draw is keyed by the
*stable identity of the thing being drawn for*, never by event-execution order. Each
call-stage execution — identified by ``(template, per-template request index, path)``
where ``path`` is the chain of ``(edge index, attempt)`` pairs from the template root —
derives its own substream via ``spawn_rng`` and consumes, in a fixed order: one LB
uniform, the service-time draw, then one routing uniform per probabilistic child edge
(drawn unconditionally). A placement/LB/binding change therefore cannot shift which
random variate any other request receives: paired comparisons see identical environment
randomness by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import simpy

from sim.arrivals import iter_arrival_times
from sim.runtime import ClusterRuntime, Ledger, PodRuntime, RequestRecord, ServiceCallRecord
from sim.spec import BackendSpec, CallEdgeSpec, CallSpec, ServiceSpec, TemplateSpec, WorkloadSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import numpy as np

OK = "completed"
TIMEOUT = "timeout"
DROP_NO_REPLICA = "dropped_no_replica"
DROP_QUEUE_FULL = "dropped_queue_full"


class BackendRuntime:
    """A shared-state backend: latency is a function of concurrent in-flight accesses."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.inflight = 0
        self.peak_inflight = 0

    def enter(self) -> float:
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        return self.spec.latency_s(float(self.inflight))

    def leave(self) -> None:
        self.inflight -= 1


class ReplicaRuntime:
    """The queueing station attached to one READY pod."""

    def __init__(self, env: simpy.Environment, pod: PodRuntime, service: ServiceSpec) -> None:
        self.env = env
        self.pod = pod
        self.service = service
        self.servers = simpy.Resource(env, capacity=service.concurrency)

    @property
    def node_id(self) -> int:
        assert self.pod.node_id is not None
        return self.pod.node_id

    def queue_full(self) -> bool:
        cap = self.service.queue_capacity
        return cap is not None and len(self.servers.queue) >= cap


class ServiceRuntime:
    """Per-service state: ready replicas, load balancing, HPA busy accounting."""

    def __init__(self, spec: ServiceSpec) -> None:
        self.spec = spec
        self.replicas: dict[int, ReplicaRuntime] = {}  # by pod uid, insertion order
        self._rr = 0
        self.busy_millicore_seconds = 0.0  # cumulative; HPA reads deltas
        self.served = 0

    def add_replica(self, replica: ReplicaRuntime) -> None:
        self.replicas[replica.pod.uid] = replica

    def remove_replica(self, uid: int) -> None:
        self.replicas.pop(uid, None)

    def ready_count(self) -> int:
        return len(self.replicas)

    def select(self, policy: str, u: float) -> ReplicaRuntime | None:
        """Pick a replica: ``round_robin`` (default, deterministic), ``random`` (uses the
        caller's pre-drawn uniform ``u`` from the call-instance substream — CRN keying),
        or ``least_outstanding`` (ties -> lowest pod uid)."""
        pool = list(self.replicas.values())
        if not pool:
            return None
        if policy == "round_robin":
            choice = pool[self._rr % len(pool)]
            self._rr += 1
            return choice
        if policy == "random":
            return pool[min(int(u * len(pool)), len(pool) - 1)]
        if policy == "least_outstanding":
            return min(pool, key=lambda r: (r.pod.outstanding, r.pod.uid))
        raise ValueError(f"unknown LB policy {policy}")


class ApplicationModel:
    """Owns the request lifecycle. The engine wires it to the cluster runtime and starts
    one arrival process per template."""

    def __init__(
        self,
        env: simpy.Environment,
        cluster: ClusterRuntime,
        workload: WorkloadSpec,
        ledger: Ledger,
        rng_for: Callable[..., np.random.Generator],
        lb_policy: str = "round_robin",
    ) -> None:
        self.env = env
        self.cluster = cluster
        self.workload = workload
        self.ledger = ledger
        self.lb_policy = lb_policy
        self.records: list[RequestRecord] = []  # every offered request, injection order
        self.service_calls: list[ServiceCallRecord] = []
        self.backends = {b.name: BackendRuntime(b) for b in workload.backends}
        self.services = [ServiceRuntime(s) for s in workload.services]
        self._rng_for = rng_for
        self._arrival_rngs = [rng_for("arrivals", t.id) for t in workload.templates]
        self._injected: list[int] = [0] * len(workload.templates)

    def _stage_rng(
        self, template_id: int, req_index: int, path: tuple[tuple[int, int], ...]
    ) -> np.random.Generator:
        """The call-instance substream: keyed by the stable identity of one stage
        execution (template, per-template request index, (edge, attempt) chain)."""
        flat = [x for pair in path for x in pair]
        return self._rng_for("call", template_id, req_index, *flat)

    # -- replica lifecycle hooks (called by the engine when pods become READY/leave) --
    def replica_ready(self, pod: PodRuntime) -> None:
        service = self.services[pod.service.id]
        service.add_replica(ReplicaRuntime(self.env, pod, pod.service))

    def replica_leaving(self, pod: PodRuntime) -> None:
        self.services[pod.service.id].remove_replica(pod.uid)

    # -- arrivals -------------------------------------------------------------
    def arrivals(self, template: TemplateSpec, cutoff_s: float) -> Generator[Any, Any, None]:
        """Open arrivals by cumulative-intensity inversion, through ``cutoff_s``.

        Poisson draws are unit-exponential hazard increments; deterministic arrivals
        occur at integer intensity targets.  Neither sequence restarts at a rate
        boundary, so splitting an equal-rate segment cannot perturb the trace.
        """
        rng = self._arrival_rngs[template.id]
        spec = template.arrival
        for at_s in iter_arrival_times(spec, cutoff_s, rng):
            if at_s > self.env.now:
                yield self.env.timeout(at_s - self.env.now)
            req_index = self._injected[template.id]
            self._injected[template.id] += 1
            rid = self.ledger.request_injected()
            record = RequestRecord(rid=rid, template=template.id, arrival_s=self.env.now)
            self.records.append(record)
            self.env.process(self._request(template, req_index, record))

    # -- request execution -----------------------------------------------------
    def _request(
        self, template: TemplateSpec, req_index: int, record: RequestRecord
    ) -> Generator[Any, Any, None]:
        status = yield from self._root_call(template.root, template.id, req_index)
        record.end_s = self.env.now
        record.status = status
        self.ledger.request_terminal(record.rid, status)

    def _root_call(
        self, call: CallSpec, template_id: int, req_index: int
    ) -> Generator[Any, Any, str]:
        """Select a replica (LB) and execute the root stage. Ingress network transit is
        not modeled (documented)."""
        rng = self._stage_rng(template_id, req_index, ())
        u = float(rng.random())  # always drawn: stage-stream consumption is fixed
        replica = self.services[call.service].select(self.lb_policy, u)
        if replica is None:
            return DROP_NO_REPLICA
        return (yield from self._call_on(replica, call, template_id, req_index, (), rng))

    def _children(
        self,
        call: CallSpec,
        template_id: int,
        req_index: int,
        path: tuple[tuple[int, int], ...],
        src_node: int,
        rng: np.random.Generator,
    ) -> Generator[Any, Any, str]:
        """Fan-out to sampled children; fan-in on all of them (order-deterministic).
        Routing uniforms come from the parent stage's substream, one per probabilistic
        edge, drawn unconditionally (consumption never depends on other components)."""
        included: list[tuple[int, CallEdgeSpec]] = []
        for edge_index, edge in enumerate(call.children):
            if edge.probability >= 1.0:
                included.append((edge_index, edge))
            elif float(rng.random()) < edge.probability:
                included.append((edge_index, edge))
        if not included:
            return OK
        blocking = call.mode == "blocking"
        procs = [
            self.env.process(
                self._child_call(edge, edge_index, template_id, req_index, path, src_node, blocking)
            )
            for edge_index, edge in included
        ]
        yield simpy.AllOf(self.env, procs)
        statuses = [p.value for p in procs]
        for status in statuses:
            if status != OK:
                return status
        return OK

    def _child_call(
        self,
        edge: CallEdgeSpec,
        edge_index: int,
        template_id: int,
        req_index: int,
        path: tuple[tuple[int, int], ...],
        src_node: int,
        blocking: bool,
    ) -> Generator[Any, Any, str]:
        """One child call with per-edge timeout + bounded retries (blocking edges)."""
        attempts = 1 + (edge.max_retries if blocking else 0)
        status = DROP_NO_REPLICA
        for attempt in range(attempts):
            child_path = (*path, (edge_index, attempt))
            proc = self.env.process(
                self._attempt(edge, template_id, req_index, child_path, src_node, blocking)
            )
            cid = self.ledger.subcall_started()
            self.env.process(self._watch_subcall(proc, cid))
            if blocking and edge.timeout_s is not None:
                deadline = self.env.timeout(edge.timeout_s)
                yield proc | deadline
                if proc.triggered:
                    status = proc.value
                else:
                    status = TIMEOUT
                    self.ledger.subcall_abandoned(cid)
                    self.env.process(self._watch_orphan(proc, cid))
            else:
                status = yield proc
            if status == OK:
                return OK
        return status

    def _attempt(
        self,
        edge: CallEdgeSpec,
        template_id: int,
        req_index: int,
        path: tuple[tuple[int, int], ...],
        src_node: int,
        blocking: bool,
    ) -> Generator[Any, Any, str]:
        """One network-request-network attempt against the child subtree. The request
        payload travels to whichever replica the LB picked; transit latency/bytes are
        charged around the child stage; the response leg is charged for blocking edges
        only (a forward handoff has no waiting caller to reply to)."""
        rng = self._stage_rng(template_id, req_index, path)
        u = float(rng.random())
        service = self.services[edge.child.service]
        replica = service.select(self.lb_policy, u)
        if replica is None:
            return DROP_NO_REPLICA
        dst_node = replica.node_id
        if dst_node != src_node:
            self.cluster.transfer_bytes(src_node, dst_node, edge.request_kb)
            yield self.env.timeout(self.cluster.oneway_latency_s(src_node, dst_node))
        status = yield from self._call_on(replica, edge.child, template_id, req_index, path, rng)
        if status == OK and blocking and dst_node != src_node:
            self.cluster.transfer_bytes(dst_node, src_node, edge.response_kb)
            yield self.env.timeout(self.cluster.oneway_latency_s(dst_node, src_node))
        return status

    def _call_on(
        self,
        replica: ReplicaRuntime,
        call: CallSpec,
        template_id: int,
        req_index: int,
        path: tuple[tuple[int, int], ...],
        rng: np.random.Generator,
    ) -> Generator[Any, Any, str]:
        """Execute one stage on an already-selected replica. ``rng`` is this stage's
        substream (LB uniform already consumed by the caller)."""
        if replica.pod.uid not in self.services[call.service].replicas:
            # The replica left the pool between selection and arrival (eviction race).
            return DROP_NO_REPLICA
        service = self.services[call.service]
        pod = replica.pod
        spec = replica.service
        node = self.cluster.nodes[replica.node_id]
        service_time = spec.service_time.sample(rng)  # drawn before any yield: ID-keyed
        # Admission is destination-local and atomic with joining the SimPy resource.
        # In particular, child calls check *after* cross-node transit: checking before
        # transit lets an arbitrary burst bypass a bounded queue while in flight.
        # No yield occurs between this check and ``request()``, so simultaneous events
        # cannot over-admit the FIFO bound.
        if replica.queue_full():
            return DROP_QUEUE_FULL
        call_record = ServiceCallRecord(
            template=template_id,
            request_index=req_index,
            path=path,
            service=call.service,
            pod_uid=pod.uid,
            start_s=self.env.now,
        )
        self.service_calls.append(call_record)
        status = OK
        pod.outstanding += 1
        try:
            with replica.servers.request() as slot:
                yield slot
                get = node.hold_cpu(spec.cpu_demand)
                if get is not None:
                    yield get
                node.held(spec.cpu_demand, +1)
                t0 = self.env.now
                try:
                    yield self.env.timeout(service_time)
                finally:
                    node.held(spec.cpu_demand, -1)
                    node.release_cpu(spec.cpu_demand)
                    service.busy_millicore_seconds += spec.cpu_demand * (self.env.now - t0)
                    service.served += 1
                if spec.backend is not None:
                    backend = self.backends[spec.backend]
                    latency = backend.enter()
                    try:
                        yield self.env.timeout(latency)
                    finally:
                        backend.leave()
                if call.mode == "blocking" and call.children:
                    status = yield from self._children(
                        call, template_id, req_index, path, replica.node_id, rng
                    )
                    if status != OK:
                        return status
        finally:
            pod.outstanding -= 1
            if pod.outstanding == 0 and pod.drained is not None:
                pod.drained.succeed()
                pod.drained = None
            call_record.end_s = self.env.now
            call_record.status = status
        if call.mode == "forward" and call.children:
            status = yield from self._children(
                call, template_id, req_index, path, replica.node_id, rng
            )
            # Forward descendants execute after this replica has released its server;
            # they are not part of this provider-stage response time.
            return status
        return status

    def _watch_subcall(self, proc: Any, cid: int) -> Generator[Any, Any, None]:
        yield proc
        self.ledger.subcall_finished(cid)

    def _watch_orphan(self, proc: Any, cid: int) -> Generator[Any, Any, None]:
        yield proc
        self.ledger.orphan_finished(cid)

    # -- terminal accounting ------------------------------------------------------
    def terminal_request_count(self) -> int:
        return sum(1 for r in self.records if r.terminal)
