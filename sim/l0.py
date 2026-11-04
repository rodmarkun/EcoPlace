"""L0 analytical evaluator (TODO 2.7; plan §3.3): ~1 ms/candidate crash/validity/
gross-quality filter.

Given a frozen ``SnapshotLibrary``, ``L0Evaluator.evaluate(scorer)``:

1. **Executes the policy** over each snapshot: pending pods are placed in the captured
   scheduler-queue order through the *same* Filter -> Score -> NormalizeScore ->
   tie-break pipeline as the emulator (``sim.scheduler.filter_feasible`` +
   ``score_and_select``, honoring the config's ``quantize_scores`` flag; allocations
   update as the assignment is built).
2. **Scores the resulting static assignment with closed forms** (no event loop):
   power-curve device energy including idle and suspended nodes, allocated cloud energy
   (baseline floor included), allocated network energy from cross-node call-graph
   traffic (request bytes; response bytes on blocking edges only — matching the DES
   accounting), per-template path latency (Erlang-C M/M/c sojourn per station with the
   backend access folded into the effective service time; fan-in as max-of-expectations
   for **both** blocking and forward stages — the DES runs children concurrently in both
   modes), arrival windows (``start_s``/``arrival_cutoff_s``) applied to energy and
   traffic totals; constraint checks (unschedulable pods, station saturation rho >= 1,
   per-template latency SLOs, services with traffic but no replica).

Documented estimation limits (L0 is a gross-quality filter, not fitness of record —
each of these is an explicitly labeled heuristic, plan §3.3):

- Per-replica arrival streams are treated as independent Poisson thinnings of the
  service's flow; the DES load-balancer is round-robin by default, so per-replica
  streams are actually more regular than Poisson (Erlang-C is conservative here).
- Saturation and sojourn use the nominal (peak) arrival rates of overlapping templates;
  energy/traffic totals use time-averaged rates over each template's arrival window.
- Device power is evaluated at the time-averaged utilization (the power curve is
  piecewise-linear, so this is exact only within a linear segment).
- Fan-in latency is max-of-expectations (a lower bound on the expected makespan);
  server occupancy by blocking downstream waits is ignored in rho.
- Cold start / image pull / retries / timeouts / HPA / rebalancing do not exist at L0 —
  L1/L2 own all dynamic effects.

If L0 cannot rank (G-ladder), it demotes to a validity filter (§3.3).

Raw component outputs are exposed (energy categories, idle-floor-normalized energy,
violations); the Phase-3 fitness aggregation (lexicographic shortfalls, §4.1) consumes
these — it is deliberately not implemented here.

``brute_force_optimum`` is an **independent enumeration oracle**: it applies candidate
assignments directly (no scorer, no shared pipeline) and computes their energy with a
self-contained closed-form implementation, so evaluator bugs cannot hide inside their
own oracle (R1-10). The verification suite checks both against hand-computed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sim.energy import BoundaryWeights, idle_floor_j
from sim.scheduler import PolicyError, Scorer, filter_feasible, score_and_select
from sim.spec import TIER_CLOUD, CallSpec, EpisodeConfig

if TYPE_CHECKING:
    from sim.snapshot import Snapshot, SnapshotLibrary


@dataclass
class L0SnapshotResult:
    feasible: bool
    violations: list[str]
    decisions: list[tuple[int, int]]  # (pending dense pod id, chosen node id)
    device_j: float
    cloud_j: float
    network_j: float
    closed_boundary_j: float
    normalized_energy: float  # closed boundary / idle floor (V1 normalizer)
    template_latency_est_s: dict[int, float]


@dataclass
class L0Evaluation:
    """Aggregate over the library. ``valid=False`` means the candidate crashed or
    produced malformed scores — the crash-filter outcome; quality fields are then
    meaningless and set to None. Host/simulator errors are never absorbed here: they
    raise (a candidate must not be penalized for our bugs — R1-12)."""

    valid: bool
    error: str | None
    feasible: bool  # every snapshot feasible
    mean_normalized_energy: float | None
    violations: list[str] = field(default_factory=list)
    per_snapshot: list[L0SnapshotResult] = field(default_factory=list)


def _iter_edges_with_mode(call: CallSpec, visits: float = 1.0) -> list[tuple[CallSpec, Any, float]]:
    """Flatten a template DAG into (parent_call, edge, expected_visits_of_parent)."""
    out = []
    for edge in call.children:
        out.append((call, edge, visits))
        out.extend(_iter_edges_with_mode(edge.child, visits * edge.probability))
    return out


def _erlang_c_wait_s(lam: float, mean_service_s: float, servers: int) -> float:
    """Mean M/M/c waiting time Wq (exact for Poisson arrivals, exponential service).
    Returns inf at or beyond saturation."""
    if mean_service_s <= 0 or lam <= 0:
        return 0.0
    a = lam * mean_service_s  # offered load in erlangs
    c = servers
    rho = a / c
    if rho >= 1.0:
        return float("inf")
    # P(wait) via the Erlang C formula, computed with a numerically stable recursion.
    term = 1.0  # a^k / k! running term, k = 0
    acc = term
    for k in range(1, c):
        term *= a / k
        acc += term
    term *= a / c  # a^c / c!
    tail = term / (1.0 - rho)
    p_wait = tail / (acc + tail)
    mu = 1.0 / mean_service_s
    return p_wait / (c * mu - lam)


def _graph_for_pending_pod(
    graph: dict[str, Any], dense_id: int, pending: frozenset[int]
) -> dict[str, list]:
    """Restrict a snapshot call graph to one pending pod's decision view.

    The snapshot projects every pending pod as serving so that no graph-aware scorer
    sees a constant view; a single scheduling decision, however, only involves *this*
    pod plus the already-serving replicas. Drop every edge incident on a different
    pending pod. Column order is preserved, so the result is deterministic.
    """
    others = pending - {dense_id}
    if not others:
        return {key: list(graph[key]) for key in ("src", "dst", "rate", "payload_kb")}
    keep = [
        index
        for index, (s, d) in enumerate(zip(graph["src"], graph["dst"], strict=True))
        if s not in others and d not in others
    ]
    return {
        key: [graph[key][index] for index in keep]
        for key in ("src", "dst", "rate", "payload_kb")
    }


class L0Evaluator:
    def __init__(self, library: SnapshotLibrary, weights: BoundaryWeights) -> None:
        self.library = library
        self.weights = weights
        config = library.config
        self._quantize = config.scheduler.quantize_scores
        self._floor_j = idle_floor_j(config.cluster, config.energy, config.horizon_s, weights)
        # Pre-thawed plain working copies of each snapshot ctx (snapshots themselves
        # are deep-frozen; the evaluator owns its mutable copies).
        self._plain: list[dict[str, Any]] = [s.to_dict() for s in library.snapshots]
        # Per-service offered rates and template edges (policy-free precomputation).
        self._service_rate_peak: dict[int, float] = {}
        self._service_rate_avg: dict[int, float] = {}
        self._backend_rate_peak: dict[str, float] = {}
        self._template_edges: dict[int, list[tuple[CallSpec, Any, float]]] = {}
        self._arrival_integral: dict[int, float] = {}
        # Rate deltas at activity-window endpoints. A sweep computes the true maximum
        # simultaneous aggregate load: inactive and disjoint templates must not be
        # summed into a fictitious peak (R2-6).
        rate_events: dict[float, dict[int, float]] = {}
        for template in config.workload.templates:
            cutoff = config.cutoff_s
            segments = template.arrival.segments(cutoff)
            offered_integral = (
                float(len(template.arrival.realized_times_s))
                if template.arrival.arrivals_realized
                else template.arrival.integrated_rate(cutoff)
            )
            self._arrival_integral[template.id] = offered_integral
            edges = _iter_edges_with_mode(template.root)
            self._template_edges[template.id] = edges
            visits: dict[int, float] = {template.root.service: 1.0}
            for parent_call, edge, parent_visits in edges:
                del parent_call
                v = parent_visits * edge.probability
                visits[edge.child.service] = visits.get(edge.child.service, 0.0) + v
            for sid, v in visits.items():
                self._service_rate_avg[sid] = (
                    self._service_rate_avg.get(sid, 0.0) + offered_integral * v / config.horizon_s
                )
                for start, end, segment_rate in segments:
                    rate = segment_rate * v
                    start_deltas = rate_events.setdefault(start, {})
                    start_deltas[sid] = start_deltas.get(sid, 0.0) + rate
                    end_deltas = rate_events.setdefault(end, {})
                    end_deltas[sid] = end_deltas.get(sid, 0.0) - rate
        current: dict[int, float] = {}
        endpoints = sorted(rate_events)
        backend_by_service = {
            service.id: service.backend
            for service in config.workload.services
            if service.backend is not None
        }
        for index, at_s in enumerate(endpoints[:-1]):
            for sid, delta in rate_events[at_s].items():
                current[sid] = current.get(sid, 0.0) + delta
            if endpoints[index + 1] <= at_s:
                continue
            for sid, rate in current.items():
                self._service_rate_peak[sid] = max(self._service_rate_peak.get(sid, 0.0), rate)
            backend_rates: dict[str, float] = {}
            for sid, rate in current.items():
                backend = backend_by_service.get(sid)
                if backend is not None:
                    backend_rates[backend] = backend_rates.get(backend, 0.0) + rate
            for backend, rate in backend_rates.items():
                self._backend_rate_peak[backend] = max(
                    self._backend_rate_peak.get(backend, 0.0), rate
                )
        # Effective mean service occupancy per service: service time + backend access
        # (the DES holds the server slot through the backend access).
        self._mean_occupancy_s: dict[int, float] = {}
        for svc in config.workload.services:
            mean_s = svc.service_time.mean()
            if svc.backend is not None:
                backend = config.workload.backend_by_name(svc.backend)
                lam_b = self._backend_rate_peak.get(svc.backend, 0.0)
                # Fixed point of L = curve(1 + lam_b * L): mean concurrency via
                # Little's law (documented approximation).
                lat = backend.latency_s(1.0)
                for _ in range(8):
                    lat = backend.latency_s(1.0 + lam_b * lat)
                mean_s += lat
            self._mean_occupancy_s[svc.id] = mean_s

    # -- policy execution over one snapshot -------------------------------------
    def evaluate(self, scorer: Scorer) -> L0Evaluation:
        per_snapshot: list[L0SnapshotResult] = []
        try:
            for snapshot, plain in zip(self.library.snapshots, self._plain, strict=True):
                per_snapshot.append(self._evaluate_snapshot(scorer, snapshot, plain))
        except PolicyError as exc:  # crash filter: the POLICY failed (host errors raise)
            return L0Evaluation(
                valid=False,
                error=f"{type(exc).__name__}: {exc}",
                feasible=False,
                mean_normalized_energy=None,
            )
        feasible = all(r.feasible for r in per_snapshot)
        violations = [v for r in per_snapshot for v in r.violations]
        mean_norm = sum(r.normalized_energy for r in per_snapshot) / len(per_snapshot)
        return L0Evaluation(
            valid=True,
            error=None,
            feasible=feasible,
            mean_normalized_energy=mean_norm,
            violations=violations,
            per_snapshot=per_snapshot,
        )

    def _evaluate_snapshot(
        self, scorer: Scorer, snapshot: Snapshot, plain: dict[str, Any]
    ) -> L0SnapshotResult:
        ctx = plain["ctx"]
        nodes = [dict(n) for n in ctx["nodes"]]
        physical_suspended = tuple(
            plain.get(
                "power_suspended_by_node",
                tuple(node["suspended"] for node in nodes),
            )
        )
        pods = [dict(p) for p in ctx["pods"]]
        work_ctx = {
            "nodes": nodes,
            "pods": pods,
            "graph": ctx["graph"],
            "now_ms": ctx["now_ms"],
        }
        nodes_by_id = {n["id"]: n for n in nodes}
        violations: list[str] = []
        decisions: list[tuple[int, int]] = []
        serving_dense_ids = {
            dense for dense, phase in enumerate(snapshot.pod_phases) if phase == "ready"
        }
        pending_set = frozenset(snapshot.pending_dense_ids)
        for dense_id in snapshot.pending_dense_ids:
            pod = pods[dense_id]
            feasible = filter_feasible(pod, nodes)
            if not feasible:
                violations.append(f"unschedulable: pod {dense_id}")
                continue
            # The snapshot graph projects *every* still-pending pod as serving; the DES
            # scoring path projects only the pod in its own cycle. Narrow to that view
            # so both rungs show a graph-aware scorer the same edge set: this pod's
            # prospective edges plus the serving-to-serving edges. (Rate shares still
            # carry the snapshot's replica count -- an L0 approximation, and L0 is the
            # gross-quality filter rung whose fidelity is measured by G-ladder.)
            work_ctx["graph"] = _graph_for_pending_pod(ctx["graph"], dense_id, pending_set)
            # The shared emulator pipeline: Score -> Normalize (per config) -> tie-break.
            best, _scores = score_and_select(
                scorer, pod, feasible, nodes_by_id, work_ctx, self._quantize
            )
            decisions.append((dense_id, best))
            # L0's static replay treats a successfully placed pending decision as the
            # resulting serving replica. Existing lifecycle states serve only if the
            # snapshot captured them READY; BINDING/PULLING/STARTING/TERMINATING pods
            # retain allocation/energy but provide no queueing capacity.
            serving_dense_ids.add(dense_id)
            pod["current_node"] = best
            nodes_by_id[best]["cpu_allocated"] += pod["cpu_request"]
            nodes_by_id[best]["mem_allocated"] += pod["mem_request"]
        return self._score_assignment(
            nodes,
            pods,
            serving_dense_ids,
            decisions,
            violations,
            physical_suspended,
        )

    # -- closed-form scoring of a static assignment --------------------------------
    def _score_assignment(
        self,
        nodes: list[dict[str, Any]],
        pods: list[dict[str, Any]],
        serving_dense_ids: set[int],
        decisions: list[tuple[int, int]],
        violations: list[str],
        physical_suspended: tuple[bool, ...],
    ) -> L0SnapshotResult:
        config = self.library.config
        services = config.workload.services
        replicas_by_service: dict[int, list[int]] = {}
        for dense_id, pod in enumerate(pods):
            if dense_id in serving_dense_ids and pod["current_node"] >= 0:
                replicas_by_service.setdefault(pod["service"], []).append(pod["current_node"])
        # Station sojourn per service: Erlang-C M/M/c on the peak per-replica rate.
        sojourn: dict[int, float] = {}
        for sid, placed in replicas_by_service.items():
            spec = services[sid]
            mean_occ = self._mean_occupancy_s[sid]
            lam = self._service_rate_peak.get(sid, 0.0) / len(placed)
            rho = lam * mean_occ / spec.concurrency
            if rho >= 1.0:
                violations.append(f"saturated: service {sid} rho={rho:.3f}")
                sojourn[sid] = float("inf")
            else:
                sojourn[sid] = mean_occ + _erlang_c_wait_s(lam, mean_occ, spec.concurrency)
        for sid, rate in self._service_rate_peak.items():
            if rate > 0 and sid not in replicas_by_service:
                violations.append(f"no_replica: service {sid}")

        # Energy closed forms (time-averaged rates over each template's window).
        cluster = config.cluster
        horizon = config.horizon_s
        device_j = 0.0
        cloud_millicores = 0.0
        busy = {n["id"]: 0.0 for n in nodes}
        for sid, placed in replicas_by_service.items():
            spec = services[sid]
            mean_s = spec.service_time.mean()  # CPU is held for the service time only
            lam_avg = self._service_rate_avg.get(sid, 0.0) / len(placed)
            for nid in placed:
                busy[nid] += lam_avg * mean_s * spec.cpu_demand
        for n in nodes:
            nid = n["id"]
            cls = cluster.node_class_of(nid)
            if cluster.nodes[nid].tier == TIER_CLOUD:
                cloud_millicores += n["cpu_allocated"]
                continue
            if physical_suspended[nid]:
                device_j += cls.suspended_power_w * horizon
            else:
                u = min(busy[nid] / n["cpu_capacity"], 1.0)
                device_j += cls.power_w(u) * horizon
        cloud_j = (
            (config.energy.cloud_floor_vcpu + cloud_millicores / 1000.0)
            * (horizon / 3600.0)
            * config.energy.cloud_j_per_vcpu_hour
        )
        network_j = self._network_j(replicas_by_service)

        weights = self.weights
        closed = weights.device * device_j + weights.cloud * cloud_j + weights.network * network_j
        # Latency estimates + SLO checks.
        latency_est: dict[int, float] = {}
        for template in config.workload.templates:
            est = self._template_latency_s(
                template.root, sojourn, replicas_by_service, cluster.latency_ms
            )
            latency_est[template.id] = est
            if template.slo_latency_s is not None and est > template.slo_latency_s:
                violations.append(
                    f"slo: template {template.id} est={est:.4f}s > {template.slo_latency_s}s"
                )
        return L0SnapshotResult(
            feasible=not violations,
            violations=violations,
            decisions=decisions,
            device_j=device_j,
            cloud_j=cloud_j,
            network_j=network_j,
            closed_boundary_j=closed,
            normalized_energy=closed / self._floor_j,
            template_latency_est_s=latency_est,
        )

    def _network_j(self, replicas_by_service: dict[int, list[int]]) -> float:
        """Expected cross-node bytes from configured edge rates over each template's
        arrival window, uniform LB split. Byte accounting matches the DES: request bytes
        on every cross-node call; response bytes on blocking edges only."""
        config = self.library.config
        total_bytes = 0.0
        for template in config.workload.templates:
            offered = self._arrival_integral[template.id]
            for parent_call, edge, parent_visits in self._template_edges[template.id]:
                srcs = replicas_by_service.get(parent_call.service)
                dsts = replicas_by_service.get(edge.child.service)
                if not srcs or not dsts:
                    continue
                cross = sum(1 for s in srcs for d in dsts if s != d) / (len(srcs) * len(dsts))
                kb = edge.request_kb + (edge.response_kb if parent_call.mode == "blocking" else 0.0)
                total_bytes += offered * parent_visits * edge.probability * cross * kb * 1000.0
        return total_bytes * config.energy.network_j_per_byte

    def _template_latency_s(
        self,
        call: Any,
        sojourn: dict[int, float],
        replicas_by_service: dict[int, list[int]],
        latency_ms: Any,
    ) -> float:
        """Recursive path-latency estimate. Children run concurrently in the DES for
        both modes (AllOf fan-in), so both use the max-of-expectations makespan
        estimate; blocking edges pay two network hops, forward handoffs one."""
        own = sojourn.get(call.service, float("inf"))
        if own == float("inf"):
            return float("inf")
        child_terms = []
        for edge in call.children:
            child_t = self._template_latency_s(edge.child, sojourn, replicas_by_service, latency_ms)
            net = self._expected_net_s(
                call.service, edge.child.service, replicas_by_service, latency_ms
            )
            hops = 2.0 if call.mode == "blocking" else 1.0
            child_terms.append((edge.probability, hops * net + child_t))
        if not child_terms:
            return own
        return own + max(p * t for p, t in child_terms)

    def _expected_net_s(
        self,
        src_sid: int,
        dst_sid: int,
        replicas_by_service: dict[int, list[int]],
        latency_ms: Any,
    ) -> float:
        srcs = replicas_by_service.get(src_sid)
        dsts = replicas_by_service.get(dst_sid)
        if not srcs or not dsts:
            return float("inf")
        total = sum(latency_ms[s][d] for s in srcs for d in dsts)
        return total / (len(srcs) * len(dsts)) / 1000.0


# ----------------------------------------------------------------- independent oracle
def _oracle_visits(call: CallSpec, acc: dict[int, float], visits: float) -> None:
    acc[call.service] = acc.get(call.service, 0.0) + visits
    for edge in call.children:
        _oracle_visits(edge.child, acc, visits * edge.probability)


def _oracle_energy_j(
    config: EpisodeConfig,
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    weights: BoundaryWeights,
    physical_suspended: tuple[bool, ...],
) -> float:
    """Self-contained closed-boundary energy of a static assignment — deliberately
    independent of ``L0Evaluator`` (its verification oracle, R1-10). Implements the
    same *specified* model (time-averaged busy -> power curve; allocated cloud incl.
    floor; cross-node bytes with blocking-edge responses) from scratch."""
    horizon = config.horizon_s
    services = config.workload.services
    placed: dict[int, list[int]] = {}
    for p in pods:
        if p["current_node"] >= 0:
            placed.setdefault(p["service"], []).append(p["current_node"])
    # Per-service time-averaged arrival rates.
    avg_rate: dict[int, float] = {}
    for t in config.workload.templates:
        offered = (
            float(len(t.arrival.realized_times_s))
            if t.arrival.arrivals_realized
            else t.arrival.integrated_rate(config.cutoff_s)
        )
        visits: dict[int, float] = {}
        _oracle_visits(t.root, visits, 1.0)
        for sid, v in visits.items():
            avg_rate[sid] = avg_rate.get(sid, 0.0) + offered * v / horizon
    busy: dict[int, float] = {}
    for sid, node_list in placed.items():
        svc = services[sid]
        lam = avg_rate.get(sid, 0.0) / len(node_list)
        for nid in node_list:
            busy[nid] = busy.get(nid, 0.0) + lam * svc.service_time.mean() * svc.cpu_demand
    device = 0.0
    cloud_mc = 0.0
    for n in nodes:
        cls = config.cluster.node_class_of(n["id"])
        if config.cluster.nodes[n["id"]].tier == TIER_CLOUD:
            cloud_mc += n["cpu_allocated"]
        elif physical_suspended[n["id"]]:
            device += cls.suspended_power_w * horizon
        else:
            device += cls.power_w(min(busy.get(n["id"], 0.0) / n["cpu_capacity"], 1.0)) * horizon
    cloud = (
        (config.energy.cloud_floor_vcpu + cloud_mc / 1000.0)
        * (horizon / 3600.0)
        * config.energy.cloud_j_per_vcpu_hour
    )
    net_bytes = 0.0

    def walk(call: CallSpec, visits: float, offered: float) -> None:
        nonlocal net_bytes
        for edge in call.children:
            srcs = placed.get(call.service)
            dsts = placed.get(edge.child.service)
            v = visits * edge.probability
            if srcs and dsts:
                cross = sum(1 for s in srcs for d in dsts if s != d) / (len(srcs) * len(dsts))
                kb = edge.request_kb + (edge.response_kb if call.mode == "blocking" else 0.0)
                net_bytes += offered * v * cross * kb * 1000.0
            walk(edge.child, v, offered)

    for t in config.workload.templates:
        offered = (
            float(len(t.arrival.realized_times_s))
            if t.arrival.arrivals_realized
            else t.arrival.integrated_rate(config.cutoff_s)
        )
        walk(t.root, 1.0, offered)
    network = net_bytes * config.energy.network_j_per_byte
    return weights.device * device + weights.cloud * cloud + weights.network * network


def brute_force_optimum(
    config: EpisodeConfig, snapshot: Snapshot, weights: BoundaryWeights
) -> tuple[float, list[tuple[int, int]]]:
    """Enumerate every capacity-feasible assignment of the snapshot's pending pods and
    return the minimum closed-boundary energy and one argmin assignment. Assignments
    are applied **directly** (no scorer, no shared pipeline) and scored by the
    independent oracle implementation — verification-suite ground truth for the
    enumerable two-node placement micro-scenarios (exponential; tiny cases only)."""
    from itertools import product

    base = snapshot.to_dict()
    pending = snapshot.pending_dense_ids
    node_ids = [n["id"] for n in base["ctx"]["nodes"]]
    physical_suspended = tuple(
        base.get(
            "power_suspended_by_node",
            tuple(node["suspended"] for node in base["ctx"]["nodes"]),
        )
    )
    best: float | None = None
    best_assign: list[tuple[int, int]] = []
    for combo in product(node_ids, repeat=len(pending)):
        nodes = [dict(n) for n in base["ctx"]["nodes"]]
        pods = [dict(p) for p in base["ctx"]["pods"]]
        by_id = {n["id"]: n for n in nodes}
        ok = True
        for dense_id, nid in zip(pending, combo, strict=True):
            pod = pods[dense_id]
            node = by_id[nid]
            if (
                node["suspended"]
                or node["cpu_allocated"] + pod["cpu_request"] > node["cpu_capacity"]
                or node["mem_allocated"] + pod["mem_request"] > node["mem_capacity"]
            ):
                ok = False
                break
            pod["current_node"] = nid
            node["cpu_allocated"] += pod["cpu_request"]
            node["mem_allocated"] += pod["mem_request"]
        if not ok:
            continue
        energy = _oracle_energy_j(config, nodes, pods, weights, physical_suspended)
        if best is None or energy < best:
            best = energy
            best_assign = list(zip(pending, combo, strict=True))
    if best is None:
        raise ValueError("no feasible assignment exists for this snapshot")
    return best, best_assign
