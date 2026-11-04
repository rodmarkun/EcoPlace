"""Scheduling-cycle emulator, scope matched to the pinned kube-scheduler (TODO 2.2, S2).

Reproduced semantics: **serial scheduling cycles** (one pod at a time through
PreScore -> Score -> NormalizeScore -> Reserve) with **asynchronous binding** (a separate
process per bound pod; cycles continue while bindings are in flight), assumed-pod state
(Reserve immediately adds the pod to the scheduler cache; a binding failure Unreserves
it), informer **cache staleness** for externally-caused changes (rebalancer binds,
terminations, suspend/wake propagate to the cache after ``informer_delay_s``), binding
failures with exponential retry/backoff, and a **documented deterministic tie-break**
(quantized-score maximum, ties -> lowest node id — a stated divergence from the real
scheduler's randomized ``selectHost``, handled at G2 by tie-free fixtures).

Every reproduced-or-pinned-off behavior is an explicit entry in ``EMULATOR_COVERAGE``
below — this list is the Phase-6 input for the G2 decision-conformance coverage
enumeration (plan §3.5). Pinned-off features are *absent from both systems* during G2
(disabled in the real scheduler's profile), never silently approximated.

Scorers: the evolved-policy path goes through the PlacementScript **batch API** (one
boundary crossing per cycle, mirroring the T1 PreScore batch design); plain-Python
callables are supported for baselines via ``CallableScorer``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import simpy

from sim.runtime import (
    AWAKE,
    BINDING,
    PENDING,
    READY,
    SUSPENDED,
    ClusterRuntime,
    HostError,
    Ledger,
    PodRuntime,
    PolicyError,
    SimulationError,
)
from sim.spec import SchedulerParams, WorkloadSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import numpy as np


# --------------------------------------------------------------- G2 coverage enumeration
@dataclass(frozen=True)
class SchedulerPin:
    """The pinned real-scheduler configuration the emulator is matched against (S2).
    G2 (Phase 6) streams identical fixtures to this exact scheduler build/profile; any
    change to the pin re-opens the coverage list."""

    kubernetes_version: str
    profile_name: str
    score_plugins: tuple[tuple[str, int], ...]  # (plugin, weight)
    filter_plugins: tuple[str, ...]
    percentage_of_nodes_to_score: int
    queue_backoff_initial_s: float
    queue_backoff_max_s: float
    disabled_plugins_note: str


PINNED_SCHEDULER = SchedulerPin(
    kubernetes_version="v1.30.14",
    profile_name="continuum-g2",
    # Exactly one Score path: the T1 PlacementScript plugin (PreScore batch -> Score
    # CycleState lookup -> explicit NormalizeScore), weight 1. All default score
    # plugins are disabled in the G2 profile.
    score_plugins=(("PlacementScriptScore", 1),),
    # NodeResourcesFit (cpu+memory, LeastAllocated scoring DISABLED - filter only) and
    # node readiness (TaintToleration is disabled; readiness enters via the emulator's
    # suspended-node filter, mirrored on KWOK by cordoning suspended nodes).
    filter_plugins=("NodeResourcesFit", "NodeUnschedulable"),
    percentage_of_nodes_to_score=100,
    queue_backoff_initial_s=1.0,
    queue_backoff_max_s=10.0,
    disabled_plugins_note=(
        "all default Score plugins, PostFilter/preemption, Permit, PodTopologySpread, "
        "InterPodAffinity, VolumeBinding, TaintToleration disabled in the G2 profile; "
        "PriorityClass unused in fixtures"
    ),
)


@dataclass(frozen=True)
class CoverageEntry:
    behavior: str
    # "reproduced": semantics matched to the pinned scheduler on the covered surface.
    # "approximated": deliberately simplified; divergence characterized in notes; G2
    #                 fixtures must avoid or bound the approximation.
    # "pinned_off": feature disabled in BOTH systems during G2.
    # "documented_divergence": known behavioral difference, handled by fixture design.
    status: str
    notes: str


EMULATOR_COVERAGE: tuple[CoverageEntry, ...] = (
    CoverageEntry(
        "serial_scheduling_cycles",
        "reproduced",
        "one pod per cycle; PreScore -> Score -> NormalizeScore -> Reserve in one step",
    ),
    CoverageEntry(
        "asynchronous_binding",
        "reproduced",
        "binding runs as its own process; cycles proceed while bindings are in flight",
    ),
    CoverageEntry(
        "assumed_pod_cache",
        "reproduced",
        "Reserve assumes the pod in the cache (full reservation record: node + "
        "resources); bind confirms; failure unreserves; transitions are strict "
        "(assume->confirm|unreserve exactly once, violations raise)",
    ),
    CoverageEntry(
        "assumed_pod_expiration",
        "pinned_off",
        "kube expires assumed pods after a TTL if bind never confirms; in-model bindings "
        "always terminate, so expiration cannot trigger",
    ),
    CoverageEntry(
        "informer_cache_staleness",
        "approximated",
        "externally-caused truth changes reach the cache after a FIXED informer_delay_s "
        "(real informers have variable, load-dependent latency); G2 deterministic "
        "fixtures run with delay 0 where the real path is not exercised",
    ),
    CoverageEntry(
        "binding_failure_retry_backoff",
        "reproduced",
        "injected/validated bind failures unreserve + requeue with per-pod exponential "
        "backoff (initial/max = the pinned queue config, kube defaults 1s/10s)",
    ),
    CoverageEntry(
        "scheduling_queue",
        "approximated",
        "activeQ is FIFO over arrival; a failed pod re-enters after its per-pod backoff "
        "expires (timer), i.e. ordering among retried pods is by backoff-expiry time, "
        "not creation sequence. Kubernetes' unschedulable pool + cluster-event "
        "activation (QueueingHint) is NOT modeled: retries are purely timer-driven. "
        "G2 fixtures must not depend on event-driven requeue ordering",
    ),
    CoverageEntry(
        "pod_priority_queue",
        "pinned_off",
        "no PriorityClass; queue order is FIFO — G2 profile disables priorities",
    ),
    CoverageEntry(
        "postfilter_preemption",
        "pinned_off",
        "unschedulable pods requeue with backoff; no victim selection in either system",
    ),
    CoverageEntry(
        "permit_plugins",
        "pinned_off",
        "no Permit stage; nothing gates between Reserve and Bind",
    ),
    CoverageEntry(
        "node_sampling_percentage",
        "pinned_off",
        "all feasible nodes are scored (percentageOfNodesToScore=100 in the G2 profile)",
    ),
    CoverageEntry(
        "score_quantization",
        "reproduced",
        "NormalizeScore maps raw policy scores to int [0,100] per cycle (min->0, "
        "max->100), matching the T1 plugin's quantized CycleState scores; toggleable "
        "(quantize_scores=False is an emulator-only diagnostic mode, never used at G2)",
    ),
    CoverageEntry(
        "tie_breaking",
        "documented_divergence",
        "quantized-score ties -> lowest node id (deterministic); the real selectHost "
        "picks uniformly among maxima — G2 acceptance fixtures are tie-free",
    ),
    CoverageEntry(
        "filter_plugins",
        "reproduced",
        "NodeResourcesFit limited to cpu+memory requests vs allocatable incl. assumed "
        "pods (no extended resources, no init/overhead accounting — fixtures restricted "
        "accordingly) and node readiness (suspended nodes unschedulable, mirrored by "
        "cordoning on KWOK); all other Filters pinned off in the G2 profile",
    ),
    CoverageEntry(
        "bind_time_capacity_revalidation",
        "approximated",
        "binding fails if truth capacity no longer fits at bind time. This is a KUBELET "
        "admission analogue (OutOfCpu/OutOfMemory rejection), NOT kube-apiserver "
        "behavior (the API server performs no aggregate fit check on Bind). Divergence "
        "from kube: a kubelet-rejected pod goes Failed and is replaced by its "
        "controller; here the same pod object requeues. Toggleable via SchedulerParams; "
        "not exercised by deterministic G2 fixtures",
    ),
    CoverageEntry(
        "scorer_error_requeue",
        "reproduced",
        "a policy error in PreScore fails the cycle and requeues the pod with backoff "
        "(T1 framework-error semantics; circuit-breaker rollback is Phase-6 scope). "
        "Host/simulator errors are NOT absorbed into this path (they raise)",
    ),
    CoverageEntry(
        "pod_list_staleness_in_ctx",
        "pinned_off",
        "the ClusterCtx pod list is built from truth; only node allocations/power states "
        "carry informer staleness (simplification, listed for G2 honesty)",
    ),
    CoverageEntry(
        "g2_event_trace",
        "reproduced",
        "the decision log records reserves AND unschedulable cycles, scorer errors, "
        "binding failures (injected/validation), and assume/confirm/unreserve cache "
        "transitions — the full event surface the G2 trace diff consumes",
    ),
)


def coverage_as_dicts() -> list[dict[str, str]]:
    return [
        {"behavior": e.behavior, "status": e.status, "notes": e.notes} for e in EMULATOR_COVERAGE
    ]


# ------------------------------------------------------------------------- scorers
class Scorer(Protocol):
    """Batch scorer contract: mirrors the PlacementScript batch ABI shape."""

    name: str

    def score_batch(
        self, pod: dict[str, Any], nodes: list[dict[str, Any]], ctx: dict[str, Any]
    ) -> list[float]: ...


class PlacementScriptScorer:
    """Adapter over the native PyO3 batch API (one boundary crossing per cycle)."""

    def __init__(self, policy: Any) -> None:  # placementscript.Policy
        self._policy = policy
        self.name = f"ps:{policy.name}@{policy.version}"

    def score_batch(
        self, pod: dict[str, Any], nodes: list[dict[str, Any]], ctx: dict[str, Any]
    ) -> list[float]:
        try:
            return list(self._policy.score_batch(pod, nodes, ctx))
        except Exception as exc:
            # The native boundary has a distinct HostError class. Translate that
            # exact type, never a spoofable class name, into the simulator host error.
            import placementscript as ps

            if isinstance(exc, ps.HostError):
                raise HostError(str(exc)) from exc
            raise


class CallableScorer:
    """Adapter for plain-Python scorers (baselines). ``fn(pod, node, ctx) -> float`` must
    be pure and deterministic; -0.0 is normalized on the boundary like the native path."""

    def __init__(self, name: str, fn: Callable[[dict, dict, dict], float]) -> None:
        self.name = f"py:{name}"
        self._fn = fn

    def score_batch(
        self, pod: dict[str, Any], nodes: list[dict[str, Any]], ctx: dict[str, Any]
    ) -> list[float]:
        out = []
        for node in nodes:
            v = float(self._fn(pod, node, ctx))
            if not math.isfinite(v):
                raise ValueError(f"scorer {self.name} returned a non-finite score")
            out.append(0.0 if v == 0.0 else v)  # -0.0 -> 0.0, matching the native boundary
        return out


def normalize_scores(raw: list[float]) -> list[int]:
    """NormalizeScore: affine map of the cycle's raw scores onto int [0,100]
    (min -> 0, max -> 100; all-equal -> all 100), matching the T1 plugin behavior."""
    lo, hi = min(raw), max(raw)
    if hi == lo:
        return [100] * len(raw)
    return [round(100.0 * (v - lo) / (hi - lo)) for v in raw]


def select_best(node_ids: list[int], scores: list[float | int]) -> int:
    """Deterministic tie-break: maximum score, ties -> lowest node id."""
    best = node_ids[0]
    best_score = scores[0]
    for nid, s in zip(node_ids[1:], scores[1:], strict=True):
        if s > best_score or (s == best_score and nid < best):
            best, best_score = nid, s
    return best


def filter_feasible(pod: dict[str, Any], nodes: list[dict[str, Any]]) -> list[int]:
    """The shared Filter stage (NodeResourcesFit cpu+mem, node readiness) over
    ctx-shaped node dicts. Used by the emulator (cache view) AND L0 (snapshot view) —
    one pipeline, two state sources (R1-9)."""
    out = []
    for n in nodes:
        if n["suspended"]:
            continue
        if n["cpu_allocated"] + pod["cpu_request"] > n["cpu_capacity"]:
            continue
        if n["mem_allocated"] + pod["mem_request"] > n["mem_capacity"]:
            continue
        out.append(n["id"])
    return out


def score_and_select(
    scorer: Scorer,
    pod: dict[str, Any],
    feasible: list[int],
    nodes_by_id: dict[int, dict[str, Any]],
    ctx: dict[str, Any],
    quantize: bool,
) -> tuple[int, list[float | int]]:
    """The shared Score -> NormalizeScore -> tie-break pipeline (emulator + L0).
    Raises ``PolicyError`` when the scorer misbehaves (its fault), and lets host-side
    errors propagate untouched (never charged to the candidate — R1-12)."""
    candidates = [nodes_by_id[nid] for nid in feasible]
    try:
        raw = [float(v) for v in scorer.score_batch(pod, candidates, ctx)]
    except HostError:
        raise  # our fault, not the policy's
    except PolicyError:
        raise
    except Exception as exc:
        raise PolicyError(f"{type(exc).__name__}: {exc}") from exc
    if len(raw) != len(feasible):
        raise PolicyError(f"scorer returned {len(raw)} scores for {len(feasible)} candidates")
    if any(not math.isfinite(v) for v in raw):
        raise PolicyError("scorer returned a non-finite score")
    scores: list[float | int] = list(normalize_scores(raw)) if quantize else list(raw)
    return select_best(feasible, scores), scores


# ------------------------------------------------------------------------ cache
class SchedulerCache:
    """The scheduler's view of node allocations and power states.

    Scheduler-initiated changes (assume/confirm/unreserve) apply immediately; external
    truth changes arrive as change records and apply after ``informer_delay_s``.

    Assumed pods are full **reservation records** (uid -> (node, cpu, mem)) with strict
    transitions: ``assume`` of an already-assumed pod, or ``confirm``/``unreserve`` of an
    unknown pod or against the wrong node, raises ``SimulationError`` (a corrupted cache
    must never be silently repaired — R1-6).
    """

    def __init__(
        self, env: simpy.Environment, cluster: ClusterRuntime, informer_delay_s: float
    ) -> None:
        self.env = env
        self._delay = informer_delay_s
        self.cpu_alloc = [n.cpu_allocated for n in cluster.nodes]
        self.mem_alloc = [n.mem_allocated for n in cluster.nodes]
        self._power_suspended = [not n.awake for n in cluster.nodes]
        self._unavailable = [not n.schedulable for n in cluster.nodes]
        # PlacementScript ABI compatibility: ``suspended`` means unavailable for new
        # placement, whether by physical power state or planned cordon.
        self.suspended = [
            power or unavailable
            for power, unavailable in zip(self._power_suspended, self._unavailable, strict=True)
        ]
        self.assumed: dict[int, tuple[int, int, int]] = {}  # uid -> (node, cpu, mem)
        self._pending_changes: list[dict[str, Any]] = []  # scheduled, not yet applied
        cluster.subscribe(self._deliver)

    def _deliver(self, change: dict[str, Any]) -> None:
        if self._delay <= 0.0:
            self._apply(change)
        else:
            self._pending_changes.append(change)
            self.env.process(self._delayed(change))

    def _delayed(self, change: dict[str, Any]) -> Generator[Any, Any, None]:
        yield self.env.timeout(self._delay)
        if change in self._pending_changes:  # not already force-flushed
            self._pending_changes.remove(change)
            self._apply(change)

    def _apply(self, change: dict[str, Any]) -> None:
        if change["kind"] == "alloc":
            self.cpu_alloc[change["node"]] += change["cpu"]
            self.mem_alloc[change["node"]] += change["mem"]
        elif change["kind"] == "power":
            nid = change["node"]
            self._power_suspended[nid] = change["suspended"]
            self.suspended[nid] = self._power_suspended[nid] or self._unavailable[nid]
        elif change["kind"] == "availability":
            nid = change["node"]
            self._unavailable[nid] = change["unavailable"]
            self.suspended[nid] = self._power_suspended[nid] or self._unavailable[nid]

    def flush_pending(self) -> None:
        """Apply every undelivered informer change now (end-of-episode reconciliation)."""
        pending, self._pending_changes = self._pending_changes, []
        for change in pending:
            self._apply(change)

    # scheduler-initiated paths (no staleness)
    def assume(self, pod: PodRuntime, node_id: int) -> None:
        if pod.uid in self.assumed:
            raise SimulationError(f"pod {pod.uid} assumed twice")
        if not 0 <= node_id < len(self.cpu_alloc):
            raise SimulationError(f"assume of pod {pod.uid} on unknown node {node_id}")
        cpu, mem = pod.service.cpu_request, pod.service.mem_request
        self.assumed[pod.uid] = (node_id, cpu, mem)
        self.cpu_alloc[node_id] += cpu
        self.mem_alloc[node_id] += mem

    def confirm(self, pod: PodRuntime, node_id: int) -> None:
        rec = self.assumed.get(pod.uid)
        if rec is None or rec[0] != node_id:
            raise SimulationError(
                f"confirm of pod {pod.uid} on node {node_id} without matching assume {rec}"
            )
        del self.assumed[pod.uid]
        # allocation already counted at assume

    def unreserve(self, pod: PodRuntime, node_id: int) -> None:
        rec = self.assumed.get(pod.uid)
        if rec is None or rec[0] != node_id:
            raise SimulationError(
                f"unreserve of pod {pod.uid} on node {node_id} without matching assume {rec}"
            )
        _, cpu, mem = rec
        if self.cpu_alloc[node_id] < cpu or self.mem_alloc[node_id] < mem:
            raise SimulationError(
                f"unreserve of pod {pod.uid} would make cache allocation negative"
            )
        del self.assumed[pod.uid]
        self.cpu_alloc[node_id] -= cpu
        self.mem_alloc[node_id] -= mem

    def verify_against_truth(self, cluster: ClusterRuntime) -> None:
        """End-of-episode reconciliation (R1-6): after flushing pending informer
        deliveries, the cache must equal truth plus the still-assumed reservations, and
        no allocation may be negative."""
        self.flush_pending()
        expected_cpu = [n.cpu_allocated for n in cluster.nodes]
        expected_mem = [n.mem_allocated for n in cluster.nodes]
        for node_id, cpu, mem in self.assumed.values():
            expected_cpu[node_id] += cpu
            expected_mem[node_id] += mem
        for i, n in enumerate(cluster.nodes):
            if self.cpu_alloc[i] < 0 or self.mem_alloc[i] < 0:
                raise SimulationError(f"negative cache allocation on node {i}")
            if self.cpu_alloc[i] != expected_cpu[i] or self.mem_alloc[i] != expected_mem[i]:
                raise SimulationError(
                    f"cache/truth reconciliation failed on node {i}: cache "
                    f"({self.cpu_alloc[i]}, {self.mem_alloc[i]}) != truth+assumed "
                    f"({expected_cpu[i]}, {expected_mem[i]})"
                )
            # Power-state agreement is only defined for settled states: a node caught
            # mid-transition (SUSPENDING/WAKING) at the horizon has not emitted its
            # completion notification yet, by design.
            if n.power_state in (AWAKE, SUSPENDED):
                expected_unavailable = not n.awake or not n.schedulable
                if self.suspended[i] is not expected_unavailable:
                    raise SimulationError(f"cache availability stale for node {i} after flush")
            elif not n.schedulable and self.suspended[i] is not True:
                # Power transitions notify only on completion; planned availability
                # remains independently observable during that transition.
                raise SimulationError(f"cache planned availability stale for node {i} after flush")


# --------------------------------------------------------------------- ctx building
def build_ctx(
    cluster: ClusterRuntime,
    workload: WorkloadSpec,
    cache: SchedulerCache | None,
    now_s: float,
    cutoff_s: float,
    include_pod: PodRuntime | None = None,
    prospective: Sequence[PodRuntime] | None = None,
) -> tuple[dict[str, Any], dict[int, int], dict[int, int]]:
    """Build a spec-§4-shaped ``ClusterCtx`` dict.

    Node allocations/power come from the scheduler ``cache`` view when given (emulator
    path — staleness applies), or from truth when ``cache is None`` (rebalancer path:
    the controller reads current state). Pod ids are remapped to a dense 0..P-1 space
    (live pods in creation order); returns ``(ctx, uid_to_dense, dense_to_uid)``.

    The call graph carries the *configured* per-edge rates (spec rates x expected
    visits), split uniformly across serving replica pairs — the observation schema every
    method sees, per §6.2 'identical observations'.

    ``prospective`` names pods projected into the call-graph expansion as if serving
    (see :func:`_pod_level_graph`).  It defaults to ``include_pod`` alone on the
    scheduling path, so the pod being scored observes the edges it will carry; pass an
    explicit sequence (possibly empty) to override.  The rebalancer path passes neither
    and is unchanged.
    """
    spec = cluster.spec
    n = len(cluster.nodes)
    nodes_out = []
    for node in cluster.nodes:
        cls = node.node_class
        suspended = (
            cache.suspended[node.id]
            if cache is not None
            else (not node.awake or not node.schedulable)
        )
        nodes_out.append(
            {
                "id": node.id,
                "tier": node.spec.tier,
                "cpu_capacity": node.spec.cpu_capacity,
                "cpu_allocated": (
                    cache.cpu_alloc[node.id] if cache is not None else node.cpu_allocated
                ),
                "mem_capacity": node.spec.mem_capacity,
                "mem_allocated": (
                    cache.mem_alloc[node.id] if cache is not None else node.mem_allocated
                ),
                "power_idle_w": float(cls.power_idle_w),
                "power_max_w": float(cls.power_max_w),
                "suspended": suspended,
                "suspendable": cls.suspendable and node.schedulable,
                "latency_to_ms": [float(spec.latency_ms[node.id][j]) for j in range(n)],
            }
        )
    live = cluster.live_pods()
    if include_pod is not None and include_pod not in live:
        live.append(include_pod)
    live.sort(key=lambda p: p.uid)
    if prospective is None:
        prospective_uids = frozenset() if include_pod is None else frozenset({include_pod.uid})
    else:
        prospective_uids = frozenset(p.uid for p in prospective)
    uid_to_dense = {p.uid: i for i, p in enumerate(live)}
    dense_to_uid = {i: p.uid for i, p in enumerate(live)}
    pods_out = [
        {
            "id": i,
            "service": p.service.id,
            "app": p.service.app,
            "cpu_request": p.service.cpu_request,
            "mem_request": p.service.mem_request,
            "stateful": p.service.stateful,
            "has_pdb": p.has_pdb,
            "image_mb": p.service.image_mb,
            "current_node": p.node_id if p.node_id is not None else -1,
        }
        for i, p in enumerate(live)
    ]
    ctx = {
        "nodes": nodes_out,
        "pods": pods_out,
        "graph": _pod_level_graph(
            workload, live, uid_to_dense, now_s, cutoff_s, prospective_uids
        ),
        "now_ms": int(now_s * 1000),
    }
    return ctx, uid_to_dense, dense_to_uid


def _pod_level_graph(
    workload: WorkloadSpec,
    live: list[PodRuntime],
    uid_to_dense: dict[int, int],
    now_s: float,
    cutoff_s: float,
    prospective_uids: frozenset[int] = frozenset(),
) -> dict[str, list]:
    """Expand edges over READY replicas with the DES/L0 activity-window mask.

    ``prospective_uids`` names pods that are *not yet* READY but whose placement is
    being decided right now.  They are projected into the expansion as if they were
    serving, which is the only way a **scoring** policy can observe the call-graph
    edges the pod it is placing will actually carry: the pod under decision is by
    construction not READY, so a READY-only expansion gives every graph-aware scorer a
    constant (empty) view and collapses it onto ``first_fit``.  The projection is part
    of the shared observation schema (plan §6.2 "identical observations"), so every
    method -- hand-written, GP, GPHH, random-AST and LLM -- sees it identically.  Rate
    shares use the post-placement replica count, which is what the edge will carry once
    the pod is READY.  Policies that never read ``ctx.graph`` are bit-for-bit
    unaffected.
    """
    src: list[int] = []
    dst: list[int] = []
    rate: list[float] = []
    payload: list[float] = []
    ready_by_service: dict[int, list[int]] = {}
    for p in live:
        if p.phase == READY or p.uid in prospective_uids:
            ready_by_service.setdefault(p.service.id, []).append(p.uid)
    for template in workload.templates:
        active = template.arrival.start_s <= now_s < cutoff_s
        configured_rate = template.arrival.rate_at(now_s) if active else 0.0
        for parent_sid, edge, visits, parent_mode in iter_template_edges(template.root):
            edge_rate = configured_rate * visits * edge.probability
            srcs = ready_by_service.get(parent_sid, [])
            dsts = ready_by_service.get(edge.child.service, [])
            if not srcs or not dsts:
                continue
            share = edge_rate / (len(srcs) * len(dsts))
            kb = edge.request_kb + (edge.response_kb if parent_mode == "blocking" else 0.0)
            for s_uid in srcs:
                for d_uid in dsts:
                    src.append(uid_to_dense[s_uid])
                    dst.append(uid_to_dense[d_uid])
                    rate.append(float(share))
                    payload.append(float(kb))
    return {"src": src, "dst": dst, "rate": rate, "payload_kb": payload}


def iter_template_edges(call: Any, visits: float = 1.0) -> list[tuple[int, Any, float, str]]:
    """Flatten into (parent service, edge, expected visits, parent call mode)."""
    out = []
    for edge in call.children:
        out.append((call.service, edge, visits, call.mode))
        out.extend(iter_template_edges(edge.child, visits * edge.probability))
    return out


# ------------------------------------------------------------------------- emulator
class SchedulerEmulator:
    """The serial-cycle scheduler process. ``on_bound(pod)`` is the engine hook that
    starts the pod's startup lifecycle (image pull, cold start).

    ``rng_for`` spawns keyed substreams: bind latency and bind-failure injection use
    *separate* streams keyed by the pod's stable identity (service, per-service ordinal,
    binding attempt) — a placement change never shifts another pod's bind draws (R1-7).
    """

    def __init__(
        self,
        env: simpy.Environment,
        cluster: ClusterRuntime,
        workload: WorkloadSpec,
        cutoff_s: float,
        params: SchedulerParams,
        scorer: Scorer,
        ledger: Ledger,
        rng_for: Callable[..., np.random.Generator],
        on_bound: Callable[[PodRuntime], Any],
        on_cycle: Callable[[PodRuntime, int], None] | None = None,
    ) -> None:
        self.env = env
        self.cluster = cluster
        self.workload = workload
        self.cutoff_s = cutoff_s
        self.params = params
        self.scorer = scorer
        self.ledger = ledger
        self._rng_for = rng_for
        self._on_bound = on_bound
        self._on_cycle = on_cycle
        self.queue: simpy.Store = simpy.Store(env)
        self.cache = SchedulerCache(env, cluster, params.informer_delay_s)
        self._backoff_until: dict[int, float] = {}  # pod uid -> backoff expiry
        self._in_cycle: PodRuntime | None = None  # pod inside the current cycle
        # The G2 event trace: reserve decisions AND unschedulable / scorer-error /
        # bind-failure / cache-transition events (R1-5: the trace diff needs all of it).
        self.decision_log: list[dict[str, Any]] = []
        env.process(self._run())

    def _log(self, event: str, pod: PodRuntime, **extra: Any) -> None:
        self.decision_log.append(
            {
                "t": self.env.now,
                "event": event,
                "pod_uid": pod.uid,
                "service": pod.service.id,
                "attempt": pod.scheduling_attempts,
                **extra,
            }
        )

    # -- queue ---------------------------------------------------------------
    def enqueue(self, pod: PodRuntime) -> None:
        if pod.phase != PENDING:
            raise ValueError(f"enqueue of non-pending pod {pod.uid}")
        self.queue.put(pod)

    def _requeue_with_backoff(self, pod: PodRuntime) -> None:
        delay = min(
            self.params.backoff_initial_s * (2 ** max(pod.scheduling_attempts - 1, 0)),
            self.params.backoff_max_s,
        )
        self._backoff_until[pod.uid] = self.env.now + delay
        self.env.process(self._backoff(pod, delay))

    def _backoff(self, pod: PodRuntime, delay: float) -> Generator[Any, Any, None]:
        yield self.env.timeout(delay)
        self._backoff_until.pop(pod.uid, None)
        if pod.phase == PENDING:  # may have been terminated meanwhile (HPA scale-down)
            self.queue.put(pod)

    def pending_in_queue_order(self) -> list[tuple[int, float]]:
        """Every PENDING pod as ``(uid, earliest-schedulable time)``: the in-cycle pod
        (popped from the activeQ, not yet reserved) first, then activeQ pods in FIFO
        store order (eligible now), then backed-off pods by (expiry, uid). This is the
        queue state a snapshot captures (R1-8: real queue order, not uid sort)."""
        now = self.env.now
        out: list[tuple[int, float]] = []
        queued: set[int] = set()
        if self._in_cycle is not None and self._in_cycle.phase == PENDING:
            out.append((self._in_cycle.uid, now))
            queued.add(self._in_cycle.uid)
        for pod in self.queue.items:
            if pod.phase == PENDING and pod.uid not in queued:
                out.append((pod.uid, now))
                queued.add(pod.uid)
        backoff = sorted(
            (t, uid)
            for uid, t in self._backoff_until.items()
            if uid not in queued and self.cluster.pods[uid].phase == PENDING
        )
        out.extend((uid, t) for t, uid in backoff)
        return out

    # -- the serial cycle ------------------------------------------------------
    def _run(self) -> Generator[Any, Any, None]:
        while True:
            pod = yield self.queue.get()
            if pod.phase != PENDING:
                continue  # terminated while queued
            self._in_cycle = pod  # still an unscheduled pod (snapshots must see it)
            try:
                yield from self._cycle(pod)
            finally:
                self._in_cycle = None

    def _cycle(self, pod: PodRuntime) -> Generator[Any, Any, None]:
        pod.scheduling_attempts += 1
        if self._on_cycle is not None:
            self._on_cycle(pod, self.ledger.scheduling_cycles)
        self.ledger.scheduling_cycles += 1
        if self.params.cycle_overhead_s > 0:
            yield self.env.timeout(self.params.cycle_overhead_s)
            if pod.phase != PENDING:
                return  # terminated during the framework overhead window
        # Host-side view construction happens OUTSIDE the policy-error guard: a
        # failure here is a simulator/data bug and must crash, never bias fitness
        # against the candidate (R1-12).
        ctx, uid_to_dense, _ = build_ctx(
            self.cluster,
            self.workload,
            self.cache,
            self.env.now,
            self.cutoff_s,
            include_pod=pod,
        )
        pod_dict = ctx["pods"][uid_to_dense[pod.uid]]
        nodes_by_id = {n["id"]: n for n in ctx["nodes"]}
        feasible = filter_feasible(pod_dict, ctx["nodes"])
        if not feasible:
            self.ledger.unschedulable_cycles += 1
            self._log("unschedulable", pod)
            self._requeue_with_backoff(pod)
            return
        try:
            node_id, scores = score_and_select(
                self.scorer,
                pod_dict,
                feasible,
                nodes_by_id,
                ctx,
                self.params.quantize_scores,
            )
        except PolicyError as exc:  # the policy's fault: framework error, requeue (T1)
            self.ledger.scorer_errors += 1
            self._log("scorer_error", pod, error=str(exc))
            self._requeue_with_backoff(pod)
            return
        # Reserve: assume the pod in the cache; binding is asynchronous.
        pod.phase = BINDING
        self.cache.assume(pod, node_id)
        self.ledger.reserved(pod.uid, node_id)
        self._log("reserve", pod, node=node_id, feasible=feasible, scores=scores)
        self.env.process(self._bind(pod, node_id))

    def _bind(self, pod: PodRuntime, node_id: int) -> Generator[Any, Any, None]:
        """Asynchronous binding with failure injection + truth revalidation. Latency and
        failure draws come from separate substreams keyed by the pod's stable identity
        and this binding attempt (R1-7: no shared mutable bind stream)."""
        sid, attempt = pod.service.id, pod.scheduling_attempts
        latency_rng = self._rng_for("bind-latency", sid, *pod.rng_identity, attempt)
        yield self.env.timeout(self.params.bind_latency.sample(latency_rng))
        injected = False
        if self.params.bind_failure_prob > 0.0:
            failure_rng = self._rng_for("bind-failure", sid, *pod.rng_identity, attempt)
            injected = float(failure_rng.random()) < self.params.bind_failure_prob
        node = self.cluster.nodes[node_id]
        valid = node.awake and node.schedulable
        capacity_ok = (
            node.cpu_allocated + pod.service.cpu_request <= node.spec.cpu_capacity
            and node.mem_allocated + pod.service.mem_request <= node.spec.mem_capacity
        )
        if pod.phase != BINDING:
            # Terminated during binding (HPA scale-down race): release the reservation.
            self.cache.unreserve(pod, node_id)
            self.ledger.unreserved(pod.uid, injected_failure=False)
            self._log("bind_failure", pod, node=node_id, reason="pod_terminated")
            return
        if injected or not valid or not capacity_ok:
            self.cache.unreserve(pod, node_id)
            self.ledger.unreserved(pod.uid, injected_failure=injected)
            reason = (
                "injected_conflict"
                if injected
                else (
                    "node_not_schedulable"
                    if node.awake and not node.schedulable
                    else ("node_not_awake" if not valid else "kubelet_admission_capacity")
                )
            )
            self._log("bind_failure", pod, node=node_id, reason=reason)
            pod.binding_failures += 1
            pod.phase = PENDING
            self._requeue_with_backoff(pod)
            return
        self.cluster.bind(pod, node_id, notify=False)
        self.cache.confirm(pod, node_id)
        self.ledger.bind_confirmed(pod.uid)
        self._log("bind_success", pod, node=node_id)
        self._on_bound(pod)
