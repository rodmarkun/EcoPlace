"""Actuation twins (TODO 2.3): budgeted rebalancer, HPA, node power-state controller.

- **Rebalancer**: every window, builds a ``ClusterState``/``CallGraphCtx`` view from
  truth (the controller reads current API state; no informer staleness on this path,
  documented), asks the proposer for moves under a hard per-window budget, and passes
  every proposed action through the **action gateway**, which validates the *whole
  proposal transactionally* (R1-3): shadow destination reservations across the window's
  accepted moves, duplicate-pod rejection, budget cap, PDB compliance, stateful +
  explicit exclusions, wake-before-move-to-suspended ordering, and the cumulative
  episode disruption caps (moves / bytes / unavailability-seconds) enforced **before
  execution** — bytes via their known upper bound (state transfer + image if the
  destination may need a pull), unavailability via a conservative structural pre-charge
  (wake latency + image-pull time + cold-start mean + a configured slack), trued up with
  the measured value afterwards. The drain component of unavailability is workload-
  dependent, so a single admitted move can exceed its pre-charge; the measured excess is
  reported as ``unavailability_overshoot_s`` (documented approximation — the cap is
  hard at admission, exact in the ledger). Rejected actions are counted, never silently
  dropped. Accepted victims are **locked synchronously at admission** (removed from the
  LB pool, phase TERMINATING) so no same-instant HPA/rebalancer action can select them
  twice. Moves are break-before-make: the old replica drains gracefully, then the new
  pod is bound directly to the destination (the scheduler observes it via its
  informer); per-move unavailability is the eviction-start -> new-replica-ready gap;
  migration bytes are image bytes (if the destination lacks the image) plus the
  service's declared state-transfer bytes, which are also metered in the
  network-allocated energy category.
- **HPA**: single shared config (C3): per-service busy-CPU/requested-CPU utilization over
  the sync window, kube-style tolerance band, ceil scaling, scale-down stabilization
  (max desired over the stabilization window), min/max replica bounds from the
  ``ServiceSpec``. Scale-ups create Pending pods through the scheduler and count
  in-flight replicas (PENDING/BINDING/PULLING/STARTING — never TERMINATING) against the
  desired count; scale-downs terminate the newest ready replicas first (documented
  victim rule), locking victims synchronously at selection.
- **Power controller**: optional auto-suspend of empty suspendable nodes after an idle
  timeout; wake happens via rebalancer ``wake`` actions (priced by the power model's
  published transition energy/latency) or on-demand by this controller's policy. Node
  classes without transition parameters are non-suspendable (enforced in the spec).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol

from sim.energy import MIB
from sim.runtime import (
    AWAKE,
    BINDING,
    PENDING,
    PULLING,
    READY,
    STARTING,
    SUSPENDED,
    TERMINATING,
    ClusterRuntime,
    HostError,
    Ledger,
    PodRuntime,
    PolicyError,
)
from sim.scheduler import build_ctx
from sim.spec import HpaParams, RebalancerParams, WorkloadSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import simpy

    from sim.app import ApplicationModel
    from sim.scheduler import SchedulerEmulator

MOVE = 0
WAKE = 1


class Proposer(Protocol):
    """Rebalancing proposer contract, mirroring the PolicyBundle ``propose_moves``."""

    name: str

    def propose_moves(
        self, state: dict[str, Any], graph: dict[str, Any], budget: int
    ) -> list[dict[str, int]]: ...


class PlacementScriptProposer:
    def __init__(self, policy: Any) -> None:  # placementscript.Policy
        self._policy = policy
        self.name = f"ps:{policy.name}@{policy.version}"

    def propose_moves(
        self, state: dict[str, Any], graph: dict[str, Any], budget: int
    ) -> list[dict[str, int]]:
        try:
            return [dict(m) for m in self._policy.propose_moves(state, graph, budget)]
        except Exception as exc:
            import placementscript as ps

            if isinstance(exc, ps.HostError):
                raise HostError(str(exc)) from exc
            raise


class CallableProposer:
    def __init__(self, name: str, fn: Callable[[dict, dict, int], list[dict]]) -> None:
        self.name = f"py:{name}"
        self._fn = fn

    def propose_moves(
        self, state: dict[str, Any], graph: dict[str, Any], budget: int
    ) -> list[dict[str, int]]:
        return [dict(m) for m in self._fn(state, graph, budget)]


class Rebalancer:
    """Budgeted rebalancing controller twin."""

    def __init__(
        self,
        env: simpy.Environment,
        cluster: ClusterRuntime,
        workload: WorkloadSpec,
        cutoff_s: float,
        app: ApplicationModel,
        scheduler: SchedulerEmulator,
        params: RebalancerParams,
        proposer: Proposer,
        ledger: Ledger,
        on_bound: Callable[[PodRuntime], Any],
    ) -> None:
        self.env = env
        self.cluster = cluster
        self.workload = workload
        self.cutoff_s = cutoff_s
        self.app = app
        self.scheduler = scheduler
        self.params = params
        self.proposer = proposer
        self.ledger = ledger
        self._on_bound = on_bound
        self.total_moves = 0
        self.total_bytes_mb = 0.0
        self.total_unavailability_s = 0.0
        self.unavailability_overshoot_s = 0.0  # measured excess over the pre-charge
        self._precharged_bytes_mb = 0.0  # outstanding admission reservations
        self._precharged_unavail_s = 0.0
        # Destination reservations persist across controller windows until each move
        # either binds to its admitted destination or explicitly falls back.
        self._dest_cpu_reserved: dict[int, int] = {}
        self._dest_mem_reserved: dict[int, int] = {}
        self._open_moves: dict[int, dict[str, Any]] = {}  # old pod uid -> commitment
        self.open_moves_censored = 0
        self.budget_exceeded = False
        self._horizon_finalized = False
        self.rejections: list[dict[str, Any]] = []
        if params.enabled:
            env.process(self._run())

    def _run(self) -> Generator[Any, Any, None]:
        while True:
            yield self.env.timeout(self.params.window_s)
            self._window()

    def _window(self) -> None:
        ctx, _, dense_to_uid = build_ctx(
            self.cluster,
            self.workload,
            cache=None,
            now_s=self.env.now,
            cutoff_s=self.cutoff_s,
        )
        state = {"nodes": ctx["nodes"], "pods": ctx["pods"]}
        budget = self.params.budget_per_window
        try:
            raw = self.proposer.propose_moves(state, ctx["graph"], budget)
            if not isinstance(raw, list | tuple):
                raise PolicyError(
                    f"proposer returned {type(raw).__name__}, expected a list of actions"
                )
            proposals = []
            for index, action in enumerate(raw):
                if not isinstance(action, dict):
                    raise PolicyError(f"proposal {index} is {type(action).__name__}, expected dict")
                proposals.append(dict(action))
        except HostError:
            raise
        except PolicyError as exc:
            self.ledger.rebalancer_contract_violations += 1
            self.rejections.append(
                {"t": self.env.now, "action": None, "reason": f"proposer_error: {exc}"}
            )
            return
        except Exception as exc:
            # Unknown exceptions crossing a policy adapter are candidate failures.
            self.ledger.rebalancer_contract_violations += 1
            self.rejections.append(
                {"t": self.env.now, "action": None, "reason": f"proposer_error: {exc}"}
            )
            return
        accepted = self._gateway(proposals, dense_to_uid, budget)
        wake_procs: dict[int, Any] = {}
        for action in accepted:
            if action["kind"] == WAKE:
                self.ledger.rebalancer_wakes_executed += 1
                wake_procs[action["dest"]] = self.cluster.wake(action["dest"])
        for action in accepted:
            if action["kind"] == MOVE:
                self.ledger.rebalancer_moves_executed += 1
                self.total_moves += 1
                pod = self.cluster.pods[action["pod_uid"]]
                # Lock the victim synchronously (same instant as admission): it leaves
                # the LB pool and its phase flips before any other controller can act.
                self.app.replica_leaving(pod)
                pod.phase = TERMINATING
                self._open_moves[pod.uid] = {
                    "evict_start": self.env.now,
                    "dest": action["dest"],
                    "cpu": pod.service.cpu_request,
                    "mem": pod.service.mem_request,
                    "bytes_precharge_mb": action["bytes_precharge_mb"],
                    "unavail_precharge_s": action["unavail_precharge_s"],
                    "replacement": None,
                    "state_transferred": False,
                    "bytes_settled": False,
                    "unavail_settled": False,
                    "dest_reserved": True,
                }
                self.env.process(
                    self._move(
                        pod,
                        action["dest"],
                        wake_procs.get(action["dest"]),
                    )
                )

    def _reject(self, action: dict[str, Any], reason: str) -> None:
        self.ledger.rebalancer_actions_rejected += 1
        self.rejections.append({"t": self.env.now, "action": dict(action), "reason": reason})

    def _move_precharges(self, pod: PodRuntime, dest_node: Any) -> tuple[float, float]:
        """Conservative admission pre-charges for one move: known migration bytes (image
        cache can only grow, so 'image not cached now' upper-bounds the pull) and the
        structural unavailability bound (wake + pull + cold-start mean + slack)."""
        service = pod.service
        # The image may ultimately be pulled on a fallback destination, so admission
        # reserves its full size independent of the current destination cache. This is
        # a true upper bound because one replacement pulls at most one image.
        bytes_mb = service.state_transfer_mb + service.image_mb
        pull_s = 0.0
        if service.id not in dest_node.image_cache:
            pull_s = service.image_mb / dest_node.node_class.image_pull_mib_per_s
        wake_s = (
            dest_node.node_class.wake_latency_s or 0.0 if dest_node.power_state != AWAKE else 0.0
        )
        unavail_s = (
            wake_s + pull_s + service.cold_start.mean() + self.params.unavailability_precharge_s
        )
        return bytes_mb, unavail_s

    def _gateway(
        self, proposals: list[dict[str, int]], dense_to_uid: dict[int, int], budget: int
    ) -> list[dict[str, Any]]:
        """The fixed action gateway (plan §3.5): validates the whole proposal list as a
        transaction — shadow destination reservations, duplicate rejection, and
        pre-execution cap enforcement (R1-3)."""
        if len(proposals) > budget:
            # Bundle-contract violation (spec §5.5): counted; excess actions dropped.
            self.ledger.rebalancer_contract_violations += 1
            proposals = proposals[:budget]
        accepted: list[dict[str, Any]] = []
        accepted_moves = 0
        woken: set[int] = set()
        seen_pods: set[int] = set()
        shadow_cpu: dict[int, int] = {}  # newly reserved millicores this window
        shadow_mem: dict[int, int] = {}
        window_bytes_mb = 0.0
        window_unavail_s = 0.0
        nodes = self.cluster.nodes
        excluded = set(self.params.excluded_pods)
        p = self.params
        for raw in proposals:
            if not isinstance(raw, dict):
                self._reject({"malformed": repr(raw)}, "malformed action")
                continue
            kind = raw.get("kind")
            if kind == WAKE:
                nid = raw.get("dest", -1)
                if not isinstance(nid, int) or not 0 <= nid < len(nodes):
                    self._reject(raw, "wake: unknown node")
                    continue
                node = nodes[nid]
                if not node.schedulable:
                    self._reject(raw, "wake: node unavailable by planned churn")
                    continue
                if not node.node_class.suspendable:
                    self._reject(raw, "wake: node class not suspendable")
                    continue
                if node.power_state != SUSPENDED or nid in woken:
                    self._reject(raw, "wake: node not suspended")
                    continue
                woken.add(nid)
                accepted.append({"kind": WAKE, "dest": nid})
                continue
            if kind != MOVE:
                self._reject(raw, "unknown action kind")
                continue
            dense_pod = raw.get("pod", -1)
            if not isinstance(dense_pod, int) or dense_pod not in dense_to_uid:
                self._reject(raw, "move: unknown pod")
                continue
            pod = self.cluster.pods[dense_to_uid[dense_pod]]
            if pod.uid in seen_pods:
                self._reject(raw, "move: duplicate pod in one proposal")
                continue
            if pod.phase != READY:
                self._reject(raw, "move: pod not ready")
                continue
            if pod.service.stateful:
                self._reject(raw, "move: stateful pod excluded")
                continue
            if pod.has_pdb:
                self._reject(raw, "move: PDB-protected pod excluded")
                continue
            if pod.uid in excluded:
                self._reject(raw, "move: pod on the exclusion list")
                continue
            dest = raw.get("dest", -1)
            if not isinstance(dest, int) or not 0 <= dest < len(nodes):
                self._reject(raw, "move: unknown destination")
                continue
            if dest == pod.node_id:
                self._reject(raw, "move: destination equals current node")
                continue
            dest_node = nodes[dest]
            if not dest_node.schedulable:
                self._reject(raw, "move: destination unavailable by planned churn")
                continue
            if dest_node.power_state != AWAKE and dest not in woken:
                self._reject(raw, "move: destination suspended without prior wake")
                continue
            # Shadow reservation: this window's already-accepted moves count against
            # the destination's remaining capacity (transactional validation).
            res_cpu = self._dest_cpu_reserved.get(dest, 0) + shadow_cpu.get(dest, 0)
            res_mem = self._dest_mem_reserved.get(dest, 0) + shadow_mem.get(dest, 0)
            if (
                dest_node.cpu_allocated + res_cpu + pod.service.cpu_request
                > dest_node.spec.cpu_capacity
                or dest_node.mem_allocated + res_mem + pod.service.mem_request
                > dest_node.spec.mem_capacity
            ):
                self._reject(raw, "move: destination capacity would be exceeded")
                continue
            # Cumulative disruption caps, enforced BEFORE execution (hard constraints,
            # plan §5): moves exactly; bytes by known upper bound; unavailability by the
            # conservative structural pre-charge.
            bytes_pre, unavail_pre = self._move_precharges(pod, dest_node)
            if p.max_total_moves is not None and (
                self.total_moves + accepted_moves + 1 > p.max_total_moves
            ):
                self._reject(raw, "move: cumulative move cap reached")
                continue
            if p.max_total_bytes_mb is not None and (
                self.total_bytes_mb + self._precharged_bytes_mb + window_bytes_mb + bytes_pre
                > p.max_total_bytes_mb
            ):
                self._reject(raw, "move: cumulative bytes cap would be exceeded")
                continue
            if p.max_total_unavailability_s is not None and (
                self.total_unavailability_s
                + self._precharged_unavail_s
                + window_unavail_s
                + unavail_pre
                > p.max_total_unavailability_s
            ):
                self._reject(raw, "move: cumulative unavailability cap would be exceeded")
                continue
            seen_pods.add(pod.uid)
            shadow_cpu[dest] = shadow_cpu.get(dest, 0) + pod.service.cpu_request
            shadow_mem[dest] = shadow_mem.get(dest, 0) + pod.service.mem_request
            window_bytes_mb += bytes_pre
            window_unavail_s += unavail_pre
            accepted.append(
                {
                    "kind": MOVE,
                    "pod_uid": pod.uid,
                    "dest": dest,
                    "bytes_precharge_mb": bytes_pre,
                    "unavail_precharge_s": unavail_pre,
                }
            )
            accepted_moves += 1
        self._precharged_bytes_mb += window_bytes_mb
        self._precharged_unavail_s += window_unavail_s
        for dest, cpu in shadow_cpu.items():
            self._dest_cpu_reserved[dest] = self._dest_cpu_reserved.get(dest, 0) + cpu
        for dest, mem in shadow_mem.items():
            self._dest_mem_reserved[dest] = self._dest_mem_reserved.get(dest, 0) + mem
        return accepted

    def _move(
        self,
        pod: PodRuntime,
        dest: int,
        wake_proc: Any | None,
    ) -> Generator[Any, Any, None]:
        """Break-before-make migration with graceful drain. The victim was locked
        synchronously at admission (LB pool exit + TERMINATING)."""
        move = self._open_moves[pod.uid]
        service = pod.service
        if pod.outstanding > 0:
            pod.drained = self.env.event()
            yield pod.drained
        self.cluster.unbind(pod, reason="rebalanced")
        move["state_transferred"] = True
        if service.state_transfer_mb > 0:
            # Migration state is copied on every replacement path, including fallback.
            self.cluster.meter.add_bytes(service.state_transfer_mb * MIB)
        if wake_proc is not None:
            yield wake_proc  # the prior wake action must complete before we bind
        dest_node = self.cluster.nodes[dest]
        new_pod = self.cluster.create_pod(
            service,
            origin="rebalancer",
            rng_identity=("rebalancer", *pod.rng_identity, "move"),
        )
        move["replacement"] = new_pod
        if (
            not dest_node.schedulable
            or dest_node.power_state != AWAKE
            or (
                dest_node.cpu_allocated + service.cpu_request > dest_node.spec.cpu_capacity
                or dest_node.mem_allocated + service.mem_request > dest_node.spec.mem_capacity
            )
        ):
            # Destination changed while draining: the replacement goes through the
            # scheduler instead (never lost). Pre-charges are released; unavailability
            # is measured by watching the replacement to readiness.
            self._release_destination(pod.uid)
            self.scheduler.enqueue(new_pod)
            self.env.process(self._watch_fallback(pod.uid, new_pod))
            return
        image_mb = service.image_mb if service.id not in dest_node.image_cache else 0.0
        self._release_destination(pod.uid)
        self.cluster.bind(new_pod, dest, notify=True)
        self._settle_bytes(pod.uid, service.state_transfer_mb + image_mb)
        startup = self._on_bound(new_pod)
        # Unavailability contribution: eviction start -> replacement ready.
        yield startup
        self._settle_unavailability(pod.uid, new_pod)

    def _watch_fallback(self, move_uid: int, pod: PodRuntime) -> Generator[Any, Any, None]:
        """Settle at the replacement's exact READY/TERMINATED lifecycle event."""
        yield pod.ready_event | pod.terminated_event
        move = self._open_moves[move_uid]
        actual_bytes = (
            pod.service.state_transfer_mb if move["state_transferred"] else 0.0
        ) + pod.startup_image_mb
        self._settle_bytes(move_uid, actual_bytes)
        self._settle_unavailability(move_uid, pod)

    def _release_destination(self, move_uid: int) -> None:
        move = self._open_moves[move_uid]
        if not move["dest_reserved"]:
            return
        dest = move["dest"]
        self._dest_cpu_reserved[dest] -= move["cpu"]
        self._dest_mem_reserved[dest] -= move["mem"]
        if self._dest_cpu_reserved[dest] == 0:
            del self._dest_cpu_reserved[dest]
        if self._dest_mem_reserved[dest] == 0:
            del self._dest_mem_reserved[dest]
        move["dest_reserved"] = False

    def _settle_bytes(self, move_uid: int, actual_mb: float) -> None:
        move = self._open_moves[move_uid]
        if move["bytes_settled"]:
            raise HostError(f"move {move_uid} bytes settled twice")
        self._precharged_bytes_mb -= move["bytes_precharge_mb"]
        self.total_bytes_mb += actual_mb
        move["bytes_settled"] = True
        if (
            self.params.max_total_bytes_mb is not None
            and self.total_bytes_mb > self.params.max_total_bytes_mb + 1e-12
        ):
            raise HostError("actual migration bytes exceeded their admission upper bound")

    def _settle_unavailability(self, move_uid: int, pod: PodRuntime) -> None:
        move = self._open_moves[move_uid]
        if move["unavail_settled"]:
            raise HostError(f"move {move_uid} unavailability settled twice")
        precharge_s = move["unavail_precharge_s"]
        self._precharged_unavail_s -= precharge_s
        terminal_s = pod.ready_s if pod.ready_s is not None else pod.terminated_s
        if terminal_s is None:
            raise HostError(f"move {move_uid} settled without a lifecycle timestamp")
        measured = terminal_s - move["evict_start"]
        self.total_unavailability_s += measured
        if measured > precharge_s:
            self.unavailability_overshoot_s += measured - precharge_s
        move["unavail_settled"] = True
        self._refresh_budget_status()

    def _refresh_budget_status(self) -> None:
        cap = self.params.max_total_unavailability_s
        if cap is not None and self.total_unavailability_s > cap + 1e-12:
            # Drain time has no finite structural upper bound. An admitted move that
            # exceeds its pre-charge is therefore an explicit infeasible disruption
            # outcome, never silently represented as cap-compliant.
            self.budget_exceeded = True

    def finalize(self, horizon_s: float) -> None:
        """Right-censor every open move at the episode horizon and close reservations."""
        if self._horizon_finalized:
            raise HostError("rebalancer finalized twice")
        self._horizon_finalized = True
        for move_uid, move in self._open_moves.items():
            if not move["bytes_settled"]:
                pod = move["replacement"]
                actual = (
                    self.cluster.pods[move_uid].service.state_transfer_mb
                    if move["state_transferred"]
                    else 0.0
                )
                if pod is not None:
                    actual += pod.startup_image_mb
                self._settle_bytes(move_uid, actual)
            if not move["unavail_settled"]:
                measured = max(0.0, horizon_s - move["evict_start"])
                self._precharged_unavail_s -= move["unavail_precharge_s"]
                self.total_unavailability_s += measured
                if measured > move["unavail_precharge_s"]:
                    self.unavailability_overshoot_s += measured - move["unavail_precharge_s"]
                move["unavail_settled"] = True
                self.open_moves_censored += 1
            self._release_destination(move_uid)
        self._refresh_budget_status()
        if abs(self._precharged_bytes_mb) > 1e-9 or abs(self._precharged_unavail_s) > 1e-9:
            raise HostError("rebalancer disruption precharges did not close at horizon")
        if self._dest_cpu_reserved or self._dest_mem_reserved:
            raise HostError("rebalancer destination reservations did not close at horizon")


class Hpa:
    """Horizontal pod autoscaler twin — one config for every method (C3)."""

    def __init__(
        self,
        env: simpy.Environment,
        cluster: ClusterRuntime,
        workload: WorkloadSpec,
        app: ApplicationModel,
        scheduler: SchedulerEmulator,
        params: HpaParams,
        ledger: Ledger,
    ) -> None:
        self.env = env
        self.cluster = cluster
        self.workload = workload
        self.app = app
        self.scheduler = scheduler
        self.params = params
        self.ledger = ledger
        self._last_busy = [0.0] * len(workload.services)
        self._desired_history: list[list[tuple[float, int]]] = [[] for _ in workload.services]
        self._scale_up_generation = [0] * len(workload.services)
        self.actions: list[dict[str, Any]] = []
        if params.enabled:
            env.process(self._run())

    def _run(self) -> Generator[Any, Any, None]:
        while True:
            yield self.env.timeout(self.params.sync_period_s)
            for service in self.workload.services:
                if service.activation_s <= self.env.now:
                    self._sync_service(service.id)

    def _sync_service(self, sid: int) -> None:
        p = self.params
        service_rt = self.app.services[sid]
        spec = self.workload.services[sid]
        ready = [r.pod for r in service_rt.replicas.values()]
        current = len(ready)
        busy = service_rt.busy_millicore_seconds
        window_busy = busy - self._last_busy[sid]
        self._last_busy[sid] = busy
        if current == 0:
            return  # nothing ready to measure; scheduler/startup will catch up
        requested = current * spec.cpu_request
        utilization = window_busy / (p.sync_period_s * requested) if requested else 0.0
        ratio = utilization / p.target_utilization
        if abs(ratio - 1.0) <= p.tolerance:
            desired = current
        else:
            desired = math.ceil(current * ratio)
        desired = max(spec.min_replicas, min(spec.max_replicas, desired))
        history = self._desired_history[sid]
        history.append((self.env.now, desired))
        cutoff = self.env.now - p.scale_down_stabilization_s
        history[:] = [(t, d) for t, d in history if t >= cutoff]
        if desired < current:
            desired = max(d for _, d in history)  # scale-down stabilization
        # ``max_replicas`` is a hard cap on *all live pod objects*, including draining
        # TERMINATING replicas. This is deliberately stricter than treating max as only
        # the desired serving count: a slow drain cannot be hidden by creating a new
        # replica above the advertised bound.
        in_flight_phases = (PENDING, BINDING, PULLING, STARTING)
        in_flight = sum(
            1
            for pod in self.cluster.pods.values()
            if pod.live and pod.service.id == sid and pod.phase in in_flight_phases
        )
        terminating = sum(
            1
            for pod in self.cluster.pods.values()
            if pod.live and pod.service.id == sid and pod.phase == TERMINATING
        )
        live = current + in_flight + terminating
        if desired > live:
            n = min(desired - live, spec.max_replicas - live)
            generation = self._scale_up_generation[sid]
            self._scale_up_generation[sid] += 1
            for index in range(n):
                pod = self.cluster.create_pod(
                    spec,
                    origin="hpa",
                    rng_identity=("hpa", generation, index),
                )
                self.scheduler.enqueue(pod)
            self.ledger.hpa_scale_ups += 1
            self.actions.append({"t": self.env.now, "service": sid, "action": "up", "n": n})
        elif desired < current:
            victims = sorted(ready, key=lambda pod: -pod.uid)[: current - desired]
            for pod in victims:
                # Lock the victim synchronously (same instant as selection): no other
                # controller acting at this timestamp can select it too.
                self.app.replica_leaving(pod)
                pod.phase = TERMINATING
                self.env.process(self._terminate(pod))
            self.ledger.hpa_scale_downs += 1
            self.actions.append(
                {"t": self.env.now, "service": sid, "action": "down", "n": len(victims)}
            )

    def _terminate(self, pod: PodRuntime) -> Generator[Any, Any, None]:
        """Drain + unbind a victim already locked (LB pool exit, TERMINATING)."""
        if pod.outstanding > 0:
            pod.drained = self.env.event()
            yield pod.drained
        self.cluster.unbind(pod, reason="hpa_scale_down")


class PowerController:
    """Optional auto-suspend of empty suspendable nodes after an idle timeout."""

    def __init__(
        self,
        env: simpy.Environment,
        cluster: ClusterRuntime,
        suspend_after_idle_s: float | None,
    ) -> None:
        self.env = env
        self.cluster = cluster
        self.threshold = suspend_after_idle_s
        if suspend_after_idle_s is not None:
            env.process(self._run(suspend_after_idle_s))

    def _run(self, threshold: float) -> Generator[Any, Any, None]:
        period = max(threshold / 4.0, 1e-3)
        while True:
            yield self.env.timeout(period)
            for node in self.cluster.nodes:
                if (
                    node.node_class.suspendable
                    and node.schedulable
                    and node.power_state == AWAKE
                    and not node.pods
                    and self.env.now - node.last_emptied_s >= threshold
                ):
                    self.cluster.suspend(node.id)
