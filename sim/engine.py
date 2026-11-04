"""Episode engine (plan §3.4): wires the C11 application model, the scheduling-cycle
emulator, the actuation twins, and the energy meter into one terminating episode.

Determinism contract (S6): for a fixed ``(EpisodeConfig, seed, scorer, proposer)`` the
episode is bitwise-reproducible — ``EpisodeResult.digest()`` is identical across runs
and processes **within the pinned environment** (the locked interpreter version,
dependency set — NumPy distribution algorithms included — and CPU architecture recorded
by ``uv.lock``/CI). Cross-platform bitwise identity is *not* claimed: NumPy/libm code
paths may differ across architectures, and Python-callable scorers/proposers are only as
deterministic as their own code. All randomness flows through per-component streams
derived via ``analysis.tracking`` (``derive_seed``/``spawn_rng``); the seed argument is
**required** (docs/SEEDS.md: omission is a TypeError, never a silent entropy default).

Simultaneous-event tie rule (documented, tested): events scheduled for the same
simulation time execute in *scheduling order* (SimPy orders its heap by
``(time, priority, insertion_count)``; every event this simulator creates uses the
default priority except the snapshot/advance boundary below). Where cross-component
ordering at equal timestamps matters semantically (energy accrual before a power-state
change; unreserve before requeue; victim locking at controller decisions), it is
enforced *within* a single process step, never left to heap order.

Snapshot/advance boundary semantics (documented, tested): "state at time t" always
means *after every event scheduled strictly before t and before any event scheduled at
t*. Timed snapshots fire through URGENT-priority events, and ``advance(t)`` stops at the
same boundary, so a timed snapshot from a full run and a replay stopped at ``t`` observe
identical state — including at controller sync instants.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import simpy

from analysis.tracking import spawn_rng
from sim.actuation import Hpa, PowerController, Proposer, Rebalancer
from sim.app import ApplicationModel
from sim.energy import EnergyMeter, EnergyReport
from sim.runtime import (
    PULLING,
    READY,
    STARTING,
    TERMINATED,
    ClusterRuntime,
    Ledger,
    PodEvent,
    PodRuntime,
    RequestRecord,
    ServiceCallRecord,
    SimulationError,
    pod_event,
)
from sim.scheduler import SchedulerEmulator, Scorer
from sim.snapshot import Snapshot, take_snapshot
from sim.spec import EpisodeConfig, config_sha256

if TYPE_CHECKING:
    from collections.abc import Generator


_EPISODE_RESULT_AUTHORITY = object()


@dataclass
class EpisodeResult:
    """Everything an episode produces. Canonically serializable; the digest is the
    deterministic-replay witness (same seed -> identical digest).

    ``records`` holds **every offered request** (plan §5), including requests still in
    flight at the horizon (status ``"in_flight"``): no constraint denominator ever
    shrinks because work did not finish."""

    config_sha256: str
    seed: int
    scorer: str
    proposer: str | None
    horizon_s: float
    records: list[RequestRecord]
    service_calls: list[ServiceCallRecord]
    pods: list[PodEvent]
    ledger: dict[str, int]
    energy: EnergyReport
    decision_log: list[dict[str, Any]]
    hpa_actions: list[dict[str, Any]]
    rebalancer: dict[str, Any]
    snapshots: list[Snapshot] = field(default_factory=list)
    churn_events: list[dict[str, Any]] = field(default_factory=list)
    _runtime_seal: str | None = field(default=None, repr=False, compare=False)
    _authority: object | None = field(default=None, repr=False, compare=False)

    def verify_runtime_origin(self) -> None:
        """Reject constructed or post-run-mutated objects at evidence boundaries."""
        if self._authority is not _EPISODE_RESULT_AUTHORITY or self._runtime_seal is None:
            raise ValueError("EpisodeResult was not issued by Simulation.run()")
        if self.digest() != self._runtime_seal:
            raise ValueError("EpisodeResult changed after Simulation.run() sealed it")

    # -- warm-up-bias analysis support (S6): windowed statistics ---------------
    def latency_stats(
        self, window_start_s: float = 0.0, window_end_s: float | None = None
    ) -> dict[str, float]:
        """Latency statistics over completed requests that *arrived* inside the window
        (supports warm-up truncation; episodes are terminating by design). Note the
        completed-only conditioning: use ``slo_satisfaction``/``offered_outcomes`` for
        constraint metrics — they never condition on completion."""
        end = self.horizon_s if window_end_s is None else window_end_s
        lat = sorted(
            r.latency_s
            for r in self.records
            if r.status == "completed" and window_start_s <= r.arrival_s < end
        )
        if not lat:
            return {"count": 0.0}
        return {
            "count": float(len(lat)),
            "mean": statistics.fmean(lat),
            "p50": lat[len(lat) // 2],
            "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
            "max": lat[-1],
        }

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def offered_outcomes(
        self, template_id: int | None = None, window_start_s: float = 0.0
    ) -> dict[str, int]:
        """Outcome counts over the full offered cohort (incl. ``in_flight``)."""
        out: dict[str, int] = {}
        for r in self.records:
            if template_id is not None and r.template != template_id:
                continue
            if r.arrival_s < window_start_s:
                continue
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def slo_satisfaction(
        self, template_id: int, slo_latency_s: float, window_start_s: float = 0.0
    ) -> float:
        """Fraction of **offered** requests (plan §5) of this template, arriving after
        the window start, that completed within the SLO. The denominator is the offered
        cohort: drops, timeouts, and requests still in flight at the horizon all count
        against satisfaction."""
        offered = [
            r for r in self.records if r.template == template_id and r.arrival_s >= window_start_s
        ]
        if not offered:
            return 1.0
        ok = sum(1 for r in offered if r.status == "completed" and r.latency_s <= slo_latency_s)
        return ok / len(offered)

    def goodput(self, template_id: int | None = None, window_start_s: float = 0.0) -> float:
        """Completed / offered over the cohort (plan §5 goodput constraint input)."""
        outcomes = self.offered_outcomes(template_id, window_start_s)
        offered = sum(outcomes.values())
        if offered == 0:
            return 1.0
        return outcomes.get("completed", 0) / offered

    def to_canonical_dict(self) -> dict[str, Any]:
        # R4-2: episode serialization is a sanctioned report consumer. A field-edited
        # report must not bypass verify_covers by flowing directly into an archive.
        self.energy.verify_seal()
        out = {
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "scorer": self.scorer,
            "proposer": self.proposer,
            "horizon_s": self.horizon_s,
            "records": [[r.rid, r.template, r.arrival_s, r.end_s, r.status] for r in self.records],
            "service_calls": [
                [
                    call.template,
                    call.request_index,
                    [list(edge_attempt) for edge_attempt in call.path],
                    call.service,
                    call.pod_uid,
                    call.start_s,
                    call.end_s,
                    call.status,
                ]
                for call in self.service_calls
            ],
            "pods": [
                [
                    p.uid,
                    p.service,
                    p.origin,
                    p.node_id,
                    p.created_s,
                    p.bound_s,
                    p.ready_s,
                    p.terminated_s,
                    p.phase,
                    p.termination_reason,
                    p.binding_failures,
                    p.scheduling_attempts,
                ]
                for p in self.pods
            ],
            "ledger": self.ledger,
            "energy": {
                "device_j": self.energy.device_j,
                "cloud_j": self.energy.cloud_j,
                "network_j": self.energy.network_j,
                "device_j_per_node": list(self.energy.device_j_per_node),
                "transition_j_per_node": list(self.energy.transition_j_per_node),
                "cloud_vcpu_hours": self.energy.cloud_vcpu_hours,
                "network_bytes": self.energy.network_bytes,
                # R2-1: the episode exposes its metering identity — per-node
                # transition counts, cluster digest, horizon — so held-out/fitness
                # consumers feed BoundaryCertificate.verify_covers from the result.
                "transition_counts": list(self.energy.transition_counts),
                "cluster_sha256": self.energy.cluster_sha256,
                "coefficients_sha256": self.energy.coefficients.sha256(),
                # §4.3: every figure carries its provenance record, serialized in full.
                "provenance": {b: rec.to_dict() for b, rec in self.energy.all_provenance().items()},
            },
            "decision_log": self.decision_log,
            "hpa_actions": self.hpa_actions,
            "rebalancer": self.rebalancer,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }
        if self.churn_events:
            out["churn_events"] = self.churn_events
        return out

    def digest(self) -> str:
        return self.canonical_digest()

    def canonical_digest(self, canonical: dict[str, Any] | None = None) -> str:
        """Digest of the canonical report.

        Pass an already-built ``to_canonical_dict()`` result to avoid building a second
        copy: on a study-B episode the canonical dict carries ~38k request records and
        ~155k service calls, so building it twice per instance doubles a multi-hundred-MB
        transient. The digest is byte-identical either way -- it is the SHA-256 of the
        same sorted JSON.
        """
        if canonical is None:
            canonical = self.to_canonical_dict()
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


class Simulation:
    """One terminating episode. Construct, then ``run()`` exactly once."""

    def __init__(
        self,
        config: EpisodeConfig,
        seed: int,
        scorer: Scorer,
        proposer: Proposer | None = None,
        lb_policy: str = "round_robin",
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an explicit int (docs/SEEDS.md)")
        if config.rebalancer.enabled and proposer is None:
            raise ValueError("rebalancer enabled but no proposer given")
        self.config = config
        self.seed = seed
        self.scorer = scorer
        self.proposer = proposer
        self.env = simpy.Environment()
        self.ledger = Ledger()
        self.meter = EnergyMeter(config.cluster, config.energy)
        self.cluster = ClusterRuntime(self.env, config.cluster, self.meter, self.ledger)
        self.app = ApplicationModel(
            self.env,
            self.cluster,
            config.workload,
            self.ledger,
            rng_for=self._rng,
            lb_policy=lb_policy,
        )
        self.scheduler = SchedulerEmulator(
            self.env,
            self.cluster,
            config.workload,
            config.cutoff_s,
            config.scheduler,
            scorer,
            self.ledger,
            rng_for=self._rng,
            on_bound=self._start_pod,
            on_cycle=self._on_cycle if config.snapshot_cycles else None,
        )
        self.hpa = Hpa(
            self.env,
            self.cluster,
            config.workload,
            self.app,
            self.scheduler,
            config.hpa,
            self.ledger,
        )
        self.rebalancer = (
            Rebalancer(
                self.env,
                self.cluster,
                config.workload,
                config.cutoff_s,
                self.app,
                self.scheduler,
                config.rebalancer,
                proposer,
                self.ledger,
                on_bound=self._start_pod,
            )
            if proposer is not None
            else None
        )
        self.power = PowerController(self.env, self.cluster, config.power.suspend_after_idle_s)
        self.snapshots: list[Snapshot] = []
        self.churn_events: list[dict[str, Any]] = []
        self._ran = False

    def _rng(self, *path: str | int) -> Any:
        return spawn_rng(self.seed, "sim", *path)

    # -- pod lifecycle ---------------------------------------------------------
    def _start_pod(self, pod: PodRuntime) -> Any:
        """Image pull + cold start, then the replica joins the LB pool (TODO 2.1).
        Returns the startup process (the rebalancer measures unavailability on it)."""
        return self.env.process(self._startup(pod))

    def _startup(self, pod: PodRuntime) -> Generator[Any, Any, None]:
        assert pod.node_id is not None
        node = self.cluster.nodes[pod.node_id]
        service = pod.service
        pod.phase = PULLING
        if service.id not in node.image_cache:
            pod.startup_image_mb = float(service.image_mb)
            pull_s = service.image_mb / node.node_class.image_pull_mib_per_s
            if self.config.energy.include_image_pull_bytes:
                self.cluster.account_image_pull(node.id, service.image_mb)
            yield self.env.timeout(pull_s)
            node.image_cache.add(service.id)
        if pod.phase == TERMINATED:
            return
        pod.phase = STARTING
        # Cold-start draw keyed by the pod's semantic origin/action identity — never
        # by the cross-origin creation order (CRN, R2-2).
        cold_rng = self._rng("cold-start", service.id, *pod.rng_identity)
        cold = service.cold_start.sample(cold_rng)
        if cold > 0:
            yield self.env.timeout(cold)
        if pod.phase == TERMINATED:
            return
        pod.phase = READY
        pod.ready_s = self.env.now
        self.app.replica_ready(pod)
        if not pod.ready_event.triggered:
            pod.ready_event.succeed()

    # -- initial placement -------------------------------------------------------
    def _create_initial_pods(self) -> None:
        for service in self.config.workload.services:
            if service.activation_s == 0.0:
                self._activate_service(service)

    def _activate_service(self, service: Any) -> None:
        for replica_index in range(service.initial_replicas):
            origin = "initial" if service.activation_s == 0.0 else "onboarding"
            identity = (
                ("initial", replica_index)
                if origin == "initial"
                else ("onboarding", service.app, service.id, replica_index)
            )
            pod = self.cluster.create_pod(
                service,
                origin=origin,
                rng_identity=identity,
            )
            self.scheduler.enqueue(pod)

    def _activation_timer(self, service: Any) -> Generator[Any, Any, None]:
        yield self.env.timeout(service.activation_s - self.env.now)
        self._activate_service(service)

    def _snapshot_timer(self, at_s: float) -> Generator[Any, Any, None]:
        """Timed snapshot at the documented boundary: an URGENT event at ``at_s`` fires
        before any normal event scheduled at ``at_s``, so the captured state is exactly
        the ``advance(at_s)`` replay state (before-event boundary semantics)."""
        ev = simpy.Event(self.env)
        ev._ok = True
        ev._value = None
        self.env.schedule(ev, priority=simpy.events.URGENT, delay=at_s - self.env.now)
        yield ev
        self.snapshots.append(
            take_snapshot(self.cluster, self.config, self.env.now, scheduler=self.scheduler)
        )

    def _churn_timer(self, event: Any) -> Generator[Any, Any, None]:
        """Apply planned scheduling-capacity churn at a normal-priority event."""
        yield self.env.timeout(event.at_s - self.env.now)
        self.cluster.set_schedulable(event.node_id, event.schedulable)
        self.churn_events.append(
            {
                "t": self.env.now,
                "node": event.node_id,
                "schedulable": event.schedulable,
                "reason": event.reason,
            }
        )

    def _on_cycle(self, pod: PodRuntime, cycle_index: int) -> None:
        """Decision-point snapshot: the cycle's pod (and everything queued behind it)
        is still pending, so L0 replay has real placement decisions to make."""
        del pod
        if cycle_index in self.config.snapshot_cycles:
            self.snapshots.append(
                take_snapshot(self.cluster, self.config, self.env.now, scheduler=self.scheduler)
            )

    # -- run -----------------------------------------------------------------
    def start(self) -> None:
        """Create the initial pods and start arrival/snapshot processes (once)."""
        if self._ran:
            raise RuntimeError("a Simulation instance runs exactly once; build a new one")
        self._ran = True
        self._create_initial_pods()
        for service in self.config.workload.services:
            if service.activation_s > 0.0:
                self.env.process(self._activation_timer(service))
        # Schedule drift before ordinary arrivals so equal-time normal events see the
        # new capacity state. Timed snapshots remain URGENT and observe the pre-event
        # boundary, matching ``advance(t)``.
        for event in self.config.node_churn:
            self.env.process(self._churn_timer(event))
        for template in self.config.workload.templates:
            self.env.process(self.app.arrivals(template, self.config.cutoff_s))
        for t in self.config.snapshot_times_s:
            self.env.process(self._snapshot_timer(t))

    def advance(self, until_s: float) -> None:
        """Advance simulation time to the documented boundary at ``until_s``: every
        event strictly before ``until_s`` has executed; no event at ``until_s`` has
        (SimPy's ``run(until=t)`` stop event is URGENT — same boundary as the timed
        snapshots). With ``start()`` this is the snapshot/'restore' API (TODO 2.6): full
        state at time t is *reconstructed by deterministic replay* —
        ``Simulation(config, seed, ...).start(); advance(t)`` yields bit-identical state
        (in-flight coroutines included) within the pinned environment, which a
        serialized byte-blob checkpoint of Python generators could not honestly provide.
        Frozen ``Snapshot`` objects are the declarative export of the
        placement-relevant state for L0."""
        self.env.run(until=until_s)

    def snapshot_now(self) -> Snapshot:
        return take_snapshot(self.cluster, self.config, self.env.now, scheduler=self.scheduler)

    def run(self) -> EpisodeResult:
        config = self.config
        self.start()
        self.advance(config.horizon_s)
        if self.rebalancer is not None:
            self.rebalancer.finalize(config.horizon_s)
        # Close the books: independent live-state close + conservation verification +
        # scheduler-cache/truth reconciliation (R1-2, R1-6).
        report = self.meter.finish(config.horizon_s)
        self.ledger.close()
        self.ledger.verify(requests=self.app.records, pods=self.cluster.pods.values())
        live_pods = sum(1 for p in self.cluster.pods.values() if p.live)
        if live_pods != self.ledger.pods_live_at_end:
            raise SimulationError(
                f"truth live-pod count {live_pods} != ledger {self.ledger.pods_live_at_end}"
            )
        self.scheduler.cache.verify_against_truth(self.cluster)
        if self.ledger.reservations() != {
            uid: rec[0] for uid, rec in self.scheduler.cache.assumed.items()
        }:
            raise SimulationError("ledger reservations do not match scheduler assumed pods")
        for node in self.cluster.nodes:
            if (
                node.cpu_allocated > node.spec.cpu_capacity
                or node.mem_allocated > node.spec.mem_capacity
            ):
                raise SimulationError(f"truth capacity exceeded on node {node.id}")
        result = EpisodeResult(
            config_sha256=config_sha256(config),
            seed=self.seed,
            scorer=self.scorer.name,
            proposer=self.proposer.name if self.proposer is not None else None,
            horizon_s=config.horizon_s,
            # Detach retained truth from live SimPy generators.  Generator finalizers
            # may run when the Simulation is garbage-collected after the horizon; they
            # must not mutate the already sealed EpisodeResult.
            records=[replace(record) for record in self.app.records],
            service_calls=[replace(call) for call in self.app.service_calls],
            pods=[pod_event(p) for p in self.cluster.pods.values()],
            ledger=self.ledger.as_dict(),
            energy=report,
            decision_log=self.scheduler.decision_log,
            hpa_actions=self.hpa.actions,
            rebalancer={
                "moves": self.rebalancer.total_moves if self.rebalancer else 0,
                "bytes_mb": self.rebalancer.total_bytes_mb if self.rebalancer else 0.0,
                "unavailability_s": (
                    self.rebalancer.total_unavailability_s if self.rebalancer else 0.0
                ),
                "unavailability_overshoot_s": (
                    self.rebalancer.unavailability_overshoot_s if self.rebalancer else 0.0
                ),
                "open_moves_censored": (
                    self.rebalancer.open_moves_censored if self.rebalancer else 0
                ),
                "budget_exceeded": (self.rebalancer.budget_exceeded if self.rebalancer else False),
                "rejections": self.rebalancer.rejections if self.rebalancer else [],
            },
            snapshots=self.snapshots,
            churn_events=self.churn_events,
        )
        result._authority = _EPISODE_RESULT_AUTHORITY
        result._runtime_seal = result.digest()
        return result


def run_episode(
    config: EpisodeConfig,
    seed: int,
    scorer: Scorer,
    proposer: Proposer | None = None,
    lb_policy: str = "round_robin",
) -> EpisodeResult:
    """Convenience one-shot runner (explicit seed, no default — docs/SEEDS.md)."""
    return Simulation(config, seed, scorer, proposer, lb_policy).run()
