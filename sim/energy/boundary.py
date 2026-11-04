"""Closed modeled energy boundary (TODO 2.4; plan §3.4/C16/S5, §4.1/§4.3).

Three accounting categories that are **never silently summed**:

1. ``device_j``  — simulated device energy for fog/far-edge node classes, integrated from
   the class power curve over utilization, **including idle, suspended, and transition
   energy** for every node in the cluster (awake-idle nodes burn idle power; suspended
   nodes burn ``suspended_power_w``; suspend/wake transitions add their published pulse).
2. ``cloud_j``   — *allocated* cloud energy: bound-pod vCPU-hours on tier-0 nodes times a
   per-vCPU-hour coefficient (a point-estimate allocation model, per CCF/Teads usage).
3. ``network_j`` — *allocated* network energy: bytes transferred times a per-byte
   coefficient.

``EnergyReport`` exposes the three categories separately and has **no implicit total**;
the only way to combine them is ``closed_boundary_j``, which takes explicit weights and
returns the value together with a ``ProvenanceRecord`` (§4.3). The idle-floor normalizer
and the max-power ceiling (plan §4.1 fitness / §6.3 penalty, V1/W2) are first-class
functions here: ``idle_floor_j`` and ``ceiling_j``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass

from sim.spec import TIER_CLOUD, ClusterSpec, EnergyCoefficients, EnergySourceRecord

KB = 1000.0
MIB = float(2**20)

_CERT_SEAL_TAG = "boundary-certificate-seal-v1:"
_VERIFIED_SEAL_TAG = "verified-instance-energy-seal-v1:"
_REPORT_SEAL_TAG = "energy-report-seal-v1:"


def _finite_nonneg(name: str, v: float) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return float(v)


def _seal(tag: str, payload: dict) -> str:
    return hashlib.sha256(
        (tag + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    """Attached to every reported energy figure (plan §4.3): physical boundary,
    included/excluded components, source dataset + version, allocation-vs-simulation
    status, parameter ranges swept, the bundle set the figure survives, and the
    structured pinned-source references behind the coefficients (R2-2: id, URL/DOI,
    access date, retained-artifact content hash — machine-readable, not prose)."""

    boundary: str  # e.g. "device", "cloud-allocated", "network-allocated", "closed-composite"
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    source_status: str  # "modeled" | "allocated-point-estimate" | "modeled-composite"
    source: str  # dataset + version/DOI (Phase 3 pins; dev placeholder until then)
    parameter_ranges: str  # swept ranges, or the point-estimate statement
    bundle_set: str  # bundle set the figure survives ("single-config" pre-Phase-3)
    coefficients_version: str
    coefficients_sha256: str
    notes: str = ""
    source_records: tuple[EnergySourceRecord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class BoundaryWeights:
    """Explicit closed-boundary combination weights (pre-registered in P2; defaults are
    the identity combination used by development fixtures)."""

    device: float = 1.0
    cloud: float = 1.0
    network: float = 1.0

    def __post_init__(self) -> None:
        # Negative/NaN weights would destroy the ordering and floor/ceiling dominance.
        _finite_nonneg("weights.device", self.device)
        _finite_nonneg("weights.cloud", self.cloud)
        _finite_nonneg("weights.network", self.network)


def _records_of_kind(
    c: EnergyCoefficients, kinds: tuple[str, ...]
) -> tuple[EnergySourceRecord, ...]:
    return tuple(r for r in c.source_records if r.kind in kinds)


@dataclass(frozen=True)
class EnergyReport:
    """The three never-silently-summed categories. Access them individually, or combine
    explicitly through ``closed_boundary_j`` (which brands the result a composite).

    The report carries its metering identity (R2-1): the digest of the cluster it was
    metered on, the horizon, and the per-node suspend/wake transition counts — so
    ``BoundaryCertificate.verify_covers`` can check the *same-boundary* preconditions
    instead of trusting the caller. Only :meth:`EnergyMeter.finish` produces a valid
    ``seal`` (R3-1): a hand-assembled or field-edited report fails at verification —
    verified tokens are minted from metered episodes only."""

    device_j: float
    cloud_j: float
    network_j: float
    device_j_per_node: tuple[float, ...]
    transition_j_per_node: tuple[float, ...]
    cloud_vcpu_hours: float
    network_bytes: float
    coefficients: EnergyCoefficients
    cluster_sha256: str
    horizon_s: float
    transition_counts: tuple[int, ...]
    seal: str = ""

    def __post_init__(self) -> None:
        n = len(self.device_j_per_node)
        if len(self.transition_j_per_node) != n or len(self.transition_counts) != n:
            raise ValueError("per-node report vectors must all cover the same node set (R3-1)")
        for c in self.transition_counts:
            if not isinstance(c, int) or isinstance(c, bool) or c < 0:
                raise ValueError("transition counts must be non-negative ints (R3-1)")
        # R4-2: a meter seal authenticates bytes, not semantics. Validate every
        # scientifically material scalar before a report can exist, then check the
        # accounting identities that tie category totals to their meter bases.
        for name in (
            "device_j",
            "cloud_j",
            "network_j",
            "cloud_vcpu_hours",
            "network_bytes",
            "horizon_s",
        ):
            _finite_nonneg(name, getattr(self, name))
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be a finite positive number")
        for label, values in (
            ("device_j_per_node", self.device_j_per_node),
            ("transition_j_per_node", self.transition_j_per_node),
        ):
            for i, value in enumerate(values):
                _finite_nonneg(f"{label}[{i}]", value)
        if not isinstance(self.cluster_sha256, str) or len(self.cluster_sha256) != 64:
            raise ValueError("cluster_sha256 must be a 64-character digest")
        if not math.isclose(
            self.device_j,
            math.fsum(self.device_j_per_node),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("device_j does not equal the sum of device_j_per_node (R4-2)")
        if any(
            t > d for t, d in zip(self.transition_j_per_node, self.device_j_per_node, strict=True)
        ):
            raise ValueError("transition energy cannot exceed total device energy per node")
        expected_cloud = self.cloud_vcpu_hours * self.coefficients.cloud_j_per_vcpu_hour
        if not math.isclose(self.cloud_j, expected_cloud, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("cloud_j does not match cloud_vcpu_hours x coefficient (R4-2)")
        expected_network = self.network_bytes * self.coefficients.network_j_per_byte
        if not math.isclose(self.network_j, expected_network, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("network_j does not match network_bytes x coefficient (R4-2)")
        if not self.seal:
            raise ValueError(
                "EnergyReport carries no seal — reports are produced only by "
                "EnergyMeter.finish() (R3-1); hand-assembled reports cannot enter "
                "verified-energy tokens"
            )

    def payload(self) -> dict:
        """Canonical seal payload: every field except the seal itself."""
        return {
            "device_j": self.device_j,
            "cloud_j": self.cloud_j,
            "network_j": self.network_j,
            "device_j_per_node": list(self.device_j_per_node),
            "transition_j_per_node": list(self.transition_j_per_node),
            "cloud_vcpu_hours": self.cloud_vcpu_hours,
            "network_bytes": self.network_bytes,
            "coefficients_sha256": self.coefficients.sha256(),
            "cluster_sha256": self.cluster_sha256,
            "horizon_s": self.horizon_s,
            "transition_counts": list(self.transition_counts),
        }

    def verify_seal(self) -> None:
        if self.seal != _seal(_REPORT_SEAL_TAG, self.payload()):
            raise ValueError(
                "EnergyReport seal mismatch — the report's fields were changed after "
                "metering (R3-1); only EnergyMeter.finish() output is verifiable"
            )

    def provenance(self, boundary: str) -> ProvenanceRecord:
        self.verify_seal()
        c = self.coefficients
        table = {
            "device": ProvenanceRecord(
                boundary="device",
                included=("fog/edge node power incl. idle, suspended, transitions",),
                excluded=("cloud device power (allocated instead)", "network", "carbon"),
                source_status="modeled",
                source=c.source,
                parameter_ranges=c.parameter_ranges,
                bundle_set=c.bundle_set,
                coefficients_version=c.version,
                coefficients_sha256=c.sha256(),
                source_records=_records_of_kind(c, ("machine", "transition")),
            ),
            "cloud-allocated": ProvenanceRecord(
                boundary="cloud-allocated",
                included=(
                    "bound-pod vCPU-hours on tier-0 nodes x per-vCPU-hour coefficient",
                    f"baseline allocation floor ({c.cloud_floor_vcpu} vCPU, whole horizon)",
                ),
                excluded=("device power of cloud machines", "PUE unless folded in", "carbon"),
                source_status="allocated-point-estimate",
                source=c.source,
                parameter_ranges=c.parameter_ranges,
                bundle_set=c.bundle_set,
                coefficients_version=c.version,
                coefficients_sha256=c.sha256(),
                source_records=_records_of_kind(c, ("cloud-allocation",)),
            ),
            "network-allocated": ProvenanceRecord(
                boundary="network-allocated",
                included=(
                    "call payload bytes (request; response on blocking edges)",
                    "rebalancer state-transfer bytes",
                )
                + (("image-pull bytes",) if c.include_image_pull_bytes else ()),
                excluded=("per-hop path modeling", "idle network equipment", "carbon"),
                source_status="allocated-point-estimate",
                source=c.source,
                parameter_ranges=c.parameter_ranges,
                bundle_set=c.bundle_set,
                coefficients_version=c.version,
                coefficients_sha256=c.sha256(),
                source_records=_records_of_kind(c, ("network-allocation",)),
                notes=(
                    "path-scope assumption (stated, R1-5): one per-byte allocation "
                    "coefficient is applied to ALL cross-node transfers — edge/fog "
                    "LAN hops included — although the cited sources scope their "
                    "figures to inter-data-center (CCF) or core+fixed-line access "
                    "(Aslan) paths. This is a synthetic all-transfer allocation "
                    "assumption; the coefficient choice is a structural factor "
                    "(bundles swap coefficient entries) and the value is swept in "
                    "the GSA."
                ),
            ),
        }
        return table[boundary]

    def closed_boundary_j(self, weights: BoundaryWeights) -> tuple[float, ProvenanceRecord]:
        """The only sanctioned combination of the three categories (explicit weights)."""
        self.verify_seal()
        value = (
            weights.device * self.device_j
            + weights.cloud * self.cloud_j
            + weights.network * self.network_j
        )
        c = self.coefficients
        record = ProvenanceRecord(
            boundary="closed-composite",
            included=(
                f"device x {weights.device}",
                f"cloud-allocated x {weights.cloud}",
                f"network-allocated x {weights.network}",
            ),
            excluded=("carbon",),
            source_status="modeled-composite",
            source=c.source,
            parameter_ranges=c.parameter_ranges,
            # R2-2: the composite carries the coefficients' own bundle-set label —
            # never a hardcoded pre-Phase-3 branding.
            bundle_set=c.bundle_set,
            coefficients_version=c.version,
            coefficients_sha256=c.sha256(),
            notes="composite of modeled device energy and allocation point estimates",
            source_records=tuple(c.source_records),
        )
        return value, record

    def all_provenance(self) -> dict[str, ProvenanceRecord]:
        """The three per-category records (serialized with every EpisodeResult — §4.3:
        figures carry their provenance, not just a coefficient hash)."""
        self.verify_seal()
        return {b: self.provenance(b) for b in ("device", "cloud-allocated", "network-allocated")}


class EnergyMeter:
    """Piecewise-constant power integration over the episode.

    The runtime calls ``set_utilization`` / ``set_state`` / ``set_cloud_alloc`` /
    ``add_bytes`` at state-change instants; energy accumulates as P x dt between changes.
    Deterministic: pure arithmetic on event-ordered updates.
    """

    def __init__(self, cluster: ClusterSpec, coefficients: EnergyCoefficients) -> None:
        self._cluster = cluster
        self._coeff = coefficients
        n = len(cluster.nodes)
        self._classes = [cluster.node_class_of(i) for i in range(n)]
        self._tiers = [cluster.nodes[i].tier for i in range(n)]
        # state: "awake" | "suspended" | "transition" (transition power = idle power)
        self._state = ["suspended" if s.initially_suspended else "awake" for s in cluster.nodes]
        self._busy_millicores = [0.0] * n
        self._last_t = [0.0] * n
        self._device_j = [0.0] * n
        self._transition_j = [0.0] * n
        # Currently-allocated vCPUs on cloud nodes + the baseline allocation floor
        # (cloud_floor_vcpu is allocated for the whole horizon: metered, floored, and
        # ceilinged identically — the category boundary is consistent).
        self._cloud_vcpu = coefficients.cloud_floor_vcpu
        self._cloud_vcpu_hours = 0.0
        self._cloud_last_t = 0.0
        self._bytes = 0.0
        self._transition_counts = [0] * n  # per-node suspend/wake pulses (ceiling audit)

    # -- device -------------------------------------------------------------
    def _node(self, node_id: int) -> int:
        if (
            not isinstance(node_id, int)
            or isinstance(node_id, bool)
            or not 0 <= node_id < len(self._cluster.nodes)
        ):
            raise ValueError(f"node_id must be an integer in [0, {len(self._cluster.nodes)})")
        return node_id

    @staticmethod
    def _time(now: float) -> float:
        return _finite_nonneg("meter time", now)

    def _accrue(self, node_id: int, now: float) -> None:
        node_id = self._node(node_id)
        now = self._time(now)
        dt = now - self._last_t[node_id]
        if dt < 0:
            raise ValueError("time went backwards in EnergyMeter")
        cls = self._classes[node_id]
        state = self._state[node_id]
        if state == "awake":
            u = self._busy_millicores[node_id] / self._cluster.nodes[node_id].cpu_capacity
            p = cls.power_w(u)
        elif state == "suspended":
            p = cls.suspended_power_w
        else:  # transition window: idle power for its duration; pulse added separately
            p = cls.power_idle_w
        if self._tiers[node_id] != TIER_CLOUD:
            self._device_j[node_id] += p * dt
        self._last_t[node_id] = now

    def set_busy_millicores(self, node_id: int, busy: float, now: float) -> None:
        node_id = self._node(node_id)
        busy = _finite_nonneg("busy_millicores", busy)
        if busy > self._cluster.nodes[node_id].cpu_capacity:
            raise ValueError("busy_millicores cannot exceed node CPU capacity")
        self._accrue(node_id, now)
        self._busy_millicores[node_id] = busy

    def set_state(self, node_id: int, state: str, now: float) -> None:
        node_id = self._node(node_id)
        if state not in ("awake", "suspended", "transition"):
            raise ValueError(f"unknown node state {state}")
        self._accrue(node_id, now)
        self._state[node_id] = state

    def add_transition_pulse(self, node_id: int, energy_j: float) -> None:
        """Published suspend/wake transition energy, accounted as device energy."""
        node_id = self._node(node_id)
        energy_j = _finite_nonneg("transition energy_j", energy_j)
        self._transition_counts[node_id] += 1
        if self._tiers[node_id] != TIER_CLOUD:
            self._transition_j[node_id] += energy_j

    def transition_counts(self) -> tuple[int, ...]:
        return tuple(self._transition_counts)

    # -- cloud allocation ---------------------------------------------------
    def _accrue_cloud(self, now: float) -> None:
        now = self._time(now)
        dt = now - self._cloud_last_t
        if dt < 0:
            raise ValueError("time went backwards in EnergyMeter")
        self._cloud_vcpu_hours += self._cloud_vcpu * dt / 3600.0
        self._cloud_last_t = now

    def set_cloud_alloc_millicores(self, total_millicores: float, now: float) -> None:
        total_millicores = _finite_nonneg("cloud allocation millicores", total_millicores)
        self._accrue_cloud(now)
        self._cloud_vcpu = self._coeff.cloud_floor_vcpu + total_millicores / 1000.0

    # -- network ------------------------------------------------------------
    def add_bytes(self, n_bytes: float) -> None:
        n_bytes = _finite_nonneg("network bytes", n_bytes)
        total = self._bytes + n_bytes
        if not math.isfinite(total):
            raise ValueError("cumulative network bytes must remain finite")
        self._bytes = total

    # -- report ---------------------------------------------------------------
    def finish(self, horizon_s: float) -> EnergyReport:
        horizon_s = self._time(horizon_s)
        if horizon_s <= 0:
            raise ValueError("horizon_s must be a finite positive number")
        for node_id in range(len(self._cluster.nodes)):
            self._accrue(node_id, horizon_s)
        self._accrue_cloud(horizon_s)
        per_node = tuple(
            self._device_j[i] + self._transition_j[i] for i in range(len(self._cluster.nodes))
        )
        fields = {
            "device_j": math.fsum(per_node),
            "cloud_j": self._cloud_vcpu_hours * self._coeff.cloud_j_per_vcpu_hour,
            "network_j": self._bytes * self._coeff.network_j_per_byte,
            "device_j_per_node": per_node,
            "transition_j_per_node": tuple(self._transition_j),
            "cloud_vcpu_hours": self._cloud_vcpu_hours,
            "network_bytes": self._bytes,
            # Metering identity (R2-1): what verify_covers checks the certificate
            # against — recorded by the meter, not asserted by the caller.
            "cluster_sha256": cluster_sha256(self._cluster),
            "horizon_s": float(horizon_s),
            "transition_counts": self.transition_counts(),
        }
        payload = dict(fields)
        payload["device_j_per_node"] = list(fields["device_j_per_node"])
        payload["transition_j_per_node"] = list(fields["transition_j_per_node"])
        payload["transition_counts"] = list(fields["transition_counts"])
        payload["coefficients_sha256"] = self._coeff.sha256()
        return EnergyReport(
            **fields,
            coefficients=self._coeff,
            seal=_seal(_REPORT_SEAL_TAG, payload),  # R3-1: meter-authenticated report
        )


# ------------------------------------------------------------------- floor and ceiling
@dataclass(frozen=True)
class BoundaryBounds:
    """Explicit episode-level bounds needed for the ceiling M (W2: the ceiling covers the
    *complete* closed boundary — max node power + bounded transition energy + cloud and
    network allocation ceilings). Callers must state their bounds; there is no default
    'infinite' or silently-zero component."""

    max_transitions_per_node: int
    network_bytes_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.max_transitions_per_node, int) or isinstance(
            self.max_transitions_per_node, bool
        ):
            raise ValueError("max_transitions_per_node must be an int")
        if self.max_transitions_per_node < 0:
            raise ValueError("max_transitions_per_node must be >= 0")
        _finite_nonneg("network_bytes_bound", self.network_bytes_bound)


def idle_floor_j(
    cluster: ClusterSpec,
    coefficients: EnergyCoefficients,
    horizon_s: float,
    weights: BoundaryWeights,
) -> float:
    """Policy-independent idle-floor energy I (plan §4.1, V1): the closed-boundary energy
    of the horizon with every default-awake node idle (suspended-by-default nodes at their
    suspended draw) plus allocation floors. Raises if I is not strictly positive (W2:
    idle-floor positivity is verified for every evaluable instance)."""

    device = 0.0
    for node in cluster.nodes:
        cls = cluster.node_class_of(node.id)
        if node.tier == TIER_CLOUD:
            continue
        # power_w(0.0) is what the meter charges an idle awake node (curve-aware).
        p = cls.suspended_power_w if node.initially_suspended else cls.power_w(0.0)
        device += p * horizon_s
    cloud = (
        coefficients.cloud_floor_vcpu * (horizon_s / 3600.0) * coefficients.cloud_j_per_vcpu_hour
    )
    network = 0.0  # allocation floor: no traffic -> no allocated network energy
    value = weights.device * device + weights.cloud * cloud + weights.network * network
    if not value > 0.0:
        raise ValueError(
            "idle-floor energy must be strictly positive (V1/W2); this cluster/coefficient "
            "combination yields a zero floor — not an evaluable instance"
        )
    return value


def ceiling_j(
    cluster: ClusterSpec,
    coefficients: EnergyCoefficients,
    horizon_s: float,
    weights: BoundaryWeights,
    bounds: BoundaryBounds,
) -> float:
    """Max-power ceiling energy M (plan §6.3 penalty bound, W2): all nodes awake at the
    maximum of their power curve for the whole horizon, plus the bounded transition
    energy, plus the cloud allocation ceiling (every tier-0 vCPU allocated for the whole
    horizon), plus the stated network-bytes bound. Upper-bounds any policy's closed
    boundary under the same weights and bounds."""

    device = 0.0
    cloud_vcpu = coefficients.cloud_floor_vcpu  # the baseline allocation is metered too
    for node in cluster.nodes:
        cls = cluster.node_class_of(node.id)
        if node.tier == TIER_CLOUD:
            cloud_vcpu += node.cpu_capacity / 1000.0
            continue
        device += cls.max_curve_power_w() * horizon_s
        if cls.suspendable:
            per_transition = max(cls.suspend_energy_j or 0.0, cls.wake_energy_j or 0.0)
            device += bounds.max_transitions_per_node * per_transition
    cloud = cloud_vcpu * (horizon_s / 3600.0) * coefficients.cloud_j_per_vcpu_hour
    network = bounds.network_bytes_bound * coefficients.network_j_per_byte
    return weights.device * device + weights.cloud * cloud + weights.network * network


@dataclass(frozen=True)
class EnergyFigure:
    """A reportable figure: value + provenance, the §4.3 unit of reporting."""

    value_j: float
    provenance: ProvenanceRecord


# ------------------------------------------------------------------- certificate
@dataclass(frozen=True)
class VerifiedInstanceEnergy:
    """The verified result token (R2-1): one episode's closed-boundary energy plus the
    complete instance identity it was verified under — produced **only** by
    :meth:`BoundaryCertificate.verify_covers`. Fitness outcomes
    (``sim.energy.fitness.InstanceOutcome``) and the certificate's regret/penalty
    methods consume these tokens, never bare floats, so an energy figure cannot cross
    boundaries without carrying its verification.

    Like every seal in this package this is mistake-proofing, not cryptography (the
    scheme is public); the durable boundary is the P2 registration.
    """

    scenario_id: str
    seed: int
    bundle_id: str
    cluster_sha256: str
    coefficients_sha256: str
    horizon_s: float
    energy_j: float
    idle_floor_j: float
    ceiling_j: float
    transition_counts: tuple[int, ...]
    network_bytes: float
    #: seal of the issuing certificate (R3-1): binds the token to the *complete*
    #: certificate payload — weights, bounds, node count included — so two
    #: certificates whose derived I and M coincide are still distinct boundaries.
    certificate_seal: str = ""
    seal: str = ""

    def payload(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "bundle_id": self.bundle_id,
            "cluster_sha256": self.cluster_sha256,
            "coefficients_sha256": self.coefficients_sha256,
            "horizon_s": self.horizon_s,
            "energy_j": self.energy_j,
            "idle_floor_j": self.idle_floor_j,
            "ceiling_j": self.ceiling_j,
            "transition_counts": list(self.transition_counts),
            "network_bytes": self.network_bytes,
            "certificate_seal": self.certificate_seal,
        }

    def __post_init__(self) -> None:
        if not self.seal or not self.certificate_seal:
            raise ValueError(
                "VerifiedInstanceEnergy carries no seal — it is produced only by "
                "BoundaryCertificate.verify_covers (R2-1/R3-1)"
            )

    def verify_seal(self) -> None:
        if self.seal != _seal(_VERIFIED_SEAL_TAG, self.payload()):
            raise ValueError(
                "VerifiedInstanceEnergy seal mismatch — the token's fields were "
                "changed after verification (R2-1); re-verify the episode's report"
            )


@dataclass(frozen=True)
class BoundaryCertificate:
    """The common-boundary identity for one evaluation instance (R1-6/R2-1, W2).

    W2's penalty-dominance property is only a theorem when the ceiling M, the idle
    floor I, and every evaluated policy's energy E are computed over the *same*
    cluster, coefficients, weights, horizon, and bounds. A bare ``(M, ref, I)``
    triple cannot witness that. The certificate binds all of it: it is constructed
    once per instance by :func:`certify_boundary` (the only seal-producing
    constructor — a hand-built certificate fails at every use), which computes I —
    positivity verified — and M from the same inputs and digests the cluster and
    coefficients. :meth:`verify_covers` then checks every episode's report against
    the full identity — same cluster digest, same horizon, same coefficient set,
    transition counts (from the report's own metering) within the stated bound,
    network bytes within the stated bound, and E ≤ M — and returns a sealed
    :class:`VerifiedInstanceEnergy` token, the only currency the fitness layer and
    the regret/penalty methods accept. A caller who states bounds an episode then
    exceeds fails loudly at verification, not silently in the algebra.

    The concrete bound *values* per registered instance are frozen in P2 (W2); this
    class is the Phase-3 mechanism that makes them checkable.
    """

    scenario_id: str
    seed: int
    bundle_id: str
    horizon_s: float
    weights: BoundaryWeights
    bounds: BoundaryBounds
    cluster_sha256: str
    coefficients_sha256: str
    idle_floor_j: float
    ceiling_j: float
    n_nodes: int = 0
    seal: str = ""

    def payload(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "bundle_id": self.bundle_id,
            "horizon_s": self.horizon_s,
            "weights": [self.weights.device, self.weights.cloud, self.weights.network],
            "bounds": [
                self.bounds.max_transitions_per_node,
                self.bounds.network_bytes_bound,
            ],
            "cluster_sha256": self.cluster_sha256,
            "coefficients_sha256": self.coefficients_sha256,
            "idle_floor_j": self.idle_floor_j,
            "ceiling_j": self.ceiling_j,
            "n_nodes": self.n_nodes,
        }

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("certificate seed must be an int (explicit, docs/SEEDS.md)")
        if not self.idle_floor_j > 0:
            raise ValueError("certificate idle floor must be strictly positive (V1/W2)")
        if not math.isfinite(self.ceiling_j) or self.ceiling_j <= 0:
            raise ValueError("certificate ceiling must be finite and positive (W2)")
        if not self.seal:
            raise ValueError(
                "BoundaryCertificate carries no seal — construct it through "
                "certify_boundary(), the only constructor that computes I and M from "
                "one boundary (R2-1)"
            )

    def verify_seal(self) -> None:
        if self.seal != _seal(_CERT_SEAL_TAG, self.payload()):
            raise ValueError(
                "BoundaryCertificate seal mismatch — fields were changed after "
                "certification (R2-1); I and M no longer witness one boundary"
            )

    def verify_covers(self, report: EnergyReport) -> VerifiedInstanceEnergy:
        """Check one episode's report against this certificate and return the sealed
        verified-energy token. Raises if the episode escaped the certified boundary:
        different cluster, different horizon, different coefficients, more
        transitions than bounded (from the report's own metering counts), more bytes
        than bounded, or energy above the ceiling."""
        self.verify_seal()
        report.verify_seal()  # R3-1: only meter-produced reports mint tokens
        if report.cluster_sha256 != self.cluster_sha256:
            raise ValueError(
                "report was metered on a different cluster than certified — not the "
                "same closed boundary (R2-1)"
            )
        if len(report.transition_counts) != self.n_nodes:
            raise ValueError(
                f"report covers {len(report.transition_counts)} nodes but the "
                f"certificate certifies {self.n_nodes} — transition-count coverage "
                "is incomplete (R3-1)"
            )
        if report.horizon_s != self.horizon_s:
            raise ValueError(
                f"report covers horizon {report.horizon_s} s but the certificate "
                f"certifies {self.horizon_s} s — not the same closed boundary (R2-1)"
            )
        if report.coefficients.sha256() != self.coefficients_sha256:
            raise ValueError(
                "report was metered under different coefficients than certified — "
                "not the same closed boundary"
            )
        over = [
            (i, n)
            for i, n in enumerate(report.transition_counts)
            if n > self.bounds.max_transitions_per_node
        ]
        if over:
            raise ValueError(
                f"episode exceeded the certified transition bound "
                f"({self.bounds.max_transitions_per_node}/node): nodes {over} — the "
                "ceiling M does not cover this episode; re-certify with honest bounds"
            )
        if report.network_bytes > self.bounds.network_bytes_bound:
            raise ValueError(
                f"episode transferred {report.network_bytes} bytes > certified bound "
                f"{self.bounds.network_bytes_bound} — the ceiling M does not cover "
                "this episode; re-certify with honest bounds"
            )
        energy, _ = report.closed_boundary_j(self.weights)
        if energy > self.ceiling_j:
            raise ValueError(
                f"episode energy {energy} J exceeds the certified ceiling "
                f"{self.ceiling_j} J — W2 dominance precondition violated"
            )
        fields = {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "bundle_id": self.bundle_id,
            "cluster_sha256": self.cluster_sha256,
            "coefficients_sha256": self.coefficients_sha256,
            "horizon_s": self.horizon_s,
            "energy_j": energy,
            "idle_floor_j": self.idle_floor_j,
            "ceiling_j": self.ceiling_j,
            "transition_counts": tuple(report.transition_counts),
            "network_bytes": report.network_bytes,
            # R3-1: the token cites the issuing certificate's own seal, which covers
            # the complete payload — weights and bounds included.
            "certificate_seal": self.seal,
        }
        payload = dict(fields)
        payload["transition_counts"] = list(fields["transition_counts"])
        return VerifiedInstanceEnergy(**fields, seal=_seal(_VERIFIED_SEAL_TAG, payload))

    def _check_token(self, token: VerifiedInstanceEnergy, role: str) -> None:
        token.verify_seal()
        # R3-1: comparing the certificate seal covers the *complete* boundary —
        # scenario/seed/bundle/cluster/coefficients/horizon/weights/bounds/I/M —
        # so two certificates whose derived I and M coincide are still distinct.
        if token.certificate_seal != self.seal:
            raise ValueError(
                f"{role} token was verified under a different certificate than this "
                "one — the W2 common-boundary precondition fails (R2-1/R3-1)"
            )

    # -- regret under the certified boundary (plan §6.3 1b) --------------------
    def feasible_regret(
        self, energy: VerifiedInstanceEnergy, reference: VerifiedInstanceEnergy
    ) -> float:
        """Checkpoint regret over two *verified* tokens of this same certificate
        (R2-1: the incumbent and the reference must both carry the boundary)."""
        from sim.energy.fitness import feasible_regret

        self.verify_seal()
        self._check_token(energy, "incumbent")
        self._check_token(reference, "reference")
        return feasible_regret(energy.energy_j, reference.energy_j, self.idle_floor_j)

    def penalty_regret(self, reference: VerifiedInstanceEnergy) -> float:
        """P(instance) = (M − ref)/I + 1 with M, I from this certificate — the
        reference must be a verified token of this same certificate (R2-1), which
        already witnesses ref ≤ M."""
        from sim.energy.fitness import penalty_regret

        self.verify_seal()
        self._check_token(reference, "reference")
        return penalty_regret(self.ceiling_j, reference.energy_j, self.idle_floor_j)


def cluster_sha256(cluster: ClusterSpec) -> str:
    """Canonical digest of a cluster spec (nested frozen dataclasses of scalars)."""
    payload = dataclasses.asdict(cluster)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def certify_boundary(
    cluster: ClusterSpec,
    coefficients: EnergyCoefficients,
    horizon_s: float,
    weights: BoundaryWeights,
    bounds: BoundaryBounds,
    *,
    scenario_id: str,
    seed: int,
    bundle_id: str,
) -> BoundaryCertificate:
    """Compute I and M from one set of inputs and bind them to the instance identity.

    This is the only sanctioned way to obtain the (I, M) pair used by the fitness
    normalizer and the §6.3 penalty: both come from the same cluster, coefficients,
    weights, horizon, and bounds, so the W2 dominance algebra is about one boundary.
    """
    if not (isinstance(horizon_s, int | float) and math.isfinite(horizon_s) and horizon_s > 0):
        raise ValueError("horizon_s must be a finite positive number")
    floor = idle_floor_j(cluster, coefficients, horizon_s, weights)
    ceiling = ceiling_j(cluster, coefficients, horizon_s, weights, bounds)
    fields = {
        "scenario_id": scenario_id,
        "seed": seed,
        "bundle_id": bundle_id,
        "horizon_s": float(horizon_s),
        "weights": weights,
        "bounds": bounds,
        "cluster_sha256": cluster_sha256(cluster),
        "coefficients_sha256": coefficients.sha256(),
        "idle_floor_j": floor,
        "ceiling_j": ceiling,
        "n_nodes": len(cluster.nodes),
    }
    payload = dict(fields)
    payload["weights"] = [weights.device, weights.cloud, weights.network]
    payload["bounds"] = [bounds.max_transitions_per_node, bounds.network_bytes_bound]
    return BoundaryCertificate(**fields, seal=_seal(_CERT_SEAL_TAG, payload))
