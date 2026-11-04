"""EnergyModelBundle assembly (TODO 3.3; plan §4.1, S4/T4).

An :class:`EnergyModelBundle` is the pre-registered unit of energy modeling: a
*jointly sampled whole-machine tuple* per node class (power curve **and** capacity
from the same published machine), plus cloud/network allocation coefficients and the
transition-cost table.

Joint-tuple integrity is enforced at four layers, honestly labeled:

1. :func:`bundle_from_registry` is the **only public bundle factory** (R2-5): it
   accepts registry entry *ids*, resolves every coefficient value, curve point,
   capacity, and provenance record from the VERIFIED registry internally, and
   re-validates the result — there is no public path that seals caller-supplied
   values (the raw assembler is private).
2. :func:`machine_tuple_from_registry` is the only public tuple factory (R3-2) —
   every tuple derives from a single VERIFIED registry entry + the validated
   mapping (the raw constructor is private), and carries a content **seal** over
   all of its fields. Consumption (``node_class_binding()``,
   ``node_class_spec()``, ``node_class_specs()``, ``energy_coefficients()``)
   re-verifies the seal, so a field-swapped copy (e.g. ``dataclasses.replace``
   combining one machine's power with another's capacity) fails loudly at the point
   of use. Simulator cluster construction consumes capacities through the
   seal-verifying ``node_class_binding()`` bridge, never by reading fields off the
   dataclass.
3. :func:`validate_bundle` re-derives every tuple from its cited registry entry and
   compares exactly — the registry-grounded check (no impossible devices) — and
   checks the capacity anchors (R2-8) and structured source records (R2-2) against
   the registry.
4. The seal is a mistake-proofing control, **not** a cryptographic boundary: the
   sealing scheme is public, so a determined operator can forge it. The durable
   integrity boundary is the P2 registration hashing the full bundle manifests
   (:meth:`EnergyModelBundle.manifest_sha256`, pinned by
   ``holdout_gate.bundle_list_sha256``).

Curve families (T4 held-out structural diversity):

- ``piecewise-joint`` — the published joint (normalized throughput, power) points,
  linearly interpolated (development family; exactly what Phase 2's
  ``NodeClassSpec`` consumes).
- ``parametric-fan`` — the nonlinear parametric form of Fan, Weber & Barroso,
  "Power Provisioning for a Warehouse-sized Computer" (ISCA 2007, DOI
  10.1145/1250662.1250665): ``P(u) = P_idle + (P_max − P_idle)(2u − u^r)``, with the
  exponent ``r`` fitted to the machine's own published points by deterministic
  least squares **on the physically monotone interval r ∈ [1, 2]** — for r in that
  interval, ``2u − u^r`` is nondecreasing on [0, 1] and bounded by [0, 1], so
  ``P_idle ≤ P(u) ≤ P_max`` everywhere (an unconstrained fit can dip below idle
  power at low load and reward a policy for adding load — R1-2). The fitted
  exponent and the maximum absolute residual are recorded in the tuple; machines
  whose published curve is poorly served by the constrained family simply show
  large recorded residuals — that is the point of a structurally different family.
  For Phase 2 consumption the fitted curve is evaluated on a fixed dense grid
  (``PARAMETRIC_GRID_N`` points, validated nondecreasing); the grid is part of the
  family definition.

Development vs. held-out split (T4): different machines, different curve families,
different allocation-coefficient entries — enforced by :func:`validate_split`.
The role rosters, bundle ids, and manifest hashes used by that validator come from
the separately stored sealed energy-analysis registration; the builders below are
construction helpers, not runtime authority. Any P2 change requires a new
registration payload and hash (see ``README.md`` §"P2 pre-registration touchpoints").
The sanctioned held-out evaluation path is ``holdout_gate.py`` (run-once, tagged
commit) — a **procedural honest-operator control**: the gate and the
``acknowledged_one_shot`` flag stop accidental access, not a dishonest operator;
the timestamped P2 registration is the durable provenance boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

from sim.energy.capacity import CapacityMapping, DomainAnchor, validate_mapping
from sim.energy.registration import load_energy_analysis_registration
from sim.energy.sources import (
    ArmSbcMachine,
    Provenance,
    SourceRegistry,
    SpecPowerMachine,
    TransitionMeasurement,
    load_pinned_sources,
)
from sim.spec import EnergyCoefficients, EnergySourceRecord, NodeClassSpec

PIECEWISE = "piecewise-joint"
PARAMETRIC = "parametric-fan"
_FAMILIES = (PIECEWISE, PARAMETRIC)

PARAMETRIC_GRID_N = 101  # fixed dense grid for evaluating the fitted parametric form
# Deterministic fit search interval for the exponent, restricted to the physically
# monotone domain (R1-2): for r in [1, 2], g(u) = 2u - u^r has
# g'(u) = 2 - r*u^(r-1) >= 0 on [0, 1] and 0 <= g(u) <= 1, so
# P_idle <= P(u) <= P_max everywhere. An unconstrained fit (e.g. r ~ 0.72 on the
# 4-socket held-out machines) dips below idle power at low load.
_FIT_R_LO, _FIT_R_HI = 1.0, 2.0
# Machines publishing too few interior points to identify the exponent (e.g. SBC
# characterizations with idle + one load point) use the exponent published by Fan et
# al. themselves ("an empirical value of r = 1.4"), rather than an unidentified fit.
FAN_DEFAULT_R = 1.4
_MIN_POINTS_FOR_FIT = 4

_MACHINE_SEAL_TAG = "machine-tuple-seal-v1:"
_BUNDLE_SEAL_TAG = "energy-model-bundle-seal-v1:"


MANIFEST_SCHEMA = "energy-model-bundle-manifest-v1"  # explicit schema/version (R2-6)


def _canonical(obj: object) -> str:
    # allow_nan=False (R2-6): a NaN/Infinity anywhere in a seal/manifest payload is a
    # hard error, never a silently-sealed non-standard token.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _req_str(name: str, v: object) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty string")
    return v


def _finite(name: str, v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def _source_record(entry_id: str, kind: str, prov: Provenance) -> EnergySourceRecord:
    """Structured source reference (R2-2) carried from the registry through the
    manifest and coefficients into every downstream ProvenanceRecord."""
    return EnergySourceRecord(
        entry_id=entry_id,
        kind=kind,
        source_name=prov.source_name,
        url=prov.url,
        doi=prov.doi,
        version_or_date_accessed=prov.version_or_date_accessed,
        status=prov.status,
        content_sha256=prov.content_sha256,
    )


# ---------------------------------------------------------------- parametric fit
def fan_power_w(u: float, p_idle: float, p_max: float, r: float) -> float:
    """Fan et al. (2007) parametric power model; exact at u=0 and u=1."""
    uu = min(max(u, 0.0), 1.0)
    return p_idle + (p_max - p_idle) * (2.0 * uu - uu**r)


def fit_fan_exponent(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    """Deterministic 1-D least-squares fit of the exponent ``r`` to the published
    joint points (idle and peak are matched exactly by the functional form), over the
    physically monotone interval ``r ∈ [1, 2]``. Returns ``(r, max_abs_residual_w)``.
    Golden-section search with a fixed iteration count — no randomness, no
    tolerance-dependent termination."""
    p_idle = points[0][1]
    p_max = points[-1][1]
    if points[0][0] != 0.0 or points[-1][0] != 1.0:
        raise ValueError("fit expects joint points spanning u=0..1")
    if len(points) < _MIN_POINTS_FOR_FIT:
        # Endpoints match exactly; interior points cannot identify r -> use the
        # exponent published by Fan et al. (cited in the module docstring).
        r = FAN_DEFAULT_R
        resid = max(abs(fan_power_w(u, p_idle, p_max, r) - w) for u, w in points)
        return r, resid

    def sse(r: float) -> float:
        return sum((fan_power_w(u, p_idle, p_max, r) - w) ** 2 for u, w in points)

    inv_phi = (5**0.5 - 1) / 2
    a, b = _FIT_R_LO, _FIT_R_HI
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(80):  # fixed iteration count -> deterministic across platforms
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = sse(d)
    r = (a + b) / 2.0
    resid = max(abs(fan_power_w(u, p_idle, p_max, r) - w) for u, w in points)
    return r, resid


def parametric_grid(
    points: tuple[tuple[float, float], ...], r: float
) -> tuple[tuple[float, float], ...]:
    if not _FIT_R_LO <= r <= _FIT_R_HI:
        raise ValueError(
            f"parametric exponent r={r} outside the monotone interval [{_FIT_R_LO}, {_FIT_R_HI}]"
        )
    p_idle, p_max = points[0][1], points[-1][1]
    n = PARAMETRIC_GRID_N
    grid = tuple((i / (n - 1), fan_power_w(i / (n - 1), p_idle, p_max, r)) for i in range(n))
    # Defense in depth: r in [1, 2] guarantees both properties; assert them anyway so
    # a future family change cannot silently reintroduce below-idle power (R1-2).
    for (_, w0), (u1, w1) in zip(grid, grid[1:], strict=False):
        if w1 < w0:
            raise ValueError(f"parametric grid not nondecreasing at u={u1} (r={r})")
    if grid[0][1] < p_idle or grid[-1][1] > p_max + 1e-9:
        raise ValueError("parametric grid escapes the [idle, max] power band")
    return grid


# ---------------------------------------------------------------- machine tuples
@dataclass(frozen=True)
class MachineTuple:
    """One node class: power curve + capacity from the *same* published machine, plus
    the transition entry applied to it (or None → non-suspendable, per plan §3.4).

    Only the module's sealing constructors (public:
    :func:`machine_tuple_from_registry`) produce a valid ``seal``; every consumption
    path re-verifies it, so mixing one machine's power with another's capacity
    (however the copy was made) fails at the point of use."""

    node_class_name: str
    machine_id: str
    curve_family: str
    power_curve_points: tuple[tuple[float, float], ...]
    capacity_millicores: int
    capacity_rule: str
    #: the validated anchor that priced the capacity (None → core-count fallback);
    #: materialized in the manifest so the registered artifact carries it (R2-8).
    capacity_anchor: DomainAnchor | None
    transition_entry_id: str | None
    suspend_energy_j: float | None
    suspend_latency_s: float | None
    wake_energy_j: float | None
    wake_latency_s: float | None
    suspended_power_w: float
    #: Phase-2 simulation parameter, materialized in the manifest (R2-6): it affects
    #: episode timing and (with include_image_pull_bytes) allocated network energy.
    image_pull_mib_per_s: float = 25.0
    fit_exponent_r: float | None = None
    fit_max_abs_residual_w: float | None = None
    seal: str = ""

    def __post_init__(self) -> None:
        _req_str("node_class_name", self.node_class_name)
        _req_str("machine_id", self.machine_id)
        _req_str("capacity_rule", self.capacity_rule)
        if self.curve_family not in _FAMILIES:
            raise ValueError(f"unknown curve family {self.curve_family!r}")
        if self.curve_family == PARAMETRIC and self.fit_exponent_r is None:
            raise ValueError("parametric tuples must record their fitted exponent")
        cap = self.capacity_millicores
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f"{self.machine_id}: capacity_millicores must be a positive int")
        if not _finite("image_pull_mib_per_s", self.image_pull_mib_per_s) > 0:
            raise ValueError(f"{self.machine_id}: image_pull_mib_per_s must be > 0")
        if _finite("suspended_power_w", self.suspended_power_w) < 0:
            raise ValueError(f"{self.machine_id}: suspended_power_w must be >= 0")
        for name in ("suspend_energy_j", "suspend_latency_s", "wake_energy_j", "wake_latency_s"):
            v = getattr(self, name)
            if v is not None and _finite(name, v) < 0:
                raise ValueError(f"{self.machine_id}: {name} must be >= 0")
        for u, w in self.power_curve_points:
            _finite("curve utilization", u)
            _finite("curve watts", w)
        if not self.seal:
            raise ValueError(
                "MachineTuple carries no seal — construct it through "
                "machine_tuple_from_registry() (joint-tuple discipline)"
            )
        idle = self.power_curve_points[0][1]
        if any(w < idle for _, w in self.power_curve_points):
            raise ValueError(
                f"{self.machine_id}: curve power below the idle point — a loaded state "
                "must never draw less than idle (W2 floor/ceiling sanity)"
            )

    def payload(self) -> dict:
        """Canonical seal/manifest payload: every field except the seal itself."""
        return {
            "node_class_name": self.node_class_name,
            "machine_id": self.machine_id,
            "curve_family": self.curve_family,
            "power_curve_points": [list(p) for p in self.power_curve_points],
            "capacity_millicores": self.capacity_millicores,
            "capacity_rule": self.capacity_rule,
            "capacity_anchor": (self.capacity_anchor.payload() if self.capacity_anchor else None),
            "transition_entry_id": self.transition_entry_id,
            "suspend_energy_j": self.suspend_energy_j,
            "suspend_latency_s": self.suspend_latency_s,
            "wake_energy_j": self.wake_energy_j,
            "wake_latency_s": self.wake_latency_s,
            "suspended_power_w": self.suspended_power_w,
            "image_pull_mib_per_s": self.image_pull_mib_per_s,
            "fit_exponent_r": self.fit_exponent_r,
            "fit_max_abs_residual_w": self.fit_max_abs_residual_w,
        }

    def verify_seal(self) -> None:
        expected = _machine_seal(self.payload())
        if self.seal != expected:
            raise ValueError(
                f"{self.node_class_name}: machine-tuple seal mismatch — the tuple's "
                f"fields do not derive together from one source entry "
                f"({self.machine_id}); joint-tuple discipline violated (was a field "
                "replaced after construction?)"
            )

    def node_class_spec(self) -> NodeClassSpec:
        """Bridge into Phase 2 (``sim.spec.NodeClassSpec`` enforces the W2 ceiling
        constraints: suspended ≤ idle ≤ max curve power). Verifies the seal first."""
        self.verify_seal()
        pts = self.power_curve_points
        return NodeClassSpec(
            name=self.node_class_name,
            power_idle_w=pts[0][1],
            power_max_w=pts[-1][1],
            power_curve_points=pts,
            suspend_energy_j=self.suspend_energy_j,
            suspend_latency_s=self.suspend_latency_s,
            wake_energy_j=self.wake_energy_j,
            wake_latency_s=self.wake_latency_s,
            suspended_power_w=self.suspended_power_w,
            image_pull_mib_per_s=self.image_pull_mib_per_s,
        )

    def node_class_binding(self) -> NodeClassBinding:
        """The single seal-verifying consumption bridge (R2-5): everything simulator
        cluster construction needs — the Phase 2 spec *and* the capacity — from one
        seal check. Reading ``capacity_millicores`` off the dataclass directly
        bypasses the seal; this method is the sanctioned path."""
        return NodeClassBinding(
            spec=self.node_class_spec(),  # verifies the seal
            capacity_millicores=self.capacity_millicores,
            machine=self,
        )


@dataclass(frozen=True)
class NodeClassBinding:
    """What one sealed machine tuple contributes to cluster construction (R2-5):
    the Phase 2 ``NodeClassSpec`` plus the jointly-derived capacity, tied to the
    sealed tuple it came from. Not freely constructible (R3-2): construction
    re-verifies the tuple's seal and that spec/capacity are exactly the tuple's own
    derivation — a binding with a swapped capacity or spec fails immediately."""

    spec: NodeClassSpec
    capacity_millicores: int
    machine: MachineTuple

    def __post_init__(self) -> None:
        self.machine.verify_seal()
        if (
            self.capacity_millicores != self.machine.capacity_millicores
            or self.spec != self.machine.node_class_spec()
        ):
            raise ValueError(
                "NodeClassBinding diverges from its sealed machine tuple — bindings "
                "are produced by MachineTuple.node_class_binding(), never assembled "
                "from loose values (R3-2)"
            )


def _machine_seal(payload: dict) -> str:
    return hashlib.sha256((_MACHINE_SEAL_TAG + _canonical(payload)).encode()).hexdigest()


def _curve_for_family(
    joint_points: tuple[tuple[float, float], ...], family: str
) -> tuple[tuple[float, float], float | None, float | None]:
    if family == PIECEWISE:
        return joint_points, None, None
    r, resid = fit_fan_exponent(joint_points)
    return parametric_grid(joint_points, r), r, resid


def _machine_tuple(
    node_class_name: str,
    machine: SpecPowerMachine | ArmSbcMachine,
    mapping: CapacityMapping,
    family: str,
    transition: TransitionMeasurement | None,
    *,
    image_pull_mib_per_s: float = 25.0,
) -> MachineTuple:
    """The raw seal-producing tuple constructor — **private** (R3-2): it derives
    everything from the supplied source-entry object, so a caller-fabricated entry
    object would be sealed as-is. The public factory is
    :func:`machine_tuple_from_registry`, which resolves the entry from the VERIFIED
    registry by id.

    Classes without a transition entry are non-suspendable in-scenario (plan §3.4:
    classes without published transition data never suspend), which
    ``NodeClassSpec.suspendable`` enforces via the all-four-params rule.
    """
    curve, r, resid = _curve_for_family(machine.joint_curve_points(), family)
    anchor = mapping.anchor_used(machine)
    fields = {
        "node_class_name": node_class_name,
        "machine_id": machine.machine_id,
        "curve_family": family,
        "power_curve_points": curve,
        "capacity_millicores": mapping.capacity_millicores(machine),
        "capacity_rule": mapping.rule_used(machine),
        "capacity_anchor": anchor,
        "transition_entry_id": transition.entry_id if transition else None,
        "suspend_energy_j": transition.suspend_energy_j if transition else None,
        "suspend_latency_s": transition.suspend_s if transition else None,
        "wake_energy_j": transition.wake_energy_j if transition else None,
        "wake_latency_s": transition.wake_s if transition else None,
        "suspended_power_w": transition.suspended_power_w if transition else 0.0,
        "image_pull_mib_per_s": float(image_pull_mib_per_s),
        "fit_exponent_r": r,
        "fit_max_abs_residual_w": resid,
    }
    payload = dict(fields)
    payload["power_curve_points"] = [list(p) for p in curve]
    payload["capacity_anchor"] = anchor.payload() if anchor else None
    return MachineTuple(**fields, seal=_machine_seal(payload))


def machine_tuple_from_registry(
    registry: SourceRegistry,
    mapping: CapacityMapping,
    *,
    node_class_name: str,
    machine_id: str,
    curve_family: str,
    transition_entry_id: str | None = None,
    image_pull_mib_per_s: float = 25.0,
) -> MachineTuple:
    """The registry-backed public tuple factory (R2-5/R3-2): callers supply entry
    *ids*; the machine and transition entries resolve from the VERIFIED registry and
    the mapping is registry-validated — no path seals caller-supplied source values."""
    registry.require_canonical()
    verified = registry.verified()
    machine_entries = verified.specpower | verified.arm_sbc
    validate_mapping(mapping, machine_entries)  # R2-8
    machine = machine_entries.get(machine_id)
    if machine is None:
        raise ValueError(
            f"machine entry {machine_id!r} is not a VERIFIED registry entry "
            "(provisional/unknown entries never enter tuples)"
        )
    transition = None
    if transition_entry_id is not None:
        transition = verified.transitions.get(transition_entry_id)
        if transition is None:
            raise ValueError(
                f"transition entry {transition_entry_id!r} is not a VERIFIED registry entry"
            )
    return _machine_tuple(
        node_class_name,
        machine,
        mapping,
        curve_family,
        transition,
        image_pull_mib_per_s=image_pull_mib_per_s,
    )


# ---------------------------------------------------------------- bundle
def _bundle_manifest(
    *,
    bundle_id: str,
    role: str,
    curve_family: str,
    machines: tuple[MachineTuple, ...],
    cloud_entry_id: str,
    cloud_j_per_vcpu_hour: float,
    cloud_swept_j_per_vcpu_hour: tuple[float, float],
    network_entry_id: str,
    network_j_per_byte: float,
    network_swept_j_per_byte: tuple[float, float],
    cloud_floor_vcpu: float,
    include_image_pull_bytes: bool,
    notes: str,
    source_records: tuple[EnergySourceRecord, ...],
) -> dict:
    """The complete canonical bundle description: every number a simulator run
    depends on (curve points, capacities, capacity anchors, transitions, fitted
    exponents, image-pull rates, swept ranges, allocation coefficients, the
    image-pull-bytes accounting toggle, role) plus the structured source records
    (R2-2/R2-6). Versioned by an explicit schema tag. This is what P2 pins
    (``holdout_gate.bundle_list_sha256``) and what the bundle seal covers."""
    return {
        "schema": MANIFEST_SCHEMA,
        "bundle_id": bundle_id,
        "role": role,
        "curve_family": curve_family,
        "machines": [{"payload": m.payload(), "seal": m.seal} for m in machines],
        "cloud_entry_id": cloud_entry_id,
        "cloud_j_per_vcpu_hour": cloud_j_per_vcpu_hour,
        "cloud_swept_j_per_vcpu_hour": list(cloud_swept_j_per_vcpu_hour),
        "network_entry_id": network_entry_id,
        "network_j_per_byte": network_j_per_byte,
        "network_swept_j_per_byte": list(network_swept_j_per_byte),
        "cloud_floor_vcpu": cloud_floor_vcpu,
        "include_image_pull_bytes": include_image_pull_bytes,
        "notes": notes,
        "source_records": [dataclasses.asdict(r) for r in source_records],
    }


def _bundle_seal(manifest: dict) -> str:
    return hashlib.sha256((_BUNDLE_SEAL_TAG + _canonical(manifest)).encode()).hexdigest()


@dataclass(frozen=True)
class EnergyModelBundle:
    """Jointly-sampled whole-machine tuples + allocation coefficients + transitions.

    ``role`` is ``"development"`` or ``"held-out-structural"``; the sanctioned
    held-out evaluation path is the run-once driver (``holdout_gate.py``) — a
    procedural control, per the module docstring. Only :func:`bundle_from_registry`
    (the registry-backed public factory, R2-5) produces bundles for use; the raw
    assembler is private. Consumption methods re-verify the seal."""

    bundle_id: str
    role: str
    curve_family: str
    machines: tuple[MachineTuple, ...]
    cloud_entry_id: str
    cloud_j_per_vcpu_hour: float
    cloud_swept_j_per_vcpu_hour: tuple[float, float]
    network_entry_id: str
    network_j_per_byte: float
    network_swept_j_per_byte: tuple[float, float]
    cloud_floor_vcpu: float = 0.0
    include_image_pull_bytes: bool = True
    notes: str = ""
    source_records: tuple[EnergySourceRecord, ...] = field(default_factory=tuple)
    seal: str = ""

    def __post_init__(self) -> None:
        _req_str("bundle_id", self.bundle_id)
        _req_str("cloud_entry_id", self.cloud_entry_id)
        _req_str("network_entry_id", self.network_entry_id)
        if not isinstance(self.notes, str):
            raise ValueError(f"{self.bundle_id}: notes must be a string")
        if self.role not in ("development", "held-out-structural"):
            raise ValueError(f"unknown bundle role {self.role!r}")
        if self.curve_family not in _FAMILIES:
            raise ValueError(f"unknown curve family {self.curve_family!r}")
        if not self.machines:
            raise ValueError(f"{self.bundle_id}: a bundle needs at least one machine tuple")
        names = [m.node_class_name for m in self.machines]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.bundle_id}: duplicate node class names")
        if any(m.curve_family != self.curve_family for m in self.machines):
            raise ValueError(f"{self.bundle_id}: mixed curve families inside one bundle")
        for lo, hi, point in (
            (*self.cloud_swept_j_per_vcpu_hour, self.cloud_j_per_vcpu_hour),
            (*self.network_swept_j_per_byte, self.network_j_per_byte),
        ):
            _finite("swept low", lo), _finite("swept high", hi), _finite("point", point)
            if not (0 < lo <= point <= hi):
                raise ValueError(f"{self.bundle_id}: swept range must bracket the point estimate")
        if _finite("cloud_floor_vcpu", self.cloud_floor_vcpu) < 0:
            raise ValueError(f"{self.bundle_id}: cloud_floor_vcpu must be >= 0 (R2-6)")
        if not isinstance(self.include_image_pull_bytes, bool):
            raise ValueError(f"{self.bundle_id}: include_image_pull_bytes must be a bool")
        if not all(isinstance(r, EnergySourceRecord) for r in self.source_records):
            raise ValueError(f"{self.bundle_id}: source_records must be EnergySourceRecord")
        cited = {m.machine_id for m in self.machines}
        cited |= {m.transition_entry_id for m in self.machines if m.transition_entry_id}
        cited |= {self.cloud_entry_id, self.network_entry_id}
        recorded = {r.entry_id for r in self.source_records}
        if recorded != cited:
            raise ValueError(
                f"{self.bundle_id}: source_records must cover exactly the cited "
                f"entries (R2-2): cited {sorted(cited)}, recorded {sorted(recorded)}"
            )
        if not self.seal:
            raise ValueError(
                "EnergyModelBundle carries no seal — construct it through "
                "bundle_from_registry() (the registry-backed public factory)"
            )

    # -- integrity ------------------------------------------------------------
    def manifest(self) -> dict:
        return _bundle_manifest(
            bundle_id=self.bundle_id,
            role=self.role,
            curve_family=self.curve_family,
            machines=self.machines,
            cloud_entry_id=self.cloud_entry_id,
            cloud_j_per_vcpu_hour=self.cloud_j_per_vcpu_hour,
            cloud_swept_j_per_vcpu_hour=self.cloud_swept_j_per_vcpu_hour,
            network_entry_id=self.network_entry_id,
            network_j_per_byte=self.network_j_per_byte,
            network_swept_j_per_byte=self.network_swept_j_per_byte,
            cloud_floor_vcpu=self.cloud_floor_vcpu,
            include_image_pull_bytes=self.include_image_pull_bytes,
            notes=self.notes,
            source_records=self.source_records,
        )

    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.manifest()).encode()).hexdigest()

    def verify_seal(self) -> None:
        for m in self.machines:
            m.verify_seal()
        if self.seal != _bundle_seal(self.manifest()):
            raise ValueError(
                f"{self.bundle_id}: bundle seal mismatch — fields were changed after "
                "assembly (joint-tuple / coefficient discipline violated)"
            )

    def machine_ids(self) -> frozenset[str]:
        """Return source-machine ids only after re-verifying the complete bundle.

        Structural-separation checks consume these ids, so this accessor is a
        scientific consumption boundary just like ``node_class_specs()``: a
        capacity-edited tuple or manifest-invalid bundle must fail here rather than
        contribute ids to an apparently valid split (R5-1/R1-3).
        """
        self.verify_seal()
        return frozenset(m.machine_id for m in self.machines)

    def node_class_specs(self) -> dict[str, NodeClassSpec]:
        self.verify_seal()
        return {m.node_class_name: m.node_class_spec() for m in self.machines}

    def node_class_bindings(self) -> dict[str, NodeClassBinding]:
        """The seal-verifying cluster-construction bridge (R2-5): per node class, the
        Phase 2 spec *and* its jointly-derived capacity from one verification."""
        self.verify_seal()
        return {m.node_class_name: m.node_class_binding() for m in self.machines}

    def energy_coefficients(self) -> EnergyCoefficients:
        """The Phase 2 coefficient object, with the §4.3 provenance anchor populated
        from the pinned entries (replacing the Phase 2 development placeholder).
        Verifies the seal first; carries the bundle-set label, the manifest hash, and
        the structured source records (R2-2) so every downstream ProvenanceRecord
        names the bundle it survives and the entries behind it."""
        self.verify_seal()
        return EnergyCoefficients(
            version=f"bundle:{self.bundle_id}",
            cloud_j_per_vcpu_hour=self.cloud_j_per_vcpu_hour,
            network_j_per_byte=self.network_j_per_byte,
            include_image_pull_bytes=self.include_image_pull_bytes,
            cloud_floor_vcpu=self.cloud_floor_vcpu,
            source_records=self.source_records,
            source=(
                f"EnergyModelBundle {self.bundle_id} ({self.role}); machines: "
                + ", ".join(sorted(self.machine_ids()))
                + f"; cloud allocation: {self.cloud_entry_id}; "
                + f"network allocation: {self.network_entry_id}; "
                + f"bundle manifest sha256: {self.manifest_sha256()}"
            ),
            parameter_ranges=(
                "swept (not CI): cloud "
                f"{self.cloud_swept_j_per_vcpu_hour[0]:.6g}-"
                f"{self.cloud_swept_j_per_vcpu_hour[1]:.6g} J/vCPUh; network "
                f"{self.network_swept_j_per_byte[0]:.6g}-"
                f"{self.network_swept_j_per_byte[1]:.6g} J/B"
            ),
            bundle_set=f"{self.role}:{self.bundle_id}",
        )


def _assemble_bundle(
    *,
    bundle_id: str,
    role: str,
    curve_family: str,
    machines: tuple[MachineTuple, ...],
    cloud_entry_id: str,
    cloud_j_per_vcpu_hour: float,
    cloud_swept_j_per_vcpu_hour: tuple[float, float],
    network_entry_id: str,
    network_j_per_byte: float,
    network_swept_j_per_byte: tuple[float, float],
    cloud_floor_vcpu: float = 0.0,
    include_image_pull_bytes: bool = True,
    notes: str = "",
    source_records: tuple[EnergySourceRecord, ...] = (),
) -> EnergyModelBundle:
    """The raw seal-producing assembler — **private** (R2-5): it seals whatever it is
    given. The public factory is :func:`bundle_from_registry`, which accepts registry
    entry ids and resolves every coefficient value internally."""
    manifest = _bundle_manifest(
        bundle_id=bundle_id,
        role=role,
        curve_family=curve_family,
        machines=machines,
        cloud_entry_id=cloud_entry_id,
        cloud_j_per_vcpu_hour=cloud_j_per_vcpu_hour,
        cloud_swept_j_per_vcpu_hour=cloud_swept_j_per_vcpu_hour,
        network_entry_id=network_entry_id,
        network_j_per_byte=network_j_per_byte,
        network_swept_j_per_byte=network_swept_j_per_byte,
        cloud_floor_vcpu=cloud_floor_vcpu,
        include_image_pull_bytes=include_image_pull_bytes,
        notes=notes,
        source_records=source_records,
    )
    return EnergyModelBundle(
        bundle_id=bundle_id,
        role=role,
        curve_family=curve_family,
        machines=machines,
        cloud_entry_id=cloud_entry_id,
        cloud_j_per_vcpu_hour=cloud_j_per_vcpu_hour,
        cloud_swept_j_per_vcpu_hour=cloud_swept_j_per_vcpu_hour,
        network_entry_id=network_entry_id,
        network_j_per_byte=network_j_per_byte,
        network_swept_j_per_byte=network_swept_j_per_byte,
        cloud_floor_vcpu=cloud_floor_vcpu,
        include_image_pull_bytes=include_image_pull_bytes,
        notes=notes,
        source_records=source_records,
        seal=_bundle_seal(manifest),
    )


def bundle_from_registry(
    registry: SourceRegistry,
    mapping: CapacityMapping,
    *,
    bundle_id: str,
    role: str,
    curve_family: str,
    machines: tuple[tuple[str, str, str | None], ...],
    cloud_entry_id: str,
    network_entry_id: str,
    cloud_floor_vcpu: float = 0.0,
    include_image_pull_bytes: bool = True,
    image_pull_mib_per_s: float = 25.0,
    notes: str = "",
) -> EnergyModelBundle:
    """The registry-backed public bundle factory (R2-5): callers supply **entry ids**
    — ``machines`` is ``(node_class_name, machine_entry_id, transition_entry_id |
    None)`` triples plus the cloud/network allocation entry ids — and every
    coefficient value, curve point, capacity, and source record is resolved from the
    VERIFIED registry internally, then re-validated by :func:`validate_bundle`.
    There is no parameter through which a caller can inject a coefficient value."""
    registry.require_canonical()
    verified = registry.verified()
    machine_entries = verified.specpower | verified.arm_sbc
    validate_mapping(mapping, machine_entries)  # R2-8
    tuples = []
    records: dict[str, EnergySourceRecord] = {}
    for node_class_name, machine_id, transition_id in machines:
        machine = machine_entries.get(machine_id)
        if machine is None:
            raise ValueError(
                f"{bundle_id}: machine entry {machine_id!r} is not a VERIFIED "
                "registry entry (provisional/unknown entries never enter bundles)"
            )
        transition = None
        if transition_id is not None:
            transition = verified.transitions.get(transition_id)
            if transition is None:
                raise ValueError(
                    f"{bundle_id}: transition entry {transition_id!r} is not a "
                    "VERIFIED registry entry"
                )
            records[transition_id] = _source_record(
                transition_id, "transition", transition.provenance
            )
        records[machine_id] = _source_record(machine_id, "machine", machine.provenance)
        tuples.append(
            _machine_tuple(
                node_class_name,
                machine,
                mapping,
                curve_family,
                transition,
                image_pull_mib_per_s=image_pull_mib_per_s,
            )
        )
    cloud = verified.cloud_allocation.get(cloud_entry_id)
    if cloud is None:
        raise ValueError(
            f"{bundle_id}: cloud allocation entry {cloud_entry_id!r} is not a "
            "VERIFIED registry entry"
        )
    net = verified.network_allocation.get(network_entry_id)
    if net is None:
        raise ValueError(
            f"{bundle_id}: network allocation entry {network_entry_id!r} is not a "
            "VERIFIED registry entry"
        )
    records[cloud_entry_id] = _source_record(cloud_entry_id, "cloud-allocation", cloud.provenance)
    records[network_entry_id] = _source_record(
        network_entry_id, "network-allocation", net.provenance
    )
    bundle = _assemble_bundle(
        bundle_id=bundle_id,
        role=role,
        curve_family=curve_family,
        machines=tuple(tuples),
        cloud_entry_id=cloud.entry_id,
        cloud_j_per_vcpu_hour=cloud.point_j_per_vcpu_hour,
        cloud_swept_j_per_vcpu_hour=cloud.swept_j_per_vcpu_hour(),
        network_entry_id=net.entry_id,
        network_j_per_byte=net.j_per_byte,
        network_swept_j_per_byte=net.swept_j_per_byte(),
        cloud_floor_vcpu=cloud_floor_vcpu,
        include_image_pull_bytes=include_image_pull_bytes,
        notes=notes,
        source_records=tuple(sorted(records.values(), key=lambda r: (r.kind, r.entry_id))),
    )
    validate_bundle(bundle, registry, mapping)
    return bundle


# ---------------------------------------------------------------- validation
def validate_bundle(
    bundle: EnergyModelBundle, registry: SourceRegistry, mapping: CapacityMapping
) -> None:
    """Registry-grounded validator: every tuple must be exactly re-derivable from
    its cited source entry (joint-tuple discipline — no impossible devices), every
    entry a default-set bundle cites must be VERIFIED, the capacity anchors must be
    registry-backed (R2-8), and the structured source records must match the cited
    entries' provenance (R2-2). Also verifies the seals."""
    registry.require_canonical()
    bundle.verify_seal()
    entries = registry.all_entries()
    validate_mapping(mapping, registry.specpower | registry.arm_sbc)  # R2-8
    expected_records: dict[str, EnergySourceRecord] = {}
    for m in bundle.machines:
        src = entries.get(m.machine_id)
        if src is None:
            raise ValueError(f"{bundle.bundle_id}: unknown machine {m.machine_id}")
        if not isinstance(src, SpecPowerMachine | ArmSbcMachine):
            raise ValueError(f"{bundle.bundle_id}: {m.machine_id} is not a machine entry")
        if not src.provenance.is_verified:
            raise ValueError(
                f"{bundle.bundle_id}: machine {m.machine_id} is not VERIFIED "
                "(provisional entries are excluded from bundle sets)"
            )
        transition = None
        if m.transition_entry_id is not None:
            transition = registry.transitions.get(m.transition_entry_id)
            if transition is None:
                raise ValueError(
                    f"{bundle.bundle_id}: unknown transition entry {m.transition_entry_id}"
                )
            if not transition.provenance.is_verified:
                raise ValueError(
                    f"{bundle.bundle_id}: transition {m.transition_entry_id} is not VERIFIED"
                )
            expected_records[m.transition_entry_id] = _source_record(
                m.transition_entry_id, "transition", transition.provenance
            )
        expected_records[m.machine_id] = _source_record(m.machine_id, "machine", src.provenance)
        expected = _machine_tuple(
            m.node_class_name,
            src,
            mapping,
            m.curve_family,
            transition,
            image_pull_mib_per_s=m.image_pull_mib_per_s,
        )
        if expected != m:
            raise ValueError(
                f"{bundle.bundle_id}: tuple {m.node_class_name} is not re-derivable from "
                f"{m.machine_id} — joint-tuple discipline violated"
            )
        m.node_class_spec()  # W2 spec-level constraints must hold
    cloud = registry.cloud_allocation.get(bundle.cloud_entry_id)
    if cloud is None or not cloud.provenance.is_verified:
        raise ValueError(f"{bundle.bundle_id}: cloud allocation entry missing or unverified")
    if bundle.cloud_j_per_vcpu_hour != cloud.point_j_per_vcpu_hour or (
        bundle.cloud_swept_j_per_vcpu_hour != cloud.swept_j_per_vcpu_hour()
    ):
        raise ValueError(f"{bundle.bundle_id}: cloud coefficients diverge from {cloud.entry_id}")
    net = registry.network_allocation.get(bundle.network_entry_id)
    if net is None or not net.provenance.is_verified:
        raise ValueError(f"{bundle.bundle_id}: network allocation entry missing or unverified")
    if bundle.network_j_per_byte != net.j_per_byte or (
        bundle.network_swept_j_per_byte != net.swept_j_per_byte()
    ):
        raise ValueError(f"{bundle.bundle_id}: network coefficients diverge from {net.entry_id}")
    expected_records[cloud.entry_id] = _source_record(
        cloud.entry_id, "cloud-allocation", cloud.provenance
    )
    expected_records[net.entry_id] = _source_record(
        net.entry_id, "network-allocation", net.provenance
    )
    if set(bundle.source_records) != set(expected_records.values()):
        raise ValueError(
            f"{bundle.bundle_id}: structured source records diverge from the cited "
            "registry entries' provenance (R2-2)"
        )


def validate_split(
    development: tuple[EnergyModelBundle, ...], held_out: tuple[EnergyModelBundle, ...]
) -> None:
    """Fail-closed, registration-anchored T4 structural-split validator.

    The authoritative development/held-out bundle ids, manifest hashes, and
    complete node-class rosters come from the content-sealed registration artifact,
    not from runtime calls to the replaceable bundle builders (R7-1). Caller
    subsets or self-consistent reduced rosters therefore cannot define what
    "complete" means. Every supplied bundle is independently validated against the
    canonical source registry and must match its registered manifest before
    machine, curve-family, and allocation-entry separation is checked.
    """
    if not development or not held_out:
        raise ValueError("development and held-out bundle sets must both be non-empty")

    registration = load_energy_analysis_registration()
    registration.verify_seal()
    registry = load_pinned_sources()
    mapping = default_capacity_mapping(registry)
    supplied_by_role = {
        "development": development,
        "held-out-structural": held_out,
    }

    for role, supplied in supplied_by_role.items():
        registered_manifests = dict(registration.bundle_manifests(role))
        expected_ids = set(registered_manifests)
        expected_roster = frozenset(registration.node_class_roster(role))
        supplied_ids = [bundle.bundle_id for bundle in supplied]
        if len(supplied_ids) != len(set(supplied_ids)):
            raise ValueError(f"{role} bundle ids must be unique")
        if set(supplied_ids) != expected_ids:
            raise ValueError(
                f"{role} bundle ids must equal the canonical registered set: "
                f"expected {sorted(expected_ids)}, got {sorted(supplied_ids)}"
            )

        for bundle in supplied:
            if bundle.role != role:
                raise ValueError(f"{bundle.bundle_id}: role must be {role}")
            roster = frozenset(
                machine.node_class_name for machine in bundle.machines
            )
            if len(roster) != len(bundle.machines) or roster != expected_roster:
                raise ValueError(
                    f"{bundle.bundle_id}: node-class roster must equal the canonical "
                    f"complete roster {sorted(expected_roster)}, got {sorted(roster)}"
                )
            # Source validity and registration identity are independent checks:
            # builders are not consulted for either authority decision.
            validate_bundle(bundle, registry, mapping)
            if bundle.manifest_sha256() != registered_manifests[bundle.bundle_id]:
                raise ValueError(
                    f"{bundle.bundle_id}: manifest differs from the canonical "
                    "sealed split registration"
                )

    dev_machines = frozenset().union(*(b.machine_ids() for b in development))
    ho_machines = frozenset().union(*(b.machine_ids() for b in held_out))
    if dev_machines & ho_machines:
        raise ValueError(f"split violation: shared machines {sorted(dev_machines & ho_machines)}")
    dev_fam = {b.curve_family for b in development}
    ho_fam = {b.curve_family for b in held_out}
    if dev_fam & ho_fam:
        raise ValueError(f"split violation: shared curve families {sorted(dev_fam & ho_fam)}")
    dev_alloc = {b.cloud_entry_id for b in development} | {b.network_entry_id for b in development}
    ho_alloc = {b.cloud_entry_id for b in held_out} | {b.network_entry_id for b in held_out}
    if dev_alloc & ho_alloc:
        raise ValueError(
            f"split violation: shared allocation entries {sorted(dev_alloc & ho_alloc)}"
        )


# ---------------------------------------------------------------- canonical bundle builders
# Inputs used to prepare the sealed split/design registration. Changing these
# builders alone cannot change confirmatory authority; a new registered payload and
# canonical hash are required (R7-1).
# Development machines: AMD-EPYC-era and low-power-Xeon machines, piecewise family.
# Held-out machines: different vendors and CPU families (Intel Xeon Platinum / X5675),
# parametric family, different allocation entries.
DEV_REFERENCE_MACHINE = "specpower-fujitsu-rx100s7-2011"
DEV_REFERENCE_CAPACITY_MILLICORES = 4000  # 4 enabled cores x 1000 mc (declared anchor)
# HPL-domain anchor (far-edge boards; R1-1): the RPi 3B's published 3.7 GFLOPS
# (stock cooling, Cloutier et al. Table 2) is anchored at 4 cores x 1000 mc — the
# declared constant tying the HPL domain to capacity units. Other HPL-domain boards
# scale by their published GFLOPS ratio (Jetson TX-1: 16.0/3.7). HPL numbers are
# never compared with ssj_ops (capacity.py rejects cross-domain anchoring).
FAR_EDGE_REFERENCE_MACHINE = "armsbc-rpi3b-cloutier2016"
FAR_EDGE_REFERENCE_CAPACITY_MILLICORES = 4000

DEV_FOG_MACHINES: tuple[str, ...] = (
    "specpower-lenovo-sr665v3-2023",
    "specpower-supermicro-as1115sv-2023",
    "specpower-fujitsu-rx100s7-2011",
)
HELD_OUT_FOG_MACHINES: tuple[str, ...] = (
    "specpower-asus-esc4000e11-2023",
    "specpower-h3c-r6900g6-2023",
    "specpower-ibm-x3550m3-2011",
)
DEV_FAR_EDGE_MACHINES: tuple[str, ...] = ("armsbc-rpi3b-cloutier2016",)
HELD_OUT_FAR_EDGE_MACHINES: tuple[str, ...] = ("armsbc-jetsontx1-cloutier2016",)
# Transition entries are published class-level measurements; they apply only to the
# device class they plausibly describe (servers), and classes without published
# transition data are non-suspendable in-scenario (plan §3.4 -> None). The transition
# machines (Grid'5000 Dell R720/R630) are *not* the power-curve machines — no source
# publishes both; the composition is stated, and the joint-tuple discipline governs
# the (power curve, capacity) pair specifically (plan §4.1).
DEV_FOG_TRANSITION: str | None = "transition-taurus-ijhpca2018"
HELD_OUT_FOG_TRANSITION: str | None = "transition-paravance-ica3pp2016"
DEV_FAR_EDGE_TRANSITION: str | None = None  # no published SBC suspend measurements pinned
HELD_OUT_FAR_EDGE_TRANSITION: str | None = None
DEV_CLOUD_ALLOC = "ccf-aws-cpu"
HELD_OUT_CLOUD_ALLOC = "ccf-gcp-cpu"
DEV_NETWORK_ALLOC = "ccf-network"
HELD_OUT_NETWORK_ALLOC = "aslan2017-network"


def default_capacity_mapping(registry: SourceRegistry) -> CapacityMapping:
    """The draft two-domain mapping (frozen in P2): SPECpower domain anchored on the
    development reference machine; HPL domain anchored on the RPi 3B. Anchors are
    built by the registry-backed factory and validated against the registry (R2-8)."""
    registry.require_canonical()
    ref = registry.specpower[DEV_REFERENCE_MACHINE]
    mapping = CapacityMapping.anchored_on(ref, DEV_REFERENCE_CAPACITY_MILLICORES)
    far_edge_ref = registry.arm_sbc[FAR_EDGE_REFERENCE_MACHINE]
    mapping = mapping.with_anchor(
        DomainAnchor.from_machine(far_edge_ref, FAR_EDGE_REFERENCE_CAPACITY_MILLICORES)
    )
    validate_mapping(mapping, registry.specpower | registry.arm_sbc)
    return mapping


def _build_set(
    registry: SourceRegistry,
    mapping: CapacityMapping,
    *,
    role: str,
    family: str,
    prefix: str,
    fog_ids: tuple[str, ...],
    far_edge_ids: tuple[str, ...],
    fog_transition_id: str | None,
    far_edge_transition_id: str | None,
    cloud_id: str,
    network_id: str,
) -> tuple[EnergyModelBundle, ...]:
    bundles = []
    for i, fog_id in enumerate(fog_ids):
        far_edge_id = far_edge_ids[i % len(far_edge_ids)]
        bundles.append(
            bundle_from_registry(
                registry,
                mapping,
                bundle_id=f"{prefix}{i + 1}",
                role=role,
                curve_family=family,
                machines=(
                    ("fog-server", fog_id, fog_transition_id),
                    ("far-edge", far_edge_id, far_edge_transition_id),
                ),
                cloud_entry_id=cloud_id,
                network_entry_id=network_id,
            )
        )
    return tuple(bundles)


def build_development_bundles(
    registry: SourceRegistry, mapping: CapacityMapping | None = None
) -> tuple[EnergyModelBundle, ...]:
    """The draft development set (visible to evolution and tuning). VERIFIED entries
    only (enforced by ``bundle_from_registry``, which also re-validates every bundle
    against the registry); raises if a pinned entry is missing or unverified."""
    m = mapping or default_capacity_mapping(registry)
    return _build_set(
        registry,
        m,
        role="development",
        family=PIECEWISE,
        prefix="dev-",
        fog_ids=DEV_FOG_MACHINES,
        far_edge_ids=DEV_FAR_EDGE_MACHINES,
        fog_transition_id=DEV_FOG_TRANSITION,
        far_edge_transition_id=DEV_FAR_EDGE_TRANSITION,
        cloud_id=DEV_CLOUD_ALLOC,
        network_id=DEV_NETWORK_ALLOC,
    )


def build_held_out_bundles(
    registry: SourceRegistry,
    mapping: CapacityMapping | None = None,
    *,
    acknowledged_one_shot: bool = False,
) -> tuple[EnergyModelBundle, ...]:
    """The draft held-out structural set. Refuses casual access: the sanctioned
    callers are ``holdout_gate.py`` (run-once, tagged commit) and split-validation
    tooling that never evaluates policies on these bundles. This is a procedural
    honest-operator control (see the module docstring), not a technical barrier."""
    if not acknowledged_one_shot:
        raise PermissionError(
            "held-out bundles are evaluated once, by sim/energy/holdout_gate.py, after "
            "code and tuning freeze at a tagged commit (plan §4.1). Pass "
            "acknowledged_one_shot=True only from the gate driver or from split "
            "validation that does not evaluate policies."
        )
    m = mapping or default_capacity_mapping(registry)
    return _build_set(
        registry,
        m,
        role="held-out-structural",
        family=PARAMETRIC,
        prefix="holdout-",
        fog_ids=HELD_OUT_FOG_MACHINES,
        far_edge_ids=HELD_OUT_FAR_EDGE_MACHINES,
        fog_transition_id=HELD_OUT_FOG_TRANSITION,
        far_edge_transition_id=HELD_OUT_FAR_EDGE_TRANSITION,
        cloud_id=HELD_OUT_CLOUD_ALLOC,
        network_id=HELD_OUT_NETWORK_ALLOC,
    )
