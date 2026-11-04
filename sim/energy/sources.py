"""Pinned published sources for EnergyModelBundles (TODO 3.1; plan §4.1).

Every numeric entry in this layer carries a :class:`Provenance` record — source name,
URL/DOI, version/date accessed, the exact field transcribed, and a verification
status. The two statuses are:

- ``"VERIFIED"`` — the value was fetched from the cited public source at pinning time
  (the record says how; SPECpower entries are machine-parsed from the result HTML by
  ``data/fetch_specpower.py``, never hand-transcribed).
- ``"PROVISIONAL — needs verification"`` — the value could not be verified against its
  source at pinning time. Provisional entries are **excluded from the default bundle
  sets** (:func:`SourceRegistry.verified`); they exist only so the machinery can be
  exercised and the verification debt is explicit.

Joint-tuple discipline (plan §4.1): SPECpower load levels are consumed as joint
(throughput, power) points per published machine. :class:`SpecPowerMachine` keeps the
whole row set together; there is no API that returns a power curve detached from the
machine's own throughput/capacity identity — bundle assembly (``bundles.py``) builds
both the power curve and the capacity from the *same* entry.

Allocation coefficients (CCF/Teads, network intensity) are point-estimate allocation
models. Sources that publish only a point estimate get a **swept range** chosen at
pinning time and labeled a sweep — never a confidence interval (plan §4.1: no CIs
invented for sources that publish none).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

VERIFIED = "VERIFIED"
PROVISIONAL = "PROVISIONAL — needs verification"
_STATUSES = (VERIFIED, PROVISIONAL)
_REGISTRY_SEAL_TAG = "pinned-source-registry-v1:"
_LOADER_AUTHORITY = object()


def _req_str(name: str, v: object) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty string")
    return v


def _finite(name: str, v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def _pos(name: str, v: object) -> float:
    out = _finite(name, v)
    if out <= 0:
        raise ValueError(f"{name} must be > 0")
    return out


def _nonneg(name: str, v: object) -> float:
    out = _finite(name, v)
    if out < 0:
        raise ValueError(f"{name} must be >= 0")
    return out


# ------------------------------------------------------------------ provenance
@dataclass(frozen=True)
class Provenance:
    """The non-negotiable per-number provenance record (TODO 3.1)."""

    source_name: str
    url: str | None
    doi: str | None
    version_or_date_accessed: str
    field_transcribed: str
    status: str
    verification_method: str
    notes: str = ""
    #: sha256 of the retained raw source artifact (fetched page/PDF), when one exists.
    #: Carried end-to-end so downstream provenance records can cite the content hash.
    content_sha256: str | None = None
    #: path of the retained raw artifact, relative to ``sim/energy/data/``. When set,
    #: :func:`load_pinned_sources` re-hashes the file and refuses a mismatch with
    #: ``content_sha256`` (R2-2: retained hashes are *verified* at load, not asserted).
    artifact_path: str | None = None

    def __post_init__(self) -> None:
        _req_str("provenance.source_name", self.source_name)
        _req_str("provenance.version_or_date_accessed", self.version_or_date_accessed)
        _req_str("provenance.field_transcribed", self.field_transcribed)
        _req_str("provenance.verification_method", self.verification_method)
        if self.url is None and self.doi is None:
            raise ValueError("provenance needs a URL or a DOI")
        if self.status not in _STATUSES:
            raise ValueError(f"provenance.status must be one of {_STATUSES}, got {self.status!r}")
        if self.artifact_path is not None:
            if self.content_sha256 is None:
                raise ValueError(
                    "a retained artifact_path requires its content_sha256 (nothing to verify)"
                )
            # R3-6: retained artifacts live under the pinned data directory only —
            # absolute or parent-traversing paths could name a file outside it.
            ap = Path(self.artifact_path)
            if ap.is_absolute() or ".." in ap.parts:
                raise ValueError(
                    f"artifact_path {self.artifact_path!r} must be a plain relative "
                    "path under the pinned data directory (R3-6)"
                )

    @property
    def is_verified(self) -> bool:
        return self.status == VERIFIED

    @staticmethod
    def from_dict(d: dict) -> Provenance:
        return Provenance(
            source_name=d["source_name"],
            url=d.get("url"),
            doi=d.get("doi"),
            version_or_date_accessed=d["version_or_date_accessed"],
            field_transcribed=d["field_transcribed"],
            status=d["status"],
            verification_method=d["verification_method"],
            notes=d.get("notes", ""),
            content_sha256=d.get("content_sha256"),
            artifact_path=d.get("artifact_path"),
        )

    def verify_retained_artifact(self, data_dir: Path) -> None:
        """Re-hash the retained raw artifact (if one is named) and refuse a mismatch
        with the recorded ``content_sha256`` (R2-2). Missing files fail loudly too —
        a record must never cite an artifact the repository does not retain."""
        if self.artifact_path is None:
            return
        path = (data_dir / self.artifact_path).resolve()
        if not path.is_relative_to(data_dir.resolve()):  # R3-6: containment, resolved
            raise ValueError(
                f"retained artifact {self.artifact_path!r} escapes the pinned data "
                f"directory {data_dir} (R3-6)"
            )
        if not path.is_file():
            raise ValueError(
                f"provenance cites retained artifact {self.artifact_path!r} but the "
                f"file does not exist under {data_dir}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.content_sha256:
            raise ValueError(
                f"retained artifact {self.artifact_path!r} hash mismatch: recorded "
                f"{self.content_sha256}, file hashes to {actual} — the artifact was "
                "modified after pinning or the record is wrong"
            )


# ------------------------------------------------------------------ SPECpower
@dataclass(frozen=True)
class SpecPowerLoadPoint:
    """One published joint row: target/actual load, throughput, average power."""

    target_load_frac: float  # 0.1 .. 1.0
    actual_load_frac: float
    ssj_ops: float
    avg_active_power_w: float

    def __post_init__(self) -> None:
        for n in ("target_load_frac", "actual_load_frac"):
            object.__setattr__(self, n, _pos(n, getattr(self, n)))
        object.__setattr__(self, "ssj_ops", _pos("ssj_ops", self.ssj_ops))
        object.__setattr__(
            self, "avg_active_power_w", _pos("avg_active_power_w", self.avg_active_power_w)
        )


@dataclass(frozen=True)
class SpecPowerMachine:
    """A whole published machine: the joint load-level table plus active idle.

    The load levels and the idle point stay together — a bundle that uses this
    machine's power curve necessarily uses this machine's throughput for capacity.
    """

    machine_id: str
    vendor: str
    model: str
    cpu: str
    cpu_enabled: str
    memory_gb: str
    test_date: str
    hardware_availability: str
    publication: str
    overall_ssj_ops_per_watt: float
    load_points: tuple[SpecPowerLoadPoint, ...]
    active_idle_w: float
    provenance: Provenance

    def __post_init__(self) -> None:
        _req_str("machine_id", self.machine_id)
        object.__setattr__(self, "active_idle_w", _pos("active_idle_w", self.active_idle_w))
        pts = tuple(self.load_points)
        object.__setattr__(self, "load_points", pts)
        if len(pts) != 10:
            raise ValueError(f"{self.machine_id}: SPECpower reports 10 load levels, got {len(pts)}")
        fracs = [p.target_load_frac for p in pts]
        if fracs != sorted(set(fracs)) or fracs[0] != 0.1 or fracs[-1] != 1.0:
            raise ValueError(f"{self.machine_id}: target loads must ascend 0.1..1.0")
        ops = [p.ssj_ops for p in pts]
        if ops != sorted(set(ops)):
            raise ValueError(f"{self.machine_id}: ssj_ops must strictly ascend with load")
        if self.active_idle_w >= min(p.avg_active_power_w for p in pts):
            raise ValueError(
                f"{self.machine_id}: active idle power must be below every load-level power"
            )

    #: Benchmark domain of the published throughput (capacity anchoring never crosses
    #: domains — HPL GFLOPS and ssj_ops are never numerically compared).
    benchmark_domain: str = "specpower-ssj2008"

    @property
    def peak_ssj_ops(self) -> float:
        return self.load_points[-1].ssj_ops

    @property
    def max_power_w(self) -> float:
        return max(p.avg_active_power_w for p in self.load_points)

    def joint_curve_points(self) -> tuple[tuple[float, float], ...]:
        """(normalized throughput, watts) with the idle point at 0.0 — the joint
        whole-machine curve.

        The load coordinate is the row's **measured throughput fraction**
        ``ssj_ops / peak_ssj_ops`` — the joint (throughput, power) point exactly as
        published (plan §4.1), not the nominal target-load label (the two differ by up
        to ~1.5 percentage points in the pinned set). Treating this throughput
        fraction as the utilization coordinate of the node power curve is the T4
        modeling assumption, stated in ``capacity.py``.
        """
        peak = self.peak_ssj_ops
        return ((0.0, self.active_idle_w),) + tuple(
            (p.ssj_ops / peak, p.avg_active_power_w) for p in self.load_points
        )


# ------------------------------------------------------------------ ARM SBC
@dataclass(frozen=True)
class ArmSbcMachine:
    """A peer-reviewed ARM SBC power characterization (far-edge node class).

    ``load_points`` are (load fraction in [0,1], watts) as published — the load
    coordinate's published meaning is recorded in ``load_metric``.
    """

    machine_id: str
    board: str
    cpu: str
    cores: int
    load_metric: str
    load_points: tuple[tuple[float, float], ...]
    active_idle_w: float
    peak_throughput_ops: float | None  # None: source publishes no throughput point
    #: Benchmark domain of the published throughput point (e.g. "hpl-gflops-cpu-dp").
    #: Required whenever a throughput point is present: capacity anchoring is only
    #: valid within one benchmark domain (HPL GFLOPS are never compared with ssj_ops).
    benchmark_domain: str | None
    provenance: Provenance

    def __post_init__(self) -> None:
        _req_str("machine_id", self.machine_id)
        _req_str("load_metric", self.load_metric)
        if not isinstance(self.cores, int) or isinstance(self.cores, bool) or self.cores < 1:
            raise ValueError(f"{self.machine_id}: cores must be a positive int")
        object.__setattr__(self, "active_idle_w", _pos("active_idle_w", self.active_idle_w))
        pts = tuple((_nonneg("load", u), _pos("watts", w)) for u, w in self.load_points)
        object.__setattr__(self, "load_points", pts)
        if len(pts) < 1:
            raise ValueError(f"{self.machine_id}: need at least one loaded power point")
        us = [u for u, _ in pts]
        if us != sorted(set(us)) or us[-1] != 1.0 or (us and us[0] == 0.0):
            raise ValueError(
                f"{self.machine_id}: load points must strictly ascend and end at 1.0; "
                "the idle point is supplied separately as active_idle_w"
            )
        if self.peak_throughput_ops is not None:
            object.__setattr__(
                self, "peak_throughput_ops", _pos("peak_throughput_ops", self.peak_throughput_ops)
            )
        if (self.peak_throughput_ops is None) != (self.benchmark_domain is None):
            raise ValueError(
                f"{self.machine_id}: peak_throughput_ops and benchmark_domain must be "
                "supplied together — a throughput point is only usable within its "
                "published benchmark domain"
            )
        if self.benchmark_domain is not None:
            _req_str("benchmark_domain", self.benchmark_domain)

    @property
    def max_power_w(self) -> float:
        return max(w for _, w in self.load_points)

    def joint_curve_points(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, self.active_idle_w),) + self.load_points


# ------------------------------------------------------------------ transitions
@dataclass(frozen=True)
class TransitionMeasurement:
    """Published suspend/wake (or off/on) transition measurements (Orgerie lineage).

    Raw published fields are kept as published (durations, powers); the derived
    per-transition energies used by the simulator are computed by a stated formula and
    recorded next to the raw fields (``derivation``), so the transformation is
    auditable rather than baked in.
    """

    entry_id: str
    device_described: str
    suspend_s: float
    suspend_energy_j: float
    wake_s: float
    wake_energy_j: float
    suspended_power_w: float  # draw while in the low-power state (0.0 = fully off)
    derivation: str
    provenance: Provenance

    def __post_init__(self) -> None:
        _req_str("entry_id", self.entry_id)
        _req_str("device_described", self.device_described)
        _req_str("derivation", self.derivation)
        for n in ("suspend_s", "suspend_energy_j", "wake_s", "wake_energy_j", "suspended_power_w"):
            object.__setattr__(self, n, _nonneg(n, getattr(self, n)))


# ------------------------------------------------------------------ allocation
@dataclass(frozen=True)
class CloudAllocationSource:
    """Per-vCPU allocation coefficients (CCF/Teads lineage): W per vCPU at min/max
    utilization. Point estimates; ``swept_range_w_per_vcpu_hour`` is the pre-registered
    sweep for sensitivity analysis, not a confidence interval."""

    entry_id: str
    min_w_per_vcpu: float
    max_w_per_vcpu: float
    assumed_allocation_utilization: float
    swept_range_w_per_vcpu: tuple[float, float]
    source_publishes_range: bool
    provenance: Provenance

    def __post_init__(self) -> None:
        _req_str("entry_id", self.entry_id)
        object.__setattr__(self, "min_w_per_vcpu", _pos("min_w_per_vcpu", self.min_w_per_vcpu))
        object.__setattr__(self, "max_w_per_vcpu", _pos("max_w_per_vcpu", self.max_w_per_vcpu))
        if self.max_w_per_vcpu < self.min_w_per_vcpu:
            raise ValueError(f"{self.entry_id}: max W/vCPU below min")
        u = _finite("assumed_allocation_utilization", self.assumed_allocation_utilization)
        if not 0.0 <= u <= 1.0:
            raise ValueError("assumed_allocation_utilization must be in [0,1]")
        lo, hi = self.swept_range_w_per_vcpu
        lo, hi = _pos("swept low", lo), _pos("swept high", hi)
        object.__setattr__(self, "swept_range_w_per_vcpu", (lo, hi))
        if not lo <= self.point_w_per_vcpu <= hi:
            raise ValueError(f"{self.entry_id}: swept range must contain the point estimate")

    @property
    def point_w_per_vcpu(self) -> float:
        """The allocation point estimate: linear between published min and max W/vCPU
        at the assumed allocation utilization (the CCF usage of the Teads figures)."""
        u = self.assumed_allocation_utilization
        return self.min_w_per_vcpu + (self.max_w_per_vcpu - self.min_w_per_vcpu) * u

    @property
    def point_j_per_vcpu_hour(self) -> float:
        return self.point_w_per_vcpu * 3600.0

    def swept_j_per_vcpu_hour(self) -> tuple[float, float]:
        lo, hi = self.swept_range_w_per_vcpu
        return lo * 3600.0, hi * 3600.0


@dataclass(frozen=True)
class NetworkAllocationSource:
    """Per-byte network allocation coefficient (kWh/GB lineage). Point estimate with a
    pre-registered swept range (a sweep, not a CI).

    ``path_scope_published`` records the physical path scope the source motivates its
    coefficient for (e.g. inter-data-center vs. Internet core + access). The simulator
    applies one coefficient to *every* cross-node transfer — a **synthetic all-transfer
    allocation assumption** that does not match any single published scope; the
    assumption is declared on every network-allocated provenance record, and the
    coefficient-to-path assignment is treated as a structural sensitivity factor (the
    dev/held-out entries differ and the swept ranges span the published scopes)."""

    entry_id: str
    kwh_per_gb: float
    swept_range_kwh_per_gb: tuple[float, float]
    source_publishes_range: bool
    path_scope_published: str
    provenance: Provenance

    def __post_init__(self) -> None:
        _req_str("entry_id", self.entry_id)
        _req_str("path_scope_published", self.path_scope_published)
        object.__setattr__(self, "kwh_per_gb", _pos("kwh_per_gb", self.kwh_per_gb))
        lo, hi = self.swept_range_kwh_per_gb
        lo, hi = _pos("swept low", lo), _pos("swept high", hi)
        object.__setattr__(self, "swept_range_kwh_per_gb", (lo, hi))
        if not lo <= self.kwh_per_gb <= hi:
            raise ValueError(f"{self.entry_id}: swept range must contain the point estimate")

    @property
    def j_per_byte(self) -> float:
        # 1 kWh = 3.6e6 J; 1 GB = 1e9 bytes.
        return self.kwh_per_gb * 3.6e6 / 1e9

    def swept_j_per_byte(self) -> tuple[float, float]:
        lo, hi = self.swept_range_kwh_per_gb
        return lo * 3.6e6 / 1e9, hi * 3.6e6 / 1e9


# ------------------------------------------------------------------ registry
def _pinned_data_sha256(data_dir: Path) -> str:
    """Digest the exact source-data closure used to issue a registry (R4-3).

    The two machine-readable registries and every retained artifact are included.
    Fetch scripts are tooling rather than scientific inputs and are intentionally
    outside this digest.
    """
    paths = [data_dir / "specpower.json", data_dir / "pinned_sources.json"]
    artifacts = data_dir / "artifacts"
    if artifacts.is_dir():
        paths.extend(p for p in artifacts.rglob("*") if p.is_file())
    manifest = {
        p.relative_to(data_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _registry_payload(
    groups: tuple[tuple[str, Mapping[str, object]], ...], source_data_sha256: str
) -> dict:
    return {
        "source_data_sha256": source_data_sha256,
        "groups": {
            name: {
                entry_id: dataclasses.asdict(entry) for entry_id, entry in sorted(entries.items())
            }
            for name, entries in groups
        },
    }


def _registry_seal(payload: dict) -> str:
    return hashlib.sha256(
        (
            _REGISTRY_SEAL_TAG
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SourceRegistry:
    """Immutable, loader-issued pinned sources (R4-3).

    ``VERIFIED`` is a property of the published record, not authority for an arbitrary
    caller-built Python object. Consequently direct construction is refused: only
    :func:`load_pinned_sources` issues the private authority token and a seal over the
    full entry payload plus the source-data digest. Bundle factories additionally call
    :meth:`require_canonical`, binding injection to the repository's pinned data.
    """

    specpower: Mapping[str, SpecPowerMachine] = field(default_factory=dict)
    arm_sbc: Mapping[str, ArmSbcMachine] = field(default_factory=dict)
    transitions: Mapping[str, TransitionMeasurement] = field(default_factory=dict)
    cloud_allocation: Mapping[str, CloudAllocationSource] = field(default_factory=dict)
    network_allocation: Mapping[str, NetworkAllocationSource] = field(default_factory=dict)
    source_data_sha256: str = ""
    seal: str = ""
    _authority: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _LOADER_AUTHORITY:
            raise ValueError(
                "SourceRegistry is loader-issued only; caller-built VERIFIED objects "
                "are not pinned-source authority (R4-3)"
            )
        for name in (
            "specpower",
            "arm_sbc",
            "transitions",
            "cloud_allocation",
            "network_allocation",
        ):
            object.__setattr__(self, name, types.MappingProxyType(dict(getattr(self, name))))
        if not isinstance(self.source_data_sha256, str) or len(self.source_data_sha256) != 64:
            raise ValueError("source_data_sha256 must be a 64-character digest")
        # Duplicate ids are invalid before a registry can be sealed.
        self._all_entries_unverified()
        if not self.seal:
            raise ValueError("loader-issued SourceRegistry requires a content seal")
        self.verify_seal()

    @staticmethod
    def _issue(
        *,
        specpower: Mapping[str, SpecPowerMachine],
        arm_sbc: Mapping[str, ArmSbcMachine],
        transitions: Mapping[str, TransitionMeasurement],
        cloud_allocation: Mapping[str, CloudAllocationSource],
        network_allocation: Mapping[str, NetworkAllocationSource],
        source_data_sha256: str,
    ) -> SourceRegistry:
        groups = (
            ("specpower", specpower),
            ("arm_sbc", arm_sbc),
            ("transitions", transitions),
            ("cloud_allocation", cloud_allocation),
            ("network_allocation", network_allocation),
        )
        return SourceRegistry(
            specpower=specpower,
            arm_sbc=arm_sbc,
            transitions=transitions,
            cloud_allocation=cloud_allocation,
            network_allocation=network_allocation,
            source_data_sha256=source_data_sha256,
            seal=_registry_seal(_registry_payload(groups, source_data_sha256)),
            _authority=_LOADER_AUTHORITY,
        )

    def _groups(self) -> tuple[tuple[str, Mapping[str, object]], ...]:
        return (
            ("specpower", self.specpower),
            ("arm_sbc", self.arm_sbc),
            ("transitions", self.transitions),
            ("cloud_allocation", self.cloud_allocation),
            ("network_allocation", self.network_allocation),
        )

    def verify_seal(self) -> None:
        expected = _registry_seal(_registry_payload(self._groups(), self.source_data_sha256))
        if self.seal != expected:
            raise ValueError(
                "SourceRegistry seal mismatch — a pinned source object changed after "
                "loader validation (R4-3)"
            )

    def require_canonical(self) -> None:
        """Require the exact repository-pinned data closure before scientific use.

        The digest comparison binds the files, while the payload comparison also
        makes this boundary total if internal Python helpers are called directly:
        no registry containing caller-modified objects is canonical merely because
        it repeats the canonical files' digest (R4-3).
        """
        self.verify_seal()
        canonical = _pinned_data_sha256(DATA_DIR)
        if self.source_data_sha256 != canonical:
            raise ValueError(
                "SourceRegistry was not loaded from the canonical pinned-data digest "
                f"(R4-3): {self.source_data_sha256[:12]} != {canonical[:12]}"
            )
        specpower = _load_specpower(DATA_DIR / "specpower.json")
        arm, transitions, cloud, network = _load_other(DATA_DIR / "pinned_sources.json")
        canonical_groups = (
            ("specpower", specpower),
            ("arm_sbc", arm),
            ("transitions", transitions),
            ("cloud_allocation", cloud),
            ("network_allocation", network),
        )
        canonical_seal = _registry_seal(_registry_payload(canonical_groups, canonical))
        if self.seal != canonical_seal:
            raise ValueError(
                "SourceRegistry payload differs from the canonical pinned source objects (R4-3)"
            )

    def _all_entries_unverified(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for _, group in self._groups():
            for key, value in group.items():
                if key in out:
                    raise ValueError(f"duplicate source entry id {key!r}")
                out[key] = value
        return out

    def all_entries(self) -> dict[str, object]:
        self.verify_seal()
        return self._all_entries_unverified()

    def verified(self) -> SourceRegistry:
        """The registry restricted to VERIFIED entries — the only entries the default
        bundle sets may draw from (TODO 3.1 data-integrity rule)."""

        self.verify_seal()

        def keep(d: Mapping) -> dict:
            return {k: v for k, v in d.items() if v.provenance.is_verified}

        return SourceRegistry._issue(
            specpower=keep(self.specpower),
            arm_sbc=keep(self.arm_sbc),
            transitions=keep(self.transitions),
            cloud_allocation=keep(self.cloud_allocation),
            network_allocation=keep(self.network_allocation),
            source_data_sha256=self.source_data_sha256,
        )


def _load_specpower(path: Path) -> dict[str, SpecPowerMachine]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "specpower-joint-tuples-v1":
        raise ValueError(f"{path}: unexpected schema {payload.get('schema')!r}")
    out = {}
    for machine_id, e in payload["entries"].items():
        out[machine_id] = SpecPowerMachine(
            machine_id=machine_id,
            vendor=e["vendor"],
            model=e["model"],
            cpu=e["cpu"],
            cpu_enabled=e["cpu_enabled"],
            memory_gb=str(e["memory_gb"]),
            test_date=e["test_date"],
            hardware_availability=e["hardware_availability"],
            publication=e["publication"],
            overall_ssj_ops_per_watt=e["overall_ssj_ops_per_watt"],
            load_points=tuple(
                SpecPowerLoadPoint(
                    target_load_frac=lp["target_load_pct"] / 100.0,
                    actual_load_frac=lp["actual_load_pct"] / 100.0,
                    ssj_ops=lp["ssj_ops"],
                    avg_active_power_w=lp["avg_active_power_w"],
                )
                # canonicalize to ascending load (result pages publish 100% -> 10%)
                for lp in sorted(e["load_levels"], key=lambda x: x["target_load_pct"])
            ),
            active_idle_w=e["active_idle_w"],
            provenance=Provenance.from_dict(e["provenance"]),
        )
    return out


def _load_other(path: Path) -> tuple[dict, dict, dict, dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "pinned-sources-v1":
        raise ValueError(f"{path}: unexpected schema {payload.get('schema')!r}")
    arm = {}
    for e in payload.get("arm_sbc", []):
        arm[e["machine_id"]] = ArmSbcMachine(
            machine_id=e["machine_id"],
            board=e["board"],
            cpu=e["cpu"],
            cores=e["cores"],
            load_metric=e["load_metric"],
            load_points=tuple((p[0], p[1]) for p in e["load_points"]),
            active_idle_w=e["active_idle_w"],
            peak_throughput_ops=e.get("peak_throughput_ops"),
            benchmark_domain=e.get("benchmark_domain"),
            provenance=Provenance.from_dict(e["provenance"]),
        )
    trans = {}
    for e in payload.get("transitions", []):
        trans[e["entry_id"]] = TransitionMeasurement(
            entry_id=e["entry_id"],
            device_described=e["device_described"],
            suspend_s=e["suspend_s"],
            suspend_energy_j=e["suspend_energy_j"],
            wake_s=e["wake_s"],
            wake_energy_j=e["wake_energy_j"],
            suspended_power_w=e["suspended_power_w"],
            derivation=e["derivation"],
            provenance=Provenance.from_dict(e["provenance"]),
        )
    cloud = {}
    for e in payload.get("cloud_allocation", []):
        cloud[e["entry_id"]] = CloudAllocationSource(
            entry_id=e["entry_id"],
            min_w_per_vcpu=e["min_w_per_vcpu"],
            max_w_per_vcpu=e["max_w_per_vcpu"],
            assumed_allocation_utilization=e["assumed_allocation_utilization"],
            swept_range_w_per_vcpu=tuple(e["swept_range_w_per_vcpu"]),
            source_publishes_range=e["source_publishes_range"],
            provenance=Provenance.from_dict(e["provenance"]),
        )
    net = {}
    for e in payload.get("network_allocation", []):
        net[e["entry_id"]] = NetworkAllocationSource(
            entry_id=e["entry_id"],
            kwh_per_gb=e["kwh_per_gb"],
            swept_range_kwh_per_gb=tuple(e["swept_range_kwh_per_gb"]),
            source_publishes_range=e["source_publishes_range"],
            path_scope_published=e["path_scope_published"],
            provenance=Provenance.from_dict(e["provenance"]),
        )
    return arm, trans, cloud, net


def load_pinned_sources(data_dir: Path | None = None) -> SourceRegistry:
    """Load and validate every pinned source entry (raises on schema/provenance gaps;
    re-hashes every retained raw artifact against its recorded content hash — R2-2)."""
    d = DATA_DIR if data_dir is None else data_dir
    specpower = _load_specpower(d / "specpower.json")
    arm, trans, cloud, net = _load_other(d / "pinned_sources.json")
    reg = SourceRegistry._issue(
        specpower=specpower,
        arm_sbc=arm,
        transitions=trans,
        cloud_allocation=cloud,
        network_allocation=net,
        source_data_sha256=_pinned_data_sha256(d),
    )
    for entry in reg.all_entries().values():  # duplicate-id check + artifact verification
        entry.provenance.verify_retained_artifact(d)
    return reg
