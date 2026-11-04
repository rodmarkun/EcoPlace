"""Declarative episode specifications (Phase 2; plan §3.4).

Everything the simulator consumes is described by frozen dataclasses defined here, with a
canonical JSON round-trip so that scenarios are seeded, replayable *artifacts* (plan §6.1).
Phase 4 generators will emit these specs; Phase 3 EnergyModelBundles will populate
``NodeClassSpec`` power parameters and ``EnergyCoefficients`` from published sources. The
numeric defaults in this file are **testing defaults** for development fixtures only — they
are placeholders with no provenance and must never appear in a confirmatory run (every
confirmatory parameter arrives via a pre-registered bundle or a fitted workload model).

Units, used consistently everywhere: time in **seconds** (float), CPU in **millicores**,
memory in **MiB**, power in **watts**, energy in **joules**, network payloads in **KB**
(1 KB = 1000 bytes), images in **MiB** (1 MiB = 2**20 bytes), rates in **per second**.
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
from typing import Any

TIER_CLOUD = 0
TIER_FOG = 1
TIER_EDGE = 2

# Host-domain limits (spec §4.1: the PlacementScript boundary validators enforce these on
# every ctx; the episode spec enforces them at construction so no legal episode can ever
# produce an out-of-domain ctx).
MAX_NODES = 512
MAX_PODS = 16_384
MAX_GRAPH_EDGES = 65_536
MAX_PROPOSAL_BUDGET = 16_384
MAX_RESOURCE = 1 << 40
# A stale scheduler view can contain one capacity-worth of observed truth plus one
# capacity-worth of assumed reservations. Keeping allocatable at half the wire maximum
# proves every reachable exported allocation remains within MAX_RESOURCE exactly.
MAX_NODE_CAPACITY = MAX_RESOURCE // 2
MAX_WATTS = 1e6
MAX_LATENCY_MS = 1e7
MAX_RATE = 1e9
MAX_PAYLOAD_KB = 1e9
MAX_NOW_MS = 1 << 53
MAX_GROUP_ID = (1 << 32) - 1
MAX_SERVICES = 512
MAX_TEMPLATES = 256
MAX_CALL_DEPTH = 32
MAX_TEMPLATE_EDGES = 4096


def _num(
    name: str,
    v: Any,
    *,
    nonneg: bool = True,
    maximum: float | None = None,
) -> float:
    """Validate one numeric field: a real, finite number (bool rejected); optionally
    nonnegative. Returns the float value. Every numeric spec field goes through this
    (finiteness/NaN/negativity destroy energy ordering and ceiling dominance — W2)."""
    if not isinstance(v, int | float) or isinstance(v, bool):
        raise ValueError(f"{name} must be a number, got {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {v!r}")
    if nonneg and f < 0:
        raise ValueError(f"{name} must be >= 0, got {v!r}")
    if maximum is not None and f > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}, got {v!r}")
    return f


def _int(
    name: str,
    v: Any,
    *,
    minimum: int | None = 0,
    maximum: int | None = None,
) -> int:
    """Validate a true integer count/id (bool and fractional floats rejected)."""
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {v}")
    return v


def _bool(name: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{name} must be a bool, got {v!r}")
    return v


# --------------------------------------------------------------------------- distributions
@dataclass(frozen=True)
class DistributionSpec:
    """A pluggable nonnegative-duration distribution (C11 'fitted service-time
    distributions' interface — Phase 4 fits parameters; the family set is fixed here).

    kinds:
      - ``deterministic``: params ``{"value": v}`` — always ``v``.
      - ``exponential``:   params ``{"mean": m}`` — Exp with mean ``m``.
      - ``lognormal``:     params ``{"mu": μ, "sigma": σ}`` of the underlying normal.
      - ``gamma``:         params ``{"shape": k, "scale": θ}``.
    """

    kind: str
    params: Mapping[str, float]

    _KINDS = ("deterministic", "exponential", "lognormal", "gamma")

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(f"unknown distribution kind {self.kind!r} (allowed: {self._KINDS})")
        required = {
            "deterministic": {"value"},
            "exponential": {"mean"},
            "lognormal": {"mu", "sigma"},
            "gamma": {"shape", "scale"},
        }[self.kind]
        if set(self.params) != required:
            raise ValueError(f"{self.kind} requires params {sorted(required)}")
        # mu may be negative (log-scale location); everything else must be nonnegative.
        checked = {
            k: _num(f"{self.kind} param {k}", v, nonneg=(k != "mu")) for k, v in self.params.items()
        }
        if self.kind == "gamma" and (checked["shape"] <= 0 or checked["scale"] <= 0):
            raise ValueError("gamma shape and scale must be > 0")
        if self.kind == "exponential" and checked["mean"] <= 0:
            raise ValueError("exponential mean must be > 0")
        # Freeze: params must not be mutable through the frozen dataclass (R1: frozen
        # dataclasses carrying mutable dicts are not actually immutable).
        object.__setattr__(self, "params", types.MappingProxyType(dict(checked)))

    def mean(self) -> float:
        """Analytic mean (used by the L0 closed forms)."""
        import math

        p = self.params
        if self.kind == "deterministic":
            return p["value"]
        if self.kind == "exponential":
            return p["mean"]
        if self.kind == "lognormal":
            return math.exp(p["mu"] + p["sigma"] ** 2 / 2.0)
        return p["shape"] * p["scale"]  # gamma

    def sample(self, rng: Any) -> float:
        """One draw from an explicitly seeded ``numpy.random.Generator`` (docs/SEEDS.md)."""
        p = self.params
        if self.kind == "deterministic":
            return float(p["value"])
        if self.kind == "exponential":
            return float(rng.exponential(p["mean"]))
        if self.kind == "lognormal":
            return float(rng.lognormal(p["mu"], p["sigma"]))
        return float(rng.gamma(p["shape"], p["scale"]))


def det(value: float) -> DistributionSpec:
    return DistributionSpec("deterministic", {"value": value})


def expo(mean: float) -> DistributionSpec:
    return DistributionSpec("exponential", {"mean": mean})


# --------------------------------------------------------------------------- cluster
@dataclass(frozen=True)
class NodeClassSpec:
    """A node hardware class. Phase 3 populates these from EnergyModelBundle machines
    (SPECpower joint points via ``power_curve_points``); the linear idle→max default is a
    development placeholder.

    Suspendability rule (plan §3.4/C17-void): a class is suspendable **only if all four
    transition parameters are provided**; classes lacking published transition data are
    non-suspendable in-scenario — this is enforced here, not left to convention.
    """

    name: str
    power_idle_w: float
    power_max_w: float
    # Optional joint (utilization in [0,1], watts) curve points; linear interpolation.
    # None -> linear between idle and max.
    power_curve_points: tuple[tuple[float, float], ...] | None = None
    suspend_energy_j: float | None = None
    suspend_latency_s: float | None = None
    wake_energy_j: float | None = None
    wake_latency_s: float | None = None
    suspended_power_w: float = 0.0
    image_pull_mib_per_s: float = 25.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("node class name must be a non-empty string")
        for pname in ("power_idle_w", "power_max_w", "suspended_power_w"):
            checked = _num(
                f"class {self.name}: {pname}",
                getattr(self, pname),
                maximum=MAX_WATTS,
            )
            object.__setattr__(self, pname, checked)
        pull_rate = _num(f"class {self.name}: image_pull_mib_per_s", self.image_pull_mib_per_s)
        object.__setattr__(self, "image_pull_mib_per_s", pull_rate)
        if pull_rate <= 0:
            raise ValueError(f"class {self.name}: image_pull_mib_per_s must be > 0")
        for pname in ("suspend_energy_j", "suspend_latency_s", "wake_energy_j", "wake_latency_s"):
            v = getattr(self, pname)
            if v is not None:
                object.__setattr__(self, pname, _num(f"class {self.name}: {pname}", v))
        if self.power_max_w < self.power_idle_w:
            raise ValueError(f"class {self.name}: need 0 <= idle <= max power")
        # Ceiling-dominance requirement (W2): every power state a node can occupy —
        # awake (curve), transition window (metered at idle power), and suspended — must
        # be bounded by the awake power curve's maximum, otherwise ceiling_j() is not an
        # upper bound over legal traces. Suspended draw above idle is therefore rejected.
        if self.suspended_power_w > self.power_idle_w:
            raise ValueError(
                f"class {self.name}: suspended_power_w ({self.suspended_power_w}) must "
                f"not exceed power_idle_w ({self.power_idle_w}) — suspended draw above "
                "the awake floor breaks ceiling dominance (W2) and is physically dubious"
            )
        if self.power_curve_points is not None:
            pts = tuple(
                (
                    _num(f"class {self.name}: curve util", p[0]),
                    _num(f"class {self.name}: curve watts", p[1], maximum=MAX_WATTS),
                )
                if len(p) == 2
                else tuple(p)
                for p in self.power_curve_points
            )
            object.__setattr__(self, "power_curve_points", pts)
            if len(pts) < 2 or any(len(p) != 2 for p in pts):
                raise ValueError(f"class {self.name}: curve needs >= 2 (util, watts) points")
            utils = [u for u, _ in pts]
            if utils != sorted(utils) or utils[0] != 0.0 or utils[-1] != 1.0:
                raise ValueError(f"class {self.name}: curve utils must ascend from 0.0 to 1.0")

    @property
    def suspendable(self) -> bool:
        params = (
            self.suspend_energy_j,
            self.suspend_latency_s,
            self.wake_energy_j,
            self.wake_latency_s,
        )
        return all(p is not None for p in params)

    def power_w(self, utilization: float) -> float:
        """Wall power at CPU utilization ``u`` in [0,1] (awake)."""
        u = min(max(utilization, 0.0), 1.0)
        if self.power_curve_points is None:
            return self.power_idle_w + (self.power_max_w - self.power_idle_w) * u
        pts = self.power_curve_points
        for (u0, p0), (u1, p1) in zip(pts, pts[1:], strict=False):
            if u <= u1:
                if u1 == u0:
                    return p1
                return p0 + (p1 - p0) * (u - u0) / (u1 - u0)
        return pts[-1][1]

    def max_curve_power_w(self) -> float:
        """Upper bound of the awake power curve (for the ceiling M, W2)."""
        if self.power_curve_points is None:
            return self.power_max_w
        return max(self.power_max_w, max(p for _, p in self.power_curve_points))


@dataclass(frozen=True)
class NodeSpec:
    id: int
    node_class: str
    tier: int  # 0 cloud, 1 fog, 2 far edge
    cpu_capacity: int  # millicores
    mem_capacity: int  # MiB
    initially_suspended: bool = False

    def __post_init__(self) -> None:
        _int(f"node {self.id}: id", self.id)
        if not isinstance(self.node_class, str) or not self.node_class:
            raise ValueError(f"node {self.id}: node_class must be a non-empty string")
        _int(f"node {self.id}: tier", self.tier)
        if self.tier not in (TIER_CLOUD, TIER_FOG, TIER_EDGE):
            raise ValueError(f"node {self.id}: tier must be 0/1/2")
        _int(
            f"node {self.id}: cpu_capacity",
            self.cpu_capacity,
            minimum=1,
            maximum=MAX_NODE_CAPACITY,
        )
        _int(
            f"node {self.id}: mem_capacity",
            self.mem_capacity,
            minimum=1,
            maximum=MAX_NODE_CAPACITY,
        )
        _bool(f"node {self.id}: initially_suspended", self.initially_suspended)


@dataclass(frozen=True)
class ClusterSpec:
    classes: tuple[NodeClassSpec, ...]
    nodes: tuple[NodeSpec, ...]
    latency_ms: tuple[tuple[float, ...], ...]  # NxN RTT/2 one-way latency, ms, symmetric

    def __post_init__(self) -> None:
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "latency_ms", tuple(tuple(row) for row in self.latency_ms))
        if not 1 <= len(self.nodes) <= MAX_NODES:
            raise ValueError(
                f"cluster must have 1..{MAX_NODES} nodes (spec §4.1 host domain), "
                f"got {len(self.nodes)}"
            )
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate node class names")
        by_name = {c.name: c for c in self.classes}
        for i, n in enumerate(self.nodes):
            if n.id != i:
                raise ValueError("node ids must be dense 0..N-1 in order")
            if n.node_class not in by_name:
                raise ValueError(f"node {n.id}: unknown class {n.node_class}")
            if n.initially_suspended and not by_name[n.node_class].suspendable:
                raise ValueError(f"node {n.id}: class {n.node_class} is not suspendable")
        n = len(self.nodes)
        if len(self.latency_ms) != n or any(len(row) != n for row in self.latency_ms):
            raise ValueError("latency_ms must be NxN")
        checked_latency = []
        for row in self.latency_ms:
            checked_row = tuple(_num("latency_ms", value) for value in row)
            if any(value > MAX_LATENCY_MS for value in checked_row):
                raise ValueError(f"latency out of the spec §4.1 domain [0, {MAX_LATENCY_MS:g}] ms")
            checked_latency.append(checked_row)
        object.__setattr__(self, "latency_ms", tuple(checked_latency))
        for i in range(n):
            if self.latency_ms[i][i] != 0.0:
                raise ValueError("latency_ms diagonal must be 0")
            for j in range(n):
                if self.latency_ms[j][i] != self.latency_ms[i][j]:
                    raise ValueError("latency_ms must be symmetric")

    def node_class_of(self, node_id: int) -> NodeClassSpec:
        by_name = {c.name: c for c in self.classes}
        return by_name[self.nodes[node_id].node_class]


# --------------------------------------------------------------------------- workload
@dataclass(frozen=True)
class BackendSpec:
    """A shared-state backend with a per-backend concurrency→latency curve (C11).

    ``latency_curve``: ascending ``(concurrency, latency_s)`` points; access latency at
    in-flight count c (including the accessing op) interpolates linearly and extrapolates
    with the last segment's slope (monotone curves stay monotone).
    """

    name: str
    latency_curve: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("backend name must be a non-empty string")
        raw_pts = tuple(tuple(p) for p in self.latency_curve)
        pts = tuple(
            (
                _num(f"backend {self.name}: concurrency point", p[0]),
                _num(f"backend {self.name}: latency point", p[1]),
            )
            if len(p) == 2
            else p
            for p in raw_pts
        )
        object.__setattr__(self, "latency_curve", pts)
        if len(pts) < 1:
            raise ValueError(f"backend {self.name}: need >= 1 curve point")
        for c, _lat in pts:
            if c <= 0:
                raise ValueError(f"backend {self.name}: concurrency points must be > 0")
        cs = [c for c, _ in pts]
        if cs != sorted(cs) or len(set(cs)) != len(cs):
            raise ValueError(f"backend {self.name}: concurrency points must strictly ascend")

    def latency_s(self, concurrency: float) -> float:
        pts = self.latency_curve
        if len(pts) == 1:
            return pts[0][1]
        if concurrency <= pts[0][0]:
            return pts[0][1]
        for (c0, l0), (c1, l1) in zip(pts, pts[1:], strict=False):
            if concurrency <= c1:
                return l0 + (l1 - l0) * (concurrency - c0) / (c1 - c0)
        (c0, l0), (c1, l1) = pts[-2], pts[-1]
        slope = (l1 - l0) / (c1 - c0)
        return max(0.0, l1 + slope * (concurrency - c1))


@dataclass(frozen=True)
class ServiceSpec:
    id: int
    name: str
    app: int
    cpu_request: int  # millicores (scheduling request)
    mem_request: int  # MiB
    service_time: DistributionSpec
    # Millicores actually held from the node's capacity while processing one request
    # (node-capacity contention, C11). Defaults to cpu_request at construction sites.
    cpu_demand: int
    concurrency: int = 1  # per-replica servers
    queue_capacity: int | None = None  # per-replica FIFO bound; None = unbounded
    cold_start: DistributionSpec = field(default_factory=lambda: det(0.0))
    image_mb: int = 50  # MiB; image identity == service id (pull cached per node)
    backend: str | None = None  # BackendSpec.name accessed once per request served
    stateful: bool = False
    pdb_protected: bool = False  # replicas count against a PodDisruptionBudget at its limit
    state_transfer_mb: float = 0.0  # extra bytes moved when the rebalancer migrates a replica
    initial_replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 8
    # Phase-4 onboarding: statically enumerated in the sealed workload, but no pods
    # are created before this absolute episode time.
    activation_s: float = 0.0
    category: str = "existing"

    def __post_init__(self) -> None:
        _int(f"service {self.name}: id", self.id)
        _int(f"service {self.name}: app", self.app, maximum=MAX_GROUP_ID)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("service name must be a non-empty string")
        activation = _num(f"service {self.name}: activation_s", self.activation_s)
        object.__setattr__(self, "activation_s", activation)
        if not isinstance(self.category, str) or not self.category:
            raise ValueError(f"service {self.name}: category must be a non-empty string")
        _int(f"service {self.name}: concurrency", self.concurrency, minimum=1)
        _int(
            f"service {self.name}: cpu_demand",
            self.cpu_demand,
            maximum=MAX_RESOURCE,
        )
        _int(
            f"service {self.name}: cpu_request",
            self.cpu_request,
            maximum=MAX_RESOURCE,
        )
        _int(
            f"service {self.name}: mem_request",
            self.mem_request,
            maximum=MAX_RESOURCE,
        )
        _int(
            f"service {self.name}: image_mb",
            self.image_mb,
            maximum=MAX_RESOURCE,
        )
        object.__setattr__(
            self,
            "state_transfer_mb",
            _num(f"service {self.name}: state_transfer_mb", self.state_transfer_mb),
        )
        if self.queue_capacity is not None:
            _int(f"service {self.name}: queue_capacity", self.queue_capacity, minimum=1)
        _bool(f"service {self.name}: stateful", self.stateful)
        _bool(f"service {self.name}: pdb_protected", self.pdb_protected)
        _int(f"service {self.name}: initial_replicas", self.initial_replicas, minimum=1)
        _int(f"service {self.name}: min_replicas", self.min_replicas, minimum=1)
        _int(f"service {self.name}: max_replicas", self.max_replicas, minimum=1)
        if not self.min_replicas <= self.max_replicas:
            raise ValueError(f"service {self.name}: need 0 < min_replicas <= max_replicas")
        if not self.min_replicas <= self.initial_replicas <= self.max_replicas:
            raise ValueError(f"service {self.name}: initial_replicas outside [min, max]")


@dataclass(frozen=True)
class CallEdgeSpec:
    """One template edge: the parent stage calls ``child`` with probability
    ``probability`` (sampled per request). ``timeout_s``/``max_retries`` apply to
    *blocking* child calls only (a forward handoff has no waiting caller to time out)."""

    child: CallSpec
    probability: float = 1.0
    timeout_s: float | None = None
    max_retries: int = 0
    request_kb: float = 1.0
    response_kb: float = 1.0

    def __post_init__(self) -> None:
        probability = _num("edge probability", self.probability)
        object.__setattr__(self, "probability", probability)
        if not 0.0 < probability <= 1.0:
            raise ValueError("edge probability must be in (0, 1]")
        _int("max_retries", self.max_retries)
        if self.timeout_s is not None:
            timeout = _num("timeout_s", self.timeout_s)
            object.__setattr__(self, "timeout_s", timeout)
            if timeout <= 0:
                raise ValueError("timeout_s must be > 0")
        request = _num("request_kb", self.request_kb, maximum=MAX_PAYLOAD_KB)
        response = _num("response_kb", self.response_kb, maximum=MAX_PAYLOAD_KB)
        if request + response > MAX_PAYLOAD_KB:
            raise ValueError(f"request_kb + response_kb must be <= {MAX_PAYLOAD_KB:g}")
        object.__setattr__(self, "request_kb", request)
        object.__setattr__(self, "response_kb", response)


@dataclass(frozen=True)
class CallSpec:
    """A template stage: process at ``service``, then invoke children (fan-out), and —
    in ``blocking`` mode — wait for all of them (fan-in) while holding the server slot;
    in ``forward`` mode release the server first (pipeline handoff, Jackson-style)."""

    service: int
    children: tuple[CallEdgeSpec, ...] = ()
    mode: str = "blocking"

    def __post_init__(self) -> None:
        _int("call service", self.service)
        object.__setattr__(self, "children", tuple(self.children))
        if self.mode not in ("blocking", "forward"):
            raise ValueError("mode must be 'blocking' or 'forward'")
        if self.mode == "forward":
            for e in self.children:
                if e.timeout_s is not None or e.max_retries != 0:
                    raise ValueError("forward edges cannot carry timeout/retries")


@dataclass(frozen=True)
class ArrivalSpec:
    """Open arrivals for one template.

    ``rate_per_s`` applies from ``start_s``. ``rate_schedule`` is an optional,
    strictly ordered sequence of ``(absolute_time_s, new_rate_per_s)`` changes. This
    is a piecewise-constant non-homogeneous process.  Arrivals are defined on the
    *cumulative intensity* rather than by restarting a clock at each boundary.  This
    makes an artificial split of one equal-rate interval observationally inert and
    preserves common-random-number pairing (Phase 4 drift scenarios).  A scheduled
    rate may be zero, which represents a closed interval with no offered traffic.
    """

    kind: str
    rate_per_s: float
    start_s: float = 0.0
    rate_schedule: tuple[tuple[float, float], ...] = ()
    # Scenario artifacts may seal exact offered timestamps for one episode seed.
    # Empty preserves the stochastic legacy/development generator.
    realized_times_s: tuple[float, ...] = ()
    arrivals_realized: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("poisson", "deterministic"):
            raise ValueError("arrival kind must be 'poisson' or 'deterministic'")
        rate = _num("rate_per_s", self.rate_per_s, maximum=MAX_RATE)
        object.__setattr__(self, "rate_per_s", rate)
        if rate <= 0:
            raise ValueError("rate_per_s must be > 0")
        start = _num("start_s", self.start_s)
        object.__setattr__(self, "start_s", start)
        raw = tuple(tuple(point) for point in self.rate_schedule)
        checked: list[tuple[float, float]] = []
        for point in raw:
            if len(point) != 2:
                raise ValueError("arrival rate_schedule entries must be (time_s, rate_per_s)")
            at = _num("arrival rate change time", point[0])
            new_rate = _num("arrival scheduled rate", point[1], maximum=MAX_RATE)
            if new_rate < 0:
                raise ValueError("arrival scheduled rates must be >= 0")
            checked.append((at, new_rate))
        times = [at for at, _ in checked]
        if any(at <= start for at in times) or times != sorted(set(times)):
            raise ValueError(
                "arrival rate_schedule times must be unique, strictly increasing, "
                "and later than start_s"
            )
        object.__setattr__(self, "rate_schedule", tuple(checked))
        realized = tuple(_num("realized arrival time", at) for at in self.realized_times_s)
        if realized != tuple(sorted(set(realized))) or any(at < start for at in realized):
            raise ValueError("realized arrival times must be unique, increasing, and >= start_s")
        object.__setattr__(self, "realized_times_s", realized)
        _bool("arrivals_realized", self.arrivals_realized)
        if realized and not self.arrivals_realized:
            raise ValueError("realized_times_s requires arrivals_realized=True")

    def rate_at(self, at_s: float) -> float:
        """Configured offered rate at ``at_s``; zero before the template starts."""
        at = _num("arrival query time", at_s)
        if at < self.start_s:
            return 0.0
        rate = self.rate_per_s
        for change_at, changed_rate in self.rate_schedule:
            if change_at > at:
                break
            rate = changed_rate
        return rate

    def segments(self, until_s: float) -> tuple[tuple[float, float, float], ...]:
        """Return non-empty ``(start, end, rate)`` intervals before ``until_s``."""
        until = _num("arrival segment limit", until_s)
        if until <= self.start_s:
            return ()
        points = ((self.start_s, self.rate_per_s), *self.rate_schedule)
        out: list[tuple[float, float, float]] = []
        for i, (start, rate) in enumerate(points):
            if start >= until:
                break
            end = points[i + 1][0] if i + 1 < len(points) else until
            end = min(end, until)
            if end > start:
                out.append((start, end, rate))
        return tuple(out)

    def integrated_rate(self, until_s: float, from_s: float = 0.0) -> float:
        """Integral of the configured intensity on ``[from_s, until_s)``."""
        until = _num("arrival integral limit", until_s)
        start_at = _num("arrival integral start", from_s)
        if until <= start_at:
            return 0.0
        return math.fsum(
            (end - max(start, start_at)) * rate
            for start, end, rate in self.segments(until)
            if end > start_at
        )

    def time_at_integrated_rate(self, target: float, until_s: float) -> float | None:
        """Invert cumulative intensity from ``start_s`` through ``until_s``.

        ``target`` is strictly positive.  The return is the earliest time whose
        cumulative intensity reaches it, or ``None`` when the finite horizon does not.
        Segment endpoints use the right-hand rate convention of :meth:`rate_at`.
        """
        wanted = _num("arrival integrated-rate target", target)
        if wanted <= 0:
            raise ValueError("arrival integrated-rate target must be > 0")
        accumulated = 0.0
        for start, end, rate in self.segments(until_s):
            mass = (end - start) * rate
            if rate > 0 and wanted <= accumulated + mass:
                return start + (wanted - accumulated) / rate
            accumulated += mass
        return None


@dataclass(frozen=True)
class NodeChurnSpec:
    """One planned scheduling-capacity churn event (Phase 4).

    ``schedulable=False`` cordons a node and ``True`` uncordons it. Existing pods
    continue serving; this deliberately models planned membership/capacity churn,
    not crash failure or pod eviction. That narrower semantic is replayable without
    pretending the DES models kubelet-failure cancellation.
    """

    at_s: float
    node_id: int
    schedulable: bool
    reason: str = "planned-capacity-churn"

    def __post_init__(self) -> None:
        at = _num("node churn time", self.at_s)
        if at <= 0:
            raise ValueError("node churn time must be > 0 (initial state belongs in ClusterSpec)")
        object.__setattr__(self, "at_s", at)
        _int("node churn node_id", self.node_id)
        _bool("node churn schedulable", self.schedulable)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("node churn reason must be a non-empty string")


@dataclass(frozen=True)
class TemplateSpec:
    id: int
    name: str
    root: CallSpec
    arrival: ArrivalSpec
    # End-to-end latency SLO for this template's app-level constraint checks.
    slo_latency_s: float | None = None

    def __post_init__(self) -> None:
        _int(f"template {self.name}: id", self.id)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("template name must be a non-empty string")
        if self.slo_latency_s is not None:
            slo = _num(f"template {self.name}: slo_latency_s", self.slo_latency_s)
            object.__setattr__(self, "slo_latency_s", slo)
            if slo <= 0:
                raise ValueError("slo_latency_s must be > 0 or None")


@dataclass(frozen=True)
class WorkloadSpec:
    services: tuple[ServiceSpec, ...]
    templates: tuple[TemplateSpec, ...]
    backends: tuple[BackendSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "services", tuple(self.services))
        object.__setattr__(self, "templates", tuple(self.templates))
        object.__setattr__(self, "backends", tuple(self.backends))
        if not 1 <= len(self.services) <= MAX_SERVICES:
            raise ValueError(f"workload must have 1..{MAX_SERVICES} services")
        if not 1 <= len(self.templates) <= MAX_TEMPLATES:
            raise ValueError(f"workload must have 1..{MAX_TEMPLATES} templates")
        for i, s in enumerate(self.services):
            if s.id != i:
                raise ValueError("service ids must be dense 0..S-1 in order")
        max_live_pods = sum(service.max_replicas for service in self.services)
        if max_live_pods > MAX_PODS:
            raise ValueError(
                f"sum(max_replicas) is {max_live_pods}, above the PlacementScript "
                f"host-domain maximum {MAX_PODS}"
            )
        backend_names = {b.name for b in self.backends}
        if len(backend_names) != len(self.backends):
            raise ValueError("duplicate backend names")
        for s in self.services:
            if s.backend is not None and s.backend not in backend_names:
                raise ValueError(f"service {s.name}: unknown backend {s.backend}")
        tids = [t.id for t in self.templates]
        if tids != list(range(len(tids))):
            raise ValueError("template ids must be dense 0..T-1 in order")
        expanded_edges = 0
        for t in self.templates:
            if self._check_call(t.root, depth=0) > MAX_TEMPLATE_EDGES:
                raise ValueError(f"template {t.id}: more than {MAX_TEMPLATE_EDGES} edges")
            latest_activation = max(
                self.services[sid].activation_s for sid in self._call_services(t.root)
            )
            if t.arrival.start_s < latest_activation:
                raise ValueError(
                    f"template {t.id}: arrivals start at {t.arrival.start_s} before "
                    f"a referenced service activates at {latest_activation}"
                )
            expanded_edges += self._max_expanded_edges(t.root)
        if expanded_edges > MAX_GRAPH_EDGES:
            raise ValueError(
                f"worst-case expanded pod graph has {expanded_edges} edges, above the "
                f"PlacementScript host-domain maximum {MAX_GRAPH_EDGES}"
            )

    def _check_call(self, call: CallSpec, depth: int) -> int:
        """Validate one call subtree; returns its edge count. Depth-bounded."""
        if depth > MAX_CALL_DEPTH:
            raise ValueError(f"template call depth exceeds {MAX_CALL_DEPTH}")
        if not 0 <= call.service < len(self.services):
            raise ValueError(f"template references unknown service {call.service}")
        edges = len(call.children)
        for e in call.children:
            edges += self._check_call(e.child, depth + 1)
        return edges

    def _call_services(self, call: CallSpec) -> set[int]:
        out = {call.service}
        for edge in call.children:
            out.update(self._call_services(edge.child))
        return out

    def _max_expanded_edges(self, call: CallSpec) -> int:
        """Worst-case pod-level columns if every service is at max_replicas/READY."""
        source_replicas = self.services[call.service].max_replicas
        edges = 0
        for edge in call.children:
            dest_replicas = self.services[edge.child.service].max_replicas
            edges += source_replicas * dest_replicas
            edges += self._max_expanded_edges(edge.child)
        return edges

    def service(self, sid: int) -> ServiceSpec:
        return self.services[sid]

    def backend_by_name(self, name: str) -> BackendSpec:
        return next(b for b in self.backends if b.name == name)


# --------------------------------------------------------------------------- controllers
@dataclass(frozen=True)
class SchedulerParams:
    """Scheduling-cycle emulator knobs (S2). See ``sim.scheduler.EMULATOR_COVERAGE`` for
    the G2 reproduced/pinned-off enumeration these parameterize."""

    informer_delay_s: float = 0.0  # staleness of externally-caused cache updates
    bind_latency: DistributionSpec = field(default_factory=lambda: det(0.01))
    bind_failure_prob: float = 0.0  # injected API-conflict probability (seeded stream)
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 10.0
    cycle_overhead_s: float = 0.0  # serial per-cycle latency (framework overhead)
    quantize_scores: bool = True  # NormalizeScore -> int in [0,100], per the T1 plugin
    bind_time_capacity_check: bool = True  # revalidate fit against truth at bind

    def __post_init__(self) -> None:
        for name in ("informer_delay_s", "cycle_overhead_s"):
            object.__setattr__(self, name, _num(name, getattr(self, name)))
        failure_prob = _num("bind_failure_prob", self.bind_failure_prob)
        object.__setattr__(self, "bind_failure_prob", failure_prob)
        if not 0.0 <= failure_prob <= 1.0:
            raise ValueError("bind_failure_prob in [0,1]")
        initial = _num("backoff_initial_s", self.backoff_initial_s)
        maximum = _num("backoff_max_s", self.backoff_max_s)
        object.__setattr__(self, "backoff_initial_s", initial)
        object.__setattr__(self, "backoff_max_s", maximum)
        if initial <= 0 or maximum < initial:
            raise ValueError("need 0 < backoff_initial_s <= backoff_max_s")
        _bool("quantize_scores", self.quantize_scores)
        _bool("bind_time_capacity_check", self.bind_time_capacity_check)
        if not self.bind_time_capacity_check:
            raise ValueError(
                "bind_time_capacity_check must stay enabled for evaluable episodes; "
                "truth overcommit is outside the energy ceiling domain"
            )


@dataclass(frozen=True)
class HpaParams:
    """HPA twin — one config shared by every method in a comparison (C3)."""

    enabled: bool = False
    sync_period_s: float = 15.0
    target_utilization: float = 0.6  # busy-CPU / requested-CPU across ready replicas
    scale_down_stabilization_s: float = 300.0
    tolerance: float = 0.1  # no action within +/- tolerance of target (kube default 0.1)

    def __post_init__(self) -> None:
        _bool("hpa.enabled", self.enabled)
        target = _num("target_utilization", self.target_utilization)
        object.__setattr__(self, "target_utilization", target)
        if not 0.0 < target <= 1.0:
            raise ValueError("target_utilization in (0,1]")
        sync = _num("sync_period_s", self.sync_period_s)
        object.__setattr__(self, "sync_period_s", sync)
        if sync <= 0:
            raise ValueError("sync_period_s must be > 0")
        object.__setattr__(
            self,
            "scale_down_stabilization_s",
            _num("scale_down_stabilization_s", self.scale_down_stabilization_s),
        )
        tolerance = _num("tolerance", self.tolerance)
        object.__setattr__(self, "tolerance", tolerance)
        if tolerance >= 1.0:
            raise ValueError("tolerance must be in [0,1)")


@dataclass(frozen=True)
class RebalancerParams:
    """Budgeted rebalancing controller twin (plan §3.1/§3.4)."""

    enabled: bool = False
    window_s: float = 60.0
    budget_per_window: int = 4
    max_total_moves: int | None = None  # cumulative disruption caps (episode-level)
    max_total_bytes_mb: float | None = None
    max_total_unavailability_s: float | None = None
    excluded_pods: tuple[int, ...] = ()  # extra stateful exclusions by pod uid
    # Conservative per-move unavailability pre-charge basis (seconds) added on top of
    # the structural bound (wake latency + image pull + cold-start mean) when admitting
    # a move against max_total_unavailability_s. The cap is enforced *before* execution
    # against this bound and trued up with the measured value afterwards.
    unavailability_precharge_s: float = 1.0

    def __post_init__(self) -> None:
        _bool("rebalancer.enabled", self.enabled)
        object.__setattr__(self, "excluded_pods", tuple(self.excluded_pods))
        window = _num("window_s", self.window_s)
        object.__setattr__(self, "window_s", window)
        if window <= 0:
            raise ValueError("window_s must be > 0")
        _int(
            "budget_per_window",
            self.budget_per_window,
            maximum=MAX_PROPOSAL_BUDGET,
        )
        for cap_name in ("max_total_moves", "max_total_bytes_mb", "max_total_unavailability_s"):
            v = getattr(self, cap_name)
            if v is not None:
                if cap_name == "max_total_moves":
                    _int(cap_name, v)
                else:
                    object.__setattr__(self, cap_name, _num(cap_name, v))
        object.__setattr__(
            self,
            "unavailability_precharge_s",
            _num("unavailability_precharge_s", self.unavailability_precharge_s),
        )
        for uid in self.excluded_pods:
            _int("excluded pod uid", uid)


@dataclass(frozen=True)
class PowerParams:
    """Node power-state controller. Suspension is only ever applied to empty,
    suspendable nodes; wake happens on demand (scheduler binds to a suspended node are
    filtered out, so wakes come from rebalancer ``wake`` actions or this controller)."""

    suspend_after_idle_s: float | None = None  # None = never auto-suspend

    def __post_init__(self) -> None:
        if self.suspend_after_idle_s is not None:
            delay = _num("suspend_after_idle_s", self.suspend_after_idle_s)
            object.__setattr__(self, "suspend_after_idle_s", delay)
            if delay <= 0:
                raise ValueError("suspend_after_idle_s must be > 0 or None")


@dataclass(frozen=True)
class EnergySourceRecord:
    """One structured pinned-source reference carried end-to-end (§4.3; Phase-3 R2-2).

    Minimal Phase-2 touch (documented, like ``bundle_set``): Phase 3's bundle layer
    populates these from the pinned source registry so every downstream
    ``ProvenanceRecord`` cites the machine/transition/allocation entries — id, kind,
    source name, URL/DOI, version/access date, and the retained-artifact content hash —
    structurally, not only inside a prose string.
    """

    entry_id: str
    kind: str  # "machine" | "transition" | "cloud-allocation" | "network-allocation"
    source_name: str
    url: str | None
    doi: str | None
    version_or_date_accessed: str
    status: str
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("entry_id", "kind", "source_name", "version_or_date_accessed", "status"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v:
                raise ValueError(f"source record {name} must be a non-empty string")
        if self.url is None and self.doi is None:
            raise ValueError(f"source record {self.entry_id}: needs a URL or a DOI")


@dataclass(frozen=True)
class EnergyCoefficients:
    """One versioned coefficient/parameter config for the closed modeled energy boundary
    (C16/S5). Phase 3 populates values from pinned public sources; the defaults here are
    development placeholders (status 'modeled', no provenance).

    ``cloud_floor_vcpu`` is a **baseline cloud allocation** (control-plane/reservation
    floor) that is allocated for the whole horizon regardless of policy: it enters the
    metered cloud category, the idle floor, and the ceiling identically, so floor <=
    actual <= ceiling holds within the cloud category by construction.

    ``source`` is the §4.3 provenance anchor (dataset + version/DOI); Phase 3 bundles
    must replace the placeholder.
    """

    version: str = "dev-0"
    cloud_j_per_vcpu_hour: float = 10800.0  # ~3 Wh per vCPU-hour placeholder
    network_j_per_byte: float = 1.0e-8
    include_image_pull_bytes: bool = True
    cloud_floor_vcpu: float = 0.0  # baseline cloud allocation (metered + floor + ceiling)
    source: str = "development-placeholder (no provenance; Phase 3 pins sources)"
    parameter_ranges: str = "point estimates only (no swept ranges at Phase 2)"
    # §4.3: the bundle set a figure survives. Phase 3 bundles populate this
    # ("<role>:<bundle_id>"); the default is the pre-bundle development placeholder.
    bundle_set: str = "single-config (development placeholder; no bundle set)"
    # §4.3 structured source references (R2-2): populated by Phase-3 bundles from the
    # pinned registry; empty for development-placeholder coefficients.
    source_records: tuple[EnergySourceRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in ("cloud_j_per_vcpu_hour", "network_j_per_byte", "cloud_floor_vcpu"):
            object.__setattr__(self, name, _num(name, getattr(self, name)))
        _bool("include_image_pull_bytes", self.include_image_pull_bytes)
        for name in ("version", "source", "parameter_ranges", "bundle_set"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"energy {name} must be a non-empty string")
        object.__setattr__(self, "source_records", tuple(self.source_records))
        if not all(isinstance(r, EnergySourceRecord) for r in self.source_records):
            raise ValueError("source_records must be EnergySourceRecord instances")

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(dataclasses.asdict(self), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


# --------------------------------------------------------------------------- episode
@dataclass(frozen=True)
class EpisodeConfig:
    """A complete, terminating episode (S6: terminating by design — arrivals stop at
    ``arrival_cutoff_s``; the run ends at ``horizon_s``; whatever is still in flight is
    counted, never lost)."""

    cluster: ClusterSpec
    workload: WorkloadSpec
    horizon_s: float
    arrival_cutoff_s: float | None = None  # None -> horizon_s
    scheduler: SchedulerParams = field(default_factory=SchedulerParams)
    hpa: HpaParams = field(default_factory=HpaParams)
    rebalancer: RebalancerParams = field(default_factory=RebalancerParams)
    power: PowerParams = field(default_factory=PowerParams)
    energy: EnergyCoefficients = field(default_factory=EnergyCoefficients)
    snapshot_times_s: tuple[float, ...] = ()
    # Decision-point snapshots: capture at the start of these scheduling cycles
    # (0-based cycle index; the pod of that cycle plus everything queued behind it is
    # pending in the snapshot — this is what makes L0 snapshots policy-discriminating).
    snapshot_cycles: tuple[int, ...] = ()
    # Planned cordon/uncordon events. Existing pods remain; see NodeChurnSpec.
    node_churn: tuple[NodeChurnSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_times_s",
            tuple(_num("snapshot time", t) for t in self.snapshot_times_s),
        )
        object.__setattr__(self, "snapshot_cycles", tuple(self.snapshot_cycles))
        object.__setattr__(self, "node_churn", tuple(self.node_churn))
        horizon = _num("horizon_s", self.horizon_s)
        object.__setattr__(self, "horizon_s", horizon)
        if horizon <= 0:
            raise ValueError("horizon_s must be > 0")
        if horizon * 1000.0 > MAX_NOW_MS:
            raise ValueError(f"horizon_s exceeds the PlacementScript now_ms maximum {MAX_NOW_MS}")
        cutoff = self.arrival_cutoff_s
        if cutoff is not None:
            checked_cutoff = _num("arrival_cutoff_s", cutoff)
            object.__setattr__(self, "arrival_cutoff_s", checked_cutoff)
            if not 0 < checked_cutoff <= self.horizon_s:
                raise ValueError("arrival_cutoff_s must be in (0, horizon_s]")
        for t in self.snapshot_times_s:
            if not 0 <= t <= self.horizon_s:
                raise ValueError("snapshot times must be within the horizon")
        for cycle in self.snapshot_cycles:
            _int("snapshot cycle index", cycle)
        for template in self.workload.templates:
            if any(at >= self.cutoff_s for at, _ in template.arrival.rate_schedule):
                raise ValueError(
                    f"template {template.id}: every arrival rate change must occur "
                    "strictly before arrival_cutoff_s"
                )
            if any(at >= self.cutoff_s for at in template.arrival.realized_times_s):
                raise ValueError(
                    f"template {template.id}: realized arrivals must be strictly before cutoff"
                )
        for service in self.workload.services:
            if service.activation_s >= self.horizon_s:
                raise ValueError(
                    f"service {service.id}: activation_s must be strictly before horizon_s"
                )
        ordered = tuple(sorted(self.node_churn, key=lambda event: (event.at_s, event.node_id)))
        if self.node_churn != ordered:
            raise ValueError("node_churn must be in canonical (time, node_id) order")
        states = [True] * len(self.cluster.nodes)
        seen: set[tuple[float, int]] = set()
        for event in self.node_churn:
            if not 0 <= event.node_id < len(states):
                raise ValueError(f"node churn references unknown node {event.node_id}")
            if event.at_s >= self.horizon_s:
                raise ValueError("node churn events must occur strictly before horizon_s")
            identity = (event.at_s, event.node_id)
            if identity in seen:
                raise ValueError("duplicate node churn event for one node at one time")
            seen.add(identity)
            if states[event.node_id] == event.schedulable:
                raise ValueError(
                    f"node {event.node_id} churn does not change schedulability at {event.at_s}"
                )
            states[event.node_id] = event.schedulable
            if not any(states):
                raise ValueError("node churn cannot make every node unschedulable")

    @property
    def cutoff_s(self) -> float:
        return self.horizon_s if self.arrival_cutoff_s is None else self.arrival_cutoff_s


# --------------------------------------------------------------------------- JSON I/O
def _to_jsonable(obj: Any) -> Any:
    # Phase-4 temporal fields are omitted when empty so every legacy Phase-2/3
    # EpisodeConfig keeps its byte-for-byte canonical encoding and hash.  New
    # artifacts carry the fields explicitly as soon as they are non-empty.
    if isinstance(obj, ArrivalSpec):
        out = {
            "kind": obj.kind,
            "rate_per_s": obj.rate_per_s,
            "start_s": obj.start_s,
        }
        if obj.rate_schedule:
            out["rate_schedule"] = _to_jsonable(obj.rate_schedule)
        if obj.realized_times_s:
            out["realized_times_s"] = _to_jsonable(obj.realized_times_s)
        if obj.arrivals_realized:
            out["arrivals_realized"] = True
        return out
    if isinstance(obj, ServiceSpec):
        out = {
            f.name: _to_jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name not in ("activation_s", "category")
        }
        if obj.activation_s != 0.0:
            out["activation_s"] = obj.activation_s
        if obj.category != "existing":
            out["category"] = obj.category
        return out
    if isinstance(obj, EpisodeConfig):
        out = {
            f.name: _to_jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name != "node_churn"
        }
        if obj.node_churn:
            out["node_churn"] = _to_jsonable(obj.node_churn)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple | list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, Mapping):  # includes the frozen MappingProxyType params
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def episode_config_to_json(config: EpisodeConfig) -> str:
    """Canonical JSON encoding (sorted keys — the basis for config hashing)."""
    return json.dumps(_to_jsonable(config), sort_keys=True, indent=2)


def _dist(d: dict[str, Any]) -> DistributionSpec:
    return DistributionSpec(d["kind"], dict(d["params"]))


def _call(d: dict[str, Any]) -> CallSpec:
    return CallSpec(
        service=d["service"],
        children=tuple(
            CallEdgeSpec(
                child=_call(e["child"]),
                probability=e.get("probability", 1.0),
                timeout_s=e.get("timeout_s"),
                max_retries=e.get("max_retries", 0),
                request_kb=e.get("request_kb", 1.0),
                response_kb=e.get("response_kb", 1.0),
            )
            for e in d.get("children", ())
        ),
        mode=d.get("mode", "blocking"),
    )


def episode_config_from_json(text: str) -> EpisodeConfig:
    d = json.loads(text)
    cluster = ClusterSpec(
        classes=tuple(
            NodeClassSpec(
                **{
                    **c,
                    "power_curve_points": (
                        tuple(tuple(p) for p in c["power_curve_points"])
                        if c.get("power_curve_points") is not None
                        else None
                    ),
                }
            )
            for c in d["cluster"]["classes"]
        ),
        nodes=tuple(NodeSpec(**n) for n in d["cluster"]["nodes"]),
        latency_ms=tuple(tuple(row) for row in d["cluster"]["latency_ms"]),
    )
    workload = WorkloadSpec(
        services=tuple(
            ServiceSpec(
                **{
                    **s,
                    "service_time": _dist(s["service_time"]),
                    "cold_start": _dist(s["cold_start"]) if "cold_start" in s else det(0.0),
                }
            )
            for s in d["workload"]["services"]
        ),
        templates=tuple(
            TemplateSpec(
                id=t["id"],
                name=t["name"],
                root=_call(t["root"]),
                arrival=ArrivalSpec(
                    **{
                        **t["arrival"],
                        "rate_schedule": tuple(
                            tuple(point) for point in t["arrival"].get("rate_schedule", ())
                        ),
                        "realized_times_s": tuple(t["arrival"].get("realized_times_s", ())),
                        "arrivals_realized": t["arrival"].get("arrivals_realized", False),
                    }
                ),
                slo_latency_s=t.get("slo_latency_s"),
            )
            for t in d["workload"]["templates"]
        ),
        backends=tuple(
            BackendSpec(name=b["name"], latency_curve=tuple(tuple(p) for p in b["latency_curve"]))
            for b in d["workload"].get("backends", ())
        ),
    )
    return EpisodeConfig(
        cluster=cluster,
        workload=workload,
        horizon_s=d["horizon_s"],
        arrival_cutoff_s=d.get("arrival_cutoff_s"),
        scheduler=SchedulerParams(
            **{
                **d.get("scheduler", {}),
                **(
                    {"bind_latency": _dist(d["scheduler"]["bind_latency"])}
                    if "scheduler" in d and "bind_latency" in d["scheduler"]
                    else {}
                ),
            }
        ),
        hpa=HpaParams(**d.get("hpa", {})),
        rebalancer=RebalancerParams(
            **{
                **d.get("rebalancer", {}),
                **(
                    {"excluded_pods": tuple(d["rebalancer"]["excluded_pods"])}
                    if "rebalancer" in d and "excluded_pods" in d["rebalancer"]
                    else {}
                ),
            }
        ),
        power=PowerParams(**d.get("power", {})),
        energy=EnergyCoefficients(
            **{
                **d.get("energy", {}),
                "source_records": tuple(
                    EnergySourceRecord(**record)
                    for record in d.get("energy", {}).get("source_records", ())
                ),
            }
        ),
        snapshot_times_s=tuple(d.get("snapshot_times_s", ())),
        snapshot_cycles=tuple(d.get("snapshot_cycles", ())),
        node_churn=tuple(NodeChurnSpec(**event) for event in d.get("node_churn", ())),
    )


def load_episode_config(path: str | Path) -> EpisodeConfig:
    return episode_config_from_json(Path(path).read_text())


def save_episode_config(config: EpisodeConfig, path: str | Path) -> None:
    Path(path).write_text(episode_config_to_json(config) + "\n")


def config_sha256(config: EpisodeConfig) -> str:
    return hashlib.sha256(episode_config_to_json(config).encode()).hexdigest()
