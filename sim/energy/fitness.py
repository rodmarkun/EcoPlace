"""Fitness aggregation over development bundles (TODO 3.4; plan §4.1/§5/§6.3,
corrections V1/V2/W2/W3).

This module is pure arithmetic over :class:`InstanceOutcome` records — it never runs
the simulator. Phase 5's ``evaluate.py`` verifies every episode's report through
``BoundaryCertificate.verify_covers`` (which returns the sealed
``VerifiedInstanceEnergy`` token — R2-1), wraps the tokens into outcomes, and calls
:func:`evaluate_candidate`; the contract is documented in ``sim/energy/README.md``.
The module-level :func:`feasible_regret`/:func:`penalty_regret`/:func:`penalty_dominates`
are the bare arithmetic kernels (exposed so tests and P2 sensitivity analyses can
assert the W2 dominance algebra); the *identity-checked* regret path is
``BoundaryCertificate.feasible_regret``/``penalty_regret`` over verified tokens.

The total order (V2/W3), implemented exactly as the plan defines it:

1. A candidate is **feasible** only if it satisfies every C20 constraint in **every**
   development bundle (all primitive shortfalls exactly zero on every instance).
2. All infeasible candidates rank strictly below all feasible ones.
3. Infeasible candidates are ordered lexicographically on a 4-component vector built
   with these aggregation axes, in order: per primitive constraint component, **max
   across development bundles** within each scenario × seed instance → **max within
   each constraint category** (across apps for SLO; across drop vs. timeout; across
   the disruption components) → **mean across scenario × seed instances**, yielding
   (SLO, goodput, drop/timeout, disruption) in that pre-registered priority.
4. Residual ties: total summed shortfall — literally the **sum** of primitive
   component shortfalls *before within-category reduction* (W3; the
   max-across-bundles step is part of the per-component definition and precedes it),
   totalled over the frozen scenario × seed grid — then the canonical proposal
   sequence number assigned at candidate creation (never asynchronous completion
   order). Comparability across candidates is guaranteed because every candidate is
   evaluated on the same frozen ``instance_keys`` grid (enforced, not assumed).
5. Feasible candidates are ordered by fitness = **mean over development scenario ×
   bundle instances of closed-boundary energy / instance idle-floor energy** (V1: the
   normalizer is policy-independent), with the proposal sequence number as the final
   deterministic tie-break.

Shortfall formulas (V2), all dimensionless with strictly positive denominators:

- "≥ target" constraints (SLO satisfaction, goodput): ``max(0, (target − achieved) /
  target)`` — targets are pre-registered strictly positive.
- "≤ cap" constraints (drop/timeout rates, disruption components): ``max(0, (value −
  cap) / cap)``; a constraint whose permitted cap is **zero** uses its pre-registered
  positive scaling constant as denominator: ``max(0, value / scale)``.

Penalty (W2): ``P(instance) = (M − reference energy) / I + 1`` where ``M`` is the
complete-closed-boundary ceiling (``sim.energy.ceiling_j``: max node power + bounded
transition energy + cloud/network allocation ceilings) and ``I`` the instance idle
floor (``sim.energy.idle_floor_j``, positivity-verified). Because any feasible
policy's closed-boundary energy E satisfies E ≤ M (ceiling dominance, Phase 2), every
feasible regret (E − ref)/I ≤ (M − ref)/I < P — the penalty strictly exceeds any
feasible regret by construction (asserted in tests, not assumed).
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field

from sim.energy.boundary import VerifiedInstanceEnergy

CATEGORY_ORDER: tuple[str, ...] = ("slo", "goodput", "drop_timeout", "disruption")


def _finite(name: str, v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def _fraction(name: str, v: object) -> float:
    """A physical rate/satisfaction fraction: finite and within [0, 1] (R3-4)."""
    out = _finite(name, v)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {out}")
    return out


def _frozen(name: str, m: Mapping) -> Mapping:
    """Defensive snapshot (R3-4): a caller mutating its dict after construction must
    not bypass validation — specs and outcomes keep an immutable copy."""
    if not isinstance(m, Mapping):
        raise ValueError(f"{name} must be a mapping")
    snapshot = dict(m)
    for key in snapshot:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings (R4-7), got {key!r}")
    return types.MappingProxyType(snapshot)


# ------------------------------------------------------------------ shortfalls
def ge_shortfall(target: float, achieved: float) -> float:
    """Normalized shortfall for a '≥ target' constraint (V2). Target must be a
    finite number > 0 (R3-4: a NaN target must raise, never soften to 0)."""
    if not _finite("target", target) > 0:
        raise ValueError("'>= target' constraints require a strictly positive target")
    _finite("achieved", achieved)
    return max(0.0, (target - achieved) / target)


def le_shortfall(cap: float, value: float, zero_cap_scale: float | None = None) -> float:
    """Normalized shortfall for a '≤ cap' constraint (V2). The cap must be a finite
    number >= 0 (R3-4: a NaN cap previously yielded a silent zero shortfall via
    ``max(0.0, nan)`` — it must raise); a zero cap requires its pre-registered
    positive scaling constant."""
    if _finite("cap", cap) < 0:
        raise ValueError("caps must be >= 0")
    _finite("value", value)
    if cap == 0.0:
        if zero_cap_scale is None or not _finite("zero_cap_scale", zero_cap_scale) > 0:
            raise ValueError(
                "a zero cap requires a pre-registered strictly positive scaling constant"
            )
        return max(0.0, value / zero_cap_scale)
    return max(0.0, (value - cap) / cap)


# ------------------------------------------------------------------ specs
@dataclass(frozen=True)
class ConstraintSpec:
    """The C20 constraint set (plan §5), pre-registered (frozen in P2).

    ``zero_cap_scales`` must name a strictly positive scale for every component whose
    cap is zero — validated here so the order is total from construction on.
    """

    slo_targets: Mapping[str, float]  # per-app required SLO satisfaction (offered cohort)
    goodput_target: float
    drop_cap: float
    timeout_cap: float
    disruption_caps: Mapping[str, float]  # e.g. moves, bytes_mb, unavailability_s
    zero_cap_scales: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # R3-4: snapshot every mapping — later mutation of the caller's dict must
        # not smuggle unvalidated (NaN/out-of-domain) values past this validation.
        object.__setattr__(self, "slo_targets", _frozen("slo_targets", self.slo_targets))
        object.__setattr__(
            self, "disruption_caps", _frozen("disruption_caps", self.disruption_caps)
        )
        object.__setattr__(
            self, "zero_cap_scales", _frozen("zero_cap_scales", self.zero_cap_scales)
        )
        if not self.slo_targets:
            raise ValueError("at least one SLO target is required")
        for app, t in self.slo_targets.items():
            if not _fraction(f"slo_targets[{app}]", t) > 0:
                raise ValueError(f"SLO target for {app!r} must be strictly positive (V2)")
        if not _fraction("goodput_target", self.goodput_target) > 0:
            raise ValueError("goodput target must be strictly positive (V2)")
        for name, cap in (("drop", self.drop_cap), ("timeout", self.timeout_cap)):
            _fraction(f"{name}_cap", cap)  # drop/timeout caps are rates (R3-4)
            self._check_zero_cap(name, cap)
        if not self.disruption_caps:
            raise ValueError("at least one disruption cap is required")
        for comp, cap in self.disruption_caps.items():
            if _finite(f"disruption_caps[{comp}]", cap) < 0:
                raise ValueError(f"disruption cap {comp!r} must be >= 0")
            self._check_zero_cap(comp, cap)
        for comp, scale in self.zero_cap_scales.items():
            if not _finite(f"zero_cap_scales[{comp}]", scale) > 0:
                raise ValueError(f"zero_cap_scales[{comp!r}] must be strictly positive")

    def _check_zero_cap(self, component: str, cap: float) -> None:
        if cap == 0.0:
            scale = self.zero_cap_scales.get(component)
            if scale is None or not _finite(f"zero_cap_scales[{component}]", scale) > 0:
                raise ValueError(
                    f"component {component!r} has a zero cap and needs a strictly "
                    "positive pre-registered scaling constant in zero_cap_scales (V2)"
                )

    def component_names(self) -> tuple[tuple[str, str], ...]:
        """Canonical (category, component) enumeration, deterministically ordered."""
        out: list[tuple[str, str]] = []
        out += [("slo", app) for app in sorted(self.slo_targets)]
        out.append(("goodput", "goodput"))
        out += [("drop_timeout", c) for c in ("drop", "timeout")]
        out += [("disruption", c) for c in sorted(self.disruption_caps)]
        return tuple(out)


@dataclass(frozen=True)
class InstanceOutcome:
    """What one (scenario, seed, bundle) instance produced for one candidate.

    The energy side is a sealed :class:`~sim.energy.boundary.VerifiedInstanceEnergy`
    token — produced only by ``BoundaryCertificate.verify_covers`` (R2-1): the
    instance identity, closed-boundary energy, and the policy-independent V1/W2
    bounds (I, M) all come from the verified token, never as caller-typed bare
    floats, so an outcome cannot cite an energy that escaped its certified boundary.
    The token seal is re-verified here at construction.
    """

    verified: VerifiedInstanceEnergy
    slo_achieved: Mapping[str, float]
    goodput: float
    drop_rate: float
    timeout_rate: float
    disruption: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.verified, VerifiedInstanceEnergy):
            raise ValueError(
                "InstanceOutcome requires a VerifiedInstanceEnergy token from "
                "BoundaryCertificate.verify_covers (R2-1) — bare energy floats are "
                "not accepted"
            )
        self.verified.verify_seal()
        # Totality guard (V2/R3-4): every achieved metric must be finite AND within
        # its physical domain — satisfaction/goodput/rates are fractions in [0, 1],
        # disruption components are non-negative counts/amounts. A NaN or negative
        # rate would corrupt shortfalls and the total order downstream. The
        # mappings are snapshotted so later caller-side mutation cannot bypass this.
        object.__setattr__(self, "slo_achieved", _frozen("slo_achieved", self.slo_achieved))
        object.__setattr__(self, "disruption", _frozen("disruption", self.disruption))
        for app, v in self.slo_achieved.items():
            _fraction(f"slo_achieved[{app}]", v)
        _fraction("goodput", self.goodput)
        _fraction("drop_rate", self.drop_rate)
        _fraction("timeout_rate", self.timeout_rate)
        for comp, v in self.disruption.items():
            if _finite(f"disruption[{comp}]", v) < 0:
                raise ValueError(f"disruption[{comp!r}] must be >= 0")

    # -- identity and energy, all from the verified token -----------------------
    @property
    def scenario_id(self) -> str:
        return self.verified.scenario_id

    @property
    def seed(self) -> int:
        return self.verified.seed

    @property
    def bundle_id(self) -> str:
        return self.verified.bundle_id

    @property
    def energy_j(self) -> float:
        return self.verified.energy_j

    @property
    def idle_floor_j(self) -> float:
        return self.verified.idle_floor_j

    @property
    def ceiling_j(self) -> float:
        return self.verified.ceiling_j

    def key(self) -> tuple[str, int]:
        return (self.scenario_id, self.seed)


def primitive_shortfalls(
    spec: ConstraintSpec, outcome: InstanceOutcome
) -> dict[tuple[str, str], float]:
    """Every primitive component's normalized shortfall for one instance (V2)."""
    out: dict[tuple[str, str], float] = {}
    for app in sorted(spec.slo_targets):
        if app not in outcome.slo_achieved:
            raise ValueError(f"outcome missing SLO achievement for app {app!r}")
        out[("slo", app)] = ge_shortfall(spec.slo_targets[app], outcome.slo_achieved[app])
    out[("goodput", "goodput")] = ge_shortfall(spec.goodput_target, outcome.goodput)
    out[("drop_timeout", "drop")] = le_shortfall(
        spec.drop_cap, outcome.drop_rate, spec.zero_cap_scales.get("drop")
    )
    out[("drop_timeout", "timeout")] = le_shortfall(
        spec.timeout_cap, outcome.timeout_rate, spec.zero_cap_scales.get("timeout")
    )
    for comp in sorted(spec.disruption_caps):
        if comp not in outcome.disruption:
            raise ValueError(f"outcome missing disruption component {comp!r}")
        out[("disruption", comp)] = le_shortfall(
            spec.disruption_caps[comp], outcome.disruption[comp], spec.zero_cap_scales.get(comp)
        )
    return out


# ------------------------------------------------------------------ aggregation
@dataclass(frozen=True)
class CandidateFitness:
    """The complete fitness record; ``sort_key()`` realizes the total order.

    Lower is better everywhere (feasible fitness is normalized energy; infeasibility
    components are shortfalls)."""

    feasible: bool
    infeasibility_vector: tuple[float, float, float, float]
    summed_primitive_shortfall: float
    normalized_energy: float | None  # mean E/I over instances; None when infeasible
    proposal_seq: int
    n_instances: int
    n_bundles: int

    def sort_key(self) -> tuple:
        if self.feasible:
            return (0, self.normalized_energy, self.proposal_seq)
        return (
            1,
            *self.infeasibility_vector,
            self.summed_primitive_shortfall,
            self.proposal_seq,
        )


def evaluate_candidate(
    spec: ConstraintSpec,
    outcomes: list[InstanceOutcome] | tuple[InstanceOutcome, ...],
    *,
    development_bundle_ids: tuple[str, ...],
    instance_keys: tuple[tuple[str, int], ...],
    expected_certificate_seals: Mapping[tuple[str, int, str], str],
    proposal_seq: int,
) -> CandidateFitness:
    """Aggregate one candidate's instance outcomes into the V1/V2/W3 fitness.

    ``instance_keys`` is the frozen (scenario_id, seed) grid every candidate in a
    campaign is scored on (pinned in P2); ``outcomes`` must cover **exactly** the
    cartesian product ``instance_keys × development_bundle_ids`` — no cherry-picked
    subsets, no extras (plan §4.1: feasibility quantifies over *every* development
    bundle on *every* registered instance, and the aggregation axes are only
    comparable across candidates when the grid is identical — R1-7).

    ``expected_certificate_seals`` is the P2/campaign registry for that exact
    cartesian grid. A token's self-seal proves only that the token was not edited;
    fitness also requires that its issuing certificate is the registered certificate
    for ``(scenario, seed, bundle)`` (R4-1).
    """
    if isinstance(proposal_seq, bool) or not isinstance(proposal_seq, int) or proposal_seq < 0:
        raise ValueError("proposal_seq must be the canonical creation sequence number (W3)")
    bundle_set = frozenset(development_bundle_ids)
    if len(development_bundle_ids) != len(bundle_set) or not bundle_set:
        raise ValueError("development_bundle_ids must be non-empty and unique")
    expected_keys = frozenset(instance_keys)
    if len(instance_keys) != len(expected_keys) or not expected_keys:
        raise ValueError("instance_keys must be a non-empty, duplicate-free grid")
    for k in expected_keys:
        if (
            not isinstance(k, tuple)
            or len(k) != 2
            or not isinstance(k[0], str)
            or not isinstance(k[1], int)
            or isinstance(k[1], bool)
        ):
            raise ValueError(f"instance key {k!r} must be a (scenario_id, seed) pair")
    if not outcomes:
        raise ValueError("no instance outcomes supplied")
    if not isinstance(expected_certificate_seals, Mapping):
        raise ValueError("expected_certificate_seals must be a mapping (R4-1)")
    expected_certificate_keys = {
        (scenario, seed, bundle) for scenario, seed in expected_keys for bundle in bundle_set
    }
    if set(expected_certificate_seals) != expected_certificate_keys:
        raise ValueError(
            "expected_certificate_seals must cover exactly the frozen "
            "instance x development-bundle grid (R4-1)"
        )
    for key, seal in expected_certificate_seals.items():
        if not isinstance(seal, str) or len(seal) != 64:
            raise ValueError(f"registered certificate seal for {key!r} is not a digest")

    # Group by scenario x seed instance; the groups must equal the frozen grid and
    # each group must cover every development bundle exactly once.
    groups: dict[tuple[str, int], dict[str, InstanceOutcome]] = {}
    for o in outcomes:
        o.verified.verify_seal()  # R2-1: only certified-boundary energies are scored
        if o.bundle_id not in bundle_set:
            raise ValueError(f"outcome cites unknown bundle {o.bundle_id!r}")
        if o.key() not in expected_keys:
            raise ValueError(f"outcome cites instance {o.key()} outside the frozen grid (R1-7)")
        certificate_key = (o.scenario_id, o.seed, o.bundle_id)
        expected_seal = expected_certificate_seals[certificate_key]
        if o.verified.certificate_seal != expected_seal:
            raise ValueError(
                f"outcome {certificate_key!r} was verified by an unregistered "
                "boundary certificate (R4-1)"
            )
        per = groups.setdefault(o.key(), {})
        if o.bundle_id in per:
            raise ValueError(f"duplicate outcome for {o.key()} bundle {o.bundle_id}")
        per[o.bundle_id] = o
    missing_keys = expected_keys - groups.keys()
    if missing_keys:
        raise ValueError(
            f"candidate not evaluated on frozen instances {sorted(missing_keys)} — "
            "partial grids are not scoreable (R1-7)"
        )
    for key, per in groups.items():
        missing = bundle_set - per.keys()
        if missing:
            raise ValueError(f"instance {key} missing bundles {sorted(missing)}")

    components = spec.component_names()
    ordered_keys = sorted(groups)
    per_instance_category: list[dict[str, float]] = []
    per_instance_primitive_sum: list[float] = []
    for key in ordered_keys:
        per = groups[key]
        # Axis 1: per primitive component, max across development bundles.
        comp_max: dict[tuple[str, str], float] = dict.fromkeys(components, 0.0)
        for bundle_id in sorted(per):
            for comp, v in primitive_shortfalls(spec, per[bundle_id]).items():
                comp_max[comp] = max(comp_max[comp], v)
        # W3 tie-break input: primitive sums *before* within-category reduction.
        per_instance_primitive_sum.append(math.fsum(comp_max[c] for c in components))
        # Axis 2: max within each constraint category.
        cat: dict[str, float] = dict.fromkeys(CATEGORY_ORDER, 0.0)
        for (category, _), v in comp_max.items():
            cat[category] = max(cat[category], v)
        per_instance_category.append(cat)

    # Axis 3: mean across scenario x seed instances -> the 4-vector.
    n = len(ordered_keys)
    vector = tuple(math.fsum(c[cat] for c in per_instance_category) / n for cat in CATEGORY_ORDER)
    # W3 tie-break, literal: the *total sum* of primitive shortfalls over the frozen
    # grid — not a mean (comparable across candidates because the grid is enforced
    # identical above; R1-7).
    summed = math.fsum(per_instance_primitive_sum)
    feasible = summed == 0.0

    normalized_energy = None
    if feasible:
        # V1: mean over development scenario x bundle instances of E / instance idle floor.
        ratios = [o.energy_j / o.idle_floor_j for o in outcomes]
        normalized_energy = math.fsum(ratios) / len(ratios)

    return CandidateFitness(
        feasible=feasible,
        infeasibility_vector=vector,  # type: ignore[arg-type]
        summed_primitive_shortfall=summed,
        normalized_energy=normalized_energy,
        proposal_seq=proposal_seq,
        n_instances=n,
        n_bundles=len(bundle_set),
    )


def rank_candidates(fitnesses: list[CandidateFitness]) -> list[CandidateFitness]:
    """Best-first ranking under the total order (deterministic: the proposal sequence
    number breaks every residual tie). Duplicate sequence numbers are rejected —
    with them, exact ties would fall back to caller input order, which can be
    asynchronous completion order (W3 forbids that; R1-7)."""
    seqs = [f.proposal_seq for f in fitnesses]
    if len(seqs) != len(set(seqs)):
        raise ValueError(
            "duplicate proposal_seq values — the canonical creation sequence number "
            "must be unique per candidate (W3), otherwise the order is not total"
        )
    return sorted(fitnesses, key=CandidateFitness.sort_key)


# ------------------------------------------------------------------ regret + penalty
def feasible_regret(energy_j: float, reference_energy_j: float, idle_floor_j: float) -> float:
    """Checkpoint regret (plan §6.3, 1b): (incumbent − reference) / instance idle floor."""
    if not idle_floor_j > 0:
        raise ValueError("idle floor must be strictly positive (V1)")
    return (energy_j - reference_energy_j) / idle_floor_j


def penalty_regret(ceiling_j: float, reference_energy_j: float, idle_floor_j: float) -> float:
    """P(instance) = (M − reference)/I + 1 (W2). Strictly exceeds any feasible regret
    on the instance because every policy's closed-boundary energy E ≤ M (ceiling
    dominance) implies (E − ref)/I ≤ (M − ref)/I < P. Finite and model-derived — not
    an arbitrary constant."""
    if not idle_floor_j > 0:
        raise ValueError("idle floor must be strictly positive (V1)")
    if not math.isfinite(ceiling_j):
        raise ValueError("ceiling M must be finite (W2: complete closed boundary)")
    return (ceiling_j - reference_energy_j) / idle_floor_j + 1.0


def penalty_dominates(
    energy_j: float, ceiling_j: float, reference_energy_j: float, idle_floor_j: float
) -> bool:
    """The W2 dominance property for one instance: any policy energy within the
    ceiling produces a regret strictly below the penalty. Exposed so tests and P2
    sensitivity analyses assert it rather than assume it."""
    if energy_j > ceiling_j:
        raise ValueError("energy exceeds the ceiling — W2 precondition violated upstream")
    return feasible_regret(energy_j, reference_energy_j, idle_floor_j) < penalty_regret(
        ceiling_j, reference_energy_j, idle_floor_j
    )
