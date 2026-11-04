"""Workload→capacity mapping (TODO 3.2; plan §4.1, correction T4).

**This module is a stated modeling assumption, not a consistency guarantee.** The
plan's earlier "internally consistent by construction" claim is withdrawn (T4): the
construction *rules* below are published and applied consistently, but the transfer
from benchmark domain to microservice domain is an assumption that goes into the
paper's threats-to-validity section. The enumerated limitations are in
:data:`LIMITATIONS` so the paper can cite them mechanically.

The rules
---------

Node capacity is expressed in **abstract capacity units** — operationally the
simulator's millicores (kube resource semantics; see ``sim/spec.py``).

1. **Domain-scoped reference anchoring (primary rule):** throughput-based capacity
   scaling is only valid **within one benchmark domain** — SPECpower ``ssj_ops`` and
   HPL GFLOPS are never numerically compared. Each domain that contributes machines
   declares its own :class:`DomainAnchor`: one pinned machine assigned a
   pre-registered capacity. Every other machine publishing a throughput point in the
   *same* domain gets ``capacity = anchor_capacity × (peak_throughput /
   anchor_peak_throughput)``, rounded to the nearest millicore. A machine publishing
   a throughput point in a domain with **no registered anchor is rejected**, never
   silently mapped. For SPECpower machines the throughput point is the machine's own
   calibrated 100%-target-load ssj_ops — taken from the *same published result* as
   the power curve, which is what keeps the (power curve, capacity) pair a joint
   whole-machine tuple; for the ARM boards it is the published HPL GFLOPS figure
   from the same characterization paper as the power figures.
2. **Load coordinate:** the machine's measured normalized throughput (each published
   row's throughput divided by the machine's own peak) is used as the utilization
   coordinate of the node power curve. This is an assumption — benchmark throughput
   fractions are not CPU-utilization readings.
3. **Core-count fallback (weaker rule):** boards whose source publishes **no**
   benchmark throughput point at all (e.g. a fitted power-vs-CPU-load model) get
   ``capacity = cores × millicores_per_core_fallback`` (kube-style allocatable
   millicores). This is explicitly the weaker rule: it assumes nothing about
   throughput comparability, only that a core is scheduled as 1000 millicores.

All rules are deterministic, published in the artifact, and frozen in P2 with the
bundle lists (anchor choices included).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from sim.energy.sources import ArmSbcMachine, SpecPowerMachine

_ANCHOR_SEAL_TAG = "domain-anchor-seal-v1:"


def _anchor_seal(payload: dict) -> str:
    return hashlib.sha256(
        (
            _ANCHOR_SEAL_TAG
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ).encode()
    ).hexdigest()


#: Threats-to-validity inputs (T4). Referenced by the paper; keep enumerable.
LIMITATIONS: tuple[str, ...] = (
    "Benchmark-domain transfer: SPECpower_ssj2008 exercises a server-side Java "
    "transactional workload and HPL a dense linear-algebra kernel; microservice RPC "
    "serving has different instruction mix, memory behavior, and I/O — throughput "
    "ratios between machines need not transfer.",
    "Throughput-linearity: capacity units scale linearly in the single published peak "
    "throughput point; no per-load-level capacity correction is applied.",
    "Load-coordinate identification: the measured normalized-throughput fraction is "
    "used as the power-curve utilization coordinate; it is not a CPU-utilization "
    "measurement.",
    "Cross-domain anchor commensurability: each benchmark domain's anchor capacity is "
    "a declared constant (millicores per unit of that domain's throughput); capacities "
    "are throughput-faithful within a domain, but the *relative* scale between the "
    "SPECpower domain and the HPL domain is an assumption, so x86-calibrated service "
    "demands are not throughput-accurate on far-edge classes.",
    "Core-count fallback: boards whose source publishes no benchmark throughput "
    "point get cores × 1000 millicores (kube-allocatable convention) — a scheduling "
    "convention, not a throughput statement.",
    "Memory capacity is taken from kube-style declared allocatable values, not from "
    "any benchmark; only CPU capacity is throughput-anchored.",
    "Reference-class choice: capacities within a domain are relative to its declared "
    "anchor machine; a different anchor rescales that domain's capacities by a common "
    "factor (within-domain rankings under fixed workloads are unaffected, absolute "
    "headroom is).",
)


@dataclass(frozen=True)
class DomainAnchor:
    """One benchmark domain's pre-registered reference point (plan §4.1).

    Build anchors with :meth:`from_machine` (domain and peak taken from the cited
    entry's own published values) and validate mappings against the registry with
    :func:`validate_anchor` / :func:`validate_mapping` (R2-8: a hand-typed anchor
    whose domain/peak diverge from the cited source is rejected, not silently used).
    Only ``reference_capacity_millicores`` is a declared constant — the P2-frozen
    choice; everything else is source data.
    """

    domain: str
    reference_machine_id: str
    reference_peak_throughput_ops: float
    reference_capacity_millicores: int
    seal: str = ""

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("anchor domain required")
        if not self.reference_machine_id:
            raise ValueError("anchor reference_machine_id required")
        if not self.reference_peak_throughput_ops > 0:
            raise ValueError("anchor reference peak throughput must be > 0")
        v = self.reference_capacity_millicores
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError("anchor reference_capacity_millicores must be a positive int")
        if not self.seal:
            raise ValueError(
                "DomainAnchor carries no seal — construct it through "
                "DomainAnchor.from_machine (the source-entry-backed factory; a "
                "hand-typed anchor cannot price capacities — R2-8/R3)"
            )

    def verify_seal(self) -> None:
        if self.seal != _anchor_seal(self.payload()):
            raise ValueError(
                f"anchor for domain {self.domain!r} has a seal mismatch — its "
                "fields were changed after construction (R2-8/R3)"
            )

    @staticmethod
    def from_machine(
        machine: SpecPowerMachine | ArmSbcMachine, capacity_millicores: int
    ) -> DomainAnchor:
        """The only seal-producing anchor factory (R2-8/R3): domain and peak come
        from the machine entry's own published throughput point, never typed by
        hand; only the capacity is the declared (P2-frozen) constant."""
        domain, peak = _domain_and_peak(machine)
        if domain is None or peak is None:
            raise ValueError(
                f"{machine.machine_id} publishes no benchmark throughput point and "
                "cannot anchor a domain"
            )
        payload = {
            "domain": domain,
            "reference_machine_id": machine.machine_id,
            "reference_peak_throughput_ops": peak,
            "reference_capacity_millicores": capacity_millicores,
        }
        return DomainAnchor(
            domain=domain,
            reference_machine_id=machine.machine_id,
            reference_peak_throughput_ops=peak,
            reference_capacity_millicores=capacity_millicores,
            seal=_anchor_seal(payload),
        )

    def payload(self) -> dict:
        """Canonical anchor payload — materialized in every machine-tuple manifest
        (R2-8: the registered artifact carries the validated anchor, not just a
        rule name)."""
        return {
            "domain": self.domain,
            "reference_machine_id": self.reference_machine_id,
            "reference_peak_throughput_ops": self.reference_peak_throughput_ops,
            "reference_capacity_millicores": self.reference_capacity_millicores,
        }


def validate_anchor(anchor: DomainAnchor, machine: SpecPowerMachine | ArmSbcMachine) -> None:
    """Registry-grounded anchor check (R2-8): the anchor's seal must verify and its
    machine, domain, and peak must match the cited entry's published values exactly."""
    anchor.verify_seal()
    if machine.machine_id != anchor.reference_machine_id:
        raise ValueError(
            f"anchor cites {anchor.reference_machine_id!r} but was validated against "
            f"{machine.machine_id!r}"
        )
    domain, peak = _domain_and_peak(machine)
    if domain != anchor.domain or peak != anchor.reference_peak_throughput_ops:
        raise ValueError(
            f"anchor for {anchor.reference_machine_id} diverges from the cited source: "
            f"anchor says ({anchor.domain!r}, {anchor.reference_peak_throughput_ops}), "
            f"the pinned entry publishes ({domain!r}, {peak}) — anchors are built with "
            "DomainAnchor.from_machine, never hand-typed"
        )


@dataclass(frozen=True)
class CapacityMapping:
    """The pre-registered scaling (plan §4.1): one declared anchor per benchmark
    domain plus the core-count fallback constant. This dataclass is the artifact
    that gets frozen in P2 (anchor machines, anchor capacities, fallback constant).
    """

    anchors: tuple[DomainAnchor, ...]
    millicores_per_core_fallback: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(self.anchors))
        if not self.anchors:
            raise ValueError("at least one domain anchor is required")
        for a in self.anchors:
            a.verify_seal()  # R3: a field-edited anchor never enters a mapping
        domains = [a.domain for a in self.anchors]
        if len(domains) != len(set(domains)):
            raise ValueError("one anchor per benchmark domain")
        v = self.millicores_per_core_fallback
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError("millicores_per_core_fallback must be a positive int")

    @staticmethod
    def anchored_on(
        machine: SpecPowerMachine | ArmSbcMachine, capacity_millicores: int
    ) -> CapacityMapping:
        """Single-domain mapping anchored on one machine's own published throughput."""
        return CapacityMapping(anchors=(DomainAnchor.from_machine(machine, capacity_millicores),))

    def with_anchor(self, anchor: DomainAnchor) -> CapacityMapping:
        return CapacityMapping(
            anchors=(*self.anchors, anchor),
            millicores_per_core_fallback=self.millicores_per_core_fallback,
        )

    def anchor_for(self, domain: str) -> DomainAnchor | None:
        for a in self.anchors:
            # R4-4: frozen dataclasses can still be field-edited via object.__setattr__.
            # Every pricing path therefore rechecks the source-bound anchor seal.
            a.verify_seal()
            if a.domain == domain:
                return a
        return None

    # -- rule 1: domain-scoped throughput anchoring ---------------------------
    def capacity_from_throughput(self, domain: str, peak_throughput_ops: float) -> int:
        if not peak_throughput_ops > 0:
            raise ValueError("peak throughput must be > 0")
        anchor = self.anchor_for(domain)
        if anchor is None:
            raise ValueError(
                f"no capacity anchor registered for benchmark domain {domain!r} — "
                "throughput points are never compared across domains (register an "
                "anchor for this domain or exclude the machine)"
            )
        ratio = peak_throughput_ops / anchor.reference_peak_throughput_ops
        out = round(anchor.reference_capacity_millicores * ratio)
        return max(1, int(out))

    def capacity_millicores(self, machine: SpecPowerMachine | ArmSbcMachine) -> int:
        """Apply the applicable rule; the returned value is only meaningful together
        with the machine's own power curve (joint-tuple discipline)."""
        domain, peak = _domain_and_peak(machine)
        if domain is not None and peak is not None:
            return self.capacity_from_throughput(domain, peak)
        return self.capacity_from_cores(machine.cores)

    # -- rule 3: core-count fallback -------------------------------------------
    def capacity_from_cores(self, cores: int) -> int:
        if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
            raise ValueError("cores must be a positive int")
        return cores * self.millicores_per_core_fallback

    def rule_used(self, machine: SpecPowerMachine | ArmSbcMachine) -> str:
        domain, peak = _domain_and_peak(machine)
        if domain is not None and peak is not None:
            return f"throughput-anchored[{domain}]"
        return "core-count-fallback"

    def anchor_used(self, machine: SpecPowerMachine | ArmSbcMachine) -> DomainAnchor | None:
        """The anchor that priced this machine's capacity (None → core-count fallback).
        Materialized into the machine-tuple manifest (R2-8)."""
        domain, peak = _domain_and_peak(machine)
        if domain is not None and peak is not None:
            anchor = self.anchor_for(domain)
            if anchor is None:
                raise ValueError(f"no capacity anchor registered for domain {domain!r}")
            return anchor
        return None

    def payload(self) -> dict:
        """Canonical mapping payload (anchors + fallback constant) — the P2-frozen
        capacity-mapping artifact (R2-8)."""
        for anchor in self.anchors:
            anchor.verify_seal()
        return {
            "anchors": [a.payload() for a in sorted(self.anchors, key=lambda a: a.domain)],
            "millicores_per_core_fallback": self.millicores_per_core_fallback,
        }


def validate_mapping(
    mapping: CapacityMapping,
    entries: dict[str, SpecPowerMachine | ArmSbcMachine],
) -> None:
    """Every anchor in the mapping must be registry-backed (R2-8): its reference
    machine must exist among the supplied entries and its domain/peak must match that
    entry's published values. Run by ``validate_bundle`` and the default-mapping
    builder before any capacity is consumed."""
    for anchor in mapping.anchors:
        machine = entries.get(anchor.reference_machine_id)
        if machine is None:
            raise ValueError(
                f"anchor for domain {anchor.domain!r} cites unknown machine "
                f"{anchor.reference_machine_id!r}"
            )
        validate_anchor(anchor, machine)


def _domain_and_peak(
    machine: SpecPowerMachine | ArmSbcMachine,
) -> tuple[str | None, float | None]:
    if isinstance(machine, SpecPowerMachine):
        return machine.benchmark_domain, machine.peak_ssj_ops
    return machine.benchmark_domain, machine.peak_throughput_ops
