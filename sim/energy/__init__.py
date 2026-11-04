"""sim.energy — closed energy boundary (Phase 2) + EnergyModelBundles (Phase 3).

Phase 2's boundary machinery (``EnergyMeter``, ``idle_floor_j``, ``ceiling_j``, the
three never-silently-summed categories) lives in ``sim.energy.boundary`` and is
re-exported here unchanged, so every existing ``from sim.energy import ...`` keeps
working.

Phase 3 (plan §4, TODO 3.1–3.5) adds, as submodules:

- ``sim.energy.sources``   — pinned published sources with per-number provenance (3.1)
- ``sim.energy.capacity``  — workload→capacity mapping, a stated assumption (3.2, T4)
- ``sim.energy.bundles``   — EnergyModelBundle assembly + dev/held-out split (3.3)
- ``sim.energy.fitness``   — V1/V2/W2/W3 fitness aggregation for Phase 5 (3.4)
- ``sim.energy.gsa``       — stratified dependency-preserving GSA harness (3.5)
- ``sim.energy.holdout_gate`` — run-once held-out bundle driver (3.3, T4)

See ``sim/energy/README.md`` for the architecture and the Phase 5 contract.
"""

from sim.energy.boundary import (
    KB,
    MIB,
    BoundaryBounds,
    BoundaryCertificate,
    BoundaryWeights,
    EnergyFigure,
    EnergyMeter,
    EnergyReport,
    ProvenanceRecord,
    VerifiedInstanceEnergy,
    ceiling_j,
    certify_boundary,
    cluster_sha256,
    idle_floor_j,
)
from sim.energy.bundles import (
    EnergyModelBundle,
    MachineTuple,
    NodeClassBinding,
    build_development_bundles,
    build_held_out_bundles,
    bundle_from_registry,
    default_capacity_mapping,
    machine_tuple_from_registry,
    validate_bundle,
    validate_split,
)
from sim.energy.capacity import CapacityMapping, DomainAnchor, validate_anchor, validate_mapping
from sim.energy.fitness import (
    CandidateFitness,
    ConstraintSpec,
    InstanceOutcome,
    evaluate_candidate,
    feasible_regret,
    penalty_dominates,
    penalty_regret,
    rank_candidates,
)
from sim.energy.registration import (
    EnergyAnalysisRegistration,
    load_energy_analysis_registration,
)
from sim.energy.sources import Provenance, SourceRegistry, load_pinned_sources

# sim.energy.gsa is intentionally NOT imported here: it pulls SALib/numpy sampling
# machinery that the simulator core never needs — import it explicitly.

__all__ = [
    "KB",
    "MIB",
    "BoundaryBounds",
    "BoundaryCertificate",
    "BoundaryWeights",
    "CandidateFitness",
    "CapacityMapping",
    "ConstraintSpec",
    "DomainAnchor",
    "EnergyFigure",
    "EnergyAnalysisRegistration",
    "EnergyMeter",
    "EnergyModelBundle",
    "EnergyReport",
    "InstanceOutcome",
    "MachineTuple",
    "NodeClassBinding",
    "Provenance",
    "ProvenanceRecord",
    "SourceRegistry",
    "VerifiedInstanceEnergy",
    "build_development_bundles",
    "build_held_out_bundles",
    "bundle_from_registry",
    "ceiling_j",
    "certify_boundary",
    "cluster_sha256",
    "default_capacity_mapping",
    "evaluate_candidate",
    "feasible_regret",
    "idle_floor_j",
    "load_pinned_sources",
    "load_energy_analysis_registration",
    "machine_tuple_from_registry",
    "penalty_dominates",
    "penalty_regret",
    "rank_candidates",
    "validate_anchor",
    "validate_bundle",
    "validate_mapping",
    "validate_split",
]
