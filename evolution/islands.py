"""Constraint-profile islands and lightweight elite criteria (TODO 5.6).

The scientific survivor path always uses ``CandidateFitness.sort_key`` unchanged.
LOC and calibrated decision-cost penalties define a separate elite view over the full
archive; they cannot make an infeasible program beat a feasible one and never alter
the registered Phase-3 fitness record.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import placementscript as ps

from sim.energy import (
    CandidateFitness,
    ConstraintSpec,
    build_development_bundles,
    load_pinned_sources,
)
from sim.scheduler import filter_feasible
from sim.snapshot import SnapshotLibrary, initial_snapshot
from sim.spec import load_episode_config

from .evaluation import TaskEvaluationConfig, config_for_development_bundle


@dataclass(frozen=True)
class IslandProfile:
    island_id: str
    slo_target_multiplier: float
    goodput_target_multiplier: float
    drop_cap_multiplier: float
    timeout_cap_multiplier: float
    disruption_cap_multiplier: float

    def __post_init__(self) -> None:
        if not self.island_id:
            raise ValueError("island_id must be non-empty")
        for field in (
            "slo_target_multiplier",
            "goodput_target_multiplier",
            "drop_cap_multiplier",
            "timeout_cap_multiplier",
            "disruption_cap_multiplier",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")

    def constraints(self, base: ConstraintSpec) -> ConstraintSpec:
        return ConstraintSpec(
            slo_targets={
                app: target * self.slo_target_multiplier for app, target in base.slo_targets.items()
            },
            goodput_target=base.goodput_target * self.goodput_target_multiplier,
            drop_cap=base.drop_cap * self.drop_cap_multiplier,
            timeout_cap=base.timeout_cap * self.timeout_cap_multiplier,
            disruption_caps={
                name: cap * self.disruption_cap_multiplier
                for name, cap in base.disruption_caps.items()
            },
            zero_cap_scales=base.zero_cap_scales,
        )


@dataclass(frozen=True)
class LatencyCalibration:
    calibration_id: str
    status: str
    intercept_us: float
    us_per_fuel: float

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("decision latency calibration id is required")
        if self.status not in {"development-provisional", "p1-frozen"}:
            raise ValueError("unknown decision latency calibration status")
        for field in ("intercept_us", "us_per_fuel"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")


@dataclass(frozen=True)
class ArchiveCriteria:
    loc_soft_limit: int
    loc_scale: float
    decision_latency_soft_us: float
    decision_latency_scale_us: float
    loc_weight: float
    decision_latency_weight: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.loc_soft_limit, int)
            or isinstance(self.loc_soft_limit, bool)
            or self.loc_soft_limit < 1
        ):
            raise ValueError("loc_soft_limit must be a positive int")
        for field in (
            "loc_scale",
            "decision_latency_soft_us",
            "decision_latency_scale_us",
            "loc_weight",
            "decision_latency_weight",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")


@dataclass(frozen=True)
class IslandsConfig:
    profiles: tuple[IslandProfile, ...]
    survivor_selection: str
    elite_view_selection: str
    archive: ArchiveCriteria
    latency_calibration: LatencyCalibration

    def __post_init__(self) -> None:
        ids = [profile.island_id for profile in self.profiles]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("island profiles must be non-empty and unique")
        if self.survivor_selection != "phase3-lexicographic":
            raise ValueError("survivor selection must preserve the Phase-3 total order")
        if self.elite_view_selection != "feasibility-complexity-fitness":
            raise ValueError("unknown elite view selection strategy")


@dataclass(frozen=True)
class DecisionCost:
    source_sha256: str
    probe_sha256: str
    score_batch_fuel: int
    score_batch_memory_bytes: int
    calibrated_latency_us: float
    calibration_id: str
    calibration_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "probe_sha256": self.probe_sha256,
            "score_batch_fuel": self.score_batch_fuel,
            "score_batch_memory_bytes": self.score_batch_memory_bytes,
            "calibrated_latency_us": self.calibrated_latency_us,
            "calibration_id": self.calibration_id,
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True)
class ArchiveAssessment:
    loc: int
    decision_cost: DecisionCost
    loc_penalty: float
    decision_latency_penalty: float
    total_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "loc": self.loc,
            "decision_cost": self.decision_cost.to_dict(),
            "loc_penalty": self.loc_penalty,
            "decision_latency_penalty": self.decision_latency_penalty,
            "total_penalty": self.total_penalty,
        }

    def elite_key(self, fitness: CandidateFitness) -> tuple[Any, ...]:
        scientific = fitness.sort_key()
        # Feasibility remains the first discriminator. This is an elite-view key only;
        # the campaign survivor key below is the unmodified scientific order.
        return (scientific[0], self.total_penalty, *scientific[1:])


def load_islands_config(path: str | Path) -> IslandsConfig:
    raw = json.loads(Path(path).read_text())["islands"]
    profiles = tuple(IslandProfile(**item) for item in raw["profiles"])
    return IslandsConfig(
        profiles=profiles,
        survivor_selection=raw["survivor_selection"],
        elite_view_selection=raw["elite_view_selection"],
        archive=ArchiveCriteria(**raw["archive_criteria"]),
        latency_calibration=LatencyCalibration(**raw["decision_latency_calibration"]),
    )


def policy_loc(source: str) -> int:
    """Physical implementation LOC, excluding comments, blanks, and identity fields."""
    lines = source.splitlines()
    has_blocks = any("EVOLVE-BLOCK-START" in line for line in lines)
    inside = not has_blocks
    count = 0
    for line in lines:
        if "EVOLVE-BLOCK-START" in line:
            inside = True
            continue
        if "EVOLVE-BLOCK-END" in line:
            inside = False
            continue
        stripped = line.strip()
        if inside and stripped and not stripped.startswith("#"):
            if not stripped.startswith(("POLICY_NAME", "POLICY_VERSION")):
                count += 1
    return count


def measure_decision_cost(
    source: str,
    task: TaskEvaluationConfig,
    calibration: LatencyCalibration,
) -> DecisionCost:
    """Measure deterministic interpreter work on the canonical initial decision probe."""
    policy = ps.Policy(source)
    scenario = task.scenarios[0]
    base = load_episode_config(scenario.path)
    bundle = build_development_bundles(load_pinned_sources())[0]
    config = config_for_development_bundle(base, bundle)
    library = SnapshotLibrary(
        config=config,
        snapshots=(initial_snapshot(config),),
        meta={"role": "decision-cost-probe", "bundle_id": bundle.bundle_id},
    )
    plain = library.snapshots[0].to_dict()["ctx"]
    if not plain["pods"]:
        raise ValueError("decision-cost probe needs at least one pending pod")
    pod = plain["pods"][0]
    feasible_ids = filter_feasible(pod, plain["nodes"])
    if not feasible_ids:
        raise ValueError("decision-cost probe has no feasible node")
    nodes_by_id = {node["id"]: node for node in plain["nodes"]}
    nodes = [nodes_by_id[node_id] for node_id in feasible_ids]
    _scores, fuel, memory = policy.score_batch_cost(pod, nodes, plain)
    calibrated_us = calibration.intercept_us + fuel * calibration.us_per_fuel
    return DecisionCost(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        probe_sha256=library.digest(),
        score_batch_fuel=fuel,
        score_batch_memory_bytes=memory,
        calibrated_latency_us=calibrated_us,
        calibration_id=calibration.calibration_id,
        calibration_status=calibration.status,
    )


def assess_for_archive(
    source: str,
    decision_cost: DecisionCost,
    criteria: ArchiveCriteria,
) -> ArchiveAssessment:
    loc = policy_loc(source)
    loc_penalty = (
        max(0.0, (loc - criteria.loc_soft_limit) / criteria.loc_scale) * criteria.loc_weight
    )
    latency_penalty = (
        max(
            0.0,
            (decision_cost.calibrated_latency_us - criteria.decision_latency_soft_us)
            / criteria.decision_latency_scale_us,
        )
        * criteria.decision_latency_weight
    )
    return ArchiveAssessment(
        loc=loc,
        decision_cost=decision_cost,
        loc_penalty=loc_penalty,
        decision_latency_penalty=latency_penalty,
        total_penalty=loc_penalty + latency_penalty,
    )


def survivor_key(fitness: CandidateFitness) -> tuple[Any, ...]:
    """The only survivor-selection key: exact Phase-3 fitness, unchanged."""
    return fitness.sort_key()
