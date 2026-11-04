"""Authoritative Phase-5 PlacementScript evaluation over development bundles.

The evaluator is deliberately a thin orchestration layer.  Closed-boundary
certification and the C20 total order remain owned by :mod:`sim.energy`; this module
only turns sealed episode results into ``InstanceOutcome`` records and archives the
evidence needed to reproduce that conversion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import placementscript as ps

from sim.actuation import PlacementScriptProposer
from sim.energy import (
    BoundaryBounds,
    BoundaryWeights,
    CandidateFitness,
    ConstraintSpec,
    InstanceOutcome,
    build_development_bundles,
    certify_boundary,
    evaluate_candidate,
    load_pinned_sources,
)
from sim.engine import EpisodeResult, run_episode
from sim.scheduler import PlacementScriptScorer
from sim.spec import ClusterSpec, EpisodeConfig, NodeSpec, load_episode_config

TASK_CONFIG_SCHEMA = "evolved-continuum-task-v1"
CANDIDATE_FAILURE_SCHEMA = "evolved-continuum-candidate-failure-v1"


_PLACEMENTSCRIPT_RUNTIME_FAILURES = {
    "TrapError": "placementscript-runtime-trap",
    "FuelExhausted": "placementscript-fuel-exhausted",
    "OutOfMemory": "placementscript-out-of-memory",
}


def candidate_failure_metadata(
    error: BaseException | str,
    *,
    stage: str,
    policy_component: str = "scorer",
    scenario_id: str | None = None,
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Return the canonical, archive-safe reason for a candidate execution failure."""
    message = str(error)
    exception_type = type(error).__name__ if isinstance(error, BaseException) else "PolicyError"
    for runtime_type in _PLACEMENTSCRIPT_RUNTIME_FAILURES:
        if isinstance(error, BaseException) and type(error).__name__ == runtime_type:
            exception_type = runtime_type
            break
        if f"{runtime_type}:" in message or message.startswith(runtime_type):
            exception_type = runtime_type
            break
    category = _PLACEMENTSCRIPT_RUNTIME_FAILURES.get(
        exception_type, "policy-execution-error"
    )

    failure: dict[str, Any] = {
        "schema": CANDIDATE_FAILURE_SCHEMA,
        "failure_category": category,
        "exception_type": exception_type,
        "message": message,
        "stage": stage,
        "policy_component": policy_component,
    }
    if scenario_id is not None:
        failure["scenario_id"] = scenario_id
    if bundle_id is not None:
        failure["bundle_id"] = bundle_id
    return failure


def _episode_failure_metadata(
    error: BaseException | str,
    *,
    stage: str,
    policy_component: str,
    scenario_id: str,
    seed: int,
    bundle_id: str,
) -> dict[str, Any]:
    """Add the explicit episode identity to canonical policy-failure metadata."""
    failure = candidate_failure_metadata(
        error,
        stage=stage,
        policy_component=policy_component,
        scenario_id=scenario_id,
        bundle_id=bundle_id,
    )
    failure["seed"] = seed
    return failure


class PolicyExecutionError(RuntimeError):
    """A policy fault observed by the simulator and rejected at evaluation boundary."""

    def __init__(self, failure: dict[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__(
            f"{self.failure['failure_category']}: {self.failure['message']}"
        )


def policy_identity_for_source(source: str) -> dict[str, str]:
    """Bind a PlacementScript identity to the exact executable source bytes."""
    policy = ps.Policy(source)
    artifact_sha256 = hashlib.sha256(source.encode()).hexdigest()
    identity_sha256 = hashlib.sha256(
        json.dumps(
            {
                "name": policy.name,
                "version": policy.version,
                "representation": "placementscript-source",
                "artifact_sha256": artifact_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "name": policy.name,
        "version": policy.version,
        "identity_sha256": identity_sha256,
        "artifact_sha256": artifact_sha256,
    }


@dataclass(frozen=True)
class ScenarioGrid:
    scenario_id: str
    path: Path
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class TaskEvaluationConfig:
    scenarios: tuple[ScenarioGrid, ...]
    bundle_ids: tuple[str, ...]
    constraints: ConstraintSpec
    weights: BoundaryWeights
    bounds: BoundaryBounds

    @property
    def instance_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (scenario.scenario_id, seed) for scenario in self.scenarios for seed in scenario.seeds
        )


@dataclass(frozen=True)
class InstanceEvidence:
    scenario_id: str
    seed: int
    bundle_id: str
    episode_digest: str
    certificate_seal: str
    energy_j: float
    idle_floor_j: float
    ceiling_j: float
    slo_achieved: dict[str, float]
    goodput: float
    drop_rate: float
    timeout_rate: float
    disruption: dict[str, float]
    boundary_certificate: dict[str, Any]
    verified_energy: dict[str, Any]
    episode: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PolicyEvaluation:
    fitness: CandidateFitness
    instances: tuple[InstanceEvidence, ...]
    policy_name: str
    policy_version: str
    policy_identity_sha256: str
    policy_artifact_sha256: str

    def __post_init__(self) -> None:
        for label, digest in (
            ("policy_identity_sha256", self.policy_identity_sha256),
            ("policy_artifact_sha256", self.policy_artifact_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    def without_episode_ledgers(self) -> PolicyEvaluation:
        """Same evaluation with horizon-scaled episode ledgers replaced by their lengths.

        Every instance keeps ``episode_digest`` and ``certificate_seal``, so the full episode
        stays pinned; only the in-memory copy of ``records`` / ``service_calls`` /
        ``snapshots`` is dropped. Fitness, energies and policy identity are untouched. Use this
        before retaining an evaluation for the lifetime of a run: on a 420 s episode the three
        ledgers cost 60 to 90 MB of resident memory per evaluation (measured 2026-09-04), which
        is what pushed five concurrent 224.0-work roots past 16 GB of RAM into swap.
        """
        return replace(
            self,
            instances=tuple(
                replace(instance, episode=compact_episode_ledgers(instance.episode))
                for instance in self.instances
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Canonical, JSON-safe public record; the tuple is the fitness authority."""
        return {
            "fitness": {
                "feasible": self.fitness.feasible,
                "infeasibility_vector": list(self.fitness.infeasibility_vector),
                "summed_primitive_shortfall": self.fitness.summed_primitive_shortfall,
                "normalized_energy": self.fitness.normalized_energy,
                "proposal_seq": self.fitness.proposal_seq,
                "n_instances": self.fitness.n_instances,
                "n_bundles": self.fitness.n_bundles,
                "sort_key": list(self.fitness.sort_key()),
                "direction": "lower-is-better",
            },
            "policy": {
                "name": self.policy_name,
                "version": self.policy_version,
                "identity_sha256": self.policy_identity_sha256,
                "artifact_sha256": self.policy_artifact_sha256,
            },
            "instances": [instance.to_dict() for instance in self.instances],
        }


# Per-episode ledgers that scale with the *horizon*, not with anything a retained result is
# for. Shared by the campaign archive writer and by the arms that keep results in memory.
OMITTED_EPISODE_LEDGERS = ("service_calls", "records", "snapshots")


def compact_episode_ledgers(value: Any) -> Any:
    """Recursively replace bulk per-episode ledgers with their lengths (idempotent)."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in OMITTED_EPISODE_LEDGERS and isinstance(item, list):
                out[key] = {
                    "omitted": "horizon-scaled episode ledger",
                    "length": len(item),
                    "pinned_by": "episode_digest + certificate_seal",
                }
            else:
                out[key] = compact_episode_ledgers(item)
        return out
    if isinstance(value, list):
        return [compact_episode_ledgers(item) for item in value]
    return value


def _positive_ints(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON array")
    out = tuple(values)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in out):
        raise ValueError(f"{name} must contain explicit non-negative integer seeds")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return out


def load_task_evaluation_config(path: str | Path) -> TaskEvaluationConfig:
    """Load a development-only task config and fail closed on held-out paths."""
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    if raw.get("schema") != TASK_CONFIG_SCHEMA:
        raise ValueError(f"task config schema must be {TASK_CONFIG_SCHEMA!r}")
    if raw.get("data_role") != "development-only":
        raise ValueError("Phase-5 evolution task may consume development data only")

    repository_root = next(
        (
            parent
            for parent in config_path.parents
            if (parent / "pyproject.toml").is_file() and (parent / "sim").is_dir()
        ),
        None,
    )
    if repository_root is None:
        raise ValueError("task config must live inside an EvolvedContinuum checkout")
    scenarios = []
    seen_scenario_ids: set[str] = set()
    for i, item in enumerate(raw.get("scenarios", [])):
        scenario_id = item.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"scenarios[{i}].id must be a non-empty string")
        if scenario_id in seen_scenario_ids:
            raise ValueError(f"duplicate scenario id {scenario_id!r}")
        seen_scenario_ids.add(scenario_id)
        scenario_path = (repository_root / item["path"]).resolve()
        try:
            relative = scenario_path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError("scenario paths must stay inside the repository") from exc
        lowered = {part.lower() for part in relative.parts}
        if {"acceptance", "private", "holdout"} & lowered:
            raise ValueError("evolution task cannot read acceptance/private/holdout scenarios")
        scenarios.append(
            ScenarioGrid(
                scenario_id=scenario_id,
                path=scenario_path,
                seeds=_positive_ints(f"scenarios[{i}].seeds", item.get("seeds")),
            )
        )
    if not scenarios:
        raise ValueError("task config requires at least one development scenario")

    bundle_ids = tuple(raw.get("development_bundle_ids", ()))
    if not bundle_ids or any(not isinstance(v, str) or not v for v in bundle_ids):
        raise ValueError("development_bundle_ids must contain non-empty strings")
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("development_bundle_ids must be unique")

    constraint_raw = raw["constraints"]
    constraints = ConstraintSpec(
        slo_targets=constraint_raw["slo_targets"],
        goodput_target=constraint_raw["goodput_target"],
        drop_cap=constraint_raw["drop_cap"],
        timeout_cap=constraint_raw["timeout_cap"],
        disruption_caps=constraint_raw["disruption_caps"],
        zero_cap_scales=constraint_raw.get("zero_cap_scales", {}),
    )
    weight_raw = raw["boundary_weights"]
    bound_raw = raw["boundary_bounds"]
    return TaskEvaluationConfig(
        scenarios=tuple(scenarios),
        bundle_ids=bundle_ids,
        constraints=constraints,
        weights=BoundaryWeights(**weight_raw),
        bounds=BoundaryBounds(**bound_raw),
    )


def config_for_development_bundle(base: EpisodeConfig, bundle: Any) -> EpisodeConfig:
    """Bind non-cloud nodes through the bundle's sealed spec+capacity bridge."""
    bindings = bundle.node_class_bindings()
    tier_binding = {1: bindings["fog-server"], 2: bindings["far-edge"]}
    cloud_classes = tuple(
        cls
        for cls in base.cluster.classes
        if any(node.tier == 0 and node.node_class == cls.name for node in base.cluster.nodes)
    )
    nodes: list[NodeSpec] = []
    used_non_cloud: set[str] = set()
    for node in base.cluster.nodes:
        if node.tier == 0:
            nodes.append(node)
            continue
        binding = tier_binding[node.tier]
        binding.machine.verify_seal()
        used_non_cloud.add(binding.spec.name)
        nodes.append(
            replace(
                node,
                node_class=binding.spec.name,
                cpu_capacity=binding.capacity_millicores,
            )
        )
    bundle_classes = tuple(bindings[name].spec for name in sorted(used_non_cloud))
    cluster = ClusterSpec(
        classes=cloud_classes + bundle_classes,
        nodes=tuple(nodes),
        latency_ms=base.cluster.latency_ms,
    )
    return replace(base, cluster=cluster, energy=bundle.energy_coefficients())


def _outcome_from_episode(
    result: EpisodeResult,
    config: EpisodeConfig,
    task: TaskEvaluationConfig,
    *,
    scenario_id: str,
    seed: int,
    bundle_id: str,
    retain_episode_ledgers: bool = True,
) -> tuple[InstanceOutcome, InstanceEvidence, str]:
    result.verify_runtime_origin()
    certificate = certify_boundary(
        config.cluster,
        config.energy,
        config.horizon_s,
        task.weights,
        task.bounds,
        scenario_id=scenario_id,
        seed=seed,
        bundle_id=bundle_id,
    )
    verified = certificate.verify_covers(result.energy)
    templates = {template.name: template for template in config.workload.templates}
    unknown = set(task.constraints.slo_targets) - templates.keys()
    if unknown:
        raise ValueError(
            f"constraint names missing from scenario {scenario_id!r}: {sorted(unknown)}"
        )
    slo: dict[str, float] = {}
    for name in sorted(task.constraints.slo_targets):
        latency_s = templates[name].slo_latency_s
        if latency_s is None:
            raise ValueError(f"template {name!r} has no SLO latency")
        slo[name] = result.slo_satisfaction(templates[name].id, latency_s)
    offered = len(result.records)
    counts = result.status_counts()
    denominator = max(1, offered)
    disruption = {
        name: float(result.rebalancer[name]) for name in sorted(task.constraints.disruption_caps)
    }
    outcome = InstanceOutcome(
        verified=verified,
        slo_achieved=slo,
        goodput=result.goodput(),
        drop_rate=sum(count for status, count in counts.items() if status.startswith("dropped"))
        / denominator,
        timeout_rate=counts.get("timeout", 0) / denominator,
        disruption=disruption,
    )
    canonical = result.to_canonical_dict()
    episode_digest = result.canonical_digest(canonical)
    if not retain_episode_ledgers:
        # Drop the horizon-scaled ledgers *before* this instance joins the accumulating
        # evidence list. The digest above was taken from the complete canonical report,
        # so the full episode stays pinned by `episode_digest` + `certificate_seal`.
        canonical = compact_episode_ledgers(canonical)
    evidence = InstanceEvidence(
        scenario_id=scenario_id,
        seed=seed,
        bundle_id=bundle_id,
        episode_digest=episode_digest,
        certificate_seal=certificate.seal,
        energy_j=verified.energy_j,
        idle_floor_j=verified.idle_floor_j,
        ceiling_j=verified.ceiling_j,
        slo_achieved=slo,
        goodput=outcome.goodput,
        drop_rate=outcome.drop_rate,
        timeout_rate=outcome.timeout_rate,
        disruption=disruption,
        boundary_certificate={**certificate.payload(), "seal": certificate.seal},
        verified_energy={**verified.payload(), "seal": verified.seal},
        episode=canonical,
    )
    return outcome, evidence, certificate.seal


def evaluate_policy_source(
    source: str,
    task: TaskEvaluationConfig,
    *,
    proposal_seq: int,
    require_complete_development_set: bool = True,
    config_transform: Any | None = None,
    retain_episode_ledgers: bool = True,
) -> PolicyEvaluation:
    """Evaluate one PlacementScript candidate on the frozen scenario x bundle grid."""
    policy = ps.Policy(source)
    identity = policy_identity_for_source(source)
    return evaluate_policy_adapter(
        PlacementScriptScorer(policy),
        PlacementScriptProposer(policy),
        task,
        proposal_seq=proposal_seq,
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
        require_complete_development_set=require_complete_development_set,
        config_transform=config_transform,
        retain_episode_ledgers=retain_episode_ledgers,
    )


def _raise_for_policy_execution_failure(
    result: EpisodeResult,
    *,
    scenario_id: str,
    seed: int,
    bundle_id: str,
) -> None:
    scorer_errors = result.ledger.get("scorer_errors", 0)
    if scorer_errors:
        first = next(
            (
                record.get("error", "policy scorer failed without an error record")
                for record in result.decision_log
                if record.get("event") == "scorer_error"
            ),
            "policy scorer failed without an error record",
        )
        raise PolicyExecutionError(
            _episode_failure_metadata(
                first,
                stage="episode",
                policy_component="scorer",
                scenario_id=scenario_id,
                seed=seed,
                bundle_id=bundle_id,
            )
        )

    first = next(
        (
            record.get("reason", "policy proposer execution failed")
            for record in result.rebalancer.get("rejections", [])
            if str(record.get("reason", "")).startswith("proposer_error:")
        ),
        None,
    )
    if first is not None:
        raise PolicyExecutionError(
            _episode_failure_metadata(
                first,
                stage="episode",
                policy_component="proposer",
                scenario_id=scenario_id,
                seed=seed,
                bundle_id=bundle_id,
            )
        )


def evaluate_policy_adapter(
    scorer: Any,
    proposer: Any,
    task: TaskEvaluationConfig,
    *,
    proposal_seq: int,
    policy_name: str,
    policy_version: str,
    policy_identity_sha256: str,
    policy_artifact_sha256: str,
    require_complete_development_set: bool = True,
    config_transform: Any | None = None,
    retain_episode_ledgers: bool = True,
) -> PolicyEvaluation:
    """Evaluate any PolicyBundle adapter through the authoritative DES/fitness path.

    ``retain_episode_ledgers=False`` compacts each instance's ``records`` /
    ``service_calls`` / ``snapshots`` as the instance is produced, instead of after the
    whole evaluation is built. Fitness, energies, digests and seals are unchanged; what
    changes is peak resident memory, which otherwise grows with the number of instances
    times the episode horizon. Callers that read raw ledgers (independent calibration,
    workload sensitivity) must keep the default.

    Baselines use this boundary so they receive exactly the same scenario x bundle
    instances, scheduler observations, rebalancer gateway, certificates, and C20
    ordering as PlacementScript candidates.  The caller supplies immutable identity;
    result provenance never depends on a Python class name or memory address.
    """
    if not isinstance(policy_name, str) or not policy_name:
        raise ValueError("policy_name must be a non-empty string")
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("policy_version must be a non-empty string")
    for label, digest in (
        ("policy_identity_sha256", policy_identity_sha256),
        ("policy_artifact_sha256", policy_artifact_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if not isinstance(proposal_seq, int) or isinstance(proposal_seq, bool) or proposal_seq < 0:
        raise ValueError("proposal_seq must be a non-negative int")
    registry = load_pinned_sources()
    bundles = build_development_bundles(registry)
    available = {bundle.bundle_id: bundle for bundle in bundles}
    if require_complete_development_set and tuple(available) != task.bundle_ids:
        raise ValueError(
            "task development_bundle_ids must equal the complete canonical development set"
        )
    unknown_bundles = set(task.bundle_ids) - available.keys()
    if unknown_bundles:
        raise ValueError(f"unknown development bundles: {sorted(unknown_bundles)}")

    outcomes: list[InstanceOutcome] = []
    evidence: list[InstanceEvidence] = []
    certificate_seals: dict[tuple[str, int, str], str] = {}
    for scenario in task.scenarios:
        base = load_episode_config(scenario.path)
        for seed in scenario.seeds:
            for bundle_id in task.bundle_ids:
                config = config_for_development_bundle(base, available[bundle_id])
                if config_transform is not None:
                    config = config_transform(config)
                result = run_episode(
                    config,
                    seed,
                    scorer,
                    proposer if config.rebalancer.enabled else None,
                )
                _raise_for_policy_execution_failure(
                    result,
                    scenario_id=scenario.scenario_id,
                    seed=seed,
                    bundle_id=bundle_id,
                )
                outcome, instance, seal = _outcome_from_episode(
                    result,
                    config,
                    task,
                    scenario_id=scenario.scenario_id,
                    seed=seed,
                    bundle_id=bundle_id,
                    retain_episode_ledgers=retain_episode_ledgers,
                )
                outcomes.append(outcome)
                evidence.append(instance)
                certificate_seals[(scenario.scenario_id, seed, bundle_id)] = seal

    fitness = evaluate_candidate(
        task.constraints,
        outcomes,
        development_bundle_ids=task.bundle_ids,
        instance_keys=task.instance_keys,
        expected_certificate_seals=certificate_seals,
        proposal_seq=proposal_seq,
    )
    return PolicyEvaluation(
        fitness=fitness,
        instances=tuple(evidence),
        policy_name=policy_name,
        policy_version=policy_version,
        policy_identity_sha256=policy_identity_sha256,
        policy_artifact_sha256=policy_artifact_sha256,
    )
