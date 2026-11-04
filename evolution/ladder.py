"""Deterministic L0 -> L1 -> L2 evaluation ladder with epsilon audits (TODO 5.4).

Promotion and audit decisions depend only on registered configuration, candidate
content, and the canonical proposal sequence.  Wall-clock timing is intentionally not
part of the decision path or canonical trace; G0/G-ladder measurements determine the
registered rung shapes before campaigns and descriptive timings live in sidecars.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import placementscript as ps

from sim.actuation import PlacementScriptProposer
from sim.energy import build_development_bundles, load_pinned_sources
from sim.l0 import L0Evaluator
from sim.scheduler import PlacementScriptScorer
from sim.snapshot import SnapshotLibrary, initial_snapshot

from .evaluation import (
    CANDIDATE_FAILURE_SCHEMA,
    PolicyEvaluation,
    PolicyExecutionError,
    ScenarioGrid,
    TaskEvaluationConfig,
    candidate_failure_metadata,
    config_for_development_bundle,
    evaluate_policy_adapter,
    evaluate_policy_source,
    policy_identity_for_source,
)


@dataclass(frozen=True)
class L0Budget:
    mode: str
    max_mean_normalized_energy: float

    def __post_init__(self) -> None:
        if self.mode not in {"gross-quality", "validity-only"}:
            raise ValueError("L0 mode must be gross-quality or validity-only")
        if not math.isfinite(self.max_mean_normalized_energy) or (
            self.max_mean_normalized_energy <= 0
        ):
            raise ValueError("L0 normalized-energy threshold must be finite and positive")


@dataclass(frozen=True)
class L1Budget:
    scenario_ids: tuple[str, ...]
    seeds_per_scenario: int
    bundle_ids: tuple[str, ...]
    horizon_s: float
    max_normalized_energy: float
    max_summed_primitive_shortfall: float

    def __post_init__(self) -> None:
        if not self.scenario_ids or len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError("L1 scenario_ids must be non-empty and unique")
        if (
            not isinstance(self.seeds_per_scenario, int)
            or isinstance(self.seeds_per_scenario, bool)
            or self.seeds_per_scenario < 1
        ):
            raise ValueError("L1 seeds_per_scenario must be a positive int")
        if not self.bundle_ids or len(self.bundle_ids) != len(set(self.bundle_ids)):
            raise ValueError("L1 bundle_ids must be non-empty and unique")
        for name, value in (
            ("horizon_s", self.horizon_s),
            ("max_normalized_energy", self.max_normalized_energy),
            ("max_summed_primitive_shortfall", self.max_summed_primitive_shortfall),
        ):
            if not math.isfinite(value) or value < 0 or (name == "horizon_s" and value == 0):
                raise ValueError(f"L1 {name} must be finite and non-negative")


@dataclass(frozen=True)
class LadderConfig:
    registration_status: str
    registration_source: str
    l0: L0Budget
    l1: L1Budget
    epsilon_audit_fraction: float
    audit_root_seed: int

    def __post_init__(self) -> None:
        if self.registration_status not in {"development-provisional", "p1-frozen"}:
            raise ValueError("unknown ladder budget registration status")
        if not self.registration_source:
            raise ValueError("ladder budget registration source is required")
        if not math.isfinite(self.epsilon_audit_fraction) or not (
            0.0 <= self.epsilon_audit_fraction <= 1.0
        ):
            raise ValueError("epsilon_audit_fraction must be within [0, 1]")
        if (
            not isinstance(self.audit_root_seed, int)
            or isinstance(self.audit_root_seed, bool)
            or self.audit_root_seed < 0
        ):
            raise ValueError("audit_root_seed must be an explicit non-negative int")

    @property
    def confirmatory_ready(self) -> bool:
        return self.registration_status == "p1-frozen"


@dataclass(frozen=True)
class BehavioralNoveltyConfig:
    """Development-only deterministic L1 decision-trace gate."""

    tolerance: float
    signature_seed: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("behavioral novelty tolerance must be finite and positive")
        if (
            not isinstance(self.signature_seed, int)
            or isinstance(self.signature_seed, bool)
            or not 0 <= self.signature_seed < 2**63
        ):
            raise ValueError("behavioral novelty signature seed must be explicit")


@dataclass(frozen=True)
class LadderResult:
    promoted_to_l2: bool
    evaluation: PolicyEvaluation | None
    trace: tuple[dict[str, Any], ...]
    budget_registration_status: str
    budget_registration_source: str
    policy_name: str
    policy_version: str
    policy_identity_sha256: str
    policy_artifact_sha256: str
    candidate_failure: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.policy_name or not self.policy_version:
            raise ValueError("ladder policy identity requires name and version")
        for label, digest in (
            ("policy_identity_sha256", self.policy_identity_sha256),
            ("policy_artifact_sha256", self.policy_artifact_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if self.candidate_failure is not None:
            if self.candidate_failure.get("schema") != CANDIDATE_FAILURE_SCHEMA:
                raise ValueError("ladder candidate failure has an unknown schema")
            if self.promoted_to_l2 or self.evaluation is not None:
                raise ValueError("an invalid candidate cannot carry fitness of record")

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted_to_l2": self.promoted_to_l2,
            "budget_registration_status": self.budget_registration_status,
            "budget_registration_source": self.budget_registration_source,
            "policy": {
                "name": self.policy_name,
                "version": self.policy_version,
                "identity_sha256": self.policy_identity_sha256,
                "artifact_sha256": self.policy_artifact_sha256,
            },
            "trace": list(self.trace),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "candidate_failure": self.candidate_failure,
        }

    def without_episode_ledgers(self) -> LadderResult:
        """Same result with horizon-scaled episode ledgers dropped from memory.

        Fitness, policy identity, per-rung work accounting and digests are unchanged, so
        ``EvaluatorWorkBudget.work_for_trace`` and every selection key give the same answer.
        Call this before retaining a result for the lifetime of a run (see
        ``PolicyEvaluation.without_episode_ledgers``).
        """
        from .evaluation import compact_episode_ledgers

        return dataclasses.replace(
            self,
            evaluation=(
                None if self.evaluation is None else self.evaluation.without_episode_ledgers()
            ),
            trace=tuple(compact_episode_ledgers(list(self.trace))),
        )


def load_ladder_config(path: str | Path) -> LadderConfig:
    raw = json.loads(Path(path).read_text())["ladder"]
    registration = raw["budget_registration"]
    return LadderConfig(
        registration_status=registration["status"],
        registration_source=registration["source"],
        l0=L0Budget(**raw["l0"]),
        l1=L1Budget(
            scenario_ids=tuple(raw["l1"]["scenario_ids"]),
            seeds_per_scenario=raw["l1"]["seeds_per_scenario"],
            bundle_ids=tuple(raw["l1"]["bundle_ids"]),
            horizon_s=raw["l1"]["horizon_s"],
            max_normalized_energy=raw["l1"]["max_normalized_energy"],
            max_summed_primitive_shortfall=raw["l1"]["max_summed_primitive_shortfall"],
        ),
        epsilon_audit_fraction=raw["epsilon_audit_fraction"],
        audit_root_seed=raw["audit_root_seed"],
    )


def epsilon_audit_selected(
    config: LadderConfig,
    *,
    proposal_seq: int,
    rejected_at: str,
) -> bool:
    """Stable Bernoulli draw independent of async completion order and wall clock."""
    if rejected_at not in {"L0", "L1"}:
        raise ValueError("epsilon audits apply only to L0/L1 rejects")
    if not isinstance(proposal_seq, int) or isinstance(proposal_seq, bool) or proposal_seq < 0:
        raise ValueError("proposal_seq must be a non-negative int")
    payload = f"ladder-audit-v1:{config.audit_root_seed}:{proposal_seq}:{rejected_at}"
    draw = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") / 2**64
    return draw < config.epsilon_audit_fraction


def _l1_task(task: TaskEvaluationConfig, budget: L1Budget) -> TaskEvaluationConfig:
    available_scenarios = {scenario.scenario_id: scenario for scenario in task.scenarios}
    unknown_scenarios = set(budget.scenario_ids) - available_scenarios.keys()
    unknown_bundles = set(budget.bundle_ids) - set(task.bundle_ids)
    if unknown_scenarios or unknown_bundles:
        raise ValueError(
            f"L1 budget cites unknown grid entries: scenarios={sorted(unknown_scenarios)}, "
            f"bundles={sorted(unknown_bundles)}"
        )
    scenarios = tuple(
        ScenarioGrid(
            scenario_id=scenario_id,
            path=available_scenarios[scenario_id].path,
            seeds=available_scenarios[scenario_id].seeds[: budget.seeds_per_scenario],
        )
        for scenario_id in budget.scenario_ids
    )
    if any(len(scenario.seeds) != budget.seeds_per_scenario for scenario in scenarios):
        raise ValueError("L1 budget requests more seeds than the task grid provides")
    return replace(task, scenarios=scenarios, bundle_ids=budget.bundle_ids)


def _shorten_episode(horizon_s: float):
    def shorten(config):
        horizon = min(config.horizon_s, horizon_s)
        # Preserve the registered arrival/drain proportion. Letting arrivals run to
        # the shortened horizon fabricates timeout/in-flight violations at L1.
        cutoff = min(config.cutoff_s, horizon * config.cutoff_s / config.horizon_s)
        return replace(
            config,
            horizon_s=horizon,
            arrival_cutoff_s=cutoff,
            snapshot_times_s=tuple(t for t in config.snapshot_times_s if t <= horizon),
            node_churn=tuple(event for event in config.node_churn if event.at_s <= horizon),
        )

    return shorten


def _run_l0(source: str, task: TaskEvaluationConfig) -> dict[str, Any]:
    policy = ps.Policy(source)
    return _run_l0_adapter(PlacementScriptScorer(policy), task)


def _run_l0_adapter(scorer: Any, task: TaskEvaluationConfig) -> dict[str, Any]:
    bundles = build_development_bundles(load_pinned_sources())
    by_id = {bundle.bundle_id: bundle for bundle in bundles}
    instances = []
    all_valid = True
    all_feasible = True
    normalized = []
    for scenario in task.scenarios:
        from sim.spec import load_episode_config

        base = load_episode_config(scenario.path)
        for bundle_id in task.bundle_ids:
            config = config_for_development_bundle(base, by_id[bundle_id])
            library = SnapshotLibrary(
                config=config,
                snapshots=(initial_snapshot(config),),
                meta={
                    "role": "L0-initial-snapshot",
                    "scenario_id": scenario.scenario_id,
                    "bundle_id": bundle_id,
                },
            )
            result = L0Evaluator(library, task.weights).evaluate(scorer)
            all_valid = all_valid and result.valid
            all_feasible = all_feasible and result.feasible
            if result.mean_normalized_energy is not None:
                normalized.append(result.mean_normalized_energy)
            instances.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "bundle_id": bundle_id,
                    "snapshot_library_sha256": library.digest(),
                    "snapshot_library": json.loads(library.to_json()),
                    "valid": result.valid,
                    "feasible": result.feasible,
                    "mean_normalized_energy": result.mean_normalized_energy,
                    "error": result.error,
                    "violations": result.violations,
                    "per_snapshot": [
                        {
                            "feasible": snapshot.feasible,
                            "violations": snapshot.violations,
                            "decisions": [list(item) for item in snapshot.decisions],
                            "device_j": snapshot.device_j,
                            "cloud_j": snapshot.cloud_j,
                            "network_j": snapshot.network_j,
                            "closed_boundary_j": snapshot.closed_boundary_j,
                            "normalized_energy": snapshot.normalized_energy,
                            "template_latency_est_s": {
                                str(template_id): (
                                    value if math.isfinite(value) else "positive-infinity"
                                )
                                for template_id, value in snapshot.template_latency_est_s.items()
                            },
                        }
                        for snapshot in result.per_snapshot
                    ],
                }
            )
    return {
        "valid": all_valid,
        "feasible": all_feasible,
        "mean_normalized_energy": (math.fsum(normalized) / len(normalized) if normalized else None),
        "instances": instances,
    }


def _run_l1(
    source: str,
    task: TaskEvaluationConfig,
    config: LadderConfig,
    proposal_seq: int,
    *,
    retain_episode_ledgers: bool = True,
) -> PolicyEvaluation:
    return evaluate_policy_source(
        source,
        _l1_task(task, config.l1),
        proposal_seq=proposal_seq,
        require_complete_development_set=False,
        config_transform=_shorten_episode(config.l1.horizon_s),
        retain_episode_ledgers=retain_episode_ledgers,
    )


def _run_l1_adapter(
    scorer: Any,
    proposer: Any,
    task: TaskEvaluationConfig,
    config: LadderConfig,
    proposal_seq: int,
    *,
    policy_name: str,
    policy_version: str,
    policy_identity_sha256: str,
    policy_artifact_sha256: str,
    retain_episode_ledgers: bool = True,
) -> PolicyEvaluation:
    return evaluate_policy_adapter(
        scorer,
        proposer,
        _l1_task(task, config.l1),
        proposal_seq=proposal_seq,
        policy_name=policy_name,
        policy_version=policy_version,
        policy_identity_sha256=policy_identity_sha256,
        policy_artifact_sha256=policy_artifact_sha256,
        require_complete_development_set=False,
        config_transform=_shorten_episode(config.l1.horizon_s),
        retain_episode_ledgers=retain_episode_ledgers,
    )


def _fitness_summary(evaluation: PolicyEvaluation) -> dict[str, Any]:
    return {
        "fitness_sort_key": list(evaluation.fitness.sort_key()),
        "feasible": evaluation.fitness.feasible,
        "normalized_energy": evaluation.fitness.normalized_energy,
        "infeasibility_vector": list(evaluation.fitness.infeasibility_vector),
        "summed_primitive_shortfall": evaluation.fitness.summed_primitive_shortfall,
    }


def _evaluation_trace(evaluation: PolicyEvaluation) -> dict[str, Any]:
    return {**_fitness_summary(evaluation), "evaluation": evaluation.to_dict()}


def _quantize_behavior(value: Any, tolerance: float) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("behavior signature refuses non-finite trace values")
        return {"quantized_float": round(value / tolerance)}
    if isinstance(value, list):
        return [_quantize_behavior(item, tolerance) for item in value]
    if isinstance(value, dict):
        return {
            key: _quantize_behavior(value[key], tolerance) for key in sorted(value)
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"behavior signature cannot encode {type(value).__name__}")


def l1_behavior_signature(
    evaluation: PolicyEvaluation,
    config: BehavioralNoveltyConfig,
) -> dict[str, Any]:
    """Hash only policy-visible L1 decisions/actions, not energy or fitness."""
    instances = []
    for instance in evaluation.instances:
        episode = instance.episode
        if not isinstance(episode, dict):
            raise ValueError("L1 behavior signature requires an archived episode")
        for field in ("decision_log", "rebalancer"):
            if field not in episode:
                raise ValueError(f"L1 behavior signature requires episode.{field}")
        instances.append(
            {
                "scenario_id": instance.scenario_id,
                "seed": instance.seed,
                "bundle_id": instance.bundle_id,
                "decision_log": episode["decision_log"],
                "rebalancer": episode["rebalancer"],
            }
        )
    payload = {
        "schema": "l1-behavior-signature-v1",
        "signature_seed": config.signature_seed,
        "tolerance": config.tolerance,
        "instances": _quantize_behavior(instances, config.tolerance),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema": payload["schema"],
        "signature_seed": config.signature_seed,
        "tolerance": config.tolerance,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _failure_for_rung(
    error: PolicyExecutionError | ps.PlacementScriptError,
    rung: str,
) -> dict[str, Any]:
    if isinstance(error, PolicyExecutionError):
        failure = dict(error.failure)
        failure["stage"] = rung
        return failure
    return candidate_failure_metadata(error, stage=rung)


def _l0_failure(result: dict[str, Any]) -> dict[str, Any] | None:
    invalid = next(
        (instance for instance in result["instances"] if not instance["valid"]),
        None,
    )
    if invalid is None:
        return None
    return candidate_failure_metadata(
        invalid.get("error") or "L0 policy execution failed without an error record",
        stage="L0",
        scenario_id=invalid["scenario_id"],
        bundle_id=invalid["bundle_id"],
    )


def _invalid_ladder_result(
    *,
    trace: list[dict[str, Any]],
    failure: dict[str, Any],
    ladder: LadderConfig,
    policy_name: str,
    policy_version: str,
    policy_identity_sha256: str,
    policy_artifact_sha256: str,
) -> LadderResult:
    return LadderResult(
        False,
        None,
        tuple(trace),
        ladder.registration_status,
        ladder.registration_source,
        policy_name,
        policy_version,
        policy_identity_sha256,
        policy_artifact_sha256,
        failure,
    )


def evaluate_through_ladder(
    source: str,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    *,
    proposal_seq: int,
    behavioral_novelty: BehavioralNoveltyConfig | None = None,
    known_l1_signatures: frozenset[str] = frozenset(),
    retain_episode_ledgers: bool = True,
) -> LadderResult:
    """Evaluate PlacementScript with deterministic promotion and epsilon audits.

    ``retain_episode_ledgers=False`` keeps every rung decision, fitness value, digest
    and seal identical while compacting the per-episode ledgers each instance carries.
    Use it whenever the result is only consumed as fitness plus archive evidence.
    """
    policy = ps.Policy(source)
    identity = policy_identity_for_source(source)
    return evaluate_adapter_through_ladder(
        PlacementScriptScorer(policy),
        PlacementScriptProposer(policy),
        task,
        ladder,
        proposal_seq=proposal_seq,
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
        behavioral_novelty=behavioral_novelty,
        known_l1_signatures=known_l1_signatures,
        retain_episode_ledgers=retain_episode_ledgers,
    )


def evaluate_adapter_through_ladder(
    scorer: Any,
    proposer: Any,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    *,
    proposal_seq: int,
    policy_name: str,
    policy_version: str,
    policy_identity_sha256: str,
    policy_artifact_sha256: str,
    behavioral_novelty: BehavioralNoveltyConfig | None = None,
    known_l1_signatures: frozenset[str] = frozenset(),
    retain_episode_ledgers: bool = True,
) -> LadderResult:
    """Evaluate any PolicyBundle through the exact registered L0/L1/L2 ladder.

    Audits consume work and archive evidence but never reverse a rejection.  This is
    the shared Phase-7 comparison boundary; no baseline owns a cheaper evaluator.
    """
    trace: list[dict[str, Any]] = []
    try:
        l0 = _run_l0_adapter(scorer, task)
    except (PolicyExecutionError, ps.PlacementScriptError) as exc:
        failure = _failure_for_rung(exc, "L0")
        trace.append(
            {
                "rung": "L0",
                "role": ladder.l0.mode,
                "decision": "invalid",
                "result": {"valid": False, "failure": failure},
            }
        )
        return _invalid_ladder_result(
            trace=trace,
            failure=failure,
            ladder=ladder,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_identity_sha256=policy_identity_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
        )
    failure = _l0_failure(l0)
    if failure is not None:
        trace.append(
            {
                "rung": "L0",
                "role": ladder.l0.mode,
                "decision": "invalid",
                "result": {**l0, "failure": failure},
            }
        )
        return _invalid_ladder_result(
            trace=trace,
            failure=failure,
            ladder=ladder,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_identity_sha256=policy_identity_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
        )
    l0_quality_pass = (
        l0["valid"]
        and l0["feasible"]
        and l0["mean_normalized_energy"] is not None
        and l0["mean_normalized_energy"] <= ladder.l0.max_mean_normalized_energy
    )
    l0_promote = l0["valid"] and (ladder.l0.mode == "validity-only" or l0_quality_pass)
    trace.append(
        {
            "rung": "L0",
            "role": ladder.l0.mode,
            "decision": "promote" if l0_promote else "reject",
            "result": l0,
        }
    )
    if not l0_promote:
        audited = epsilon_audit_selected(ladder, proposal_seq=proposal_seq, rejected_at="L0")
        audit_record: dict[str, Any] = {"selected": audited, "target_rung": "L1"}
        if audited:
            try:
                audit_record["result"] = _evaluation_trace(
                    _run_l1_adapter(
                        scorer,
                        proposer,
                        task,
                        ladder,
                        proposal_seq,
                        policy_name=policy_name,
                        policy_version=policy_version,
                        policy_identity_sha256=policy_identity_sha256,
                        policy_artifact_sha256=policy_artifact_sha256,
                        retain_episode_ledgers=retain_episode_ledgers,
                    )
                )
            except (PolicyExecutionError, ps.PlacementScriptError) as exc:
                failure = _failure_for_rung(exc, "L1")
                audit_record["result"] = {"valid": False, "failure": failure}
                trace.append({"rung": "epsilon-audit", "rejected_at": "L0", **audit_record})
                return _invalid_ladder_result(
                    trace=trace,
                    failure=failure,
                    ladder=ladder,
                    policy_name=policy_name,
                    policy_version=policy_version,
                    policy_identity_sha256=policy_identity_sha256,
                    policy_artifact_sha256=policy_artifact_sha256,
                )
        trace.append({"rung": "epsilon-audit", "rejected_at": "L0", **audit_record})
        return LadderResult(
            False,
            None,
            tuple(trace),
            ladder.registration_status,
            ladder.registration_source,
            policy_name,
            policy_version,
            policy_identity_sha256,
            policy_artifact_sha256,
        )

    try:
        l1 = _run_l1_adapter(
            scorer,
            proposer,
            task,
            ladder,
            proposal_seq,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_identity_sha256=policy_identity_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
            retain_episode_ledgers=retain_episode_ledgers,
        )
    except (PolicyExecutionError, ps.PlacementScriptError) as exc:
        failure = _failure_for_rung(exc, "L1")
        trace.append(
            {
                "rung": "L1",
                "decision": "invalid",
                "result": {"valid": False, "failure": failure},
            }
        )
        return _invalid_ladder_result(
            trace=trace,
            failure=failure,
            ladder=ladder,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_identity_sha256=policy_identity_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
        )
    energy_within_cap = l1.fitness.normalized_energy is None or (
        l1.fitness.normalized_energy <= ladder.l1.max_normalized_energy
    )
    l1_promote = energy_within_cap and (
        l1.fitness.summed_primitive_shortfall <= ladder.l1.max_summed_primitive_shortfall
    )
    trace.append(
        {
            "rung": "L1",
            "decision": "promote" if l1_promote else "reject",
            "result": _evaluation_trace(l1),
        }
    )
    if not l1_promote:
        audited = epsilon_audit_selected(ladder, proposal_seq=proposal_seq, rejected_at="L1")
        audit_record = {"selected": audited, "target_rung": "L2"}
        if audited:
            try:
                audit_record["result"] = _evaluation_trace(
                    evaluate_policy_adapter(
                        scorer,
                        proposer,
                        task,
                        proposal_seq=proposal_seq,
                        policy_name=policy_name,
                        policy_version=policy_version,
                        policy_identity_sha256=policy_identity_sha256,
                        policy_artifact_sha256=policy_artifact_sha256,
                        retain_episode_ledgers=retain_episode_ledgers,
                    )
                )
            except (PolicyExecutionError, ps.PlacementScriptError) as exc:
                failure = _failure_for_rung(exc, "L2")
                audit_record["result"] = {"valid": False, "failure": failure}
                trace.append({"rung": "epsilon-audit", "rejected_at": "L1", **audit_record})
                return _invalid_ladder_result(
                    trace=trace,
                    failure=failure,
                    ladder=ladder,
                    policy_name=policy_name,
                    policy_version=policy_version,
                    policy_identity_sha256=policy_identity_sha256,
                    policy_artifact_sha256=policy_artifact_sha256,
                )
        trace.append({"rung": "epsilon-audit", "rejected_at": "L1", **audit_record})
        return LadderResult(
            False,
            None,
            tuple(trace),
            ladder.registration_status,
            ladder.registration_source,
            policy_name,
            policy_version,
            policy_identity_sha256,
            policy_artifact_sha256,
        )

    if behavioral_novelty is not None:
        signature = l1_behavior_signature(l1, behavioral_novelty)
        trace[-1]["behavior_signature"] = signature
        inert = signature["sha256"] in known_l1_signatures
        trace.append(
            {
                "rung": "behavioral-novelty",
                "decision": "inert-not-promoted" if inert else "novel-promote",
                "signature": signature,
                "known_signature_count": len(known_l1_signatures),
            }
        )
        if inert:
            return LadderResult(
                False,
                None,
                tuple(trace),
                ladder.registration_status,
                ladder.registration_source,
                policy_name,
                policy_version,
                policy_identity_sha256,
                policy_artifact_sha256,
            )

    try:
        l2 = evaluate_policy_adapter(
            scorer,
            proposer,
            task,
            proposal_seq=proposal_seq,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_identity_sha256=policy_identity_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
            retain_episode_ledgers=retain_episode_ledgers,
        )
    except (PolicyExecutionError, ps.PlacementScriptError) as exc:
        failure = _failure_for_rung(exc, "L2")
        trace.append(
            {
                "rung": "L2",
                "decision": "invalid",
                "result": {"valid": False, "failure": failure},
            }
        )
        return _invalid_ladder_result(
            trace=trace,
            failure=failure,
            ladder=ladder,
            policy_name=policy_name,
            policy_version=policy_version,
            policy_identity_sha256=policy_identity_sha256,
            policy_artifact_sha256=policy_artifact_sha256,
        )
    trace.append({"rung": "L2", "decision": "fitness-of-record", "result": _evaluation_trace(l2)})
    return LadderResult(
        True,
        l2,
        tuple(trace),
        ladder.registration_status,
        ladder.registration_source,
        policy_name,
        policy_version,
        policy_identity_sha256,
        policy_artifact_sha256,
    )
