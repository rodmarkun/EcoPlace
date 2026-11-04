"""Execute or resume one matched all-baselines development comparison."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analysis.tracking import derive_seed
from baselines.gphh import ClassicGPHH, GPHHConfig
from baselines.handwritten import all_handwritten
from baselines.ppo import load_frozen_ppo_bundle
from baselines.random_ast import RandomASTConfig, RandomASTFloor
from evolution.campaign import load_campaign_plan
from evolution.codex_engine import load_codex_engine_config
from evolution.evaluation import ScenarioGrid, load_task_evaluation_config
from evolution.ladder import evaluate_through_ladder, load_ladder_config
from evolution.operators import AblationProbeInterleaveOperator
from evolution.profiles import (
    DEEP_SEARCH_FEEDBACK_PROFILE,
    DEEP_SEARCH_NOREPAIR_PROFILE,
    DEEP_SEARCH_PROFILE,
    ENHANCED_PROFILE,
    LEAN_DEEP_PROFILE,
    MATCHED_PROFILE,
)

from .initialization import (
    read_seed_archive,
    run_random_ast_with_archive,
)
from .novelty import AdaptiveNoveltyEvaluator
from .preflight import preflight_from_paths
from .support import (
    BOUNDARY,
    MODEL_MANIFEST,
    ROOT,
    PreparedRun,
    RunSizeBudgetExceeded,
    canonical_json,
    canonical_sha256,
    enforce_run_size,
    evaluation_seeds,
    file_sha256,
    sealed_artifact_exists,
    size_guard,
    write_sealed,
)


class ProposalEngineJournalStalled(RuntimeError):
    """The campaign process stayed alive beyond a pending dispatch reservation."""


def _pending_dispatch_reservation_ms(path: Path) -> int | None:
    if not path.exists():
        return None
    pending: dict[str, Any] | None = None
    try:
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("type") == "dispatch_reserved":
                pending = event.get("payload")
            elif event.get("type") in {"dispatch_settled", "dispatch_abandoned"}:
                pending = None
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(pending, dict):
        return None
    reserved_ms = pending.get("reserved_wall_clock_ms")
    if not isinstance(reserved_ms, int) or isinstance(reserved_ms, bool):
        return None
    return reserved_ms


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("task_config", type=Path)
    parser.add_argument("budget", help="matched normalized-work units, e.g. 58.8")
    parser.add_argument("root_seed", type=int)
    parser.add_argument(
        "--profile",
        choices=(
            MATCHED_PROFILE,
            ENHANCED_PROFILE,
            LEAN_DEEP_PROFILE,
            DEEP_SEARCH_PROFILE,
            DEEP_SEARCH_FEEDBACK_PROFILE,
            DEEP_SEARCH_NOREPAIR_PROFILE,
        ),
        default=MATCHED_PROFILE,
        help="explicit evolution profile; enhanced-llm-dev and lean-deep are development-only",
    )
    parser.add_argument(
        "--proposals-per-call",
        type=int,
        default=1,
        metavar="K",
        help="request K diverse LLM candidates per model call; default 1",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="explicitly enable the selected live proposal-engine arm",
    )
    parser.add_argument("--engine", choices=("codex", "openrouter"), default="codex")
    parser.add_argument(
        "--engine-config",
        type=Path,
        metavar="PATH",
        help="pinned engine config; defaults to the selected engine's committed config",
    )
    parser.add_argument(
        "--ablation-probe-share",
        type=float,
        default=0.0,
        metavar="P",
        help=(
            "development-only share of proposals routed through the LLM-free "
            "semantic ablation operator; default 0"
        ),
    )
    parser.add_argument(
        "--ablation-prompt-hint-share",
        type=float,
        default=0.0,
        metavar="P",
        help=(
            "development-only share of LLM proposals receiving the semantic "
            "ablation hint; default 0"
        ),
    )
    parser.add_argument(
        "--seed-archive",
        type=int,
        metavar="K",
        help=(
            "population parity: initialize each DSL arm with six references plus K "
            "explicitly seeded random-AST mutants (K must make the population equal GPHH)"
        ),
    )
    parser.add_argument(
        "--initial",
        type=Path,
        metavar="PATH",
        help=(
            "evolution seed policy shared by every DSL arm; defaults to "
            "evolution/task/initial.py. The dev-continuum-* family needs "
            "evolution/task/initial_tight.py, whose dependency-aware seed is the "
            "only hand-written shape that starts feasible there"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="explicit in-repository root for newly created run directories",
    )
    parser.add_argument(
        "--gp-config",
        type=Path,
        metavar="PATH",
        help=(
            "optional GPOperatorConfig JSON; omitted, the committed 0.45/0.35/0.20 "
            "mix is used and the run-directory name is unchanged"
        ),
    )
    return parser.parse_args()


def _seeded_task(prepared: PreparedRun):
    task = load_task_evaluation_config(prepared.task_config)
    replicates = prepared.harness_config["evaluation_replicates"]
    seeds = evaluation_seeds(prepared.root_seed, prepared.scenario_id, replicates)
    task = dataclasses.replace(
        task,
        scenarios=tuple(
            ScenarioGrid(scenario.scenario_id, scenario.path, seeds)
            for scenario in task.scenarios
        ),
    )
    ladder = load_ladder_config(prepared.task_config)
    ladder = dataclasses.replace(
        ladder,
        audit_root_seed=derive_seed(
            prepared.root_seed,
            "evolution",
            "ladder-audit",
        ),
    )
    return task, ladder


def _fitness_record(fitness: Any) -> dict[str, Any]:
    return {
        "feasible": fitness.feasible,
        "infeasibility_vector": list(fitness.infeasibility_vector),
        "summed_primitive_shortfall": fitness.summed_primitive_shortfall,
        "normalized_energy": fitness.normalized_energy,
        "proposal_seq": fitness.proposal_seq,
        "n_instances": fitness.n_instances,
        "n_bundles": fitness.n_bundles,
        "sort_key": list(fitness.sort_key()),
        "direction": "lower-is-better",
    }


def _compact_evaluation(evaluation: Any) -> dict[str, Any]:
    """Retain fitness-authoritative scalars while omitting full episode ledgers."""
    energies = []
    for instance in evaluation.instances:
        energies.append(
            {
                "scenario_id": instance.scenario_id,
                "seed": instance.seed,
                "bundle_id": instance.bundle_id,
                "episode_digest": instance.episode_digest,
                "certificate_seal": instance.certificate_seal,
                "energy_j": instance.energy_j,
                "idle_floor_j": instance.idle_floor_j,
                "ceiling_j": instance.ceiling_j,
                "slo_achieved": instance.slo_achieved,
                "goodput": instance.goodput,
                "drop_rate": instance.drop_rate,
                "timeout_rate": instance.timeout_rate,
                "disruption": instance.disruption,
                "boundary_certificate_sha256": canonical_sha256(
                    instance.boundary_certificate
                ),
                "verified_energy_sha256": canonical_sha256(instance.verified_energy),
            }
        )
    return {
        "fitness": _fitness_record(evaluation.fitness),
        "policy": {
            "name": evaluation.policy_name,
            "version": evaluation.policy_version,
            "identity_sha256": evaluation.policy_identity_sha256,
            "artifact_sha256": evaluation.policy_artifact_sha256,
        },
        "energies": energies,
        "omitted_inline_fields": [
            "instances[].episode",
            "instances[].boundary_certificate",
            "instances[].verified_energy",
        ],
    }


def _compact_ladder(result: Any) -> dict[str, Any]:
    trace = []
    for record in result.trace:
        compact = {
            key: value
            for key, value in record.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        for key in ("behavior_signature", "signature"):
            if isinstance(record.get(key), dict):
                compact[key] = record[key]
        trace.append(compact)
    return {
        "promoted_to_l2": result.promoted_to_l2,
        "budget_registration_status": result.budget_registration_status,
        "budget_registration_source": result.budget_registration_source,
        "policy": {
            "name": result.policy_name,
            "version": result.policy_version,
            "identity_sha256": result.policy_identity_sha256,
            "artifact_sha256": result.policy_artifact_sha256,
        },
        "trace": trace,
        "evaluation": (
            None if result.evaluation is None else _compact_evaluation(result.evaluation)
        ),
        "candidate_failure": result.candidate_failure,
    }


def _compact_candidate(arm: str, candidate: Any) -> dict[str, Any]:
    result = candidate.result
    record = {
        "candidate_id": f"{arm}:{candidate.proposal_seq}",
        "proposal_seq": candidate.proposal_seq,
        "generation": getattr(candidate, "generation", None),
        "candidate_microwork": candidate.candidate_microwork,
        "cumulative_microwork": candidate.cumulative_microwork,
        "selection_key": list(candidate.selection_key),
        "phase": getattr(candidate, "proposal_metadata", {}).get("phase"),
        "ladder": _compact_ladder(result),
    }
    if hasattr(candidate, "tree"):
        expression = str(candidate.tree)
        record["representation"] = {
            "kind": "deap-primitive-tree",
            "sha256": hashlib.sha256(expression.encode()).hexdigest(),
        }
    else:
        record["representation"] = {
            "kind": "placementscript-source",
            "sha256": result.policy_artifact_sha256,
            "parent_sha256": candidate.parent_sha256,
        }
        record["proposal_metadata_sha256"] = canonical_sha256(
            candidate.proposal_metadata
        )
    return record


def _compact_adaptive_run(arm: str, run: Any) -> dict[str, Any]:
    candidates = [_compact_candidate(arm, candidate) for candidate in run.candidates]
    champion = None
    if run.champion is not None:
        champion_candidate = next(
            candidate
            for candidate in candidates
            if candidate["proposal_seq"] == run.champion.proposal_seq
        )
        champion = {
            "candidate_id": champion_candidate["candidate_id"],
            "proposal_seq": champion_candidate["proposal_seq"],
            "generation": champion_candidate["generation"],
            "identity": champion_candidate["ladder"]["policy"],
            "fitness": champion_candidate["ladder"]["evaluation"]["fitness"],
            "energies": champion_candidate["ladder"]["evaluation"]["energies"],
        }
    record = {
        "schema": "evolved-continuum-compact-adaptive-run-v1",
        "source_schema": (
            "evolved-continuum-classic-gphh-run-v2"
            if arm == "classic_gphh"
            else "evolved-continuum-random-ast-run-v1"
        ),
        "root_seed": run.root_seed,
        "config": dataclasses.asdict(run.config),
        "spent_microwork": run.spent_microwork,
        "stop_reason": run.stop_reason,
        "champion": champion,
        "champion_proposal_seq": (
            None if run.champion is None else run.champion.proposal_seq
        ),
        "candidates": candidates,
    }
    if hasattr(run, "invalid_proposals"):
        record["invalid_proposals"] = run.invalid_proposals
        record["initial_source_sha256"] = run.initial_source_sha256
    if hasattr(run, "population_proposal_seqs_by_generation"):
        record["population_proposal_seqs_by_generation"] = [
            list(generation)
            for generation in run.population_proposal_seqs_by_generation
        ]
    return {**record, "sha256": canonical_sha256(record)}


def _plateau_summary(candidates: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    feasible = [
        candidate
        for candidate in candidates
        if candidate["ladder"]["evaluation"] is not None
        and candidate["ladder"]["evaluation"]["fitness"]["feasible"]
    ]
    levels: list[float] = []
    for value in sorted(
        candidate["ladder"]["evaluation"]["fitness"]["normalized_energy"]
        for candidate in feasible
    ):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
    buckets = []
    for level in levels:
        members = [
            candidate
            for candidate in feasible
            if abs(
                candidate["ladder"]["evaluation"]["fitness"]["normalized_energy"]
                - level
            )
            <= tolerance
        ]
        buckets.append(
            {
                "normalized_energy": level,
                "candidate_count": len(members),
                "first_proposal_seq": min(item["proposal_seq"] for item in members),
                "first_cumulative_normalized_work": min(
                    item["cumulative_microwork"] for item in members
                )
                / 1_000_000,
            }
        )
    return {
        "distinct_tolerance": tolerance,
        "l2_feasible_candidates": len(feasible),
        "distinct_feasible_energy_plateaus": len(buckets),
        "buckets": buckets,
    }


def _level_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    levels: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        for trace in candidate["ladder"]["trace"]:
            rung = trace["rung"]
            summary = levels.setdefault(
                rung,
                {"visited_candidates": 0, "passed_candidates": 0, "failed_candidates": 0},
            )
            summary["visited_candidates"] += 1
            decision = trace.get("decision")
            if trace.get("passed") is True or decision in {
                "promote",
                "fitness-of-record",
            }:
                summary["passed_candidates"] += 1
            elif trace.get("passed") is False or decision in {
                "reject",
                "invalid",
                "inert-not-promoted",
            }:
                summary["failed_candidates"] += 1
            elif decision == "novel-promote":
                summary["passed_candidates"] += 1
    return dict(sorted(levels.items()))


def _input_references(prepared: PreparedRun) -> list[dict[str, str]]:
    references = []
    for role, path in (
        ("input-manifest", prepared.run_dir / "inputs/input_manifest.json"),
        ("runtime-task-config", prepared.task_config),
        ("campaign-config", prepared.campaign_config),
        *(
            (("gp-operator-config", prepared.gp_config_path),)
            if prepared.gp_config_path is not None
            else ()
        ),
    ):
        references.append(
            {
                "role": role,
                "path": str(path.relative_to(prepared.run_dir)),
                "sha256": file_sha256(path),
            }
        )
    return references


def _write_arm_result(
    prepared: PreparedRun, path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    guard = size_guard(prepared.harness_config)
    wrapped = write_sealed(
        path,
        payload,
        max_bytes=int(guard["max_arm_result_bytes"]),
    )
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    return wrapped


def _evaluation_payload(
    arm: str, role: str, evaluation: Any, *, tolerance: float
) -> dict[str, Any]:
    compact = _compact_evaluation(evaluation)
    return {
        "schema": "evolved-continuum-allbaselines-evaluation-arm-v2",
        "status": "DEVELOPMENT_EVALUATION_COMPLETE",
        "development_only": True,
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "arm": arm,
        "role": role,
        "evaluation": compact,
        "champion": {
            "candidate_id": arm,
            "proposal_seq": compact["fitness"]["proposal_seq"],
            "identity": compact["policy"],
            "fitness": compact["fitness"],
            "energies": compact["energies"],
        },
        "summaries": {
            "plateau_histogram": {
                "distinct_tolerance": tolerance,
                "l2_feasible_candidates": int(compact["fitness"]["feasible"]),
                "distinct_feasible_energy_plateaus": int(
                    compact["fitness"]["feasible"]
                ),
                "buckets": (
                    []
                    if not compact["fitness"]["feasible"]
                    else [
                        {
                            "normalized_energy": compact["fitness"][
                                "normalized_energy"
                            ],
                            "candidate_count": 1,
                            "first_proposal_seq": compact["fitness"]["proposal_seq"],
                            "first_cumulative_normalized_work": 0.0,
                        }
                    ]
                ),
            },
            "ladder_levels": {"L2": {"visited_candidates": 1}},
        },
        "accounting": {
            "mode": "frozen-evaluation-only",
            "registered_max_microwork": None,
            "actual_microwork": 0,
            "effective_terminal_microwork": None,
            "proposal_calls": 0,
        },
        "mechanical_health": {
            "complete": True,
            "sealed_result": True,
            "full_episode_ledgers_inlined": False,
        },
    }


def _run_evaluation_only(prepared: PreparedRun, task: Any, ladder: Any) -> None:
    arms = prepared.run_dir / "arms"
    initial_path = arms / "initial" / "result.json"
    if not sealed_artifact_exists(initial_path):
        initial = evaluate_through_ladder(
            prepared.initial_path.read_text(),
            task,
            ladder,
            proposal_seq=0,
        )
        if not initial.promoted_to_l2 or initial.evaluation is None:
            raise ValueError("common initial policy did not reach L2 on the named dev scenario")
        payload = _evaluation_payload(
            "initial",
            "common-initial-policy",
            initial.evaluation,
            tolerance=prepared.harness_config["distinct_energy_tolerance"],
        )
        payload["initial_source_sha256"] = file_sha256(prepared.initial_path)
        payload["references"] = _input_references(prepared)
        _write_arm_result(prepared, initial_path, payload)

    for index, bundle in enumerate(all_handwritten()):
        path = arms / bundle.name / "result.json"
        if sealed_artifact_exists(path):
            continue
        evaluation = bundle.evaluate(task, proposal_seq=10_000 + index)
        payload = _evaluation_payload(
            bundle.name,
            "hand-written-evaluation-only-reference",
            evaluation,
            tolerance=prepared.harness_config["distinct_energy_tolerance"],
        )
        payload["references"] = _input_references(prepared)
        _write_arm_result(
            prepared,
            path,
            payload,
        )

    ppo_path = arms / "ppo_frozen" / "result.json"
    if not sealed_artifact_exists(ppo_path):
        ppo_config = prepared.harness_config["ppo"]
        ppo_dir = (ROOT / ppo_config["artifact_dir"]).resolve()
        bundle = load_frozen_ppo_bundle(
            ppo_dir,
            expected_receipt_sha256=ppo_config["expected_training_receipt_sha256"],
            require_p2_frozen=False,
        )
        evaluation = bundle.evaluate(task, proposal_seq=20_000)
        payload = _evaluation_payload(
            "ppo_frozen",
            "frozen-learned-policy-evaluation-only-reference",
            evaluation,
            tolerance=prepared.harness_config["distinct_energy_tolerance"],
        )
        payload["training_receipt_sha256"] = ppo_config[
            "expected_training_receipt_sha256"
        ]
        payload["model_artifact_sha256"] = bundle.artifact_sha256
        payload["references"] = _input_references(prepared)
        _write_arm_result(prepared, ppo_path, payload)


def _adaptive_payload(
    arm: str,
    run: dict[str, Any],
    *,
    spent_microwork: int,
    stop_reason: str,
    maximum: int,
    tolerance: float,
    references: list[dict[str, str]],
) -> dict[str, Any]:
    complete = stop_reason == "normalized-work-budget-exhausted"
    return {
        "schema": "evolved-continuum-allbaselines-adaptive-arm-v2",
        "status": "DEVELOPMENT_ADAPTIVE_COMPLETE" if complete else "DEVELOPMENT_INCOMPLETE",
        "development_only": True,
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "arm": arm,
        "run": run,
        "champion": run["champion"],
        "summaries": {
            "plateau_histogram": _plateau_summary(run["candidates"], tolerance),
            "ladder_levels": _level_summary(run["candidates"]),
        },
        "references": references,
        "accounting": {
            "mode": "adaptive-normalized-work",
            "registered_max_microwork": maximum,
            "actual_microwork": spent_microwork,
            "effective_terminal_microwork": maximum if complete else spent_microwork,
            "proposal_calls": 0,
        },
        "mechanical_health": {
            "complete": complete,
            "stop_reason": stop_reason,
            "run_sha256_verified": run.get("sha256") is not None,
            "sealed_result": True,
            "full_episode_ledgers_inlined": False,
        },
    }


def _run_classic_gphh(prepared: PreparedRun, task: Any, ladder: Any, work: Any) -> None:
    path = prepared.run_dir / "arms/classic_gphh/result.json"
    if sealed_artifact_exists(path):
        return
    population = prepared.harness_config["classic_gphh_population_size"]
    maximum_evaluations = prepared.max_microwork // min(work.rung_microwork.values()) + 1
    config = GPHHConfig(
        population_size=population,
        generations=maximum_evaluations,
        elite_size=2,
        tournament_size=3,
    )
    novelty = (
        AdaptiveNoveltyEvaluator(
            prepared.root_seed,
            prepared.evolution_profile.behavioral_novelty_tolerance,
        )
        if prepared.evolution_profile.behavioral_novelty_enabled
        else None
    )
    run = ClassicGPHH(
        config,
        **({"evaluate_fn": novelty.evaluate_bundle} if novelty is not None else {}),
    ).run(
        task,
        ladder,
        work,
        root_seed=derive_seed(
            prepared.root_seed,
            "allbaselines",
            "arm",
            "classic-gphh",
        ),
    )
    record = _compact_adaptive_run("classic_gphh", run)
    # Sidecar (2026-09-05): the compact record keeps only each tree's sha256, which is why
    # sealed classic_gphh champions could never be re-scored on the fresh grid. Write the
    # expressions next to the result so the held-out bridge can rebuild them with
    # ``gp.PrimitiveTree.from_string``. Separate file: result.json bytes are unchanged.
    write_gphh_trees_sidecar(path.with_name("trees.json"), run.candidates)
    _write_arm_result(
        prepared,
        path,
        _adaptive_payload(
            "classic_gphh",
            record,
            spent_microwork=run.spent_microwork,
            stop_reason=run.stop_reason,
            maximum=prepared.max_microwork,
            tolerance=prepared.harness_config["distinct_energy_tolerance"],
            references=_input_references(prepared),
        ),
    )


def write_gphh_trees_sidecar(trees_path: Path, candidates: Any) -> None:
    """Write classic_gphh tree expressions next to the arm result.

    The arm directory does not exist yet when the arm finishes (``_write_arm_result`` creates
    it), so the parent is created here. Missing this cost the whole 2026-09-05 grid: every
    cell died with FileNotFoundError after finishing its classic_gphh search.
    """
    if trees_path.exists():
        return
    trees_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "evolved-continuum-classic-gphh-trees-v1",
        "arm": "classic_gphh",
        "trees": [
            {
                "candidate_id": f"classic_gphh:{candidate.proposal_seq}",
                "proposal_seq": candidate.proposal_seq,
                "generation": candidate.generation,
                "expression": str(candidate.tree),
                "tree_sha256": hashlib.sha256(str(candidate.tree).encode()).hexdigest(),
            }
            for candidate in candidates
        ],
    }
    trees_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_random_ast(prepared: PreparedRun, task: Any, ladder: Any, work: Any) -> None:
    path = prepared.run_dir / "arms/random_ast/result.json"
    if sealed_artifact_exists(path):
        return
    maximum_evaluations = prepared.max_microwork // min(work.rung_microwork.values()) + 1
    config = RandomASTConfig(
        max_proposals=(
            maximum_evaluations
            * prepared.harness_config["random_ast_proposal_safety_factor"]
        )
    )
    novelty = (
        AdaptiveNoveltyEvaluator(
            prepared.root_seed,
            prepared.evolution_profile.behavioral_novelty_tolerance,
        )
        if prepared.evolution_profile.behavioral_novelty_enabled
        else None
    )
    floor = RandomASTFloor(
        config,
        **({"evaluate_fn": novelty.evaluate_source} if novelty is not None else {}),
    )
    if prepared.ablation_probe_probability > 0.0:
        floor.operator = AblationProbeInterleaveOperator(
            floor.operator,
            probability=prepared.ablation_probe_probability,
        )
    arm_seed = derive_seed(
        prepared.root_seed,
        "allbaselines",
        "arm",
        "random-ast",
    )
    if prepared.population_parity:
        archive = read_seed_archive(
            prepared.run_dir / "inputs/seed_archive.json",
            expected_size=prepared.initialization_population_size,
            initial_path=prepared.initial_path,
        )
        run = run_random_ast_with_archive(
            floor,
            archive["sources"],
            task,
            ladder,
            work,
            root_seed=arm_seed,
        )
    else:
        run = floor.run(
            prepared.initial_path.read_text(),
            task,
            ladder,
            work,
            root_seed=arm_seed,
        )
    # Persist candidate sources the way the LLM and GP arms do. Without this the floor
    # arm's champions exist only in memory, so the zero-shot transfer endpoint could not
    # score them -- and the floor is precisely the arm that endpoint is most informative
    # about, since a champion that won by deleting a normalization term has no reason to
    # survive a cluster it never saw. Classic GPHH stays out: it evolves DEAP trees, not
    # PlacementScript, so there is no source to carry across.
    candidates_dir = prepared.run_dir / "archives" / "random_ast" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for candidate in run.candidates:
        target = candidates_dir / f"{candidate.proposal_seq:08d}.ps"
        if not target.exists():
            target.write_text(candidate.source, encoding="utf-8")

    record = _compact_adaptive_run("random_ast", run)
    payload = _adaptive_payload(
        "random_ast",
        record,
        spent_microwork=run.spent_microwork,
        stop_reason=run.stop_reason,
        maximum=prepared.max_microwork,
        tolerance=prepared.harness_config["distinct_energy_tolerance"],
        references=_input_references(prepared),
    )
    if prepared.population_parity:
        payload["initialization"] = {
            "mode": "population-parity",
            "candidate_count": prepared.initialization_population_size,
            "seed_archive_payload_sha256": archive["payload_sha256"],
            "charging": "EvaluatorWorkBudget.work_for_trace",
        }
    _write_arm_result(prepared, path, payload)


def _read_driver(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    previous = "0" * 64
    for index, line in enumerate(path.read_text().splitlines()):
        event = json.loads(line)
        if line != canonical_json(event):
            raise ValueError("adaptive driver journal is not canonical JSON")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if (
            body.get("event_seq") != index
            or body.get("previous_event_sha256") != previous
            or event.get("event_sha256") != canonical_sha256(body)
        ):
            raise ValueError("adaptive driver journal hash chain failed")
        events.append(event)
        previous = event["event_sha256"]
    return events


def _append_driver(path: Path, kind: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        events = _read_driver(path)
        body = {
            "event_seq": len(events),
            "previous_event_sha256": (
                events[-1]["event_sha256"] if events else "0" * 64
            ),
            "type": kind,
            "payload": payload,
        }
        event = {**body, "event_sha256": canonical_sha256(body)}
        with path.open("a") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _has_terminal(archive: Path) -> bool:
    events = archive / "events.jsonl"
    if not events.exists():
        return False
    with events.open() as stream:
        return any(json.loads(line).get("type") == "campaign_terminal" for line in stream)


def _run_with_size_guard(
    command: list[str],
    prepared: PreparedRun,
    *,
    proposal_journal: Path | None = None,
    journal_watchdog_multiplier: float | None = None,
) -> tuple[int, RuntimeError | None]:
    """Monitor adaptive storage and pending proposal-journal liveness."""
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    process = subprocess.Popen(command, cwd=ROOT)
    poll_seconds = float(size_guard(prepared.harness_config)["poll_interval_seconds"])
    violation: RuntimeError | None = None
    journal_signature: tuple[int, int] | None = None
    journal_progress_at = time.monotonic()
    while True:
        try:
            return_code = process.wait(timeout=poll_seconds)
            break
        except subprocess.TimeoutExpired:
            try:
                enforce_run_size(prepared.run_dir, prepared.harness_config)
            except RunSizeBudgetExceeded as exc:
                violation = exc
                process.terminate()
                try:
                    return_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait()
                break
            if proposal_journal is not None and journal_watchdog_multiplier is not None:
                try:
                    stat = proposal_journal.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                except FileNotFoundError:
                    signature = None
                if signature != journal_signature:
                    journal_signature = signature
                    journal_progress_at = time.monotonic()
                reserved_ms = _pending_dispatch_reservation_ms(proposal_journal)
                if reserved_ms is None:
                    journal_progress_at = time.monotonic()
                else:
                    watchdog_s = reserved_ms / 1000.0 * journal_watchdog_multiplier
                    stalled_s = time.monotonic() - journal_progress_at
                    if stalled_s > watchdog_s:
                        violation = ProposalEngineJournalStalled(
                            "proposal-engine journal stalled with a pending dispatch for "
                            f"{stalled_s:.3f}s; watchdog limit is {watchdog_s:.3f}s "
                            f"({journal_watchdog_multiplier:g} x {reserved_ms}ms reservation)"
                        )
                        process.terminate()
                        try:
                            return_code = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            return_code = process.wait()
                        break
    if violation is None:
        try:
            enforce_run_size(prepared.run_dir, prepared.harness_config)
        except RunSizeBudgetExceeded as exc:
            violation = exc
    return return_code, violation


def _run_campaign_arm(prepared: PreparedRun, *, arm: str) -> None:
    archive_name = "same_dsl_gp" if arm == "gp" else "llm"
    archive = prepared.run_dir / "archives" / archive_name
    if _has_terminal(archive):
        return
    module = "evolution.run_campaign"
    parity_prefix: list[str] = []
    if prepared.population_parity:
        module = "experiments.allbaselines_harness.seeded_campaign"
        parity_prefix = [
            "--seed-archive",
            str(prepared.run_dir / "inputs/seed_archive.json"),
            "--seed-archive-size",
            str(prepared.initialization_population_size),
        ]
    command = [
        sys.executable,
        "-m",
        module,
        *parity_prefix,
        "--development",
        "--campaign_id",
        "h00",
        "--arm",
        arm,
        "--archive_dir",
        str(archive),
        "--campaign_config",
        str(prepared.campaign_config),
        "--task_config",
        str(prepared.task_config),
        "--model_manifest",
        str(MODEL_MANIFEST),
        "--initial",
        str(prepared.initial_path),
    ]
    if prepared.engine == "codex":
        command.extend(["--codex_engine_config", str(prepared.engine_config)])
    else:
        command.extend(["--openrouter_engine_config", str(prepared.engine_config)])
    if prepared.gp_config_path is not None and arm == "gp":
        command.extend(["--gp-config", str(prepared.gp_config_path)])
    if arm == "llm":
        if prepared.engine == "codex":
            command.extend(["--proposal_engine", "codex-cli-subscription"])
        else:
            command.extend(["--proposal_engine", "openrouter"])
    journal = archive / "driver-invocations.jsonl"
    prior = _read_driver(journal)
    invocation = sum(event["type"] == "started" for event in prior)
    _append_driver(
        journal,
        "started",
        {
            "invocation_id": invocation,
            "arm": archive_name,
            "resume": (archive / "manifest.json").exists(),
            "command": command[1:],
            "started_unix_ns": time.time_ns(),
        },
    )
    started = time.perf_counter_ns()
    codex_config = (
        load_codex_engine_config(prepared.engine_config)
        if arm == "llm" and prepared.engine == "codex"
        else None
    )
    return_code, guard_violation = _run_with_size_guard(
        command,
        prepared,
        proposal_journal=(
            archive / "proposal-engine-usage.jsonl" if arm == "llm" else None
        ),
        journal_watchdog_multiplier=(
            None if codex_config is None else codex_config.journal_watchdog_multiplier
        ),
    )
    if isinstance(guard_violation, ProposalEngineJournalStalled):
        _append_driver(
            journal,
            "proposal_engine_watchdog_stalled",
            {
                "invocation_id": invocation,
                "arm": archive_name,
                "reason": str(guard_violation),
                "resumable": True,
                "detected_unix_ns": time.time_ns(),
            },
        )
    _append_driver(
        journal,
        "finished",
        {
            "invocation_id": invocation,
            "arm": archive_name,
            "exit_code": return_code,
            "elapsed_ms": max(0, (time.perf_counter_ns() - started) // 1_000_000),
            "finished_unix_ns": time.time_ns(),
        },
    )
    if guard_violation is not None:
        raise guard_violation
    if return_code != 0 and not _has_terminal(archive):
        raise RuntimeError(f"{archive_name} campaign stopped without a terminal event")


def main() -> int:
    args = _arguments()
    prepared, _preflight = preflight_from_paths(
        args.scenario,
        args.task_config,
        args.budget,
        args.root_seed,
        include_llm=args.include_llm,
        profile_name=args.profile,
        ablation_probe_probability=args.ablation_probe_share,
        ablation_prompt_hint_probability=args.ablation_prompt_hint_share,
        proposals_per_call=args.proposals_per_call,
        seed_archive_random_mutants=args.seed_archive,
        initial_path=args.initial,
        runs_root=args.runs_root,
        engine=args.engine,
        engine_config=args.engine_config,
        gp_config_path=args.gp_config,
        inspect_cli=True,
    )
    task, ladder = _seeded_task(prepared)
    plan = load_campaign_plan(prepared.campaign_config)
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    _run_evaluation_only(prepared, task, ladder)
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    _run_campaign_arm(prepared, arm="gp")
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    _run_classic_gphh(prepared, task, ladder, plan.evaluator_work)
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    _run_random_ast(prepared, task, ladder, plan.evaluator_work)
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    if prepared.include_llm:
        _run_campaign_arm(prepared, arm="llm")
        enforce_run_size(prepared.run_dir, prepared.harness_config)

    from .summary import summarize

    wrapped = summarize(prepared)
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    print(
        json.dumps(
            {
                "status": wrapped["payload"]["status"],
                "report": str(prepared.run_dir / "REPORT.md"),
                "payload_sha256": wrapped["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
