"""Deterministic fixed archives and fail-closed parity charging for the harness."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import placementscript as ps
from shinka.database import Program

from analysis.tracking import derive_seed
from baselines.random_ast import (
    RandomASTCandidate,
    RandomASTConfig,
    RandomASTFloor,
    RandomASTRun,
)
from evolution.campaign import CampaignRunner, EvaluatorWorkBudget
from evolution.evaluation import TaskEvaluationConfig, policy_identity_for_source
from evolution.ladder import LadderConfig

from .support import (
    BOUNDARY,
    INITIAL,
    ROOT,
    PreparedRun,
    read_sealed,
    write_sealed,
)

_REFERENCE_FILES = {
    "latency_greedy": "latency_greedy.ps",
    "energy_greedy": "energy_greedy.ps",
    "first_fit": "first_fit.ps",
    "least_allocated": "kube_least_allocated.ps",
    "most_allocated": "kube_most_allocated.ps",
    "network_overhead": "network_overhead.ps",
    "graph_headroom": "graph_headroom.ps",
}


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _evolve_shell(source: str) -> tuple[str, str]:
    lines = source.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if "EVOLVE-BLOCK-START" in line]
    ends = [index for index, line in enumerate(lines) if "EVOLVE-BLOCK-END" in line]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("fixed archive source needs one ordered EVOLVE-BLOCK pair")
    return "".join(lines[: starts[0] + 1]), "".join(lines[ends[0] :])


def _reference_body(source: str) -> str:
    offset = source.find("def ")
    if offset < 0:
        raise ValueError("reference PlacementScript contains no function definitions")
    return source[offset:].rstrip() + "\n"


def _reference_sources(names: list[str], initial_path: Path) -> list[dict[str, Any]]:
    initial = initial_path.read_text()
    prefix, suffix = _evolve_shell(initial)
    records = []
    for name in names:
        filename = _REFERENCE_FILES.get(name)
        if filename is None:
            raise ValueError(f"unknown fixed-archive reference policy {name!r}")
        path = ROOT / "runtime/policies" / filename
        original = path.read_text()
        source = prefix + _reference_body(original) + suffix
        ps.Policy(source)
        if _evolve_shell(source) != (prefix, suffix):
            raise ValueError("reference policy changed the fixed campaign shell")
        records.append(
            {
                "kind": "handwritten_reference",
                "name": name,
                "source_path": str(path.relative_to(ROOT)),
                "source_path_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": _sha256(source),
                "source": source,
                "identity_note": (
                    "Reference implementation transplanted into the common immutable "
                    "campaign identity shell; score/proposer behavior is retained."
                ),
            }
        )
    return records


def materialize_seed_archive(prepared: PreparedRun) -> Path | None:
    """Create or verify the common eight-member DSL initialization archive."""
    if not prepared.population_parity:
        return None
    path = prepared.run_dir / "inputs/seed_archive.json"
    names = list(prepared.harness_config["seed_archive_reference_policies"])
    records = _reference_sources(names, prepared.initial_path)
    seen = {record["source_sha256"] for record in records}
    if len(seen) != len(records):
        raise ValueError("fixed reference archive contains duplicate policy sources")

    floor = RandomASTFloor(RandomASTConfig())
    mutant_count = int(prepared.seed_archive_random_mutants)
    for mutant_index in range(mutant_count):
        created = None
        attempt_records = []
        for attempt in range(64):
            seed = derive_seed(
                prepared.root_seed,
                "allbaselines",
                "seed-archive",
                "random-ast-mutant",
                mutant_index,
                attempt,
            )
            rng = random.Random(seed)
            parent_record = rng.choice(records[: len(names)])
            parent = Program(
                id=f"seed-archive-parent-{mutant_index}-{attempt}",
                code=parent_record["source"],
                language="placementscript",
                correct=True,
            )
            offspring = floor.operator(parent, (), rng)
            attempt_records.append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    "parent_source_sha256": parent_record["source_sha256"],
                    "valid": offspring.valid,
                }
            )
            if not offspring.valid:
                continue
            digest = _sha256(offspring.source)
            if digest in seen:
                continue
            ps.Policy(offspring.source)
            if _evolve_shell(offspring.source) != _evolve_shell(
                prepared.initial_path.read_text()
            ):
                raise ValueError("random-AST archive mutant changed the fixed shell")
            created = {
                "kind": "seeded_random_ast_mutant",
                "name": f"random_ast_mutant_{mutant_index}",
                "seed": seed,
                "source_sha256": digest,
                "source": offspring.source,
                "proposal_metadata": offspring.metadata,
                "generation_attempts": attempt_records,
            }
            seen.add(digest)
            break
        if created is None:
            raise RuntimeError("could not create a distinct seeded random-AST archive mutant")
        records.append(created)

    expected_size = prepared.harness_config["classic_gphh_population_size"]
    if len(records) != expected_size:
        raise ValueError("fixed DSL archive size differs from GPHH generation 0")
    payload = {
        "schema": "evolved-continuum-seed-archive-v1",
        "status": "development-only",
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "root_seed": prepared.root_seed,
        "random_ast_mutants": mutant_count,
        "population_size": len(records),
        "gphh_generation0_population_size": expected_size,
        "entries": records,
    }
    write_sealed(path, payload)
    return path


def read_seed_archive(
    path: Path, *, expected_size: int, initial_path: Path = INITIAL
) -> dict[str, Any]:
    wrapped = read_sealed(path)
    payload = wrapped["payload"]
    if (
        payload.get("schema") != "evolved-continuum-seed-archive-v1"
        or payload.get("status") != "development-only"
        or payload.get("confirmatory_ready") is not False
        or payload.get("interpretation_boundary") != BOUNDARY
        or payload.get("population_size") != expected_size
        or payload.get("gphh_generation0_population_size") != expected_size
    ):
        raise ValueError("seed archive does not satisfy population-parity constraints")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_size:
        raise ValueError("seed archive entry count differs from its population size")
    digests = []
    shell = _evolve_shell(initial_path.read_text())
    for entry in entries:
        source = entry.get("source")
        if not isinstance(source, str):
            raise ValueError("seed archive entry lacks source text")
        ps.Policy(source)
        digest = _sha256(source)
        if entry.get("source_sha256") != digest or _evolve_shell(source) != shell:
            raise ValueError("seed archive source or immutable shell differs")
        digests.append(digest)
    if len(set(digests)) != len(digests):
        raise ValueError("seed archive source hashes are not unique")
    return {**wrapped, "sources": tuple(entry["source"] for entry in entries)}


def seeded_campaign_runner(
    seed_archive: dict[str, Any],
) -> type[CampaignRunner]:
    """Build a CampaignRunner seam whose first N charged candidates are fixed."""
    sources = seed_archive["sources"]
    payload_sha256 = seed_archive["payload_sha256"]

    class SeedArchiveCampaignRunner(CampaignRunner):
        def __init__(self, **kwargs: Any) -> None:
            provenance = dict(kwargs["provenance"])
            provenance["seed_archive"] = {
                "payload_sha256": payload_sha256,
                "population_size": len(sources),
                "charging": "CampaignRunner evaluator-work ledger",
            }
            kwargs["provenance"] = provenance
            kwargs["initial_source"] = sources[0]
            super().__init__(**kwargs)

        def _proposal(
            self,
            created: dict[str, Any],
            *,
            initial_offspring: Any | None = None,
        ) -> dict[str, Any]:
            """Serve the fixed archive, then delegate to the real proposer.

            The keyword must be accepted and forwarded: batched LLM proposals hand the
            runner a pre-fetched sibling through `initial_offspring`, and this override
            silently swallowed it. Population parity and batching had never been used
            together -- `--seed-archive` predates `--proposals-per-call` -- so the first
            campaign that combined them died with `_proposal() got an unexpected keyword
            argument 'initial_offspring'` on its first batched candidate. Every
            deterministic arm survived (they do not batch); only the LLM arm was lost.
            """
            seq = created["proposal_seq"]
            if seq >= len(sources):
                return super()._proposal(created, initial_offspring=initial_offspring)
            source = sources[seq]
            return {
                "valid": True,
                "source": source,
                "error": None,
                "metadata": {
                    "operator": "fixed_seed_archive",
                    "phase": "initialization",
                    "seed_archive_index": seq,
                    "seed_archive_payload_sha256": payload_sha256,
                },
                "attempts": [],
            }

    return SeedArchiveCampaignRunner


@dataclass(frozen=True)
class InitializationCharging:
    candidate_count: int
    actual_microwork: int
    candidate_microwork: tuple[int, ...]


def run_random_ast_with_archive(
    floor: RandomASTFloor,
    sources: tuple[str, ...],
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    work: EvaluatorWorkBudget,
    *,
    root_seed: int,
) -> RandomASTRun:
    """Run Random AST with fixed initialization evaluated inside its work budget."""
    rng = random.Random(root_seed)
    archive: list[str] = []
    candidates: list[RandomASTCandidate] = []
    spent = 0
    for proposal_seq, source in enumerate(sources):
        if spent + work.max_candidate_microwork > work.max_microwork:
            raise ValueError("budget cannot charge the complete fixed Random-AST archive")
        result = floor.evaluate_fn(source, task, ladder, proposal_seq)
        expected_identity = policy_identity_for_source(source)
        if (
            result.policy_name != expected_identity["name"]
            or result.policy_version != expected_identity["version"]
            or result.policy_identity_sha256 != expected_identity["identity_sha256"]
            or result.policy_artifact_sha256 != expected_identity["artifact_sha256"]
        ):
            raise ValueError("Random-AST initialization returned a mismatched identity")
        consumed = work.work_for_trace(result.trace)
        if consumed > work.max_candidate_microwork:
            raise ValueError("Random-AST initialization trace exceeds candidate work")
        spent += consumed
        candidates.append(
            RandomASTCandidate(
                proposal_seq=proposal_seq,
                parent_sha256=_sha256(source),
                source=source,
                proposal_metadata={
                    "operator": "fixed_seed_archive",
                    "phase": "initialization",
                    "seed_archive_index": proposal_seq,
                },
                result=result,
                candidate_microwork=consumed,
                cumulative_microwork=spent,
            )
        )
        archive.append(source)

    invalid = 0
    stop_reason = "candidate-limit"
    first_search_seq = len(sources)
    for proposal_seq in range(first_search_seq, floor.config.max_proposals):
        if spent + work.max_candidate_microwork > work.max_microwork:
            stop_reason = "normalized-work-budget-exhausted"
            break
        parent_source = rng.choice(archive)
        parent = Program(
            id=f"random-parent-{proposal_seq}",
            code=parent_source,
            language="placementscript",
            correct=True,
        )
        offspring = floor.operator(parent, (), rng)
        if not offspring.valid:
            invalid += 1
            continue
        result = floor.evaluate_fn(offspring.source, task, ladder, proposal_seq)
        expected_identity = policy_identity_for_source(offspring.source)
        if (
            result.policy_name != expected_identity["name"]
            or result.policy_version != expected_identity["version"]
            or result.policy_identity_sha256 != expected_identity["identity_sha256"]
            or result.policy_artifact_sha256 != expected_identity["artifact_sha256"]
        ):
            raise ValueError("Random-AST evaluator returned a mismatched policy identity")
        consumed = work.work_for_trace(result.trace)
        if consumed > work.max_candidate_microwork:
            raise ValueError("Random-AST ladder trace exceeds candidate work")
        spent += consumed
        candidates.append(
            RandomASTCandidate(
                proposal_seq=proposal_seq,
                parent_sha256=_sha256(parent_source),
                source=offspring.source,
                proposal_metadata={**offspring.metadata, "phase": "search"},
                result=result,
                candidate_microwork=consumed,
                cumulative_microwork=spent,
            )
        )
        archive.append(offspring.source)

    feasible = [
        candidate
        for candidate in candidates
        if candidate.result.promoted_to_l2
        and candidate.result.evaluation is not None
        and candidate.result.evaluation.fitness.feasible
    ]
    champion = min(feasible, key=lambda candidate: candidate.selection_key) if feasible else None
    return RandomASTRun(
        root_seed=root_seed,
        initial_source_sha256=_sha256(sources[0]),
        config=floor.config,
        candidates=tuple(candidates),
        champion=champion,
        spent_microwork=spent,
        invalid_proposals=invalid,
        stop_reason=stop_reason,
    )


def validate_parity_charging(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    expected_count: int,
    work: EvaluatorWorkBudget,
) -> dict[str, Any]:
    """Fail closed unless every initialization uses the registered ladder meter."""
    required = {"classic_gphh", "same_dsl_gp", "random_ast"}
    if not required <= set(rows_by_arm):
        raise ValueError("parity charging lacks a required non-LLM adaptive arm")
    records = {}
    for arm, rows in sorted(rows_by_arm.items()):
        if len(rows) != expected_count:
            raise ValueError(f"{arm} initialization count differs from generation 0")
        candidate_work = []
        for row in rows:
            trace = row.get("trace")
            actual = row.get("candidate_microwork")
            if not isinstance(trace, tuple) or actual != work.work_for_trace(trace):
                raise ValueError(f"{arm} initialization is not charged by the common meter")
            candidate_work.append(actual)
        records[arm] = {
            "candidate_count": len(rows),
            "actual_microwork": sum(candidate_work),
            "candidate_microwork": candidate_work,
            "charging_rule": "EvaluatorWorkBudget.work_for_trace",
        }
    return {
        "enabled": True,
        "verified": True,
        "population_size": expected_count,
        "reference_arm": "classic_gphh generation 0",
        "registered_rung_microwork": dict(work.rung_microwork),
        "arms": records,
    }


__all__ = [
    "materialize_seed_archive",
    "read_seed_archive",
    "run_random_ast_with_archive",
    "seeded_campaign_runner",
    "validate_parity_charging",
]
