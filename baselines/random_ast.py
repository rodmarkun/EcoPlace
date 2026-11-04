"""Unguided PlacementScript AST-mutation floor (TODO 7.5).

This control deliberately has no semantic, fitness, inspiration, or text-feedback
guidance in proposal generation.  It chooses archive parents uniformly, applies one
uniformly random same-type subtree splice or fixed-domain literal replacement, and
uses the common L0/L1/L2 ladder.  Fitness is consulted only after evaluation to name
the best sampled policy; it never controls which policy is mutated next.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Protocol

import placementscript as ps
from shinka.core import ProposalOffspring
from shinka.database import Program

from evolution.campaign import EvaluatorWorkBudget
from evolution.evaluation import TaskEvaluationConfig, policy_identity_for_source
from evolution.ladder import LadderConfig, LadderResult, evaluate_through_ladder
from evolution.operators import validate_scoped_edit


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _sites(source: str) -> list[dict[str, Any]]:
    return [
        dict(site)
        for site in ps.enumerate_subtrees(source)
        if site["func"] is not None and site["kind"] != "func"
    ]


def _compatible(target: dict[str, Any], donor: dict[str, Any]) -> bool:
    if target["is_expr"] != donor["is_expr"]:
        return False
    if target["is_expr"]:
        return target["type"] == donor["type"]
    return target["kind"] == donor["kind"]


@dataclass(frozen=True)
class RandomASTConfig:
    max_proposals: int = 100
    max_attempts_per_candidate: int = 24
    subtree_probability: float = 0.5
    int_domain: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    # PlacementScript exposes the sign as a separate unary node, so numeric-literal
    # patches themselves must remain non-negative lexical tokens.
    float_domain: tuple[float, ...] = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0)

    def __post_init__(self) -> None:
        for name in ("max_proposals", "max_attempts_per_candidate"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if not math.isfinite(self.subtree_probability) or not (
            0.0 <= self.subtree_probability <= 1.0
        ):
            raise ValueError("subtree_probability must be within [0, 1]")
        if not self.int_domain or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.int_domain
        ):
            raise ValueError("int_domain must contain ints")
        if not self.float_domain or any(
            type(value) is not float or not math.isfinite(value)
            for value in self.float_domain
        ):
            raise ValueError("float_domain must contain finite floats")
        if len(set(self.int_domain)) != len(self.int_domain) or len(
            set(self.float_domain)
        ) != len(self.float_domain):
            raise ValueError("literal domains must not contain duplicates")


class RandomASTOperator:
    """One seeded, uninformed random edit under Shinka's proposal contract."""

    name = "random_ast"

    def __init__(self, config: RandomASTConfig | None = None) -> None:
        self.config = config or RandomASTConfig()

    def __call__(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
    ) -> ProposalOffspring:
        if not isinstance(rng, random.Random):
            raise TypeError("random-AST proposer requires an explicit random.Random")
        try:
            ps.Policy(parent.code)
        except ps.PlacementScriptError as exc:
            return ProposalOffspring(
                source=None,
                error=f"invalid parent PlacementScript: {exc}",
                metadata={"operator": self.name, "invalid_attempts": 1},
            )

        failures = []
        for attempt in range(1, self.config.max_attempts_per_candidate + 1):
            operation = (
                "subtree"
                if rng.random() < self.config.subtree_probability
                else "literal"
            )
            try:
                child, detail = (
                    self._subtree(parent.code, rng)
                    if operation == "subtree"
                    else self._literal(parent.code, rng)
                )
                if child == parent.code:
                    raise ValueError("random edit produced unchanged source")
                validate_scoped_edit(parent.code, child)
                policy = ps.Policy(child)
            except (ValueError, ps.PlacementScriptError) as exc:
                failures.append(
                    {"attempt": attempt, "operation": operation, "error": str(exc)}
                )
                continue
            return ProposalOffspring(
                source=child,
                metadata={
                    "operator": self.name,
                    "operation": operation,
                    "attempts": attempt,
                    "invalid_attempts": len(failures),
                    "retries": attempt - 1,
                    "parent_sha256": _sha256(parent.code),
                    "offspring_sha256": _sha256(child),
                    "policy_name": policy.name,
                    "policy_version": policy.version,
                    "inspirations_offered": len(inspirations),
                    "inspirations_consumed": 0,
                    "feedback_consumed": False,
                    "parent_selection": "uniform-archive",
                    "detail": detail,
                    "invalid_history": failures,
                    "api_costs": 0.0,
                },
            )
        return ProposalOffspring(
            source=None,
            error=(
                "no valid random-AST offspring after "
                f"{self.config.max_attempts_per_candidate} attempts"
            ),
            metadata={
                "operator": self.name,
                "attempts": self.config.max_attempts_per_candidate,
                "invalid_attempts": len(failures),
                "retries": self.config.max_attempts_per_candidate - 1,
                "inspirations_offered": len(inspirations),
                "inspirations_consumed": 0,
                "feedback_consumed": False,
                "invalid_history": failures,
                "api_costs": 0.0,
            },
        )

    @staticmethod
    def _subtree(source: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
        sites = _sites(source)
        if not sites:
            raise ValueError("no mutable AST sites")
        target = rng.choice(sites)
        donors = [
            site
            for site in sites
            if site["id"] != target["id"] and _compatible(target, site)
        ]
        if not donors:
            raise ValueError("no distinct compatible subtree")
        donor = rng.choice(donors)
        child = ps.crossover(source, source, target["id"], donor["id"])
        return child, {
            "kind": "uniform-same-parent-splice",
            "target_id": target["id"],
            "donor_id": donor["id"],
            "node_kind": target["kind"],
            "node_type": target["type"],
        }

    def _literal(
        self, source: str, rng: random.Random
    ) -> tuple[str, dict[str, Any]]:
        literals = [
            dict(literal)
            for literal in ps.enumerate_numeric_literals(source)
            if not literal["is_const"]
        ]
        if not literals:
            raise ValueError("no mutable numeric literals")
        target = rng.choice(literals)
        before = target["text"]
        domain = self.config.float_domain if target["is_float"] else self.config.int_domain
        choices = [self._literal_text(value, target["is_float"]) for value in domain]
        choices = [value for value in choices if value != before]
        if not choices:
            raise ValueError("literal domain has no changed replacement")
        replacement = rng.choice(choices)
        child = ps.patch_numeric_literals(source, {target["id"]: replacement})
        return child, {
            "kind": "uniform-fixed-domain-literal",
            "literal_id": target["id"],
            "before": before,
            "after": replacement,
        }

    @staticmethod
    def _literal_text(value: int | float, is_float: bool) -> str:
        if is_float:
            text = f"{float(value):.12g}"
            return text if "." in text or "e" in text.lower() else f"{text}.0"
        return str(value)


class EvaluateRandomAST(Protocol):
    def __call__(
        self,
        source: str,
        task: TaskEvaluationConfig,
        ladder: LadderConfig,
        proposal_seq: int,
    ) -> LadderResult: ...


@dataclass(frozen=True)
class RandomASTCandidate:
    proposal_seq: int
    parent_sha256: str
    source: str
    proposal_metadata: dict[str, Any]
    result: LadderResult
    candidate_microwork: int
    cumulative_microwork: int

    @property
    def selection_key(self) -> tuple[Any, ...]:
        if self.result.promoted_to_l2 and self.result.evaluation is not None:
            return (0, *self.result.evaluation.fitness.sort_key())
        return (1, self.proposal_seq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_seq": self.proposal_seq,
            "parent_sha256": self.parent_sha256,
            "source_sha256": _sha256(self.source),
            "proposal_metadata": self.proposal_metadata,
            "candidate_microwork": self.candidate_microwork,
            "cumulative_microwork": self.cumulative_microwork,
            "selection_key": list(self.selection_key),
            "ladder": self.result.to_dict(),
        }


@dataclass(frozen=True)
class RandomASTRun:
    root_seed: int
    initial_source_sha256: str
    config: RandomASTConfig
    candidates: tuple[RandomASTCandidate, ...]
    champion: RandomASTCandidate | None
    spent_microwork: int
    invalid_proposals: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-random-ast-run-v1",
            "root_seed": self.root_seed,
            "initial_source_sha256": self.initial_source_sha256,
            "config": dataclasses.asdict(self.config),
            "spent_microwork": self.spent_microwork,
            "invalid_proposals": self.invalid_proposals,
            "stop_reason": self.stop_reason,
            "champion_proposal_seq": (
                None if self.champion is None else self.champion.proposal_seq
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        return {**payload, "sha256": _sha256(_canonical(payload))}


def _default_evaluate(
    source: str,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    proposal_seq: int,
) -> LadderResult:
    return evaluate_through_ladder(source, task, ladder, proposal_seq=proposal_seq)


class RandomASTFloor:
    def __init__(
        self,
        config: RandomASTConfig | None = None,
        *,
        evaluate_fn: EvaluateRandomAST = _default_evaluate,
    ) -> None:
        self.config = config or RandomASTConfig()
        self.operator = RandomASTOperator(self.config)
        self.evaluate_fn = evaluate_fn

    def run(
        self,
        initial_source: str,
        task: TaskEvaluationConfig,
        ladder: LadderConfig,
        work: EvaluatorWorkBudget,
        *,
        root_seed: int,
    ) -> RandomASTRun:
        if (
            not isinstance(root_seed, int)
            or isinstance(root_seed, bool)
            or not 0 <= root_seed < 2**63
        ):
            raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
        ps.Policy(initial_source)
        rng = random.Random(root_seed)
        archive = [initial_source]
        candidates = []
        invalid = 0
        spent = 0
        stop_reason = "candidate-limit"
        for proposal_seq in range(self.config.max_proposals):
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
            offspring = self.operator(parent, (), rng)
            if not offspring.valid:
                invalid += 1
                continue
            result = self.evaluate_fn(
                offspring.source, task, ladder, proposal_seq
            )
            expected_identity = policy_identity_for_source(offspring.source)
            if (
                result.policy_name != expected_identity["name"]
                or result.policy_version != expected_identity["version"]
                or result.policy_identity_sha256
                != expected_identity["identity_sha256"]
                or result.policy_artifact_sha256
                != expected_identity["artifact_sha256"]
            ):
                raise ValueError("random-AST evaluator returned a mismatched policy identity")
            consumed = work.work_for_trace(result.trace)
            if consumed > work.max_candidate_microwork:
                raise ValueError("ladder trace exceeds registered per-candidate work")
            spent += consumed
            # See baselines/gphh.py: drop horizon-scaled episode ledgers before retaining.
            result = result.without_episode_ledgers()
            candidate = RandomASTCandidate(
                proposal_seq=proposal_seq,
                parent_sha256=_sha256(parent_source),
                source=offspring.source,
                proposal_metadata=offspring.metadata,
                result=result,
                candidate_microwork=consumed,
                cumulative_microwork=spent,
            )
            candidates.append(candidate)
            archive.append(offspring.source)
        feasible = [
            candidate
            for candidate in candidates
            if candidate.result.promoted_to_l2
            and candidate.result.evaluation is not None
            and candidate.result.evaluation.fitness.feasible
        ]
        champion = (
            min(feasible, key=lambda candidate: candidate.selection_key)
            if feasible
            else None
        )
        return RandomASTRun(
            root_seed=root_seed,
            initial_source_sha256=_sha256(initial_source),
            config=self.config,
            candidates=tuple(candidates),
            champion=champion,
            spent_microwork=spent,
            invalid_proposals=invalid,
            stop_reason=stop_reason,
        )


__all__ = [
    "RandomASTCandidate",
    "RandomASTConfig",
    "RandomASTFloor",
    "RandomASTOperator",
    "RandomASTRun",
]
