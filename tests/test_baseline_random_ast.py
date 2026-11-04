from __future__ import annotations

import random
from pathlib import Path

import placementscript as ps
import pytest
from shinka.database import Program

from baselines.random_ast import RandomASTConfig, RandomASTFloor, RandomASTOperator
from evolution.campaign import EvaluatorWorkBudget
from evolution.evaluation import PolicyEvaluation, policy_identity_for_source
from evolution.ladder import LadderResult
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]


def _source() -> str:
    return (ROOT / "evolution" / "task" / "initial.py").read_text()


def _program(code: str | None = None) -> Program:
    return Program(
        id="parent",
        code=code or _source(),
        language="placementscript",
        correct=True,
        text_feedback="do something highly informed",
    )


def _rejected_result(seq: int, source: str | None = None) -> LadderResult:
    del seq
    identity = policy_identity_for_source(source or _source())
    return LadderResult(
        promoted_to_l2=False,
        evaluation=None,
        trace=({"rung": "L0", "selected": False},),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


def _work(maximum: int) -> EvaluatorWorkBudget:
    return EvaluatorWorkBudget(
        registration_status="development-provisional",
        registration_source="test",
        max_microwork=maximum,
        rung_microwork={"L0": 1, "L1": 2, "L2": 3},
        checkpoints_microwork=(0, maximum),
    )


def test_random_ast_is_reproducible_valid_and_ignores_inspirations() -> None:
    operator = RandomASTOperator()
    inspiration = Program(
        id="inspiration",
        code=(ROOT / "runtime" / "policies" / "energy_greedy.ps").read_text(),
        language="placementscript",
        correct=True,
    )
    first = operator(_program(), (inspiration,), random.Random(81))
    second = operator(_program(), (), random.Random(81))
    assert first.source == second.source
    assert first.valid
    assert first.source != _source()
    ps.Policy(first.source)
    assert first.metadata["inspirations_offered"] == 1
    assert first.metadata["inspirations_consumed"] == 0
    assert first.metadata["feedback_consumed"] is False


@pytest.mark.parametrize("seed", range(20))
def test_random_ast_seed_pool_returns_changed_loadable_policy(seed: int) -> None:
    result = RandomASTOperator()(_program(), (), random.Random(seed))
    assert result.valid, result.error
    assert result.source != _source()
    ps.Policy(result.source)
    assert result.metadata["operator"] == "random_ast"
    assert result.metadata["api_costs"] == 0.0


def test_floor_uses_shared_ladder_work_budget_and_is_reproducible() -> None:
    seen = []

    def evaluate(source, task, ladder, proposal_seq):
        del task, ladder
        seen.append((proposal_seq, source))
        return _rejected_result(proposal_seq, source)

    floor = RandomASTFloor(
        RandomASTConfig(max_proposals=20), evaluate_fn=evaluate
    )
    first = floor.run(_source(), None, None, _work(7), root_seed=17)
    first_seen = list(seen)
    seen.clear()
    second = floor.run(_source(), None, None, _work(7), root_seed=17)
    assert first.to_dict() == second.to_dict()
    assert first_seen == seen
    assert first.stop_reason == "normalized-work-budget-exhausted"
    assert first.spent_microwork == 2
    assert len(first.candidates) == 2
    assert first.champion is None
    assert all(candidate.candidate_microwork == 1 for candidate in first.candidates)


def test_floor_requires_explicit_seed() -> None:
    floor = RandomASTFloor(
        RandomASTConfig(max_proposals=1),
        evaluate_fn=lambda source, task, ladder, seq: _rejected_result(seq, source),
    )
    with pytest.raises(ValueError, match="root_seed"):
        floor.run(_source(), None, None, _work(6), root_seed=-1)


def test_random_ast_never_deploys_an_infeasible_l2_candidate() -> None:
    def infeasible(source, task, ladder, seq):
        del task, ladder
        identity = policy_identity_for_source(source)
        evaluation = PolicyEvaluation(
            CandidateFitness(False, (0.1, 0.0, 0.0, 0.0), 0.1, None, seq, 1, 1),
            (),
            identity["name"],
            identity["version"],
            identity["identity_sha256"],
            identity["artifact_sha256"],
        )
        return LadderResult(
            True,
            evaluation,
            ({"rung": "L2"},),
            "development-provisional",
            "test",
            identity["name"],
            identity["version"],
            identity["identity_sha256"],
            identity["artifact_sha256"],
        )

    run = RandomASTFloor(
        RandomASTConfig(max_proposals=1), evaluate_fn=infeasible
    ).run(_source(), None, None, _work(6), root_seed=3)
    assert run.candidates
    assert run.champion is None
