from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
from deap import gp

from baselines.gphh import (
    ClassicGPHH,
    GPHHConfig,
    GPHHProposer,
    crossover_trees,
    generate_tree,
    gphh_bundle,
    mutate_tree,
    primitive_set,
)
from evolution.campaign import EvaluatorWorkBudget
from evolution.evaluation import PolicyEvaluation, load_task_evaluation_config
from evolution.ladder import LadderResult, load_ladder_config
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]


def _context() -> tuple[dict, list[dict], dict]:
    nodes = [
        {
            "id": 0,
            "tier": 0,
            "cpu_capacity": 4000,
            "cpu_allocated": 1000,
            "mem_capacity": 8000,
            "mem_allocated": 1000,
            "power_idle_w": 80.0,
            "power_max_w": 200.0,
            "suspended": False,
            "suspendable": False,
            "latency_to_ms": [0.0, 20.0],
        },
        {
            "id": 1,
            "tier": 2,
            "cpu_capacity": 2000,
            "cpu_allocated": 0,
            "mem_capacity": 4000,
            "mem_allocated": 0,
            "power_idle_w": 4.0,
            "power_max_w": 10.0,
            "suspended": False,
            "suspendable": True,
            "latency_to_ms": [20.0, 0.0],
        },
    ]
    pods = [
        {
            "id": 0,
            "service": 0,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": -1,
        },
        {
            "id": 1,
            "service": 1,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": 1,
        },
    ]
    graph = {"src": [0], "dst": [1], "rate": [5.0], "payload_kb": [2.0]}
    return pods[0], nodes, {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}


def _fake_evaluate(bundle, task, ladder, proposal_seq):
    del task, ladder
    pod, nodes, ctx = _context()
    score = sum(bundle.scorer.score_batch(pod, nodes, ctx))
    fitness = CandidateFitness(
        feasible=True,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=abs(score) + proposal_seq / 1_000_000,
        proposal_seq=proposal_seq,
        n_instances=1,
        n_bundles=1,
    )
    evaluation = PolicyEvaluation(
        fitness,
        (),
        bundle.name,
        bundle.version,
        bundle.identity_sha256,
        bundle.artifact_sha256,
    )
    return LadderResult(
        True,
        evaluation,
        (
            {"rung": "L0"},
            {"rung": "L1"},
            {"rung": "L2"},
        ),
        "development-provisional",
        "test",
        bundle.name,
        bundle.version,
        bundle.identity_sha256,
        bundle.artifact_sha256,
    )


def _inputs():
    task = load_task_evaluation_config(ROOT / "evolution" / "task" / "config.json")
    ladder = load_ladder_config(ROOT / "evolution" / "task" / "config.json")
    work = EvaluatorWorkBudget(
        registration_status="development-provisional",
        registration_source="test",
        max_microwork=36,
        rung_microwork={"L0": 1, "L1": 2, "L2": 3},
        checkpoints_microwork=(0, 18, 36),
    )
    return task, ladder, work


def test_primitive_set_is_deap_strongly_typed() -> None:
    pset = primitive_set()
    assert isinstance(pset, gp.PrimitiveSetTyped)
    assert pset.ret is float
    assert pset.primitives[bool]
    assert any(bool in primitive.args for primitive in pset.primitives[float])


def test_seeded_generation_variation_and_scoring_are_deterministic() -> None:
    import random

    pset = primitive_set()
    a = generate_tree(pset, random.Random(17), min_depth=1, max_depth=3)
    b = generate_tree(pset, random.Random(17), min_depth=1, max_depth=3)
    assert str(a) == str(b)
    mutated = mutate_tree(a, pset, random.Random(3), max_depth=3)
    left, right = crossover_trees(a, mutated, random.Random(9))
    for tree in (a, mutated, left, right):
        bundle = gphh_bundle(tree, pset)
        pod, nodes, ctx = _context()
        assert bundle.scorer.score_batch(pod, nodes, ctx) == bundle.scorer.score_batch(
            pod, nodes, ctx
        )


def test_gphh_proposer_scores_one_counterfactual_and_updates_later_context() -> None:
    class RecordingScorer:
        name = "recording"

        def __init__(self) -> None:
            self.calls = []

        def score_batch(self, pod, nodes, ctx):
            self.calls.append(tuple(item["current_node"] for item in ctx["pods"]))
            preferred = 1 if pod["id"] == 0 else 0
            return [float(node["id"] == preferred) for node in nodes]

    _pod, nodes, ctx = _context()
    ctx["pods"][0]["current_node"] = 0
    scorer = RecordingScorer()
    moves = GPHHProposer(scorer).propose_moves(
        {"nodes": nodes, "pods": ctx["pods"]},
        ctx["graph"],
        2,
    )
    assert moves == [
        {"kind": 0, "pod": 0, "dest": 1},
        {"kind": 0, "pod": 1, "dest": 0},
    ]
    assert len(scorer.calls) == 2
    assert scorer.calls[1][0] == 1


def test_classic_gphh_is_reproducible_and_stops_on_primary_budget() -> None:
    task, ladder, work = _inputs()
    config = GPHHConfig(population_size=3, generations=4, elite_size=1)
    first = ClassicGPHH(config, evaluate_fn=_fake_evaluate).run(
        task, ladder, work, root_seed=20260722
    )
    second = ClassicGPHH(config, evaluate_fn=_fake_evaluate).run(
        task, ladder, work, root_seed=20260722
    )
    assert first.to_dict() == second.to_dict()
    assert first.spent_microwork == work.max_microwork
    assert first.stop_reason == "normalized-work-budget-exhausted"
    assert first.champion is not None
    assert len({candidate.proposal_seq for candidate in first.candidates}) == len(first.candidates)
    assert all(
        len(generation) == len(set(generation))
        for generation in first.population_proposal_seqs_by_generation
    )
    assert len(first.to_dict()["sha256"]) == 64


def test_gphh_never_deploys_an_infeasible_l2_candidate() -> None:
    def infeasible(bundle, task, ladder, proposal_seq):
        result = _fake_evaluate(bundle, task, ladder, proposal_seq)
        assert result.evaluation is not None
        fitness = dataclasses.replace(result.evaluation.fitness, feasible=False)
        return dataclasses.replace(
            result,
            evaluation=dataclasses.replace(result.evaluation, fitness=fitness),
        )

    task, ladder, work = _inputs()
    run = ClassicGPHH(
        GPHHConfig(population_size=2, generations=1, elite_size=1),
        evaluate_fn=infeasible,
    ).run(task, ladder, work, root_seed=4)
    assert run.champion is None
    for candidate in run.candidates:
        assert candidate.result.evaluation is not None
        assert candidate.selection_key == (
            0,
            *candidate.result.evaluation.fitness.sort_key(),
        )


def test_classic_gphh_requires_explicit_valid_seed() -> None:
    task, ladder, work = _inputs()
    with pytest.raises(ValueError, match="root_seed"):
        ClassicGPHH(evaluate_fn=_fake_evaluate).run(task, ladder, work, root_seed=-1)


def test_gphh_bundle_traverses_real_shared_ladder() -> None:
    task, ladder, _work = _inputs()
    pset = primitive_set()
    tree = gp.PrimitiveTree.from_string("cpu_free_after", pset)
    result = gphh_bundle(tree, pset).evaluate_through_ladder(task, ladder, proposal_seq=0)
    assert result.promoted_to_l2
    assert [record["rung"] for record in result.trace] == ["L0", "L1", "L2"]
    assert result.evaluation is not None
    assert result.evaluation.policy_name == "classic_gphh"
    assert result.evaluation.policy_identity_sha256 == gphh_bundle(tree, pset).identity_sha256
    assert (
        result.evaluation.policy_artifact_sha256 == hashlib.sha256(str(tree).encode()).hexdigest()
    )
    bundle = gphh_bundle(tree, pset)
    assert result.to_dict()["policy"] == {
        "name": "classic_gphh",
        "version": bundle.version,
        "identity_sha256": bundle.identity_sha256,
        "artifact_sha256": hashlib.sha256(str(tree).encode()).hexdigest(),
    }
