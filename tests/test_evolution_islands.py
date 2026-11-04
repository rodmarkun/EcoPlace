from __future__ import annotations

from pathlib import Path

from evolution.evaluation import load_task_evaluation_config
from evolution.islands import (
    assess_for_archive,
    load_islands_config,
    measure_decision_cost,
    policy_loc,
    survivor_key,
)
from sim.energy import CandidateFitness

TASK = Path(__file__).parents[1] / "evolution" / "task"


def _fitness(*, feasible: bool, energy: float | None, seq: int) -> CandidateFitness:
    return CandidateFitness(
        feasible=feasible,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=energy,
        proposal_seq=seq,
        n_instances=1,
        n_bundles=1,
    )


def test_islands_apply_distinct_constraint_tightness_profiles():
    task = load_task_evaluation_config(TASK / "config.json")
    config = load_islands_config(TASK / "config.json")
    by_id = {profile.island_id: profile for profile in config.profiles}

    balanced = by_id["balanced"].constraints(task.constraints)
    tight = by_id["qos-tight"].constraints(task.constraints)
    explorer = by_id["feasibility-explorer"].constraints(task.constraints)
    assert dict(balanced.slo_targets) == dict(task.constraints.slo_targets)
    assert tight.slo_targets["rpc"] > balanced.slo_targets["rpc"]
    assert tight.drop_cap < balanced.drop_cap
    assert explorer.goodput_target < balanced.goodput_target
    assert explorer.timeout_cap > balanced.timeout_cap


def test_policy_loc_counts_only_mutable_implementation():
    source = (TASK / "initial.py").read_text()

    assert policy_loc(source) > 0
    assert policy_loc("POLICY_NAME: str = \"x\"\nPOLICY_VERSION: str = \"1\"\n") == 0


def test_decision_cost_is_deterministic_and_archive_penalties_are_separate():
    source = (TASK / "initial.py").read_text()
    task = load_task_evaluation_config(TASK / "config.json")
    config = load_islands_config(TASK / "config.json")
    first = measure_decision_cost(source, task, config.latency_calibration)
    second = measure_decision_cost(source, task, config.latency_calibration)
    assessment = assess_for_archive(source, first, config.archive)

    assert first == second
    assert first.score_batch_fuel > 0
    assert first.calibrated_latency_us > 0
    assert assessment.total_penalty >= 0

    better = _fitness(feasible=True, energy=1.0, seq=1)
    worse = _fitness(feasible=True, energy=2.0, seq=2)
    assert survivor_key(better) < survivor_key(worse)
    assert survivor_key(better) == better.sort_key()
    assert assessment.elite_key(better)[0] == 0


def test_elite_view_never_places_infeasible_ahead_of_feasible():
    source = (TASK / "initial.py").read_text()
    task = load_task_evaluation_config(TASK / "config.json")
    config = load_islands_config(TASK / "config.json")
    cost = measure_decision_cost(source, task, config.latency_calibration)
    assessment = assess_for_archive(source, cost, config.archive)

    feasible = _fitness(feasible=True, energy=100.0, seq=3)
    infeasible = _fitness(feasible=False, energy=None, seq=4)
    assert assessment.elite_key(feasible) < assessment.elite_key(infeasible)
