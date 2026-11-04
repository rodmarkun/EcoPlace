from __future__ import annotations

import pytest

from baselines.assignment_oracles import (
    AssignmentInstance,
    FirstFitDecreasingAssignmentOracle,
    HighsMILPOracle,
    MILPConfig,
    NSGA2AssignmentOracle,
    NSGA2Config,
    assignment_objectives,
)


def _instance() -> AssignmentInstance:
    nodes = (
        {
            "id": 0,
            "cpu_capacity": 4,
            "cpu_allocated": 0,
            "mem_capacity": 4,
            "mem_allocated": 0,
            "power_idle_w": 10.0,
            "power_max_w": 18.0,
            "suspended": False,
            "latency_to_ms": [0.0, 10.0],
        },
        {
            "id": 1,
            "cpu_capacity": 4,
            "cpu_allocated": 0,
            "mem_capacity": 4,
            "mem_allocated": 0,
            "power_idle_w": 4.0,
            "power_max_w": 8.0,
            "suspended": False,
            "latency_to_ms": [10.0, 0.0],
        },
    )
    pods = tuple(
        {
            "id": pod_id,
            "cpu_request": 2,
            "mem_request": 1,
        }
        for pod_id in range(3)
    )
    graph = {"src": [0, 1], "dst": [1, 2], "rate": [2.0, 2.0], "payload_kb": [1.0, 1.0]}
    return AssignmentInstance("small", nodes, pods, graph)


def test_assignment_objectives_reject_infeasible_and_price_network() -> None:
    instance = _instance()
    with pytest.raises(ValueError, match="capacity"):
        assignment_objectives(instance, (0, 0, 0))
    colocated = assignment_objectives(instance, (1, 1, 0))
    separated = assignment_objectives(instance, (1, 0, 1))
    assert colocated[1] < separated[1]


def test_full_queue_ffd_orders_items_then_first_fits() -> None:
    result = FirstFitDecreasingAssignmentOracle().run(_instance())
    assert result.decreasing_order == (0, 1, 2)
    assert result.assignment == (0, 0, 1)
    assert result.to_dict()["c20_lower_bound"] is False


def test_nsga2_is_seeded_bounded_and_returns_only_feasible_pareto_points() -> None:
    config = NSGA2Config(population_size=8, generations=3)
    oracle = NSGA2AssignmentOracle(config)
    first = oracle.run(_instance(), root_seed=44)
    second = oracle.run(_instance(), root_seed=44)
    assert first.to_dict() == second.to_dict()
    assert first.objective_evaluations == config.objective_evaluation_budget == 32
    assert first.pareto
    for solution in first.pareto:
        assert assignment_objectives(_instance(), solution.assignment) == (
            solution.energy_proxy_w,
            solution.network_cost_kb_ms_s,
        )
    assert "not-reusable-policy" in first.to_dict()["role"]


def test_highs_milp_proves_and_recomputes_small_instance_optimum() -> None:
    pytest.importorskip("highspy")
    config = MILPConfig(energy_weight=1.0, network_weight=0.1)
    first = HighsMILPOracle(config).solve(_instance(), root_seed=17)
    second = HighsMILPOracle(config).solve(_instance(), root_seed=17)
    assert first.to_dict() == second.to_dict()
    assert first.model_status == "kOptimal"
    energy, network = assignment_objectives(_instance(), first.assignment.assignment)
    assert first.scalar_objective == pytest.approx(energy + 0.1 * network)
    # Exhaustive confirmation is practical only in this tiny regression fixture.
    feasible = []
    for assignment in ((a, b, c) for a in range(2) for b in range(2) for c in range(2)):
        try:
            objectives = assignment_objectives(_instance(), assignment)
        except ValueError:
            continue
        feasible.append(objectives[0] + 0.1 * objectives[1])
    assert first.scalar_objective == min(feasible)


def test_milp_refuses_instances_above_contextual_limit() -> None:
    pytest.importorskip("highspy")
    with pytest.raises(ValueError, match="small-MILP"):
        HighsMILPOracle(MILPConfig(max_pods=2)).solve(_instance(), root_seed=1)
