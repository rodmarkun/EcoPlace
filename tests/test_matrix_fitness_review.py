"""Independent aggregation tests; manufactured outcomes, no simulator calls."""

import copy

import pytest

from experiments.validation_20260909.verify_matrix_fitness import fitness
from sim.energy.fitness import ConstraintSpec


def data():
    spec = ConstraintSpec({"a": 1.0, "b": 1.0}, 1.0, 1.0, 1.0, {"moves": 1.0})
    rows = [
        {
            "scenario_id": "s",
            "seed": seed,
            "bundle_id": bundle,
            "slo_achieved": {"a": 1.0, "b": 1.0},
            "goodput": 1.0,
            "drop_rate": 0.0,
            "timeout_rate": 0.0,
            "disruption": {"moves": 0.0},
            "energy_j": energy,
            "idle_floor_j": floor,
        }
        for seed in (1, 2)
        for bundle, energy, floor in [("x", 2.0, 1.0), ("y", 3.0, 3.0)]
    ]
    return spec, rows, [(r["scenario_id"], r["seed"], r["bundle_id"]) for r in rows]


def test_energy_uses_mean_of_ratios():
    spec, rows, ids = data()
    result = fitness(spec, rows, ids)
    assert result["feasible"] is True
    assert result["normalized_energy"] == 1.5
    assert result["sort_key"] == [0, 1.5, 0]


def test_bundle_max_precedes_category_max_and_instance_mean():
    spec, rows, ids = data()
    rows[0]["slo_achieved"]["a"] = 0.6
    rows[1]["slo_achieved"]["b"] = 0.4
    rows[2]["slo_achieved"].update(a=0.8, b=0.9)
    result = fitness(spec, rows, ids)
    assert result["infeasibility_vector"] == pytest.approx([0.4, 0, 0, 0])
    assert result["summed_primitive_shortfall"] == pytest.approx(1.3)
    assert result["feasible"] is False and result["normalized_energy"] is None


def test_missing_duplicate_and_nonfinite_metrics_rejected():
    spec, rows, ids = data()
    for broken in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match="grid"):
            fitness(spec, broken, ids)
    bad = copy.deepcopy(rows)
    bad[0]["goodput"] = float("nan")
    with pytest.raises(ValueError, match="delivery"):
        fitness(spec, bad, ids)


def test_zero_cap_has_explicit_scale():
    _, rows, ids = data()
    spec = ConstraintSpec(
        {"a": 1.0, "b": 1.0}, 1.0, 0.0, 1.0, {"moves": 0.0}, {"drop": 0.1, "moves": 4.0}
    )
    rows[0]["drop_rate"] = 0.02
    rows[0]["disruption"]["moves"] = 2
    result = fitness(spec, rows, ids)
    assert result["infeasibility_vector"] == pytest.approx([0, 0, 0.1, 0.25])
    assert result["summed_primitive_shortfall"] == pytest.approx(0.7)
