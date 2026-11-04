from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import placementscript as ps

from evolution.evaluation import load_task_evaluation_config
from evolution.ladder import (
    epsilon_audit_selected,
    evaluate_through_ladder,
    load_ladder_config,
)

ROOT = Path(__file__).parents[1]
TASK = ROOT / "evolution" / "task"


def test_initial_policy_is_loadable_and_scoped_to_evolve_blocks():
    source = (TASK / "initial.py").read_text()
    policy = ps.Policy(source)

    assert policy.name == "least_allocated"
    assert source.count("EVOLVE-BLOCK-START") == 1
    assert source.count("EVOLVE-BLOCK-END") == 1
    assert source.index("POLICY_VERSION") < source.index("EVOLVE-BLOCK-START")


def test_task_config_pins_complete_development_grid():
    task = load_task_evaluation_config(TASK / "config.json")

    assert task.bundle_ids == ("dev-1", "dev-2", "dev-3")
    assert task.instance_keys == (
        ("dev-small-cluster", 104729),
        ("dev-small-cluster", 130363),
    )


def test_seed_policy_runs_through_ladder_with_exact_phase3_fitness():
    task = load_task_evaluation_config(TASK / "config.json")
    ladder = load_ladder_config(TASK / "config.json")
    result = evaluate_through_ladder(
        (TASK / "initial.py").read_text(),
        task,
        ladder,
        proposal_seq=17,
    )
    assert result.evaluation is not None
    record = result.evaluation.to_dict()

    assert [item["rung"] for item in result.trace] == ["L0", "L1", "L2"]
    assert len(result.trace[1]["result"]["evaluation"]["instances"]) == 1
    assert result.promoted_to_l2
    assert record["fitness"]["sort_key"] == list(result.evaluation.fitness.sort_key())
    assert record["fitness"]["proposal_seq"] == 17
    assert record["fitness"]["n_instances"] == 2
    assert record["fitness"]["n_bundles"] == 3
    assert len(record["instances"]) == 6
    assert all(len(instance["certificate_seal"]) == 64 for instance in record["instances"])
    assert all(
        instance["episode"]["energy"]["device_j_per_node"] for instance in record["instances"]
    )
    assert all(instance["boundary_certificate"]["seal"] for instance in record["instances"])
    assert all(instance["verified_energy"]["seal"] for instance in record["instances"])
    assert result.trace[0]["result"]["instances"][0]["per_snapshot"]
    assert result.trace[0]["result"]["instances"][0]["snapshot_library"]["payload"]
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_epsilon_audit_selection_is_seeded_and_order_independent():
    ladder = dataclasses.replace(
        load_ladder_config(TASK / "config.json"),
        epsilon_audit_fraction=0.5,
    )
    forward = [
        epsilon_audit_selected(ladder, proposal_seq=seq, rejected_at="L0") for seq in range(20)
    ]
    reverse = {
        seq: epsilon_audit_selected(ladder, proposal_seq=seq, rejected_at="L0")
        for seq in reversed(range(20))
    }

    assert forward == [reverse[seq] for seq in range(20)]
    assert any(forward) and not all(forward)


def test_epsilon_audit_observes_but_does_not_reverse_l0_rejection():
    task = load_task_evaluation_config(TASK / "config.json")
    ladder = load_ladder_config(TASK / "config.json")
    ladder = dataclasses.replace(
        ladder,
        l0=dataclasses.replace(ladder.l0, max_mean_normalized_energy=0.1),
        epsilon_audit_fraction=1.0,
    )

    result = evaluate_through_ladder(
        (TASK / "initial.py").read_text(),
        task,
        ladder,
        proposal_seq=18,
    )

    assert not result.promoted_to_l2
    assert result.evaluation is None
    assert result.trace[0]["decision"] == "reject"
    assert result.trace[1]["rung"] == "epsilon-audit"
    assert result.trace[1]["selected"] is True
    assert result.trace[1]["target_rung"] == "L1"
    assert "fitness_sort_key" in result.trace[1]["result"]
