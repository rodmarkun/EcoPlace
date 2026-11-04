from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from baselines.ppo import GraphEncodingConfig, PlacementEpisode, PPOCorpus, PPOHyperparameters
from baselines.ppo_retraining import (
    PeriodicPPOArm,
    PeriodicPPOPlan,
    RetrainingBatch,
    load_periodic_ppo_plan,
)

ROOT = Path(__file__).parents[1]


def _episode(source: str, split: str, decision_step: int) -> PlacementEpisode:
    node = {
        "id": 0,
        "tier": 0,
        "cpu_capacity": 4,
        "cpu_allocated": 0,
        "mem_capacity": 4,
        "mem_allocated": 0,
        "power_idle_w": 1.0,
        "power_max_w": 2.0,
        "suspended": False,
        "suspendable": False,
        "latency_to_ms": [0.0],
    }
    pod = {
        "id": 0,
        "service": 0,
        "app": 0,
        "cpu_request": 1,
        "mem_request": 1,
        "stateful": False,
        "has_pdb": False,
        "image_mb": 1,
        "current_node": -1,
    }
    ctx = {
        "nodes": [node],
        "pods": [pod],
        "graph": {"src": [], "dst": [], "rate": [], "payload_kb": []},
        "now_ms": decision_step,
    }
    return PlacementEpisode(
        f"episode-{source}-{decision_step}",
        source * 64,
        source * 64,
        "online-history-before-step",
        split,
        decision_step,
        ctx,
        (0,),
    )


def _plan() -> PeriodicPPOPlan:
    return PeriodicPPOPlan(
        status="exploratory-only",
        registration_source="test",
        interval_steps=10,
        max_retrains=2,
        max_total_timesteps=16,
        step_unit="placement-decision",
        hyperparameters=PPOHyperparameters(0.001, 8, 8, 1, 0.9, 0.0, (8,), 8),
        encoding=GraphEncodingConfig(max_nodes=1),
    )


def _batch(step: int) -> RetrainingBatch:
    return RetrainingBatch(
        step,
        PPOCorpus("online-history-before-step", "train", (_episode("a", "train", step - 2),)),
        PPOCorpus(
            "online-history-before-step",
            "validation",
            (_episode("b", "validation", step - 1),),
        ),
    )


def test_periodic_retraining_is_seeded_scheduled_bounded_and_exploratory(tmp_path: Path) -> None:
    calls = []

    def train(training, validation, budget, encoding, output, seed):
        calls.append((training, validation, budget, encoding, output, seed))
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "trial-000.zip"
        artifact.write_bytes(f"model:{seed}".encode())
        payload = {
            "data_role": budget.data_role,
            "root_seed": seed,
            "training_corpus_sha256": training.sha256,
            "validation_corpus_sha256": validation.sha256,
            "actual_total_timesteps": budget.trials[0].total_timesteps,
            "champion_artifact": "trial-000.zip",
            "champion_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "trials": [{"hyperparameters": dataclasses.asdict(budget.trials[0])}],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}

    arm = PeriodicPPOArm(_plan(), train_fn=train)
    first = arm.run((_batch(10), _batch(20)), tmp_path / "a", root_seed=7)
    first_seeds = [call[-1] for call in calls]
    calls.clear()
    second = arm.run((_batch(10), _batch(20)), tmp_path / "a", root_seed=7)
    assert first.to_dict() == second.to_dict()
    assert first_seeds == [call[-1] for call in calls]
    assert first.total_timesteps == 16
    assert first.to_dict()["role"] == "exploratory-periodic-retraining"


def test_periodic_retraining_rejects_future_leak_and_off_schedule(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly before"):
        RetrainingBatch(
            10,
            PPOCorpus("online-history-before-step", "train", (_episode("a", "train", 10),)),
            PPOCorpus(
                "online-history-before-step",
                "validation",
                (_episode("b", "validation", 9),),
            ),
        )
    arm = PeriodicPPOArm(_plan(), train_fn=lambda *args: {})
    with pytest.raises(ValueError, match="outside"):
        arm.run((_batch(11),), tmp_path, root_seed=1)


def test_periodic_plan_has_one_fixed_trial_and_checked_in_plan_is_exploratory() -> None:
    budget = _plan().training_budget()
    assert len(budget.trials) == 1
    assert budget.data_role == "online-history-before-step"
    checked_in = load_periodic_ppo_plan(ROOT / "baselines" / "ppo_retraining.json")
    assert checked_in.status == "exploratory-only"
    assert checked_in.max_retrains == 5
    assert checked_in.max_total_timesteps == 81_920


def test_online_training_role_forbids_hyperparameter_search() -> None:
    hyperparameters = PPOHyperparameters(0.001, 8, 8, 1, 0.9, 0.0, (8,), 8)
    from baselines.ppo import PPOTuningBudget

    with pytest.raises(ValueError, match="one P2-fixed"):
        PPOTuningBudget(
            "development-provisional",
            "test",
            "online-history-before-step",
            (hyperparameters, hyperparameters),
            16,
        )
