"""``retain_episode_ledgers=False`` must change memory, never the science.

Regression for 2026-09-17: study B of the adaptive budget pilots aborted inside its
initialization when an evaluation child crossed a 2 GiB resident ceiling. One L1+L2
evaluation there retained 16 episodes' ``records`` / ``service_calls`` / ``snapshots``
(a 265 MB pickle, ~106 MiB resident per retained episode), and the campaign consumes
none of them -- it consumes fitness plus archive evidence, and the archive compacts
those ledgers when it writes. Compacting each instance as it is produced is therefore
a resource change only, and this test pins that: identical fitness, identical rung
decisions, identical episode digests and certificate seals, ledgers replaced by their
lengths, and a materially smaller retained payload.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from evolution.evaluation import OMITTED_EPISODE_LEDGERS, load_task_evaluation_config
from evolution.feedback import generate_feedback, load_feedback_profiles
from evolution.ladder import evaluate_through_ladder, load_ladder_config

ROOT = Path(__file__).parents[1]
TASK = ROOT / "evolution" / "task"


def _instances(result) -> list[dict]:
    return [
        instance
        for stage in result.trace
        for instance in (
            ((stage.get("result") or {}).get("evaluation") or {}).get("instances") or []
        )
    ]


def _strip_episodes(payload: dict) -> dict:
    """The trace with every ``episode`` removed, so the rest can be compared exactly."""
    scrubbed = json.loads(json.dumps(payload, sort_keys=True))
    evaluations = [scrubbed.get("evaluation")]
    for stage in scrubbed["trace"]:
        evaluations.append((stage.get("result") or {}).get("evaluation") or {})
    for evaluation in evaluations:
        for instance in (evaluation or {}).get("instances") or []:
            instance.pop("episode", None)
    return scrubbed


@pytest.fixture(scope="module")
def pair():
    task = load_task_evaluation_config(TASK / "config.json")
    ladder = load_ladder_config(TASK / "config.json")
    source = (TASK / "initial.py").read_text()
    full = evaluate_through_ladder(source, task, ladder, proposal_seq=17)
    slim = evaluate_through_ladder(
        source, task, ladder, proposal_seq=17, retain_episode_ledgers=False
    )
    return full, slim


def test_compacting_ledgers_preserves_fitness_decisions_digests_and_seals(pair) -> None:
    full, slim = pair

    assert full.promoted_to_l2 and slim.promoted_to_l2
    assert full.evaluation is not None and slim.evaluation is not None
    assert slim.evaluation.fitness == full.evaluation.fitness
    assert slim.evaluation.fitness.sort_key() == full.evaluation.fitness.sort_key()
    assert [stage["rung"] for stage in slim.trace] == [stage["rung"] for stage in full.trace]
    assert [stage.get("decision") for stage in slim.trace] == [
        stage.get("decision") for stage in full.trace
    ]
    assert [(i["scenario_id"], i["seed"], i["bundle_id"]) for i in _instances(slim)] == [
        (i["scenario_id"], i["seed"], i["bundle_id"]) for i in _instances(full)
    ]
    assert [i["episode_digest"] for i in _instances(slim)] == [
        i["episode_digest"] for i in _instances(full)
    ]
    assert [i["certificate_seal"] for i in _instances(slim)] == [
        i["certificate_seal"] for i in _instances(full)
    ]
    assert _strip_episodes(slim.to_dict()) == _strip_episodes(full.to_dict())


def test_compacting_ledgers_replaces_bulk_lists_and_shrinks_the_retained_result(pair) -> None:
    full, slim = pair

    retained = _instances(slim)
    assert retained, "the seed policy must reach at least one evaluated rung"
    for instance in retained:
        for ledger in OMITTED_EPISODE_LEDGERS:
            assert instance["episode"][ledger]["omitted"] == "horizon-scaled episode ledger"
            assert isinstance(instance["episode"][ledger]["length"], int)
        assert instance["episode"]["ledger"], "non-bulk episode fields must survive"

    for instance, reference in zip(retained, _instances(full), strict=True):
        # `_energy_signals` builds LLM feedback from `episode.energy`; compaction must
        # not touch it, or the operator would see different prompts.
        assert instance["episode"]["energy"] == reference["episode"]["energy"]
        assert instance["episode"]["ledger"] == reference["episode"]["ledger"]

    for instance, reference in zip(retained, _instances(full), strict=True):
        for ledger in OMITTED_EPISODE_LEDGERS:
            assert instance["episode"][ledger]["length"] == len(reference["episode"][ledger])

    assert len(pickle.dumps(slim)) * 4 < len(pickle.dumps(full))


def test_compacting_ledgers_leaves_operator_feedback_byte_identical(pair) -> None:
    full, slim = pair
    task = load_task_evaluation_config(TASK / "config.json")
    profiles = load_feedback_profiles(TASK / "config.json")

    for name, profile in profiles.items():
        rendered = [
            generate_feedback(result, profile, constraints=task.constraints)
            for result in (full, slim)
        ]
        assert rendered[0] == rendered[1], f"feedback profile {name!r} changed under compaction"
