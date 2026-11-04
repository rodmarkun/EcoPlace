"""Retained ladder results must not carry horizon-scaled episode ledgers.

Regression for 2026-09-04: ``ClassicGPHH.run`` and ``RandomASTFloor.run`` kept every
candidate's full ``LadderResult`` (with ``records`` / ``service_calls`` / ``snapshots`` for
three bundles, 60-90 MB resident per evaluation) for the whole run; five concurrent
224.0-work roots exceeded 16 GB of RAM and went into swap. The harness drops
``instances[].episode`` when it writes arm results, so slimming the retained copy cannot
change any sealed output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from evolution.evaluation import (
    OMITTED_EPISODE_LEDGERS,
    InstanceEvidence,
    PolicyEvaluation,
    compact_episode_ledgers,
)
from evolution.ladder import LadderResult

DIGEST = "ab" * 32


def _episode(n: int) -> dict:
    return {
        "records": [[i, "ali-trace-0", 1.0 * i, 2.0 * i, "ok"] for i in range(n)],
        "service_calls": [{"rid": i, "src": 7, "dst": i % 8} for i in range(8 * n)],
        "snapshots": [{"t": i} for i in range(12)],
        "decision_log": [{"pod": i, "node": i % 15} for i in range(20)],
        "rebalancer": {"moves": 3},
        "energy_j": 123.0,
    }


def _evaluation(n: int) -> PolicyEvaluation:
    instance = InstanceEvidence(
        scenario_id="dev-continuum-tight",
        seed=1000151,
        bundle_id="dev-1",
        episode_digest=DIGEST,
        certificate_seal=DIGEST,
        energy_j=1.0,
        idle_floor_j=1.0,
        ceiling_j=2.0,
        slo_achieved={"ali-trace-0": 0.95},
        goodput=0.99,
        drop_rate=0.0,
        timeout_rate=0.0,
        disruption={"moves": 0.0},
        boundary_certificate={"ok": True},
        verified_energy={"ok": True},
        episode=_episode(n),
    )
    fitness = SimpleNamespace(
        feasible=True,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=0.83,
        proposal_seq=1,
        n_instances=2,
        n_bundles=1,
        sort_key=lambda: (0.83, 1),
    )
    return PolicyEvaluation(
        fitness=fitness,  # type: ignore[arg-type]
        instances=(instance, instance),
        policy_name="p",
        policy_version="v",
        policy_identity_sha256=DIGEST,
        policy_artifact_sha256=DIGEST,
    )


def _result(n: int) -> LadderResult:
    return LadderResult(
        promoted_to_l2=True,
        evaluation=_evaluation(n),
        trace=(
            {"level": "L1", "microwork": 300000, "result": {"episode": _episode(n)}},
            {"level": "L2", "microwork": 1000000},
        ),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name="p",
        policy_version="v",
        policy_identity_sha256=DIGEST,
        policy_artifact_sha256=DIGEST,
    )


def test_compact_episode_ledgers_replaces_only_bulk_lists_and_is_idempotent() -> None:
    compact = compact_episode_ledgers(_episode(50))
    for key in OMITTED_EPISODE_LEDGERS:
        assert compact[key]["omitted"] == "horizon-scaled episode ledger"
    assert compact["records"]["length"] == 50
    assert compact["service_calls"]["length"] == 400
    assert compact["decision_log"] == _episode(50)["decision_log"]
    assert compact["rebalancer"] == {"moves": 3}
    assert compact["energy_j"] == 123.0
    assert compact_episode_ledgers(compact) == compact


def test_without_episode_ledgers_keeps_fitness_identity_and_work_fields() -> None:
    full = _result(2000)
    slim = full.without_episode_ledgers()

    assert slim.evaluation is not None and full.evaluation is not None
    assert slim.evaluation.fitness is full.evaluation.fitness
    assert slim.policy_identity_sha256 == full.policy_identity_sha256
    assert [record["microwork"] for record in slim.trace] == [300000, 1000000]
    for instance in slim.evaluation.instances:
        assert instance.episode_digest == DIGEST
        assert instance.episode["records"]["length"] == 2000
        assert "omitted" in instance.episode["service_calls"]
    assert slim.trace[0]["result"]["episode"]["records"]["length"] == 2000

    # The whole point: the retained copy is small.
    assert len(json.dumps(slim.to_dict())) < len(json.dumps(full.to_dict())) / 50


def test_without_episode_ledgers_handles_invalid_candidates() -> None:
    from evolution.ladder import CANDIDATE_FAILURE_SCHEMA

    failed = LadderResult(
        promoted_to_l2=False,
        evaluation=None,
        trace=({"level": "L0", "microwork": 100000},),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name="p",
        policy_version="v",
        policy_identity_sha256=DIGEST,
        policy_artifact_sha256=DIGEST,
        candidate_failure={"schema": CANDIDATE_FAILURE_SCHEMA, "reason": "parse"},
    )
    assert failed.without_episode_ledgers().to_dict() == failed.to_dict()
