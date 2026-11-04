from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import placementscript as ps
import pytest
from shinka.core import ProposalOffspring

import evolution.ladder as ladder_module
from baselines.adapter import PolicyBundleAdapter
from evolution.campaign import CampaignRunner, EventArchive, load_campaign_plan
from evolution.evaluation import (
    PolicyEvaluation,
    PolicyExecutionError,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import load_feedback_profiles
from evolution.islands import ArchiveAssessment, DecisionCost, load_islands_config
from evolution.ladder import LadderResult, evaluate_through_ladder, load_ladder_config
from sim.actuation import PlacementScriptProposer
from sim.energy import CandidateFitness
from sim.scheduler import PlacementScriptScorer

ROOT = Path(__file__).parents[1]
TASK = ROOT / "evolution" / "task"
CAMPAIGNS = ROOT / "evolution" / "campaigns.json"


def _failure_source(kind: str) -> str:
    if kind == "trap":
        body = "    return 1.0 / (float(node.cpu_allocated) * 0.0)\n"
    elif kind == "fuel":
        body = (
            "    acc = 0.0\n"
            "    for i in range(100000000):\n"
            "        acc += 1.0\n"
            "    return acc\n"
        )
    else:
        body = (
            "    xs: list[int] = []\n"
            "    for i in range(100000000):\n"
            "        xs.append(i)\n"
            "    return 0.0\n"
        )
    return (
        f'POLICY_NAME: str = "{kind}-candidate"\n'
        'POLICY_VERSION: str = "1"\n'
        "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
        f"{body}"
        "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
        "budget: int) -> list[Move]:\n"
        "    out: list[Move] = []\n"
        "    return out\n"
    )


def _runtime_inputs() -> tuple[dict, list[dict], dict]:
    nodes = [
        {
            "id": 0,
            "tier": 1,
            "cpu_capacity": 4000,
            "cpu_allocated": 0,
            "mem_capacity": 8192,
            "mem_allocated": 0,
            "power_idle_w": 2.0,
            "power_max_w": 8.0,
            "suspended": False,
            "suspendable": True,
            "latency_to_ms": [0.0],
        }
    ]
    pods = [
        {
            "id": 0,
            "service": 0,
            "app": 0,
            "cpu_request": 100,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": -1,
        }
    ]
    graph = {"src": [], "dst": [], "rate": [], "payload_kb": []}
    return pods[0], nodes, {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}


def _execute_score(scorer: object, kind: str) -> None:
    pod, nodes, context = _runtime_inputs()
    if kind == "memory":
        scorer._policy.score(pod, nodes[0], context, fuel=10_000_000_000)
    else:
        scorer.score_batch(pod, nodes, context)


def _successful_l0() -> dict:
    return {
        "valid": True,
        "feasible": True,
        "mean_normalized_energy": 1.0,
        "instances": [],
    }


def _successful_evaluation(source: str, proposal_seq: int) -> PolicyEvaluation:
    identity = policy_identity_for_source(source)
    return PolicyEvaluation(
        fitness=CandidateFitness(
            feasible=True,
            infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
            summed_primitive_shortfall=0.0,
            normalized_energy=1.0,
            proposal_seq=proposal_seq,
            n_instances=1,
            n_bundles=1,
        ),
        instances=(),
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


def _successful_ladder_result(source: str, proposal_seq: int) -> LadderResult:
    identity = policy_identity_for_source(source)
    return LadderResult(
        promoted_to_l2=True,
        evaluation=_successful_evaluation(source, proposal_seq),
        trace=tuple(
            {"rung": rung, "decision": "promote" if rung != "L2" else "fitness-of-record"}
            for rung in ("L0", "L1", "L2")
        ),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


@pytest.mark.parametrize(
    ("kind", "failure_category"),
    [
        ("trap", "placementscript-runtime-trap"),
        ("fuel", "placementscript-fuel-exhausted"),
        ("memory", "placementscript-out-of-memory"),
    ],
)
@pytest.mark.parametrize("rung", ["L0", "L1", "L2"])
def test_runtime_failures_invalidate_at_the_rung(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    failure_category: str,
    rung: str,
) -> None:
    source = _failure_source(kind)
    task = load_task_evaluation_config(TASK / "config.json")
    ladder = dataclasses.replace(
        load_ladder_config(TASK / "config.json"),
        epsilon_audit_fraction=0.0,
    )

    if rung == "L0":
        monkeypatch.setattr(
            ladder_module,
            "_run_l0_adapter",
            lambda scorer, _task: _execute_score(scorer, kind),
        )
    else:
        monkeypatch.setattr(
            ladder_module,
            "_run_l0_adapter",
            lambda _scorer, _task: _successful_l0(),
        )
    if rung == "L1":
        monkeypatch.setattr(
            ladder_module,
            "_run_l1_adapter",
            lambda scorer, *_args, **_kwargs: _execute_score(scorer, kind),
        )
    elif rung == "L2":
        monkeypatch.setattr(
            ladder_module,
            "_run_l1_adapter",
            lambda _scorer, _proposer, _task, _ladder, proposal_seq, **_kwargs: (
                _successful_evaluation(source, proposal_seq)
            ),
        )
        monkeypatch.setattr(
            ladder_module,
            "evaluate_policy_adapter",
            lambda scorer, *_args, **_kwargs: _execute_score(scorer, kind),
        )

    result = evaluate_through_ladder(source, task, ladder, proposal_seq=17)

    assert result.promoted_to_l2 is False
    assert result.evaluation is None
    assert result.candidate_failure == result.trace[-1]["result"]["failure"]
    assert result.candidate_failure["failure_category"] == failure_category
    assert result.candidate_failure["stage"] == rung
    assert result.trace[-1]["rung"] == rung
    assert result.trace[-1]["decision"] == "invalid"


@pytest.mark.parametrize(
    ("kind", "failure_category"),
    [
        ("trap", "placementscript-runtime-trap"),
        ("fuel", "placementscript-fuel-exhausted"),
    ],
)
def test_trapping_reference_policy_fails_loudly(
    kind: str, failure_category: str
) -> None:
    task = load_task_evaluation_config(TASK / "config.json")
    source = _failure_source(kind)
    policy = ps.Policy(source)
    reference = PolicyBundleAdapter(
        name=f"trapping-reference-{kind}",
        version="1",
        scorer=PlacementScriptScorer(policy),
        proposer=PlacementScriptProposer(policy),
        representation="test-reference",
        artifact_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )

    with pytest.raises(PolicyExecutionError) as caught:
        reference.evaluate(task, proposal_seq=23)

    assert caught.value.failure["failure_category"] == failure_category
    assert caught.value.failure["stage"] == "episode"


class RuntimeFailureOperator:
    name = "runtime-failure-test"

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __call__(self, parent, inspirations, rng):
        del inspirations, rng
        replacement = (
            "    return 1.0 / (float(node.cpu_allocated) * 0.0)\n"
            if self.kind == "trap"
            else (
                "    acc = 0.0\n"
                "    for i in range(100000000):\n"
                "        acc += 1.0\n"
                "    return acc\n"
            )
        )
        source = parent.code.replace(
            "    return free_fraction_after(pod, node)\n",
            replacement,
        )
        if source == parent.code:
            raise AssertionError("test parent lacks the registered score return")
        return ProposalOffspring(
            source=source,
            metadata={"operator": self.name, "invalid_attempts": 0, "retries": 0},
        )


def _assessment(source: str, task: object, islands: object) -> ArchiveAssessment:
    del task, islands
    if "* 0.0" in source or "range(100000000)" in source:
        raise AssertionError("a rung-invalid candidate reached archive assessment")
    digest = hashlib.sha256(source.encode()).hexdigest()
    return ArchiveAssessment(
        loc=10,
        decision_cost=DecisionCost(
            source_sha256=digest,
            probe_sha256="a" * 64,
            score_batch_fuel=100,
            score_batch_memory_bytes=64,
            calibrated_latency_us=2.0,
            calibration_id="test",
            calibration_status="development-provisional",
        ),
        loc_penalty=0.0,
        decision_latency_penalty=0.0,
        total_penalty=0.0,
    )


def _campaign(path: Path, *, failure_kind: str = "trap") -> CampaignRunner:
    plan = dataclasses.replace(load_campaign_plan(CAMPAIGNS), candidate_budget=3)
    task = load_task_evaluation_config(TASK / "config.json")
    ladder = load_ladder_config(TASK / "config.json")
    islands = load_islands_config(TASK / "config.json")
    feedback = load_feedback_profiles(TASK / "config.json")[plan.feedback_profile]
    return CampaignRunner(
        definition=plan.campaigns[0],
        plan=plan,
        arm="gp",
        operator=RuntimeFailureOperator(failure_kind),
        initial_source=(TASK / "initial.py").read_text(),
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        archive_dir=path,
        provenance={"test": "runtime-trap-fixed-seed"},
        run_purpose="determinism-smoke",
        evaluate_fn=lambda source, task, ladder, proposal_seq: evaluate_through_ladder(
            source, task, ladder, proposal_seq=proposal_seq
        ),
        assess_fn=_assessment,
    )


@pytest.mark.parametrize(
    ("kind", "failure_category", "source_marker"),
    [
        ("trap", "placementscript-runtime-trap", "* 0.0"),
        ("fuel", "placementscript-fuel-exhausted", "range(100000000)"),
    ],
)
def test_runtime_invalid_candidate_is_accounted_archived_and_deterministic(
    tmp_path: Path,
    kind: str,
    failure_category: str,
    source_marker: str,
) -> None:
    first = _campaign(tmp_path / "first", failure_kind=kind)
    second = _campaign(tmp_path / "second", failure_kind=kind)

    first_digest = first.run()
    second_digest = second.run()

    assert first_digest == second_digest
    assert (tmp_path / "first" / "events.jsonl").read_bytes() == (
        tmp_path / "second" / "events.jsonl"
    ).read_bytes()
    completed = list(first._completed().values())
    invalid = [payload for payload in completed[1:] if not payload["proposal"]["valid"]]
    assert len(invalid) == 2
    assert all(payload["proposal"]["source"] is not None for payload in invalid)
    assert all(source_marker in payload["proposal"]["source"] for payload in invalid)
    assert all(
        payload["proposal"]["metadata"]["evaluation_invalid_attempts"] == 1
        for payload in invalid
    )
    assert all(
        payload["proposal"]["metadata"]["candidate_failure"]["failure_category"]
        == failure_category
        for payload in invalid
    )
    assert all(payload["decision"]["eligible"] is False for payload in invalid)
    assert all(payload["evaluation"]["evaluation"] is None for payload in invalid)
    assert all(
        payload["evaluator_work"]["candidate_microwork"]
        == first.plan.evaluator_work.rung_microwork["L0"]
        for payload in invalid
    )
    assert first._spent_microwork() == sum(
        payload["evaluator_work"]["candidate_microwork"] for payload in completed
    )
    assert first._terminal()["reason"] == "determinism-smoke-candidate-target"
    assert first._terminal()["completed_candidates"] == 3
    assert len(list((tmp_path / "first" / "candidates").glob("*.ps"))) == 3

    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    reloaded = EventArchive(tmp_path / "first", manifest)
    reloaded.validate_semantics()
    assert reloaded.digest() == first_digest


def test_runtime_trap_in_archive_assessment_also_invalidates_candidate(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path / "assessment")
    campaign.evaluate_fn = lambda source, _task, _ladder, proposal_seq: (
        _successful_ladder_result(source, proposal_seq)
    )

    def trap_on_offspring(source: str, task: object, islands: object) -> ArchiveAssessment:
        if "* 0.0" in source:
            raise ps.TrapError("decision-cost probe trapped")
        return _assessment(source, task, islands)

    campaign.assess_fn = trap_on_offspring
    campaign.run()

    invalid = list(campaign._completed().values())[1:]
    assert all(payload["proposal"]["valid"] is False for payload in invalid)
    assert all(payload["decision"]["eligible"] is False for payload in invalid)
    assert all(
        payload["proposal"]["metadata"]["candidate_failure"]["stage"]
        == "archive-assessment"
        for payload in invalid
    )
    assert all(
        payload["evaluator_work"]["candidate_microwork"]
        == campaign.plan.evaluator_work.max_candidate_microwork
        for payload in invalid
    )
