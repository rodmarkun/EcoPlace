from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest
from shinka.core import ProposalOffspring

from evolution.evaluation import (
    PolicyEvaluation,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import load_feedback_profiles
from evolution.g_ablate import REPORT_SCHEMA, load_protocol, run_acceptance_gate, run_gate
from evolution.gp_operator import PlacementScriptGPOperator
from evolution.islands import ArchiveAssessment, DecisionCost, load_islands_config
from evolution.ladder import LadderResult, load_ladder_config
from evolution.llm_operator import LLMCompletion, OpenWeightLLMOperator
from evolution.model import ApiCostLedger, load_model_track
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]
TASK_CONFIG = ROOT / "evolution/task/config.json"
CAMPAIGNS = ROOT / "evolution/campaigns.json"
PROTOCOL = ROOT / "evolution/g_ablate.json"
MODEL = ROOT / "evolution/open_model.json"
INITIAL = ROOT / "evolution/task/initial.py"


def fake_evaluate(source, task, ladder, proposal_seq):
    del task, ladder
    identity = policy_identity_for_source(source)
    evaluation = PolicyEvaluation(
        fitness=CandidateFitness(
            feasible=True,
            infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
            summed_primitive_shortfall=0.0,
            normalized_energy=1.0 + proposal_seq / 100.0,
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
    return LadderResult(
        promoted_to_l2=True,
        evaluation=evaluation,
        trace=(
            {
                "rung": "L2",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            },
        ),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


def fake_assess(source, task, islands):
    del task, islands
    digest = hashlib.sha256(source.encode()).hexdigest()
    return ArchiveAssessment(
        loc=8,
        decision_cost=DecisionCost(
            source_sha256=digest,
            probe_sha256="b" * 64,
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


class DeterministicCompletionClient:
    def __init__(self, source: str) -> None:
        self.source = source

    def complete(self, *, model, prompt, generation, seed):
        return LLMCompletion(
            content=f"```python\n{self.source}```",
            cost_microusd=100,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.source.split()),
            raw_response={"model": model, "generation": generation, "seed": seed},
        )


def context():
    protocol = dataclasses.replace(load_protocol(PROTOCOL), candidate_budget=3)
    task = load_task_evaluation_config(TASK_CONFIG)
    ladder = load_ladder_config(TASK_CONFIG)
    islands = load_islands_config(TASK_CONFIG)
    feedback = load_feedback_profiles(TASK_CONFIG)[protocol.feedback_profile]
    initial = INITIAL.read_text()
    changed = initial.replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.3")
    track = load_model_track(MODEL)

    def llm_factory():
        return OpenWeightLLMOperator(
            track,
            DeterministicCompletionClient(changed),
            ApiCostLedger(track.max_api_costs_microusd),
            proposal_modes=("rewrite",),
        )

    return protocol, task, ladder, islands, feedback, initial, llm_factory


def test_paired_gate_compares_fresh_complete_archives_for_both_real_operator_paths(tmp_path):
    protocol, task, ladder, islands, feedback, initial, llm_factory = context()

    report = run_gate(
        protocol=protocol,
        campaign_plan_path=CAMPAIGNS,
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        initial_source=initial,
        operator_factories={"llm": llm_factory, "gp": PlacementScriptGPOperator},
        archive_root=tmp_path / "gate",
        provenance={"test": "fixed"},
        evaluate_fn=fake_evaluate,
        assess_fn=fake_assess,
    )

    assert report["status"] == "DEVELOPMENT_PASS"
    assert report["acceptance_evidence"] is False
    assert all(arm["bitwise_identical"] for arm in report["arms"].values())
    assert all(arm["coverage_passed"] for arm in report["arms"].values())
    assert (tmp_path / "gate/report.json").is_file()


class TaggedCloneOperator:
    name = "tagged_clone"

    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __call__(self, parent, inspirations, rng: random.Random):
        del inspirations
        return ProposalOffspring(
            source=parent.code,
            metadata={"tag": self.tag, "draw": rng.randrange(1_000_000)},
        )


def test_gate_detects_cross_run_archive_difference(tmp_path):
    protocol, task, ladder, islands, feedback, initial, llm_factory = context()
    created = 0

    def differing_factory():
        nonlocal created
        created += 1
        return TaggedCloneOperator(created)

    report = run_gate(
        protocol=protocol,
        campaign_plan_path=CAMPAIGNS,
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        initial_source=initial,
        operator_factories={"llm": llm_factory, "gp": differing_factory},
        archive_root=tmp_path / "gate",
        provenance={"test": "fixed"},
        evaluate_fn=fake_evaluate,
        assess_fn=fake_assess,
    )

    assert report["status"] == "DEVELOPMENT_FAIL"
    assert report["arms"]["llm"]["bitwise_identical"] is True
    assert report["arms"]["gp"]["bitwise_identical"] is False


class AlwaysInvalidOperator:
    name = "always_invalid"

    def __call__(self, parent, inspirations, rng):
        del parent, inspirations
        return ProposalOffspring(
            source=None,
            error="deterministic failure",
            metadata={"draw": rng.random(), "api_costs_microusd": 0},
        )


def test_gate_rejects_bitwise_identical_all_invalid_primary_arms(tmp_path):
    protocol, task, ladder, islands, feedback, initial, _llm_factory = context()

    report = run_gate(
        protocol=protocol,
        campaign_plan_path=CAMPAIGNS,
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        initial_source=initial,
        operator_factories={
            "llm": AlwaysInvalidOperator,
            "gp": AlwaysInvalidOperator,
        },
        archive_root=tmp_path / "gate",
        provenance={"test": "fixed"},
        evaluate_fn=fake_evaluate,
        assess_fn=fake_assess,
    )

    assert report["status"] == "DEVELOPMENT_FAIL"
    assert all(arm["bitwise_identical"] for arm in report["arms"].values())
    assert all(not arm["coverage_passed"] for arm in report["arms"].values())


def test_p1_protocol_requires_a_predeclared_frozen_tag():
    protocol = load_protocol(PROTOCOL)

    try:
        dataclasses.replace(protocol, status="p1-frozen", frozen_tag=None)
    except ValueError as exc:
        assert "predeclared Git tag" in str(exc)
    else:
        raise AssertionError("P1 protocol accepted without a frozen tag")


def test_injected_gate_api_cannot_emit_acceptance_pass(tmp_path):
    protocol, task, ladder, islands, feedback, initial, llm_factory = context()
    frozen = dataclasses.replace(
        protocol,
        status="p1-frozen",
        frozen_tag="g-ablate-p1-v1",
        proposal_endpoint_origin="https://proposal.example",
        proposal_host_handshake_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="development-only"):
        run_gate(
            protocol=frozen,
            campaign_plan_path=CAMPAIGNS,
            task=task,
            ladder=ladder,
            islands=islands,
            feedback_profile=feedback,
            initial_source=initial,
            operator_factories={"llm": llm_factory, "gp": PlacementScriptGPOperator},
            archive_root=tmp_path / "forged",
            provenance={"forged": True},
            evaluate_fn=fake_evaluate,
            assess_fn=fake_assess,
        )


def test_acceptance_api_rejects_archives_inside_either_source_repository():
    for archive in (ROOT / "acceptance", ROOT.parent / "ShinkaEvolve" / "acceptance"):
        with pytest.raises(ValueError, match="outside both source repositories"):
            run_acceptance_gate(
                archive_root=archive,
                proposal_endpoint="https://proposal.example",
                weight_directory=ROOT.parent / "weights",
            )


def test_acceptance_api_uses_a_fresh_isolated_worker(tmp_path, monkeypatch):
    archive = tmp_path / "acceptance"
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        archive.mkdir()
        (archive / "report.json").write_text(
            json.dumps(
                {
                    "schema": REPORT_SCHEMA,
                    "status": "FAIL",
                    "acceptance_evidence": True,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("evolution.g_ablate.subprocess.run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("PYTHONHOME", "/untrusted")

    report = run_acceptance_gate(
        archive_root=archive,
        proposal_endpoint="https://proposal.example",
        weight_directory=tmp_path / "weights",
    )

    assert report["status"] == "FAIL"
    assert seen["command"][1:4] == ["-I", "-m", "evolution.g_ablate"]
    assert "--acceptance-worker" in seen["command"]
    assert seen["kwargs"]["cwd"] == ROOT
    assert "PYTHONPATH" not in seen["kwargs"]["env"]
    assert "PYTHONHOME" not in seen["kwargs"]["env"]
    assert seen["kwargs"]["env"]["PYTHONNOUSERSITE"] == "1"
