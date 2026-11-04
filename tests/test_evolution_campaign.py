from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path

import pytest
from shinka.core import ProposalOffspring
from shinka.database import Program

from analysis.tracking import derive_seed
from evolution.campaign import (
    AUTHORIZATION_SCHEMA,
    CONFIRMATORY_PREREQUISITES,
    CampaignRunner,
    load_campaign_plan,
    verify_confirmatory_authorization,
)
from evolution.evaluation import (
    PolicyEvaluation,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import load_feedback_profiles
from evolution.islands import (
    ArchiveAssessment,
    DecisionCost,
    load_islands_config,
)
from evolution.ladder import LadderResult, load_ladder_config
from evolution.llm_operator import LLMCompletion, OpenWeightLLMOperator
from evolution.model import ApiCostLedger, load_model_track
from evolution.operators import AblationProbeInterleaveOperator
from evolution.run_campaign import (
    CANONICAL_INPUTS,
    CONFIRMATORY_RECEIPT_SCHEMA,
    _verify_canonical_tagged_inputs,
    _write_confirmatory_receipt,
)
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]
TASK = ROOT / "evolution" / "task"
CAMPAIGNS = ROOT / "evolution" / "campaigns.json"
MODEL = ROOT / "evolution" / "open_model.json"


class CloneOperator:
    name = "clone"

    def __call__(self, parent, inspirations, rng):
        return ProposalOffspring(
            source=parent.code,
            metadata={
                "operator": self.name,
                "prompt": f"parent={parent.id};inspirations={len(inspirations)}",
                "draw": rng.randrange(1_000_000),
                "raw_response": {"content": parent.code},
            },
        )


def fake_evaluate(source, task, ladder, proposal_seq):
    del task, ladder
    identity = policy_identity_for_source(source)
    fitness = CandidateFitness(
        feasible=True,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=1.0 + proposal_seq / 100.0,
        proposal_seq=proposal_seq,
        n_instances=2,
        n_bundles=3,
    )
    evaluation = PolicyEvaluation(
        fitness=fitness,
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
                "decision": "fitness-of-record",
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
    cost = DecisionCost(
        source_sha256=digest,
        probe_sha256="a" * 64,
        score_batch_fuel=100,
        score_batch_memory_bytes=64,
        calibrated_latency_us=2.0,
        calibration_id="test",
        calibration_status="development-provisional",
    )
    return ArchiveAssessment(
        loc=10,
        decision_cost=cost,
        loc_penalty=0.0,
        decision_latency_penalty=0.0,
        total_penalty=0.0,
    )


def runner(
    path,
    *,
    evaluate_fn=fake_evaluate,
    plan_transform=None,
    run_purpose="determinism-smoke",
    arm="gp",
    terminal_budget_policy="carry-to-work-limit",
    evaluation_seed_mode="campaign-derived",
    operator=None,
):
    plan = dataclasses.replace(load_campaign_plan(CAMPAIGNS), candidate_budget=3)
    if plan_transform is not None:
        plan = plan_transform(plan)
    task = load_task_evaluation_config(TASK / "config.json")
    ladder = load_ladder_config(TASK / "config.json")
    islands = load_islands_config(TASK / "config.json")
    feedback = load_feedback_profiles(TASK / "config.json")[plan.feedback_profile]
    return CampaignRunner(
        definition=plan.campaigns[0],
        plan=plan,
        arm=arm,
        operator=operator or CloneOperator(),
        initial_source=(TASK / "initial.py").read_text(),
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        archive_dir=path,
        provenance={"test": "fixed"},
        run_purpose=run_purpose,
        terminal_budget_policy=terminal_budget_policy,
        evaluation_seed_mode=evaluation_seed_mode,
        evaluate_fn=evaluate_fn,
        assess_fn=fake_assess,
    )


def test_campaign_archives_creation_before_every_completion_and_full_evidence(tmp_path):
    campaign = runner(tmp_path / "archive")
    digest = campaign.run()

    assert len(digest) == 64
    types = [event["type"] for event in campaign.archive.events]
    assert types == [
        "candidate_created",
        "candidate_completed",
        "candidate_created",
        "candidate_completed",
        "candidate_created",
        "candidate_completed",
        "campaign_terminal",
    ]
    completed = [
        event["payload"]
        for event in campaign.archive.events
        if event["type"] == "candidate_completed"
    ]
    assert [item["proposal_seq"] for item in completed] == [0, 1, 2]
    assert all(item["evaluation"] is not None for item in completed)
    assert all(item["decision"]["survivor_key"] is not None for item in completed)
    assert all(item["decision"]["elite_view"]["key"] is not None for item in completed)
    assert [item["evaluator_work"]["cumulative_microwork"] for item in completed] == [
        1_000_000,
        2_000_000,
        3_000_000,
    ]
    assert "prompt" in completed[1]["proposal"]["metadata"]
    assert len(list((tmp_path / "archive" / "candidates").glob("*.ps"))) == 3
    terminal = campaign.archive.events[-1]["payload"]
    assert terminal["reason"] == "determinism-smoke-candidate-target"
    assert terminal["usable_for_confirmatory_analysis"] is False


def test_ablation_flags_off_preserve_the_matched_archive_byte_identity(tmp_path):
    campaign = runner(
        tmp_path / "archive",
        plan_transform=lambda plan: dataclasses.replace(
            plan,
            ablation_probe_probability=0.0,
            ablation_prompt_hint_probability=0.0,
        ),
    )

    assert campaign.run() == (
        "566765cb203a74d9ae0349a886dfbf22ab481a4e9452924cc6e18ff77bf6befa"
    )
    assert hashlib.sha256(campaign.archive.manifest_path.read_bytes()).hexdigest() == (
        "e14e36e36774d96ab482d6b0ee6d63babddf0d2b042547d89964f69b08b9761b"
    )
    assert hashlib.sha256(campaign.archive.events_path.read_bytes()).hexdigest() == (
        "cd8bf63f60680cefa2fe29cab255a08013c7009cb85eb4d56e0df3cbd4258e90"
    )


def test_ablation_campaign_archives_route_kind_site_and_explicit_seed(tmp_path):
    operator = AblationProbeInterleaveOperator(
        CloneOperator(),
        probability=0.8,
        shared_numeric_probability=0.2,
    )
    campaign = runner(
        tmp_path / "archive",
        operator=operator,
        plan_transform=lambda plan: dataclasses.replace(
            plan,
            ablation_probe_probability=0.8,
        ),
    )
    campaign.run()

    attempts = [
        campaign._completed()[seq]["proposal"]["attempts"][0]
        for seq in (1, 2)
    ]
    assert all(attempt["metadata"]["route"] == "ablation_probe" for attempt in attempts)
    assert all(
        attempt["ablation_probe_seed"]
        == attempt["metadata"]["ablation_probe_seed"]
        for attempt in attempts
    )
    assert all(
        attempt["metadata"]["ablation_kind"]
        == attempt["metadata"]["ablation_site"]["kind"]
        for attempt in attempts
    )


def test_campaign_derives_every_component_from_its_single_root(tmp_path):
    campaign = runner(tmp_path / "archive")
    definition = campaign.definition
    scenario = campaign.task.scenarios[0]

    assert scenario.seeds == tuple(
        derive_seed(
            definition.root_seed,
            "sim",
            "scenario",
            scenario.scenario_id,
            "replicate",
            replicate,
        )
        for replicate in range(definition.evaluation_replicates)
    )
    assert campaign.ladder.audit_root_seed == derive_seed(
        definition.root_seed, "evolution", "ladder-audit"
    )


def test_proposal_rngs_split_paired_shared_and_arm_specific_paths(tmp_path):
    gp = runner(tmp_path / "gp", arm="gp")
    llm = runner(tmp_path / "llm", arm="llm")
    gp.run()
    llm.run()
    gp_attempt = gp._completed()[1]["proposal"]["attempts"][0]
    llm_attempt = llm._completed()[1]["proposal"]["attempts"][0]

    assert gp_attempt["shared_route_seed"] == llm_attempt["shared_route_seed"]
    assert gp_attempt["shared_numeric_seed"] == llm_attempt["shared_numeric_seed"]
    assert gp_attempt["arm_primary_seed"] != llm_attempt["arm_primary_seed"]
    assert gp_attempt["arm_primary_seed"] == derive_seed(
        gp.definition.root_seed,
        "evolution",
        "gp",
        "primary-proposal",
        1,
        0,
    )


def test_registered_work_budget_stops_before_a_full_candidate_would_exceed_it(tmp_path):
    def limited(plan):
        return dataclasses.replace(
            plan,
            candidate_budget=5,
            evaluator_work=dataclasses.replace(
                plan.evaluator_work,
                max_microwork=2_400_000,
                checkpoints_microwork=(0, 2_400_000),
            ),
        )

    campaign = runner(
        tmp_path / "archive",
        plan_transform=limited,
        run_purpose="scientific-full-work",
    )

    campaign.run()

    assert len(campaign._completed()) == 2
    assert campaign._spent_microwork() == 2_000_000
    terminal = campaign.archive.events[-1]["payload"]
    assert terminal["reason"] == "normalized-work-budget-exhausted"
    assert terminal["budget_carry_microwork"] == 400_000
    assert terminal["unused_microwork_discarded"] == 0
    assert terminal["effective_terminal_microwork"] == 2_400_000
    assert terminal["usable_for_confirmatory_analysis"] is False
    assert terminal["confirmatory_postflight_receipt_required"] is True


def test_trigger_local_campaign_discards_remainder_and_preserves_presealed_seeds(tmp_path):
    campaign = runner(
        tmp_path / "archive",
        plan_transform=lambda plan: dataclasses.replace(
            plan,
            evaluator_work=dataclasses.replace(
                plan.evaluator_work,
                max_microwork=2_400_000,
                checkpoints_microwork=(0, 2_400_000),
            ),
        ),
        run_purpose="scientific-full-work",
        terminal_budget_policy="discard-at-trigger",
        evaluation_seed_mode="presealed-task",
    )
    original_seeds = load_task_evaluation_config(TASK / "config.json").instance_keys
    campaign.run()
    terminal = campaign._terminal()
    assert terminal is not None
    assert terminal["budget_carry_microwork"] == 0
    assert terminal["unused_microwork_discarded"] == 400_000
    assert campaign.task.instance_keys == original_seeds


def test_confirmatory_postflight_receipt_is_exclusive_and_idempotent(tmp_path):
    path = tmp_path / "campaign.confirmatory-postflight.json"
    payload = {
        "schema": CONFIRMATORY_RECEIPT_SCHEMA,
        "status": "PASS",
        "archive_sha256": "a" * 64,
    }

    _write_confirmatory_receipt(path, payload)
    first = path.read_bytes()
    _write_confirmatory_receipt(path, payload)

    assert path.read_bytes() == first
    with pytest.raises(RuntimeError, match="receipt differs"):
        _write_confirmatory_receipt(path, {**payload, "archive_sha256": "b" * 64})


def test_scientific_candidate_ceiling_marks_archive_unusable_and_fails(tmp_path):
    archive = tmp_path / "archive"
    campaign = runner(archive, run_purpose="scientific-full-work")

    with pytest.raises(RuntimeError, match="safety ceiling"):
        campaign.run()

    terminal = campaign.archive.events[-1]
    assert terminal["type"] == "campaign_terminal"
    assert terminal["payload"]["status"] == "unusable"
    assert terminal["payload"]["reason"] == "candidate-safety-ceiling-reached"
    before = campaign.archive.events_path.read_bytes()
    with pytest.raises(RuntimeError, match="archive is unusable"):
        runner(archive, run_purpose="scientific-full-work").run()
    assert campaign.archive.events_path.read_bytes() == before


def test_resume_after_creation_reuses_sequence_and_is_bitwise_identical(tmp_path):
    class CrashOnce:
        def __init__(self):
            self.crashed = False

        def __call__(self, source, task, ladder, proposal_seq):
            if not self.crashed:
                self.crashed = True
                raise RuntimeError("simulated interruption")
            return fake_evaluate(source, task, ladder, proposal_seq)

    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="interruption"):
        runner(interrupted, evaluate_fn=CrashOnce()).run()
    resumed = runner(interrupted)
    resumed_digest = resumed.run()

    clean = runner(tmp_path / "clean")
    clean_digest = clean.run()
    assert resumed_digest == clean_digest
    assert (interrupted / "events.jsonl").read_bytes() == (
        tmp_path / "clean" / "events.jsonl"
    ).read_bytes()


def test_resume_discards_only_an_unterminated_event_tail(tmp_path):
    clean = runner(tmp_path / "archive")
    digest = clean.run()
    events = clean.archive.events_path
    with events.open("ab") as stream:
        stream.write(b'{"event_seq":999')

    resumed = runner(tmp_path / "archive")

    assert resumed.run() == digest
    assert events.read_bytes().endswith(b"\n")


def test_resume_repairs_materialized_candidates_after_terminal_append(tmp_path):
    archive = tmp_path / "archive"
    clean = runner(archive)
    digest = clean.run()
    missing = archive / "candidates" / "00000001.ps"
    missing.unlink()

    resumed = runner(archive)

    assert resumed.run() == digest
    assert missing.is_file()


def test_invalid_event_is_rejected_before_it_can_corrupt_archive(tmp_path):
    campaign = runner(tmp_path / "archive")
    campaign.run()
    before = campaign.archive.events_path.read_bytes()

    with pytest.raises(ValueError, match="terminal event must be the final event"):
        campaign.archive.append("unregistered", {})

    assert campaign.archive.events_path.read_bytes() == before


def test_confirmatory_authorization_is_exactly_hash_and_tag_bound(tmp_path):
    base = load_campaign_plan(CAMPAIGNS)
    frozen_tag = "p2-campaign-v1"
    payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "frozen_tag": frozen_tag,
        "prerequisites": CONFIRMATORY_PREREQUISITES,
        "proposal_endpoint_origin": "https://proposal.example",
        "proposal_host_handshake_sha256": "b" * 64,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(encoded)
    plan = dataclasses.replace(
        base,
        status="p2-frozen",
        frozen_tag=frozen_tag,
        confirmatory_authorization_sha256=hashlib.sha256(encoded).hexdigest(),
        evaluator_work=dataclasses.replace(base.evaluator_work, registration_status="p2-frozen"),
    )

    assert verify_confirmatory_authorization(plan, authorization) == payload

    authorization.write_bytes(encoded + b"\n")
    with pytest.raises(ValueError, match="hash pin"):
        verify_confirmatory_authorization(plan, authorization)


def test_confirmatory_inputs_reject_an_off_repository_config(tmp_path):
    supplied = dict(CANONICAL_INPUTS)
    supplied["campaign_config"] = tmp_path / "campaigns.json"

    with pytest.raises(ValueError, match="canonical repository path"):
        _verify_canonical_tagged_inputs(supplied, tag="unused")


class FakeCompletionClient:
    def __init__(self, source):
        self.source = source

    def complete(self, *, model, prompt, generation, seed):
        return LLMCompletion(
            content=f"```python\n{self.source}```",
            cost_microusd=1000,
            prompt_tokens=100,
            completion_tokens=200,
            raw_response={
                "model": model,
                "seed": seed,
                "generation": generation,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        )


class FailingCompletionClient:
    def complete(self, **kwargs):
        del kwargs
        raise TimeoutError("ambiguous post-dispatch timeout")


class UnpricedCompletionClient(FakeCompletionClient):
    def complete(self, **kwargs):
        completion = super().complete(**kwargs)
        return dataclasses.replace(completion, cost_microusd=None)


def test_open_weight_operator_archives_prompt_response_seed_and_cost():
    source = (TASK / "initial.py").read_text()
    changed = source.replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.3")
    track = load_model_track(MODEL)
    ledger = ApiCostLedger(track.max_api_costs_microusd)
    operator = OpenWeightLLMOperator(
        track,
        FakeCompletionClient(changed),
        ledger,
        proposal_modes=("rewrite",),
    )
    program = Program(id="p", code=source, timestamp=0.0)
    offspring = operator(program, (), random.Random(123))

    assert offspring.valid
    assert offspring.metadata["prompt"]
    assert offspring.metadata["raw_response"]
    assert isinstance(offspring.metadata["call_seed"], int)
    assert offspring.metadata["api_costs_microusd"] == 1000
    assert ledger.spent_microusd == 1000


def test_ambiguous_proposal_failure_consumes_full_cost_reservation():
    source = (TASK / "initial.py").read_text()
    track = load_model_track(MODEL)
    ledger = ApiCostLedger(track.max_api_costs_microusd)
    operator = OpenWeightLLMOperator(track, FailingCompletionClient(), ledger)
    program = Program(id="p", code=source, timestamp=0.0)

    offspring = operator(program, (), random.Random(123))

    assert not offspring.valid
    assert offspring.metadata["cost_uncertain"] is True
    assert ledger.spent_microusd == track.max_call_reservation_microusd


def test_settled_empty_completion_charges_known_cost_not_full_reservation():
    from experiments.matched_transport_20260909.client import SettledEmptyCompletion

    class EmptyClient:
        def complete(self, **kwargs):
            raise SettledEmptyCompletion(
                {"cost_microusd": 1234, "finish_reason": "length", "prompt_tokens": 1,
                 "completion_tokens": 2, "reasoning_tokens": 2}
            )

    source = (TASK / "initial.py").read_text()
    track = load_model_track(MODEL)
    ledger = ApiCostLedger(track.max_api_costs_microusd)
    operator = OpenWeightLLMOperator(track, EmptyClient(), ledger)
    program = Program(id="p", code=source, timestamp=0.0)

    offspring = operator(program, (), random.Random(123))

    assert not offspring.valid
    assert offspring.metadata["cost_uncertain"] is False
    assert offspring.metadata["api_costs_microusd"] == 1234
    assert offspring.metadata["empty_completion"]["finish_reason"] == "length"
    assert ledger.spent_microusd == 1234

def test_unpriced_success_conservatively_consumes_full_cost_reservation():
    source = (TASK / "initial.py").read_text()
    changed = source.replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.3")
    track = load_model_track(MODEL)
    ledger = ApiCostLedger(track.max_api_costs_microusd)
    operator = OpenWeightLLMOperator(
        track,
        UnpricedCompletionClient(changed),
        ledger,
        proposal_modes=("rewrite",),
    )

    offspring = operator(Program(id="p", code=source), (), random.Random(123))

    assert offspring.valid
    assert offspring.metadata["reported_cost_microusd"] is None
    assert offspring.metadata["cost_basis"] == "conservative-reservation"
    assert ledger.spent_microusd == track.max_call_reservation_microusd


def test_proposal_cost_cap_refuses_dispatch_without_crashing():
    source = (TASK / "initial.py").read_text()
    track = load_model_track(MODEL)
    client = FakeCompletionClient(source)
    ledger = ApiCostLedger(track.max_call_reservation_microusd - 1)
    operator = OpenWeightLLMOperator(track, client, ledger)

    offspring = operator(Program(id="p", code=source, timestamp=0.0), (), random.Random(123))

    assert not offspring.valid
    assert offspring.metadata["budget_exhausted"] is True
    assert offspring.metadata["api_costs_microusd"] == 0
    assert ledger.spent_microusd == 0
