from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from shinka.core import ProposalOffspring
from shinka.database import Program

import evolution.ladder as ladder_module
from evolution.campaign import CampaignRunner, EvaluatorWorkBudget, load_campaign_plan
from evolution.codex_engine import load_codex_engine_config
from evolution.evaluation import (
    InstanceEvidence,
    PolicyEvaluation,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import (
    ENHANCED_LLM_DEV_PROFILE,
    FeedbackProfile,
    generate_feedback,
    load_feedback_profiles,
)
from evolution.islands import ArchiveAssessment, DecisionCost, load_islands_config
from evolution.ladder import (
    BehavioralNoveltyConfig,
    evaluate_adapter_through_ladder,
    l1_behavior_signature,
    load_ladder_config,
)
from evolution.llm_operator import (
    LLMCompletion,
    OpenWeightLLMOperator,
    ValidationRepair,
)
from evolution.model import ApiCostLedger, load_model_track
from evolution.profiles import ENHANCED_PROFILE, MATCHED_PROFILE, evolution_profile
from experiments.allbaselines_harness import support
from experiments.allbaselines_harness.initialization import materialize_seed_archive
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]
INITIAL = ROOT / "evolution/task/initial.py"
MODEL = ROOT / "evolution/open_model.json"
TASK = ROOT / "evolution/task/config_dev_medium.json"
CAMPAIGNS = ROOT / "evolution/campaigns.json"


class SequenceCompletionClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete(self, *, model, prompt, generation, seed):
        self.requests.append({"model": model, "prompt": prompt, "seed": seed})
        content = self.responses.pop(0)
        return LLMCompletion(
            content=content,
            cost_microusd=1000,
            prompt_tokens=100,
            completion_tokens=100,
            raw_response={"content": content, "generation": generation, "seed": seed},
        )


def _operator(profile_name: str, responses: list[str]) -> tuple[OpenWeightLLMOperator, object]:
    track = load_model_track(MODEL)
    client = SequenceCompletionClient(responses)
    operator = OpenWeightLLMOperator(
        track,
        client,
        ApiCostLedger(track.max_api_costs_microusd),
        proposal_modes=("rewrite",),
        profile=evolution_profile(profile_name),
    )
    return operator, client


@pytest.mark.parametrize(
    ("old", "invalid_construct"),
    (
        (
            "    return free_fraction_after(pod, node)",
            "    cpu_util(node)\n    return free_fraction_after(pod, node)",
        ),
        (
            "    return (cpu_frac + mem_frac) / 2.0",
            "    return min(cpu_frac, mem_frac, 1.0)",
        ),
        (
            "    return (cpu_frac + mem_frac) / 2.0",
            "    return (cpu_frac, mem_frac)",
        ),
    ),
    ids=("bare-expression", "three-argument-min", "tuple-return"),
)
def test_grammar_retry_reuses_mode_and_appends_exact_local_error_and_excerpt(
    old: str,
    invalid_construct: str,
) -> None:
    parent = INITIAL.read_text()
    invalid = parent.replace(old, invalid_construct)
    valid = parent.replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.30")
    operator, client = _operator(
        ENHANCED_PROFILE,
        [f"```python\n{invalid}```", f"```python\n{valid}```"],
    )
    program = Program(id="parent", code=parent)

    first = operator(program, (), random.Random(101))
    failure = first.metadata["validation_failure"]
    repair = ValidationRepair(**failure)
    second = operator.propose_with_repair(
        program,
        (),
        random.Random(202),
        mode=first.metadata["proposal_mode"],
        repair=repair,
    )

    assert not first.valid and second.valid
    assert len(client.requests) == 2
    retry_prompt = client.requests[1]["prompt"]
    assert f"VALIDATION ERROR (verbatim): {failure['error']}" in retry_prompt
    assert failure["offending_excerpt"] in retry_prompt
    assert "VALID PLACEMENTSCRIPT EXEMPLARS" in retry_prompt
    assert first.metadata["proposal_mode"] == second.metadata["proposal_mode"] == "rewrite"
    assert client.requests[0]["seed"] != client.requests[1]["seed"]
    # Offline deterministic fixture rate: one loadable proposal from two charged calls.
    assert sum((first.valid, second.valid)) / len(client.requests) == 0.5


def test_matched_prompt_and_retry_flags_are_off() -> None:
    parent = INITIAL.read_text()
    operator, _client = _operator(MATCHED_PROFILE, [])
    prompt = operator._prompt(
        Program(id="p", code=parent),
        (),
        "rewrite",
        profile=evolution_profile(MATCHED_PROFILE),
    )
    repair = ValidationRepair("parse error (line 1): x", "1: broken")

    assert "VALID PLACEMENTSCRIPT EXEMPLARS" not in prompt
    assert "substantial structural rewrite" not in prompt
    with pytest.raises(ValueError, match="disabled"):
        operator.propose_with_repair(
            Program(id="p", code=parent),
            (),
            random.Random(1),
            mode="rewrite",
            repair=repair,
        )


@pytest.mark.parametrize("profile_name", (MATCHED_PROFILE, ENHANCED_PROFILE))
def test_mode_selection_is_seed_deterministic_in_both_flag_states(profile_name: str) -> None:
    track = load_model_track(MODEL)
    operator = OpenWeightLLMOperator(
        track,
        SequenceCompletionClient([]),
        ApiCostLedger(track.max_api_costs_microusd),
        profile=evolution_profile(profile_name),
    )

    def draws() -> list[str]:
        rng = random.Random(987654321)
        return [operator._select_mode(("diff", "rewrite", "crossover"), rng) for _ in range(200)]

    first = draws()
    assert first == draws()
    if profile_name == ENHANCED_PROFILE:
        assert first.count("rewrite") > first.count("diff")


def _fake_policy_evaluation(proposal_seq: int = 0) -> PolicyEvaluation:
    fitness = CandidateFitness(
        feasible=True,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=1.1,
        proposal_seq=proposal_seq,
        n_instances=1,
        n_bundles=1,
    )
    instance = InstanceEvidence(
        scenario_id="dev-test",
        seed=123,
        bundle_id="dev-1",
        episode_digest="1" * 64,
        certificate_seal="2" * 64,
        energy_j=110.0,
        idle_floor_j=10.0,
        ceiling_j=210.0,
        slo_achieved={"app": 1.0},
        goodput=1.0,
        drop_rate=0.0,
        timeout_rate=0.0,
        disruption={"moves": 0.0},
        boundary_certificate={},
        verified_energy={},
        episode={
            "decision_log": [{"t": 1.0, "event": "reserve", "node": 2}],
            "rebalancer": {"moves": [], "rejections": []},
            "energy": {"device_j": 70.0, "cloud_j": 30.0, "network_j": 10.0},
        },
    )
    return PolicyEvaluation(
        fitness=fitness,
        instances=(instance,),
        policy_name="test",
        policy_version="1",
        policy_identity_sha256="a" * 64,
        policy_artifact_sha256="b" * 64,
    )


def test_behavioral_novelty_skips_l2_deterministically_and_charges_l1(monkeypatch) -> None:
    evaluation = _fake_policy_evaluation()
    l2_calls = 0

    monkeypatch.setattr(
        ladder_module,
        "_run_l0_adapter",
        lambda scorer, task: {
            "valid": True,
            "feasible": True,
            "mean_normalized_energy": 1.0,
            "instances": [],
        },
    )
    monkeypatch.setattr(
        ladder_module,
        "_run_l1_adapter",
        lambda *args, **kwargs: evaluation,
    )

    def l2(*args, **kwargs):
        nonlocal l2_calls
        l2_calls += 1
        return evaluation

    monkeypatch.setattr(ladder_module, "evaluate_policy_adapter", l2)
    task = load_task_evaluation_config(TASK)
    ladder = load_ladder_config(TASK)
    config = BehavioralNoveltyConfig(tolerance=1e-9, signature_seed=444)
    kwargs = {
        "proposal_seq": 0,
        "policy_name": "test",
        "policy_version": "1",
        "policy_identity_sha256": "a" * 64,
        "policy_artifact_sha256": "b" * 64,
        "behavioral_novelty": config,
    }

    novel = evaluate_adapter_through_ladder(object(), object(), task, ladder, **kwargs)
    signature = next(
        record["behavior_signature"]["sha256"]
        for record in novel.trace
        if "behavior_signature" in record
    )
    inert = evaluate_adapter_through_ladder(
        object(),
        object(),
        task,
        ladder,
        **kwargs,
        known_l1_signatures=frozenset({signature}),
    )
    ungated = evaluate_adapter_through_ladder(
        object(),
        object(),
        task,
        ladder,
        **{key: value for key, value in kwargs.items() if key != "behavioral_novelty"},
        known_l1_signatures=frozenset({signature}),
    )

    assert l2_calls == 2  # novel + explicitly ungated; inert never reaches L2.
    assert not inert.promoted_to_l2 and ungated.promoted_to_l2
    assert any(record.get("decision") == "inert-not-promoted" for record in inert.trace)
    work = EvaluatorWorkBudget(
        "development-provisional",
        "test",
        1_400_000,
        {"L0": 100_000, "L1": 300_000, "L2": 1_000_000},
        (0, 1_400_000),
    )
    assert work.work_for_trace(inert.trace) == 400_000
    assert l1_behavior_signature(evaluation, config) == l1_behavior_signature(
        evaluation, config
    )


def test_enhanced_feedback_contains_constraints_energy_and_recent_rejections() -> None:
    task = load_task_evaluation_config(TASK)
    profile = FeedbackProfile(ENHANCED_LLM_DEV_PROFILE, True)
    evaluation = _fake_policy_evaluation()
    result = SimpleNamespace(
        promoted_to_l2=True,
        trace=(
            {
                "rung": "L2",
                "decision": "fitness-of-record",
                "result": {
                    "feasible": True,
                    "normalized_energy": 1.1,
                    "infeasibility_vector": [0.0, 0.0, 0.0, 0.0],
                    "evaluation": evaluation.to_dict(),
                },
            },
        ),
    )

    feedback = generate_feedback(
        result,
        profile,
        constraints=task.constraints,
        recent_rejections=("parse error (line 7): expected expression",),
    )
    categories = [signal.category for signal in feedback.signals]

    assert categories[:4] == [
        "binding_constraints",
        "energy_device",
        "energy_cloud",
        "energy_network",
    ]
    assert categories[-1] == "recent_rejected_attempt"


def test_profiles_control_numeric_and_initialization_without_mutating_matched() -> None:
    matched = evolution_profile(MATCHED_PROFILE)
    enhanced = evolution_profile(ENHANCED_PROFILE)

    assert matched.numeric_interleave_enabled and matched.shared_numeric_probability == 0.2
    assert not enhanced.numeric_interleave_enabled
    assert enhanced.shared_numeric_probability == 0.0
    assert not matched.seeded_initial_archive_enabled
    assert enhanced.seeded_initial_archive_enabled and enhanced.seeded_random_mutants == 5


def test_same_seed_materializes_identical_enhanced_archives_and_matched_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    common = (
        "dev-medium-cluster",
        TASK,
        "58.8",
        800_000_011,
    )
    monkeypatch.setattr(support, "RUNS", tmp_path / "enhanced-a")
    left = support.prepare_run(
        *common,
        include_llm=False,
        profile_name=ENHANCED_PROFILE,
    )
    monkeypatch.setattr(support, "RUNS", tmp_path / "enhanced-b")
    right = support.prepare_run(
        *common,
        include_llm=False,
        profile_name=ENHANCED_PROFILE,
    )
    left_seed = materialize_seed_archive(left)
    right_seed = materialize_seed_archive(right)

    assert left_seed is not None and right_seed is not None
    assert hashlib.sha256(left_seed.read_bytes()).hexdigest() == hashlib.sha256(
        right_seed.read_bytes()
    ).hexdigest()
    assert load_campaign_plan(left.campaign_config).evolution_profile == ENHANCED_PROFILE
    assert load_feedback_profiles(left.task_config)[ENHANCED_PROFILE].enabled
    assert (
        load_codex_engine_config(left.codex_config).profile_policy[ENHANCED_PROFILE]
        == "development-only"
    )
    assert support.required_call_ceiling(left.harness_config, left.max_microwork) == 150

    monkeypatch.setattr(support, "RUNS", tmp_path / "matched-a")
    matched_left = support.prepare_run(
        *common,
        include_llm=False,
    )
    monkeypatch.setattr(support, "RUNS", tmp_path / "matched-b")
    matched_right = support.prepare_run(
        *common,
        include_llm=False,
    )
    assert matched_left.campaign_config.read_bytes() == matched_right.campaign_config.read_bytes()
    assert not matched_left.population_parity


class _CloneOperator:
    name = "enhanced-deterministic-clone"

    def __call__(self, parent, inspirations, rng):
        del inspirations
        return ProposalOffspring(
            source=parent.code,
            metadata={"operator": self.name, "draw": rng.randrange(2**31)},
        )


class _RepairingOperator:
    name = "enhanced-deterministic-repair"

    def __call__(self, parent, inspirations, rng):
        del parent, inspirations, rng
        return ProposalOffspring(
            source=None,
            error="parse error (line 7): expected expression",
            metadata={
                "proposal_mode": "rewrite",
                "validation_failure": {
                    "error": "parse error (line 7): expected expression",
                    "offending_excerpt": "7: broken",
                },
            },
        )

    def propose_with_repair(self, parent, inspirations, rng, *, mode, repair):
        del inspirations, rng
        assert mode == "rewrite"
        assert repair == ValidationRepair(
            "parse error (line 7): expected expression", "7: broken"
        )
        return ProposalOffspring(
            source=parent.code,
            metadata={"proposal_mode": mode, "repaired": True},
        )


def _enhanced_evaluate(
    source,
    task,
    ladder,
    proposal_seq,
    *,
    behavioral_novelty,
    known_l1_signatures,
):
    del task, ladder
    identity = policy_identity_for_source(source)
    base = _fake_policy_evaluation(proposal_seq)
    evaluation = dataclasses.replace(
        base,
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )
    signature = l1_behavior_signature(evaluation, behavioral_novelty)
    result = {
        "feasible": True,
        "normalized_energy": evaluation.fitness.normalized_energy,
        "infeasibility_vector": [0.0, 0.0, 0.0, 0.0],
        "summed_primitive_shortfall": 0.0,
        "evaluation": evaluation.to_dict(),
    }
    inert = signature["sha256"] in known_l1_signatures
    trace = (
        {"rung": "L0", "decision": "promote", "result": {"valid": True}},
        {
            "rung": "L1",
            "decision": "promote",
            "result": result,
            "behavior_signature": signature,
        },
        {
            "rung": "behavioral-novelty",
            "decision": "inert-not-promoted" if inert else "novel-promote",
            "signature": signature,
        },
    )
    if not inert:
        trace = (*trace, {"rung": "L2", "decision": "fitness-of-record", "result": result})
    return ladder_module.LadderResult(
        promoted_to_l2=not inert,
        evaluation=None if inert else evaluation,
        trace=trace,
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


_enhanced_evaluate.supports_behavioral_novelty = True


def _enhanced_assess(source, task, islands):
    del task, islands
    digest = hashlib.sha256(source.encode()).hexdigest()
    return ArchiveAssessment(
        loc=1,
        decision_cost=DecisionCost(
            source_sha256=digest,
            probe_sha256="1" * 64,
            score_batch_fuel=1,
            score_batch_memory_bytes=0,
            calibrated_latency_us=1.0,
            calibration_id="test",
            calibration_status="development-provisional",
        ),
        loc_penalty=0.0,
        decision_latency_penalty=0.0,
        total_penalty=0.0,
    )


def _enhanced_runner(path, operator=None):
    plan = dataclasses.replace(
        load_campaign_plan(CAMPAIGNS),
        candidate_budget=3,
        shared_numeric_probability=0.0,
        feedback_profile=ENHANCED_PROFILE,
        evolution_profile=ENHANCED_PROFILE,
    )
    task = load_task_evaluation_config(ROOT / "evolution/task/config.json")
    return CampaignRunner(
        definition=plan.campaigns[0],
        plan=plan,
        arm="llm",
        operator=_CloneOperator() if operator is None else operator,
        initial_source=INITIAL.read_text(),
        task=task,
        ladder=load_ladder_config(ROOT / "evolution/task/config.json"),
        islands=load_islands_config(ROOT / "evolution/task/config.json"),
        feedback_profile=FeedbackProfile(ENHANCED_PROFILE, True),
        archive_dir=path,
        provenance={"test": "enhanced-profile-determinism"},
        run_purpose="determinism-smoke",
        evaluate_fn=_enhanced_evaluate,
        assess_fn=_enhanced_assess,
    )


def test_same_seed_produces_identical_enhanced_campaign_archive(tmp_path) -> None:
    left = _enhanced_runner(tmp_path / "left")
    right = _enhanced_runner(tmp_path / "right")

    assert left.run() == right.run()
    assert left.archive.events_path.read_bytes() == right.archive.events_path.read_bytes()
    controls = json.loads(left.archive.manifest_path.read_text())["campaign"][
        "evolution_controls"
    ]
    assert not controls["numeric_interleave_enabled"]
    assert all(
        controls[name]
        for name in (
            "grammar_aware_retry_enabled",
            "prompt_exemplars_enabled",
            "behavioral_novelty_enabled",
            "structural_rewrite_bias_enabled",
            "diverse_inspirations_enabled",
            "simulator_feedback_enabled",
            "seeded_initial_archive_enabled",
        )
    )


def test_campaign_archives_every_seeded_grammar_repair_attempt(tmp_path) -> None:
    campaign = _enhanced_runner(tmp_path / "repair", _RepairingOperator())
    campaign.run()
    attempts = campaign._completed()[1]["proposal"]["attempts"]

    assert len(attempts) == 2
    assert attempts[0]["repair_of_attempt"] is None
    assert attempts[1]["repair_of_attempt"] == 0
    assert attempts[0]["error"] == "parse error (line 7): expected expression"
    assert attempts[0]["arm_primary_seed"] != attempts[1]["arm_primary_seed"]
    assert attempts[1]["metadata"]["repaired"] is True
