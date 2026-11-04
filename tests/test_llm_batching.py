from __future__ import annotations

import dataclasses
import hashlib
import random
from pathlib import Path

import pytest
from shinka.database import Program

from evolution.campaign import CampaignRunner, load_campaign_plan
from evolution.evaluation import (
    PolicyEvaluation,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import load_feedback_profiles
from evolution.islands import ArchiveAssessment, DecisionCost, load_islands_config
from evolution.ladder import LadderResult, load_ladder_config
from evolution.llm_operator import (
    LLMCompletion,
    OpenWeightLLMOperator,
    parse_batched_proposals,
)
from evolution.model import ApiCostLedger, load_model_track
from evolution.profiles import LEAN_DEEP_PROFILE, MATCHED_PROFILE, evolution_profile
from experiments.allbaselines_harness import support
from experiments.allbaselines_harness.matching import (
    ADAPTIVE_ARMS,
    EVALUATION_ONLY_ARMS,
    validate_matching,
)
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]
INITIAL = ROOT / "evolution/task/initial.py"
TASK = ROOT / "evolution/task/config.json"
CAMPAIGNS = ROOT / "evolution/campaigns.json"
MODEL = ROOT / "evolution/open_model.json"


def _source(threshold: str) -> str:
    return INITIAL.read_text().replace("hi_u - lo_u < 0.25", f"hi_u - lo_u < {threshold}")


def _block(index: int, source: str, *, close: bool = True) -> str:
    ending = f"\n<<<EVOLVED_CONTINUUM_PROPOSAL:{index}:END>>>" if close else ""
    return (
        f"<<<EVOLVED_CONTINUUM_PROPOSAL:{index}:BEGIN>>>\n"
        f"```python\n{source}```{ending}"
    )


class BatchClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete(self, *, model, prompt, generation, seed):
        self.requests.append(
            {"model": model, "prompt": prompt, "generation": generation, "seed": seed}
        )
        content = self.responses.pop(0)
        return LLMCompletion(
            content=content,
            cost_microusd=1_000,
            prompt_tokens=120,
            completion_tokens=30,
            raw_response={"content": content, "seed": seed},
        )


def _operator(
    responses: list[str],
    *,
    proposals_per_call: int,
    profile: str = LEAN_DEEP_PROFILE,
) -> tuple[OpenWeightLLMOperator, BatchClient, ApiCostLedger]:
    track = load_model_track(MODEL)
    client = BatchClient(responses)
    ledger = ApiCostLedger(track.max_api_costs_microusd)
    return (
        OpenWeightLLMOperator(
            track,
            client,
            ledger,
            proposal_modes=("rewrite",),
            profile=evolution_profile(profile),
            proposals_per_call=proposals_per_call,
        ),
        client,
        ledger,
    )


def _run_batch(operator: OpenWeightLLMOperator, size: int = 3):
    return operator.propose_batch(
        Program(id="parent", code=INITIAL.read_text()),
        (),
        random.Random(202_608_25),
        proposals_per_call=size,
        shared_call_id="test:llm:call:00000000",
        candidate_attempt_seeds=tuple(range(700, 700 + size)),
    )


def test_batch_parser_accepts_k_valid_candidates_and_requests_structural_diversity() -> None:
    response = "\n".join(
        _block(index, _source(threshold))
        for index, threshold in enumerate(("0.30", "0.31", "0.32"), start=1)
    )
    operator, client, _ledger = _operator([response], proposals_per_call=3)

    offspring = _run_batch(operator)

    assert all(item.valid for item in offspring)
    assert len(client.requests) == 1
    prompt = str(client.requests[0]["prompt"])
    assert "BATCH DELIMITER CONTRACT" in prompt
    assert "genuinely different structural idea" in prompt
    assert "<<<EVOLVED_CONTINUUM_PROPOSAL:N:BEGIN>>>" in prompt


@pytest.mark.parametrize(
    ("response", "valid_slots", "error_fragment"),
    (
        (
            "\n".join((_block(1, _source("0.30")), _block(2, _source("0.31")))),
            (True, True, False),
            "fewer than",
        ),
        (
            "\n".join(
                (
                    _block(1, _source("0.30")),
                    _block(2, _source("0.31"), close=False),
                    _block(3, _source("0.32")),
                )
            ),
            (True, False, True),
            "unmatched delimiters",
        ),
        (
            "\n".join(
                (
                    _block(1, _source("0.30")),
                    _block(2, _source("0.30")),
                    _block(3, _source("0.32")),
                )
            ),
            (True, False, True),
            "duplicates batch slot 1",
        ),
    ),
    ids=("fewer-than-k", "malformed-subset", "duplicate-subset"),
)
def test_batch_parser_rejects_only_bad_slots(
    response: str,
    valid_slots: tuple[bool, ...],
    error_fragment: str,
) -> None:
    operator, client, _ledger = _operator([response], proposals_per_call=3)

    offspring = _run_batch(operator)

    assert tuple(item.valid for item in offspring) == valid_slots
    assert error_fragment in next(item.error for item in offspring if not item.valid)
    assert len(client.requests) == 1


def test_batch_validation_rejects_one_malformed_program_without_losing_siblings() -> None:
    response = "\n".join(
        (
            _block(1, _source("0.30")),
            _block(2, "this is not a PlacementScript policy"),
            _block(3, _source("0.32")),
        )
    )
    operator, client, _ledger = _operator([response], proposals_per_call=3)

    offspring = _run_batch(operator)

    assert tuple(item.valid for item in offspring) == (True, False, True)
    assert offspring[1].metadata["validation_failure"]["error"] == offspring[1].error
    assert len(client.requests) == 1


def test_batch_parser_requires_k_greater_than_one() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        parse_batched_proposals("", 1)


def test_batch_attempt_seeds_and_call_accounting_are_exactly_once() -> None:
    response = "\n".join(
        _block(index, _source(threshold))
        for index, threshold in enumerate(("0.30", "0.31", "0.32"), start=1)
    )
    operator, _client, ledger = _operator([response], proposals_per_call=3)

    offspring = _run_batch(operator)
    metadata = [item.metadata for item in offspring]

    assert [item["candidate_attempt_seed"] for item in metadata] == [700, 701, 702]
    assert len({item["candidate_attempt_seed"] for item in metadata}) == 3
    assert {item["shared_call_id"] for item in metadata} == {
        "test:llm:call:00000000"
    }
    assert sum(item["prompt_tokens"] for item in metadata) == 120
    assert sum(item["completion_tokens"] for item in metadata) == 30
    assert sum(item["api_costs_microusd"] for item in metadata) == 1_000
    assert sum("raw_response" in item for item in metadata) == 1
    assert sum("prompt" in item for item in metadata) == 1
    assert ledger.spent_microusd == 1_000


def _evaluation(
    source,
    task,
    ladder,
    proposal_seq,
    *,
    behavioral_novelty,
    known_l1_signatures,
):
    del task, ladder, behavioral_novelty
    identity = policy_identity_for_source(source)
    signature = (
        "a" * 64
        if "< 0.30" in source or "< 0.31" in source
        else "b" * 64
        if "< 0.32" in source
        else "c" * 64
    )
    fitness = CandidateFitness(
        feasible=True,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=1.0 + proposal_seq / 100.0,
        proposal_seq=proposal_seq,
        n_instances=1,
        n_bundles=1,
    )
    evaluation = PolicyEvaluation(
        fitness=fitness,
        instances=(),
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )
    inert = signature in known_l1_signatures
    trace = (
        {"rung": "L0", "decision": "promote"},
        {
            "rung": "L1",
            "decision": "promote",
            "behavior_signature": {"sha256": signature},
        },
        {
            "rung": "behavioral-novelty",
            "decision": "inert-not-promoted" if inert else "novel-promote",
            "signature": {"sha256": signature},
        },
        *(({"rung": "L2", "decision": "fitness-of-record"},) if not inert else ()),
    )
    return LadderResult(
        promoted_to_l2=not inert,
        evaluation=None if inert else evaluation,
        trace=trace,
        budget_registration_status="development-provisional",
        budget_registration_source="batch-test",
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


_evaluation.supports_behavioral_novelty = True


def _plain_evaluation(source, task, ladder, proposal_seq):
    return _evaluation(
        source,
        task,
        ladder,
        proposal_seq,
        behavioral_novelty=None,
        known_l1_signatures=set(),
    )


def _assess(source, task, islands):
    del task, islands
    digest = hashlib.sha256(source.encode()).hexdigest()
    return ArchiveAssessment(
        loc=1,
        decision_cost=DecisionCost(
            source_sha256=digest,
            probe_sha256="d" * 64,
            score_batch_fuel=1,
            score_batch_memory_bytes=0,
            calibrated_latency_us=1.0,
            calibration_id="batch-test",
            calibration_status="development-provisional",
        ),
        loc_penalty=0.0,
        decision_latency_penalty=0.0,
        total_penalty=0.0,
    )


def _runner(
    path: Path,
    operator: OpenWeightLLMOperator,
    *,
    proposals_per_call: int,
    profile_name: str = LEAN_DEEP_PROFILE,
):
    profile = evolution_profile(profile_name)
    plan = dataclasses.replace(
        load_campaign_plan(CAMPAIGNS),
        candidate_budget=4,
        shared_numeric_probability=profile.shared_numeric_probability,
        feedback_profile=profile.feedback_profile,
        evolution_profile=profile.name,
        proposals_per_call=proposals_per_call,
    )
    return CampaignRunner(
        definition=plan.campaigns[0],
        plan=plan,
        arm="llm",
        operator=operator,
        initial_source=INITIAL.read_text(),
        task=load_task_evaluation_config(TASK),
        ladder=load_ladder_config(TASK),
        islands=load_islands_config(TASK),
        feedback_profile=load_feedback_profiles(TASK)[profile.feedback_profile],
        archive_dir=path,
        provenance={"test": "batch-determinism"},
        run_purpose="determinism-smoke",
        evaluate_fn=_evaluation if profile.behavioral_novelty_enabled else _plain_evaluation,
        assess_fn=_assess,
    )


def test_batched_campaign_is_deterministic_and_novelty_rejects_behavioral_duplicate(
    tmp_path,
) -> None:
    response = "\n".join(
        _block(index, _source(threshold))
        for index, threshold in enumerate(("0.30", "0.31", "0.32"), start=1)
    )
    left_operator, left_client, _ledger = _operator([response], proposals_per_call=3)
    right_operator, right_client, _ledger = _operator([response], proposals_per_call=3)
    left = _runner(tmp_path / "left", left_operator, proposals_per_call=3)
    right = _runner(tmp_path / "right", right_operator, proposals_per_call=3)

    assert left.run() == right.run()
    assert left.archive.manifest_path.read_bytes() == right.archive.manifest_path.read_bytes()
    assert left.archive.events_path.read_bytes() == right.archive.events_path.read_bytes()
    assert len(left_client.requests) == len(right_client.requests) == 1
    completed = left._completed()
    assert completed[2]["evaluation"]["promoted_to_l2"] is False
    assert completed[2]["evaluation"]["trace"][-1]["decision"] == "inert-not-promoted"
    call_ids = {
        completed[seq]["shared_call_id"] for seq in (1, 2, 3)
    }
    assert call_ids == {"c00:llm:call:00000000"}
    attempts = [completed[seq]["proposal"]["attempts"][0] for seq in (1, 2, 3)]
    assert len({attempt["arm_primary_seed"] for attempt in attempts}) == 3
    assert sum(
        attempt["metadata"].get("prompt_tokens", 0) for attempt in attempts
    ) == 120


def test_batched_campaign_resume_replays_archived_call_without_redispatch(
    tmp_path,
) -> None:
    response = "\n".join(
        _block(index, _source(threshold))
        for index, threshold in enumerate(("0.30", "0.31", "0.32"), start=1)
    )
    clean_operator, clean_client, _ledger = _operator([response], proposals_per_call=3)
    clean = _runner(tmp_path / "clean", clean_operator, proposals_per_call=3)
    clean.run()

    interrupted_operator, interrupted_client, _ledger = _operator(
        [response], proposals_per_call=3
    )
    interrupted = _runner(
        tmp_path / "interrupted", interrupted_operator, proposals_per_call=3
    )
    original_complete = interrupted._complete

    def stop_after_owner(created, *, proposal=None):
        result = original_complete(created, proposal=proposal)
        if created["proposal_seq"] == 1:
            raise RuntimeError("simulated crash after batch owner archive")
        return result

    interrupted._complete = stop_after_owner
    with pytest.raises(RuntimeError, match="simulated crash"):
        interrupted.run()
    assert len(interrupted_client.requests) == 1

    resumed_operator, resumed_client, _ledger = _operator([], proposals_per_call=3)
    resumed = _runner(
        tmp_path / "interrupted", resumed_operator, proposals_per_call=3
    )
    resumed.run()

    assert len(resumed_client.requests) == 0
    assert len(clean_client.requests) == 1
    assert clean.archive.events_path.read_bytes() == resumed.archive.events_path.read_bytes()


def test_explicit_k1_campaign_archive_is_byte_identical_to_default_k1(tmp_path) -> None:
    responses = [f"```python\n{_source('0.30')}```"] * 3
    implicit_operator, _client, _ledger = _operator(
        list(responses), proposals_per_call=1, profile=MATCHED_PROFILE
    )
    explicit_operator, _client, _ledger = _operator(
        list(responses), proposals_per_call=1, profile=MATCHED_PROFILE
    )

    implicit = _runner(
        tmp_path / "implicit",
        implicit_operator,
        proposals_per_call=1,
        profile_name=MATCHED_PROFILE,
    )
    explicit = _runner(
        tmp_path / "explicit",
        explicit_operator,
        proposals_per_call=1,
        profile_name=MATCHED_PROFILE,
    )

    assert implicit.run() == explicit.run()
    assert (
        implicit.archive.manifest_path.read_bytes()
        == explicit.archive.manifest_path.read_bytes()
    )
    assert implicit.archive.events_path.read_bytes() == explicit.archive.events_path.read_bytes()
    assert b"proposals_per_call" not in implicit.archive.manifest_path.read_bytes()


def test_lean_deep_contains_only_supported_controls() -> None:
    profile = evolution_profile(LEAN_DEEP_PROFILE)

    assert profile.development_only
    assert profile.behavioral_novelty_enabled
    assert profile.shared_numeric_probability == 0.0
    assert not profile.numeric_interleave_enabled
    assert not profile.grammar_aware_retry_enabled
    assert not profile.prompt_exemplars_enabled
    assert not profile.structural_rewrite_bias_enabled
    assert profile.proposal_mode_weights is None
    assert not profile.diverse_inspirations_enabled
    assert not profile.simulator_feedback_enabled
    assert not profile.seeded_initial_archive_enabled
    assert profile.seeded_random_mutants == 0


@pytest.mark.parametrize(
    ("budget", "microwork", "call_ceiling"),
    (("294.0", 294_000_000, 750), ("588.0", 588_000_000, 1_500)),
)
def test_deep_budget_inputs_scale_guards_and_preserve_work_matching(
    tmp_path,
    monkeypatch,
    budget: str,
    microwork: int,
    call_ceiling: int,
) -> None:
    monkeypatch.setattr(support, "RUNS", tmp_path)
    prepared = support.prepare_run(
        "dev-medium-cluster",
        ROOT / "evolution/task/config_dev_medium.json",
        budget,
        800_000_011,
        include_llm=True,
        profile_name=LEAN_DEEP_PROFILE,
        proposals_per_call=3,
    )
    plan = load_campaign_plan(prepared.campaign_config)

    assert prepared.max_microwork == microwork
    assert plan.evaluator_work.max_microwork == microwork
    assert plan.candidate_budget == microwork // 100_000 + 2
    assert plan.proposals_per_call == 3
    assert support.required_call_ceiling(prepared.harness_config, microwork) == call_ceiling
    assert support.enforce_run_size(prepared.run_dir, prepared.harness_config) > 0

    accounting = {
        arm: {
            "mode": "adaptive-normalized-work",
            "registered_max_microwork": microwork,
            "actual_microwork": microwork - 100_000,
            "effective_terminal_microwork": microwork,
        }
        for arm in ADAPTIVE_ARMS
    }
    accounting.update(
        {
            arm: {
                "mode": "frozen-evaluation-only",
                "registered_max_microwork": None,
                "actual_microwork": 0,
                "effective_terminal_microwork": None,
            }
            for arm in EVALUATION_ONLY_ARMS
        }
    )
    receipt = validate_matching(
        accounting,
        matched_max_microwork=microwork,
        include_llm=True,
    )
    assert receipt["verified"] is True
