from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import pytest
from shinka.database import Program

from evolution.campaign import CampaignRunner, load_campaign_plan, verify_confirmatory_authorization
from evolution.codex_engine import (
    CODEX_ENGINE_NAME,
    FAILURE_CONTENT_FILTER,
    FAILURE_EMPTY_OUTPUT,
    FAILURE_NONZERO_EXIT,
    FAILURE_STARTUP_STALL,
    FAILURE_TIMEOUT,
    FAILURE_UNPARSEABLE_OUTPUT,
    TERMINAL_CALL_CEILING,
    TERMINAL_CIRCUIT_BREAKER,
    TERMINAL_WALL_CLOCK_CEILING,
    CodexCLICompletionClient,
    CodexCLICompletionError,
    CodexUsageLedger,
    load_codex_engine_config,
    validate_codex_development_profile,
)
from evolution.evaluation import (
    PolicyEvaluation,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import REQUIRED_PROFILES, load_feedback_profiles
from evolution.islands import ArchiveAssessment, DecisionCost, load_islands_config
from evolution.ladder import LadderResult, load_ladder_config
from evolution.llm_operator import OpenWeightLLMOperator
from evolution.model import ApiCostLedger
from evolution.operators import SharedNumericInterleaveOperator
from evolution.run_campaign import (
    _verify_canonical_tagged_inputs,
    _write_confirmatory_receipt,
)
from experiments.allbaselines_harness.preflight import (
    ENGINE_PROBE_MARKER,
    run_engine_path_probe,
)
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]
TASK_CONFIG = ROOT / "evolution/task/config.json"
CAMPAIGN_CONFIG = ROOT / "evolution/campaigns.json"
CODEX_CONFIG = ROOT / "evolution/codex_engine.json"
INITIAL = ROOT / "evolution/task/initial.py"
STUB = ROOT / "tests/fixtures/codex_stub.py"


@pytest.fixture
def stub_codex(tmp_path, monkeypatch):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    executable = binary_dir / "codex"
    shutil.copy2(STUB, executable)
    executable.chmod(0o755)
    completion = tmp_path / "completion.txt"
    completion.write_text("canned completion")
    state = tmp_path / "state.txt"
    log = tmp_path / "invocations.jsonl"
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text("{}")
    monkeypatch.setenv("PATH", f"{binary_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CODEX_STUB_COMPLETION_FILE", str(completion))
    monkeypatch.setenv("CODEX_STUB_STATE", str(state))
    monkeypatch.setenv("CODEX_STUB_LOG", str(log))
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "success")
    return {"completion": completion, "state": state, "log": log}


def _config(**changes):
    return dataclasses.replace(load_codex_engine_config(CODEX_CONFIG), **changes)


def _complete(client: CodexCLICompletionClient, *, seed: int = 101):
    return client.complete(
        model=client.config.model_id,
        prompt="rendered proposal prompt",
        generation=client.config.generation,
        seed=seed,
    )


def test_success_uses_isolated_read_only_json_cli_and_accounts_tokens(stub_codex):
    client = CodexCLICompletionClient(_config())

    completion = _complete(client)

    assert completion.content == "canned completion"
    assert completion.prompt_tokens == 111
    assert completion.completion_tokens == 222
    raw = completion.raw_response
    attempt = raw["attempts"][0]
    assert raw["session_id"] == "stub-session-0"
    assert raw["engine_descriptor"] == client.engine_descriptor
    assert client.engine_descriptor == {
        "engine": "codex-cli-subscription",
        "model_id": "gpt-5.6-sol",
        "cli_version": "0.144.99",
        "a1_reference": (
            "ContinuumPlacementScopedDown.md#researcher-amendment-a1-2026-08-09"
        ),
        "development_only": True,
        "live_deterministic": False,
        "replay_deterministic": True,
        "sampling_control": "subscription-service-managed",
    }
    invocation = json.loads(stub_codex["log"].read_text())
    assert invocation["prompt"] == "rendered proposal prompt"
    assert invocation["cwd_entries_before"] == []
    assert Path(invocation["cwd"]) != ROOT
    assert invocation["argv"][invocation["argv"].index("--sandbox") + 1] == "read-only"
    assert invocation["argv"][invocation["argv"].index("-m") + 1] == "gpt-5.6-sol"
    assert "--json" in invocation["argv"]
    assert attempt["raw_output"]["stdout"]
    assert attempt["process"]["exit_code"] == 0
    assert raw["accounting"]["calls"] == 1
    assert raw["accounting"]["total_tokens"] == 333


def test_timeout_is_classified_and_charged_to_wall_clock(stub_codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "timeout")
    monkeypatch.setenv("CODEX_STUB_SLEEP_S", "1")
    client = CodexCLICompletionClient(_config(timeout_s=0.05))

    with pytest.raises(CodexCLICompletionError) as caught:
        _complete(client)

    assert caught.value.category == FAILURE_TIMEOUT
    attempt = caught.value.archive_metadata["raw_response"]["attempts"][0]
    assert attempt["process"]["exit_code"] is None
    assert attempt["process"]["duration_ms"] >= 1
    assert client.usage_ledger.calls == 1


def test_nonzero_exit_is_classified_and_raw_stderr_is_retained(stub_codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "nonzero")
    client = CodexCLICompletionClient(_config())

    with pytest.raises(CodexCLICompletionError) as caught:
        _complete(client)

    assert caught.value.category == FAILURE_NONZERO_EXIT
    attempt = caught.value.archive_metadata["raw_response"]["attempts"][0]
    assert attempt["process"]["exit_code"] == 17
    assert "stub nonzero failure" in attempt["raw_output"]["stderr"]
    assert "raw excerpt" in str(caught.value)
    assert "stub nonzero failure" in str(caught.value)


@pytest.mark.parametrize(
    ("mode", "category"),
    [("empty", FAILURE_EMPTY_OUTPUT), ("malformed", FAILURE_UNPARSEABLE_OUTPUT)],
)
def test_empty_and_malformed_output_are_distinct(stub_codex, monkeypatch, mode, category):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", mode)
    client = CodexCLICompletionClient(_config())

    with pytest.raises(CodexCLICompletionError) as caught:
        _complete(client)

    assert caught.value.category == category


def test_never_outputting_process_fails_with_bounded_actionable_startup_error(
    stub_codex, monkeypatch
):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "timeout")
    monkeypatch.setenv("CODEX_STUB_SLEEP_S", "2")
    client = CodexCLICompletionClient(
        _config(timeout_s=1.0, startup_grace_s=0.05)
    )

    started = time.monotonic()
    with pytest.raises(CodexCLICompletionError) as caught:
        _complete(client)

    assert time.monotonic() - started < 0.75
    assert caught.value.category == FAILURE_STARTUP_STALL
    assert "no recognized JSONL event within 0.05s" in str(caught.value)
    assert "saw event types []" in str(caught.value)
    assert "stdout=''" in str(caught.value)
    attempt = caught.value.archive_metadata["raw_response"]["attempts"][0]
    assert attempt["process"]["exit_code"] is None
    assert attempt["accounting_delta"]["duration_ms"] < 750


def test_renamed_event_types_fail_fast_and_report_seen_types(stub_codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "renamed-events")
    monkeypatch.setenv("CODEX_STUB_SLEEP_S", "2")
    client = CodexCLICompletionClient(
        _config(timeout_s=1.0, startup_grace_s=0.05)
    )

    with pytest.raises(CodexCLICompletionError) as caught:
        _complete(client)

    assert caught.value.category == FAILURE_STARTUP_STALL
    assert "response.begun" in str(caught.value)
    assert "thread.opened" in str(caught.value)
    assert "thread.started" in str(caught.value)
    assert len(str(caught.value).encode()) < 5_000


def test_valid_events_accept_forward_compatible_extra_fields(stub_codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "extra-fields")
    client = CodexCLICompletionClient(_config())

    completion = _complete(client)

    assert completion.content == "canned completion"
    assert all(
        event.get("unexpected_extra_field") == {"future": True}
        for event in completion.raw_response["attempts"][0]["events"]
    )


def test_engine_path_preflight_requires_a_parsed_exact_marker(
    stub_codex, monkeypatch
):
    stub_codex["completion"].write_text(ENGINE_PROBE_MARKER)
    config = _config(timeout_s=1.0, startup_grace_s=0.05)
    client = CodexCLICompletionClient(config)

    probe = run_engine_path_probe(client, config)

    assert probe["status"] == "PARSED_COMPLETION"
    assert probe["marker"] == ENGINE_PROBE_MARKER
    assert probe["session_id"] == "stub-session-0"

    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "renamed-events")
    monkeypatch.setenv("CODEX_STUB_SLEEP_S", "2")
    failing_client = CodexCLICompletionClient(config)
    with pytest.raises(CodexCLICompletionError, match="no recognized JSONL event"):
        run_engine_path_probe(failing_client, config)


def test_content_filter_refusal_is_not_returned_as_a_completion(stub_codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "refusal")
    client = CodexCLICompletionClient(_config())

    with pytest.raises(CodexCLICompletionError) as caught:
        _complete(client)

    assert caught.value.category == FAILURE_CONTENT_FILTER


def _fake_evaluate(source, task, ladder, proposal_seq):
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


def _fake_assess(source, task, islands):
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


def _runner(tmp_path, config, *, max_operator_attempts=2):
    plan = dataclasses.replace(
        load_campaign_plan(CAMPAIGN_CONFIG),
        candidate_budget=2,
        max_operator_attempts=max_operator_attempts,
        shared_numeric_probability=0.0,
    )
    task = load_task_evaluation_config(TASK_CONFIG)
    ladder = load_ladder_config(TASK_CONFIG)
    islands = load_islands_config(TASK_CONFIG)
    feedback = load_feedback_profiles(TASK_CONFIG)[plan.feedback_profile]
    client = CodexCLICompletionClient(config)
    primary = OpenWeightLLMOperator(
        config,
        client,
        ApiCostLedger(0),
        proposal_modes=("rewrite",),
    )
    runner = CampaignRunner(
        definition=plan.campaigns[0],
        plan=plan,
        arm="llm",
        operator=SharedNumericInterleaveOperator(primary, probability=0.0),
        initial_source=INITIAL.read_text(),
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        archive_dir=tmp_path / "archive",
        provenance={"test": "codex-stub"},
        run_purpose="determinism-smoke",
        evaluate_fn=_fake_evaluate,
        assess_fn=_fake_assess,
    )
    return runner, client


def test_runner_retries_with_seeded_attempts_and_archives_full_cli_evidence(
    tmp_path, stub_codex, monkeypatch
):
    changed = INITIAL.read_text().replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.3")
    stub_codex["completion"].write_text(f"```python\n{changed}```")
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "nonzero,success")
    runner, client = _runner(tmp_path, _config(max_calls_per_campaign=2))

    digest = runner.run()

    completed = runner._completed()[1]
    attempts = completed["proposal"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["metadata"]["failure_category"] == FAILURE_NONZERO_EXIT
    assert attempts[1]["valid"] is True
    assert attempts[0]["arm_primary_seed"] != attempts[1]["arm_primary_seed"]
    assert all(attempt["metadata"]["prompt"] for attempt in attempts)
    assert all(attempt["metadata"]["raw_response"]["attempts"] for attempt in attempts)
    assert completed["proposal_engine"] == client.engine_descriptor
    assert all(
        event["payload"].get("proposal_engine") == client.engine_descriptor
        for event in runner.archive.events
    )
    terminal = runner._terminal()
    assert terminal["proposal_engine_accounting"]["calls"] == 2
    assert terminal["proposal_engine_accounting"]["total_tokens"] == 333

    resumed, resumed_client = _runner(tmp_path, _config(max_calls_per_campaign=2))
    assert resumed.run() == digest
    assert resumed_client.usage_ledger.calls == 2
    assert resumed_client.usage_ledger.to_dict()["total_tokens"] == 333


def test_call_ceiling_archives_terminal_reason_and_raises(tmp_path, stub_codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "nonzero")
    runner, _client = _runner(tmp_path, _config(max_calls_per_campaign=1))

    with pytest.raises(RuntimeError, match=TERMINAL_CALL_CEILING):
        runner.run()

    terminal = runner._terminal()
    assert terminal["status"] == "unusable"
    assert terminal["reason"] == TERMINAL_CALL_CEILING
    assert terminal["proposal_engine_accounting"]["calls"] == 1
    assert len(runner._completed()[1]["proposal"]["attempts"]) == 2


def test_usage_journal_conservatively_charges_interrupted_dispatch_on_resume(tmp_path):
    journal = tmp_path / "proposal-engine-usage.jsonl"
    first = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)
    first.bind_journal(journal)
    first.reserve_dispatch(
        reserved_wall_clock_ms=400,
        request_sha256="a" * 64,
        seed=7,
    )
    first.release_journal_lease()

    resumed = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)
    resumed.bind_journal(journal)

    assert resumed.calls == 1
    assert resumed.wall_clock_ms == 400
    assert resumed.remaining_calls == 1
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "dispatch_reserved",
        "dispatch_abandoned",
    ]
    assert events[-1]["payload"]["duration_ms"] == 400


def test_usage_journal_refuses_overlapping_resume_before_dispatch(tmp_path):
    journal = tmp_path / "proposal-engine-usage.jsonl"
    first = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)
    first.bind_journal(journal)
    first.reserve_dispatch(
        reserved_wall_clock_ms=400,
        request_sha256="a" * 64,
        seed=7,
    )
    overlapping = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)

    with pytest.raises(RuntimeError, match="concurrent resume refused before dispatch"):
        overlapping.bind_journal(journal)

    assert [json.loads(line)["type"] for line in journal.read_text().splitlines()] == [
        "dispatch_reserved"
    ]
    first.release_journal_lease()
    resumed = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)
    resumed.bind_journal(journal)
    assert resumed.calls == 1
    assert resumed.wall_clock_ms == 400
    assert resumed.consecutive_failures == 1
    resumed.release_journal_lease()


def test_consecutive_failures_open_circuit_and_resume_without_another_call(
    tmp_path, stub_codex, monkeypatch
):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "timeout")
    monkeypatch.setenv("CODEX_STUB_SLEEP_S", "2")
    config = _config(
        timeout_s=1.0,
        startup_grace_s=0.05,
        max_calls_per_campaign=10,
        max_consecutive_failures=2,
    )
    runner, client = _runner(tmp_path, config, max_operator_attempts=3)

    with pytest.raises(RuntimeError, match=TERMINAL_CIRCUIT_BREAKER):
        runner.run()

    terminal = runner._terminal()
    assert terminal["status"] == "unusable"
    assert terminal["reason"] == TERMINAL_CIRCUIT_BREAKER
    assert terminal["proposal_engine_accounting"]["calls"] == 2
    assert terminal["proposal_engine_accounting"]["consecutive_failures"] == 2
    attempts = runner._completed()[1]["proposal"]["attempts"]
    assert len(attempts) == 2
    assert all(
        attempt["metadata"]["failure_category"] == FAILURE_STARTUP_STALL
        for attempt in attempts
    )
    assert all(attempt["metadata"]["api_costs_microusd"] == 0 for attempt in attempts)
    attempt_seeds = [attempt["arm_primary_seed"] for attempt in attempts]
    stub_calls = int(stub_codex["state"].read_text())

    resumed, resumed_client = _runner(tmp_path, config, max_operator_attempts=3)
    with pytest.raises(RuntimeError, match=TERMINAL_CIRCUIT_BREAKER):
        resumed.run()

    assert int(stub_codex["state"].read_text()) == stub_calls
    assert resumed_client.usage_ledger.calls == 2
    assert [
        attempt["arm_primary_seed"]
        for attempt in resumed._completed()[1]["proposal"]["attempts"]
    ] == attempt_seeds
    events = [
        json.loads(line)
        for line in (tmp_path / "archive/proposal-engine-usage.jsonl").read_text().splitlines()
    ]
    assert [event["type"] for event in events] == [
        "dispatch_reserved",
        "dispatch_settled",
        "dispatch_reserved",
        "dispatch_settled",
    ]
    assert all(
        event["payload"].get("failure_category") == FAILURE_STARTUP_STALL
        for event in events
        if event["type"] == "dispatch_settled"
    )
    del client


def test_wall_clock_ceiling_archives_terminal_reason_and_raises(
    tmp_path, stub_codex, monkeypatch
):
    monkeypatch.setenv("CODEX_STUB_SEQUENCE", "timeout")
    monkeypatch.setenv("CODEX_STUB_SLEEP_S", "1")
    runner, _client = _runner(
        tmp_path,
        _config(timeout_s=1.0, max_wall_clock_s_per_campaign=0.05),
    )

    with pytest.raises(RuntimeError, match=TERMINAL_WALL_CLOCK_CEILING):
        runner.run()

    terminal = runner._terminal()
    assert terminal["reason"] == TERMINAL_WALL_CLOCK_CEILING
    assert terminal["proposal_engine_accounting"]["calls"] == 1
    assert terminal["proposal_engine_accounting"]["remaining_wall_clock_seconds"] == 0


def test_config_accepts_only_development_matched_and_e5_profiles():
    config = load_codex_engine_config(CODEX_CONFIG)
    for profile in REQUIRED_PROFILES:
        validate_codex_development_profile(
            config,
            development=True,
            campaign_status="development-provisional",
            feedback_profile=profile,
        )
    with pytest.raises(ValueError, match="development override"):
        validate_codex_development_profile(
            config,
            development=False,
            campaign_status="development-provisional",
            feedback_profile="matched-llm-vs-gp",
        )
    with pytest.raises(ValueError, match="development campaign"):
        validate_codex_development_profile(
            config,
            development=True,
            campaign_status="p2-frozen",
            feedback_profile="matched-llm-vs-gp",
        )
    with pytest.raises(ValueError, match="matched/E5"):
        validate_codex_development_profile(
            config,
            development=True,
            campaign_status="development-provisional",
            feedback_profile="confirmatory",
        )


def test_every_confirmatory_authority_boundary_explicitly_refuses_codex(tmp_path):
    descriptor = {"engine": CODEX_ENGINE_NAME}
    plan = load_campaign_plan(CAMPAIGN_CONFIG)
    with pytest.raises(ValueError, match="authorization refuses development-only"):
        verify_confirmatory_authorization(
            plan,
            tmp_path / "missing-authorization.json",
            engine_descriptor=descriptor,
        )
    with pytest.raises(ValueError, match="frozen-tag verification refuses development-only"):
        _verify_canonical_tagged_inputs(
            {}, tag="missing-tag", engine_descriptor=descriptor
        )
    with pytest.raises(ValueError, match="postflight receipt refuses development-only"):
        _write_confirmatory_receipt(
            tmp_path / "receipt.json",
            {"status": "PASS"},
            engine_descriptor=descriptor,
        )
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE") != "1",
    reason="set RUN_CODEX_LIVE=1 to use the researcher's authenticated subscription",
)
def test_live_codex_cli_smoke():
    client = CodexCLICompletionClient(
        _config(
            max_calls_per_campaign=1,
            max_wall_clock_s_per_campaign=600.0,
        )
    )
    completion = client.complete(
        model=client.config.model_id,
        prompt="Return exactly the text CODEX_ENGINE_LIVE_OK and nothing else.",
        generation=client.config.generation,
        seed=20260809,
    )
    assert "CODEX_ENGINE_LIVE_OK" in completion.content
    assert completion.prompt_tokens > 0
    assert completion.completion_tokens > 0


def test_operator_consumes_client_interface_without_a_codex_specific_operator(stub_codex):
    config = _config()
    client = CodexCLICompletionClient(config)
    operator = OpenWeightLLMOperator(
        config,
        client,
        ApiCostLedger(0),
        proposal_modes=("rewrite",),
    )
    source = INITIAL.read_text()
    changed = source.replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.3")
    stub_codex["completion"].write_text(f"```python\n{changed}```")

    offspring = operator(Program(id="parent", code=source), (), random.Random(7))

    assert offspring.valid
    assert operator.name == "codex_cli_subscription"
    assert offspring.metadata["engine_descriptor"] == client.engine_descriptor
    assert offspring.metadata["cost_basis"] == "subscription-no-marginal-call-charge"
    rendered = json.loads(stub_codex["log"].read_text())["prompt"]
    assert "Dictionaries, sets, tuples, None" in rendered
