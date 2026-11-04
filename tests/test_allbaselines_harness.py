from __future__ import annotations

import copy
import gzip
import json
import sys
import time
from types import SimpleNamespace

import pytest

from baselines.gphh import GPHHConfig
from baselines.handwritten import HANDWRITTEN_NAMES
from evolution.campaign import EvaluatorWorkBudget, EventArchive, load_campaign_plan
from evolution.codex_engine import CodexUsageLedger, load_codex_engine_config
from evolution.profiles import (
    DEEP_SEARCH_FEEDBACK_PROFILE,
    DEEP_SEARCH_PROFILE,
    ENHANCED_PROFILE,
    evolution_profile,
)
from experiments.allbaselines_harness import support
from experiments.allbaselines_harness.initialization import validate_parity_charging
from experiments.allbaselines_harness.matching import (
    ADAPTIVE_ARMS,
    EVALUATION_ONLY_ARMS,
    validate_matching,
)
from experiments.allbaselines_harness.preflight import verify_a1_refusals
from experiments.allbaselines_harness.runner import (
    ProposalEngineJournalStalled,
    _adaptive_payload,
    _append_driver,
    _compact_adaptive_run,
    _run_with_size_guard,
)
from experiments.allbaselines_harness.summary import (
    _adaptive_arm,
    _campaign_rows,
    arm_summary,
    gphh_gen0_rows,
    plateau_histogram,
    trajectory,
)
from experiments.allbaselines_harness.verify import verify
from experiments.deep_search_20260825.deep_search_analysis import runs_for_profile
from sim.energy import CandidateFitness


def _row(
    seq: int,
    energy: float | None,
    work: int,
    *,
    feasible: bool = True,
    generation: int | None = None,
) -> dict:
    fitness = CandidateFitness(
        feasible=feasible,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0) if feasible else (1.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0 if feasible else 1.0,
        normalized_energy=energy if feasible else None,
        proposal_seq=seq,
        n_instances=4,
        n_bundles=3,
    )
    return {
        "candidate_id": f"candidate-{seq}",
        "proposal_seq": seq,
        "generation": generation,
        "cumulative_microwork": work,
        "candidate_microwork": work,
        "trace": (),
        "phase": None,
        "fitness": fitness,
        "fitness_record": None,
    }


def test_normalized_budget_conversion_is_exact_and_never_rounds() -> None:
    assert support.parse_normalized_budget("58.8") == 58_800_000
    assert support.parse_normalized_budget("1.400000") == 1_400_000
    with pytest.raises(ValueError, match="integer microwork"):
        support.parse_normalized_budget("0.0000001")
    with pytest.raises(ValueError, match="positive"):
        support.parse_normalized_budget("0")


def test_proposal_journal_watchdog_fails_closed_and_remains_resumable(tmp_path) -> None:
    journal = tmp_path / "proposal-engine-usage.jsonl"
    ledger = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)
    ledger.bind_journal(journal)
    ledger.reserve_dispatch(
        reserved_wall_clock_ms=30,
        request_sha256="a" * 64,
        seed=2_026_082_101,
    )
    ledger.release_journal_lease()
    harness_config = copy.deepcopy(support.load_harness_config())
    harness_config["artifact_size_guard"]["poll_interval_seconds"] = 0.05
    prepared = SimpleNamespace(run_dir=tmp_path, harness_config=harness_config)

    started = time.monotonic()
    return_code, violation = _run_with_size_guard(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        prepared,
        proposal_journal=journal,
        journal_watchdog_multiplier=1.1,
    )

    assert time.monotonic() - started < 1.0
    assert return_code != 0
    assert isinstance(violation, ProposalEngineJournalStalled)
    assert "1.1 x 30ms reservation" in str(violation)
    resumed = CodexUsageLedger(max_calls=2, max_wall_clock_ms=1_000)
    resumed.bind_journal(journal)
    assert resumed.calls == 1
    assert resumed.wall_clock_ms == 30
    assert resumed.consecutive_failures == 1
    resumed.release_journal_lease()


def test_matching_gives_all_adaptive_arms_one_budget_and_frozen_arms_none() -> None:
    maximum = 5_600_000
    accounting = {
        arm: {
            "mode": "adaptive-normalized-work",
            "registered_max_microwork": maximum,
            "actual_microwork": maximum - 100_000,
            "effective_terminal_microwork": maximum,
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
        matched_max_microwork=maximum,
        include_llm=True,
    )

    assert receipt["verified"] is True
    assert set(receipt["adaptive_arms"]) == set(ADAPTIVE_ARMS)
    accounting["ppo_frozen"]["actual_microwork"] = 1
    with pytest.raises(ValueError, match="adaptation work"):
        validate_matching(
            accounting,
            matched_max_microwork=maximum,
            include_llm=True,
        )


def test_trajectory_uses_exact_phase3_total_order_for_final_incumbent() -> None:
    points = trajectory(
        [
            _row(0, None, 100_000, feasible=False),
            _row(1, 2.0, 1_500_000),
            _row(2, 2.0, 2_900_000),
            _row(3, 1.5, 4_300_000),
        ]
    )

    assert points[0]["candidate_sort_key"] == [1, 1.0, 0.0, 0.0, 0.0, 1.0, 0]
    assert points[1]["final_incumbent_sort_key_so_far"] == [0, 2.0, 1]
    assert points[2]["incumbent_improved"] is False
    assert points[3]["final_incumbent_sort_key_so_far"] == [0, 1.5, 3]


def test_plateau_histogram_groups_only_within_registered_tolerance() -> None:
    histogram = plateau_histogram(
        [
            _row(0, 1.0, 0),
            _row(1, 1.0000005, 100_000),
            _row(2, 1.000002, 200_000),
            _row(3, None, 300_000, feasible=False),
        ],
        tolerance=1e-6,
    )

    assert histogram["l2_feasible_candidates"] == 3
    assert histogram["distinct_feasible_energy_plateaus"] == 2
    assert [bucket["candidate_count"] for bucket in histogram["buckets"]] == [2, 1]


def test_arm_summary_decomposes_initialization_from_search() -> None:
    rows = [_row(0, 2.0, 1_400_000), _row(1, 1.5, 2_800_000)]
    payload = {
        "role": "adaptive-search",
        "accounting": {"actual_microwork": 2_800_000},
        "mechanical_health": {"complete": True},
    }

    summary = arm_summary(
        "same_dsl_gp",
        rows,
        payload,
        tolerance=1e-6,
        initialization_rows=rows[:1],
    )

    assert summary["best_absolute"] == 1.5
    assert summary["best_at_initialization"] == 2.0
    assert summary["search_delta"] == 0.5
    assert summary["search_improved"] is True
    assert summary["best_found_at"] == {
        "coordinate": "proposal",
        "index": 1,
        "proposal_seq": 1,
        "candidate_id": "candidate-1",
    }


def test_gphh_gen0_is_derived_only_from_generation_zero() -> None:
    rows = [
        _row(0, 2.0, 1_400_000, generation=0),
        _row(1, 1.8, 2_800_000, generation=0),
        _row(2, 1.2, 4_200_000, generation=1),
    ]

    derived = gphh_gen0_rows(rows)

    assert [row["proposal_seq"] for row in derived] == [0, 1]
    assert min(row["fitness"].normalized_energy for row in derived) == 1.8


def test_population_parity_charges_each_initialization_through_common_meter() -> None:
    work = EvaluatorWorkBudget(
        registration_status="development-provisional",
        registration_source="test",
        max_microwork=20_000_000,
        rung_microwork={"L0": 100_000, "L1": 300_000, "L2": 1_000_000},
        checkpoints_microwork=(0, 20_000_000),
    )
    trace = (
        {"rung": "L0"},
        {"rung": "L1"},
        {"rung": "L2"},
    )
    rows = []
    for seq in range(8):
        row = _row(seq, 2.0 - seq / 100, (seq + 1) * 1_400_000, generation=0)
        row["candidate_microwork"] = 1_400_000
        row["trace"] = trace
        rows.append(row)
    by_arm = {arm: [dict(row) for row in rows] for arm in ADAPTIVE_ARMS[:-1]}

    receipt = validate_parity_charging(by_arm, expected_count=8, work=work)

    assert receipt["verified"] is True
    assert all(record["actual_microwork"] == 11_200_000 for record in receipt["arms"].values())
    by_arm["random_ast"][0]["candidate_microwork"] = 1
    with pytest.raises(ValueError, match="common meter"):
        validate_parity_charging(by_arm, expected_count=8, work=work)


def _terminal_payload(max_microwork: int) -> dict:
    return {
        "status": "complete",
        "reason": "normalized-work-budget-exhausted",
        "run_purpose": "scientific-full-work",
        "registered_max_microwork": max_microwork,
        "actual_microwork": max_microwork,
        "effective_terminal_microwork": max_microwork,
        "remaining_microwork": 0,
        "budget_carry_microwork": 0,
        "unused_microwork_discarded": 0,
        "terminal_budget_policy": "carry-to-work-limit",
        "usable_for_confirmatory_analysis": False,
        "confirmatory_postflight_receipt_required": True,
    }


def _enhanced_archive(path, *, evolution_profile_name: str = ENHANCED_PROFILE) -> None:
    maximum = 1_400_000
    manifest = {
        "schema": "evolved-continuum-campaign-archive-v4",
        "campaign": {
            "arm": "llm",
            "campaign_id": "h00",
            "root_seed": 7,
            "plan_status": "development-provisional",
            "feedback_profile": ENHANCED_PROFILE,
            "evolution_profile": evolution_profile_name,
            "run_purpose": "scientific-full-work",
            "evaluation_seed_mode": "campaign-derived",
            "evaluator_work": {"max_microwork": maximum},
        },
        "provenance": {},
    }
    archive = EventArchive(path, manifest)
    archive.append("campaign_terminal", _terminal_payload(maximum))
    journal = path / "driver-invocations.jsonl"
    command = ["-m", "campaign", "--profile", ENHANCED_PROFILE]
    _append_driver(
        journal,
        "started",
        {"invocation_id": 0, "arm": "llm", "resume": False, "command": command},
    )
    _append_driver(
        journal,
        "started",
        {"invocation_id": 1, "arm": "llm", "resume": True, "command": command},
    )
    _append_driver(
        journal,
        "finished",
        {"invocation_id": 1, "arm": "llm", "exit_code": 0},
    )


def test_enhanced_resumed_archive_passes_but_profile_divergence_fails(tmp_path) -> None:
    prepared = SimpleNamespace(
        root_seed=7,
        max_microwork=1_400_000,
        population_parity=False,
        evolution_profile=evolution_profile(ENHANCED_PROFILE),
    )
    resumed = tmp_path / "llm"
    _enhanced_archive(resumed)

    rows, metadata = _campaign_rows(
        resumed,
        expected_campaign_arm="llm",
        prepared=prepared,
    )

    assert rows == []
    assert metadata["mechanical_health"]["complete"] is True
    assert metadata["mechanical_health"][
        "driver_superseded_unfinished_invocations"
    ] == [0]

    divergent = tmp_path / "divergent" / "llm"
    _enhanced_archive(divergent, evolution_profile_name="matched-llm-vs-gp")
    with pytest.raises(ValueError, match='"evolution_profile"'):
        _campaign_rows(
            divergent,
            expected_campaign_arm="llm",
            prepared=prepared,
        )


def test_prepared_run_selects_one_dev_scenario_and_pins_the_matching_budget(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(support, "RUNS", tmp_path / "runs")

    prepared = support.prepare_run(
        "dev-medium-cluster",
        support.ROOT / "evolution/task/config_dev_medium.json",
        "1.4",
        800_000_011,
        include_llm=False,
    )

    runtime_task = json.loads(prepared.task_config.read_text())
    plan = load_campaign_plan(prepared.campaign_config)
    runtime_codex = load_codex_engine_config(prepared.codex_config)
    assert [item["id"] for item in runtime_task["scenarios"]] == ["dev-medium-cluster"]
    assert runtime_task["data_role"] == "development-only"
    assert plan.evaluator_work.max_microwork == 1_400_000
    assert plan.campaigns[0].root_seed == 800_000_011
    assert len(plan.campaigns) == 10
    assert prepared.population_parity is False
    assert prepared.initialization_population_size == 1
    assert runtime_codex.max_calls_per_campaign == support.required_call_ceiling(
        prepared.harness_config, prepared.max_microwork
    )
    assert prepared.run_dir.name == (
        "dev-medium-cluster--seed-800000011--work-1400000--non-llm"
    )
    manifest = support.read_sealed(
        prepared.run_dir / "inputs/input_manifest.json"
    )["payload"]
    assert manifest["initialization_mode"] == "single-seed"
    assert manifest["seed_archive_random_mutants"] is None


def test_openrouter_run_name_is_engine_isolated_and_codex_name_is_historical(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(support, "RUNS", tmp_path / "runs")
    common = (
        "dev-medium-cluster",
        support.ROOT / "evolution/task/config_dev_medium.json",
        "1.4",
        800_000_013,
    )

    codex = support.prepare_run(*common, include_llm=True)
    openrouter = support.prepare_run(*common, include_llm=True, engine="openrouter")

    assert codex.run_dir.name == (
        "dev-medium-cluster--seed-800000013--work-1400000--all-arms"
    )
    assert "--engine-openrouter" in openrouter.run_dir.name
    assert openrouter.run_dir != codex.run_dir


def test_analysis_filter_separates_proposal_engines(tmp_path) -> None:
    codex = tmp_path / (
        "dev-x--seed-1--work-1--all-arms--deep-search--seed-archive-5"
    )
    openrouter = tmp_path / (
        "dev-x--seed-1--work-1--all-arms--deep-search--engine-openrouter"
        "--seed-archive-5"
    )
    for run in (codex, openrouter):
        run.mkdir()
        (run / "report.json").write_text("{}")

    assert runs_for_profile(tmp_path, "deep-search", "codex") == [codex]
    assert runs_for_profile(tmp_path, "deep-search", "openrouter") == [openrouter]


def test_feedback_ablation_run_name_and_campaign_profile_are_distinct(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(support, "RUNS", tmp_path / "runs")
    common = (
        "dev-medium-cluster",
        support.ROOT / "evolution/task/config_dev_medium.json",
        "1.4",
        800_000_014,
    )

    baseline = support.prepare_run(
        *common, include_llm=True, profile_name=DEEP_SEARCH_PROFILE
    )
    feedback = support.prepare_run(
        *common, include_llm=True, profile_name=DEEP_SEARCH_FEEDBACK_PROFILE
    )

    assert baseline.run_dir != feedback.run_dir
    assert f"--{DEEP_SEARCH_PROFILE}" in baseline.run_dir.name
    assert f"--{DEEP_SEARCH_FEEDBACK_PROFILE}" in feedback.run_dir.name
    assert load_campaign_plan(feedback.campaign_config).evolution_profile == (
        DEEP_SEARCH_FEEDBACK_PROFILE
    )


def test_harness_ablation_flags_are_independent_and_default_off(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(support, "RUNS", tmp_path / "runs")
    common = (
        "dev-medium-cluster",
        support.ROOT / "evolution/task/config_dev_medium.json",
        "1.4",
    )

    control = support.prepare_run(*common, 800_000_040, include_llm=False)
    before = {
        path.name: path.read_bytes()
        for path in (control.run_dir / "inputs").iterdir()
        if path.is_file()
    }
    explicit_off = support.prepare_run(
        *common,
        800_000_040,
        include_llm=False,
        ablation_probe_probability=0.0,
        ablation_prompt_hint_probability=0.0,
    )
    assert explicit_off.run_dir == control.run_dir
    assert before == {
        path.name: path.read_bytes()
        for path in (explicit_off.run_dir / "inputs").iterdir()
        if path.is_file()
    }
    assert "ablation_probe_probability" not in json.loads(
        control.campaign_config.read_text()
    )

    operator_only = support.prepare_run(
        *common,
        800_000_041,
        include_llm=False,
        ablation_probe_probability=0.2,
    )
    operator_plan = load_campaign_plan(operator_only.campaign_config)
    assert operator_plan.ablation_probe_probability == 0.2
    assert operator_plan.ablation_prompt_hint_probability == 0.0
    assert operator_only.run_dir.name.endswith("--ablation-probe-0p2")

    prompt_only = support.prepare_run(
        *common,
        800_000_042,
        include_llm=True,
        ablation_prompt_hint_probability=0.2,
    )
    prompt_plan = load_campaign_plan(prompt_only.campaign_config)
    assert prompt_plan.ablation_probe_probability == 0.0
    assert prompt_plan.ablation_prompt_hint_probability == 0.2
    assert prompt_only.run_dir.name.endswith("--ablation-prompt-0p2")

    with pytest.raises(ValueError, match="require --include-llm"):
        support.prepare_run(
            *common,
            800_000_043,
            include_llm=False,
            ablation_prompt_hint_probability=0.2,
        )


def test_population_parity_requires_exact_gphh_population_and_is_path_isolated(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(support, "RUNS", tmp_path / "runs")

    prepared = support.prepare_run(
        "dev-medium-cluster",
        support.ROOT / "evolution/task/config_dev_medium.json",
        "58.8",
        800_000_011,
        include_llm=False,
        seed_archive_random_mutants=5,
    )

    assert prepared.population_parity is True
    assert prepared.initialization_population_size == 12
    assert prepared.run_dir.name.endswith("--seed-archive-5")
    with pytest.raises(ValueError, match="equal the classic GPHH population"):
        support.prepare_run(
            "dev-medium-cluster",
            support.ROOT / "evolution/task/config_dev_medium.json",
            "58.8",
            800_000_012,
            include_llm=False,
            seed_archive_random_mutants=1,
        )


def test_a1_refusals_and_configured_hard_ceilings_are_live() -> None:
    config = support.load_harness_config()
    codex = load_codex_engine_config(support.ROOT / config["llm"]["config"])

    refusals = verify_a1_refusals(codex, feedback_profile=config["feedback_profile"])

    assert all(refusals.values())
    assert codex.max_calls_per_campaign == config["llm"]["call_ceiling_policy"][
        "minimum_calls"
    ]
    assert support.required_call_ceiling(config, 58_800_000) == 150
    assert codex.max_wall_clock_s_per_campaign == config["llm"][
        "hard_wall_clock_seconds"
    ]
    assert set(HANDWRITTEN_NAMES) < set(EVALUATION_ONLY_ARMS)


def test_sealed_result_detects_payload_tampering(tmp_path) -> None:
    path = tmp_path / "result.json"
    support.write_sealed(path, {"development_only": True, "value": 1})
    raw = json.loads(path.read_text())
    raw["payload"]["value"] = 2
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="digest mismatch"):
        support.read_sealed(path)


def test_size_guard_names_oversized_artifact_and_run_directory(tmp_path) -> None:
    config = {
        "artifact_size_guard": {
            "max_arm_result_bytes": 8,
            "max_artifact_bytes": 10,
            "max_run_bytes": 15,
            "poll_interval_seconds": 0.05,
        }
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "oversized.jsonl"
    artifact.write_bytes(b"x" * 11)

    with pytest.raises(support.RunSizeBudgetExceeded, match=str(artifact)):
        support.enforce_run_size(run_dir, config)

    artifact.write_bytes(b"x" * 8)
    second = run_dir / "second.jsonl"
    second.write_bytes(b"y" * 8)
    with pytest.raises(support.RunSizeBudgetExceeded, match=str(run_dir)):
        support.enforce_run_size(run_dir, config)

    result = run_dir / "result.json"
    with pytest.raises(support.RunSizeBudgetExceeded, match=str(result)):
        support.write_exact(result, "too-large", max_bytes=8)
    assert not result.exists()


def test_harness_config_registers_fail_closed_size_budgets() -> None:
    guard = support.size_guard(support.load_harness_config())

    assert guard["max_arm_result_bytes"] == 10 * 1024 * 1024
    assert guard["max_artifact_bytes"] == 1024 * 1024 * 1024
    assert guard["max_run_bytes"] == 4 * 1024 * 1024 * 1024
    assert guard["poll_interval_seconds"] == 1.0


def test_gzip_sealed_result_loads_and_stream_verifies_without_decompression(
    tmp_path,
) -> None:
    path = tmp_path / "result.json"
    wrapped = support.write_sealed(path, {"development_only": True, "value": 1})
    compressed = path.with_name("result.json.gz")
    with path.open("rb") as source, gzip.open(compressed, "wb") as destination:
        destination.write(source.read())
    path.unlink()

    assert support.read_sealed(path) == wrapped
    receipt = support.verify_sealed_stream(path)
    assert receipt["payload_sha256"] == wrapped["payload_sha256"]
    assert receipt["path"] == str(compressed)


def _fake_evaluation(seq: int, energy: float, ledger: str) -> SimpleNamespace:
    fitness = CandidateFitness(
        feasible=True,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0,
        normalized_energy=energy,
        proposal_seq=seq,
        n_instances=1,
        n_bundles=1,
    )
    instance = SimpleNamespace(
        scenario_id="dev-test",
        seed=123,
        bundle_id="dev-1",
        episode_digest="1" * 64,
        certificate_seal="2" * 64,
        energy_j=energy * 100.0,
        idle_floor_j=10.0,
        ceiling_j=110.0,
        slo_achieved={"latency": 1.0},
        goodput=1.0,
        drop_rate=0.0,
        timeout_rate=0.0,
        disruption={"moves": 0.0},
        boundary_certificate={"seal": "2" * 64},
        verified_energy={"component": 1.0},
        episode={"decision_log": ledger},
    )
    return SimpleNamespace(
        fitness=fitness,
        instances=(instance,),
        policy_name=f"candidate-{seq}",
        policy_version="v1",
        policy_identity_sha256=f"{seq + 3:064x}",
        policy_artifact_sha256=f"{seq + 5:064x}",
    )


def _fake_candidate(seq: int, generation: int, energy: float, ledger: str) -> SimpleNamespace:
    evaluation = _fake_evaluation(seq, energy, ledger)
    result = SimpleNamespace(
        promoted_to_l2=True,
        evaluation=evaluation,
        trace=(
            {"rung": "L0", "passed": True},
            {"rung": "L1", "passed": True},
            {"rung": "L2", "passed": True},
        ),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=evaluation.policy_name,
        policy_version=evaluation.policy_version,
        policy_identity_sha256=evaluation.policy_identity_sha256,
        policy_artifact_sha256=evaluation.policy_artifact_sha256,
        candidate_failure=None,
    )
    return SimpleNamespace(
        proposal_seq=seq,
        generation=generation,
        tree=f"tree-{seq}",
        result=result,
        candidate_microwork=1_400_000,
        cumulative_microwork=(seq + 1) * 1_400_000,
        selection_key=(0, *evaluation.fitness.sort_key()),
    )


def test_compact_completed_arm_omits_ledgers_and_preserves_prefixed_summary(
    tmp_path,
) -> None:
    ledger = "x" * (2 * 1024 * 1024)
    candidates = (
        _fake_candidate(0, 0, 2.0, ledger),
        _fake_candidate(1, 1, 1.5, ledger),
    )
    run = SimpleNamespace(
        root_seed=7,
        config=GPHHConfig(population_size=2, generations=2),
        candidates=candidates,
        champion=candidates[1],
        spent_microwork=2_800_000,
        stop_reason="normalized-work-budget-exhausted",
        population_proposal_seqs_by_generation=((0,), (1,)),
    )
    compact_run = _compact_adaptive_run("classic_gphh", run)
    compact_payload = _adaptive_payload(
        "classic_gphh",
        compact_run,
        spent_microwork=run.spent_microwork,
        stop_reason=run.stop_reason,
        maximum=run.spent_microwork,
        tolerance=1e-6,
        references=[],
    )
    compact_path = tmp_path / "compact.json"
    support.write_sealed(compact_path, compact_payload, max_bytes=10 * 1024 * 1024)

    full_candidates = []
    for candidate in candidates:
        evaluation = candidate.result.evaluation
        full_candidates.append(
            {
                "proposal_seq": candidate.proposal_seq,
                "generation": candidate.generation,
                "candidate_microwork": candidate.candidate_microwork,
                "cumulative_microwork": candidate.cumulative_microwork,
                "ladder": {
                    "trace": list(candidate.result.trace),
                    "policy": {
                        "name": evaluation.policy_name,
                        "version": evaluation.policy_version,
                        "identity_sha256": evaluation.policy_identity_sha256,
                        "artifact_sha256": evaluation.policy_artifact_sha256,
                    },
                    "evaluation": {
                        "fitness": _fitness_dict(evaluation.fitness),
                        "instances": [{"episode": {"decision_log": ledger}}],
                    },
                },
            }
        )
    legacy_body = {
        "schema": "evolved-continuum-classic-gphh-run-v2",
        "candidates": full_candidates,
    }
    legacy_run = {**legacy_body, "sha256": support.canonical_sha256(legacy_body)}
    legacy_payload = {
        **compact_payload,
        "schema": "evolved-continuum-allbaselines-adaptive-arm-v1",
        "run": legacy_run,
    }
    legacy_path = tmp_path / "legacy.json"
    support.write_sealed(legacy_path, legacy_payload)

    compact_rows, loaded_compact = _adaptive_arm(compact_path)
    legacy_rows, loaded_legacy = _adaptive_arm(legacy_path)
    compact_summary = arm_summary(
        "classic_gphh",
        compact_rows,
        loaded_compact,
        tolerance=1e-6,
        initialization_rows=gphh_gen0_rows(compact_rows),
    )
    legacy_summary = arm_summary(
        "classic_gphh",
        legacy_rows,
        loaded_legacy,
        tolerance=1e-6,
        initialization_rows=gphh_gen0_rows(legacy_rows),
    )
    fields = (
        "best",
        "best_absolute",
        "best_at_initialization",
        "search_delta",
        "search_improved",
        "candidate_count",
        "plateau_histogram",
        "trajectory",
    )
    assert {field: compact_summary[field] for field in fields} == {
        field: legacy_summary[field] for field in fields
    }
    assert compact_path.stat().st_size < 10 * 1024 * 1024
    assert compact_path.stat().st_size < legacy_path.stat().st_size / 100
    assert "decision_log" not in compact_path.read_text()


def _fitness_dict(fitness: CandidateFitness) -> dict:
    return {
        "feasible": fitness.feasible,
        "infeasibility_vector": list(fitness.infeasibility_vector),
        "summed_primitive_shortfall": fitness.summed_primitive_shortfall,
        "normalized_energy": fitness.normalized_energy,
        "proposal_seq": fitness.proposal_seq,
        "n_instances": fitness.n_instances,
        "n_bundles": fitness.n_bundles,
        "sort_key": list(fitness.sort_key()),
        "direction": "lower-is-better",
    }


def test_existing_all_arms_reports_and_search_isolation_synthesis_verify() -> None:
    reports = sorted(
        (support.HERE / "runs").glob("dev-*--work-58800000--all-arms/report.json")
    )
    synthesis = (
        support.ROOT
        / "experiments/search_isolation_20260809/runs/SYNTHESIS.json"
    )

    receipt = verify([*reports, synthesis])

    # Every discovered all-arms report must verify; the count grows as legitimate
    # campaigns are added (e.g. replication roots), so it is a floor, not a fixture.
    assert len(reports) >= 3
    assert receipt["status"] == "VERIFIED"
    assert receipt["artifact_count"] == len(reports) + 1
