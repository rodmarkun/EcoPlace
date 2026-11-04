"""G0 capacity arithmetic, authority boundaries, and canonical artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution import g0


def _timing() -> dict[str, dict]:
    return {
        "L0": {
            "repetitions": 30,
            "successful_calls": 30,
            "p50_seconds": 0.01,
            "p90_seconds": 0.02,
            "mean_seconds": 0.012,
            "failure_rate": 0.0,
            "invalid_candidate_rate": 0.0,
            "promotion_rate": 0.8,
            "instances_per_batch": 3,
            "p90_seconds_per_instance_linear_scaling_assumption": 0.02 / 3,
            "percentile_method": "fixed-test-input",
        },
        "L1": {
            "repetitions": 10,
            "successful_calls": 10,
            "p50_seconds": 0.1,
            "p90_seconds": 0.2,
            "mean_seconds": 0.12,
            "failure_rate": 0.1,
            "invalid_candidate_rate": 0.0,
            "promotion_rate": 0.5,
            "instances_per_batch": 1,
            "p90_seconds_per_instance_linear_scaling_assumption": 0.2,
            "percentile_method": "fixed-test-input",
        },
        "L2": {
            "repetitions": 5,
            "successful_calls": 5,
            "p50_seconds": 0.8,
            "p90_seconds": 1.0,
            "mean_seconds": 0.82,
            "failure_rate": 0.0,
            "invalid_candidate_rate": 0.0,
            "promotion_rate": 1.0,
            "instances_per_batch": 6,
            "p90_seconds_per_instance_linear_scaling_assumption": 1.0 / 6,
            "percentile_method": "fixed-test-input",
        },
    }


def _ceilings(*, approved: bool = False) -> dict:
    return {
        "schema": g0.CEILINGS_SCHEMA,
        "status": (g0.APPROVED_CEILINGS_STATUS if approved else g0.PLACEHOLDER_CEILINGS_STATUS),
        "researcher_hours_per_week": 40.0,
        "weeks": 10.0,
        "cpu_hours": 2000.0,
        "api_spend_microdollars": 5_000_000_000,
        "storage_gb": 2000.0,
        "wallclock_days": 42.0,
        "spend_authorized": approved,
    }


def _host() -> dict:
    return {
        "logical_cpu_count": 10,
        "platform": "fixed-test-platform",
        "machine": "arm64",
        "python": "3.12.0",
        "load_average_min_1m_5m_15m": [1.0, 1.0, 1.0],
        "load_average_max_1m_5m_15m": [2.0, 2.0, 2.0],
        "max_normalized_load_1m": 0.2,
        "sampling": "fixed-test-input",
    }


def _quarantine() -> dict:
    return {
        "task_config": "evolution/task/config.json",
        "task_config_sha256": g0.EXPECTED_TASK_SHA256,
        "data_role": "development-only",
        "pilot_binding": "experiments/pilot_codex_20260728/run_pilot.py",
        "scenarios": [g0.EXPECTED_SCENARIO],
        "seeds": list(g0.EXPECTED_SCENARIO_SEEDS),
        "bundles": list(g0.EXPECTED_BUNDLES),
    }


def _build(*, mode: str = "development-test", llm_pilot=None) -> dict:
    evidence = (
        g0.DEVELOPMENT_EVIDENCE_CLASS if mode == "development-test" else g0.GATE_EVIDENCE_CLASS
    )
    return g0.build_capacity_payload(
        mode=mode,
        run_id="fixed-test",
        evidence_class=evidence,
        timing_summary=_timing(),
        timing_payload_sha256="a" * 64,
        host_context=_host(),
        ceilings=_ceilings(approved=mode == "gate-evidence"),
        quarantine=_quarantine(),
        audit_fraction=0.1,
        assumed_generation_seconds=60.0,
        llm_pilot=llm_pilot,
        parallel_cpu_workers=4,
    )


def _llm_pilot(*, cli_failures: int = 1, invalid_bundles: int = 2) -> dict:
    successful = 10 - cli_failures
    valid = successful - invalid_bundles
    if valid < 1:
        raise ValueError("test pilot must retain at least one valid bundle")
    factor = 10 / valid
    return {
        "artifact_revision": "test-r1",
        "evidence_class": g0.DEVELOPMENT_EVIDENCE_CLASS,
        "summary": {
            "calls": 10,
            "successful_cli_calls": successful,
            "cli_failures": cli_failures,
            "valid_bundles": valid,
            "invalid_bundles": invalid_bundles,
            "l0_exercised_bundles": valid,
            "latency_seconds": {
                "p50": 20.0,
                "p90": 30.0,
                "mean": 21.0,
                "samples": [float(value) for value in range(16, 26)],
                "percentile_method": "fixed-test-input",
            },
            "rates": {
                "cli_failure_rate": cli_failures / 10,
                "invalid_bundle_rate_among_cli_successes": invalid_bundles / successful,
                "end_to_end_retry_invalid_rate": (cli_failures + invalid_bundles) / 10,
            },
            "inflation": {
                "cli_retry_factor": 10 / successful,
                "invalid_bundle_retry_factor": successful / valid,
                "retry_invalid_factor": factor,
                "retry_invalid_fraction": factor - 1.0,
            },
            "token_totals": {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 20,
                "reasoning_output_tokens": 5,
                "total_tokens": 120,
            },
            "implied_calls_per_hour_at_p90": 120.0,
        },
    }
def _write_sealed(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(g0.seal_payload(payload), sort_keys=True, indent=2) + "\n")


def test_capacity_harness_is_deterministic_for_fixed_timing_inputs() -> None:
    first = _build()
    second = _build()
    assert first == second
    assert g0.canonical_sha256(first) == g0.canonical_sha256(second)


def test_reserve_check_accepts_exact_30_percent_boundary() -> None:
    check = g0.reserve_check(70, 100)
    assert check["reserve_fraction"] == 0.3
    assert check["holds"] is True


def test_reserve_check_rejects_just_over_boundary() -> None:
    check = g0.reserve_check(70.000001, 100)
    assert check["reserve_fraction"] < 0.3
    assert check["holds"] is False


def test_development_run_cannot_emit_gate_verdict() -> None:
    payload = _build()
    assert payload["status"] in {"DEVELOPMENT_PASS", "DEVELOPMENT_FAIL"}
    assert "gate_verdict" not in payload
    assert payload["gate_verdict_policy"] == "NOT_EMITTED_FROM_DEVELOPMENT_TEST"
    assert payload["acceptance_evidence"] is False


def test_measured_llm_line_still_blocks_gate_pass_on_p1_engine_decision() -> None:
    payload = _build(mode="gate-evidence", llm_pilot=_llm_pilot())
    assert payload["llm_generation"]["status"] == "MEASURED_DEVELOPMENT_ONLY"
    assert payload["llm_generation"]["blocking_gate_pass"] is True
    assert payload["llm_generation"]["blocking_reason"] == g0.LLM_BLOCKING_REASON
    assert payload["gate_verdict"] == "FAIL"
    assert g0.LLM_BLOCKING_REASON in payload["blocking_items"]


def test_llm_line_uses_measured_retry_invalid_inflation_arithmetic() -> None:
    payload = _build(llm_pilot=_llm_pilot(cli_failures=1, invalid_bundles=2))
    llm = payload["llm_generation"]
    assert llm["base_proposals_total"] == 5300
    assert llm["retry_invalid_inflation_factor"] == pytest.approx(10 / 7)
    assert llm["retry_invalid_inflation_fraction"] == pytest.approx(3 / 7)
    assert llm["capacity_call_count_total"] == pytest.approx(5300 * (1 + 3 / 7))
    assert llm["subscription_rate_constraint"]["sequential_occupancy_hours"] == (
        pytest.approx(5300 * (10 / 7) * 30 / 3600)
    )


def test_llm_line_has_subscription_occupancy_without_cost_or_energy_fields() -> None:
    payload = _build(llm_pilot=_llm_pilot())
    llm_material = {
        "line": payload["llm_generation"],
        "generation_rows": [
            row["generation"]
            for row in payload["arm_enumeration"]["rows"]
            if row["generation"]["required"]
        ],
    }
    keys: set[str] = set()

    def collect(value) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(llm_material)
    assert not any("dollar" in key.lower() for key in keys)
    assert not any("energy" in key.lower() for key in keys)
    assert llm_material["line"]["subscription_rate_constraint"][
        "implied_calls_per_hour_at_p90"
    ] == pytest.approx(120.0)


def test_llm_capacity_seal_is_stable_for_stubbed_pilot_inputs() -> None:
    first = g0.seal_payload(_build(llm_pilot=_llm_pilot()))
    second = g0.seal_payload(_build(llm_pilot=_llm_pilot()))
    assert first == second


def test_ceilings_file_validation_accepts_only_sealed_complete_input(tmp_path: Path) -> None:
    path = tmp_path / "ceilings.json"
    payload = _ceilings()
    _write_sealed(path, payload)
    assert g0.load_ceilings(path) == payload

    tampered = json.loads(path.read_text())
    tampered["payload"]["cpu_hours"] = 1.0
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest mismatch"):
        g0.load_ceilings(path)


def test_ceilings_file_validation_fails_closed_on_missing_or_extra_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ceilings.json"
    missing = _ceilings()
    missing.pop("storage_gb")
    _write_sealed(path, missing)
    with pytest.raises(ValueError, match="fields must be exactly"):
        g0.load_ceilings(path)

    extra = {**_ceilings(), "unreviewed_override": 1}
    _write_sealed(path, extra)
    with pytest.raises(ValueError, match="fields must be exactly"):
        g0.load_ceilings(path)


def test_placeholder_ceilings_cannot_enter_gate_path(tmp_path: Path) -> None:
    path = tmp_path / "ceilings.json"
    _write_sealed(path, _ceilings())
    with pytest.raises(ValueError, match="researcher-approved"):
        g0.load_ceilings(path, gate_evidence=True)

    legacy_spend_field_is_not_authority = _ceilings(approved=True)
    legacy_spend_field_is_not_authority["spend_authorized"] = False
    _write_sealed(path, legacy_spend_field_is_not_authority)
    assert g0.load_ceilings(path, gate_evidence=True)["spend_authorized"] is False


def test_canonical_hash_is_stable_across_mapping_insertion_order() -> None:
    first = {"z": [3, 2, 1], "a": {"right": 2, "left": 1}}
    second = {"a": {"left": 1, "right": 2}, "z": [3, 2, 1]}
    assert g0.canonical_json(first) == g0.canonical_json(second)
    assert g0.seal_payload(first)["payload_sha256"] == g0.seal_payload(second)["payload_sha256"]


def test_arm_enumeration_covers_plan_methods_and_experiments() -> None:
    identifiers = {arm.arm_id for arm in g0.PROVISIONAL_ARMS}
    assert {
        "e1-llm-evolution",
        "e1-same-dsl-gp",
        "e1-classic-gphh",
        "e1-ppo-frozen",
        "e1-random-ast-floor",
        "e3-online-gphh-adaptation",
        "e5-2-text-feedback-off",
        "e5-3-numeric-patching-off",
        "e5-4-l2-only",
        "e5-5-single-population",
        "e2-i-held-out-seeds",
        "e2-ii-network-structure",
        "e2-iii-train-ticket",
    } <= identifiers
    assert len([arm for arm in g0.PROVISIONAL_ARMS if "reference" in arm.arm_id]) == 6
    assert all(arm.campaigns > 0 for arm in g0.PROVISIONAL_ARMS)


def test_timing_summary_reports_failures_invalids_and_p90() -> None:
    samples = [
        {
            "duration_seconds": duration,
            "failure": failure,
            "invalid_candidate": invalid,
            "promoted": promoted,
        }
        for duration, failure, invalid, promoted in (
            (1.0, False, False, True),
            (2.0, False, True, False),
            (3.0, True, True, None),
        )
    ]
    summary = g0.summarize_timing_samples(samples, instances_per_batch=2)
    assert summary["p50_seconds"] == 2.0
    assert summary["p90_seconds"] == pytest.approx(2.8)
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["invalid_candidate_rate"] == pytest.approx(2 / 3)
    assert summary["promotion_rate"] == 0.5


def test_llm_sample_summary_derives_retry_and_invalid_inflation() -> None:
    samples = []
    for index in range(10):
        cli_failure = index == 0
        valid_bundle = index >= 3
        samples.append(
            {
                "call_index": index,
                "dispatched": True,
                "duration_seconds": float(index + 1),
                "cli_failure_category": "timeout" if cli_failure else None,
                "valid_bundle": valid_bundle,
                "l0_exercised": valid_bundle,
                "token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                },
            }
        )

    summary = g0.summarize_llm_generation_samples(samples, hard_call_cap=10)

    assert summary["cli_failures"] == 1
    assert summary["invalid_bundles"] == 2
    assert summary["valid_bundles"] == 7
    assert summary["rates"]["cli_failure_rate"] == pytest.approx(0.1)
    assert summary["rates"]["invalid_bundle_rate_among_cli_successes"] == (
        pytest.approx(2 / 9)
    )
    assert summary["inflation"]["cli_retry_factor"] == pytest.approx(10 / 9)
    assert summary["inflation"]["invalid_bundle_retry_factor"] == pytest.approx(9 / 7)
    assert summary["inflation"]["retry_invalid_factor"] == pytest.approx(10 / 7)
    assert summary["inflation"]["retry_invalid_fraction"] == pytest.approx(3 / 7)


def test_mode_and_evidence_class_must_match() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        g0.build_capacity_payload(
            mode="development-test",
            run_id="fixed-test",
            evidence_class=g0.GATE_EVIDENCE_CLASS,
            timing_summary=_timing(),
            timing_payload_sha256="a" * 64,
            host_context=_host(),
            ceilings=_ceilings(),
            quarantine=_quarantine(),
            audit_fraction=0.1,
            assumed_generation_seconds=60.0,
            llm_pilot=None,
        )
