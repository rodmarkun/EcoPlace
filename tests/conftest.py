"""Test-suite adjustments for the public repository.

The tests were written against the full development tree. Some need inputs this repository does
not ship; those tests are skipped, naming the missing input, only while it is absent, so they run
wherever it is present. A few tests of development experiments that the paper does not use are
stale (the experiment code changed after the test was written, and that code is hash-bound to
its registration, so it is kept as it ran); they are marked as expected failures.
"""

import importlib.util
import os
import platform
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALIBABA_SAMPLE = ROOT / "experiments/alibaba_ingestion_20260809/call_graph_sample.json"


def _missing_module(*names):
    return any(importlib.util.find_spec(n) is None for n in names)


# (condition under which to skip, reason, test ids; an id without "[...]" covers all parameters)
SKIP_IF = [
    (
        lambda: not ALIBABA_SAMPLE.exists(),
        "needs experiments/alibaba_ingestion_20260809/call_graph_sample.json "
        "(doi:10.5281/zenodo.23964625)",
        [
            "tests/test_alibaba_workload_builder.py::test_builder_emits_dense_multi_template_fan_out_specs_from_selected_fits",
            "tests/test_alibaba_workload_builder.py::test_builder_is_deterministic_for_one_seed_and_seed_sensitive",
            "tests/test_alibaba_workload_builder.py::test_calibration_informed_sealer_requires_exact_typed_builder_workload",
            "tests/test_alibaba_workload_builder.py::test_exact_export_byte_tampering_fails_closed[source1-call-graph sample exact-byte]",  # noqa: E501
            "tests/test_alibaba_workload_builder.py::test_sealed_loader_revalidates_all_fits_scans_and_parent_chain",
            "tests/test_dev_tight_scenario.py::test_far_edge_capacity_is_actually_contended",
            "tests/test_dev_tight_scenario.py::test_horizon_outlives_the_published_fog_wake",
            "tests/test_dev_tight_scenario.py::test_manifest_matches_the_tree",
            "tests/test_dev_tight_scenario.py::test_queues_are_bounded",
            "tests/test_dev_tight_scenario.py::test_slo_can_actually_bind",
            "tests/test_dev_tight_scenario.py::test_the_two_scenarios_are_structurally_different",
            "tests/test_scenario_dev_realistic.py::test_hierarchical_and_zoo_topology_labels_and_shapes_are_exact",
            "tests/test_scenario_dev_realistic.py::test_hot_scenario_reuses_medium_cluster_and_adds_absorbable_flash_load",
            "tests/test_scenario_dev_realistic.py::test_manifest_rebuilds_every_config_topology_and_workload_closure",
            "tests/test_scenario_dev_realistic.py::test_sealed_richness_evidence_is_current_and_passes",
        ],
    ),
    (
        lambda: not (ROOT / "data/alibaba-clusterdata").exists(),
        "needs the pinned trace checkouts from scripts/fetch_data.sh",
        ["tests/test_scenario_sources.py::test_registry_split_is_sealed_and_retained_metadata_rehashes"],
    ),
    (
        lambda: _missing_module("gymnasium", "cloudpickle"),
        "needs the rl extra (uv sync --extra rl)",
        [
            "tests/test_baseline_ppo.py::test_training_environment_is_action_conditioned_and_masks_invalid_nodes",
            "tests/test_corrected_ab_launcher.py::test_peak_windows_are_waited_out_not_skipped",
        ],
    ),
    (
        lambda: not (ROOT / "experiments/search_isolation_20260809/runs/SYNTHESIS.json").exists(),
        "needs development campaign reports that are not published",
        ["tests/test_allbaselines_harness.py::test_existing_all_arms_reports_and_search_isolation_synthesis_verify"],
    ),
    (
        lambda: bool(os.environ.get("CI")),
        "wall-clock decision-latency budget (p99 <= 5 ms) is measured on the study machine, "
        "not on shared CI runners",
        ["tests/test_risk_gates.py::TestSeedPoliciesPassAllGates::test_seed_policy"],
    ),
    (
        lambda: (platform.system(), platform.machine()) != ("Darwin", "arm64"),
        "the sealed energy-bundle registration was computed on macOS arm64; libm results differ "
        "in the last bits on other platforms",
        [
            "tests/test_energy_bundles.py::TestSplit::test_builder_monkeypatch_cannot_redefine_registered_split",
            "tests/test_energy_bundles.py::TestSplit::test_draft_split_is_valid",
        ],
    ),
]

STALE = {
    "tests/test_independent_calibration.py::test_freeze_stops_before_artifact_write_if_loader_repair_missing":
        "validation_20260909 scenario design changed after its frozen bytes were recorded",
    "tests/test_independent_calibration.py::test_frozen_archive_round_trip_and_tamper_refusal":
        "validation_20260909 scenario design changed after its frozen bytes were recorded",
    "tests/test_matched_cli_recovery.py::test_cli_replays_or_stops_without_duplicate_dispatch[lost-response]":
        "matched_cli_pilot_20260909 now stops at the candidate safety ceiling before the "
        "retry-refusal path this case expects",
}


def _killpg_exit_race(exc):
    """evolution.codex_engine._stop_process_group checks poll() and then calls os.killpg. If
    the child exits in between, Linux raises ProcessLookupError (handled) but macOS raises
    PermissionError for an exited, not yet reaped group leader (not handled). The engine is
    hash-bound to the Study F registration as it ran, so the race is retried here instead."""
    if not isinstance(exc, PermissionError):
        return False
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "_stop_process_group":
            return True
        tb = tb.tb_next
    return False


def _stub_startup_too_slow(exc):
    """Some Codex engine tests give the stub CLI 0.05 s to print its first event. On a loaded CI
    runner the stub can miss that window and print nothing, which the engine reports as "saw
    event types []". The tests are hash-bound like the engine, so this case is retried too."""
    return isinstance(exc, AssertionError) and "saw event types []" in str(exc)


def _retry_transient_codex_failures(item, attempts=3):
    run = item.runtest

    def runtest():
        for attempt in range(attempts):
            try:
                return run()
            except (PermissionError, AssertionError) as exc:
                transient = _killpg_exit_race(exc) or _stub_startup_too_slow(exc)
                if attempt == attempts - 1 or not transient:
                    raise

    item.runtest = runtest


def _matches(nodeid, target):
    return nodeid == target or ("[" not in target and nodeid.startswith(target + "["))


def pytest_collection_modifyitems(config, items):
    active = [(reason, ids) for condition, reason, ids in SKIP_IF if condition()]
    for item in items:
        for reason, ids in active:
            if any(_matches(item.nodeid, t) for t in ids):
                item.add_marker(pytest.mark.skip(reason=reason))
                break
        if item.nodeid in STALE:
            item.add_marker(pytest.mark.xfail(reason=STALE[item.nodeid], strict=False))
        if item.nodeid.startswith("tests/test_codex_engine.py::"):
            _retry_transient_codex_failures(item)
