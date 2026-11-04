"""Fail-closed seals, seed replay, and shape invariants for the typed builder."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sim.scenarios.alibaba_workload import (
    CALIBRATION_PATH,
    CALL_GRAPH_SAMPLE_PATH,
    AlibabaWorkloadBuildConfig,
    _verify_rt_fits,
    build_calibration_informed_workload,
    load_sealed_alibaba_calibration,
)
from sim.scenarios.artifacts import (
    seal_calibration_informed_synthetic_workload,
    seal_empirical_workload,
)
from sim.scenarios.dev_medium import medium_cluster
from sim.scenarios.sources import load_source_registry


def test_sealed_loader_revalidates_all_fits_scans_and_parent_chain():
    evidence = load_sealed_alibaba_calibration()

    assert len(evidence.rt_fits) == 3734
    assert len({(fit.service, fit.metric) for fit in evidence.rt_fits}) == 3734
    assert len(evidence.calls) == 29933
    assert len({call.trace_id for call in evidence.calls}) == 991
    assert evidence.trace_sample_seed == 6399959053302358801
    assert evidence.rate_support_end_s == 1800.0
    assert len(evidence.rate_points) == 30


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (CALIBRATION_PATH, "calibration report exact-byte"),
        (CALL_GRAPH_SAMPLE_PATH, "call-graph sample exact-byte"),
    ),
)
def test_exact_export_byte_tampering_fails_closed(tmp_path, source, message):
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes() + b" ")
    kwargs = {
        "calibration_path": target if source == CALIBRATION_PATH else CALIBRATION_PATH,
        "call_graph_sample_path": (
            target if source == CALL_GRAPH_SAMPLE_PATH else CALL_GRAPH_SAMPLE_PATH
        ),
    }
    with pytest.raises(ValueError, match=message):
        load_sealed_alibaba_calibration(**kwargs)


def test_an_individual_rt_fit_seal_is_recomputed_not_merely_trusted():
    calibration = json.loads(CALIBRATION_PATH.read_text())
    calibration["rt_fits"][0]["seal"] = "0" * 64

    with pytest.raises(ValueError, match="RT-fit seal mismatch"):
        _verify_rt_fits(calibration)


def test_builder_is_deterministic_for_one_seed_and_seed_sensitive():
    registry = load_source_registry()
    config = AlibabaWorkloadBuildConfig(seed=20260817)
    first = build_calibration_informed_workload(config, registry)
    second = build_calibration_informed_workload(config, registry)
    other = build_calibration_informed_workload(
        dataclasses.replace(config, seed=20260818), registry
    )

    assert first == second
    assert first.digest == second.digest
    assert first.selected_templates == second.selected_templates
    assert other.digest != first.digest
    assert other.selected_templates != first.selected_templates


def test_builder_emits_dense_multi_template_fan_out_specs_from_selected_fits():
    result = build_calibration_informed_workload(AlibabaWorkloadBuildConfig(seed=20260817))
    workload = result.workload

    assert [service.id for service in workload.services] == list(range(len(workload.services)))
    assert 3 <= len(workload.services) <= 8
    assert len(workload.templates) == 3
    assert all(service.service_time.kind == "lognormal" for service in workload.services)
    assert all(
        service.category == "calibration-informed-development"
        for service in workload.services
    )
    assert all(template.arrival.kind == "poisson" for template in workload.templates)
    assert all(template.arrival.rate_schedule[-1][1] == 0.0 for template in workload.templates)
    assert len(result.selected_fit_seals) == len(workload.services)

    def shape(call, depth=1):
        depths = [depth]
        fanouts = [len(call.children)]
        for edge in call.children:
            child_depths, child_fanouts = shape(edge.child, depth + 1)
            depths.extend(child_depths)
            fanouts.extend(child_fanouts)
        return depths, fanouts

    shapes = [shape(template.root) for template in workload.templates]
    assert all(max(depths) >= 3 for depths, _ in shapes)
    assert all(max(fanouts) >= 2 for _, fanouts in shapes)


def test_calibration_informed_sealer_requires_exact_typed_builder_workload():
    registry = load_source_registry()
    build = build_calibration_informed_workload(
        AlibabaWorkloadBuildConfig(seed=20260817), registry
    )
    episode = dataclasses.replace(
        medium_cluster(), workload=build.workload, arrival_cutoff_s=53.0
    )
    closure = seal_calibration_informed_synthetic_workload(
        episode, registry, build=build
    )

    assert closure.provenance_label == "calibration-informed-synthetic-workload"
    assert {item.kind for item in closure.evidence} >= {
        "absolute-root-rate-shape",
        "call-graph-sample",
        "interval-mean-rt-fit",
    }
    with pytest.raises(ValueError, match="was not produced"):
        seal_calibration_informed_synthetic_workload(
            medium_cluster(), registry, build=build
        )
    with pytest.raises(ValueError, match="empirical workload sealing is unavailable"):
        seal_empirical_workload(episode, registry)
