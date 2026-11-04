"""G3 protocol, access, residual, and unavailable-evidence behavior."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess

import pytest
from sim_helpers import (
    simple_service,
    single_service_config,
    two_edge_cluster,
    v2021_dump_bytes,
)

import sim.scenarios.g3 as g3_module
import sim.scenarios.sources as source_registry_module
from sim.scenarios.alibaba import (
    fit_response_time_targets,
    parse_v2021_metrics,
    parse_v2022_metrics,
)
from sim.scenarios.artifacts import (
    seal_evidence_citing_synthetic_workload,
    seal_synthetic_workload,
)
from sim.scenarios.g3 import (
    G3AccessGate,
    _load_p1_registration_receipt_fixture_bytes,
    access_receipt_from_json,
    access_receipt_to_json,
    aggregate_instance_interval_mean_rt,
    curve_residuals,
    freeze_curve_prediction,
    freeze_protocol,
    freeze_rt_prediction,
    g3_protocol_spec_sha256,
    make_published_curve,
    prediction_from_json,
    prediction_to_json,
    protocol_from_json,
    protocol_to_json,
    published_curve_selector_sha256,
    result_from_json,
    result_from_residuals,
    result_to_json,
    rt_quantile_residuals,
    rt_shape_evidence,
    rt_shape_selector_sha256,
    run_g3_prediction_episode,
    unavailable_result,
)
from sim.scenarios.sources import CALIBRATION, DATA_PATH, FACE_VALIDITY, load_source_registry
from sim.spec import (
    ArrivalSpec,
    CallEdgeSpec,
    CallSpec,
    EpisodeConfig,
    TemplateSpec,
    WorkloadSpec,
)

TEST_CALIBRATION_PAYLOAD = v2021_dump_bytes(
    "timestamp,msname,msinstanceid,metrics,value\n"
    + "".join(
        f"{index},CAL,p0,providerRPC_RT,{1.0 + index}\n"
        f"{index},CAL,p0,providerRPC_MCR,1.0\n"
        for index in range(5)
    )
)


@pytest.fixture(autouse=True)
def _test_predictions_use_explicit_dirty_checkout_bypass(monkeypatch):
    """Production prediction freezing requires clean HEAD; tests exercise a dirty tree."""
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(g3_module, "clean_checkout_commit", lambda: checkout)


def _available_registry(tmp_path, payloads: dict[str, bytes]):
    payloads = {
        "alibaba-microservices-v2021": TEST_CALIBRATION_PAYLOAD,
        **payloads,
    }
    raw = json.loads(DATA_PATH.read_text())
    for source in raw["sources"]:
        if source["source_id"] in payloads:
            # Fixtures exercise the single-payload contract; drop any registered
            # component list so one fixture digest cannot masquerade as a component.
            source["payload_components"] = []
            source["payload_status"] = "AVAILABLE"
            source["payload_sha256"] = hashlib.sha256(
                payloads[source["source_id"]]
            ).hexdigest()
    payload = json.dumps(raw).encode()
    (tmp_path / "source-split.json").write_bytes(payload)
    return source_registry_module._load_source_registry_bytes(payload)


def _protocol(registry=None, evidence_selectors=(), predictions=()):
    registry = load_source_registry() if registry is None else registry
    spec_sha256 = g3_protocol_spec_sha256(
        source_split_sha256=registry.manifest_sha256(),
        rt_quantiles=(0.5, 0.75, 0.9, 0.95, 0.99),
        curve_log_ratio_abs_band=0.2,
        rt_log_quantile_abs_band=0.25,
        evidence_selectors=evidence_selectors,
        predictions=predictions,
    )
    receipt = json.dumps(
        {
            "schema": "g3-p1-registration-receipt-v1",
            "receipt_id": "external-test-receipt-001",
            "registration_id": "P1-draft-test",
            "authority_url": "https://osf.example.invalid/registrations/test",
            "frozen_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "registered_at_utc": "2026-07-17T10:00:00+00:00",
            "protocol_spec_sha256": spec_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    registration_receipt = _load_p1_registration_receipt_fixture_bytes(
        receipt, spec_sha256
    )
    protocol = freeze_protocol(
        registration_receipt=registration_receipt,
        source_split_sha256=registry.manifest_sha256(),
        rt_quantiles=(0.5, 0.75, 0.9, 0.95, 0.99),
        curve_log_ratio_abs_band=0.2,
        rt_log_quantile_abs_band=0.25,
        evidence_selectors=evidence_selectors,
        predictions=predictions,
    )
    return registry, protocol, registration_receipt


def _prediction_run(
    registry,
    *,
    service: str,
    rate: float,
    seed: int,
    mean_s: float = 0.02,
    calibration_fit=None,
):
    config = single_service_config(
        two_edge_cluster(),
        rate=rate,
        horizon_s=240.0,
        service_kwargs={"name": service, "mean_s": mean_s},
        arrival_cutoff_s=180.0,
    )
    if calibration_fit is None:
        model = seal_synthetic_workload(
            config, registry, generator_version="test-g3-workload-v1"
        )
    else:
        model = seal_evidence_citing_synthetic_workload(
            config,
            registry,
            rt_fits=(calibration_fit,),
            transform_config=(("test-transform", "rt-fit-only"),),
            code_versions=("test-g3-evidence-citing-v1",),
        )
    return run_g3_prediction_episode(
        config=config,
        model=model,
        registry=registry,
        seed=seed,
    )


def _calibration_fit(registry, payload=TEST_CALIBRATION_PAYLOAD):
    corpus = parse_v2021_metrics(payload)
    service = next(obs.service for obs in corpus.observations if obs.metric.endswith("_RT"))
    return fit_response_time_targets(
        corpus, registry, service, "PROVIDERRPC_RT", minimum_count=2
    )


def _provider_prediction_run(
    registry,
    *,
    service,
    rate,
    seed,
    calibration_fit=None,
    root_kwargs=None,
):
    root_config = {"mean_s": 0.005, **(root_kwargs or {})}
    root = simple_service(0, "front", **root_config)
    provider = simple_service(1, service, 0.02)
    config = EpisodeConfig(
        cluster=two_edge_cluster(),
        workload=WorkloadSpec(
            services=(root, provider),
            templates=(
                TemplateSpec(
                    id=0,
                    name="provider-rpc",
                    root=CallSpec(
                        service=0,
                        children=(CallEdgeSpec(child=CallSpec(service=1)),),
                    ),
                    arrival=ArrivalSpec("poisson", rate),
                ),
            ),
        ),
        horizon_s=240.0,
        arrival_cutoff_s=180.0,
    )
    if calibration_fit is None:
        model = seal_synthetic_workload(
            config, registry, generator_version="test-g3-provider-workload-v1"
        )
    else:
        model = seal_evidence_citing_synthetic_workload(
            config,
            registry,
            rt_fits=(calibration_fit,),
            transform_config=(("test-transform", "rt-fit-only"),),
            code_versions=("test-g3-evidence-citing-v1",),
        )
    return run_g3_prediction_episode(
        config=config,
        model=model,
        registry=registry,
        seed=seed,
    )


def _curve_runs(
    registry,
    rates=(1.0, 2.0),
    mean_s=0.02,
    calibration_payload=TEST_CALIBRATION_PAYLOAD,
):
    calibration_fit = _calibration_fit(registry, calibration_payload)
    return tuple(
            _prediction_run(
                registry,
                service="curve-service",
                rate=rate,
                seed=700 + index,
                mean_s=mean_s,
                calibration_fit=calibration_fit,
            )
            for index, rate in enumerate(rates)
    )


def _curve_prediction(
    registry,
    source_id,
    selector,
    rates=(1.0, 2.0),
    mean_s=0.02,
    calibration_payload=TEST_CALIBRATION_PAYLOAD,
):
    return freeze_curve_prediction(
        registry=registry,
        source_id=source_id,
        selector_sha256=selector,
        x_unit="requests/s",
        y_unit="ms",
        runs=_curve_runs(registry, rates, mean_s, calibration_payload),
    )


def _rt_prediction(
    registry,
    source_id,
    service,
    metric="PROVIDERRPC_RT",
    calibration_payload=TEST_CALIBRATION_PAYLOAD,
):
    calibration_fit = _calibration_fit(registry, calibration_payload)
    return freeze_rt_prediction(
        registry=registry,
        source_id=source_id,
        service=service,
        metric=metric,
        runs=tuple(
            _provider_prediction_run(
                registry,
                service=service,
                rate=1.0,
                seed=811 + index,
                calibration_fit=calibration_fit,
            )
            for index in range(10)
        ),
    )


def _preliminary_rt_values(service):
    registry = load_source_registry()
    samples = []
    for index in range(10):
        run = _provider_prediction_run(
            registry,
            service=service,
            rate=1.0,
            seed=811 + index,
        )
        samples.extend(
            (
                call.start_s,
                f"run-{index}:pod-{call.pod_uid}",
                call.latency_s * 1000.0,
            )
            for call in run.result.service_calls
            if call.service == 1
            and 60.0 <= call.start_s < 180.0
            and call.end_s is not None
        )
    return aggregate_instance_interval_mean_rt(tuple(samples))


def _published_curve(paper_content, **fields):
    manifest = {"schema": "published-curve-extraction-v1", **fields}
    return make_published_curve(
        paper_content=paper_content,
        extraction_manifest_content=json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode(),
    )


def test_missing_empirical_payloads_are_unavailable_and_never_pass():
    registry, protocol, _ = _protocol()
    result = unavailable_result(protocol, registry)
    assert result.status == "UNAVAILABLE"
    assert "teastore-published-curves" in result.unavailable_sources
    assert result.residuals == ()
    with pytest.raises(ValueError, match="cannot pass"):
        dataclasses.replace(result, status="PASS").verify()
    with pytest.raises(ValueError, match="external P1 receipt"):
        dataclasses.replace(protocol, _authority=object()).verify()
    with pytest.raises(ValueError, match="contextual evaluation"):
        dataclasses.replace(result, _authority=object()).verify()


def test_prediction_freezer_rejects_caller_values_and_mutated_results(tmp_path):
    paper = b"prediction-boundary-test"
    registry = _available_registry(tmp_path, {"deathstarbench-hotel-publication": paper})
    selector = published_curve_selector_sha256(
        source_id="deathstarbench-hotel-publication",
        page="1",
        figure_or_table="Figure 1",
        series_label="p95",
        x_unit="requests/s",
        y_unit="ms",
        extraction_method="test",
        extraction_tool_version="v1",
        extraction_code_sha256="a" * 64,
        configuration="test",
        unknown_configuration="none",
    )
    with pytest.raises(TypeError):
        freeze_curve_prediction(b'{"curve_points":[[1,1],[2,2]]}')  # type: ignore[misc]
    run = _prediction_run(
        registry,
        service="curve-service",
        rate=1.0,
        seed=999,
        calibration_fit=_calibration_fit(registry),
    )
    run.result.records[0].status = "forged"
    with pytest.raises(ValueError, match="changed after"):
        freeze_curve_prediction(
            registry=registry,
            source_id="deathstarbench-hotel-publication",
            selector_sha256=selector,
            x_unit="requests/s",
            y_unit="ms",
            runs=(run,),
        )


def test_rt_prediction_rejects_survivor_only_overload(tmp_path):
    registry = _available_registry(tmp_path, {})
    fit = _calibration_fit(registry)
    run = _provider_prediction_run(
        registry,
        service="MS_1",
        rate=5.0,
        seed=1001,
        calibration_fit=fit,
        root_kwargs={
            "mean_s": 1.0,
            "queue_capacity": 1,
            "concurrency": 1,
        },
    )
    assert any(record.status != "completed" for record in run.result.records)
    with pytest.raises(ValueError, match="every root request"):
        freeze_rt_prediction(
            registry=registry,
            source_id="alibaba-microservices-v2021",
            service="MS_1",
            metric="PROVIDERRPC_RT",
            runs=(run,),
        )


def test_rt_prediction_rejects_duplicate_runs_in_sample_count(tmp_path):
    registry = _available_registry(tmp_path, {})
    run = _provider_prediction_run(
        registry,
        service="MS_1",
        rate=1.0,
        seed=1002,
        calibration_fit=_calibration_fit(registry),
    )
    with pytest.raises(ValueError, match="distinct retained episode results"):
        freeze_rt_prediction(
            registry=registry,
            source_id="alibaba-microservices-v2021",
            service="MS_1",
            metric="PROVIDERRPC_RT",
            runs=(run,) * 10,
        )


def test_published_curve_requires_provenance_and_residuals_do_not_extrapolate(tmp_path):
    paper = b"exact DSB publication test bytes"
    registry = _available_registry(tmp_path, {"deathstarbench-hotel-publication": paper})
    selector = published_curve_selector_sha256(
        source_id="deathstarbench-hotel-publication",
        page="7",
        figure_or_table="Figure 4",
        series_label="p95 latency",
        x_unit="requests/s",
        y_unit="ms",
        extraction_method="digitized",
        extraction_tool_version="test-v1",
        extraction_code_sha256="a" * 64,
        configuration="registered test configuration",
        unknown_configuration="none",
    )
    runs = _curve_runs(registry, rates=(0.5, 1.0))
    prediction = freeze_curve_prediction(
        registry=registry,
        source_id="deathstarbench-hotel-publication",
        selector_sha256=selector,
        x_unit="requests/s",
        y_unit="ms",
        runs=runs,
    )
    assert prediction_from_json(
        prediction_to_json(prediction), registry=registry, runs=runs
    ) == prediction
    _, protocol, _ = _protocol(
        registry,
        (("deathstarbench-hotel-publication", selector),),
        (prediction,),
    )
    curve = _published_curve(
        paper,
        source_id="deathstarbench-hotel-publication",
        doi_or_url="https://example.invalid/paper",
        page="7",
        figure_or_table="Figure 4",
        series_label="p95 latency",
        x_unit="requests/s",
        y_unit="ms",
        points=prediction.curve_points,
        extraction_method="digitized",
        extraction_tool_version="test-v1",
        extraction_code_sha256="a" * 64,
        point_selection_rule="all-visible-points-in-selected-series",
        configuration="registered test configuration",
        unknown_configuration="none",
    )
    residuals = curve_residuals(
        curve,
        prediction,
        protocol,
        registry,
        CALIBRATION,
    )
    assert len(residuals) == 2
    narrow_prediction = _curve_prediction(
        registry,
        "deathstarbench-hotel-publication",
        selector,
        rates=(0.6, 0.9),
    )
    _, narrow_protocol, _ = _protocol(
        registry,
        (("deathstarbench-hotel-publication", selector),),
        (narrow_prediction,),
    )
    with pytest.raises(ValueError, match="extrapolation"):
        curve_residuals(
            curve,
            narrow_prediction,
            narrow_protocol,
            registry,
            CALIBRATION,
        )
    with pytest.raises(ValueError, match="page"):
        _published_curve(
            paper,
            source_id="teastore-published-curves",
            doi_or_url="x",
            page="",
            figure_or_table="Figure 4",
            series_label="p95",
            x_unit="requests/s",
            y_unit="ms",
            points=((1.0, 1.0), (2.0, 2.0)),
            extraction_method="digitized",
            extraction_tool_version="v1",
            extraction_code_sha256="a" * 64,
            point_selection_rule="all-visible-points-in-selected-series",
            configuration="known",
            unknown_configuration="none",
        )


def test_rt_shape_residuals_are_median_normalized_on_frozen_grid(tmp_path):
    preliminary_values = _preliminary_rt_values("MS_1")
    payload = v2021_dump_bytes(
        "timestamp,msname,msinstanceid,metrics,value\n"
        + "".join(
            f"{index},MS_1,POD_1,providerRPC_RT,{value}\n"
            f"{index},MS_1,POD_1,providerRPC_MCR,1.0\n"
            for index, value in enumerate(preliminary_values)
        )
    )
    corpus = parse_v2021_metrics(payload)
    registry = _available_registry(tmp_path, {"alibaba-microservices-v2021": payload})
    selector = rt_shape_selector_sha256(
        source_id="alibaba-microservices-v2021",
        service="MS_1",
        metric="PROVIDERRPC_RT",
    )
    prediction = _rt_prediction(
        registry,
        "alibaba-microservices-v2021",
        "MS_1",
        calibration_payload=payload,
    )
    _, protocol, _ = _protocol(
        registry, (("alibaba-microservices-v2021", selector),), (prediction,)
    )
    residuals = rt_quantile_residuals(
        corpus=corpus,
        registry=registry,
        service="MS_1",
        metric="PROVIDERRPC_RT",
        prediction=prediction,
        protocol=protocol,
    )
    assert tuple(item.coordinate for item in residuals) == protocol.rt_quantiles
    assert all(item.residual == pytest.approx(0.0) for item in residuals)
    assert aggregate_instance_interval_mean_rt(
        ((1.0, "pod-a", 2.0), (2.0, "pod-a", 4.0), (61.0, "pod-a", 6.0))
    ) == (3.0, 6.0)
    with pytest.raises(ValueError, match="at least two populated"):
        aggregate_instance_interval_mean_rt(
            ((1.0, "pod-a", 2.0), (2.0, "pod-a", 4.0))
        )


def test_rt_shape_retains_left_censored_mass_with_frozen_substitution(tmp_path):
    values = (0.0, *_preliminary_rt_values("MS_1"))
    payload = v2021_dump_bytes(
        "timestamp,msname,msinstanceid,metrics,value\n"
        + "".join(
            f"{index},MS_1,POD_1,providerRPC_RT,{value}\n"
            f"{index},MS_1,POD_1,providerRPC_MCR,1.0\n"
            for index, value in enumerate(values)
        )
    )
    registry = _available_registry(tmp_path, {"alibaba-microservices-v2021": payload})
    selector = rt_shape_selector_sha256(
        source_id="alibaba-microservices-v2021",
        service="MS_1",
        metric="PROVIDERRPC_RT",
    )
    prediction = _rt_prediction(
        registry,
        "alibaba-microservices-v2021",
        "MS_1",
        calibration_payload=payload,
    )
    _, protocol, _ = _protocol(
        registry, (("alibaba-microservices-v2021", selector),), (prediction,)
    )
    observed, _ = rt_shape_evidence(
        corpus=parse_v2021_metrics(payload),
        registry=registry,
        service="MS_1",
        metric="PROVIDERRPC_RT",
        protocol=protocol,
    )
    assert observed[0] == 0.5
    assert len(observed) == len(values)


def test_g3_marker_is_created_before_source_return_and_is_run_once(tmp_path, monkeypatch):
    payloads = {
        "alibaba-microservices-v2022": b"held-out Alibaba bytes",
        "teastore-published-curves": b"held-out TeaStore bytes",
    }
    registry = _available_registry(tmp_path, payloads)
    registry, protocol, _ = _protocol(registry)
    marker = tmp_path / "g3-opened.json"
    monkeypatch.setattr(g3_module, "require_frozen_clean_checkout", lambda commit: commit)
    gate = G3AccessGate(protocol, registry, marker)
    load_order = []

    def loader(source_id):
        assert marker.is_file()
        load_order.append(source_id)
        return payloads[source_id]

    opened, receipt = gate.open_all_face_validity(loader)
    assert {source_id for source_id, _ in opened} == {
        "alibaba-microservices-v2022",
        "teastore-published-curves",
    }
    assert tuple(load_order) == receipt.face_validity_source_ids
    assert marker.is_file()
    receipt.verify(protocol, registry, marker)
    with pytest.raises(RuntimeError, match="run once"):
        gate.open_all_face_validity(loader)
    with pytest.raises(FileExistsError):
        G3AccessGate(protocol, registry, marker).open_all_face_validity(loader)


def test_available_g3_path_is_source_bound_complete_and_recomputes_residuals(
    tmp_path, monkeypatch
):
    # The empirical fixture values are chosen from an independent preliminary run;
    # the prediction artifacts below are regenerated under the final source registry.
    preliminary_v1 = _preliminary_rt_values("A")
    preliminary_v2 = _preliminary_rt_values("MS_1")
    v1_payload = v2021_dump_bytes(
        "timestamp,msname,msinstanceid,metrics,value\n"
        + "".join(
            f"{index},A,p0,providerRPC_RT,{value}\n"
            f"{index},A,p0,providerRPC_MCR,1.0\n"
            for index, value in enumerate(preliminary_v1)
        )
    )
    v2_payload = (
        "timestamp,msname,msinstanceid,nodeid,providerrpc_rt,providerrpc_mcr\n"
        + "".join(
            f"{index},MS_1,POD_1,NODE_1,{value},0.5\n"
            for index, value in enumerate(preliminary_v2)
        )
    ).encode()
    dsb_paper = b"exact DSB publication bytes"
    tea_paper = b"exact TeaStore publication bytes"
    payloads = {
        "alibaba-microservices-v2021": v1_payload,
        "alibaba-microservices-v2022": v2_payload,
        "deathstarbench-hotel-publication": dsb_paper,
        "teastore-published-curves": tea_paper,
    }
    registry = _available_registry(tmp_path, payloads)
    selectors = (
        (
            "alibaba-microservices-v2021",
            rt_shape_selector_sha256(
                source_id="alibaba-microservices-v2021",
                service="A",
                metric="PROVIDERRPC_RT",
            ),
        ),
        (
            "alibaba-microservices-v2022",
            rt_shape_selector_sha256(
                source_id="alibaba-microservices-v2022",
                service="MS_1",
                metric="PROVIDERRPC_RT",
            ),
        ),
        (
            "deathstarbench-hotel-publication",
            published_curve_selector_sha256(
                source_id="deathstarbench-hotel-publication",
                page="1",
                figure_or_table="Figure test",
                series_label="p95",
                x_unit="requests/s",
                y_unit="ms",
                extraction_method="test fixture extraction",
                extraction_tool_version="test-v1",
                extraction_code_sha256="a" * 64,
                configuration="synthetic positive-path fixture",
                unknown_configuration="none",
            ),
        ),
        (
            "teastore-published-curves",
            published_curve_selector_sha256(
                source_id="teastore-published-curves",
                page="2",
                figure_or_table="Figure test",
                series_label="p95",
                x_unit="requests/s",
                y_unit="ms",
                extraction_method="test fixture extraction",
                extraction_tool_version="test-v1",
                extraction_code_sha256="a" * 64,
                configuration="synthetic positive-path fixture",
                unknown_configuration="none",
            ),
        ),
    )
    selector_map = dict(selectors)
    v1_prediction = _rt_prediction(
        registry, "alibaba-microservices-v2021", "A", calibration_payload=v1_payload
    )
    v2_prediction = _rt_prediction(
        registry, "alibaba-microservices-v2022", "MS_1", calibration_payload=v1_payload
    )
    dsb_prediction = _curve_prediction(
        registry,
        "deathstarbench-hotel-publication",
        selector_map["deathstarbench-hotel-publication"],
        calibration_payload=v1_payload,
    )
    tea_prediction = _curve_prediction(
        registry,
        "teastore-published-curves",
        selector_map["teastore-published-curves"],
        calibration_payload=v1_payload,
    )
    dsb = _published_curve(
        dsb_paper,
        source_id="deathstarbench-hotel-publication",
        doi_or_url="https://example.invalid/dsb",
        page="1",
        figure_or_table="Figure test",
        series_label="p95",
        x_unit="requests/s",
        y_unit="ms",
        points=dsb_prediction.curve_points,
        extraction_method="test fixture extraction",
        extraction_tool_version="test-v1",
        extraction_code_sha256="a" * 64,
        point_selection_rule="all-visible-points-in-selected-series",
        configuration="synthetic positive-path fixture",
        unknown_configuration="none",
    )
    predictions = (v1_prediction, v2_prediction, dsb_prediction, tea_prediction)
    _, protocol, registration_receipt = _protocol(registry, selectors, predictions)
    with pytest.raises(PermissionError, match="before the P1 access gate"):
        rt_quantile_residuals(
            corpus=parse_v2022_metrics(v2_payload),
            registry=registry,
            service="MS_1",
            metric="PROVIDERRPC_RT",
            prediction=v2_prediction,
            protocol=protocol,
        )
    monkeypatch.setattr(g3_module, "require_frozen_clean_checkout", lambda commit: commit)
    marker = tmp_path / "g3-positive-opened.json"
    opened_payloads, receipt = G3AccessGate(
        protocol, registry, marker
    ).open_all_face_validity(lambda source_id: payloads[source_id])
    opened_payloads = dict(opened_payloads)
    v1 = parse_v2021_metrics(v1_payload)
    v2 = parse_v2022_metrics(opened_payloads["alibaba-microservices-v2022"])
    tea = _published_curve(
        opened_payloads["teastore-published-curves"],
        source_id="teastore-published-curves",
        doi_or_url="https://example.invalid/tea",
        page="2",
        figure_or_table="Figure test",
        series_label="p95",
        x_unit="requests/s",
        y_unit="ms",
        points=tea_prediction.curve_points,
        extraction_method="test fixture extraction",
        extraction_tool_version="test-v1",
        extraction_code_sha256="a" * 64,
        point_selection_rule="all-visible-points-in-selected-series",
        configuration="synthetic positive-path fixture",
        unknown_configuration="none",
    )
    residuals = (
        *rt_quantile_residuals(
            corpus=v1,
            registry=registry,
            service="A",
            metric="PROVIDERRPC_RT",
            prediction=v1_prediction,
            protocol=protocol,
        ),
        *rt_quantile_residuals(
            corpus=v2,
            registry=registry,
            service="MS_1",
            metric="PROVIDERRPC_RT",
            prediction=v2_prediction,
            protocol=protocol,
            access_receipt=receipt,
            access_marker=marker,
        ),
        *curve_residuals(dsb, dsb_prediction, protocol, registry, CALIBRATION),
        *curve_residuals(
            tea,
            tea_prediction,
            protocol,
            registry,
            FACE_VALIDITY,
            receipt,
            marker,
        ),
    )
    result = result_from_residuals(protocol, registry, residuals, receipt, marker)
    assert result.status == "PASS"
    assert protocol_from_json(
        protocol_to_json(protocol),
        registration_receipt=registration_receipt,
        predictions=predictions,
    ) == protocol
    reloaded_receipt = access_receipt_from_json(
        access_receipt_to_json(receipt), protocol, registry, marker
    )
    assert reloaded_receipt == receipt
    assert result_from_json(
        result_to_json(result),
        protocol=protocol,
        registry=registry,
        access_receipt=receipt,
        marker=marker,
        expected_residuals=residuals,
    ) == result
    with pytest.raises(ValueError, match="source-evaluator-issued residuals"):
        result_from_json(
            result_to_json(result),
            protocol=protocol,
            registry=registry,
            access_receipt=receipt,
            marker=marker,
        )
    tampered = (dataclasses.replace(residuals[0], within_band=False), *residuals[1:])
    with pytest.raises(ValueError, match="within_band|seal"):
        result_from_residuals(protocol, registry, tampered, receipt, marker)

    # A genuinely executed, overloaded prediction fails against the same evidence.
    bad_tea_prediction = _curve_prediction(
        registry,
        "teastore-published-curves",
        selector_map["teastore-published-curves"],
        mean_s=0.05,
        calibration_payload=v1_payload,
    )
    bad_predictions = (v1_prediction, v2_prediction, dsb_prediction, bad_tea_prediction)
    _, bad_protocol, _ = _protocol(registry, selectors, bad_predictions)
    bad_marker = tmp_path / "g3-negative-opened.json"
    _, bad_receipt = G3AccessGate(
        bad_protocol, registry, bad_marker
    ).open_all_face_validity(lambda source_id: payloads[source_id])
    bad_tea = curve_residuals(
        tea,
        bad_tea_prediction,
        bad_protocol,
        registry,
        FACE_VALIDITY,
        bad_receipt,
        bad_marker,
    )
    failing = (*residuals[:-2], *bad_tea)
    assert result_from_residuals(
        bad_protocol, registry, failing, bad_receipt, bad_marker
    ).status == "FAIL"
    with pytest.raises(ValueError, match="omit or duplicate"):
        result_from_residuals(
            bad_protocol, registry, failing[:-1], bad_receipt, bad_marker
        )
