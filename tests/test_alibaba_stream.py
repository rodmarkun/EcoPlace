"""Component-aware source registration and memory-lean complete-member ingestion.

Two properties carry the Phase-4 calibration argument and are asserted here directly:

* one ``payload_sha256`` cannot authenticate two distinct parser inputs, so each corpus
  component authenticates itself and cross-component substitution fails closed;
* the streaming path is a *memory* optimisation only — on identical bytes it must produce
  the identical sealed artifacts as the whole-payload parsers, because a path that
  quietly parsed less would reproduce the 2026-07-21 truncation failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from sim_helpers import v2021_dump_bytes

import sim.scenarios.sources as source_registry_module
from sim.scenarios.alibaba import (
    CALL_GRAPH_COMPONENT,
    METRICS_COMPONENT,
    V2021,
    fit_response_time_targets,
    parse_v2021_call_graph,
    parse_v2021_metrics,
    sample_call_graphs,
)
from sim.scenarios.alibaba_stream import (
    scan_v2021_call_graph,
    scan_v2021_metrics,
)
from sim.scenarios.sources import (
    CALIBRATION,
    DATA_PATH,
    FACE_VALIDITY,
    PayloadComponent,
    load_source_registry,
)

METRIC_MEMBER = "MSRTQps_0.csv"
CALL_MEMBER = "MSCallGraph_0.csv"

REGISTERED_METRIC_SHA256 = "42514bcbbc4451ae66722a54a8a299aa33319c82bb4054dc91b197961e5bbfcf"
REGISTERED_CALL_GRAPH_SHA256 = "3fcaa9a4181649842534403a08cbcf7ab904cd537d6518fce832bf9f4f0a1e73"


def _tar_gz(path: Path, member: str, payload: bytes) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return path


def _component_registry(tmp_path: Path, components: dict[str, dict[str, object]]):
    """Rebuild the production split with v2021's components re-pointed at fixtures."""
    raw = json.loads(DATA_PATH.read_text())
    for source in raw["sources"]:
        if source["source_id"] != V2021:
            continue
        for item in source["payload_components"]:
            override = components.get(item["component"])
            if override is not None:
                item.update(override)
    payload = json.dumps(raw).encode()
    (tmp_path / "source-split.json").write_bytes(payload)
    return source_registry_module._load_source_registry_bytes(payload)


def _metric_fixture() -> bytes:
    rows = "".join(
        f"{index * 60000},svc-a,inst-{index % 3},providerRPC_RT,{0 if index < 4 else 3 + index}\n"
        f"{index * 60000},svc-a,inst-{index % 3},providerRPC_MCR,{2 + index}\n"
        for index in range(26)
    )
    # A second service that fails the activity rule: no MCR partner at all.
    rows += "".join(f"{index * 60000},svc-b,inst-9,HTTP_RT,5\n" for index in range(26))
    return v2021_dump_bytes("timestamp,msname,msinstanceid,metrics,value\n" + rows)


def _call_graph_fixture() -> bytes:
    rows = []
    for trace in range(40):
        rows.append(f"{trace},t{trace:03d},0,,rpc,entry,front,{5 + trace}\n")
        rows.append(f"{trace},t{trace:03d},0,,rpc,entry,front,-{3 + trace}\n")
        rows.append(f"{trace},t{trace:03d},0.1,front,mc,,cache,{1 + trace}\n")
    return v2021_dump_bytes(
        "timestamp,traceid,rpcid,um,rpctype,interface,dm,rt\n" + "".join(rows)
    )


# --------------------------------------------------------------------------- registry


def test_production_split_registers_both_chunk0_components_with_their_own_digests():
    registry = load_source_registry()
    source = registry.get(V2021)
    assert source.payload_status == "AVAILABLE"
    # The single-hash field stays empty: it cannot authenticate two parser inputs.
    assert source.payload_sha256 is None
    assert [item.component for item in source.payload_components] == [
        CALL_GRAPH_COMPONENT,
        METRICS_COMPONENT,
    ]
    assert source.component(METRICS_COMPONENT).sha256 == REGISTERED_METRIC_SHA256
    assert source.component(METRICS_COMPONENT).parser_input == METRIC_MEMBER
    assert source.component(CALL_GRAPH_COMPONENT).sha256 == REGISTERED_CALL_GRAPH_SHA256
    assert source.component(CALL_GRAPH_COMPONENT).parser_input == CALL_MEMBER
    assert set(source.authorized_digests()) == {
        REGISTERED_METRIC_SHA256,
        REGISTERED_CALL_GRAPH_SHA256,
    }
    assert "adopted 2026-08-09 by researcher direction" in source.notes
    assert "calibration_corpus_memo.md recommendation" in source.notes


def test_component_binding_refuses_cross_component_substitution():
    registry = load_source_registry()
    registry.require_component(V2021, CALIBRATION, METRICS_COMPONENT, REGISTERED_METRIC_SHA256)
    registry.require_component(
        V2021, CALIBRATION, CALL_GRAPH_COMPONENT, REGISTERED_CALL_GRAPH_SHA256
    )
    # The exact failure the memo flagged: one digest authorizing the wrong parser input.
    with pytest.raises(ValueError, match="hash mismatch for source component"):
        registry.require_component(
            V2021, CALIBRATION, METRICS_COMPONENT, REGISTERED_CALL_GRAPH_SHA256
        )
    with pytest.raises(ValueError, match="hash mismatch for source component"):
        registry.require_component(
            V2021, CALIBRATION, CALL_GRAPH_COMPONENT, REGISTERED_METRIC_SHA256
        )
    with pytest.raises(ValueError, match="registers no payload component"):
        registry.require_component(V2021, CALIBRATION, "resource", REGISTERED_METRIC_SHA256)
    # require_payload stays deliberately looser: any registered component digest.
    registry.require_payload(V2021, CALIBRATION, REGISTERED_METRIC_SHA256)
    registry.require_payload(V2021, CALIBRATION, REGISTERED_CALL_GRAPH_SHA256)
    with pytest.raises(ValueError, match="payload hash mismatch"):
        registry.require_payload(V2021, CALIBRATION, "0" * 64)


def test_component_invariants_fail_closed():
    registry = load_source_registry()
    source = registry.get(V2021)
    metrics = source.component(METRICS_COMPONENT)
    with pytest.raises(ValueError, match="must not also advertise a single payload hash"):
        dataclasses.replace(source, payload_sha256="a" * 64)
    with pytest.raises(ValueError, match="unique and canonically sorted"):
        dataclasses.replace(source, payload_components=(metrics, metrics))
    with pytest.raises(ValueError, match="AVAILABLE only when every"):
        dataclasses.replace(
            source,
            payload_components=(
                dataclasses.replace(
                    source.component(CALL_GRAPH_COMPONENT),
                    status="UNAVAILABLE",
                    sha256=None,
                    bytes=None,
                ),
                metrics,
            ),
        )
    with pytest.raises(ValueError, match="needs its parser-input SHA-256"):
        dataclasses.replace(metrics, sha256=None)
    with pytest.raises(ValueError, match="needs a positive byte count"):
        dataclasses.replace(metrics, bytes=0)
    with pytest.raises(ValueError, match="may not advertise bytes or a hash"):
        dataclasses.replace(metrics, status="UNAVAILABLE")
    with pytest.raises(ValueError, match="archive_path and archive_sha256"):
        dataclasses.replace(metrics, archive_sha256=None)


def test_component_archives_are_reverified_against_their_registered_digest(tmp_path):
    payload = b"not the retained archive"
    archive = tmp_path / "fixture.tar.gz"
    archive.write_bytes(payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "archive_path": "fixture.tar.gz",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
            },
            CALL_GRAPH_COMPONENT: {
                "archive_path": "fixture.tar.gz",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
            },
        },
    )
    registry.verify_payload_components(tmp_path)
    archive.write_bytes(payload + b"!")
    with pytest.raises(ValueError, match="byte count mismatch"):
        registry.verify_payload_components(tmp_path)


def test_component_aware_face_validity_source_is_refused_by_the_access_receipt(tmp_path):
    from sim.scenarios.g3 import _require_single_payload_face_validity

    raw = json.loads(DATA_PATH.read_text())
    for source in raw["sources"]:
        if source["source_id"] == "alibaba-microservices-v2022":
            source["payload_components"] = [
                {
                    "component": "metrics",
                    "status": "UNAVAILABLE",
                    "parser_input": "MSRTMCR_0.csv",
                    "sha256": None,
                    "bytes": None,
                    "archive_path": None,
                    "archive_sha256": None,
                    "archive_bytes": None,
                    "notes": "hypothetical multi-member face-validity acquisition",
                }
            ]
    registry = source_registry_module._load_source_registry_bytes(json.dumps(raw).encode())
    assert registry.get("alibaba-microservices-v2022").role == FACE_VALIDITY
    with pytest.raises(ValueError, match="component-aware receipt schema"):
        _require_single_payload_face_validity(registry)


def test_payload_component_ordering_is_total():
    left = PayloadComponent(
        "call_graph", "UNAVAILABLE", "a.csv", None, None, None, None, None, "n"
    )
    right = PayloadComponent(
        "metrics", "UNAVAILABLE", "b.csv", None, None, None, None, None, "n"
    )
    assert sorted((right, left)) == [left, right]


# ------------------------------------------------------------------- streaming parity


def test_streaming_metric_scan_reproduces_the_whole_payload_fit_exactly(tmp_path):
    payload = _metric_fixture()
    digest = hashlib.sha256(payload).hexdigest()
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": digest,
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )

    result = scan_v2021_metrics(
        archive, METRIC_MEMBER, registry, workspace=tmp_path / "work", minimum_count=20, shards=7
    )
    reference = fit_response_time_targets(
        parse_v2021_metrics(payload), registry, "svc-a", "PROVIDERRPC_RT", minimum_count=20
    )

    assert result.scan.content_sha256 == digest
    assert result.scan.content_bytes == len(payload)
    assert result.scan.truncation == "none-complete-member-streamed"
    assert result.scan.rows_read == payload.decode().count("\n") - 1
    assert result.scan.rejected == 0
    assert [(fit.service, fit.metric) for fit in result.fits] == [("svc-a", "PROVIDERRPC_RT")]
    streamed = result.fits[0]
    assert streamed.seal == reference.seal
    assert streamed.params_seconds == reference.params_seconds
    assert streamed.positive_count == reference.positive_count
    assert streamed.censored_count == reference.censored_count

    # svc-b is scanned and reported, but excluded by the activity rule, not by outcome.
    ineligible = {item.service: item for item in result.cells}["svc-b"]
    assert not ineligible.eligible
    assert ineligible.ineligibility == "rt-cell-without-unique-same-cell-mcr-partner"
    assert ineligible.unpaired_cells == 26
    assert result.services == 2


def test_shard_round_trip_survives_separator_bearing_identifiers(tmp_path):
    """Identifiers are opaque published strings; a tab or comma must not corrupt a shard."""
    hostile = "svc\twith,separators"
    rows = "".join(
        f'{index * 60000},"{hostile}",inst-0,providerRPC_RT,{4 + index}\n'
        f'{index * 60000},"{hostile}",inst-0,providerRPC_MCR,7\n'
        for index in range(22)
    )
    payload = v2021_dump_bytes("timestamp,msname,msinstanceid,metrics,value\n" + rows)
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )
    result = scan_v2021_metrics(archive, METRIC_MEMBER, registry, workspace=tmp_path / "w")
    reference = fit_response_time_targets(
        parse_v2021_metrics(payload), registry, hostile, "PROVIDERRPC_RT", minimum_count=20
    )
    assert [fit.service for fit in result.fits] == [hostile]
    assert result.fits[0].seal == reference.seal


def test_streaming_metric_scan_is_invariant_to_the_shard_count(tmp_path):
    payload = _metric_fixture()
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )
    one = scan_v2021_metrics(
        archive, METRIC_MEMBER, registry, workspace=tmp_path / "w1", shards=1
    )
    many = scan_v2021_metrics(
        archive, METRIC_MEMBER, registry, workspace=tmp_path / "w2", shards=97
    )
    assert [fit.seal for fit in one.fits] == [fit.seal for fit in many.fits]
    assert one.cells == many.cells


def test_streaming_scan_refuses_bytes_that_are_not_the_registered_component(tmp_path):
    payload = _metric_fixture()
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {METRICS_COMPONENT: {"sha256": "b" * 64, "bytes": 4242}},
    )
    with pytest.raises(ValueError, match="hash mismatch for source component"):
        scan_v2021_metrics(archive, METRIC_MEMBER, registry, workspace=tmp_path / "w")


def test_streaming_root_rate_shape_matches_the_whole_payload_transform(tmp_path):
    payload = _metric_fixture()
    digest = hashlib.sha256(payload).hexdigest()
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": digest,
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )
    from sim.scenarios.alibaba import absolute_root_rate_shape

    result = scan_v2021_metrics(
        archive,
        METRIC_MEMBER,
        registry,
        workspace=tmp_path / "w",
        root_services=("svc-a",),
        root_metric="PROVIDERRPC_MCR",
        target_positive_mean_rps=40.0,
    )
    reference = absolute_root_rate_shape(
        parse_v2021_metrics(payload),
        registry,
        "PROVIDERRPC_MCR",
        40.0,
        ("svc-a",),
        "identified-entry-root-only",
    )
    assert result.root_rate_shape is not None
    assert result.root_rate_shape.seal == reference.seal
    assert result.root_services_used == ("svc-a",)
    assert result.root_services_without_observations == ()
    assert result.root_rate_shape.to_arrival_spec(kind="poisson") == reference.to_arrival_spec(
        kind="poisson"
    )


def test_root_without_metric_observations_is_reported_not_silently_folded_in(tmp_path):
    payload = _metric_fixture()
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )
    result = scan_v2021_metrics(
        archive,
        METRIC_MEMBER,
        registry,
        workspace=tmp_path / "w",
        # svc-b reports only HTTP_RT; svc-ghost is absent from the metric member entirely.
        root_services=("svc-a", "svc-b", "svc-ghost"),
        root_metric="PROVIDERRPC_MCR",
        target_positive_mean_rps=40.0,
    )
    assert result.root_services_requested == ("svc-a", "svc-b", "svc-ghost")
    assert result.root_services_used == ("svc-a",)
    assert result.root_services_without_observations == ("svc-b", "svc-ghost")
    assert result.root_rate_shape is not None
    # The sealed artifact lists exactly the roots that contributed to the aggregate.
    assert result.root_rate_shape.root_services == ("svc-a",)
    assert dict(result.scan.facts)["root_services_requested"] == "3"
    assert dict(result.scan.facts)["root_services_used"] == "1"


def test_streaming_call_graph_sample_matches_the_whole_payload_sample(tmp_path):
    payload = _call_graph_fixture()
    digest = hashlib.sha256(payload).hexdigest()
    archive = _tar_gz(tmp_path / "c.tar.gz", CALL_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            CALL_GRAPH_COMPONENT: {
                "sha256": digest,
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )
    result = scan_v2021_call_graph(
        archive, CALL_MEMBER, registry, workspace=tmp_path / "w", sample_traces=9, seed=4242,
        shards=5,
    )
    whole = parse_v2021_call_graph(payload)
    reference = sample_call_graphs(whole, 9, 4242)

    assert result.scan.content_sha256 == digest
    assert result.distinct_traces == 40
    assert result.logical_calls == whole.report.accepted
    assert result.sample.calls == reference.calls
    assert {call.trace_id for call in result.sample.calls} == {
        call.trace_id for call in reference.calls
    }
    # Provenance is the sealed complete-member scan, never a fabricated parent corpus.
    assert result.sample.parent_corpus_seal is None
    assert result.sample.parent_scan_seal == result.scan.seal
    assert result.sample.selection_seed == 4242
    result.sample.verify_seal()
    assert result.root_services == ("front",)


def test_sampled_corpus_needs_exactly_one_parent_provenance(tmp_path):
    payload = _call_graph_fixture()
    whole = parse_v2021_call_graph(payload)
    sample = sample_call_graphs(whole, 3, 7)
    with pytest.raises(ValueError, match="lacks seed/count/parent provenance"):
        dataclasses.replace(sample, parent_scan_seal="c" * 64).verify_seal()
    with pytest.raises(ValueError, match="lacks seed/count/parent provenance"):
        dataclasses.replace(sample, parent_corpus_seal=None).verify_seal()
    with pytest.raises(ValueError, match="inconsistent selection provenance"):
        dataclasses.replace(whole, parent_scan_seal="d" * 64).verify_seal()


def test_one_closure_can_cite_both_components_of_the_same_source(tmp_path):
    """The memo's blocker, end to end.

    Under a single ``payload_sha256`` an RT fit and a call-graph sample could never be
    authorized together: whichever digest was registered de-authorized the other kind of
    evidence.  With per-component registration both bind in one workload closure.
    """
    from sim_helpers import single_service_config, two_edge_cluster

    from sim.scenarios.artifacts import seal_evidence_citing_synthetic_workload

    metric_payload = _metric_fixture()
    call_payload = _call_graph_fixture()
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": hashlib.sha256(metric_payload).hexdigest(),
                "bytes": len(metric_payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            },
            CALL_GRAPH_COMPONENT: {
                "sha256": hashlib.sha256(call_payload).hexdigest(),
                "bytes": len(call_payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            },
        },
    )
    metric_result = scan_v2021_metrics(
        _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, metric_payload),
        METRIC_MEMBER,
        registry,
        workspace=tmp_path / "wm",
    )
    call_result = scan_v2021_call_graph(
        _tar_gz(tmp_path / "c.tar.gz", CALL_MEMBER, call_payload),
        CALL_MEMBER,
        registry,
        workspace=tmp_path / "wc",
        sample_traces=5,
        seed=11,
    )
    assert metric_result.fits[0].source_content_sha256 != call_result.sample.content_sha256

    config = single_service_config(two_edge_cluster())
    closure = seal_evidence_citing_synthetic_workload(
        config,
        registry,
        rt_fits=(metric_result.fits[0],),
        call_graph_samples=(call_result.sample,),
        transform_config=(("test", "both-components"),),
        code_versions=("test-both-components-v1",),
    )
    closure.verify(config, registry)
    assert {item.source_content_sha256 for item in closure.evidence} == {
        metric_result.fits[0].source_content_sha256,
        call_result.sample.content_sha256,
    }


def test_sufficient_statistic_objective_matches_the_per_observation_objective():
    """The fit cost was made size-independent; the objective must be unchanged.

    Re-summing every observation per candidate made a complete-chunk fit of thousands of
    services unaffordable, so the negative log-likelihood is now evaluated from the log
    sample's sufficient statistics.  This pins the two forms together on a sample with
    heavy tails and censoring, where any algebra slip would show.
    """
    import math

    from sim.scenarios.alibaba import _lognormal_censored_fit

    values_s = [0.001 * (1 + (index * 37) % 991) for index in range(2000)]
    censored, threshold_s = 41, 0.001

    def reference_nll(mu: float, sigma: float) -> float:
        out = math.fsum(
            math.log(value)
            + math.log(sigma)
            + 0.5 * math.log(2.0 * math.pi)
            + ((math.log(value) - mu) ** 2) / (2.0 * sigma**2)
            for value in values_s
        )
        z = (math.log(threshold_s) - mu) / sigma
        cdf = max(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 1e-300)
        return out - censored * math.log(cdf)

    mu, sigma = _lognormal_censored_fit(values_s, censored, threshold_s)
    best = reference_nll(mu, sigma)
    # The returned point must be a local optimum of the *original* objective.
    for delta_mu, delta_sigma in ((1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-3), (0.0, -1e-3)):
        assert reference_nll(mu + delta_mu, sigma + delta_sigma) >= best
    assert sigma > 0
    assert math.isfinite(mu)


def test_scan_artifact_cannot_be_forged_or_relabelled(tmp_path):
    payload = _metric_fixture()
    archive = _tar_gz(tmp_path / "m.tar.gz", METRIC_MEMBER, payload)
    registry = _component_registry(
        tmp_path,
        {
            METRICS_COMPONENT: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "archive_path": None,
                "archive_sha256": None,
                "archive_bytes": None,
            }
        },
    )
    scan = scan_v2021_metrics(archive, METRIC_MEMBER, registry, workspace=tmp_path / "w").scan
    scan.verify_seal()
    with pytest.raises(ValueError, match="issued by the streaming scanner"):
        dataclasses.replace(scan, _authority=object()).verify_seal()
    with pytest.raises(ValueError, match="cannot accept more rows than it read"):
        dataclasses.replace(scan, rows_read=1).verify_seal()
    with pytest.raises(ValueError, match="seal mismatch"):
        dataclasses.replace(scan, parser_input="MSRTQps_9.csv").verify_seal()
    with pytest.raises(ValueError, match="complete member was streamed"):
        dataclasses.replace(scan, truncation="head-slice").verify_seal()
