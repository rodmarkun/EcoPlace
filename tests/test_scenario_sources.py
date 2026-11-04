"""Phase-4 source split and Alibaba ingestion adversarial tests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sim_helpers import v2021_dump_bytes

import sim.scenarios.sources as source_registry_module
from sim.scenarios.alibaba import (
    V2021,
    absolute_root_rate_shape,
    capacity_scaled_shape,
    fit_response_time_targets,
    normalized_load_shape,
    parse_v2021_call_graph,
    parse_v2021_metrics,
    parse_v2022_metrics,
    sample_call_graphs,
)
from sim.scenarios.sources import (
    DATA_PATH,
    FACE_VALIDITY,
    load_source_registry,
    require_frozen_clean_checkout,
)

REPO = Path(__file__).resolve().parent.parent


def _available_registry(tmp_path, source_id: str, payload: bytes):
    raw = json.loads(DATA_PATH.read_text())
    for source in raw["sources"]:
        if source["source_id"] == source_id:
            # Fixtures exercise the single-payload contract; drop any registered
            # component list so one fixture digest cannot masquerade as a component.
            source["payload_components"] = []
            source["payload_status"] = "AVAILABLE"
            source["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    payload = json.dumps(raw).encode()
    (tmp_path / "source-split.json").write_bytes(payload)
    return source_registry_module._load_source_registry_bytes(payload)


def test_registry_split_is_sealed_and_retained_metadata_rehashes():
    registry = load_source_registry()
    registry.verify_retained(REPO)
    assert registry.get(V2021).role == "calibration"
    assert registry.get("alibaba-microservices-v2022").role == FACE_VALIDITY
    with pytest.raises(ValueError, match="schema|seal mismatch"):
        dataclasses.replace(registry, schema="laundered").verify_seal()


def test_v2021_signed_pairs_collapse_and_are_input_order_invariant():
    header = "timestamp,traceid,rpcid,um,rpctype,interface,dm,rt\n"
    rows = [
        "10,t1,0.1,A,rpc,i,B,12\n",
        "10,t1,0.1,A,rpc,i,B,-7\n",
        "11,t1,0.2,A,MC,,MC,0\n",
    ]
    a = parse_v2021_call_graph(v2021_dump_bytes(header + "".join(rows)))
    b = parse_v2021_call_graph(v2021_dump_bytes(header + "".join(reversed(rows))))
    # Row ids are audit identities and therefore reflect physical order; logical
    # values and pairing do not.
    logical_a = [dataclasses.replace(call, row_ids=()) for call in a.calls]
    logical_b = [dataclasses.replace(call, row_ids=()) for call in b.calls]
    assert logical_a == logical_b
    assert a.calls[0].upstream_rt_ms == 12.0
    assert a.calls[0].provider_rt_ms == 7.0
    assert a.calls[1].left_censored_below_ms == 1.0
    assert a.report.accepted == 2


def test_call_graph_bottom_k_sampling_is_seeded_and_trace_atomic():
    text = "timestamp,traceid,rpcid,um,rpctype,interface,dm,rt\n" + "".join(
        f"{i},t{i},0,A,DB,,DB,1\n" for i in range(8)
    )
    corpus = parse_v2021_call_graph(v2021_dump_bytes(text))
    one = sample_call_graphs(corpus, 3, 91)
    two = sample_call_graphs(corpus, 3, 91)
    assert one.calls == two.calls
    assert len({call.trace_id for call in one.calls}) == 3
    assert one.selection_seed == 91
    assert one.requested_trace_count == 3
    assert one.parent_corpus_seal == corpus.seal
    assert one.transform_code_version.startswith("bottom-k-trace-sampler-v1:")
    with pytest.raises(ValueError, match="exceeds"):
        sample_call_graphs(corpus, 9, 91)
    with pytest.raises(TypeError, match="integer"):
        sample_call_graphs(corpus, True, 91)


def test_heldout_checkout_binding_rejects_dirty_or_untracked_state(monkeypatch):
    with pytest.raises(ValueError, match="full SHA-1"):
        require_frozen_clean_checkout("a" * 7)
    replies = iter(
        (
            SimpleNamespace(stdout="a" * 40 + "\n"),
            SimpleNamespace(stdout="?? post-freeze-analysis.py\n"),
        )
    )
    monkeypatch.setattr(source_registry_module.subprocess, "run", lambda *a, **k: next(replies))
    with pytest.raises(ValueError, match="clean frozen working tree"):
        require_frozen_clean_checkout("a" * 40)


def test_v2021_rate_scaling_preserves_target_mean_and_v2022_cannot_be_absolute():
    registry = load_source_registry()
    v1 = parse_v2021_metrics(
        v2021_dump_bytes(
            b"timestamp,msname,msinstanceid,metrics,value\n"
            b"0,A,p0,providerRPC_MCR,10\n"
            b"60000,A,p0,providerRPC_MCR,20\n"
        )
    )
    scaled = capacity_scaled_shape(((0, 10.0), (60000, 20.0)), 45.0)
    assert sum(value for _, value in scaled) / len(scaled) == pytest.approx(45.0)
    # v2021 is registered component-aware, so a fixture payload fails on the exact
    # metrics-component digest rather than on availability.
    with pytest.raises(ValueError, match="hash mismatch for source component"):
        absolute_root_rate_shape(
            v1,
            registry,
            "PROVIDERRPC_MCR",
            45.0,
            ("A",),
            "identified-entry-root-only",
        )

    v2 = parse_v2022_metrics(

            b"timestamp,msname,msinstanceid,nodeid,providerrpc_rt,providerrpc_mcr\n"
            b"0,MS_1,POD_1,NODE_1,9.0,0.25\n"

    )
    with pytest.raises(ValueError, match="payload.*unavailable"):
        normalized_load_shape(v2, registry)
    with pytest.raises(ValueError, match="not permitted role"):
        absolute_root_rate_shape(
            v2,
            registry,
            "PROVIDERRPC_MCR",
            10.0,
            ("MS_1",),
            "identified-entry-root-only",
        )
    with pytest.raises(ValueError, match="not permitted role"):
        fit_response_time_targets(v2, registry, "MS_1", "PROVIDERRPC_RT", minimum_count=1)


def test_censored_rt_rows_are_preserved_but_unregistered_payload_cannot_fit():
    registry = load_source_registry()
    rows = "".join(
        f"{i * 1000},A,p0,providerRPC_RT,{0 if i < 5 else 2 + i / 10}\n"
        f"{i * 1000},A,p0,providerRPC_MCR,1\n"
        for i in range(30)
    )
    corpus = parse_v2021_metrics(
        v2021_dump_bytes("timestamp,msname,msinstanceid,metrics,value\n" + rows)
    )
    assert sum(obs.left_censored_below is not None for obs in corpus.observations) == 5
    with pytest.raises(ValueError, match="hash mismatch for source component"):
        fit_response_time_targets(corpus, registry, "A", "PROVIDERRPC_RT", minimum_count=20)


def test_parser_computes_payload_hash_instead_of_trusting_a_caller_claim():
    payload = v2021_dump_bytes(
        b"timestamp,msname,msinstanceid,metrics,value\n"
        b"0,A,p0,providerRPC_MCR,10\n"
    )
    corpus = parse_v2021_metrics(payload)
    assert corpus.content_sha256 == hashlib.sha256(payload).hexdigest()
    with pytest.raises(TypeError, match="exact bytes"):
        parse_v2021_metrics(__import__("io").StringIO(payload.decode()))  # type: ignore[arg-type]


def test_inconsistent_remote_halves_are_rejected():
    payload = v2021_dump_bytes(
        b"timestamp,traceid,rpcid,um,rpctype,interface,dm,rt\n"
        b"10,t1,0.1,A,rpc,i,B,12\n"
        b"99,t1,0.1,X,rpc,j,Y,-7\n"
    )
    corpus = parse_v2021_call_graph(payload)
    assert corpus.calls == ()
    assert corpus.report.rejected_by_reason == (("inconsistent_remote_halves", 2),)


def test_absolute_root_rate_aggregates_instances_retains_gaps_and_converts_units(tmp_path):
    payload = v2021_dump_bytes(
        b"timestamp,msname,msinstanceid,metrics,value\n"
        b"0,A,p0,providerRPC_MCR,0\n"
        b"0,A,p1,providerRPC_MCR,0\n"
        b"60000,A,p0,providerRPC_MCR,10\n"
        b"60000,A,p1,providerRPC_MCR,20\n"
        b"120000,A,p0,providerRPC_MCR,0\n"
        b"120000,A,p1,providerRPC_MCR,0\n"
        b"180000,A,p0,providerRPC_MCR,10\n"
    )
    corpus = parse_v2021_metrics(payload)
    registry = _available_registry(tmp_path, V2021, payload)
    artifact = absolute_root_rate_shape(
        corpus,
        registry,
        "PROVIDERRPC_MCR",
        20.0,
        ("A",),
        "identified-entry-root-only",
    )
    assert artifact.points_relative_s == (
        (0.0, 0.0),
        (60.0, 30.0),
        (120.0, 0.0),
        (180.0, 10.0),
    )
    arrival = artifact.to_arrival_spec(kind="poisson")
    assert arrival.start_s == 60.0
    assert arrival.rate_per_s == 30.0
    assert arrival.rate_schedule == ((120.0, 0.0), (180.0, 10.0), (240.0, 0.0))
    assert arrival.time_at_integrated_rate(2400.1, 1000.0) is None
    with pytest.raises(ValueError, match="registered transform"):
        dataclasses.replace(artifact, _authority=object()).verify_seal()


def test_absolute_root_rate_rejects_missing_source_cells(tmp_path):
    payload = v2021_dump_bytes(
        b"timestamp,msname,msinstanceid,metrics,value\n"
        b"0,A,p0,providerRPC_MCR,10\n"
        b"120000,A,p0,providerRPC_MCR,20\n"
    )
    with pytest.raises(ValueError, match="missing 60-second source cell"):
        absolute_root_rate_shape(
            parse_v2021_metrics(payload),
            _available_registry(tmp_path, V2021, payload),
            "PROVIDERRPC_MCR",
            20.0,
            ("A",),
            "identified-entry-root-only",
        )


def test_rt_fit_is_typed_and_authority_issued(tmp_path):
    payload = v2021_dump_bytes(
        b"timestamp,msname,msinstanceid,metrics,value\n"
        b"0,A,p0,providerRPC_RT,0\n"
        b"0,A,p0,providerRPC_MCR,1\n"
        b"60000,A,p0,providerRPC_RT,4\n"
        b"60000,A,p0,providerRPC_MCR,1\n"
    )
    corpus = parse_v2021_metrics(payload)
    registry = _available_registry(tmp_path, V2021, payload)
    artifact = fit_response_time_targets(
        corpus, registry, "A", "PROVIDERRPC_RT", minimum_count=2
    )
    artifact.verify_seal()
    assert artifact.service == "A"
    assert artifact.metric == "PROVIDERRPC_RT"
    assert artifact.censored_count == 1
    assert artifact.activity_rule == "paired-same-cell-family-mcr-strictly-positive-v1"
    with pytest.raises(ValueError, match="registered fitter"):
        dataclasses.replace(artifact, _authority=object()).verify_seal()


def test_rt_fit_rejects_duplicate_instance_interval_cells(tmp_path):
    payload = v2021_dump_bytes(
        b"timestamp,msname,msinstanceid,metrics,value\n"
        b"0,A,p0,providerRPC_RT,2\n"
        b"0,A,p0,providerRPC_RT,3\n"
        b"0,A,p0,providerRPC_MCR,1\n"
    )
    with pytest.raises(ValueError, match="duplicate service-instance interval"):
        fit_response_time_targets(
            parse_v2021_metrics(payload),
            _available_registry(tmp_path, V2021, payload),
            "A",
            "PROVIDERRPC_RT",
            minimum_count=1,
        )


def test_v2022_zero_rt_is_inactive_only_when_paired_mcr_is_zero(tmp_path):
    payload = (
        b"timestamp,msname,msinstanceid,nodeid,providerrpc_rt,providerrpc_mcr\n"
        b"0,A,p0,n0,0,0\n"
        b"60000,A,p0,n0,4,1\n"
    )
    corpus = parse_v2022_metrics(payload)
    rt = [observation for observation in corpus.observations if observation.metric.endswith("_RT")]
    assert rt[0].activity_value == 0
    assert rt[0].left_censored_below is None
    registry = _available_registry(tmp_path, "alibaba-microservices-v2022", payload)
    with pytest.raises(ValueError, match="not permitted role"):
        fit_response_time_targets(corpus, registry, "A", "PROVIDERRPC_RT", minimum_count=1)


def test_authority_sealed_corpus_rejects_observation_replacement():
    payload = v2021_dump_bytes(
        b"timestamp,msname,msinstanceid,metrics,value\n"
        b"0,A,p0,providerRPC_RT,2\n"
        b"1,A,p0,providerRPC_RT,3\n"
    )
    corpus = parse_v2021_metrics(payload)
    forged = dataclasses.replace(corpus, observations=())
    with pytest.raises(ValueError, match="corpus seal mismatch"):
        forged.verify_seal()
