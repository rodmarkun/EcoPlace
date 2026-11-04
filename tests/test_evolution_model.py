from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.request
from pathlib import Path

import pytest

from evolution.llm_operator import VLLMCompletionClient
from evolution.model import (
    ApiCostLedger,
    WeightFile,
    load_model_track,
    verify_host_handshake,
    verify_weight_directory,
)
from evolution.run_campaign import _endpoint_origin

MODEL = Path(__file__).parents[1] / "evolution" / "open_model.json"


def _handshake(track):
    return {
        "model_id": track.model_id,
        "revision": track.revision,
        "weight_set_sha256": track.weight_set_sha256,
        "accelerator": track.host.accelerator,
        "accelerator_memory_gb": track.host.accelerator_memory_gb,
        "cpu_arch": track.host.cpu_arch,
        "os": track.host.os,
        "engine": track.host.engine,
        "engine_version": track.host.engine_version,
        "transformers_version": track.host.transformers_version,
        "dtype": track.host.dtype,
        "tensor_parallel_size": track.host.tensor_parallel_size,
        "max_model_len": track.host.max_model_len,
        "isolation": track.host.isolation,
        "request_idempotency": track.host.request_idempotency,
        "cost_enforcement": track.host.cost_enforcement,
        "max_request_cost_microusd": track.host.max_request_cost_microusd,
        "weight_verification": track.host.weight_verification,
    }


def test_exactly_one_open_weight_track_is_hash_pinned():
    track = load_model_track(MODEL)

    assert track.model_id == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert track.revision == "eaf8c2495bf23eb47dc38da1974bb29a158190e0"
    assert len(track.weights) == 16
    assert all(len(weight.sha256) == 64 for weight in track.weights)
    assert track.confirmatory_ready is False


def test_host_handshake_fails_closed_on_any_drift():
    track = load_model_track(MODEL)
    handshake = _handshake(track)
    verify_host_handshake(track, handshake)

    handshake["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision"):
        verify_host_handshake(track, handshake)


def test_api_cost_ledger_reserves_before_dispatch_and_never_overspends():
    track = load_model_track(MODEL)
    ledger = ApiCostLedger(track.max_api_costs_microusd)
    ledger.reserve(track.max_call_reservation_microusd)
    ledger.settle(track.max_call_reservation_microusd, 125000)

    assert ledger.spent_microusd == 125000
    with pytest.raises(ValueError, match="max_api_costs"):
        ledger.reserve(track.max_api_costs_microusd)


def test_model_track_rejects_unpinned_second_track_semantics(tmp_path):
    track = load_model_track(MODEL)
    manifest = json.loads(MODEL.read_text())
    manifest["tracks"].append(dict(manifest["tracks"][0]))
    doubled = tmp_path / "two-tracks.json"
    doubled.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly one"):
        load_model_track(doubled)
    with pytest.raises(ValueError, match="exactly one Apache"):
        dataclasses.replace(track, track_id="commercial")


def test_handshake_is_fetched_from_the_completion_endpoint_origin(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def read(self):
            return b'{"model_id":"test"}'

        def geturl(self):
            return "https://proposal.example/evolved-continuum/handshake"

        def close(self):
            return None

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = VLLMCompletionClient(
        "https://proposal.example",
        max_request_cost_microusd=500_000,
        revision="a" * 40,
        weight_set_sha256="b" * 64,
        timeout_s=12.0,
    )

    assert client.fetch_handshake() == {"model_id": "test"}
    assert seen == [("https://proposal.example/evolved-continuum/handshake", 12.0)]
    with pytest.raises(ValueError, match="without credentials or a path"):
        _endpoint_origin("https://proposal.example/unbound-proxy")


def test_completion_rejects_redirect_or_backend_identity_drift(monkeypatch):
    class Response:
        def __init__(self, payload, url):
            self.payload = payload
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def read(self):
            return json.dumps(self.payload).encode()

        def geturl(self):
            return self.url

        def close(self):
            return None

    track = load_model_track(MODEL)
    client = VLLMCompletionClient(
        "https://proposal.example",
        max_request_cost_microusd=track.max_call_reservation_microusd,
        revision=track.revision,
        weight_set_sha256=track.weight_set_sha256,
    )
    handshake = _handshake(track)
    digest = client.bind_handshake(handshake)
    response = {
        "model": "wrong/model",
        "revision": track.revision,
        "weight_set_sha256": track.weight_set_sha256,
        "host_handshake_sha256": digest,
        "choices": [{"message": {"content": "bad"}}],
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: Response(response, request.full_url),
    )
    with pytest.raises(ValueError, match="identity differs"):
        client.complete(model=track.model_id, prompt="p", generation={}, seed=1)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: Response(
            handshake, "https://redirected.example/evolved-continuum/handshake"
        ),
    )
    with pytest.raises(ValueError, match="redirected"):
        client.fetch_handshake()


def test_weight_directory_verifier_checks_the_registered_bytes(tmp_path):
    payload = b"pinned-test-weight"
    weight = WeightFile(
        path="model.safetensors",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    track = dataclasses.replace(
        load_model_track(MODEL),
        weights=(weight,),
        weight_set_sha256=hashlib.sha256(
            json.dumps(
                [weight.payload()],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    )
    (tmp_path / weight.path).write_bytes(payload)

    verify_weight_directory(track, tmp_path)
    (tmp_path / weight.path).write_bytes(payload + b"drift")
    with pytest.raises(ValueError, match="wrong size"):
        verify_weight_directory(track, tmp_path)
