"""Pilot accounting must fail closed, retain uncertainty and replay without spending."""

import dataclasses
import json

import pytest

from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.deepseek_pilots_20260920 import client as module
from experiments.matched_transport_20260909.client import UnresolvedDispatch


def config():
    base = load_openrouter_engine_config(module.ROOT / "evolution/openrouter_engine.json")
    return dataclasses.replace(
        base,
        model_id="deepseek/deepseek-v4.1-flash",
        max_attempts=1,
        prompt_microusd_per_million=150000,
        completion_microusd_per_million=600000,
        max_call_reservation_microusd=50000,
        max_total_microusd=600000,
    )


def raw():
    return {
        "model": "deepseek/deepseek-v4.1-flash",
        "provider": "DeepSeek",
        "choices": [{"message": {"content": "policy source"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "cost": 0.00075},
    }


@pytest.mark.parametrize("bad", ["provider", "price", "tokens"])
def test_receipts_reject_wrong_provider_price_or_usage(bad):
    response = raw()
    if bad == "provider":
        response["provider"] = "Other"
    elif bad == "price":
        response["usage"]["cost"] = 0.002
    else:
        response["usage"]["completion_tokens"] = -1
    with pytest.raises(ValueError):
        module.check_receipt(response, config())


def test_legacy_reservations_are_not_mistaken_for_free_budget():
    with module.LEGACY.open("rb") as stream:
        assert module.legacy_exposure(stream) == 697524


def test_replay_and_request_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    c = module.PilotClient(config(), journal=tmp_path / "journal", calls_dir=tmp_path / "calls")
    dispatches = []
    monkeypatch.setattr(c, "_dispatch", lambda request: dispatches.append(1) or raw())

    def complete(seed):
        return c.complete(
            model=c.config.model_id, prompt="test", generation=c.config.generation, seed=seed
        )

    first = complete(1)
    assert complete(1) == first
    assert len(dispatches) == 1
    for seed in range(2, 7):
        complete(seed)
    with pytest.raises(module.PilotLimit):
        complete(7)
    assert len(dispatches) == 6
    assert c.ledger.spent_microusd == 4500
    assert "fake-test-key" not in (tmp_path / "journal").read_text()


def test_unknown_dispatch_preserves_reservation_and_blocks_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    c = module.PilotClient(config(), journal=tmp_path / "journal", calls_dir=tmp_path / "calls")

    def fail(request):
        raise OSError("connection lost")

    monkeypatch.setattr(c, "_dispatch", fail)
    with pytest.raises(UnresolvedDispatch):
        c.complete(model=c.config.model_id, prompt="test", generation=c.config.generation, seed=1)
    with pytest.raises(UnresolvedDispatch):
        module.PilotClient(config(), journal=tmp_path / "journal", calls_dir=tmp_path / "calls")
    events = [json.loads(line) for line in (tmp_path / "journal").read_text().splitlines()]
    assert events[-1]["payload"]["reservation_microusd"] == 50000
