from __future__ import annotations

import dataclasses
import io
import json
import urllib.error
from pathlib import Path

import pytest

from evolution.llm_operator import LLMCompletion
from evolution.model import ApiCostLedger
from evolution.openrouter_engine import (
    OpenRouterCompletionClient,
    load_openrouter_engine_config,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "evolution/openrouter_engine.json"
API_KEY = "test-secret-openrouter-key"


class StubResponse:
    def __init__(self, payload, url="https://openrouter.ai/api/v1/chat/completions"):
        self.payload = payload
        self.url = url

    def read(self):
        return json.dumps(self.payload).encode()

    def geturl(self):
        return self.url

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _config(**changes):
    return dataclasses.replace(load_openrouter_engine_config(CONFIG), **changes)


def _payload(
    *,
    model="z-ai/glm-5.3-flash",
    prompt_tokens=10,
    completion_tokens=20,
    reasoning_tokens=None,
    content="proposal",
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": 999,
    }
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "id": "provider-request-id-must-not-be-archived",
        "created": 1_777_777_777,
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": usage,
    }


def _complete(client):
    return client.complete(
        model=client.config.model_id,
        prompt="write a proposal",
        generation=client.config.generation,
        seed=90210,
    )


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", ""),
        ("model_id", 3),
        ("endpoint", "http://openrouter.ai/api/v1"),
        ("endpoint", "https://user:pass@openrouter.ai/api/v1"),
        ("endpoint", "https://openrouter.ai/api/v2"),
        ("generation", []),
        ("generation", {1: "bad"}),
        ("generation", {"seed": 1}),
        ("generation", {"temperature": float("nan")}),
        ("prompt_microusd_per_million", -1),
        ("prompt_microusd_per_million", 1.5),
        ("completion_microusd_per_million", -1),
        ("completion_microusd_per_million", True),
        ("max_call_reservation_microusd", 0),
        ("max_call_reservation_microusd", 20_000_001),
        ("max_total_microusd", 0),
        ("request_timeout_s", 0),
        ("request_timeout_s", 3601),
        ("max_output_tokens", 0),
        ("max_output_tokens", 1.5),
        ("max_attempts", 0),
        ("max_attempts", 11),
        ("retry_initial_backoff_s", 0),
        ("retry_initial_backoff_s", 5),
        ("retry_max_backoff_s", 0),
        ("retry_max_backoff_s", 61),
        ("reasoning_effort", 0),
        ("reasoning_effort", -1),
        ("reasoning_effort", True),
        ("reasoning_effort", 1.5),
        ("reasoning_effort", "maximum"),
    ],
)
def test_config_rejects_malformed_fields(field, value):
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_loader_rejects_missing_and_unknown_fields(tmp_path):
    raw = json.loads(CONFIG.read_text())
    raw.pop("model_id")
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_openrouter_engine_config(path)

    raw["model_id"] = "z-ai/glm-5.3"
    raw["surprise"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_openrouter_engine_config(path)


def test_cost_is_computed_from_tokens_and_rounded_up(monkeypatch):
    # 3 * 0.075 + 2 * 0.25 = 0.725 microdollars before integral rounding.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: StubResponse(
            _payload(prompt_tokens=3, completion_tokens=2)
        ),
    )
    client = OpenRouterCompletionClient(_config())

    completion = _complete(client)

    assert completion.cost_microusd == 1
    assert client.ledger.spent_microusd == 1

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: StubResponse(
            _payload(prompt_tokens=1, completion_tokens=0)
        ),
    )
    assert _complete(client).cost_microusd == 1


def test_ledger_refuses_call_before_dispatch(monkeypatch):
    dispatched = False

    def urlopen(*_args, **_kwargs):
        nonlocal dispatched
        dispatched = True
        return StubResponse(_payload())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    config = _config(max_total_microusd=200_000)
    ledger = ApiCostLedger(config.max_total_microusd, spent_microusd=1)
    client = OpenRouterCompletionClient(config, ledger)

    with pytest.raises(ValueError, match="max_api_costs exceeded"):
        _complete(client)
    assert not dispatched


@pytest.mark.parametrize("status", [429, 503])
def test_retryable_http_errors_back_off(monkeypatch, status):
    calls = 0
    sleeps = []

    def urlopen(request, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.HTTPError(request.full_url, status, API_KEY, {}, io.BytesIO())
        return StubResponse(_payload())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)
    completion = _complete(OpenRouterCompletionClient(_config()))

    assert completion.content == "proposal"
    assert calls == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [400, 401])
def test_nonretryable_http_errors_do_not_retry(monkeypatch, status):
    calls = 0

    def urlopen(request, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(request.full_url, status, API_KEY, {}, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _delay: pytest.fail("unexpected retry"))
    with pytest.raises(RuntimeError, match=f"HTTP error {status}") as raised:
        _complete(OpenRouterCompletionClient(_config()))
    assert calls == 1
    assert API_KEY not in str(raised.value)


def test_api_key_is_excluded_from_archive_and_errors(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: StubResponse(_payload()),
    )
    completion = _complete(OpenRouterCompletionClient(_config()))
    assert API_KEY not in json.dumps(completion.raw_response)

    def fail(*_args, **_kwargs):
        raise OSError(API_KEY)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError) as raised:
        _complete(OpenRouterCompletionClient(_config()))
    assert API_KEY not in str(raised.value)


def test_api_key_falls_back_to_repo_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr("evolution.openrouter_engine._REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text(f"# local credentials\nOPENROUTER_API_KEY='{API_KEY}'\n")
    requests = []

    def urlopen(request, **_kwargs):
        requests.append(request)
        return StubResponse(_payload())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    _complete(OpenRouterCompletionClient(_config()))

    assert requests[0].headers["Authorization"] == f"Bearer {API_KEY}"


def test_complete_returns_protocol_dataclass_and_normalized_payload(monkeypatch):
    requests = []

    def urlopen(request, **_kwargs):
        requests.append(request)
        return StubResponse(
            _payload(prompt_tokens=7, completion_tokens=11, reasoning_tokens=9)
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    completion = _complete(OpenRouterCompletionClient(_config()))

    assert isinstance(completion, LLMCompletion)
    assert completion.content == "proposal"
    assert completion.prompt_tokens == 7
    assert completion.completion_tokens == 11
    assert completion.cost_microusd == 4
    assert completion.raw_response == {
        "model": "z-ai/glm-5.3-flash",
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "proposal"},
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 11,
            "reasoning_tokens": 9,
        },
        "cost_microusd": 4,
    }
    sent = json.loads(requests[0].data)
    assert sent["seed"] == 90210
    assert sent["max_tokens"] == 24000
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["messages"] == [{"role": "user", "content": "write a proposal"}]
    assert requests[0].headers["Authorization"] == f"Bearer {API_KEY}"


def test_complete_rejects_response_from_a_different_model(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: StubResponse(_payload(model="different/model")),
    )

    with pytest.raises(RuntimeError, match="response model differs from pin"):
        _complete(OpenRouterCompletionClient(_config()))


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [
        (6000, {"max_tokens": 6000}),
        ("medium", {"effort": "medium"}),
        (None, None),
    ],
)
def test_reasoning_request_field(monkeypatch, reasoning_effort, expected):
    requests = []

    def urlopen(request, **_kwargs):
        requests.append(request)
        return StubResponse(_payload())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    _complete(OpenRouterCompletionClient(_config(reasoning_effort=reasoning_effort)))

    sent = json.loads(requests[0].data)
    if expected is None:
        assert "reasoning" not in sent
    else:
        assert sent["reasoning"] == expected
        assert set(sent["reasoning"]) == set(expected)


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_content_raises_with_finish_reason_and_token_counts(monkeypatch, content):
    payload = _payload(
        prompt_tokens=13,
        completion_tokens=24_000,
        reasoning_tokens=23_900,
        content=content,
    )
    payload["choices"][0]["finish_reason"] = "length"
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: StubResponse(payload)
    )
    client = OpenRouterCompletionClient(_config())

    with pytest.raises(RuntimeError) as raised:
        _complete(client)

    message = str(raised.value)
    assert "finish_reason='length'" in message
    assert "prompt_tokens=13" in message
    assert "completion_tokens=24000" in message
    assert "reasoning_tokens=23900" in message
    assert client.ledger.spent_microusd == 6001
