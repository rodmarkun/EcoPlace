"""Pinned, budgeted OpenRouter completion transport for proposal generation."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .llm_operator import LLMCompletion
from .model import ApiCostLedger

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class OpenRouterEngineConfig:
    """Strict model, pricing, retry, and budget pin for OpenRouter."""

    model_id: str
    generation: dict[str, Any]
    prompt_microusd_per_million: int
    completion_microusd_per_million: int
    max_call_reservation_microusd: int
    max_total_microusd: int
    request_timeout_s: float
    max_output_tokens: int
    max_attempts: int
    retry_initial_backoff_s: float
    retry_max_backoff_s: float
    reasoning_effort: str | int | None
    endpoint: str = DEFAULT_ENDPOINT

    @property
    def revision(self) -> None:
        return None

    @property
    def weight_set_sha256(self) -> None:
        return None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.generation, dict) or any(
            not isinstance(key, str) for key in self.generation
        ):
            raise ValueError("generation must be an object with string keys")
        forbidden = {"model", "messages", "seed", "max_tokens"} & self.generation.keys()
        if forbidden:
            raise ValueError("generation must not override pinned request fields")
        try:
            json.dumps(self.generation, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation must contain finite JSON values") from exc

        _nonnegative_int(
            "prompt_microusd_per_million", self.prompt_microusd_per_million
        )
        _nonnegative_int(
            "completion_microusd_per_million", self.completion_microusd_per_million
        )
        _positive_int(
            "max_call_reservation_microusd", self.max_call_reservation_microusd
        )
        _positive_int("max_total_microusd", self.max_total_microusd)
        if self.max_call_reservation_microusd > self.max_total_microusd:
            raise ValueError("per-call reservation exceeds total cost cap")
        timeout = _positive_number("request_timeout_s", self.request_timeout_s)
        if timeout > 3600:
            raise ValueError("request_timeout_s must not exceed 3600")
        _positive_int("max_output_tokens", self.max_output_tokens)
        attempts = _positive_int("max_attempts", self.max_attempts)
        if attempts > 10:
            raise ValueError("max_attempts must not exceed 10")
        initial = _positive_number(
            "retry_initial_backoff_s", self.retry_initial_backoff_s
        )
        maximum = _positive_number("retry_max_backoff_s", self.retry_max_backoff_s)
        if initial > maximum:
            raise ValueError("initial retry backoff exceeds maximum retry backoff")
        if maximum > 60:
            raise ValueError("retry_max_backoff_s must not exceed 60")
        if isinstance(self.reasoning_effort, bool) or not (
            self.reasoning_effort is None
            or isinstance(self.reasoning_effort, str)
            and self.reasoning_effort in {"minimal", "low", "medium", "high"}
            or isinstance(self.reasoning_effort, int)
            and self.reasoning_effort > 0
        ):
            raise ValueError(
                "reasoning_effort must be a positive integer, minimal, low, medium, "
                "high, or null"
            )

        if not isinstance(self.endpoint, str):
            raise ValueError("endpoint must be a string")
        parsed = urlsplit(self.endpoint.rstrip("/"))
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/api/v1"
        ):
            raise ValueError("endpoint must be an HTTPS origin ending in /api/v1")


_CONFIG_FIELDS = {
    "model_id",
    "endpoint",
    "generation",
    "prompt_microusd_per_million",
    "completion_microusd_per_million",
    "max_call_reservation_microusd",
    "max_total_microusd",
    "request_timeout_s",
    "max_output_tokens",
    "max_attempts",
    "retry_initial_backoff_s",
    "retry_max_backoff_s",
    "reasoning_effort",
}


def load_openrouter_engine_config(path: str | Path) -> OpenRouterEngineConfig:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) != _CONFIG_FIELDS:
        raise ValueError(
            "OpenRouter config must contain exactly these fields: "
            f"{sorted(_CONFIG_FIELDS)}"
        )
    return OpenRouterEngineConfig(**raw)


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = _REPO_ROOT / ".env"
    try:
        lines = env_path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == "OPENROUTER_API_KEY" and value.strip():
            return value.strip().strip("'\"")
    raise RuntimeError("OpenRouter API key is not configured")


class EmptyCompletion(RuntimeError):
    """A parsed, billed response without final content, e.g. reasoning-only truncation.

    The provider outcome is known, so callers may settle its cost and treat the proposal
    as failed rather than as an uncertain dispatch.
    """

    def __init__(self, message, *, cost_microusd, finish_reason, prompt_tokens,
                 completion_tokens, reasoning_tokens):
        super().__init__(message)
        self.cost_microusd = cost_microusd
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens


class HTTPStatusFailure(RuntimeError):
    """The provider answered with a definite HTTP error status (a known outcome)."""

    def __init__(self, status):
        super().__init__(f"OpenRouter HTTP error {status}")
        self.status = status


class DispatchFailure(RuntimeError):
    """The request failed without any provider response (network error or timeout)."""


class OpenRouterCompletionClient:
    """OpenRouter HTTP client with deterministic local accounting."""

    cost_basis = "pinned-token-prices"

    def __init__(
        self,
        config: OpenRouterEngineConfig,
        ledger: ApiCostLedger | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger or ApiCostLedger(config.max_total_microusd)
        if self.ledger.cap_microusd != config.max_total_microusd:
            raise ValueError("ledger cap differs from OpenRouter config")
        base = config.endpoint.rstrip("/")
        parsed = urlsplit(base)
        host = parsed.hostname
        assert host is not None
        if ":" in host:
            host = f"[{host}]"
        port = "" if parsed.port is None else f":{parsed.port}"
        self._origin = f"{parsed.scheme}://{host}{port}"
        self._url = base + "/chat/completions"

    def _open(self, request: urllib.request.Request):
        response = urllib.request.urlopen(request, timeout=self.config.request_timeout_s)
        final = urlsplit(response.geturl())
        host = final.hostname
        if host is None:
            response.close()
            raise RuntimeError("OpenRouter response has no final origin")
        if ":" in host:
            host = f"[{host}]"
        port = "" if final.port is None else f":{final.port}"
        if f"{final.scheme}://{host}{port}" != self._origin:
            response.close()
            raise RuntimeError("OpenRouter redirected to a different origin")
        return response

    def _dispatch(self, request: urllib.request.Request) -> dict[str, Any]:
        delay = self.config.retry_initial_backoff_s
        for attempt in range(self.config.max_attempts):
            try:
                with self._open(request) as response:
                    raw = json.loads(response.read())
                if not isinstance(raw, dict):
                    raise ValueError
                return raw
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 == self.config.max_attempts:
                    raise HTTPStatusFailure(exc.code) from None
            except (ValueError, json.JSONDecodeError):
                raise RuntimeError("OpenRouter returned an invalid JSON response") from None
            except Exception:
                raise DispatchFailure("OpenRouter request failed") from None
            time.sleep(delay)
            delay = min(delay * 2, self.config.retry_max_backoff_s)
        raise AssertionError("retry loop did not terminate")

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        generation: dict[str, Any],
        seed: int,
    ) -> LLMCompletion:
        if model != self.config.model_id or generation != self.config.generation:
            raise ValueError("completion request differs from pinned OpenRouter config")
        if not isinstance(prompt, str) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("prompt must be text and seed must be an explicit integer")
        key = _api_key()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **generation,
            "seed": seed,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.reasoning_effort is not None:
            if isinstance(self.config.reasoning_effort, int):
                payload["reasoning"] = {"max_tokens": self.config.reasoning_effort}
            else:
                payload["reasoning"] = {"effort": self.config.reasoning_effort}
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, sort_keys=True).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        reservation = self.config.max_call_reservation_microusd
        self.ledger.reserve(reservation)
        try:
            raw = self._dispatch(request)
            if raw.get("model") != model:
                raise RuntimeError("OpenRouter response model differs from pin")
            choice = raw["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = raw["usage"]
            prompt_tokens = _nonnegative_int("prompt_tokens", usage["prompt_tokens"])
            completion_tokens = _nonnegative_int(
                "completion_tokens", usage["completion_tokens"]
            )
            details = usage.get("completion_tokens_details", {})
            if not isinstance(details, dict):
                raise ValueError
            reasoning_tokens = _nonnegative_int(
                "reasoning_tokens", details.get("reasoning_tokens", 0)
            )
            numerator = (
                prompt_tokens * self.config.prompt_microusd_per_million
                + completion_tokens * self.config.completion_microusd_per_million
            )
            cost = (numerator + 999_999) // 1_000_000
            self.ledger.settle(reservation, cost)
            if not isinstance(content, str) or not content.strip():
                raise EmptyCompletion(
                    "OpenRouter returned empty completion content: "
                    f"finish_reason={finish_reason!r}, prompt_tokens={prompt_tokens}, "
                    f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}",
                    cost_microusd=cost,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
        except Exception as exc:
            if self.ledger.reserved_microusd >= reservation:
                self.ledger.settle(reservation, 0)
            if isinstance(exc, RuntimeError) and "OpenRouter" in str(exc):
                raise
            raise RuntimeError("OpenRouter returned an invalid completion response") from None
        normalized = {
            "model": model,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content},
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "cost_microusd": cost,
        }
        return LLMCompletion(
            content=content,
            cost_microusd=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response=normalized,
        )
