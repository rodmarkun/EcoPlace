"""Development-only Codex CLI completion engine required by Amendment A1.

The CLI is treated strictly as a text-generation transport.  Every dispatch runs in
its own empty temporary directory, with a read-only sandbox and no repository or user
configuration context.  The campaign runner remains the sole owner of retries and of
the seeds that identify those attempts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .feedback import LEGACY_REQUIRED_PROFILES, REQUIRED_PROFILES
from .llm_operator import LLMCompletion

CODEX_ENGINE_CONFIG_SCHEMA = "evolved-continuum-codex-cli-engine-v1"
CODEX_ATTEMPT_SCHEMA = "evolved-continuum-codex-cli-attempt-v1"
CODEX_COMPLETION_SCHEMA = "evolved-continuum-codex-cli-completion-v1"
CODEX_ENGINE_NAME = "codex-cli-subscription"
CODEX_MODEL_ID = "gpt-5.6-sol"
CODEX_REQUIRED_CLI_SERIES = "0.144.x"
A1_REFERENCE = (
    "ContinuumPlacementScopedDown.md#researcher-amendment-a1-2026-08-09"
)

FAILURE_TIMEOUT = "timeout"
FAILURE_NONZERO_EXIT = "nonzero-exit"
FAILURE_EMPTY_OUTPUT = "empty-output"
FAILURE_UNPARSEABLE_OUTPUT = "unparseable-output"
FAILURE_STARTUP_STALL = "startup-no-recognized-event"
FAILURE_REMOTE_ERROR = "remote-error"
FAILURE_CONTENT_FILTER = "content-filter-refusal"
FAILURE_CALL_CEILING = "campaign-call-ceiling"
FAILURE_WALL_CLOCK_CEILING = "campaign-wall-clock-ceiling"

TERMINAL_CALL_CEILING = "codex-cli-call-ceiling-exhausted"
TERMINAL_WALL_CLOCK_CEILING = "codex-cli-wall-clock-ceiling-exhausted"
TERMINAL_CIRCUIT_BREAKER = "codex-cli-consecutive-failure-circuit-open"

DEFAULT_STARTUP_GRACE_S = 10.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_JOURNAL_WATCHDOG_MULTIPLIER = 1.25
DEFAULT_RAW_OUTPUT_EXCERPT_BYTES = 4096

_RECOGNIZED_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "item.completed",
        "turn.completed",
        "turn.failed",
        "error",
    }
)

_EXPECTED_PROFILE_POLICY = {
    name: "development-only" for name in sorted(REQUIRED_PROFILES)
}
_LEGACY_PROFILE_POLICY = {
    name: "development-only" for name in sorted(LEGACY_REQUIRED_PROFILES)
}
_EXPECTED_GENERATION = {
    "sampling_control": "subscription-service-managed",
    "seed_role": "harness-retry-accounting-only",
}
_VERSION_PATTERN = re.compile(r"(?:codex-cli\s+)?(\d+)\.(\d+)\.(\d+)(?:\b|$)")
_REFUSAL_PREFIXES = (
    "i'm sorry, but i can't",
    "i’m sorry, but i can’t",
    "i cannot assist with",
    "i can't assist with",
    "i can’t assist with",
    "i cannot help with",
    "i can't help with",
    "i can’t help with",
)
_CONTENT_FILTER_CODES = {
    "content_filter",
    "content-filter",
    "content_policy_violation",
    "content-policy-violation",
    "refusal",
    "safety_refusal",
}


def _nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
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
class CodexEngineConfig:
    """Strict configuration for one development campaign's subscription engine."""

    status: str
    engine: str
    model_id: str
    cli_executable: str
    required_cli_series: str
    timeout_s: float
    max_calls_per_campaign: int
    max_wall_clock_s_per_campaign: float
    generation: dict[str, str]
    profile_policy: dict[str, str]
    a1_reference: str
    startup_grace_s: float = DEFAULT_STARTUP_GRACE_S
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    journal_watchdog_multiplier: float = DEFAULT_JOURNAL_WATCHDOG_MULTIPLIER
    raw_output_excerpt_bytes: int = DEFAULT_RAW_OUTPUT_EXCERPT_BYTES

    def __post_init__(self) -> None:
        if self.status != "development-only":
            raise ValueError("Codex subscription engine status must be development-only")
        if self.engine != CODEX_ENGINE_NAME or self.model_id != CODEX_MODEL_ID:
            raise ValueError("Codex subscription engine/model identity differs from A1")
        if not self.cli_executable or "\x00" in self.cli_executable:
            raise ValueError("Codex CLI executable must be explicit and non-empty")
        if self.required_cli_series != CODEX_REQUIRED_CLI_SERIES:
            raise ValueError("Codex CLI series must remain pinned to 0.144.x")
        _positive_number("timeout_s", self.timeout_s)
        if (
            not isinstance(self.max_calls_per_campaign, int)
            or isinstance(self.max_calls_per_campaign, bool)
            or self.max_calls_per_campaign < 1
        ):
            raise ValueError("max_calls_per_campaign must be a positive integer")
        _positive_number(
            "max_wall_clock_s_per_campaign", self.max_wall_clock_s_per_campaign
        )
        _positive_number("startup_grace_s", self.startup_grace_s)
        if (
            not isinstance(self.max_consecutive_failures, int)
            or isinstance(self.max_consecutive_failures, bool)
            or self.max_consecutive_failures < 1
        ):
            raise ValueError("max_consecutive_failures must be a positive integer")
        _positive_number(
            "journal_watchdog_multiplier", self.journal_watchdog_multiplier
        )
        if self.journal_watchdog_multiplier <= 1.0:
            raise ValueError("journal_watchdog_multiplier must be greater than 1")
        if (
            not isinstance(self.raw_output_excerpt_bytes, int)
            or isinstance(self.raw_output_excerpt_bytes, bool)
            or self.raw_output_excerpt_bytes < 256
        ):
            raise ValueError("raw_output_excerpt_bytes must be an integer of at least 256")
        if self.generation != _EXPECTED_GENERATION:
            raise ValueError("Codex generation controls must state subscription-managed sampling")
        if self.profile_policy not in (
            _LEGACY_PROFILE_POLICY,
            _EXPECTED_PROFILE_POLICY,
        ):
            raise ValueError(
                "Codex profile policy must be the exact legacy or enhanced development set"
            )
        if self.a1_reference != A1_REFERENCE:
            raise ValueError("Codex engine must cite Researcher Amendment A1")

    # These properties are the narrow model-config surface consumed by the existing
    # LLM proposal operator.  No fake weight or host identity is manufactured.
    @property
    def revision(self) -> None:
        return None

    @property
    def weight_set_sha256(self) -> None:
        return None

    @property
    def max_call_reservation_microusd(self) -> int:
        return 0


def load_codex_engine_config(path: str | Path) -> CodexEngineConfig:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != CODEX_ENGINE_CONFIG_SCHEMA:
        raise ValueError(f"Codex engine schema must be {CODEX_ENGINE_CONFIG_SCHEMA}")
    required = {
        "schema",
        "status",
        "engine",
        "model_id",
        "cli_executable",
        "required_cli_series",
        "timeout_s",
        "max_calls_per_campaign",
        "max_wall_clock_s_per_campaign",
        "generation",
        "profile_policy",
        "a1_reference",
    }
    optional = {
        "startup_grace_s",
        "max_consecutive_failures",
        "journal_watchdog_multiplier",
        "raw_output_excerpt_bytes",
    }
    if not required <= set(raw) or not set(raw) <= required | optional:
        raise ValueError(
            "Codex engine config fields must contain exactly the required fields plus "
            f"recognized robustness controls; required={sorted(required)}, "
            f"optional={sorted(optional)}"
        )
    return CodexEngineConfig(
        status=raw["status"],
        engine=raw["engine"],
        model_id=raw["model_id"],
        cli_executable=raw["cli_executable"],
        required_cli_series=raw["required_cli_series"],
        timeout_s=raw["timeout_s"],
        max_calls_per_campaign=raw["max_calls_per_campaign"],
        max_wall_clock_s_per_campaign=raw["max_wall_clock_s_per_campaign"],
        generation=raw["generation"],
        profile_policy=raw["profile_policy"],
        a1_reference=raw["a1_reference"],
        startup_grace_s=raw.get("startup_grace_s", DEFAULT_STARTUP_GRACE_S),
        max_consecutive_failures=raw.get(
            "max_consecutive_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES
        ),
        journal_watchdog_multiplier=raw.get(
            "journal_watchdog_multiplier", DEFAULT_JOURNAL_WATCHDOG_MULTIPLIER
        ),
        raw_output_excerpt_bytes=raw.get(
            "raw_output_excerpt_bytes", DEFAULT_RAW_OUTPUT_EXCERPT_BYTES
        ),
    )


def validate_codex_development_profile(
    config: CodexEngineConfig,
    *,
    development: bool,
    campaign_status: str,
    feedback_profile: str,
) -> None:
    """Refuse any profile that could give the A1 engine confirmatory authority."""
    if not development:
        raise ValueError("Codex CLI subscription engine requires the development override")
    if campaign_status != "development-provisional":
        raise ValueError("Codex CLI subscription engine requires a development campaign plan")
    if config.profile_policy.get(feedback_profile) != "development-only":
        raise ValueError(
            "Codex CLI subscription engine accepts only matched/E5 development profiles"
        )


def refuse_codex_confirmatory_engine(
    engine_descriptor: dict[str, Any] | None,
    *,
    boundary: str,
) -> None:
    """Explicit A1 guard shared by frozen-tag, authorization, and postflight paths."""
    if engine_descriptor is not None and engine_descriptor.get("engine") == CODEX_ENGINE_NAME:
        raise ValueError(
            f"{boundary} refuses development-only engine {CODEX_ENGINE_NAME!r} under A1"
        )


@dataclass
class CodexUsageLedger:
    """Per-campaign call, wall-clock, and token accounting."""

    max_calls: int
    max_wall_clock_ms: int
    calls: int = 0
    wall_clock_ms: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        self._journal_path: Path | None = None
        self._journal_events: list[dict[str, Any]] = []
        self._pending_dispatch: dict[str, Any] | None = None
        self._journal_lock_stream: Any | None = None
        if not isinstance(self.max_calls, int) or isinstance(self.max_calls, bool):
            raise ValueError("max_calls must be an integer")
        if self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        if not isinstance(self.max_wall_clock_ms, int) or self.max_wall_clock_ms < 1:
            raise ValueError("max_wall_clock_ms must be a positive integer")
        self._check()

    def _check(self) -> None:
        for field in (
            "calls",
            "wall_clock_ms",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "consecutive_failures",
        ):
            _nonnegative_int(field, getattr(self, field))
        if self.calls > self.max_calls:
            raise ValueError("Codex call ledger exceeds its campaign ceiling")
        if self.wall_clock_ms > self.max_wall_clock_ms:
            raise ValueError("Codex wall-clock ledger exceeds its campaign ceiling")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed total input tokens")

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.calls

    @property
    def remaining_wall_clock_ms(self) -> int:
        return self.max_wall_clock_ms - self.wall_clock_ms

    @staticmethod
    def _journal_sha256(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _read_journal(self) -> list[dict[str, Any]]:
        if self._journal_path is None or not self._journal_path.exists():
            return []
        encoded = self._journal_path.read_bytes()
        if encoded and not encoded.endswith(b"\n"):
            boundary = encoded.rfind(b"\n") + 1
            with self._journal_path.open("r+b") as stream:
                stream.truncate(boundary)
                stream.flush()
                os.fsync(stream.fileno())
            encoded = encoded[:boundary]
        events = []
        previous = "0" * 64
        for index, raw_line in enumerate(encoded.splitlines()):
            line = raw_line.decode()
            event = json.loads(line)
            if line != json.dumps(
                event, sort_keys=True, separators=(",", ":"), allow_nan=False
            ):
                raise ValueError(f"Codex usage journal event {index} is not canonical JSON")
            body = {key: value for key, value in event.items() if key != "event_sha256"}
            if (
                body.get("event_seq") != index
                or body.get("previous_event_sha256") != previous
                or event.get("event_sha256") != self._journal_sha256(body)
            ):
                raise ValueError(f"Codex usage journal chain failed at event {index}")
            previous = event["event_sha256"]
            events.append(event)
        return events

    def _append_journal(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._journal_path is None:
            return
        body = {
            "event_seq": len(self._journal_events),
            "previous_event_sha256": (
                self._journal_events[-1]["event_sha256"]
                if self._journal_events
                else "0" * 64
            ),
            "type": event_type,
            "payload": payload,
        }
        event = {**body, "event_sha256": self._journal_sha256(body)}
        encoded = json.dumps(
            event, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._journal_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._journal_events.append(event)

    def _replay_journal(self) -> None:
        self.calls = 0
        self.wall_clock_ms = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_output_tokens = 0
        self.consecutive_failures = 0
        pending = None
        reserved_count = 0
        for event in self._journal_events:
            event_type = event.get("type")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Codex usage journal payload must be an object")
            if event_type == "dispatch_reserved":
                if pending is not None or payload.get("call_seq") != reserved_count:
                    raise ValueError("Codex usage journal dispatch sequence is invalid")
                reserved_ms = payload.get("reserved_wall_clock_ms")
                request_sha256 = payload.get("request_sha256")
                seed = payload.get("seed")
                if (
                    not isinstance(reserved_ms, int)
                    or isinstance(reserved_ms, bool)
                    or reserved_ms < 1
                    or not isinstance(request_sha256, str)
                    or len(request_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in request_sha256)
                    or not isinstance(seed, int)
                    or isinstance(seed, bool)
                    or not 0 <= seed < 2**63
                ):
                    raise ValueError("Codex usage journal reservation is invalid")
                pending = payload
                reserved_count += 1
                self.calls += 1
                self.wall_clock_ms += reserved_ms
                self._check()
                continue
            if event_type == "dispatch_progress":
                if pending is None or payload.get("call_seq") != pending["call_seq"]:
                    raise ValueError("Codex usage journal progress lacks its reservation")
                observed_duration_ms = payload.get("observed_duration_ms")
                output_bytes = payload.get("output_bytes")
                stream = payload.get("stream")
                event_name = payload.get("event_type")
                if (
                    not isinstance(observed_duration_ms, int)
                    or isinstance(observed_duration_ms, bool)
                    or not 0 <= observed_duration_ms <= pending["reserved_wall_clock_ms"]
                    or not isinstance(output_bytes, int)
                    or isinstance(output_bytes, bool)
                    or output_bytes < 1
                    or stream != "stdout"
                    or event_name not in _RECOGNIZED_EVENT_TYPES
                ):
                    raise ValueError("Codex usage journal progress is invalid")
                continue
            if event_type not in {"dispatch_settled", "dispatch_abandoned"}:
                raise ValueError(f"unknown Codex usage journal event {event_type!r}")
            if pending is None or payload.get("call_seq") != pending["call_seq"]:
                raise ValueError("Codex usage journal settlement lacks its reservation")
            duration_ms = payload.get("duration_ms")
            if (
                not isinstance(duration_ms, int)
                or isinstance(duration_ms, bool)
                or not 0 <= duration_ms <= pending["reserved_wall_clock_ms"]
            ):
                raise ValueError("Codex usage journal duration is invalid")
            usage = payload.get("usage")
            if not isinstance(usage, dict) or set(usage) != set(_zero_usage()):
                raise ValueError("Codex usage journal token accounting is invalid")
            checked_usage = {name: _nonnegative_int(name, usage[name]) for name in usage}
            if event_type == "dispatch_abandoned":
                if duration_ms != pending["reserved_wall_clock_ms"] or any(
                    checked_usage.values()
                ):
                    raise ValueError("an abandoned Codex dispatch must consume its reservation")
                self.consecutive_failures += 1
            else:
                self.wall_clock_ms -= pending["reserved_wall_clock_ms"] - duration_ms
                failure_category = payload.get("failure_category")
                if failure_category is not None and (
                    not isinstance(failure_category, str) or not failure_category
                ):
                    raise ValueError("Codex dispatch failure category is invalid")
                self.consecutive_failures = (
                    self.consecutive_failures + 1
                    if failure_category is not None
                    else 0
                )
            self.input_tokens += checked_usage["input_tokens"]
            self.cached_input_tokens += checked_usage["cached_input_tokens"]
            self.output_tokens += checked_usage["output_tokens"]
            self.reasoning_output_tokens += checked_usage["reasoning_output_tokens"]
            pending = None
            self._check()
        self._pending_dispatch = pending

    def bind_journal(self, path: str | Path) -> None:
        """Bind durable pre-dispatch reservations before any completion can run.

        An unterminated reservation means the prior process could have dispatched the
        external call. Resume therefore charges its full per-call wall reservation and
        never makes that call disappear from the hard campaign ceiling.
        """
        journal_path = Path(path)
        if self._journal_path is not None:
            if self._journal_path != journal_path:
                raise ValueError("Codex usage ledger is already bound to another journal")
            return
        if any(
            getattr(self, field) != 0
            for field in (
                "calls",
                "wall_clock_ms",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            )
        ):
            raise ValueError("Codex usage ledger must be empty before journal binding")
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._journal_path = journal_path
        lock_stream = journal_path.with_name(journal_path.name + ".lock").open("a+")
        try:
            fcntl.flock(
                lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            lock_stream.close()
            self._journal_path = None
            raise RuntimeError(
                "Codex usage journal already has an active campaign process; "
                f"concurrent resume refused before dispatch: {journal_path}"
            ) from exc
        self._journal_lock_stream = lock_stream
        try:
            self._journal_events = self._read_journal()
            self._replay_journal()
            if self._pending_dispatch is not None:
                pending = self._pending_dispatch
                self._append_journal(
                    "dispatch_abandoned",
                    {
                        "call_seq": pending["call_seq"],
                        "duration_ms": pending["reserved_wall_clock_ms"],
                        "usage": _zero_usage(),
                    },
                )
                self._replay_journal()
        except Exception:
            self.release_journal_lease()
            raise

    def release_journal_lease(self) -> None:
        if self._journal_lock_stream is None:
            return
        fcntl.flock(self._journal_lock_stream.fileno(), fcntl.LOCK_UN)
        self._journal_lock_stream.close()
        self._journal_lock_stream = None

    def reserve_dispatch(
        self, *, reserved_wall_clock_ms: int, request_sha256: str, seed: int
    ) -> int:
        if self._pending_dispatch is not None:
            raise ValueError("Codex usage ledger already has a pending dispatch")
        if self.remaining_calls <= 0:
            raise ValueError("Codex call ceiling exhausted")
        if (
            not isinstance(reserved_wall_clock_ms, int)
            or isinstance(reserved_wall_clock_ms, bool)
            or not 1 <= reserved_wall_clock_ms <= self.remaining_wall_clock_ms
        ):
            raise ValueError("Codex dispatch wall-clock reservation is invalid")
        payload = {
            "call_seq": self.calls,
            "reserved_wall_clock_ms": reserved_wall_clock_ms,
            "request_sha256": request_sha256,
            "seed": seed,
        }
        self._append_journal("dispatch_reserved", payload)
        self.calls += 1
        self.wall_clock_ms += reserved_wall_clock_ms
        self._pending_dispatch = payload
        self._check()
        return payload["call_seq"]

    def settle_dispatch(
        self,
        *,
        call_seq: int,
        duration_ms: int,
        usage: dict[str, int],
        failure_category: str | None = None,
    ) -> None:
        pending = self._pending_dispatch
        if pending is None or call_seq != pending["call_seq"]:
            raise ValueError("Codex dispatch settlement differs from its reservation")
        duration_ms = _nonnegative_int("duration_ms", duration_ms)
        if duration_ms > pending["reserved_wall_clock_ms"]:
            raise ValueError("Codex dispatch duration exceeds its wall-clock reservation")
        checked_usage = {
            name: _nonnegative_int(name, usage[name]) for name in _zero_usage()
        }
        if failure_category is not None and (
            not isinstance(failure_category, str) or not failure_category
        ):
            raise ValueError("failure_category must be a non-empty string or None")
        self._append_journal(
            "dispatch_settled",
            {
                "call_seq": call_seq,
                "duration_ms": duration_ms,
                "usage": checked_usage,
                "failure_category": failure_category,
            },
        )
        self.wall_clock_ms -= pending["reserved_wall_clock_ms"] - duration_ms
        self.input_tokens += checked_usage["input_tokens"]
        self.cached_input_tokens += checked_usage["cached_input_tokens"]
        self.output_tokens += checked_usage["output_tokens"]
        self.reasoning_output_tokens += checked_usage["reasoning_output_tokens"]
        self.consecutive_failures = (
            self.consecutive_failures + 1 if failure_category is not None else 0
        )
        self._pending_dispatch = None
        self._check()

    def record_progress(
        self,
        *,
        call_seq: int,
        observed_duration_ms: int,
        event_type: str,
        output_bytes: int,
    ) -> None:
        pending = self._pending_dispatch
        if pending is None or call_seq != pending["call_seq"]:
            raise ValueError("Codex dispatch progress differs from its reservation")
        observed_duration_ms = _nonnegative_int(
            "observed_duration_ms", observed_duration_ms
        )
        output_bytes = _nonnegative_int("output_bytes", output_bytes)
        if observed_duration_ms > pending["reserved_wall_clock_ms"]:
            observed_duration_ms = pending["reserved_wall_clock_ms"]
        if event_type not in _RECOGNIZED_EVENT_TYPES or output_bytes < 1:
            raise ValueError("Codex dispatch progress event is invalid")
        self._append_journal(
            "dispatch_progress",
            {
                "call_seq": call_seq,
                "observed_duration_ms": observed_duration_ms,
                "stream": "stdout",
                "event_type": event_type,
                "output_bytes": output_bytes,
            },
        )

    def account(self, *, duration_ms: int, usage: dict[str, int]) -> None:
        if self._journal_path is not None:
            raise ValueError("journal-bound Codex usage must use dispatch reservation")
        duration_ms = _nonnegative_int("duration_ms", duration_ms)
        if self.calls >= self.max_calls:
            raise ValueError("Codex call ceiling exhausted")
        if duration_ms > self.remaining_wall_clock_ms:
            raise ValueError("Codex wall-clock ceiling exhausted")
        self.calls += 1
        self.wall_clock_ms += duration_ms
        self.input_tokens += usage["input_tokens"]
        self.cached_input_tokens += usage["cached_input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.reasoning_output_tokens += usage["reasoning_output_tokens"]
        self.consecutive_failures = 0
        self._check()

    def restore(self, deltas: list[dict[str, Any]]) -> None:
        if self._journal_path is not None:
            settled = {
                event["payload"]["call_seq"]: event["payload"]
                for event in self._journal_events
                if event["type"] == "dispatch_settled"
            }
            archived_call_seqs = set()
            for delta in deltas:
                if not delta.get("dispatched"):
                    continue
                call_seq = delta.get("journal_call_seq")
                if call_seq in archived_call_seqs or call_seq not in settled:
                    raise ValueError("archive call is absent or duplicated in Codex usage journal")
                archived_call_seqs.add(call_seq)
                journal = settled[call_seq]
                if delta.get("duration_ms") != journal["duration_ms"] or any(
                    delta.get(name) != journal["usage"][name] for name in _zero_usage()
                ):
                    raise ValueError("archive call differs from Codex usage journal")
            return
        if any(
            getattr(self, field) != 0
            for field in (
                "calls",
                "wall_clock_ms",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            )
        ):
            raise ValueError("Codex usage ledger must be empty before archive restoration")
        for delta in deltas:
            if not delta.get("dispatched"):
                continue
            self.account(
                duration_ms=delta["duration_ms"],
                usage={
                    name: _nonnegative_int(name, delta[name])
                    for name in (
                        "input_tokens",
                        "cached_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                    )
                },
            )
            if delta.get("cumulative") != self.to_dict():
                raise ValueError("Codex usage ledger differs from resumable archive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_calls": self.max_calls,
            "calls": self.calls,
            "remaining_calls": self.remaining_calls,
            "max_wall_clock_seconds": self.max_wall_clock_ms / 1000.0,
            "wall_clock_seconds": self.wall_clock_ms / 1000.0,
            "remaining_wall_clock_seconds": self.remaining_wall_clock_ms / 1000.0,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "consecutive_failures": self.consecutive_failures,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


class CodexCLICompletionError(RuntimeError):
    """A classified CLI failure whose evidence is safe to retain in an archive."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        archive_metadata: dict[str, Any],
        campaign_stop_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.archive_metadata = archive_metadata
        self.campaign_stop_reason = campaign_stop_reason


@dataclass(frozen=True)
class _ParsedOutput:
    events: list[dict[str, Any]]
    session_id: str | None
    completion: str | None
    usage: dict[str, int]
    parse_error: str | None


@dataclass(frozen=True)
class _ProcessResult:
    stdout: str
    stderr: str
    exit_code: int | None
    measured_duration_ms: int
    timed_out: bool
    startup_stalled: bool
    remote_error_event_type: str | None
    seen_event_types: tuple[str, ...]


def _zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def _parse_jsonl(stdout: str) -> _ParsedOutput:
    events: list[dict[str, Any]] = []
    for index, line in enumerate(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            return _ParsedOutput(
                events=events,
                session_id=None,
                completion=None,
                usage=_zero_usage(),
                parse_error=f"line {index + 1} is not JSON: {exc.msg}",
            )
        if not isinstance(event, dict):
            return _ParsedOutput(
                events=events,
                session_id=None,
                completion=None,
                usage=_zero_usage(),
                parse_error=f"line {index + 1} is not a JSON object",
            )
        events.append(event)

    session_ids = [
        event.get("thread_id")
        for event in events
        if event.get("type") == "thread.started"
    ]
    session_id = session_ids[-1] if session_ids else None
    messages = [
        event.get("item", {}).get("text")
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "agent_message"
    ]
    completion = messages[-1] if messages else None
    usage_events = [
        event.get("usage") for event in events if event.get("type") == "turn.completed"
    ]
    usage = _zero_usage()
    if usage_events:
        raw_usage = usage_events[-1]
        if not isinstance(raw_usage, dict):
            return _ParsedOutput(
                events=events,
                session_id=session_id,
                completion=completion,
                usage=usage,
                parse_error="turn.completed usage is not an object",
            )
        for name in usage:
            value = raw_usage.get(name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return _ParsedOutput(
                    events=events,
                    session_id=session_id,
                    completion=completion,
                    usage=_zero_usage(),
                    parse_error=f"turn.completed {name} is not a non-negative integer",
                )
            usage[name] = value
        if usage["cached_input_tokens"] > usage["input_tokens"]:
            return _ParsedOutput(
                events=events,
                session_id=session_id,
                completion=completion,
                usage=_zero_usage(),
                parse_error="cached_input_tokens exceeds input_tokens",
            )
    return _ParsedOutput(
        events=events,
        session_id=session_id if isinstance(session_id, str) and session_id else None,
        completion=completion if isinstance(completion, str) else None,
        usage=usage,
        parse_error=None,
    )


def _has_content_filter_event(events: list[dict[str, Any]]) -> bool:
    def visit(value: object) -> bool:
        if isinstance(value, dict):
            return any(visit(key) or visit(item) for key, item in value.items())
        if isinstance(value, list):
            return any(visit(item) for item in value)
        if isinstance(value, str):
            lowered = value.lower().strip()
            return lowered in _CONTENT_FILTER_CODES or any(
                code in lowered for code in _CONTENT_FILTER_CODES
            )
        return False

    return any(visit(event) for event in events)


def _is_refusal(completion: str) -> bool:
    lowered = completion.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _REFUSAL_PREFIXES)


def _bounded_raw_excerpt(stdout: str, stderr: str, limit_bytes: int) -> str:
    combined = f"stdout={stdout!r}; stderr={stderr!r}"
    encoded = combined.encode(errors="replace")
    if len(encoded) <= limit_bytes:
        return combined
    suffix = b"...<bounded>"
    return (encoded[: limit_bytes - len(suffix)] + suffix).decode(errors="replace")


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:  # pragma: no cover - process-exit race
        return
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - process-exit race
            pass


class CodexCLICompletionClient:
    """One-process-per-attempt implementation of the shared completion interface."""

    operator_name = "codex_cli_subscription"
    development_only = True

    def __init__(self, config: CodexEngineConfig) -> None:
        self.config = config
        executable = shutil.which(config.cli_executable)
        if executable is None:
            raise FileNotFoundError(f"Codex CLI executable not found: {config.cli_executable}")
        self.executable = str(Path(executable).resolve())
        self._runtime_home: tempfile.TemporaryDirectory[str] | None = None
        self._subprocess_environment = self._build_subprocess_environment()
        self.cli_version = self._read_and_validate_version()
        self.engine_descriptor = {
            "engine": CODEX_ENGINE_NAME,
            "model_id": config.model_id,
            "cli_version": self.cli_version,
            "a1_reference": config.a1_reference,
            "development_only": True,
            "live_deterministic": False,
            "replay_deterministic": True,
            "sampling_control": "subscription-service-managed",
        }
        self.usage_ledger = CodexUsageLedger(
            max_calls=config.max_calls_per_campaign,
            max_wall_clock_ms=math.ceil(
                config.max_wall_clock_s_per_campaign * 1000
            ),
        )

    def _build_subprocess_environment(self) -> dict[str, str]:
        """Give every engine client private writable CLI state.

        The user's model cache may be rewritten by a different Codex CLI build while a
        campaign is live. Copy only the auth receipt into a temporary CODEX_HOME so one
        campaign client owns its cache schema and TTL for its full lifetime.
        """
        source_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        source_auth = source_root / "auth.json"
        if not source_auth.is_file():
            raise RuntimeError("nested Codex CLI needs a readable authenticated auth.json")
        self._runtime_home = tempfile.TemporaryDirectory(prefix="ec-codex-runtime-")
        runtime_root = Path(self._runtime_home.name)
        runtime_auth = runtime_root / "auth.json"
        shutil.copyfile(source_auth, runtime_auth)
        runtime_auth.chmod(0o600)
        environment = os.environ.copy()
        environment["CODEX_HOME"] = str(runtime_root)
        environment.pop("CODEX_SANDBOX", None)
        environment.pop("CODEX_SANDBOX_NETWORK_DISABLED", None)
        environment.pop("CODEX_THREAD_ID", None)
        # A1 requires the authenticated subscription, never an accidentally inherited
        # Platform key with different cost and identity semantics.
        environment.pop("CODEX_API_KEY", None)
        environment.pop("OPENAI_API_KEY", None)
        return environment

    def _read_and_validate_version(self) -> str:
        with tempfile.TemporaryDirectory(prefix="ec-codex-version-") as workspace:
            result = subprocess.run(
                [self.executable, "--version"],
                cwd=workspace,
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
                env=self._subprocess_environment,
            )
        raw = (result.stdout or result.stderr).strip()
        match = _VERSION_PATTERN.search(raw)
        if result.returncode != 0 or match is None:
            raise RuntimeError("unable to read a semantic version from Codex CLI")
        major, minor, _patch = (int(part) for part in match.groups())
        if (major, minor) != (0, 144):
            raise RuntimeError(
                f"Codex CLI {match.group(0)!r} is outside required series 0.144.x"
            )
        return ".".join(match.groups())

    def _normalized_command(self) -> list[str]:
        return [
            Path(self.executable).name,
            "exec",
            "-",
            "-m",
            self.config.model_id,
            "--sandbox",
            "read-only",
            "-C",
            "<isolated-empty-directory>",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--color",
            "never",
            "--json",
        ]

    def _command(self, workspace: str) -> list[str]:
        command = self._normalized_command()
        command[command.index("<isolated-empty-directory>")] = workspace
        command[0] = self.executable
        return command

    def _run_command(
        self,
        *,
        workspace: str,
        prompt: str,
        timeout_s: float,
        journal_call_seq: int,
    ) -> _ProcessResult:
        """Stream both pipes so startup drift is visible before the hard timeout."""
        started = time.monotonic()
        deadline = started + timeout_s
        startup_deadline = started + min(self.config.startup_grace_s, timeout_s)
        process = subprocess.Popen(
            self._command(workspace),
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._subprocess_environment,
            start_new_session=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(prompt.encode())
        process.stdin.close()

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        stdout_line_buffer = bytearray()
        seen_event_types: set[str] = set()
        recognized_event_seen = False
        timed_out = False
        startup_stalled = False
        remote_error_event_type = None
        stopped = False

        try:
            while selector.get_map():
                now = time.monotonic()
                if not stopped and now >= deadline:
                    timed_out = True
                    stopped = True
                    _stop_process_group(process)
                elif not stopped and not recognized_event_seen and now >= startup_deadline:
                    startup_stalled = True
                    stopped = True
                    _stop_process_group(process)

                next_deadline = deadline
                if not recognized_event_seen:
                    next_deadline = min(next_deadline, startup_deadline)
                wait_s = max(0.0, min(0.1, next_deadline - time.monotonic()))
                for key, _mask in selector.select(wait_s):
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    stream_name = key.data
                    buffers[stream_name].extend(chunk)
                    if stream_name != "stdout":
                        continue
                    stdout_line_buffer.extend(chunk)
                    while b"\n" in stdout_line_buffer:
                        raw_line, _, remainder = stdout_line_buffer.partition(b"\n")
                        stdout_line_buffer = bytearray(remainder)
                        try:
                            event = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if not isinstance(event, dict) or not isinstance(
                            event.get("type"), str
                        ):
                            continue
                        event_type = event["type"]
                        seen_event_types.add(event_type)
                        if event_type not in _RECOGNIZED_EVENT_TYPES:
                            continue
                        recognized_event_seen = True
                        observed_duration_ms = max(
                            0, math.ceil((time.monotonic() - started) * 1000)
                        )
                        self.usage_ledger.record_progress(
                            call_seq=journal_call_seq,
                            observed_duration_ms=observed_duration_ms,
                            event_type=event_type,
                            output_bytes=len(buffers["stdout"]),
                        )
                        if event_type in {"error", "turn.failed"} and not stopped:
                            remote_error_event_type = event_type
                            stopped = True
                            _stop_process_group(process)

                if process.poll() is not None and not selector.get_map():
                    break
        finally:
            selector.close()
            if process.poll() is None:  # pragma: no cover - defensive cleanup
                _stop_process_group(process)
        exit_code = process.wait()
        measured_duration_ms = max(
            1, math.ceil((time.monotonic() - started) * 1000)
        )
        return _ProcessResult(
            stdout=bytes(buffers["stdout"]).decode(errors="replace"),
            stderr=bytes(buffers["stderr"]).decode(errors="replace"),
            exit_code=exit_code,
            measured_duration_ms=measured_duration_ms,
            timed_out=timed_out,
            startup_stalled=startup_stalled,
            remote_error_event_type=remote_error_event_type,
            seen_event_types=tuple(sorted(seen_event_types)),
        )

    def _attempt_record(
        self,
        *,
        prompt: str,
        generation: dict[str, Any],
        seed: int,
        dispatched: bool,
        observed_duration_ms: int,
        accounted_duration_ms: int,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        parsed: _ParsedOutput,
        failure_category: str | None,
        journal_call_seq: int | None,
    ) -> dict[str, Any]:
        delta = {
            "dispatched": dispatched,
            "duration_ms": accounted_duration_ms,
            "observed_duration_ms": observed_duration_ms,
            "journal_call_seq": journal_call_seq,
            **parsed.usage,
            "cumulative": self.usage_ledger.to_dict(),
        }
        return {
            "schema": CODEX_ATTEMPT_SCHEMA,
            "engine_descriptor": self.engine_descriptor,
            "request": {
                "model": self.config.model_id,
                "seed": seed,
                "generation": generation,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_bytes": len(prompt.encode()),
            },
            "process": {
                "argv": self._normalized_command(),
                "cwd_role": "fresh-isolated-empty-temporary-directory",
                "sandbox": "read-only",
                "startup_grace_seconds": self.config.startup_grace_s,
                "runtime_state": (
                    "ephemeral-authenticated-cli-state"
                    if self._subprocess_environment is not None
                    else "researcher-authenticated-cli-state"
                ),
                "exit_code": exit_code,
                "duration_ms": observed_duration_ms,
                "accounted_duration_ms": accounted_duration_ms,
                "dispatched": dispatched,
                "environment_overrides": (
                    {
                        "CODEX_HOME": "fresh-directory-containing-auth-json-only",
                        "unset": [
                            "CODEX_SANDBOX",
                            "CODEX_SANDBOX_NETWORK_DISABLED",
                            "CODEX_THREAD_ID",
                            "CODEX_API_KEY",
                            "OPENAI_API_KEY",
                        ],
                    }
                    if self._subprocess_environment is not None
                    else {"mode": "inherited-exactly"}
                ),
            },
            "raw_output": {"stdout": stdout, "stderr": stderr},
            "events": parsed.events,
            "parsed_completion": parsed.completion,
            "parse_error": parsed.parse_error,
            "session_id": parsed.session_id,
            "token_usage": parsed.usage,
            "failure_category": failure_category,
            "accounting_delta": delta,
        }

    def _archive_metadata(
        self,
        attempt: dict[str, Any],
        *,
        failure_category: str | None,
        campaign_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        usage = attempt["token_usage"]
        return {
            "engine_descriptor": self.engine_descriptor,
            "raw_response": {
                "schema": CODEX_COMPLETION_SCHEMA,
                "engine_descriptor": self.engine_descriptor,
                "attempts": [attempt],
                "parsed_completion": attempt["parsed_completion"],
                "session_id": attempt["session_id"],
                "token_usage": usage,
                "accounting": self.usage_ledger.to_dict(),
            },
            "failure_category": failure_category,
            "campaign_stop_reason": campaign_stop_reason,
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "cached_prompt_tokens": usage["cached_input_tokens"],
            "reasoning_output_tokens": usage["reasoning_output_tokens"],
            "codex_usage_ledger": self.usage_ledger.to_dict(),
        }

    def _raise_pre_dispatch_ceiling(
        self,
        *,
        prompt: str,
        generation: dict[str, Any],
        seed: int,
        category: str,
        terminal_reason: str,
    ) -> None:
        parsed = _ParsedOutput([], None, None, _zero_usage(), None)
        attempt = self._attempt_record(
            prompt=prompt,
            generation=generation,
            seed=seed,
            dispatched=False,
            observed_duration_ms=0,
            accounted_duration_ms=0,
            exit_code=None,
            stdout="",
            stderr="",
            parsed=parsed,
            failure_category=category,
            journal_call_seq=None,
        )
        raise CodexCLICompletionError(
            category,
            terminal_reason,
            archive_metadata=self._archive_metadata(
                attempt,
                failure_category=category,
                campaign_stop_reason=terminal_reason,
            ),
            campaign_stop_reason=terminal_reason,
        )

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        generation: dict[str, Any],
        seed: int,
    ) -> LLMCompletion:
        if model != self.config.model_id or generation != self.config.generation:
            raise ValueError("Codex completion request differs from validated engine config")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError("Codex attempt seed must be an explicit integer in [0, 2**63)")
        if (
            self.usage_ledger.consecutive_failures
            >= self.config.max_consecutive_failures
        ):
            self._raise_pre_dispatch_ceiling(
                prompt=prompt,
                generation=generation,
                seed=seed,
                category=FAILURE_REMOTE_ERROR,
                terminal_reason=TERMINAL_CIRCUIT_BREAKER,
            )
        if self.usage_ledger.remaining_calls <= 0:
            self._raise_pre_dispatch_ceiling(
                prompt=prompt,
                generation=generation,
                seed=seed,
                category=FAILURE_CALL_CEILING,
                terminal_reason=TERMINAL_CALL_CEILING,
            )
        if self.usage_ledger.remaining_wall_clock_ms <= 0:
            self._raise_pre_dispatch_ceiling(
                prompt=prompt,
                generation=generation,
                seed=seed,
                category=FAILURE_WALL_CLOCK_CEILING,
                terminal_reason=TERMINAL_WALL_CLOCK_CEILING,
            )

        remaining_wall_clock_ms = self.usage_ledger.remaining_wall_clock_ms
        configured_timeout_ms = math.ceil(self.config.timeout_s * 1000)
        reserved_wall_clock_ms = min(configured_timeout_ms, remaining_wall_clock_ms)
        timeout_s = reserved_wall_clock_ms / 1000.0
        wall_clock_limited = remaining_wall_clock_ms <= configured_timeout_ms
        journal_call_seq = self.usage_ledger.reserve_dispatch(
            reserved_wall_clock_ms=reserved_wall_clock_ms,
            request_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            seed=seed,
        )
        with tempfile.TemporaryDirectory(prefix="ec-codex-proposal-") as workspace:
            if any(Path(workspace).iterdir()):  # pragma: no cover - OS contract guard
                raise RuntimeError("Codex proposal workspace was not empty before dispatch")
            process_result = self._run_command(
                workspace=workspace,
                prompt=prompt,
                timeout_s=timeout_s,
                journal_call_seq=journal_call_seq,
            )
        stdout = process_result.stdout
        stderr = process_result.stderr
        exit_code = (
            None
            if process_result.timed_out or process_result.startup_stalled
            else process_result.exit_code
        )
        measured_duration_ms = process_result.measured_duration_ms
        accounted_duration_ms = min(measured_duration_ms, reserved_wall_clock_ms)
        exceeded_wall_clock = measured_duration_ms > reserved_wall_clock_ms
        parsed = _parse_jsonl(stdout)

        category = None
        campaign_stop_reason = None
        failure_detail = None
        if process_result.startup_stalled:
            category = FAILURE_STARTUP_STALL
            failure_detail = (
                f"no recognized JSONL event within {self.config.startup_grace_s:g}s; "
                f"expected one of {sorted(_RECOGNIZED_EVENT_TYPES)!r}; "
                f"saw event types {list(process_result.seen_event_types)!r}"
            )
        elif process_result.remote_error_event_type is not None:
            category = FAILURE_REMOTE_ERROR
            failure_detail = (
                "Codex JSONL stream emitted terminal error event "
                f"{process_result.remote_error_event_type!r}"
            )
        elif process_result.timed_out:
            category = FAILURE_TIMEOUT
            failure_detail = f"hard timeout after {timeout_s:g}s"
            if wall_clock_limited:
                campaign_stop_reason = TERMINAL_WALL_CLOCK_CEILING
        elif exceeded_wall_clock:
            category = FAILURE_TIMEOUT
            failure_detail = "process exceeded its reserved wall-clock duration"
            campaign_stop_reason = TERMINAL_WALL_CLOCK_CEILING
        elif exit_code != 0:
            category = FAILURE_NONZERO_EXIT
            failure_detail = f"process exited with status {exit_code}"
        elif not stdout.strip():
            category = FAILURE_EMPTY_OUTPUT
            failure_detail = "stdout was empty"
        elif parsed.parse_error is not None:
            category = FAILURE_UNPARSEABLE_OUTPUT
            failure_detail = parsed.parse_error
        elif _has_content_filter_event(parsed.events):
            category = FAILURE_CONTENT_FILTER
            failure_detail = "JSONL stream contained a content-filter event"
        elif parsed.session_id is None:
            category = FAILURE_UNPARSEABLE_OUTPUT
            parsed = _ParsedOutput(
                parsed.events,
                None,
                parsed.completion,
                parsed.usage,
                "thread.started did not report a session id",
            )
            failure_detail = parsed.parse_error
        elif parsed.completion is None:
            category = FAILURE_UNPARSEABLE_OUTPUT
            parsed = _ParsedOutput(
                parsed.events,
                parsed.session_id,
                None,
                parsed.usage,
                "JSONL stream did not contain a final agent message",
            )
            failure_detail = parsed.parse_error
        elif not parsed.completion.strip():
            category = FAILURE_EMPTY_OUTPUT
            failure_detail = "final agent message was empty"
        elif not any(event.get("type") == "turn.completed" for event in parsed.events):
            category = FAILURE_UNPARSEABLE_OUTPUT
            parsed = _ParsedOutput(
                parsed.events,
                parsed.session_id,
                parsed.completion,
                parsed.usage,
                "JSONL stream did not contain turn.completed usage",
            )
            failure_detail = parsed.parse_error
        elif _is_refusal(parsed.completion):
            category = FAILURE_CONTENT_FILTER
            failure_detail = "final agent message was a refusal"

        if category == FAILURE_UNPARSEABLE_OUTPUT:
            failure_detail = (
                f"{failure_detail}; expected event types include 'thread.started', "
                "'item.completed' agent_message, and 'turn.completed'; "
                f"saw event types {list(process_result.seen_event_types)!r}"
            )

        self.usage_ledger.settle_dispatch(
            call_seq=journal_call_seq,
            duration_ms=accounted_duration_ms,
            usage=parsed.usage,
            failure_category=category,
        )
        if (
            category is not None
            and campaign_stop_reason is None
            and self.usage_ledger.consecutive_failures
            >= self.config.max_consecutive_failures
        ):
            campaign_stop_reason = TERMINAL_CIRCUIT_BREAKER

        attempt = self._attempt_record(
            prompt=prompt,
            generation=generation,
            seed=seed,
            dispatched=True,
            observed_duration_ms=measured_duration_ms,
            accounted_duration_ms=accounted_duration_ms,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            parsed=parsed,
            failure_category=category,
            journal_call_seq=journal_call_seq,
        )
        metadata = self._archive_metadata(
            attempt,
            failure_category=category,
            campaign_stop_reason=campaign_stop_reason,
        )
        if category is not None:
            excerpt = _bounded_raw_excerpt(
                stdout, stderr, self.config.raw_output_excerpt_bytes
            )
            raise CodexCLICompletionError(
                category,
                f"Codex CLI completion failed: {category}; {failure_detail}; "
                f"raw excerpt: {excerpt}",
                archive_metadata=metadata,
                campaign_stop_reason=campaign_stop_reason,
            )

        return LLMCompletion(
            content=parsed.completion,
            cost_microusd=None,
            prompt_tokens=parsed.usage["input_tokens"],
            completion_tokens=parsed.usage["output_tokens"],
            raw_response=metadata["raw_response"],
        )


__all__ = [
    "A1_REFERENCE",
    "CODEX_ENGINE_CONFIG_SCHEMA",
    "CODEX_ENGINE_NAME",
    "CODEX_MODEL_ID",
    "CODEX_REQUIRED_CLI_SERIES",
    "CodexCLICompletionClient",
    "CodexCLICompletionError",
    "CodexEngineConfig",
    "CodexUsageLedger",
    "FAILURE_CALL_CEILING",
    "FAILURE_CONTENT_FILTER",
    "FAILURE_EMPTY_OUTPUT",
    "FAILURE_NONZERO_EXIT",
    "FAILURE_TIMEOUT",
    "FAILURE_UNPARSEABLE_OUTPUT",
    "FAILURE_WALL_CLOCK_CEILING",
    "TERMINAL_CALL_CEILING",
    "TERMINAL_WALL_CLOCK_CEILING",
    "load_codex_engine_config",
    "refuse_codex_confirmatory_engine",
    "validate_codex_development_profile",
]
