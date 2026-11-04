#!/usr/bin/env python3
"""Canned Codex CLI-shaped process used by tests/test_codex_engine.py."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _next_mode() -> tuple[str, int]:
    state_path = Path(os.environ["CODEX_STUB_STATE"])
    index = int(state_path.read_text()) if state_path.exists() else 0
    state_path.write_text(str(index + 1))
    sequence = os.environ.get("CODEX_STUB_SEQUENCE", "success").split(",")
    return sequence[min(index, len(sequence) - 1)].strip(), index


def _emit(event: dict[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def main() -> int:
    if sys.argv[1:] == ["--version"]:
        print(os.environ.get("CODEX_STUB_VERSION", "codex-cli 0.144.99"))
        return 0
    if len(sys.argv) < 3 or sys.argv[1:3] != ["exec", "-"]:
        print("unsupported stub invocation", file=sys.stderr)
        return 64

    prompt = sys.stdin.read()
    mode, index = _next_mode()
    invocation = {
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "cwd_entries_before": sorted(path.name for path in Path.cwd().iterdir()),
        "prompt": prompt,
        "mode": mode,
        "attempt_index": index,
    }
    log_path = os.environ.get("CODEX_STUB_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(invocation, sort_keys=True) + "\n")

    if mode == "timeout":
        time.sleep(float(os.environ.get("CODEX_STUB_SLEEP_S", "2")))
        return 0
    if mode == "nonzero":
        _emit({"type": "thread.started", "thread_id": f"stub-session-{index}"})
        print("stub nonzero failure", file=sys.stderr)
        return 17
    if mode == "empty":
        return 0
    if mode == "malformed":
        print("{not-json")
        return 0
    if mode == "renamed-events":
        _emit({"type": "thread.opened", "thread_id": f"stub-session-{index}"})
        _emit({"type": "response.begun"})
        time.sleep(float(os.environ.get("CODEX_STUB_SLEEP_S", "2")))
        return 0

    completion = Path(os.environ["CODEX_STUB_COMPLETION_FILE"]).read_text()
    if mode == "refusal":
        completion = "I'm sorry, but I can't assist with that request."
    input_tokens = int(os.environ.get("CODEX_STUB_INPUT_TOKENS", "111"))
    cached_tokens = int(os.environ.get("CODEX_STUB_CACHED_TOKENS", "11"))
    output_tokens = int(os.environ.get("CODEX_STUB_OUTPUT_TOKENS", "222"))
    reasoning_tokens = int(os.environ.get("CODEX_STUB_REASONING_TOKENS", "33"))
    extra = {"unexpected_extra_field": {"future": True}} if mode == "extra-fields" else {}
    _emit(
        {
            "type": "thread.started",
            "thread_id": f"stub-session-{index}",
            **extra,
        }
    )
    _emit({"type": "turn.started", **extra})
    _emit(
        {
            "type": "item.completed",
            "item": {
                "id": f"item-{index}",
                "type": "agent_message",
                "text": completion,
                **extra,
            },
            **extra,
        }
    )
    _emit(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_output_tokens": reasoning_tokens,
            },
            **extra,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
