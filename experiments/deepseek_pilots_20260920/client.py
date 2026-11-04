"""Pinned DeepSeek transport with durable receipts and a shared project-budget lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path

from experiments.glm_budget_pilot_20260917.run import bound
from experiments.gp_tuning_20260902.recovery import digest
from experiments.matched_transport_20260909.client import DurableOpenRouterClient

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "experiments/api_budget_20260917/journal.jsonl"


class PilotLimit(SystemExit):
    pass


def save(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def legacy_exposure(stream):
    """Count completed accounting plus both historically retained reservations."""
    events = DurableOpenRouterClient._read(stream)
    spent = retained = 0
    pending = None
    for event in events[1:]:
        row = event["payload"]
        if row["type"] == "reserved":
            if pending is not None:
                raise ValueError("overlapping legacy dispatch")
            pending = row
        elif row["type"] == "completed":
            if pending is None or row["request_sha256"] != pending["request_sha256"]:
                raise ValueError("legacy completion mismatch")
            spent += row["completion"]["cost_microusd"]
            pending = None
        elif row["type"] in {
            "explicit_retry_retaining_reservation",
            "parameter_amendment_retaining_reservation",
        }:
            if pending is None:
                raise ValueError("legacy retained reservation missing")
            retained += pending["reservation_microusd"]
            pending = None
        else:
            raise ValueError("unknown legacy event")
    return spent + retained + (pending["reservation_microusd"] if pending else 0)


def check_receipt(raw, config):
    usage = raw["usage"]
    a, b = usage["prompt_tokens"], usage["completion_tokens"]
    if type(a) is not int or type(b) is not int or min(a, b) < 0:
        raise ValueError("invalid token usage")
    cost = math.ceil(
        (a * config.prompt_microusd_per_million + b * config.completion_microusd_per_million) / 1e6
    )
    billed = Decimal(str(usage["cost"]))
    if not billed.is_finite() or billed < 0 or billed * 1000000 > cost:
        raise ValueError("provider billing exceeds accounting bound")
    if cost > config.max_call_reservation_microusd:
        raise ValueError("reservation exceeded")
    if raw.get("provider", "").lower() != "deepseek":
        raise ValueError("provider differs from pin")
    return cost


class PilotClient(DurableOpenRouterClient):
    def __init__(self, *args, calls_dir, **kwargs):
        self.calls_dir = calls_dir
        calls_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)

    def complete(self, *, model, prompt, generation, seed):
        bound(prompt, self.config)
        key = digest({"model": model, "prompt": prompt, "generation": generation, "seed": seed})
        path = self.calls_dir / f"{key}.json"
        if not path.exists():
            if len(list(self.calls_dir.glob("*.json"))) >= 6:
                raise PilotLimit("six-model-request pilot ceiling reached, including repairs")
            save(
                path,
                {
                    "request_sha256": key,
                    "seed": seed,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "reserved_ceiling_microusd": self.config.max_call_reservation_microusd,
                },
            )
        self.receipt_path = self.calls_dir.parent / f"{key}-provider.json"
        self.wire_path = self.calls_dir.parent / f"{key}-wire.json"
        self.error_path = self.calls_dir.parent / f"{key}-http-error.json"
        self.started = time.monotonic()
        # All legacy clients lock this journal too. Do not mutate historical accounting.
        # The new journal is shared by A and B and has a $0.60 aggregate cap.
        with LEGACY.open("rb") as legacy:
            fcntl.flock(legacy.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if legacy_exposure(legacy) + self.config.max_total_microusd > 60000000:
                raise PilotLimit("project budget cannot fund the pilot allocation")
            result = super().complete(model=model, prompt=prompt, generation=generation, seed=seed)
        print(
            json.dumps(
                {
                    "study": self.calls_dir.parent.name,
                    "requests": len(list(self.calls_dir.glob("*.json"))),
                    "accounted_call_usd": result.cost_microusd / 1e6,
                }
            ),
            flush=True,
        )
        return result

    def _dispatch(self, request):
        raw = super()._dispatch(request)
        save(
            self.receipt_path, {"response": raw, "elapsed_seconds": time.monotonic() - self.started}
        )
        check_receipt(raw, self.config)
        return raw

    def _open(self, request):
        payload = json.loads(request.data)
        payload.pop("seed")  # Unsupported by the pinned provider; local seed is retained.
        save(self.wire_path, payload)
        corrected = urllib.request.Request(
            request.full_url,
            data=json.dumps(payload, sort_keys=True).encode(),
            headers=dict(request.header_items()),
            method=request.method,
        )
        try:
            return super()._open(corrected)
        except urllib.error.HTTPError as exc:
            body = exc.read(65536).decode("utf-8", errors="replace")
            auth = request.get_header("Authorization", "")
            if auth:
                body = body.replace(auth, "[redacted]").replace(
                    auth.removeprefix("Bearer "), "[redacted]"
                )
            save(
                self.error_path,
                {"http_status": exc.code, "body": body, "automatic_retry_allowed": False},
            )
            exc.close()
            raise
