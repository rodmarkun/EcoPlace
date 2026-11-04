"""Execute the unchanged sensitivity matrix under an explicit resource-only amendment.

The original plan, source snapshot, requests, evaluator and pilot stay byte-identical.
This independently archived controller supersedes only the total worker-time cap.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import math
from pathlib import Path

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.validation_20260909.run_workload_sensitivity import (
    DEFAULT_OUT,
    ROOT,
    guarded,
    request,
    sha,
    snapshot,
)

CAP_SECONDS = 28800


def verify_original(out: Path) -> tuple[dict, dict]:
    plan = read_sealed(out / "PLAN.json")
    frozen = read_sealed(out / "snapshot.json")
    review = read_sealed(out / "PILOT_REVIEW.json")
    if (
        digest(frozen) != plan["snapshot_sha256"]
        or snapshot() != frozen
        or review["plan_sha256"] != digest(plan)
        or review["matrix_started"]
        or review["matrix_cap_seconds"] != plan["maximum_total_worker_seconds"]
    ):
        raise ValueError("original plan, snapshot or pre-comparison pilot review changed")
    pilot_key = review["pilot_request_sha256"]
    req, result, usage = (
        read_sealed(out / "pilot" / (pilot_key + suffix))
        for suffix in (".request.json", ".result.json", ".usage.json")
    )
    if (
        digest(req) != pilot_key
        or req["plan_sha256"] != digest(plan)
        or req["pilot"] is not True
        or req["seeds"] != [plan["pilot_seed"]]
        or result["request_sha256"] != pilot_key
        or result["status"] != "evaluated"
        or usage != review["pilot_usage"]
        or usage["exit_code"] != 0
        or usage["stop_reason"] is not None
    ):
        raise ValueError("pilot receipts do not bind the successful original resource measurement")
    estimate = (
        usage["elapsed_seconds"] * len(plan["seeds"]) * plan["unique_sources"] * len(plan["cells"])
    )
    if (
        not math.isfinite(estimate)
        or estimate != review["linear_projection_seconds"]
        or not plan["maximum_total_worker_seconds"] < estimate < CAP_SECONDS
    ):
        raise ValueError("resource amendment no longer matches the measured projection")
    return plan, review


def freeze(out: Path) -> None:
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "RESOURCE_AMENDMENT.json").exists() or (out / "scores").exists():
            raise ValueError("amendment must be created once before any comparative dispatch")
        plan, review = verify_original(out)
        paths = [Path(__file__).resolve(), ROOT / "tests/test_workload_resource_amendment.py"]
        files = {str(path.relative_to(ROOT)): sha(path) for path in paths}
        amended_sources = out / "resource_amendment_sources"
        amended_sources.mkdir(exist_ok=False)
        for path in paths:
            copied = amended_sources / path.name
            copied.write_bytes(path.read_bytes())
            if sha(copied) != files[str(path.relative_to(ROOT))]:
                raise ValueError("resource controller archive copy differs")
        amended = {
            "schema": "workload-sensitivity-resource-amendment-v1",
            "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plan_sha256": digest(plan),
            "pilot_review_sha256": digest(review),
            "superseded_total_worker_seconds": plan["maximum_total_worker_seconds"],
            "maximum_total_worker_seconds": CAP_SECONDS,
            "reason": "Successful separate resource pilot projects 5.098 hours; "
            "an eight-hour cap provides approximately 57% headroom. "
            "Amended before any comparative dispatch or outcome inspection.",
            "unchanged": [
                "all 49 policy labels and 29 source programs",
                "all six settings",
                "all eight seeds and three bundles",
                "evaluator and analysis",
                "2048 MiB and 1200-second per-request limits",
                "one child, no model calls, no automatic retries",
            ],
            "scope": "Only total matrix worker time changes from four to eight hours. "
            "Original plan and pilot remain immutable; this controller consumes "
            "the original request identities and adds an amendment binding to "
            "the final completion receipt. No scientific gate is waived.",
            "files": files,
        }
        write_sealed(out / "RESOURCE_AMENDMENT.json", amended)
        (out / "RESOURCE_AMENDMENT.md").write_text(
            "# Resource amendment 1\n\n"
            + amended["recorded_at_utc"]
            + "\n\n"
            + amended["reason"]
            + "\n\n"
            + amended["scope"]
            + "\n\n"
            + "The original pilot is reused as resource evidence without rerunning or "
            "relabeling it. Sources for the amended controller and its tests are copied "
            "and hashed separately. Every scientific request remains bound to the original "
            "PLAN.json and snapshot.\n"
        )


def verify_amendment(out: Path, plan: dict) -> dict:
    amended = read_sealed(out / "RESOURCE_AMENDMENT.json")
    if (
        amended["plan_sha256"] != digest(plan)
        or amended["pilot_review_sha256"] != digest(read_sealed(out / "PILOT_REVIEW.json"))
        or amended["superseded_total_worker_seconds"] != plan["maximum_total_worker_seconds"]
        or amended["maximum_total_worker_seconds"] != CAP_SECONDS
    ):
        raise ValueError("amendment exceeds or differs from the explicit resource scope")
    for name, expected in amended["files"].items():
        if (
            sha(ROOT / name) != expected
            or sha(out / "resource_amendment_sources" / Path(name).name) != expected
        ):
            raise ValueError("amendment controller/test source changed")
    return amended


def run(out: Path) -> None:
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, _review = verify_original(out)
        amended = verify_amendment(out, plan)
        if (out / "COMPLETE.json").exists():
            raise ValueError("matrix already has a completion receipt; verify instead of rerunning")
        sources = sorted({p["source_sha256"] for p in plan["policies"]})
        usage = []
        for cell in plan["cells"]:
            for source in sources:
                verify_amendment(out, plan)
                remaining = CAP_SECONDS - sum(u["elapsed_seconds"] for u in usage)
                if remaining <= 0:
                    raise RuntimeError("amended total worker time exhausted; preserve partial grid")
                req = request(plan, source, cell, pilot=False)
                usage.append(guarded(out, req, plan, remaining))
                print(
                    f"completed {len(usage)}/{len(sources) * len(plan['cells'])}: "
                    f"{cell['id']} {source[:12]}; {usage[-1]['elapsed_seconds']:.2f}s, "
                    f"{usage[-1]['peak_rss_mib']:.1f}MiB",
                    flush=True,
                )
        verify_original(out)
        verify_amendment(out, plan)
        write_sealed(
            out / "COMPLETE.json",
            {
                "plan_sha256": digest(plan),
                "resource_amendment_sha256": digest(amended),
                "maximum_total_worker_seconds": CAP_SECONDS,
                "requests": len(usage),
                "total_worker_seconds": sum(u["elapsed_seconds"] for u in usage),
                "peak_rss_mib": max(u["peak_rss_mib"] for u in usage),
                "no_model_calls": True,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out)
    else:
        run(args.out)
