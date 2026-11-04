"""Resource-only calibration amendment; preserve original sources, jobs and pilot."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from statistics import mean

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.validation_20260909 import run_independent_calibration as runner
from experiments.validation_20260909 import summarize_independent_calibration as summary

CAP_SECONDS = 43200
SOURCE_PATHS = (
    Path(__file__).resolve(),
    runner.ROOT / "tests/test_independent_resource_amendment.py",
)


def original(out: Path) -> tuple[dict, dict]:
    plan = runner.verify_source(out)
    runner.verify_archive(out, read_sealed(out / "snapshot.json"), plan["source_archive_sha256"])
    pilot = summary.verify_phase(out, plan, "pilot")
    review = runner.pilot_review(plan, pilot["usages"])
    if (
        read_sealed(out / "PILOT_CLOSURE.json") != summary.pilot_closure(plan, pilot)
        or read_sealed(out / "PILOT_REVIEW.json") != review
        or review["fits_cap"] is not False
        or not plan["calibration_total_seconds"] < review["with_reserve_seconds"] <= CAP_SECONDS
    ):
        raise ValueError("original pilot closure or resource decision differs")
    return plan, review


def freeze(out: Path) -> None:
    with (
        runner.shared_lock(runner.MATRIX / "execution.lock"),
        runner.shared_lock(out / "execution.lock"),
    ):
        if any(
            (out / p).exists() for p in ("calibration", "COMPLETE.json", "RESOURCE_AMENDMENT.json")
        ):
            raise ValueError("amendment required before any calibration dispatch")
        plan, review = original(out)
        folder = out / "resource_amendment_sources"
        folder.mkdir(exist_ok=False)
        files = {}
        for path in SOURCE_PATHS:
            files[str(path.relative_to(runner.ROOT))] = runner.sha(path)
            copied = folder / path.name
            copied.write_bytes(path.read_bytes())
            if runner.sha(copied) != runner.sha(path):
                raise ValueError("amendment source copy differs")
        amendment = {
            "schema": "independent-calibration-resource-amendment-v1",
            "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plan_sha256": digest(plan),
            "pilot_review_sha256": digest(review),
            "superseded_total_worker_seconds": plan["calibration_total_seconds"],
            "maximum_total_worker_seconds": CAP_SECONDS,
            "reason": "Researcher authorized assessment action 1 on 13 September 2026. "
            "The completed pilot projects 7.04 worker hours, 9.15 with the frozen reserve. "
            "Twelve hours accommodates that projection while retaining a finite cap.",
            "scope": "Only total calibration worker time changes from six to twelve hours. "
            "All 972 jobs, original request identities, source/native snapshot, three completed "
            "pilot episodes, reference policies, seeds, bundles, admission rules, 2048 MiB RSS, "
            "1200-second child limits, sequential execution and no automatic retries remain. "
            "No model calls, adaptive launch or scientific gate waiver.",
            "files": files,
        }
        runner.store_once(out / "RESOURCE_AMENDMENT.json", amendment)
        (out / "RESOURCE_AMENDMENT.md").write_text(
            "# Independent calibration resource amendment\n\n"
            + amendment["recorded_at_utc"]
            + "\n\n"
            + amendment["reason"]
            + "\n\n"
            + amendment["scope"]
            + "\n"
        )


def verify_amendment(out: Path, plan: dict) -> dict:
    a = read_sealed(out / "RESOURCE_AMENDMENT.json")
    expected_paths = {str(p.relative_to(runner.ROOT)) for p in SOURCE_PATHS}
    if (
        a["plan_sha256"] != digest(plan)
        or a["pilot_review_sha256"] != digest(read_sealed(out / "PILOT_REVIEW.json"))
        or a["superseded_total_worker_seconds"] != plan["calibration_total_seconds"]
        or a["maximum_total_worker_seconds"] != CAP_SECONDS
        or set(a["files"]) != expected_paths
    ):
        raise ValueError("amendment exceeds or differs from explicit resource scope")
    for name, expected in a["files"].items():
        if (
            runner.sha(runner.ROOT / name) != expected
            or runner.sha(out / "resource_amendment_sources" / Path(name).name) != expected
        ):
            raise ValueError("amendment source changed")
    return a


def completion(plan: dict, amendment: dict, usages: list[dict]) -> dict:
    if len(usages) != len(plan["calibration_jobs"]):
        raise ValueError("incomplete calibration usage grid")
    if sum(u["elapsed_seconds"] for u in usages) > CAP_SECONDS:
        raise ValueError("amended total worker-time cap exceeded")
    return {
        "plan_sha256": digest(plan),
        "resource_amendment_sha256": digest(amendment),
        "maximum_total_worker_seconds": CAP_SECONDS,
        "requests": len(usages),
        "usage_sha256": [digest(u) for u in usages],
        "total_worker_seconds": sum(u["elapsed_seconds"] for u in usages),
        "peak_rss_mib": max(u["peak_rss_mib"] for u in usages),
        "no_model_calls": True,
    }


def run(out: Path) -> None:
    with (
        runner.shared_lock(runner.MATRIX / "execution.lock"),
        runner.shared_lock(out / "execution.lock"),
    ):
        plan, _ = original(out)
        amendment = verify_amendment(out, plan)
        if (out / "COMPLETE.json").exists():
            raise ValueError("already complete; verify without redispatch")
        usages = []
        for job in plan["calibration_jobs"]:
            verify_amendment(out, plan)
            remaining = CAP_SECONDS - sum(u["elapsed_seconds"] for u in usages)
            usages.append(runner.guarded(out, runner.request(plan, job), plan, remaining))
            print(
                f"completed calibration {len(usages)}/{len(plan['calibration_jobs'])}", flush=True
            )
        runner.verify_source(out)
        verify_amendment(out, plan)
        runner.store_once(out / "COMPLETE.json", completion(plan, amendment, usages))


def verify(out: Path) -> dict:
    complete = read_sealed(out / "COMPLETE.json")
    plan, _ = original(out)
    amendment = verify_amendment(out, plan)
    checked = summary.verify_phase(out, plan, "calibration")
    if complete != completion(plan, amendment, checked["usages"]):
        raise ValueError("completion receipt differs from verified jobs and amended cap")
    rows = checked["rows"]
    groups = []
    for cell in plan["design"]["cells"]:
        for policy in plan["design"]["fixed_references"]:
            group = [
                r
                for r in rows
                if r["job"]["cell_id"] == cell["id"] and r["job"]["label"] == policy["label"]
            ]
            feasible = [r for r in group if r["status"] == "evaluated" and r["feasible"]]
            groups.append(
                {
                    "cell_id": cell["id"],
                    "reference": policy["label"],
                    "episodes": len(group),
                    "feasible": len(feasible),
                    "policy_faults": sum(r["status"] == "policy-fault" for r in group),
                    "mean_energy_among_feasible": mean(r["normalized_energy"] for r in feasible)
                    if feasible
                    else None,
                }
            )
    runner.verify_source(out)
    verify_amendment(out, plan)
    return {
        "schema": "independent-calibration-amended-closure-v1",
        "plan_sha256": digest(plan),
        "complete_sha256": digest(complete),
        "resource_amendment_sha256": digest(amendment),
        "files_sha256": checked["files_sha256"],
        "rows": rows,
        "groups": groups,
        "admission": summary.admission(plan["design"], rows),
        "resource_usage": complete,
        "scope": "Development calibration under the disclosed twelve-hour resource amendment. "
        "The original failed six-hour pilot review is preserved. Original semantic verification "
        "and operating-level admission rules are reused unchanged. Offered traces, seeded "
        "arrivals, accounting and fitness are checked with shared model code, not independent "
        "physical validation. Full service/snapshot ledgers are absent. No adaptive launch.",
    }


def publish(out: Path) -> None:
    with (
        runner.shared_lock(runner.MATRIX / "execution.lock"),
        runner.shared_lock(out / "execution.lock"),
    ):
        report = verify(out)
        runner.preserve_or_match(out / "SUMMARY.json", report)
        path, rendered = out / "REPORT.md", summary.render(report)
        if path.exists() and path.read_text() != rendered:
            raise ValueError("existing report differs")
        if not path.exists():
            path.write_text(rendered)
        print(
            json.dumps({"verified_episodes": len(report["rows"]), "admission": report["admission"]})
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=runner.OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out)
    elif args.run:
        run(args.out)
        publish(args.out)
    else:
        publish(args.out)
