"""Freeze and run a bounded, complete fixed-policy development sensitivity grid.

No proposal operators, held-out acceptance data or external model calls are used.
An incomplete request is never silently retried. Historical evidence is read only.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import psutil

from analysis.tracking import derive_seed
from evolution.evaluation import (
    PolicyExecutionError,
    evaluate_policy_source,
    load_task_evaluation_config,
)
from experiments.allbaselines_harness.initialization import _REFERENCE_FILES
from experiments.allbaselines_harness.summary import fitness_from_record
from experiments.gp_tuning_20260902.recovery import canonical, digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot, write_sealed
from experiments.matched_control_20260909.real_pilot import historical_seed_inventory
from experiments.validation_20260909.workload_sensitivity import FRACTIONS, SHAPES, transform
from sim.spec import config_sha256, load_episode_config

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "workload_sensitivity_20260909"
TASK = ROOT / "evolution/task/config_dev_continuum_tight_calibrated.json"
SUMMARY = HERE / "fit_diagnostics_20260909_v2/SUMMARY.json"
SEED_ROOT = 9113512321037399045
PILOT_BASELINE = "cheapest_tier_bottleneck_headroom"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_source(archive: Path, pointer: dict) -> dict:
    """Verify the whole chain and final best fitness without mutating EventArchive."""
    previous, selected, best, last = "0" * 64, None, None, None
    with (archive / "events.jsonl").open() as stream:
        for index, line in enumerate(stream):
            event = json.loads(line)
            body = {k: v for k, v in event.items() if k != "event_sha256"}
            if (
                line != canonical(event) + "\n"
                or digest(body) != event["event_sha256"]
                or event["event_seq"] != index
                or event["previous_event_sha256"] != previous
            ):
                raise ValueError(f"event chain failure at {archive}:{index}")
            previous, last = event["event_sha256"], event
            if event["type"] != "candidate_completed":
                continue
            row = event["payload"]
            evaluation = (row.get("evaluation") or {}).get("evaluation")
            if evaluation is not None:
                fitness = fitness_from_record(evaluation["fitness"])
                if fitness.proposal_seq != row["proposal_seq"]:
                    raise ValueError("fitness/proposal identity mismatch")
                if best is None or fitness.sort_key() < best[0]:
                    best = (fitness.sort_key(), row["candidate_id"])
            if row["candidate_id"] == pointer["candidate_id"]:
                if selected is not None or row["proposal_seq"] != pointer["proposal_seq"]:
                    raise ValueError("ambiguous selected candidate")
                selected = row
    if (
        last is None
        or last["type"] != "campaign_terminal"
        or last["payload"]["status"] not in ("complete", "unusable")
        or selected is None
        or best is None
        or best[1] != pointer["candidate_id"]
    ):
        raise ValueError("report selection differs from a terminal archive best")
    path = archive / "candidates" / f"{pointer['proposal_seq']:08d}.ps"
    if path.read_text() != selected["proposal"]["source"]:
        raise ValueError("selected materialized source differs from durable event")
    return {
        "source": path.read_text(),
        "source_sha256": sha(path),
        "path": str(path.relative_to(ROOT)),
        "candidate_id": pointer["candidate_id"],
        "chain_head": previous,
        "events_file_sha256": sha(archive / "events.jsonl"),
        "terminal": last["payload"],
    }


def population() -> list[dict]:
    rows = []
    for root in range(800000801, 800000821):
        runs = list(
            (ROOT / "experiments/deep_search_20260825/runs_final").glob(f"*--seed-{root}--*")
        )
        if len(runs) != 1:
            raise ValueError("expected one primary archive for each of the twenty roots")
        report_path = runs[0] / "report.json"
        report = read_sealed(report_path)
        for arm in ("llm", "same_dsl_gp"):
            selection = selected_source(
                runs[0] / "archives" / arm, report["arms"][arm]["best_found_at"]
            )
            rows.append(
                {
                    "label": f"{arm}-{root}",
                    "group": arm,
                    "root": root,
                    "report": str(report_path.relative_to(ROOT)),
                    "report_file_sha256": sha(report_path),
                    **selection,
                }
            )
    for label, filename in _REFERENCE_FILES.items():
        path = ROOT / "runtime/policies" / filename
        rows.append(
            {
                "label": label,
                "group": "fixed-reference",
                "path": str(path.relative_to(ROOT)),
                "source": path.read_text(),
                "source_sha256": sha(path),
            }
        )
    for label in (PILOT_BASELINE, "root807_marginal_watts_headroom"):
        path = ROOT / "experiments/baseline_check_20260905/policies" / (label + ".py")
        rows.append(
            {
                "label": label,
                "group": "posthoc-reference",
                "path": str(path.relative_to(ROOT)),
                "source": path.read_text(),
                "source_sha256": sha(path),
            }
        )
    if len(rows) != 49 or len({r["label"] for r in rows}) != 49:
        raise ValueError("the complete fixed population must contain 49 labels")
    return rows


def snapshot() -> dict:
    state = evaluation_snapshot()
    for path in (
        Path(__file__).resolve(),
        HERE / "workload_sensitivity.py",
        SUMMARY,
        HERE / "summarize_workload_sensitivity.py",
        ROOT / "experiments/matched_control_20260909/real_pilot.py",
        ROOT / "experiments/allbaselines_harness/initialization.py",
        ROOT / "experiments/allbaselines_harness/summary.py",
        ROOT / "uv.lock",
        ROOT / "tests/test_workload_sensitivity.py",
        ROOT / "tests/test_workload_sensitivity_runner.py",
        HERE / "WORKLOAD_SENSITIVITY.md",
    ):
        state["files"][str(path.relative_to(ROOT))] = sha(path)
    return json.loads(canonical(state))


def freeze(out: Path) -> None:
    if out.exists():
        raise ValueError("freeze requires a new output directory")
    # Validate all inputs before creating a receipt or invoking an evaluator.
    policies = population()
    summary = read_sealed(SUMMARY)
    fits = next(
        s["selected_fit_errors"]
        for s in summary["scenarios"]
        if s["scenario_id"] == "dev-continuum-tight"
    )
    task = load_task_evaluation_config(TASK)
    base = load_episode_config(task.scenarios[0].path)
    cells = [
        {
            "id": f"{shape}-{fraction:.2f}",
            "shape": shape,
            "fraction": fraction,
            "unbound_config_sha256": config_sha256(
                transform(base, shape=shape, fraction=fraction, fits=fits)
            ),
        }
        for shape in SHAPES
        for fraction in FRACTIONS
    ]
    inventory = historical_seed_inventory()
    pilot = derive_seed(SEED_ROOT, "resource-pilot", 0)
    seeds = [derive_seed(SEED_ROOT, "fixed-policy-evaluation", i) for i in range(8)]
    # The completed real initialization pilot used this additional evaluation seed.
    excluded = set(inventory["seeds"]) | {3246458936443790985}
    if len(set([pilot, *seeds])) != 9 or excluded.intersection([pilot, *seeds]):
        raise ValueError("seed roles collide with one another or an inventoried historical grid")
    frozen = snapshot()
    out.mkdir(parents=True)
    write_sealed(out / "snapshot.json", frozen)
    with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
        for name in sorted(frozen["files"]):
            archive.add(ROOT / name, arcname=name)
        for i, name in enumerate(frozen["native"]):
            archive.add(name, arcname=f"native/{i}-{Path(name).name}")
    unique = {row["source_sha256"]: row["source"] for row in policies}
    for source_hash, source in unique.items():
        path = out / "policies" / (source_hash + ".ps")
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
        if sha(path) != source_hash:
            raise ValueError("copied policy differs from exact source bytes")
    for row in policies:
        del row["source"]
    plan = {
        "schema": "fixed-workload-sensitivity-v1",
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "snapshot_sha256": digest(frozen),
        "policies": policies,
        "unique_sources": len(unique),
        "cells": cells,
        "fits": fits,
        "seed_root": SEED_ROOT,
        "seeds": seeds,
        "pilot_seed": pilot,
        "excluded_seed_inventory": inventory,
        "additional_excluded_seeds": [3246458936443790985],
        "task_file_sha256": sha(TASK),
        "bundles": list(task.bundle_ids),
        "pilot_baseline": PILOT_BASELINE,
        "pilot_cell": "original-0.45",
        "rss_limit_mib": 2048,
        "worker_timeout_seconds": 1200,
        "maximum_total_worker_seconds": 14400,
        "analysis": "All 49 labels on all six cells; exact-source deduplication only. "
        "All policy faults retained as invalid. Feasibility counts by arm; "
        "paired energy only where both policies are feasible; descriptive "
        "win/loss/ties, no significance tests. Shared eight-seed, three-bundle "
        "grid is not independent root-level environmental replication.",
        "boundary": "Development sensitivity conditional on historical selection; no "
        "retraining, causal operator effect, independent data, validated "
        "per-request tails, CPU attribution estimate or P1/P2 clearance.",
    }
    if snapshot() != frozen:
        raise ValueError("source changed during freeze")
    write_sealed(out / "PLAN.json", plan)
    print(
        json.dumps(
            {
                "labels": 49,
                "distinct_sources": len(unique),
                "cells": len(cells),
                "evaluation_seeds": seeds,
                "pilot_seed": pilot,
            }
        ),
        flush=True,
    )


def request(plan: dict, source_hash: str, cell: dict, *, pilot: bool) -> dict:
    return {
        "plan_sha256": digest(plan),
        "source_sha256": source_hash,
        "cell": cell,
        "seeds": [plan["pilot_seed"]] if pilot else plan["seeds"],
        "pilot": pilot,
    }


def worker(out: Path, request_path: Path, result_path: Path) -> None:
    plan, frozen, req = (
        read_sealed(p) for p in (out / "PLAN.json", out / "snapshot.json", request_path)
    )
    if (
        snapshot() != frozen
        or digest(frozen) != plan["snapshot_sha256"]
        or req != request(plan, req["source_sha256"], req["cell"], pilot=req["pilot"])
        or req["cell"] not in plan["cells"]
        or req["source_sha256"] not in {p["source_sha256"] for p in plan["policies"]}
    ):
        raise ValueError("worker request or frozen source binding changed")
    source_path = out / "policies" / (req["source_sha256"] + ".ps")
    if sha(source_path) != req["source_sha256"] or result_path.exists():
        raise ValueError("source mismatch or attempted completed-result overwrite")
    task = load_task_evaluation_config(TASK)
    task = dataclasses.replace(
        task,
        scenarios=tuple(
            dataclasses.replace(grid, seeds=tuple(req["seeds"])) for grid in task.scenarios
        ),
    )
    cell = req["cell"]
    try:
        evaluation = evaluate_policy_source(
            source_path.read_text(),
            task,
            proposal_seq=0,
            config_transform=lambda c: transform(
                c, shape=cell["shape"], fraction=cell["fraction"], fits=plan["fits"]
            ),
        )
        result = {
            "status": "evaluated",
            "evaluation": evaluation.without_episode_ledgers().to_dict(),
        }
    except PolicyExecutionError as exc:
        result = {"status": "policy-fault", "failure": exc.failure}
    if snapshot() != frozen:
        raise ValueError("source or environment changed during evaluation")
    write_sealed(result_path, {"request_sha256": digest(req), **result})


def guarded(out: Path, req: dict, plan: dict, remaining_seconds: float) -> dict:
    key = digest(req)
    folder = out / ("pilot" if req["pilot"] else "scores")
    folder.mkdir(exist_ok=True)
    rp, resultp, usagep = (
        folder / (key + suffix) for suffix in (".request.json", ".result.json", ".usage.json")
    )
    if rp.exists():
        if read_sealed(rp) != req or not resultp.exists() or not usagep.exists():
            raise ValueError("incomplete or changed previous request: inspect, do not retry")
        result, usage = read_sealed(resultp), read_sealed(usagep)
        if (
            result["request_sha256"] != key
            or usage["request_sha256"] != key
            or usage["exit_code"] != 0
            or usage["stop_reason"] is not None
        ):
            raise ValueError("previous request is not a complete successful execution")
        return usage
    write_sealed(rp, req)
    start, peak, child, code, reason = time.monotonic(), 0, None, None, None
    timeout = min(plan["worker_timeout_seconds"], remaining_seconds)
    try:
        with (folder / (key + ".log")).open("x") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.validation_20260909.run_workload_sensitivity",
                    "--out",
                    str(out),
                    "--worker",
                    str(rp),
                    str(resultp),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            observed = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    peak = max(peak, observed.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak > plan["rss_limit_mib"] * 1024**2:
                    raise RuntimeError("worker RSS limit exceeded")
                if time.monotonic() - start > timeout:
                    raise RuntimeError("worker or total time limit exceeded")
                time.sleep(0.5)
            code = child.wait()
            if code != 0:
                raise RuntimeError(f"worker exit {code}; see {log.name}")
        result = read_sealed(resultp)
        if result["request_sha256"] != key:
            raise ValueError("completed result binding mismatch")
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            code = child.wait()
        usage = {
            "request_sha256": key,
            "exit_code": code,
            "stop_reason": reason,
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_mib": peak / 1024**2,
        }
        write_sealed(usagep, usage)
    return usage


def run(out: Path, *, pilot_only: bool) -> None:
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, frozen = read_sealed(out / "PLAN.json"), read_sealed(out / "snapshot.json")
        if snapshot() != frozen or digest(frozen) != plan["snapshot_sha256"]:
            raise ValueError("frozen input or environment mismatch")
        source = next(p["source_sha256"] for p in plan["policies"] if p["label"] == PILOT_BASELINE)
        cell = next(c for c in plan["cells"] if c["id"] == plan["pilot_cell"])
        pilot = guarded(
            out, request(plan, source, cell, pilot=True), plan, plan["worker_timeout_seconds"]
        )
        unique_sources = sorted({p["source_sha256"] for p in plan["policies"]})
        expected = (
            pilot["elapsed_seconds"] * len(plan["seeds"]) * len(unique_sources) * len(plan["cells"])
        )
        print(
            json.dumps(
                {
                    "pilot_usage": pilot,
                    "linear_runtime_projection_seconds": expected,
                    "projection_is_not_a_guarantee": True,
                }
            ),
            flush=True,
        )
        if pilot_only:
            return
        if not math.isfinite(expected) or expected > plan["maximum_total_worker_seconds"]:
            raise ValueError("pilot projection exceeds the frozen total work-time cap")
        usage = []
        for cell in plan["cells"]:
            for source in unique_sources:
                remaining = plan["maximum_total_worker_seconds"] - sum(
                    u["elapsed_seconds"] for u in usage
                )
                if remaining <= 0:
                    raise RuntimeError("total worker time exhausted; retain incomplete grid")
                req = request(plan, source, cell, pilot=False)
                usage.append(guarded(out, req, plan, remaining))
                print(
                    f"completed {len(usage)}/{len(unique_sources) * len(plan['cells'])}: "
                    f"{cell['id']} {source[:12]}",
                    flush=True,
                )
        if snapshot() != frozen:
            raise ValueError("frozen source changed during the grid")
        write_sealed(
            out / "COMPLETE.json",
            {
                "plan_sha256": digest(plan),
                "requests": len(usage),
                "total_worker_seconds": sum(u["elapsed_seconds"] for u in usage),
                "peak_rss_mib": max(u["peak_rss_mib"] for u in usage),
                "no_model_calls": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", nargs=2, type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out)
    elif args.worker:
        worker(args.out, *args.worker)
    else:
        run(args.out, pilot_only=args.pilot)


if __name__ == "__main__":
    main()
