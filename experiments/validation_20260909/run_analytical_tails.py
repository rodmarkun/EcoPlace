"""Freeze, guard and retain the complete independent-replicate tail witness."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import psutil

from analysis.tracking import derive_seed
from experiments.gp_tuning_20260902.recovery import canonical, digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot, write_sealed
from experiments.matched_control_20260909.real_pilot import historical_seed_inventory
from experiments.validation_20260909.analytical_tails import (
    ARRIVAL_RATES,
    EARLY_WINDOW,
    LATE_WINDOW,
    PRIMARY_WINDOW,
    REPLICATES,
    ROOT,
    analyze_rate,
    cohort,
    episode,
)
from experiments.validation_20260909.run_workload_sensitivity import DEFAULT_OUT as MATRIX
from sim import CallableScorer, run_episode
from sim.spec import config_sha256

HERE = Path(__file__).parent
OUT = HERE / "analytical_tails_20260909"
SEED_ROOT = 9113512321037399046


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict:
    state = evaluation_snapshot()
    for path in (
        Path(__file__).resolve(),
        HERE / "analytical_tails.py",
        HERE / "ANALYTICAL_TAILS.md",
        ROOT / "tests/test_analytical_tails.py",
        ROOT / "tests/test_analytical_tail_driver.py",
        ROOT / "uv.lock",
        ROOT / "experiments/matched_control_20260909/real_pilot.py",
        MATRIX / "PLAN.json",
        MATRIX / "RESOURCE_AMENDMENT.json",
    ):
        state["files"][str(path.relative_to(ROOT))] = sha(path)
    return json.loads(canonical(state))


def freeze(out: Path) -> None:
    if out.exists():
        raise ValueError("tail freeze requires a fresh output directory")
    matrix = read_sealed(MATRIX / "PLAN.json")
    excluded = set(historical_seed_inventory()["seeds"]) | set(matrix["seeds"])
    excluded.update((matrix["pilot_seed"], 3246458936443790985))
    fixture_manifest = json.loads((ROOT / "sim/fixtures/MANIFEST.json").read_text())
    excluded.update(fixture_manifest["acceptance_seed_matrix"])
    jobs, all_seeds = [], []
    for rate_index, rate in enumerate(ARRIVAL_RATES):
        for index in range(REPLICATES):
            seed = derive_seed(SEED_ROOT, "mm1-tail-independent-replicate", rate_index, index)
            all_seeds.append(seed)
            for horizon in (5000.0, 4800.0) if index < 4 else (5000.0,):
                jobs.append(
                    {
                        "rate": rate,
                        "replicate": index,
                        "seed": seed,
                        "horizon": horizon,
                        "config_sha256": config_sha256(episode(rate, horizon)),
                    }
                )
    if len(set(all_seeds)) != 128 or excluded.intersection(all_seeds) or len(jobs) != 136:
        raise ValueError("tail grid collides with inventoried roles or is incomplete")
    frozen = snapshot()
    out.mkdir(parents=True)
    write_sealed(out / "snapshot.json", frozen)
    with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
        for name in sorted(frozen["files"]):
            archive.add(ROOT / name, arcname=name)
        for i, name in enumerate(frozen["native"]):
            archive.add(name, arcname=f"native/{i}-{Path(name).name}")
    plan = {
        "schema": "analytical-mm1-tail-witness-v1",
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "snapshot_sha256": digest(frozen),
        "seed_root": SEED_ROOT,
        "jobs": jobs,
        "matrix_plan_sha256": digest(matrix),
        "primary_window": list(PRIMARY_WINDOW),
        "early_window": list(EARLY_WINDOW),
        "late_window": list(LATE_WINDOW),
        "rss_limit_mib": 2048,
        "worker_timeout_seconds": 600,
        "maximum_total_worker_seconds": 7200,
        "no_model_calls": True,
        "reference": "https://iadan.win.tue.nl/queueing.pdf, section4.4, equation4.13",
        "boundary": "Development analytical witness, not empirical tail validity or "
        "P1/P2 acceptance. All 128 main runs and eight horizon checks retained. "
        "Nominal replicate t intervals, not request-level binomial intervals.",
    }
    if snapshot() != frozen:
        raise ValueError("source changed during tail freeze")
    write_sealed(out / "PLAN.json", plan)
    print("frozen 128 independent main runs and eight paired horizon checks", flush=True)


def verify_source(out: Path) -> dict:
    plan, frozen = read_sealed(out / "PLAN.json"), read_sealed(out / "snapshot.json")
    if digest(frozen) != plan["snapshot_sha256"] or snapshot() != frozen:
        raise ValueError("tail source/environment drift")
    return plan


def worker(out: Path, key: str) -> None:
    plan = verify_source(out)
    req = read_sealed(out / "runs" / (key + ".request.json"))
    job = req["job"]
    if digest(req) != key or req["plan_sha256"] != digest(plan) or job not in plan["jobs"]:
        raise ValueError("tail request is outside the frozen grid")
    config = episode(job["rate"], job["horizon"])
    if config_sha256(config) != job["config_sha256"]:
        raise ValueError("tail episode changed")
    result_path = out / "runs" / (key + ".result.json")
    records_path = out / "runs" / (key + ".records.jsonl.gz")
    if result_path.exists() or records_path.exists():
        raise ValueError("refuse a completed or partial trace overwrite")
    result = run_episode(
        config, seed=job["seed"], scorer=CallableScorer("flat", lambda p, n, c: 1.0)
    )
    records = sorted(result.records, key=lambda r: r.rid)
    raw = "".join(canonical([r.rid, r.arrival_s, r.end_s, r.status]) + "\n" for r in records)
    records_path.write_bytes(gzip.compress(raw.encode(), mtime=0))
    payload = {
        "request_sha256": key,
        **job,
        "records_file_sha256": sha(records_path),
        "records_uncompressed_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "all_offered_records": len(records),
        **{
            name: cohort(records, window, job["rate"])
            for name, window in (
                ("primary", PRIMARY_WINDOW),
                ("early", EARLY_WINDOW),
                ("late", LATE_WINDOW),
            )
        },
    }
    verify_source(out)
    write_sealed(result_path, payload)


def guarded(out: Path, job: dict, plan: dict, remaining: float) -> dict:
    req = {"plan_sha256": digest(plan), "job": job}
    key = digest(req)
    folder = out / "runs"
    folder.mkdir(exist_ok=True)
    request_path = folder / (key + ".request.json")
    result_path, usage_path = (
        folder / (key + suffix) for suffix in (".result.json", ".usage.json")
    )
    if request_path.exists():
        if read_sealed(request_path) != req or not result_path.exists() or not usage_path.exists():
            raise ValueError("partial tail request retained; no automatic retry")
        result, usage = read_sealed(result_path), read_sealed(usage_path)
        if (
            result["request_sha256"] != key
            or usage["request_sha256"] != key
            or usage["exit_code"] != 0
            or usage["stop_reason"] is not None
            or sha(folder / (key + ".records.jsonl.gz")) != result["records_file_sha256"]
        ):
            raise ValueError("completed tail evidence binding mismatch")
        return usage
    write_sealed(request_path, req)
    start, peak, child, reason, code = time.monotonic(), 0, None, None, None
    try:
        with (folder / (key + ".log")).open("x") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.validation_20260909.run_analytical_tails",
                    "--out",
                    str(out),
                    "--worker",
                    key,
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
                    raise RuntimeError("tail worker exceeded RSS bound")
                if time.monotonic() - start > min(plan["worker_timeout_seconds"], remaining):
                    raise RuntimeError("tail worker or total work-time bound exceeded")
                time.sleep(0.5)
            code = child.wait()
            if code != 0 or read_sealed(result_path)["request_sha256"] != key:
                raise RuntimeError("tail worker failed or returned a mismatched result")
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
        write_sealed(usage_path, usage)
    return usage


def analyze(out: Path, plan: dict, usage: list[dict]) -> None:
    results = []
    for job in plan["jobs"]:
        key = digest({"plan_sha256": digest(plan), "job": job})
        result = read_sealed(out / "runs" / (key + ".result.json"))
        if result["request_sha256"] != key:
            raise ValueError("tail analysis result binding mismatch")
        results.append(result)
    horizon_checks = []
    for shorter in (r for r in results if r["horizon"] == 4800.0):
        longer = next(
            r
            for r in results
            if r["rate"] == shorter["rate"]
            and r["seed"] == shorter["seed"]
            and r["horizon"] == 5000.0
        )
        horizon_checks.append(
            {
                "rate": shorter["rate"],
                "seed": shorter["seed"],
                "primary_cohort_identical": shorter["primary"] == longer["primary"],
            }
        )
    rates = [
        analyze_rate([r for r in results if r["rate"] == rate and r["horizon"] == 5000.0], rate)
        for rate in ARRIVAL_RATES
    ]
    write_sealed(
        out / "RESULTS.json",
        {
            "plan_sha256": digest(plan),
            "rates": rates,
            "horizon_checks": horizon_checks,
            "replicates": results,
            "all_fixed_checks_pass": all(c["primary_cohort_identical"] for c in horizon_checks)
            and all(r["status"] == "within-fixed-tolerances" for r in rates),
            "total_worker_seconds": sum(u["elapsed_seconds"] for u in usage),
            "peak_rss_mib": max(u["peak_rss_mib"] for u in usage),
            "boundary": plan["boundary"],
        },
    )


def run(out: Path, wait_for_matrix: bool) -> None:
    # Holding the shared lock prevents simultaneous matrix and tail simulations.
    with (MATRIX / "execution.lock").open("a") as shared:
        print(
            "waiting for the sensitivity execution lock"
            if wait_for_matrix
            else "acquiring the sensitivity execution lock without waiting",
            flush=True,
        )
        flags = fcntl.LOCK_EX if wait_for_matrix else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(shared, flags)
        plan = verify_source(out)
        complete = read_sealed(MATRIX / "COMPLETE.json")
        if complete["plan_sha256"] != plan["matrix_plan_sha256"] or complete["requests"] != 174:
            raise ValueError("sensitivity matrix has not completed its declared grid")
        with (out / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (out / "RESULTS.json").exists():
                raise ValueError("tail witness already complete; verify instead of rerunning")
            usage = []
            for job in plan["jobs"]:
                remaining = plan["maximum_total_worker_seconds"] - sum(
                    u["elapsed_seconds"] for u in usage
                )
                if remaining <= 0:
                    raise RuntimeError("tail total work-time cap exhausted")
                usage.append(guarded(out, job, plan, remaining))
                print(
                    f"tail {len(usage)}/{len(plan['jobs'])}: rate={job['rate']}, "
                    f"replicate={job['replicate']}, horizon={job['horizon']}",
                    flush=True,
                )
            verify_source(out)
            analyze(out, plan, usage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker")
    parser.add_argument("--after-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out)
    elif args.worker:
        worker(args.out, args.worker)
    else:
        run(args.out, args.after_sensitivity)
