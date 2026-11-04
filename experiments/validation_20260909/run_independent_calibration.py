"""Bounded prospective calibration of the new synthetic family; no model calls.

One episode per worker. Freeze only after earlier studies release their shared lock
and the effective-config reload defect is fixed. No comparative data is read by freeze.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

from evolution.evaluation import (
    PolicyExecutionError,
    ScenarioGrid,
    TaskEvaluationConfig,
    config_for_development_bundle,
    evaluate_policy_source,
)
from experiments.gp_tuning_20260902.recovery import canonical, digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot
from experiments.validation_20260909 import independent_scenarios as design_module
from sim.energy import build_development_bundles, load_pinned_sources
from sim.energy.boundary import BoundaryBounds, BoundaryWeights
from sim.energy.fitness import ConstraintSpec
from sim.spec import (
    config_sha256,
    episode_config_from_json,
    episode_config_to_json,
    load_episode_config,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "independent_calibration_20260909_v1"
MATRIX = HERE / "workload_sensitivity_20260909"
TAILS = HERE / "analytical_tails_20260909"
MAX_SECONDS = 21600
PILOT_MAX_SECONDS = 3600
WORKER_SECONDS = 1200
RSS_MIB = 2048
RESERVE_FACTOR = 1.30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_once(path: Path, payload: dict) -> None:
    """Exclusive, fsynced evidence write; partial writes remain refusal markers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(
            json.dumps(
                {"payload": payload, "payload_sha256": digest(payload)},
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preserve_or_match(path: Path, payload: dict) -> None:
    if path.exists():
        if read_sealed(path) != payload:
            raise ValueError(f"existing evidence differs: {path}")
    else:
        store_once(path, payload)


@contextmanager
def shared_lock(path: Path):
    # This is the same inode used by both earlier simulator controllers.
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prerequisites() -> dict:
    matrix_plan = read_sealed(MATRIX / "PLAN.json")
    matrix = read_sealed(MATRIX / "COMPLETE.json")
    amendment = read_sealed(MATRIX / "RESOURCE_AMENDMENT.json")
    tail_plan = read_sealed(TAILS / "PLAN.json")
    tails = read_sealed(TAILS / "RESULTS.json")
    if (
        matrix["plan_sha256"] != digest(matrix_plan)
        or matrix["requests"] != 174
        or matrix["resource_amendment_sha256"] != digest(amendment)
        or tails["plan_sha256"] != digest(tail_plan)
        or len(tails["replicates"]) != 136
        or tail_plan["matrix_plan_sha256"] != digest(matrix_plan)
        or {(r["rate"], r["seed"], r["horizon"]) for r in tails["replicates"]}
        != {(j["rate"], j["seed"], j["horizon"]) for j in tail_plan["jobs"]}
    ):
        raise ValueError("earlier studies lack bound completion receipts")
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            MATRIX / "PLAN.json",
            MATRIX / "COMPLETE.json",
            MATRIX / "RESOURCE_AMENDMENT.json",
            TAILS / "PLAN.json",
            TAILS / "RESULTS.json",
        )
    }


def snapshot() -> dict:
    state = evaluation_snapshot()
    paths = [
        Path(__file__),
        HERE / "independent_scenarios.py",
        HERE / "summarize_independent_calibration.py",
        HERE / "INDEPENDENT_SCENARIOS.md",
        HERE / "INDEPENDENT_CALIBRATION_RUNNER.md",
        ROOT / "tests/test_independent_scenarios.py",
        ROOT / "tests/test_independent_calibration.py",
        ROOT / "tests/test_independent_calibration_summary.py",
        ROOT / "tests/test_energy_source_json.py",
        ROOT / "experiments/matched_control_20260909/real_pilot.py",
        ROOT / "uv.lock",
    ]
    paths.extend(design_module.OUT.glob("*.json"))
    design = json.loads((design_module.OUT / "DESIGN.json").read_text())
    paths.extend(ROOT / p["path"] for p in design["fixed_references"])
    for path in paths:
        state["files"][str(path.relative_to(ROOT))] = sha(path)
    return json.loads(canonical(state))


def jobs(design: dict, phase: str) -> list[dict]:
    if phase not in ("pilot", "calibration"):
        raise ValueError("unknown job phase")
    result = []
    for cell in design["cells"]:
        if phase == "pilot" and cell["id"] != "edge_heavy-two_bursts-0.45":
            continue
        for policy in design["fixed_references"]:
            if phase == "pilot" and policy["label"] != "graph_headroom":
                continue
            for effective in cell["effective_configs"]:
                role = "resource_pilot" if phase == "pilot" else "calibration"
                for seed in design["seed_design"]["roles"][role]:
                    result.append(
                        {
                            "phase": phase,
                            "cell_id": cell["id"],
                            "label": policy["label"],
                            "source_sha256": policy["source_sha256"],
                            "seed": seed,
                            "bundle_id": effective["bundle_id"],
                            "effective_config_sha256": effective["config_sha256"],
                        }
                    )
    expected = 3 if phase == "pilot" else 972
    if len(result) != expected or len({digest(j) for j in result}) != expected:
        raise ValueError("job grid incomplete or duplicated")
    return result


def verify_archive(out: Path, frozen: dict, expected_sha: str) -> None:
    archive_path = out / "sources.tar.gz"
    if sha(archive_path) != expected_sha:
        raise ValueError("source archive bytes changed")
    expected = dict(frozen["files"])
    for i, name in enumerate(frozen["native"]):
        expected[f"native/{i}-{Path(name).name}"] = frozen["native"][name]
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("source archive member set differs")
        for member in members:
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected[member.name]:
                raise ValueError("source archive member hash differs")


def freeze(out: Path) -> None:
    with shared_lock(MATRIX / "execution.lock"):
        prior = prerequisites()
        if out.exists():
            raise ValueError("freeze requires a new directory")
        design, files = design_module.build()
        design = json.loads(canonical(design))
        saved = json.loads((design_module.OUT / "DESIGN.json").read_text())
        if design != saved or any(
            (design_module.OUT / n).read_text() != v for n, v in files.items()
        ):
            raise ValueError("scenario design source or generated bytes changed")
        # Require the known public-loader repair before freezing another environment.
        base = load_episode_config(design_module.OUT / (design["cells"][0]["id"] + ".json"))
        for bundle in build_development_bundles(load_pinned_sources()):
            effective = config_for_development_bundle(base, bundle)
            if config_sha256(
                episode_config_from_json(episode_config_to_json(effective))
            ) != config_sha256(effective):
                raise ValueError("effective configuration cannot round trip")
        frozen = snapshot()
        out.mkdir()
        store_once(out / "snapshot.json", frozen)
        with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
            for name in sorted(frozen["files"]):
                archive.add(ROOT / name, arcname=name)
            for i, name in enumerate(frozen["native"]):
                archive.add(name, arcname=f"native/{i}-{Path(name).name}")
        archive_sha = sha(out / "sources.tar.gz")
        verify_archive(out, frozen, archive_sha)
        for policy in design["fixed_references"]:
            source = (ROOT / policy["path"]).read_bytes()
            if hashlib.sha256(source).hexdigest() != policy["source_sha256"]:
                raise ValueError("fixed reference source changed")
            destination = out / "policies" / (policy["source_sha256"] + ".ps")
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != source:
                    raise ValueError("source alias differs")
            else:
                destination.write_bytes(source)
        plan = {
            "schema": "independent-family-calibration-v1",
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "design": design,
            "snapshot_sha256": digest(frozen),
            "source_archive_sha256": archive_sha,
            "prior_completion_files": prior,
            "pilot_jobs": jobs(design, "pilot"),
            "calibration_jobs": jobs(design, "calibration"),
            "rss_limit_mib": RSS_MIB,
            "worker_timeout_seconds": WORKER_SECONDS,
            "pilot_total_seconds": PILOT_MAX_SECONDS,
            "calibration_total_seconds": MAX_SECONDS,
            "projection_reserve_factor": RESERVE_FACTOR,
            "one_episode_per_worker": True,
            "no_model_calls": True,
            "scientific_eligibility": "development calibration only",
        }
        if snapshot() != frozen:
            raise ValueError("source changed while freezing")
        store_once(out / "PLAN.json", plan)
        print("frozen three pilot and 972 calibration requests; no simulation launched", flush=True)


def verify_source(out: Path) -> dict:
    plan, frozen = read_sealed(out / "PLAN.json"), read_sealed(out / "snapshot.json")
    if digest(frozen) != plan["snapshot_sha256"] or snapshot() != frozen:
        raise ValueError("frozen source or environment changed")
    expected_limits = {
        "rss_limit_mib": RSS_MIB,
        "worker_timeout_seconds": WORKER_SECONDS,
        "pilot_total_seconds": PILOT_MAX_SECONDS,
        "calibration_total_seconds": MAX_SECONDS,
        "projection_reserve_factor": RESERVE_FACTOR,
    }
    if any(plan.get(k) != v for k, v in expected_limits.items()):
        raise ValueError("recorded resource limits differ from the frozen controller")
    if plan["design"] != json.loads((design_module.OUT / "DESIGN.json").read_text()):
        raise ValueError("plan design differs from its captured input")
    if any(plan[p + "_jobs"] != jobs(plan["design"], p) for p in ("pilot", "calibration")):
        raise ValueError("plan job grid differs from the fixed design")
    for name, expected in plan["prior_completion_files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("prior completion evidence changed")
    return plan


def request(plan: dict, job: dict) -> dict:
    if job["phase"] not in ("pilot", "calibration") or job not in plan[job["phase"] + "_jobs"]:
        raise ValueError("request outside the frozen grid")
    return {"plan_sha256": digest(plan), "job": job}


def validate_result(req: dict, result: dict) -> None:
    if result.get("request_sha256") != digest(req):
        raise ValueError("result request binding mismatch")
    job = req["job"]
    if result.get("effective_config_sha256") != job["effective_config_sha256"]:
        raise ValueError("result effective config identity differs")
    if result.get("status") == "policy-fault":
        failure = result.get("failure", {})
        if (failure.get("scenario_id"), failure.get("seed"), failure.get("bundle_id")) != (
            job["cell_id"],
            job["seed"],
            job["bundle_id"],
        ):
            raise ValueError("policy fault episode identity differs")
        if "evaluation" in result:
            raise ValueError("policy fault must not carry invented evaluation metrics")
        return
    if result.get("status") != "evaluated":
        raise ValueError("unknown completed result status")
    evaluation = result["evaluation"]
    f = evaluation["fitness"]
    if any(type(f[k]) is not int for k in ("n_instances", "n_bundles", "proposal_seq")) or (
        f["n_instances"],
        f["n_bundles"],
        f["proposal_seq"],
    ) != (1, 1, 0):
        raise ValueError("one-episode fitness cardinality differs")
    if (
        evaluation["policy"]["artifact_sha256"] != job["source_sha256"]
        or len(evaluation["instances"]) != 1
    ):
        raise ValueError("result source or episode count differs")
    instance = evaluation["instances"][0]
    if (instance["scenario_id"], instance["seed"], instance["bundle_id"]) != (
        job["cell_id"],
        job["seed"],
        job["bundle_id"],
    ):
        raise ValueError("measured episode identity differs")


def write_offered_trace(path: Path, records) -> dict:
    """Stream every offered root record; deterministic gzip, no extra full trace copy."""
    raw_digest, count = hashlib.sha256(), 0
    with path.open("xb") as target:
        with gzip.GzipFile(
            fileobj=target, filename="", mode="wb", mtime=0, compresslevel=6
        ) as stream:
            for row in records:
                line = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
                stream.write(line)
                raw_digest.update(line)
                count += 1
        target.flush()
        os.fsync(target.fileno())
    return {
        "file_name": path.name,
        "file_sha256": sha(path),
        "file_bytes": path.stat().st_size,
        "uncompressed_sha256": raw_digest.hexdigest(),
        "offered_records": count,
        "columns": ["rid", "template", "arrival_s", "end_s", "status"],
    }


def validate_trace_file(folder: Path, key: str, result: dict) -> None:
    path = folder / (key + ".records.jsonl.gz")
    if result["status"] == "policy-fault":
        if path.exists() or "offered_trace" in result:
            raise ValueError("policy fault must not carry an invented complete offered trace")
        return
    trace = result["offered_trace"]
    if (
        trace["file_name"] != path.name
        or trace["file_sha256"] != sha(path)
        or trace["file_bytes"] != path.stat().st_size
    ):
        raise ValueError("offered trace file binding differs")


def worker(out: Path, key: str, phase: str) -> None:
    if (
        phase not in ("pilot", "calibration")
        or len(key) != 64
        or any(c not in "0123456789abcdef" for c in key)
    ):
        raise ValueError("invalid internal worker identity")
    plan = verify_source(out)
    folder = out / phase
    req = read_sealed(folder / (key + ".request.json"))
    if key != digest(req) or req != request(plan, req["job"]) or req["job"]["phase"] != phase:
        raise ValueError("worker request differs from frozen plan")
    store_once(
        folder / (key + ".started.json"),
        {
            "request_sha256": key,
            "pid": os.getpid(),
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    job = req["job"]
    source_path = out / "policies" / (job["source_sha256"] + ".ps")
    if sha(source_path) != job["source_sha256"]:
        raise ValueError("copied policy source differs")
    design = plan["design"]
    config_path = design_module.OUT / (job["cell_id"] + ".json")
    base = load_episode_config(config_path)
    bundle = next(
        b
        for b in build_development_bundles(load_pinned_sources())
        if b.bundle_id == job["bundle_id"]
    )
    effective_sha = config_sha256(config_for_development_bundle(base, bundle))
    if effective_sha != job["effective_config_sha256"]:
        raise ValueError("effective config differs before simulation")
    task = TaskEvaluationConfig(
        scenarios=(ScenarioGrid(job["cell_id"], config_path, (job["seed"],)),),
        bundle_ids=(job["bundle_id"],),
        constraints=ConstraintSpec(**design["constraints"]),
        weights=BoundaryWeights(**design["boundary_weights"]),
        bounds=BoundaryBounds(**design["boundary_bounds"]),
    )
    try:
        evaluation = evaluate_policy_source(
            source_path.read_text(), task, proposal_seq=0, require_complete_development_set=False
        )
        trace = write_offered_trace(
            folder / (key + ".records.jsonl.gz"), evaluation.instances[0].episode["records"]
        )
        body = {
            "status": "evaluated",
            "evaluation": evaluation.without_episode_ledgers().to_dict(),
            "offered_trace": trace,
        }
    except PolicyExecutionError as error:
        body = {"status": "policy-fault", "failure": error.failure}
    result = {"request_sha256": key, "effective_config_sha256": effective_sha, **body}
    validate_result(req, result)
    verify_source(out)
    store_once(folder / (key + ".result.json"), result)


def validate_usage(req: dict, result: dict, usage: dict, plan: dict) -> None:
    validate_result(req, result)
    if (
        usage.get("request_sha256") != digest(req)
        or usage.get("result_sha256") != digest(result)
        or usage.get("exit_code") != 0
        or usage.get("stop_reason") is not None
    ):
        raise ValueError("request is not a successful completed execution")
    elapsed, rss = usage["elapsed_seconds"], usage["peak_rss_mib"]
    timeout = usage["timeout_seconds"]
    if (
        not all(math.isfinite(v) for v in (elapsed, rss, timeout))
        or elapsed <= 0
        or rss < 0
        or not 0 < timeout <= plan["worker_timeout_seconds"]
        or rss > plan["rss_limit_mib"]
        or elapsed > timeout
    ):
        raise ValueError("usage is invalid or exceeds the recorded resource bounds")


def guarded(out: Path, req: dict, plan: dict, remaining: float) -> dict:
    key = digest(req)
    folder = out / req["job"]["phase"]
    folder.mkdir(exist_ok=True)
    rp, resultp, usagep, logp = [
        folder / (key + suffix)
        for suffix in (".request.json", ".result.json", ".usage.json", ".log")
    ]
    startedp = folder / (key + ".started.json")
    artifacts = (rp, resultp, usagep, logp, startedp)
    if any(p.exists() for p in (*artifacts, folder / (key + ".records.jsonl.gz"))):
        if not all(p.exists() for p in artifacts) or read_sealed(rp) != req:
            raise ValueError("incomplete or changed attempt; inspect, do not retry")
        result, usage = read_sealed(resultp), read_sealed(usagep)
        validate_trace_file(folder, key, result)
        if read_sealed(startedp).get("request_sha256") != key or usage.get(
            "started_file_sha256"
        ) != sha(startedp):
            raise ValueError("worker-start evidence differs")
        validate_usage(req, result, usage, plan)
        return usage
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("no positive worker-time budget remains")
    store_once(rp, req)
    start, peak, child, code, reason, result_sha = time.monotonic(), 0, None, None, None, None
    timeout = min(remaining, plan["worker_timeout_seconds"])
    try:
        with logp.open("x") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.validation_20260909.run_independent_calibration",
                    "--out",
                    str(out),
                    "--worker",
                    req["job"]["phase"],
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
                    raise RuntimeError("worker RSS cap exceeded")
                if time.monotonic() - start > timeout:
                    raise RuntimeError("worker or total worker-time cap exceeded")
                time.sleep(0.25)
            code = child.wait()
            if code != 0:
                raise RuntimeError(f"worker exited {code}")
        if time.monotonic() - start > timeout:
            raise RuntimeError("completed child exceeded time allowance")
        result = read_sealed(resultp)
        validate_result(req, result)
        validate_trace_file(folder, key, result)
        if read_sealed(startedp).get("request_sha256") != key:
            raise ValueError("worker-start request identity differs")
        result_sha = digest(result)
    except BaseException as error:
        reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            code = child.wait()
        usage = {
            "request_sha256": key,
            "result_sha256": result_sha,
            "exit_code": code,
            "stop_reason": reason,
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_mib": peak / 1024**2,
            "timeout_seconds": timeout,
            "started_file_sha256": sha(startedp) if startedp.exists() else None,
        }
        store_once(usagep, usage)
    validate_usage(req, result, usage, plan)
    return usage


def pilot_review(plan: dict, usages: list[dict]) -> dict:
    if len(usages) != 3:
        raise ValueError("pilot projection requires all three bundle episodes")
    elapsed = [u["elapsed_seconds"] for u in usages]
    if any(not math.isfinite(v) or v <= 0 for v in elapsed):
        raise ValueError("invalid pilot timing")
    if sum(elapsed) > plan["pilot_total_seconds"]:
        raise ValueError("pilot total worker-time cap exceeded")
    projection = sum(elapsed) / 3 * len(plan["calibration_jobs"])
    reserved = projection * plan["projection_reserve_factor"]
    return {
        "plan_sha256": digest(plan),
        "pilot_usage_sha256": [digest(u) for u in usages],
        "linear_projection_seconds": projection,
        "with_reserve_seconds": reserved,
        "fits_cap": reserved <= plan["calibration_total_seconds"],
        "projection_is_not_a_guarantee": True,
        "quality_used_for_design": False,
    }


def run(out: Path, *, pilot_only: bool) -> None:
    with shared_lock(MATRIX / "execution.lock"), shared_lock(out / "execution.lock"):
        plan = verify_source(out)
        verify_archive(out, read_sealed(out / "snapshot.json"), plan["source_archive_sha256"])
        if (out / "COMPLETE.json").exists():
            raise ValueError("grid already complete; analyze instead of redispatching")
        usages = []
        for job in plan["pilot_jobs"]:
            remaining = plan["pilot_total_seconds"] - sum(u["elapsed_seconds"] for u in usages)
            usages.append(guarded(out, request(plan, job), plan, remaining))
        from experiments.validation_20260909 import summarize_independent_calibration as summary

        checked_pilot = summary.verify_phase(out, plan, "pilot")
        if checked_pilot["usages"] != usages:
            raise ValueError("pilot usages differ from verified phase")
        verify_source(out)
        preserve_or_match(out / "PILOT_CLOSURE.json", summary.pilot_closure(plan, checked_pilot))
        review = pilot_review(plan, usages)
        preserve_or_match(out / "PILOT_REVIEW.json", review)
        print(json.dumps(review), flush=True)
        if pilot_only:
            return
        if not review["fits_cap"]:
            raise ValueError("pilot projection plus reserve exceeds calibration cap")
        usages = []
        for job in plan["calibration_jobs"]:
            remaining = plan["calibration_total_seconds"] - sum(
                u["elapsed_seconds"] for u in usages
            )
            usages.append(guarded(out, request(plan, job), plan, remaining))
            print(
                f"completed calibration {len(usages)}/{len(plan['calibration_jobs'])}", flush=True
            )
        verify_source(out)
        if sum(u["elapsed_seconds"] for u in usages) > plan["calibration_total_seconds"]:
            raise ValueError("completed grid exceeded total worker-time cap")
        store_once(
            out / "COMPLETE.json",
            {
                "plan_sha256": digest(plan),
                "requests": len(usages),
                "usage_sha256": [digest(u) for u in usages],
                "total_worker_seconds": sum(u["elapsed_seconds"] for u in usages),
                "peak_rss_mib": max(u["peak_rss_mib"] for u in usages),
                "no_model_calls": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "KEY"))
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out)
    elif args.worker:
        worker(args.out, args.worker[1], args.worker[0])
    else:
        run(args.out, pilot_only=args.pilot)


if __name__ == "__main__":
    main()
