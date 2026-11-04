"""Recover GP evidence and optionally rescore with one RSS/time-limited child at a time.

Outputs are separate from original results. Caches bind source, seeds, task and
evaluation-code snapshot; legacy unbound caches are not silently adopted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import placementscript
import psutil

from experiments.gp_tuning_20260902 import analyze
from experiments.gp_tuning_20260902.recovery import digest, read_sealed, recover_gp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "recovered_20260909"


def write_sealed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        {"payload": payload, "payload_sha256": digest(payload)}, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def evaluation_snapshot() -> dict:
    files = {}
    for folder in ("analysis", "baselines", "evolution", "sim", "runtime/policies"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".ps"}:
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("recovery.py", "run_recovered.py", "analyze.py"):
        path = Path(__file__).parent / name
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    native_dir = Path(placementscript.__file__).parent
    native = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(native_dir.glob("*")) if p.suffix in {".py", ".so", ".dylib"}
    }
    return {
        "files": files, "native": native, "python": sys.version, "platform": platform.platform(),
        "packages": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
    }


def locate(root: int, mix: str, budget: float) -> Path:
    if mix == "current" and budget == 56.0:
        return analyze.RUNS_FINAL / analyze.final_run_name(root)
    name = analyze.tuned_run_name(root, mix, budget)
    candidates = [base / name for base in (analyze.RUNS_ROOT, analyze.EARLY_224_ROOT)]
    candidates = [p for p in candidates if (p / "archives/same_dsl_gp/events.jsonl").exists()]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one archive for {mix}/{budget}/{root}: {candidates}")
    return candidates[0]


def worker(request_path: Path, output_path: Path) -> None:
    request = read_sealed(request_path)
    if digest(evaluation_snapshot()) != request["evaluation_snapshot_sha256"]:
        raise ValueError("evaluation source/environment changed before worker started")
    task = analyze.load_task_evaluation_config(analyze.TASK_CONFIG)
    source = request["champion_source"]
    if hashlib.sha256(source.encode()).hexdigest() != request["champion_source_sha256"]:
        raise ValueError("request source digest mismatch")
    result = analyze.rescore_champion(source, request["root_seed"], task)
    if result["fresh_seeds"] != request["fresh_seeds"]:
        raise ValueError("worker fresh seed grid changed")
    if digest(evaluation_snapshot()) != request["evaluation_snapshot_sha256"]:
        raise ValueError("evaluation source/environment changed during worker execution")
    write_sealed(output_path, {"request_sha256": digest(request), "result": result})


def guarded_score(request_path: Path, output_path: Path, *, max_rss_mb: int,
                  timeout_seconds: int) -> dict:
    """Kill/reap only our evaluation child when it exceeds a measured bound."""
    command = [sys.executable, "-m", "experiments.gp_tuning_20260902.run_recovered",
               "--worker", str(request_path), str(output_path)]
    start = time.monotonic()
    peak = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            observed = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss = observed.memory_info().rss
                except psutil.NoSuchProcess:
                    break
                peak = max(peak, rss)
                if rss > max_rss_mb * 1024**2:
                    raise RuntimeError(f"evaluation exceeded RSS limit {max_rss_mb} MiB")
                if time.monotonic() - start > timeout_seconds:
                    raise RuntimeError(f"evaluation exceeded {timeout_seconds}s limit")
                time.sleep(1)
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f"evaluation failed ({returncode}); see {log.name}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return {"peak_rss_bytes": peak, "elapsed_seconds": time.monotonic() - start,
            "rss_limit_mib": max_rss_mb, "timeout_seconds": timeout_seconds}


def render(payload: dict) -> str:
    lines = ["# Recovered GP operator comparison", "",
             "Retrospective development analysis; original campaigns and reports unchanged.", "",
             "All recovered arms are validated independently of unrelated baselines.", "",
             "| GP mix | Work | Recovered | Scored | LLM wins | GP wins | Ties | Exact p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in payload["cells"]:
        result = cell["comparison"]
        p = result["exact_p"]
        p_text = "—" if p is None else f"{p:.7g}"
        lines.append(f"| {cell['mix']} | {cell['budget']} | {cell['recovered']} | "
                     f"{len(cell['pairs'])} | {result['llm_wins']} | {result['gp_wins']} | "
                     f"{result['ties']} | {p_text} |")
    lines += ["", "Counts use the original feasibility-first rule. Incomplete cells are not final.",
              "Budget controls match evaluator work, not total compute or energy.",
              "The best-cell-per-root comparison is an exploratory hindsight envelope,",
              "not the performance of a configuration chosen before held-out evaluation."]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    snapshot = evaluation_snapshot()
    snapshot_sha = digest(snapshot)
    source_path = out / "evaluation_snapshot.json"
    if source_path.exists() and digest(read_sealed(source_path)) != snapshot_sha:
        raise ValueError("evaluation snapshot changed; use a new output directory")
    write_sealed(source_path, snapshot)
    if not (out / "evaluation_sources.tar.gz").exists():
        with tarfile.open(out / "evaluation_sources.tar.gz", "w:gz") as bundle:
            for relative in snapshot["files"]:
                bundle.add(ROOT / relative, arcname=relative)
            for index, filename in enumerate(snapshot["native"]):
                bundle.add(filename, arcname=f"native/{index}-{Path(filename).name}")
    original = read_sealed(analyze.LLM_RESCORE)
    llm = {r["root_seed"]: r for r in original["rows"] if r["arm"] == "llm"}
    old_gp = {r["root_seed"]: r for r in original["rows"] if r["arm"] == "same_dsl_gp"}
    payload = {"schema": "evolved-continuum-gp-recovery-analysis-v1",
               "evaluation_snapshot_sha256": snapshot_sha,
               "original_rescore_payload_sha256": digest(original), "cells": []}

    def checkpoint() -> None:
        write_sealed(out / "RESULTS.json", payload)
        (out / "RESULTS.md").write_text(render(payload))

    # Recover the entire grid before starting any simulation, preserving parity evidence.
    recovered = {}
    for mix, budget in analyze.CELLS:
        for root in analyze.ROOTS:
            receipt = recover_gp(locate(root, mix, budget))
            name = f"{mix}-{budget}-{root}"
            write_sealed(out / "arms" / f"{name}.json", receipt)
            recovered[name] = receipt
            print(f"recovered {name} report_agrees={receipt['existing_report_agrees']}", flush=True)
    for mix, budget in analyze.CELLS:
        cell = {"mix": mix, "budget": budget, "recovered": 8, "pairs": [],
                "comparison": analyze.sign_test([])}
        payload["cells"].append(cell)
        for root in analyze.ROOTS:
            name = f"{mix}-{budget}-{root}"
            receipt = recovered[name]
            if mix == "current" and budget == 56.0:
                held = old_gp[root]
            else:
                request = {"root_seed": root,
                           "champion_source": receipt["champion_source"],
                           "champion_source_sha256": receipt["champion_source_sha256"],
                           "evaluation_snapshot_sha256": snapshot_sha,
                           "task_config_sha256": hashlib.sha256(
                               analyze.TASK_CONFIG.read_bytes()).hexdigest(),
                           "fresh_seeds": list(analyze.fresh_grid(root, analyze.SCENARIO))}
                # Multiple mixes often select byte-identical programs for the same root.
                # Reuse only when the entire source/seed/task/environment request agrees.
                path = out / "scores" / f"{digest(request)}.json"
                if not path.exists() and args.rescore:
                    request_path = out / "requests" / f"{name}.json"
                    write_sealed(request_path, request)
                    print(f"scoring {name}", flush=True)
                    usage = guarded_score(request_path, path, max_rss_mb=args.max_rss_mb,
                                          timeout_seconds=args.timeout_seconds)
                    write_sealed(out / "usage" / f"{name}.json", usage)
                    print(f"scored {name} peak_rss_mib={usage['peak_rss_bytes']/1024**2:.1f}",
                          flush=True)
                if not path.exists():
                    checkpoint()
                    continue
                cached = read_sealed(path)
                if cached["request_sha256"] != digest(request):
                    raise ValueError(f"stale rescore cache: {path}")
                held = cached["result"]
            cell["pairs"].append({"root": root,
                                  "gp_feasible": held["fresh_grid_feasible"],
                                  "gp_energy": held["fresh_grid_energy"],
                                  "llm_feasible": llm[root]["fresh_grid_feasible"],
                                  "llm_energy": llm[root]["fresh_grid_energy"]})
            cell["comparison"] = analyze.sign_test(cell["pairs"])
            checkpoint()
    payload["complete"] = all(len(c["pairs"]) == 8 for c in payload["cells"])
    payload["gp_hindsight_envelope"] = analyze.sign_test(
        analyze._oracle_pairs(payload["cells"], llm)
    )
    checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--max-rss-mb", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--worker", nargs=2, type=Path, metavar=("REQUEST", "OUTPUT"))
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "process.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
