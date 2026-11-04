"""EXPLORATORY post-study analyses of the corrected A+B study (not confirmatory).

(1) Brittleness mechanism: re-evaluate each arm's top-K search (L2) candidates on a separate
    validation grid that is disjoint from every search, final and pilot seed; re-select a
    "validated champion" per arm by the feasibility-first sort key on the validation grid; then
    test validated champions on the root's final grid (never used for selection).
(2) Energy-model sensitivity (study A): re-evaluate selected policies and the robust reference
    on the final grids under the frozen workload-sensitivity transform (CPU attribution 0.30 and
    0.60; observed mean/p95 service-time shape at 0.45).

Rules are fixed in artifacts/PLAN.json by `prepare` before any job runs. Results never alter
the registered analysis in experiments/corrected_ab_20260925/analysis/.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import psutil

from analysis.tracking import derive_seed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "artifacts"
STUDY = ROOT / "experiments/corrected_ab_20260925"
V3 = ROOT / "prereg/CORRECTED_STUDIES_20260920_v3"
TOP_K = 5
VALIDATION_SEEDS = {"A": 4, "B": 2}
SENSITIVITY_CELLS = (("original", 0.30), ("original", 0.60), ("observed-mean-p95", 0.45))
SENSITIVITY_LABELS = ("llm_champion", "gp_champion", "initialization_best",
                      "reference:latency_greedy")
WORKERS = int(os.environ.get("BRITTLENESS_WORKERS", "3"))
START_MIB = 2560


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def l2_rows(archive):
    rows = []
    for line in (archive / "events.jsonl").open():
        event = json.loads(line)
        if event["type"] != "candidate_completed":
            continue
        payload = event["payload"]
        evaluation = payload.get("evaluation") or {}
        if evaluation.get("promoted_to_l2") and evaluation.get("evaluation"):
            rows.append(payload)
    return rows


def top_k(archive, k):
    rows = sorted(l2_rows(archive),
                  key=lambda r: tuple(r["evaluation"]["evaluation"]["fitness"]["sort_key"]))
    out, seen = [], set()
    for row in rows:
        source = row["proposal"]["source"]
        key = sha_text(source)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source_sha256": key, "source": source, "proposal_seq": row["proposal_seq"],
                    "search_sort_key": row["evaluation"]["evaluation"]["fitness"]["sort_key"]})
        if len(out) == k:
            break
    return out


def occupied_seeds():
    from experiments.prospective_20260917 import build

    seeds = {r["seed"] for r in json.loads((V3 / "SEEDS.json").read_text())
             ["enumerated_seed_records"]} | build.historical_seeds()
    for plan in (ROOT / "experiments").glob("*/artifacts/PLAN.json"):
        try:
            data = json.loads(plan.read_text())
        except ValueError:
            continue
        for row in data.get("new_search_seed_grid", []) + data.get("search_seed_grid", []):
            seeds.add(row["seed"])
        for grid in data.get("audit_seed_grid", {}).values():
            seeds |= set(grid.values())
    return seeds


def label_source(campaign_id, label):
    final = STUDY / "runs" / campaign_id / "final"
    for sha, names in json.loads((final / "LABELS.json").read_text())["labels"].items():
        if label in names:
            return sha, (final / f"{sha}.ps").read_text()
    return None, None


def prepare():
    from evolution.evaluation import load_task_evaluation_config

    OUT.mkdir(parents=True)
    sources_dir = OUT / "sources"
    sources_dir.mkdir()
    occupied = occupied_seeds()
    seeds_doc = json.loads((V3 / "SEEDS.json").read_text())
    jobs, candidates = [], {}
    for study in ("A", "B"):
        scenarios = [s.scenario_id for s in
                     load_task_evaluation_config(V3 / f"task_{study}.json").scenarios]
        for row in seeds_doc["studies"][study]:
            cid = row["campaign_id"]
            validation = {}
            for scenario in scenarios:
                validation[scenario] = []
                for j in range(VALIDATION_SEEDS[study]):
                    seed = derive_seed(20261002, "exploratory-validation", study, cid, scenario, j)
                    if seed in occupied:
                        raise ValueError("validation seed collides with a registered seed")
                    occupied.add(seed)
                    validation[scenario].append(seed)
            per_arm = {}
            for arm in ("llm", "gp"):
                per_arm[arm] = top_k(STUDY / "runs" / cid / arm / "campaign", TOP_K)
            candidates[cid] = {"study": study, "validation_seeds": validation,
                               "final_seeds": {s: row["grids"][s]["final"] for s in scenarios},
                               "arms": {a: [{k: v for k, v in c.items() if k != "source"}
                                            for c in cs] for a, cs in per_arm.items()}}
            distinct = {c["source_sha256"]: c["source"] for cs in per_arm.values() for c in cs}
            for key, source in distinct.items():
                path = sources_dir / f"{key}.ps"
                if not path.exists():
                    path.write_text(source)
                jobs.append({"id": f"validation:{cid}:{key}", "kind": "validation",
                             "study": study, "root": cid, "source_sha256": key,
                             "seeds": validation, "transform": None})
            if study == "A":
                for label in SENSITIVITY_LABELS:
                    key, source = label_source(cid, label)
                    if key is None:
                        continue
                    path = sources_dir / f"{key}.ps"
                    if not path.exists():
                        path.write_text(source)
                    for shape, fraction in SENSITIVITY_CELLS:
                        job_id = f"sensitivity:{cid}:{key}:{shape}:{fraction:.2f}"
                        if any(j["id"] == job_id for j in jobs):
                            continue
                        jobs.append({"id": job_id, "kind": "sensitivity", "study": study,
                                     "root": cid, "source_sha256": key, "label": label,
                                     "seeds": candidates[cid]["final_seeds"],
                                     "transform": {"shape": shape, "fraction": fraction}})
    save_once(OUT / "PLAN.json", {
        "status": "EXPLORATORY; not part of the registered analysis",
        "authorization": "Investigator, 2 October 2026: brittleness mechanism, energy-model sensitivity "
        "and execution-cost measurements.",
        "top_k": TOP_K,
        "validation_seeds_per_scenario": VALIDATION_SEEDS,
        "selection_rule": "Per arm, the validated champion is the top-K candidate with the "
        "smallest feasibility-first sort key on the validation grid (base constraints, all "
        "three bundles); ties broken by the search sort key, then proposal sequence. Final seeds "
        "are never used for selection.",
        "test_rule": "Validated champions are scored on the root's registered final grid; "
        "existing final records are reused when the source was already scored there.",
        "sensitivity_cells": SENSITIVITY_CELLS,
        "sensitivity_labels": SENSITIVITY_LABELS,
        "sensitivity_scope": "study A only (the CPU-attribution transform applies to the "
        "Alibaba-derived service times); evaluated on the root's final grid",
        "candidates": candidates,
        "jobs": jobs,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print(json.dumps({"jobs": len(jobs),
                      "validation": sum(j["kind"] == "validation" for j in jobs),
                      "sensitivity": sum(j["kind"] == "sensitivity" for j in jobs)}))


def task_for(job):
    from evolution.evaluation import ScenarioGrid, load_task_evaluation_config

    task = load_task_evaluation_config(V3 / f"task_{job['study']}.json")
    return replace(task, scenarios=tuple(
        ScenarioGrid(scenario_id=s.scenario_id, path=s.path,
                     seeds=tuple(job["seeds"][s.scenario_id])) for s in task.scenarios))


def evaluate_job(job):
    import placementscript as ps

    from evolution.evaluation import PolicyExecutionError, evaluate_policy_source

    source = (OUT / "sources" / f"{job['source_sha256']}.ps").read_text()
    transform = None
    if job["transform"]:
        from experiments.gp_tuning_20260902.recovery import read_sealed
        from experiments.validation_20260909.workload_sensitivity import transform as wt

        summary = read_sealed(ROOT / "experiments/validation_20260909/"
                                     "fit_diagnostics_20260909_v2/SUMMARY.json")
        fits = next(s["selected_fit_errors"] for s in summary["scenarios"]
                    if s["scenario_id"] == "dev-continuum-tight")
        cell = job["transform"]

        def transform(config):
            return wt(config, shape=cell["shape"], fraction=cell["fraction"], fits=fits)

    start = time.monotonic()
    try:
        evaluation = evaluate_policy_source(source, task_for(job), proposal_seq=0,
                                            config_transform=transform,
                                            retain_episode_ledgers=False)
        fitness = evaluation.fitness
        instances = evaluation.to_dict()["instances"]
        result = {"valid": True, "feasible": fitness.feasible,
                  "normalized_energy": fitness.normalized_energy,
                  "summed_primitive_shortfall": fitness.summed_primitive_shortfall,
                  "sort_key": list(fitness.sort_key())[:-1],
                  "instances": instances}
    except (PolicyExecutionError, ps.PlacementScriptError) as exc:
        result = {"valid": False, "feasible": False, "error": f"{type(exc).__name__}: {exc}"}
    result.update(job_id=job["id"], seconds=time.monotonic() - start)
    return result


def result_path(job):
    return OUT / "results" / (hashlib.sha256(job["id"].encode()).hexdigest() + ".json")


def run(kinds):
    plan = json.loads((OUT / "PLAN.json").read_text())
    runner = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    amended = OUT / "RUNNER_AMENDMENTS.json"
    allowed = {plan["runner_sha256"]}
    if amended.exists():
        allowed |= {a["runner_sha256"] for a in json.loads(amended.read_text())["amendments"]}
    if runner not in allowed:
        raise ValueError("runner changed after the plan was fixed")
    jobs = [j for j in plan["jobs"] if j["kind"] in kinds and not result_path(j).exists()]
    if "final_test" in kinds:
        for extra in sorted(OUT.glob("FINAL_TEST_JOBS_*.json")):
            jobs += [j for j in json.loads(extra.read_text())["jobs"]
                     if not result_path(j).exists()]
    (OUT / "results").mkdir(exist_ok=True)
    # Study A first (validation, then sensitivity, then final tests), then study B.
    order = {"validation": 0, "sensitivity": 1, "final_test": 2}
    jobs.sort(key=lambda j: (j["study"] != "A", order[j["kind"]], j["id"]))
    with ProcessPoolExecutor(max_workers=WORKERS, max_tasks_per_child=1) as pool:
        pending = {}
        while jobs or pending:
            while jobs and len(pending) < WORKERS and \
                    psutil.virtual_memory().available >= START_MIB * 2**20:
                job = jobs.pop(0)
                pending[pool.submit(evaluate_job, job)] = job
            done, _ = wait(list(pending), timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                save_once(result_path(job), {"job": job, **future.result()})
                print(json.dumps({"done": job["id"][:60], "left": len(jobs) + len(pending)}),
                      flush=True)


def select_validated(study):
    """Apply the fixed selection rule for one study; write its final-test jobs."""
    plan = json.loads((OUT / "PLAN.json").read_text())
    jobs = {j["id"]: j for j in plan["jobs"]}
    selections, final_jobs = {}, []
    for cid, info in plan["candidates"].items():
        if info["study"] != study:
            continue
        selections[cid] = {}
        for arm, cands in info["arms"].items():
            scored = []
            for c in cands:
                job = jobs[f"validation:{cid}:{c['source_sha256']}"]
                result = json.loads(result_path(job).read_text())
                key = result["sort_key"] if result["valid"] else [2.0]
                scored.append((tuple(key), tuple(c["search_sort_key"]), c["proposal_seq"],
                               c["source_sha256"]))
            if not scored:
                selections[cid][arm] = None
                continue
            best = min(scored)
            selections[cid][arm] = {"source_sha256": best[3], "validation_sort_key": best[0],
                                    "search_rank": [s[3] for s in sorted(
                                        scored, key=lambda s: (s[1], s[2]))].index(best[3])}
            sha = best[3]
            existing = STUDY / "runs" / cid / "final" / f"{sha}.json"
            if not existing.exists():
                job_id = f"final_test:{cid}:{sha}"
                if all(j["id"] != job_id for j in final_jobs):
                    final_jobs.append({"id": job_id, "kind": "final_test",
                                       "study": info["study"], "root": cid,
                                       "source_sha256": sha, "seeds": info["final_seeds"],
                                       "transform": None})
    save_once(OUT / f"VALIDATED_SELECTIONS_{study}.json", {"rule": plan["selection_rule"],
                                                           "selections": selections})
    save_once(OUT / f"FINAL_TEST_JOBS_{study}.json", {"jobs": final_jobs})
    print(json.dumps({"final_test_jobs": len(final_jobs),
                      "A": sum(j["study"] == "A" for j in final_jobs),
                      "B": sum(j["study"] == "B" for j in final_jobs)}))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        prepare()
    elif action == "run":
        run(set(sys.argv[2].split(",")))
    elif action == "select":
        select_validated(sys.argv[2])
