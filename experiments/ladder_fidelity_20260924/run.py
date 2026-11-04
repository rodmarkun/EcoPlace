"""P1 G-ladder fidelity audit for the frozen v3 ladders (tasks A and B).

Every corpus policy is evaluated at L0, L1 and the full L2 grid regardless of its rung
decisions, on audit seeds disjoint from every search, final, pilot and historical seed.
The criteria in PLAN.json are fixed by ``prepare`` before any audit evaluation runs.
No search, LLM call, final grid or selected champion is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import placementscript as ps
import psutil

from analysis.tracking import derive_seed
from evolution.evaluation import ScenarioGrid, evaluate_policy_source, load_task_evaluation_config
from evolution.ladder import _run_l0, _run_l1, load_ladder_config
from experiments.prospective_20260917 import build

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "artifacts"
V1 = ROOT / "prereg/CORRECTED_STUDIES_20260917_v1"
V3 = ROOT / "prereg/CORRECTED_STUDIES_20260920_v3"
SAMPLER_SEED = 20260924
QUOTA = {"A": 30, "B": 15}
STRATA = {
    "llm_offspring": "experiments/deep_search_20260825/runs_final/*/archives/llm/candidates/*.ps",
    "glm_offspring": "experiments/deep_search_20260825/runs_glm53*/*/archives/llm/candidates/*.ps",
    "gp_offspring": "experiments/gp_tuning_20260902/runs/*/archives/same_dsl_gp/candidates/*.ps",
    "other_arms": "experiments/allbaselines_harness/runs/*/archives/*/candidates/*.ps",
}
REFERENCES = (
    "runtime/policies/*.ps",
    "experiments/baseline_check_20260905/policies/*.py",
    "experiments/baseline_check_20260905/policies_done/*.py",
)
WORKERS = 3
MIN_AVAILABLE_MIB = 768
CRITERIA = {
    "F1_false_elimination": (
        "Among policies feasible at L2, the fraction rejected at L0 or L1 must be <= 0.05. "
        "Evaluable only with at least 10 L2-feasible policies; otherwise reported as "
        "underpowered, not as a pass."
    ),
    "F2_top_k_retention": (
        "Every policy among the L2 top 5 by the registered sort key must be promoted past "
        "L1. Applies to all corpus policies, feasible or not."
    ),
    "F3_descriptive": (
        "Spearman rank correlation between L1 and L2 sort-key ranks, L1 reject rate and "
        "L2 feasibility rate are reported without a pass threshold."
    ),
    "decision": (
        "The ladder passes for a task when F1 (if evaluable) and F2 hold. A failure means "
        "the ladder must change prospectively, with budgets recertified, before that "
        "task's search; it is never relabeled as a pass."
    ),
    "rung_rules": (
        "L0 validity-only; L1 promotes when normalized energy <= max_normalized_energy "
        "(or None) and summed primitive shortfall <= max_summed_primitive_shortfall, exactly "
        "as in evolution/ladder.py and the frozen v3 task files."
    ),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def compiles(source):
    try:
        ps.Policy(source)
        return True
    except Exception:
        return False


def prepare():
    OUT.mkdir(parents=True)
    excluded = set()
    for path in sorted((V3 / "initialization").glob("*.json")):
        excluded |= {text_sha(s) for s in strings(json.loads(path.read_text())) if "def score" in s}
    rng = random.Random(SAMPLER_SEED)
    pools = {}
    for stratum, pattern in STRATA.items():
        seen, pool = set(), []
        for path in sorted(ROOT.glob(pattern)):
            source = path.read_text()
            key = text_sha(source)
            if key in seen or key in excluded:
                continue
            seen.add(key)
            pool.append((str(path.relative_to(ROOT)), key))
        pools[stratum] = pool
    references, ref_seen = [], set()
    for pattern in REFERENCES:
        for path in sorted(ROOT.glob(pattern)):
            source = path.read_text()
            key = text_sha(source)
            if key not in ref_seen and key not in excluded and compiles(source):
                ref_seen.add(key)
                references.append({"path": str(path.relative_to(ROOT)), "sha256": key})
    corpus = {}
    taken = set(ref_seen)
    for study in "AB":
        chosen = [dict(r, stratum="reference") for r in references]
        for stratum, pool in pools.items():
            order = list(pool)
            rng.shuffle(order)
            picked = 0
            for rel, key in order:
                if picked == QUOTA[study]:
                    break
                if key in taken or not compiles((ROOT / rel).read_text()):
                    continue
                taken.add(key)
                chosen.append({"path": rel, "sha256": key, "stratum": stratum})
                picked += 1
        corpus[study] = chosen
    seeds = json.loads((V1 / "SEEDS.json").read_text())
    occupied = {r["seed"] for r in seeds["enumerated_seed_records"]} | build.historical_seeds()
    for plan in (
        ROOT / "experiments/deepseek_pilots_bounded_20260923/artifacts/PLAN.json",
        ROOT / "experiments/deepseek_concurrency_20260923/artifacts/PLAN.json",
    ):
        occupied |= {r["seed"] for r in json.loads(plan.read_text())["new_search_seed_grid"]}
    grid = {}
    for study in "AB":
        task = load_task_evaluation_config(V3 / f"task_{study}.json")
        grid[study] = {}
        for scenario in task.scenarios:
            seed = derive_seed(SAMPLER_SEED, "ladder-fidelity", study, scenario.scenario_id)
            assert seed not in occupied
            occupied.add(seed)
            grid[study][scenario.scenario_id] = seed
    save(
        OUT / "PLAN.json",
        {
            "purpose": "P1 G-ladder fidelity audit of the frozen v3 ladders; no search.",
            "authorization": "Investigator, 24 September 2026: pre-launch steps 1-6.",
            "v3_freeze_sha256": sha(V3 / "FREEZE.json"),
            "task_sha256": {s: sha(V3 / f"task_{s}.json") for s in "AB"},
            "sampler_seed": SAMPLER_SEED,
            "strata": STRATA,
            "quota_per_stratum": QUOTA,
            "reference_patterns": REFERENCES,
            "pool_sizes_after_dedupe": {k: len(v) for k, v in pools.items()},
            "excluded_v3_initialization_sources": len(excluded),
            "audit_seed_grid": grid,
            "criteria": CRITERIA,
            "corpus": corpus,
            "runner_sha256": sha(__file__),
        },
    )
    print({s: len(c) for s, c in corpus.items()}, {k: len(v) for k, v in pools.items()})


def audit_task(study):
    task = load_task_evaluation_config(V3 / f"task_{study}.json")
    plan = json.loads((OUT / "PLAN.json").read_text())
    grid = plan["audit_seed_grid"][study]
    return replace(
        task,
        scenarios=tuple(
            ScenarioGrid(scenario_id=s.scenario_id, path=s.path, seeds=(grid[s.scenario_id],))
            for s in task.scenarios
        ),
    )


def fitness_record(fitness):
    return {
        "feasible": fitness.feasible,
        "normalized_energy": fitness.normalized_energy,
        "summed_primitive_shortfall": fitness.summed_primitive_shortfall,
        "infeasibility_vector": list(fitness.infeasibility_vector),
        "sort_key": [float(x) for x in fitness.sort_key()[:-1]],
    }


def evaluate_one(study, index, entry):
    source = (ROOT / entry["path"]).read_text()
    if text_sha(source) != entry["sha256"]:
        raise ValueError(f"corpus source changed: {entry['path']}")
    task = audit_task(study)
    ladder = load_ladder_config(V3 / f"task_{study}.json")
    start = time.monotonic()
    row = {"study": study, "index": index, **entry}
    try:
        l0 = _run_l0(source, task)
        row["l0_valid"] = bool(l0["valid"])
        l1 = _run_l1(source, task, ladder, index, retain_episode_ledgers=False)
        row["l1"] = fitness_record(l1.fitness)
        energy_ok = l1.fitness.normalized_energy is None or (
            l1.fitness.normalized_energy <= ladder.l1.max_normalized_energy
        )
        row["l1_promote"] = bool(
            row["l0_valid"]
            and energy_ok
            and l1.fitness.summed_primitive_shortfall <= ladder.l1.max_summed_primitive_shortfall
        )
        l2 = evaluate_policy_source(source, task, proposal_seq=index, retain_episode_ledgers=False)
        row["l2"] = fitness_record(l2.fitness)
    except Exception as exc:  # recorded, never retried silently
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.monotonic() - start
    row["peak_rss_mib"] = psutil.Process().memory_info().rss / 2**20
    return row


def run():
    plan = json.loads((OUT / "PLAN.json").read_text())
    if sha(__file__) != plan["runner_sha256"]:
        raise ValueError("runner changed after the plan was fixed")
    results = OUT / "results.jsonl"
    done = set()
    if results.exists():
        done = {(r["study"], r["index"]) for r in map(json.loads, results.open())}
    jobs = [
        (study, i, entry)
        for study in "AB"
        for i, entry in enumerate(plan["corpus"][study])
        if (study, i) not in done
    ]
    with results.open("a") as out, ProcessPoolExecutor(
        max_workers=WORKERS, max_tasks_per_child=1
    ) as pool:
        pending = set()
        while jobs or pending:
            while jobs and len(pending) < WORKERS:
                if psutil.virtual_memory().available < MIN_AVAILABLE_MIB * 2**20:
                    break
                pending.add(pool.submit(evaluate_one, *jobs.pop(0)))
            finished, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
                print(json.dumps({k: row.get(k) for k in ("study", "index", "stratum",
                      "l1_promote", "seconds")} | {"l2_feasible": row.get("l2", {}).get("feasible"),
                      "error": row.get("error")}), flush=True)
            if psutil.virtual_memory().available < MIN_AVAILABLE_MIB * 2**20 and not pending:
                raise RuntimeError("available memory below floor; stopping before new work")


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
