"""Build and freeze the Study C registration (robust-objective follow-up to studies A and B).

Study C repeats study A's matched LLM-versus-GP comparison on 30 fresh paired roots with one
registered change: every full (L2) evaluation during search uses four arrival seeds per
scenario instead of one (`evaluation_replicates: 4`). Everything else is inherited from the
frozen v3 study-A inputs. Run: `uv run python -m experiments.study_c_20261003.build prepare`
then `... freeze`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

from analysis.tracking import derive_seed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
V3 = ROOT / "prereg/CORRECTED_STUDIES_20260920_v3"
OUT = ROOT / "prereg/STUDY_C_ROBUST_20261003"
ROOTS = 30
SEARCH_REPLICATES = 4
FINAL_SEEDS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def occupied_seeds():
    from experiments.brittleness_20261002.run import occupied_seeds as base

    seeds = base()
    plan = json.loads((ROOT / "experiments/brittleness_20261002/artifacts/PLAN.json").read_text())
    for info in plan["candidates"].values():
        for values in info["validation_seeds"].values():
            seeds |= set(values)
    for row in json.loads((V3 / "SEEDS.json").read_text())["enumerated_seed_records"]:
        seeds.add(row["seed"])
    return seeds


def prepare():
    from evolution.campaign import load_campaign_plan
    from evolution.evaluation import load_task_evaluation_config
    from experiments.allbaselines_harness.initialization import read_seed_archive
    from experiments.prospective_20260917 import build
    from experiments.validation_20260909.run_independent_calibration import store_once

    if OUT.exists():
        raise ValueError("new registration directory required")
    occupied = occupied_seeds()
    task = load_task_evaluation_config(V3 / "task_A.json")
    scenarios = [s.scenario_id for s in task.scenarios]
    rows, records = [], []

    def claim(seed, role):
        if seed in occupied:
            raise ValueError(f"seed collision for {role}")
        occupied.add(seed)
        records.append({"role": role, "seed": seed})
        return seed

    for i in range(ROOTS):
        cid = f"C{i:02d}"
        root = claim(derive_seed(20261003, "study-c-robust-objective", i), f"C/root/{i}")
        grids = {}
        for sid in scenarios:
            search = [claim(derive_seed(root, "sim", "scenario", sid, "replicate", j),
                            f"C/search/{i}/{sid}/{j}") for j in range(SEARCH_REPLICATES)]
            final = [claim(derive_seed(root, "study-c", "final", sid, j), f"C/final/{i}/{sid}/{j}")
                     for j in range(FINAL_SEEDS)]
            grids[sid] = {"search": search, "final": final}
        rows.append({"campaign_id": cid, "root_seed": root, "grids": grids,
                     "evaluation_replicates": SEARCH_REPLICATES,
                     "first_arm": "llm" if i % 2 == 0 else "gp"})
    analysis_seed = claim(derive_seed(20261003, "study-c", "analysis"), "C/analysis")
    OUT.mkdir(parents=True)
    write(OUT / "SEEDS.json", {"studies": {"C": rows}, "analysis": {"C": analysis_seed},
                               "enumerated_seed_records": records})
    plan = json.loads((V3 / "campaign_A.json").read_text())
    plan["campaigns"] = [{"campaign_id": r["campaign_id"], "root_seed": r["root_seed"],
                          "evaluation_replicates": SEARCH_REPLICATES} for r in rows]
    write(OUT / "campaign_C.json", plan)
    load_campaign_plan(OUT / "campaign_C.json")
    for name in ("task_A.json", "gp.json", "engine.json"):
        shutil.copyfile(V3 / name, OUT / name.replace("task_A", "task_C"))
    for r in rows:
        path = OUT / "initialization" / f"{r['campaign_id']}.json"
        path.parent.mkdir(exist_ok=True)
        store_once(path, build.archive(r["root_seed"], "A"))
        read_seed_archive(path, expected_size=12, initial_path=build.INITIAL)
    write(OUT / "INHERITANCE.json", {
        "v3_freeze_sha256": sha(V3 / "FREEZE.json"),
        "copied_unchanged": {"task_C.json": sha(V3 / "task_A.json"), "gp.json": sha(V3 / "gp.json"),
                             "engine.json": sha(V3 / "engine.json")},
        "campaign_change": "campaign_A.json with 30 new campaign definitions; "
                           "evaluation_replicates 1 -> 4; all other fields unchanged",
    })
    shutil.copyfile(HERE / "PROTOCOL.md", OUT / "PROTOCOL.md")
    print(f"Prepared {ROOTS} roots, {len(records)} seeds; registration not yet frozen.")


def freeze():
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    if (OUT / "FREEZE.json").exists():
        raise ValueError("already frozen")
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    payload = {"schema": "study-c-design-freeze-v1",
               "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "files_sha256": files, "environment_digest": digest(evaluation_snapshot()),
               "builder_sha256": sha(__file__)}
    write(OUT / "FREEZE.json", {"payload": payload, "payload_sha256": digest(payload)})
    print(f"Frozen {len(files)} files.")


if __name__ == "__main__":
    {"prepare": prepare, "freeze": freeze}[sys.argv[1]]()
