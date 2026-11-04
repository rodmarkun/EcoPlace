"""Build and freeze the Study E registration (multi-family confirmatory study).

Study E compares the LLM and GP proposal operators under the multi-seed protocol of Study C
(`evaluation_replicates: 4`) on three task families, 30 new paired roots each:

* T (`dev-continuum-tight`): the historical task of studies A, C and D;
* W (`dev-continuum-wide`): a 17-node cluster with a different topology and site structure;
* R (`dev-continuum-drift`): the historical cluster with three nodes cordoned mid-episode and a
  sustained demand step (operational drift).

Every root starts from Study A/C's archive procedure (seven references plus five mutants).
Ladder, islands, feedback, GP and LLM settings are inherited unchanged from Study C. Run `uv run python -m experiments.study_e_20261005.build prepare` then `freeze`.
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
C = ROOT / "prereg/STUDY_C_ROBUST_20261003"
D = ROOT / "prereg/STUDY_D_SINGLESEED_20261004"
OUT = ROOT / "prereg/STUDY_E_MULTIFAMILY_20261005"
ROOTS = 30
SEARCH_REPLICATES = 4
FINAL_SEEDS = 8
FAMILIES = {
    "T": ("dev-continuum-tight", "sim/scenarios/dev/continuum_tight.json"),
    "W": ("dev-continuum-wide", "sim/scenarios/dev/continuum_wide.json"),
    "R": ("dev-continuum-drift", "sim/scenarios/dev/continuum_drift.json"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def occupied_seeds():
    from experiments.study_c_20261003.build import occupied_seeds as through_b

    seeds = through_b()
    for reg, study in ((C, "C"),):
        for row in json.loads((reg / "SEEDS.json").read_text())["enumerated_seed_records"]:
            seeds.add(row["seed"])
    seeds.add(json.loads((D / "SEEDS.json").read_text())["analysis"]["D"])
    for family in ("tight", "wide", "drift"):
        for j in range(4):  # calibration seeds (calibrate.py)
            seeds.add(derive_seed(20261005, "study-e-calibration", family, j))
    return seeds


def archive(root):
    """Study A/C's archive procedure, unchanged: seven references plus five mutants.

    Its `least_allocated`/`most_allocated` entries are the kube_* policy files, so each family's
    calibration-feasible reference (latency_greedy, kube_least_allocated, graph_headroom) is in
    every starting archive."""
    from experiments.prospective_20260917 import build

    return build.archive(root, "A")


def family_task(letter):
    sid, path = FAMILIES[letter]
    task = json.loads((C / "task_C.json").read_text())
    task["scenarios"] = [{"id": sid, "path": path, "seeds": task["scenarios"][0]["seeds"]}]
    task["ladder"]["l1"]["scenario_ids"] = [sid]
    task["task_variant"] = sid
    task["task_variant_note"] = (f"Study E family {letter}: Study C task with scenario {sid}; "
                                 "scenario seeds are campaign-derived at run time.")
    return task


def prepare():
    from evolution.campaign import load_campaign_plan
    from evolution.evaluation import load_task_evaluation_config
    from experiments.allbaselines_harness.initialization import read_seed_archive
    from experiments.prospective_20260917 import build
    from experiments.validation_20260909.run_independent_calibration import store_once

    if OUT.exists():
        raise ValueError("new registration directory required")
    occupied = occupied_seeds()
    records = []

    def claim(seed, role):
        if seed in occupied:
            raise ValueError(f"seed collision for {role}")
        occupied.add(seed)
        records.append({"role": role, "seed": seed})
        return seed

    OUT.mkdir(parents=True)
    studies, analysis = {}, {}
    drift_sid = FAMILIES["R"][0]
    for letter, (sid, _path) in FAMILIES.items():
        rows = []
        for i in range(ROOTS):
            cid = f"{letter}{i:02d}"
            root = claim(derive_seed(20261005, "study-e", letter, i), f"E/{letter}/root/{i}")
            search = [claim(derive_seed(root, "sim", "scenario", sid, "replicate", j),
                            f"E/{letter}/search/{i}/{j}") for j in range(SEARCH_REPLICATES)]
            final = [claim(derive_seed(root, "study-e", "final", sid, j),
                           f"E/{letter}/final/{i}/{j}") for j in range(FINAL_SEEDS)]
            grids = {sid: {"search": search, "final": final}}
            row = {"campaign_id": cid, "root_seed": root, "grids": grids,
                   "evaluation_replicates": SEARCH_REPLICATES,
                   "first_arm": "llm" if i % 2 == 0 else "gp",
                   "zeroshot_seed": claim(derive_seed(root, "study-e", "zeroshot"),
                                          f"E/{letter}/zeroshot/{i}")}
            if letter == "T":  # descriptive transfer of T champions to operational drift
                row["transfer_final"] = {drift_sid: [
                    claim(derive_seed(root, "study-e", "transfer", drift_sid, j),
                          f"E/T/transfer/{i}/{j}") for j in range(FINAL_SEEDS)]}
            rows.append(row)
        studies[letter] = rows
        analysis[letter] = claim(derive_seed(20261005, "study-e", "analysis", letter),
                                 f"E/{letter}/analysis")
        plan = json.loads((C / "campaign_C.json").read_text())
        plan["campaigns"] = [{"campaign_id": r["campaign_id"], "root_seed": r["root_seed"],
                              "evaluation_replicates": SEARCH_REPLICATES} for r in rows]
        write(OUT / f"campaign_{letter}.json", plan)
        load_campaign_plan(OUT / f"campaign_{letter}.json")
        write(OUT / f"task_{letter}.json", family_task(letter))
        load_task_evaluation_config(OUT / f"task_{letter}.json")
        for r in rows:
            path = OUT / "initialization" / f"{r['campaign_id']}.json"
            path.parent.mkdir(exist_ok=True)
            store_once(path, archive(r["root_seed"]))
            read_seed_archive(path, expected_size=12, initial_path=build.INITIAL)
    pooled = claim(derive_seed(20261005, "study-e", "analysis", "pooled"), "E/pooled/analysis")
    analysis["pooled"] = pooled
    write(OUT / "SEEDS.json", {"studies": studies, "analysis": analysis,
                               "enumerated_seed_records": records})
    for name in ("gp.json", "engine.json"):
        shutil.copyfile(C / name, OUT / name)
    write(OUT / "INHERITANCE.json", {
        "study_c_freeze_sha256": sha(C / "FREEZE.json"),
        "copied_unchanged": {"gp.json": sha(C / "gp.json"), "engine.json": sha(C / "engine.json")},
        "campaign_change": "campaign_C.json with new campaign definitions per family; "
                           "evaluation_replicates 4 as in Study C; all other fields unchanged",
        "task_change": "task_C.json with the scenario (and L1 scenario id) replaced per family",
        "archive": "Study A/C procedure unchanged (build.archive(root, 'A'))",
        "calibration_sha256": {p.name: sha(p) for p in sorted((HERE / "calibration").glob("*.json"))},
    })
    shutil.copyfile(HERE / "PROTOCOL.md", OUT / "PROTOCOL.md")
    print(f"Prepared {sum(len(v) for v in studies.values())} roots, {len(records)} seeds.")


def freeze():
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    if (OUT / "FREEZE.json").exists():
        raise ValueError("already frozen")
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    payload = {"schema": "study-e-design-freeze-v1",
               "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "files_sha256": files, "environment_digest": digest(evaluation_snapshot()),
               "builder_sha256": sha(__file__), "analysis_sha256": sha(HERE / "analyze.py"),
               "calibration_script_sha256": sha(HERE / "calibrate.py")}
    write(OUT / "FREEZE.json", {"payload": payload, "payload_sha256": digest(payload)})
    print(f"Frozen {len(files)} files.")


if __name__ == "__main__":
    {"prepare": prepare, "freeze": freeze}[sys.argv[1]]()
