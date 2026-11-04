"""Build and freeze the Study D registration (within-root single-seed counterpart of Study C).

Study D reruns Study C's 30 paired roots with Study C's single registered change reversed:
every full (L2) evaluation during search uses one arrival seed per scenario
(`evaluation_replicates: 1`, the study-A setting) instead of four. Root seeds, initialization
archives, final seeds and every other input are byte-identical copies of the frozen Study C
registration, so C versus D is a within-root test of the search objective. Run:
`uv run python -m experiments.study_d_20261004.build prepare` then `... freeze`.
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
OUT = ROOT / "prereg/STUDY_D_SINGLESEED_20261004"
SEARCH_REPLICATES = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare():
    from evolution.campaign import load_campaign_plan
    from experiments.allbaselines_harness.initialization import read_seed_archive
    from experiments.prospective_20260917 import build
    from experiments.gp_tuning_20260902.recovery import read_sealed

    if OUT.exists():
        raise ValueError("new registration directory required")
    read_sealed(C / "FREEZE.json")
    seeds_c = json.loads((C / "SEEDS.json").read_text())
    rows = []
    for row in seeds_c["studies"]["C"]:
        i = int(row["campaign_id"][1:])
        grids = {}
        for sid, grid in row["grids"].items():
            search = [derive_seed(row["root_seed"], "sim", "scenario", sid, "replicate", 0)]
            if search != grid["search"][:1]:
                raise ValueError("search seed derivation differs from Study C")
            grids[sid] = {"search": search, "final": list(grid["final"])}
        rows.append({"campaign_id": f"D{i:02d}", "paired_c_campaign_id": row["campaign_id"],
                     "root_seed": row["root_seed"], "grids": grids,
                     "evaluation_replicates": SEARCH_REPLICATES, "first_arm": row["first_arm"]})
    analysis_seed = derive_seed(20261004, "study-d", "analysis")
    used = {r["seed"] for r in seeds_c["enumerated_seed_records"]}
    if analysis_seed in used:
        raise ValueError("analysis seed collision")
    OUT.mkdir(parents=True)
    write(OUT / "SEEDS.json", {"studies": {"D": rows}, "analysis": {"D": analysis_seed},
                               "reused_from_study_c": "root, search replicate 0 and final seeds"})
    plan = json.loads((C / "campaign_C.json").read_text())
    plan["campaigns"] = [{"campaign_id": r["campaign_id"], "root_seed": r["root_seed"],
                          "evaluation_replicates": SEARCH_REPLICATES} for r in rows]
    write(OUT / "campaign_D.json", plan)
    load_campaign_plan(OUT / "campaign_D.json")
    shutil.copyfile(C / "task_C.json", OUT / "task_D.json")
    for name in ("gp.json", "engine.json"):
        shutil.copyfile(C / name, OUT / name)
    for r in rows:
        source = C / "initialization" / f"{r['paired_c_campaign_id']}.json"
        target = OUT / "initialization" / f"{r['campaign_id']}.json"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        read_seed_archive(target, expected_size=12, initial_path=build.INITIAL)
    write(OUT / "INHERITANCE.json", {
        "study_c_freeze_sha256": sha(C / "FREEZE.json"),
        "copied_unchanged": {
            "task_D.json": sha(C / "task_C.json"), "gp.json": sha(C / "gp.json"),
            "engine.json": sha(C / "engine.json"),
            **{f"initialization/{r['campaign_id']}.json":
               sha(C / "initialization" / f"{r['paired_c_campaign_id']}.json") for r in rows}},
        "campaign_change": "campaign_C.json with campaign ids C->D; evaluation_replicates 4 -> 1; "
                           "root seeds and all other fields unchanged",
    })
    shutil.copyfile(HERE / "PROTOCOL.md", OUT / "PROTOCOL.md")
    print(f"Prepared {len(rows)} roots; registration not yet frozen.")


def freeze():
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    if (OUT / "FREEZE.json").exists():
        raise ValueError("already frozen")
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    payload = {"schema": "study-d-design-freeze-v1",
               "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "files_sha256": files, "environment_digest": digest(evaluation_snapshot()),
               "builder_sha256": sha(__file__), "analysis_sha256": sha(HERE / "analyze.py")}
    write(OUT / "FREEZE.json", {"payload": payload, "payload_sha256": digest(payload)})
    print(f"Frozen {len(files)} files.")


if __name__ == "__main__":
    {"prepare": prepare, "freeze": freeze}[sys.argv[1]]()
