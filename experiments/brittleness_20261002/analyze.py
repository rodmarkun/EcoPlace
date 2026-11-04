"""Summarize the EXPLORATORY brittleness and sensitivity analyses (not confirmatory)."""

from __future__ import annotations

import hashlib
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"
STUDY = HERE.parents[0] / "corrected_ab_20260925"
PRIMARY = json.loads((STUDY / "analysis/PRIMARY.json").read_text())


def result_path(job_id):
    return OUT / "results" / (hashlib.sha256(job_id.encode()).hexdigest() + ".json")


def final_outcome(cid, sha):
    """Final-grid outcome of a source: the registered record, else the exploratory test."""
    registered = STUDY / "runs" / cid / "final" / f"{sha}.json"
    if registered.exists():
        record = json.loads(registered.read_text())
        if not record["valid"]:
            return {"feasible": False, "energy": None}
        trace = record["trace"]
        return {"feasible": trace["feasible"],
                "energy": trace["normalized_energy"] if trace["feasible"] else None}
    path = result_path(f"final_test:{cid}:{sha}")
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return {"feasible": record["feasible"],
            "energy": record["normalized_energy"] if record["feasible"] else None}


def summarize(rows):
    feasible = [r for r in rows if r and r["feasible"]]
    energies = [r["energy"] for r in feasible]
    return {"n": len(rows), "missing": sum(r is None for r in rows),
            "feasible_fraction": round(len(feasible) / len(rows), 3) if rows else None,
            "mean_energy_when_feasible": round(S.mean(energies), 4) if energies else None}


def direction(a, b):
    """+1 if a better than b under feasibility-first ordering (energy tie 1e-9)."""
    if a is None or b is None:
        return None
    if a["feasible"] != b["feasible"]:
        return 1 if a["feasible"] else -1
    if not a["feasible"]:
        return 0
    delta = b["energy"] - a["energy"]
    return 0 if abs(delta) <= 1e-9 else (1 if delta > 0 else -1)


def brittleness():
    report = {}
    for study in ("A", "B"):
        path = OUT / f"VALIDATED_SELECTIONS_{study}.json"
        if not path.exists():
            continue
        selections = json.loads(path.read_text())["selections"]
        outcomes = PRIMARY["studies"][study]["outcomes"]
        seeds = json.loads((STUDY.parents[1] / "prereg/CORRECTED_STUDIES_20260920_v3/"
                            "SEEDS.json").read_text())["studies"][study]
        table = {}
        for arm in ("llm", "gp"):
            registered, validated, changed, dirs = [], [], 0, []
            for row in seeds:
                cid, rec = row["campaign_id"], outcomes[str(row["root_seed"])]
                reg = {"feasible": rec[f"{arm}_champion"]["feasible"],
                       "energy": rec[f"{arm}_champion"]["energy"]}
                sel = selections[cid][arm]
                val = final_outcome(cid, sel["source_sha256"]) if sel else None
                changed += bool(sel and sel["source_sha256"] != rec[f"{arm}_champion"]
                                ["source_sha256"])
                registered.append(reg)
                validated.append(val)
                dirs.append(direction(val, reg))
            table[arm] = {"registered_champion": summarize(registered),
                          "validated_champion": summarize(validated),
                          "selections_changed": changed,
                          "validated_vs_registered_wins_losses_ties":
                              [dirs.count(1), dirs.count(-1), dirs.count(0)]}
        ref = [{"feasible": outcomes[str(r["root_seed"])]["reference:latency_greedy"]["feasible"],
                "energy": outcomes[str(r["root_seed"])]["reference:latency_greedy"]["energy"]}
               for r in seeds]
        table["latency_greedy"] = summarize(ref)
        for arm in ("llm", "gp"):
            for which, rows in (("registered", None), ("validated", None)):
                vals = []
                for row, r in zip(seeds, ref, strict=True):
                    rec = outcomes[str(row["root_seed"])]
                    if which == "registered":
                        v = {"feasible": rec[f"{arm}_champion"]["feasible"],
                             "energy": rec[f"{arm}_champion"]["energy"]}
                    else:
                        sel = selections[row["campaign_id"]][arm]
                        v = final_outcome(row["campaign_id"], sel["source_sha256"]) if sel else None
                    vals.append((v, r))
                joint = [(v["energy"], r["energy"]) for v, r in vals
                         if v and v["feasible"] and r["feasible"]]
                dirs = [direction(v, r) for v, r in vals]
                table[arm][f"{which}_vs_latency_greedy"] = {
                    "wins_losses_ties": [dirs.count(1), dirs.count(-1), dirs.count(0)],
                    "jointly_feasible": len(joint),
                    "mean_relative_energy_saving": round(S.mean((r - v) / r for v, r in joint), 4)
                    if joint else None}
        report[study] = table
    return report


def sensitivity():
    """Per root, look up each label's source result by (root, source, cell)."""
    plan = json.loads((OUT / "PLAN.json").read_text())
    cells_spec = [(shape, fraction) for shape, fraction in plan["sensitivity_cells"]]
    outcomes = PRIMARY["studies"]["A"]["outcomes"]
    seeds = json.loads((STUDY.parents[1] / "prereg/CORRECTED_STUDIES_20260920_v3/SEEDS.json")
                       .read_text())["studies"]["A"]
    out = {}
    for shape, fraction in cells_spec:
        cell = f"{shape}@{fraction:.2f}"
        per_label = {label: [] for label in plan["sensitivity_labels"]}
        for row in seeds:
            rec = outcomes[str(row["root_seed"])]
            for label in plan["sensitivity_labels"]:
                sha = rec[label]["source_sha256"]
                path = result_path(f"sensitivity:{row['campaign_id']}:{sha}:{shape}:{fraction:.2f}")
                if not path.exists():
                    per_label[label].append(None)
                    continue
                r = json.loads(path.read_text())
                per_label[label].append({"feasible": r["feasible"],
                                         "energy": r["normalized_energy"] if r["feasible"]
                                         else None})
        out[cell] = {label: summarize(rows) for label, rows in per_label.items()}
        for arm in ("llm_champion", "gp_champion"):
            joint = [(a["energy"], b["energy"]) for a, b in
                     zip(per_label[arm], per_label["reference:latency_greedy"], strict=True)
                     if a and b and a["feasible"] and b["feasible"]]
            out[cell][f"{arm}_vs_latency_greedy"] = {
                "jointly_feasible": len(joint),
                "mean_relative_saving": round(S.mean((b - a) / b for a, b in joint), 4)
                if joint else None,
                "directions_wlt": [sum(1 for d in (direction(a, b) for a, b in zip(
                    per_label[arm], per_label["reference:latency_greedy"], strict=True))
                    if d == v) for v in (1, -1, 0)]}
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    result = {}
    if which in ("all", "sensitivity"):
        result["sensitivity"] = sensitivity()
    if which in ("all", "brittleness"):
        result["brittleness"] = brittleness()
    print(json.dumps(result, indent=1))
    if which == "all":
        (HERE / "SUMMARY.json").write_text(json.dumps(result, indent=1) + "\n")
