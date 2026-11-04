"""Registered analysis of Study D (PROTOCOL.md): two primary tests, Holm-corrected."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.prospective_20260917 import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = ROOT / "prereg/STUDY_D_SINGLESEED_20261004"
RUNS = {"C": ROOT / "experiments/study_c_20261003/runs", "D": HERE / "runs"}
ALPHA = 0.05
LABELS = ["llm_champion", "gp_champion", "initialization_best", "reference:latency_greedy"]
REFERENCES = [f"reference:{n}" for n in (
    "latency_greedy", "energy_greedy", "first_fit", "least_allocated", "most_allocated",
    "kube_least_allocated", "kube_most_allocated", "network_overhead", "graph_headroom")]


def labels_of(study, campaign_id):
    path = RUNS[study] / campaign_id / "final" / "LABELS.json"
    return json.loads(path.read_text())["labels"] if path.exists() else {}


def final_record(study, campaign_id, label):
    """(source sha, record); a missing label or result file gives (sha or None, None)."""
    final = RUNS[study] / campaign_id / "final"
    sha = next((s for s, names in labels_of(study, campaign_id).items() if label in names), None)
    path = final / f"{sha}.json"
    return sha, (json.loads(path.read_text()) if sha and path.exists() else None)


def outcome(study, campaign_id, label):
    sha, record = final_record(study, campaign_id, label)
    if record is None:
        return {"status": "missing", "feasible": None, "energy": None, "source_sha256": None}
    if not record["valid"]:
        return {"status": "evaluated", "feasible": False, "energy": None, "source_sha256": sha}
    trace = record["trace"]
    return {"status": "evaluated", "feasible": bool(trace["feasible"]),
            "energy": trace["normalized_energy"] if trace["feasible"] else None,
            "source_sha256": sha}


def determinism(rows):
    """Every program scored in both studies on the same final grid must give identical traces.

    All nine references must be present and completed in both studies for every root."""
    mismatches, shared = [], 0
    for row in rows:
        c_id, d_id = row["paired_c_campaign_id"], row["campaign_id"]
        for label in REFERENCES:
            for study, cid in (("C", c_id), ("D", d_id)):
                if final_record(study, cid, label)[1] is None:
                    mismatches.append([d_id, f"{label} missing in {study}"])
        c, d = labels_of("C", c_id), labels_of("D", d_id)
        for sha in sorted(set(c) & set(d)):
            rc = final_record("C", c_id, c[sha][0])[1]
            rd = final_record("D", d_id, d[sha][0])[1]
            if rc is None or rd is None:
                continue
            shared += 1
            if rc["valid"] != rd["valid"] or rc.get("trace") != rd.get("trace"):
                mismatches.append([d_id, sha])
    return mismatches, shared


def main():
    actual = hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()
    if not actual.startswith("b42bd4fd"):
        raise ValueError("statistics module differs from the registered source")
    seeds = json.loads((REG / "SEEDS.json").read_text())
    rows = seeds["studies"]["D"]
    roots = [r["root_seed"] for r in rows]
    out = {}
    for r in rows:
        out[r["root_seed"]] = {f"{lab}@C": outcome("C", r["paired_c_campaign_id"], lab)
                               for lab in LABELS}
        out[r["root_seed"]].update({f"{lab}@D": outcome("D", r["campaign_id"], lab)
                                    for lab in LABELS})
    mismatches, shared = determinism(rows)

    def dirs(left, right):
        return [stats.direction(out[k][left], out[k][right]) for k in roots]

    def report(left, right):
        return stats.paired_report([{"root_seed": k, "left": out[k][left], "right": out[k][right]}
                                    for k in roots], roots, bootstrap_seed=seeds["analysis"]["D"])

    p1 = report("llm_champion@C", "llm_champion@D")
    c_adv = dirs("llm_champion@C", "gp_champion@C")
    d_adv = dirs("llm_champion@D", "gp_champion@D")
    interaction = [None if a is None or b is None else (a > b) - (a < b)
                   for a, b in zip(c_adv, d_adv)]
    p2 = stats.sign_summary(interaction)
    missing = p1["primary"]["missing"] + p2["missing"]
    blocked = bool(mismatches) or bool(missing)
    raw = [None if blocked else p1["primary"]["p_two_sided"], None if blocked else p2["p_two_sided"]]
    adjusted = stats.holm_two(raw)
    primary = {}
    for name, summary, adj in (("P1 llm_C vs llm_D", p1["primary"], adjusted[0]),
                               ("P2 interaction", p2, adjusted[1])):
        primary[name] = {**summary, "p_holm": adj, "directional_claim": bool(
            adj is not None and adj <= ALPHA and summary["wins"] > summary["losses"]),
            "opposite_direction": bool(
            adj is not None and adj <= ALPHA and summary["losses"] > summary["wins"])}
    primary["P1 llm_C vs llm_D"]["energy"] = p1["energy"]
    primary["P2 interaction"]["advantage_C_wlt"] = [c_adv.count(v) for v in (1, -1, 0)]
    primary["P2 interaction"]["advantage_D_wlt"] = [d_adv.count(v) for v in (1, -1, 0)]

    descriptive = {}
    for left, right in (("llm_champion@D", "gp_champion@D"), ("gp_champion@C", "gp_champion@D"),
                        ("llm_champion@D", "initialization_best@D"),
                        ("gp_champion@D", "initialization_best@D"),
                        ("llm_champion@D", "reference:latency_greedy@D"),
                        ("gp_champion@D", "reference:latency_greedy@D")):
        r = report(left, right)
        joint = [(out[k][left]["energy"], out[k][right]["energy"]) for k in roots
                 if out[k][left]["feasible"] and out[k][right]["feasible"]]
        descriptive[f"{left} vs {right}"] = {
            "wins_losses_ties": [r["primary"][k] for k in ("wins", "losses", "ties")],
            "p_unadjusted": r["primary"]["p_two_sided"], "jointly_feasible": len(joint),
            "mean_relative_energy_saving": (sum((b - a) / b for a, b in joint) / len(joint))
            if joint else None, "energy": r["energy"]}
    keys = [f"{lab}@{s}" for s in "CD" for lab in LABELS]
    feasibility = {k: sum(out[r][k]["feasible"] is True for r in roots) / len(roots) for k in keys}
    energy = {}
    for k in keys:
        values = [out[r][k]["energy"] for r in roots if out[r][k]["feasible"]]
        energy[k] = sum(values) / len(values) if values else None
    result = {"statistics_source_sha256": actual, "determinism_mismatches": mismatches,
              "determinism_shared_programs": shared,
              "confirmatory_blocked": blocked, "primary": primary, "descriptive": descriptive,
              "feasibility_frequency": feasibility, "mean_energy_when_feasible": energy,
              "outcomes": {str(k): v for k, v in out.items()}}
    (HERE / "analysis").mkdir(exist_ok=True)
    with (HERE / "analysis" / "PRIMARY.json").open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"mismatches": mismatches, "shared": shared, "primary": {
        k: {f: v[f] for f in ("wins", "losses", "ties", "missing", "p_two_sided", "p_holm",
                              "directional_claim", "opposite_direction")}
        for k, v in primary.items()}}, indent=1))
    print("feasibility", {k: round(v, 3) for k, v in feasibility.items()})
    print("energy", {k: (round(v, 3) if v else None) for k, v in energy.items()})
    for k, v in descriptive.items():
        print(" ", k, {f: v[f] for f in ("wins_losses_ties", "p_unadjusted", "jointly_feasible",
                                         "mean_relative_energy_saving")})


if __name__ == "__main__":
    main()
