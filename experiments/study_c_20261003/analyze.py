"""Registered analysis of Study C (PROTOCOL.md): one primary test, LLM vs GP champion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.prospective_20260917 import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = ROOT / "prereg/STUDY_C_ROBUST_20261003"
RUNS = HERE / "runs"
ALPHA = 0.05


def label_record(campaign_id, label):
    final = RUNS / campaign_id / "final"
    labels = json.loads((final / "LABELS.json").read_text())["labels"]
    sha = next((s for s, names in labels.items() if label in names), None)
    if sha is None:
        return {"status": "missing", "feasible": None, "energy": None, "source_sha256": None}
    record = json.loads((final / f"{sha}.json").read_text())
    if not record["valid"]:
        return {"status": "evaluated", "feasible": False, "energy": None, "source_sha256": sha}
    trace = record["trace"]
    return {"status": "evaluated", "feasible": bool(trace["feasible"]),
            "energy": trace["normalized_energy"] if trace["feasible"] else None,
            "source_sha256": sha}


def main():
    actual = hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()
    if not actual.startswith("b42bd4fd"):
        raise ValueError("statistics module differs from the registered source")
    seeds = json.loads((REG / "SEEDS.json").read_text())
    rows = seeds["studies"]["C"]
    roots = [r["root_seed"] for r in rows]
    labels = ["llm_champion", "gp_champion", "initialization_best", "reference:latency_greedy",
              "reference:network_overhead", "reference:graph_headroom"]
    outcome = {r["root_seed"]: {lab: label_record(r["campaign_id"], lab) for lab in labels}
               for r in rows}

    def contrast(left, right, seed):
        return stats.paired_report([{"root_seed": root, "left": outcome[root][left],
                                     "right": outcome[root][right]} for root in roots],
                                   roots, bootstrap_seed=seed)

    primary = contrast("llm_champion", "gp_champion", seeds["analysis"]["C"])
    p = primary["primary"]["p_two_sided"]
    primary["primary"]["directional_claim"] = bool(
        p is not None and p <= ALPHA and primary["primary"]["wins"] > primary["primary"]["losses"])
    primary["primary"]["opposite_direction"] = bool(
        p is not None and p <= ALPHA and primary["primary"]["losses"] > primary["primary"]["wins"])
    descriptive = {}
    for left, right in (("llm_champion", "initialization_best"),
                        ("gp_champion", "initialization_best"),
                        ("llm_champion", "reference:latency_greedy"),
                        ("gp_champion", "reference:latency_greedy")):
        r = contrast(left, right, seeds["analysis"]["C"])
        joint = [(outcome[root][left]["energy"], outcome[root][right]["energy"]) for root in roots
                 if outcome[root][left]["feasible"] and outcome[root][right]["feasible"]]
        descriptive[f"{left} vs {right}"] = {
            "wins_losses_ties": [r["primary"][k] for k in ("wins", "losses", "ties")],
            "p_unadjusted": r["primary"]["p_two_sided"],
            "jointly_feasible": len(joint),
            "mean_relative_energy_saving": (sum((b - a) / b for a, b in joint) / len(joint))
            if joint else None,
        }
    feasibility = {lab: sum(outcome[root][lab]["feasible"] is True for root in roots) / len(roots)
                   for lab in labels}
    energy = {}
    for lab in labels:
        values = [outcome[root][lab]["energy"] for root in roots if outcome[root][lab]["feasible"]]
        energy[lab] = sum(values) / len(values) if values else None
    same = sum(outcome[root]["llm_champion"]["source_sha256"] ==
               outcome[root]["gp_champion"]["source_sha256"] for root in roots)
    report = {"statistics_source_sha256": actual, "primary_contrast": "llm_champion vs gp_champion",
              "primary": primary, "descriptive": descriptive,
              "feasibility_frequency": feasibility, "mean_energy_when_feasible": energy,
              "llm_equals_gp_champion": same,
              "outcomes": {str(k): v for k, v in outcome.items()}}
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "PRIMARY.json").open("x") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: primary["primary"][k] for k in (
        "wins", "losses", "ties", "missing", "p_two_sided", "directional_claim",
        "opposite_direction", "conditional_win_probability_ci95")}))
    print("feasibility", {k: round(v, 3) for k, v in feasibility.items()})
    print("energy", {k: (round(v, 3) if v else None) for k, v in energy.items()})
    print("energy (primary)", primary["energy"], "same program", same)
    for k, v in descriptive.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
