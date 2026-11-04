"""Registered analysis of the corrected A+B study (v1 Endpoints and inference, v3 ANALYSIS.json).

Primary family: A = LLM champion vs GP champion; B = LLM champion vs common initialization
best. Directions, sign tests, Holm correction and intervals come unchanged from the frozen
experiments/prospective_20260917/statistics.py (sha256 pinned in ANALYSIS.json). Everything
else is descriptive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.prospective_20260917 import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3 = ROOT / "prereg/CORRECTED_STUDIES_20260920_v3"
RUNS = HERE / "runs"
OUT = HERE / "analysis"
ALPHA = 0.05


def label_record(campaign_id, label):
    """Final-grid outcome for one label in the protocol's status vocabulary."""
    final = RUNS / campaign_id / "final"
    labels = json.loads((final / "LABELS.json").read_text())["labels"]
    sha = next((s for s, names in labels.items() if label in names), None)
    if sha is None:
        return {"status": "missing", "feasible": None, "energy": None, "source_sha256": None}
    record = json.loads((final / f"{sha}.json").read_text())
    if not record["valid"]:
        # The selected program faulted on the final grid: a completed, failed placement.
        return {"status": "evaluated", "feasible": False, "energy": None,
                "source_sha256": sha, "fault": record["error"][:200]}
    trace = record["trace"]
    return {"status": "evaluated", "feasible": bool(trace["feasible"]),
            "energy": trace["normalized_energy"] if trace["feasible"] else None,
            "source_sha256": sha,
            "shortfall": trace["summed_primitive_shortfall"]}


def contrast(study, rows_by_root, left, right, roots, seed):
    rows = [{"root_seed": root, "left": rows_by_root[root][left],
             "right": rows_by_root[root][right]} for root in roots]
    return stats.paired_report(rows, roots, bootstrap_seed=seed)


def execution_summary(study, rows):
    out = []
    for row in rows:
        selection = json.loads((RUNS / row["campaign_id"] / "SELECTION.json").read_text())
        arms = {}
        for arm in ("gp", "llm"):
            info = selection["arms"][arm]
            terminal = info["terminal"] or {}
            arms[arm] = {"stage_exit": info["stage_exit"], "terminal": terminal.get("status"),
                         "reason": terminal.get("reason"),
                         "completed_candidates": info["completed_candidates"],
                         "full_records": info["full_records"]}
        out.append({"root": row["campaign_id"], "arms": arms,
                    "superseded": (RUNS / row["campaign_id"] /
                                   "SELECTION_superseded_amendment02.json").exists()})
    return out


def main():
    expected = json.loads((V3 / "ANALYSIS.json").read_text())["statistics_source_sha256"]
    actual = hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("statistics module differs from the registered source")
    seeds = json.loads((V3 / "SEEDS.json").read_text())
    references = [p.stem for p in (ROOT / "runtime/policies").glob("*.ps")]
    report = {"statistics_source_sha256": actual, "studies": {}}
    pvalues = []
    for study in ("A", "B"):
        rows = seeds["studies"][study]
        roots = [row["root_seed"] for row in rows]
        outcome = {}
        for row in rows:
            labels = ["llm_champion", "gp_champion", "initialization_best",
                      *[f"reference:{r}" for r in references]]
            if study == "B":
                labels.append("reference:cheapest_tier_bottleneck_headroom")
            outcome[row["root_seed"]] = {lab: label_record(row["campaign_id"], lab)
                                         for lab in labels}
        primary_right = "gp_champion" if study == "A" else "initialization_best"
        primary = contrast(study, outcome, "llm_champion", primary_right, roots,
                           seeds["analysis"][study])
        pvalues.append(primary["primary"]["p_two_sided"])
        descriptive = {}
        pairs = [("llm_champion", "initialization_best"), ("gp_champion", "initialization_best")]
        if study == "B":
            pairs.insert(0, ("llm_champion", "gp_champion"))
        for label in sorted(next(iter(outcome.values()))):
            if label.startswith("reference:"):
                pairs.append(("llm_champion", label))
                pairs.append(("gp_champion", label))
        for left, right in pairs:
            if (left, right) == ("llm_champion", primary_right):
                continue
            r = contrast(study, outcome, left, right, roots, seeds["analysis"]["descriptive"])
            descriptive[f"{left} vs {right}"] = {
                "wins_losses_ties": [r["primary"][k] for k in ("wins", "losses", "ties")],
                "p_unadjusted": r["primary"]["p_two_sided"],
                "feasible_left_right": [r["feasibility"]["left"], r["feasibility"]["right"]],
                "jointly_feasible": r["energy"]["jointly_feasible"],
                "median_energy_difference": r["energy"]["median"],
            }
        feasible_counts = {lab: sum(outcome[root][lab]["feasible"] is True for root in roots)
                           for lab in next(iter(outcome.values()))}
        faults = {lab: sum("fault" in outcome[root][lab] for root in roots)
                  for lab in next(iter(outcome.values()))}
        practical = primary["practical_directions"]
        report["studies"][study] = {
            "primary_contrast": f"llm_champion vs {primary_right}",
            "primary": primary,
            "practical_energy_threshold_directions": {
                "wins": practical.count(1), "losses": practical.count(-1),
                "ties": practical.count(0)},
            "feasibility_frequency": {lab: n / len(roots) for lab, n in feasible_counts.items()},
            "final_faults": {k: v for k, v in faults.items() if v},
            "descriptive": descriptive,
            "execution": execution_summary(study, rows),
            "outcomes": {str(k): v for k, v in outcome.items()},
        }
    adjusted = stats.holm_two(pvalues)
    for study, p in zip(("A", "B"), adjusted, strict=True):
        primary = report["studies"][study]["primary"]["primary"]
        primary["p_holm"] = p
        primary["directional_claim"] = bool(
            p is not None and p <= ALPHA and primary["wins"] > primary["losses"])
        primary["opposite_direction"] = bool(
            p is not None and p <= ALPHA and primary["losses"] > primary["wins"])
    OUT.mkdir(exist_ok=True)
    path = OUT / "PRIMARY.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
        stream.write("\n")
    for study in ("A", "B"):
        s = report["studies"][study]
        p = s["primary"]
        print(study, s["primary_contrast"], {k: p["primary"][k] for k in (
            "wins", "losses", "ties", "missing", "p_two_sided", "p_holm", "directional_claim",
            "opposite_direction", "conditional_win_probability_ci95")})
        print("  feasibility left/right", p["feasibility"], "energy", p["energy"])
        print("  practical (0.01)", s["practical_energy_threshold_directions"],
              "faults", s["final_faults"])


if __name__ == "__main__":
    main()
