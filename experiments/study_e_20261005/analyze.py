"""Registered analysis of Study E (PROTOCOL.md).

Primary P: LLM vs GP champion pooled over 90 roots (exact sign test, alpha 0.05).
Key secondaries, gatekept on P: K1 LLM vs oracle-best reference, K2 the LLM arm's champion
after the same number of proposal slots as the zero-shot arm used (llm_champion_atN) vs the
zero-shot champion (Holm over two). Everything else, including full LLM evolution vs zero-shot,
is descriptive.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.prospective_20260917 import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = ROOT / "prereg/STUDY_E_MULTIFAMILY_20261005"
RUNS = HERE / "runs"
ALPHA = 0.05
FAMILIES = ("T", "W", "R")
REFERENCES = [f"reference:{n}" for n in (
    "latency_greedy", "energy_greedy", "first_fit", "least_allocated", "most_allocated",
    "kube_least_allocated", "kube_most_allocated", "network_overhead", "graph_headroom")]
LABELS = ["llm_champion", "gp_champion", "initialization_best", "zeroshot_champion",
          "llm_champion_atN", *REFERENCES]
MISSING = {"status": "missing", "feasible": None, "energy": None, "source_sha256": None}


def outcome(campaign_id, label, grid="final"):
    final = RUNS / campaign_id / grid
    path = final / "LABELS.json"
    if not path.exists():
        return dict(MISSING)
    labels = json.loads(path.read_text())["labels"]
    sha = next((s for s, names in labels.items() if label in names), None)
    if sha is None or not (final / f"{sha}.json").exists():
        return dict(MISSING)
    record = json.loads((final / f"{sha}.json").read_text())
    if not record["valid"]:
        return {"status": "evaluated", "feasible": False, "energy": None, "source_sha256": sha}
    trace = record["trace"]
    return {"status": "evaluated", "feasible": bool(trace["feasible"]),
            "energy": trace["normalized_energy"] if trace["feasible"] else None,
            "source_sha256": sha}


def oracle(rows):
    """Best reference on this root's final grid: feasible first, then lowest energy."""
    if any(r["status"] == "missing" for r in rows):
        return dict(MISSING)
    feasible = [r for r in rows if r["feasible"]]
    if feasible:
        return dict(min(feasible, key=lambda r: r["energy"]))
    return dict(rows[0])  # all infeasible: any reference is an infeasible outcome


def saving(pairs):
    joint = [(a["energy"], b["energy"]) for a, b in pairs if a["feasible"] and b["feasible"]]
    return (sum((b - a) / b for a, b in joint) / len(joint) if joint else None), len(joint)


def main():
    actual = hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()
    if not actual.startswith("b42bd4fd"):
        raise ValueError("statistics module differs from the registered source")
    seeds = json.loads((REG / "SEEDS.json").read_text())
    out, family_of = {}, {}
    for fam in FAMILIES:
        for row in seeds["studies"][fam]:
            cid = row["campaign_id"]
            labels = {lab: outcome(cid, lab) for lab in LABELS}
            labels["oracle_reference"] = oracle([labels[r] for r in REFERENCES])
            if fam == "T":
                for lab in LABELS:
                    labels[f"{lab}@drift"] = outcome(cid, lab, "final_transfer_drift")
                labels["oracle_reference@drift"] = oracle(
                    [labels[f"{r}@drift"] for r in REFERENCES])
            out[row["root_seed"]] = labels
            family_of[row["root_seed"]] = fam
    every = list(out)

    def report(left, right, roots, seed):
        return stats.paired_report([{"root_seed": k, "left": out[k][left], "right": out[k][right]}
                                    for k in roots], roots, bootstrap_seed=seed)

    def summary(left, right, roots, seed):
        r = report(left, right, roots, seed)
        sav, n = saving([(out[k][left], out[k][right]) for k in roots])
        return {**{k: r["primary"][k] for k in ("wins", "losses", "ties", "missing", "p_two_sided",
                                                  "conditional_win_probability_ci95")},
                "energy": r["energy"], "mean_relative_saving": sav, "jointly_feasible": n}

    pooled_seed = seeds["analysis"]["pooled"]
    primary = summary("llm_champion", "gp_champion", every, pooled_seed)
    p = primary["p_two_sided"]
    primary["directional_claim"] = bool(p is not None and p <= ALPHA
                                        and primary["wins"] > primary["losses"])
    primary["opposite_direction"] = bool(p is not None and p <= ALPHA
                                         and primary["losses"] > primary["wins"])
    k1 = summary("llm_champion", "oracle_reference", every, pooled_seed)
    full_vs_zeroshot = summary("llm_champion", "zeroshot_champion", every, pooled_seed)
    k2 = summary("llm_champion_atN", "zeroshot_champion", every, pooled_seed)
    gate_open = primary["directional_claim"]
    adjusted = stats.holm_two([k1["p_two_sided"], k2["p_two_sided"]]) if gate_open else [None, None]
    for k, adj in ((k1, adjusted[0]), (k2, adjusted[1])):
        k["p_holm"] = adj
        k["confirmatory"] = gate_open
        k["directional_claim"] = bool(adj is not None and adj <= ALPHA and k["wins"] > k["losses"])
        k["opposite_direction"] = bool(adj is not None and adj <= ALPHA
                                       and k["losses"] > k["wins"])

    contrasts = [("llm_champion", "gp_champion"), ("llm_champion", "oracle_reference"),
                 ("llm_champion", "zeroshot_champion"), ("gp_champion", "oracle_reference"),
                 ("llm_champion_atN", "zeroshot_champion"),
                 ("llm_champion", "initialization_best"), ("gp_champion", "initialization_best"),
                 ("zeroshot_champion", "initialization_best"),
                 ("llm_champion", "reference:latency_greedy")]
    per_family = {}
    for fam in FAMILIES:
        roots = [k for k in every if family_of[k] == fam]
        fam_seed = seeds["analysis"][fam]
        per_family[fam] = {
            "contrasts": {f"{a} vs {b}": summary(a, b, roots, fam_seed) for a, b in contrasts},
            "feasibility": {lab: sum(out[k][lab]["feasible"] is True for k in roots) / len(roots)
                            for lab in [*LABELS, "oracle_reference"]},
            "evaluated": {lab: sum(out[k][lab]["status"] == "evaluated" for k in roots)
                          for lab in [*LABELS, "oracle_reference"]},
            "mean_energy_when_feasible": {
                lab: (lambda v: sum(v) / len(v) if v else None)(
                    [out[k][lab]["energy"] for k in roots if out[k][lab]["feasible"]])
                for lab in [*LABELS, "oracle_reference"]},
            "llm_equals_gp": sum(out[k]["llm_champion"]["source_sha256"] is not None
                                 and out[k]["llm_champion"]["source_sha256"]
                                 == out[k]["gp_champion"]["source_sha256"] for k in roots),
        }
    t_roots = [k for k in every if family_of[k] == "T"]
    transfer = {
        "feasibility": {lab: sum(out[k][f"{lab}@drift"]["feasible"] is True for k in t_roots)
                        / len(t_roots) for lab in [*LABELS, "oracle_reference"]},
        "contrasts": {f"{a} vs {b}": summary(f"{a}@drift", f"{b}@drift", t_roots,
                                             seeds["analysis"]["T"])
                      for a, b in (("llm_champion", "gp_champion"),
                                   ("llm_champion", "oracle_reference"))},
    }
    pooled_desc = {f"{a} vs {b}": summary(a, b, every, pooled_seed) for a, b in contrasts}
    missing_primary = primary["missing"]
    result = {"statistics_source_sha256": actual,
              "primary": primary, "key_secondary": {"K1 llm vs oracle reference": k1,
                                                    "K2 llm_atN vs zeroshot": k2,
                                                    "gate_open": gate_open},
              "confirmatory_blocked": bool(missing_primary),
              "secondary_blocked": bool(k1["missing"] or k2["missing"]),
              "pooled_descriptive": pooled_desc, "full_llm_vs_zeroshot": full_vs_zeroshot, "per_family": per_family,
              "transfer_T_to_drift": transfer,
              "outcomes": {str(k): {"family": family_of[k], **v} for k, v in out.items()}}
    (HERE / "analysis").mkdir(exist_ok=True)
    with (HERE / "analysis" / "PRIMARY.json").open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True,
                  default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
        stream.write("\n")
    show = ("wins", "losses", "ties", "missing", "p_two_sided")
    print("P  ", {k: primary[k] for k in (*show, "directional_claim")})
    print("K1 ", {k: k1[k] for k in (*show, "p_holm", "directional_claim")})
    print("K2 ", {k: k2[k] for k in (*show, "p_holm", "directional_claim")})
    for fam in FAMILIES:
        f = per_family[fam]
        print(f"== {fam}", {k: round(v, 2) for k, v in f["feasibility"].items()
                           if k in ("llm_champion", "gp_champion", "zeroshot_champion",
                                    "initialization_best", "oracle_reference")})
        for name, c in f["contrasts"].items():
            print("  ", name, (c["wins"], c["losses"], c["ties"]), c["p_two_sided"],
                  c["mean_relative_saving"], c["jointly_feasible"])
    print("transfer T->drift", {k: round(v, 2) for k, v in transfer["feasibility"].items()
                                if k in ("llm_champion", "gp_champion", "oracle_reference")})


if __name__ == "__main__":
    main()
