"""Exploratory (post hoc, not registered): how do the selected Study F policies spend energy?

Reads only sealed final-grid records (Study F, and Study E for the fixed references) and
summarizes, per task variant and label, the mean device, cloud and network energy, cloud
vCPU-hours, rebalancing moves and the worst per-template SLO attainment over the final grid.
Run from the repository root:
    .venv/bin/python -m experiments.study_f_20261007.exploratory.mechanism
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
F_RUNS = ROOT / "experiments/study_f_20261007/runs"
E_RUNS = ROOT / "experiments/study_e_20261005/runs"
OUT = Path(__file__).parent / "MECHANISM.json"
REFERENCES = ("graph_headroom", "latency_greedy", "kube_least_allocated")


def summarize(path):
    record = json.loads(path.read_text())
    if not record.get("valid", True):
        return {"valid": False}
    trace = record["trace"]
    instances = trace["evaluation"]["instances"]

    def mean(field):
        return st.mean(field(i) for i in instances)

    return {
        "valid": True,
        "feasible": trace["feasible"],
        "normalized_energy": trace["normalized_energy"],
        "device_kj": mean(lambda i: i["episode"]["energy"]["device_j"]) / 1e3,
        "cloud_kj": mean(lambda i: i["episode"]["energy"]["cloud_j"]) / 1e3,
        "network_kj": mean(lambda i: i["episode"]["energy"]["network_j"]) / 1e3,
        "cloud_vcpu_hours": mean(lambda i: i["episode"]["energy"]["cloud_vcpu_hours"]),
        "moves": mean(lambda i: i["episode"]["rebalancer"]["moves"]),
        "worst_slo": min(min(i["slo_achieved"].values()) for i in instances),
    }


def record_path(root_id, sha):
    path = F_RUNS / root_id / "final" / f"{sha}.json"
    return path if path.exists() else E_RUNS / root_id / "final" / f"{sha}.json"


def main():
    rows = {}
    for root in sorted(F_RUNS.iterdir()):
        selection = json.loads((root / "SELECTION.json").read_text())
        labels = json.loads((E_RUNS / root.name / "final/LABELS.json").read_text())["labels"]
        picks = {arm: selection["validated"][arm]["source_sha256"]
                 for arm in ("sol", "gp", "deepseek")}
        for name in REFERENCES:
            picks[name] = next(s for s, ls in labels.items() if f"reference:{name}" in ls)
        for label, sha in picks.items():
            rows.setdefault(f"{root.name[0]}|{label}", []).append(
                {"root": root.name, **summarize(record_path(root.name, sha))})
    summary = {}
    for key, values in sorted(rows.items()):
        valid = [v for v in values if v["valid"]]
        feasible = [v["normalized_energy"] for v in valid if v["feasible"]]
        summary[key] = {
            "roots": len(values), "valid": len(valid), "feasible": len(feasible),
            "median_energy_when_feasible": st.median(feasible) if feasible else None,
            **{f"mean_{k}": st.mean(v[k] for v in valid)
               for k in ("device_kj", "cloud_kj", "network_kj", "cloud_vcpu_hours", "moves",
                         "worst_slo")},
        }
        s = summary[key]
        print(f"{key:24s} feasible {s['feasible']:2d}/{s['roots']} "
              f"device {s['mean_device_kj']:6.1f} kJ  cloud {s['mean_cloud_kj']:4.1f} kJ  "
              f"vCPU-h {s['mean_cloud_vcpu_hours']:.2f}  moves {s['mean_moves']:5.1f}  "
              f"median E {s['median_energy_when_feasible'] or float('nan'):.3f}")
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
