"""Headroom calibration for Study E families (before any registration or search).

Scores the ten runtime/policies references on each candidate family with dedicated
calibration seeds (never reused by the study) across all three bundles, and reports
feasibility and normalized energy per reference. A family qualifies (gate) only if at least
one reference is feasible on every calibration seed and some reference that is infeasible
or feasible uses clearly less energy than the cheapest feasible one, i.e. there is energy
headroom for search to exploit. Run: uv run python -m experiments.study_e_20261005.calibrate
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
BASE_TASK = ROOT / "prereg/STUDY_C_ROBUST_20261003/task_C.json"
FAMILIES = {
    "tight": ("dev-continuum-tight", "sim/scenarios/dev/continuum_tight.json"),
    "wide": ("dev-continuum-wide", "sim/scenarios/dev/continuum_wide.json"),
    "drift": ("dev-continuum-drift", "sim/scenarios/dev/continuum_drift.json"),
}
REFERENCES = ("latency_greedy", "energy_greedy", "first_fit", "least_allocated",
              "most_allocated", "kube_least_allocated", "kube_most_allocated",
              "network_overhead", "graph_headroom")
CAL_SEEDS = 4


def family_task(family, seeds):
    sid, path = FAMILIES[family]
    task = json.loads(BASE_TASK.read_text())
    task["scenarios"] = [{"id": sid, "path": path, "seeds": seeds}]
    task["ladder"]["l1"]["scenario_ids"] = [sid]
    task["task_variant"] = sid
    return task


def calibration_seeds(family):
    from analysis.tracking import derive_seed

    return [derive_seed(20261005, "study-e-calibration", family, j) for j in range(CAL_SEEDS)]


def score(job):
    family, ref = job
    import tempfile

    from evolution.evaluation import evaluate_policy_source, load_task_evaluation_config
    from evolution.ladder import _evaluation_trace

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=ROOT) as f:
        json.dump(family_task(family, calibration_seeds(family)), f)
        path = f.name
    try:
        task = load_task_evaluation_config(Path(path))
        source = (ROOT / "runtime/policies" / f"{ref}.ps").read_text()
        ev = evaluate_policy_source(source, task, proposal_seq=0, retain_episode_ledgers=False)
        trace = _evaluation_trace(ev)
        inst = trace["evaluation"]["instances"]
        per = [min(i["slo_achieved"].values()) for i in inst]
        return {"family": family, "ref": ref, "feasible": bool(trace["feasible"]),
                "energy": trace["normalized_energy"],
                "shortfall": trace["summed_primitive_shortfall"],
                "instances_feasible": per}
    finally:
        Path(path).unlink()


def main():
    families = sys.argv[1:] or list(FAMILIES)
    jobs = [(f, r) for f in families for r in REFERENCES]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(score, jobs))
    out = HERE / "calibration"
    out.mkdir(exist_ok=True)
    for f in families:
        fr = [r for r in rows if r["family"] == f]
        (out / f"{f}.json").write_text(json.dumps(fr, indent=1))
        print(f"== {f}")
        for r in sorted(fr, key=lambda r: (not r["feasible"], r["energy"] or 9)):
            inst = r["instances_feasible"]
            frac = f"{min(inst):.3f}"
            print(f"  {r['ref']:22s} feasible={r['feasible']!s:5} worstSLO={frac:6} "
                  f"energy={r['energy'] if r['energy'] is None else round(r['energy'], 3)} "
                  f"shortfall={round(r['shortfall'], 4)}")


if __name__ == "__main__":
    main()
