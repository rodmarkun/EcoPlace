"""Exploratory (post hoc, not registered): power over time of the selected Study F policies.

Re-runs each run's validated Sol, GP and DeepSeek winner on its final grid (8 seeds x 3 energy
settings, exactly as in the sealed final judgement) with a read-only sampler that records the
cumulative modeled energy every STEP_S seconds, split into fog, edge, cloud and network. The
sampler only reads the energy meter, so episodes are unchanged; every instance's total is checked
against the sealed final record before it is kept.
Run from the repository root:
    .venv/bin/python -m experiments.study_f_20261007.exploratory.power_timeseries [workers]
"""

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
F_RUNS = ROOT / "experiments/study_f_20261007/runs"
E_RUNS = ROOT / "experiments/study_e_20261005/runs"
OUT = Path(__file__).parent / "power"
ARMS = ("sol", "gp", "deepseek")
STEP_S = 5.0


def cumulative(sim):
    """Cumulative fog, edge, cloud and network energy (J) at sim time, without mutating."""
    from sim.energy.boundary import TIER_CLOUD

    TIER_FOG = 1
    m, now = sim.meter, sim.env.now
    fog = edge = 0.0
    for i, cls in enumerate(m._classes):
        if m._tiers[i] == TIER_CLOUD:
            continue
        state = m._state[i]
        if state == "awake":
            p = cls.power_w(m._busy_millicores[i] / m._cluster.nodes[i].cpu_capacity)
        elif state == "suspended":
            p = cls.suspended_power_w
        else:
            p = cls.power_idle_w
        e = m._device_j[i] + m._transition_j[i] + p * (now - m._last_t[i])
        if m._tiers[i] == TIER_FOG:
            fog += e
        else:
            edge += e
    vcpu_h = m._cloud_vcpu_hours + m._cloud_vcpu * (now - m._cloud_last_t) / 3600.0
    return [fog, edge, vcpu_h * m._coeff.cloud_j_per_vcpu_hour, m._bytes * m._coeff.network_j_per_byte]


def install_sampler(sink):
    import evolution.evaluation as evaluation
    from sim.engine import Simulation

    class Sampled(Simulation):
        def start(self):
            super().start()
            samples = []
            sink.append({"seed": self.seed, "samples": samples})

            def sampler():
                while self.env.now < self.config.horizon_s:
                    samples.append(cumulative(self))
                    yield self.env.timeout(STEP_S)

            self.env.process(sampler())

    def run_episode(config, seed, scorer, proposer=None, lb_policy="round_robin"):
        result = Sampled(config, seed, scorer, proposer, lb_policy).run()
        e = result.energy
        sink[-1]["final"] = [e.device_j, e.cloud_j, e.network_j]
        return result

    evaluation.run_episode = run_episode


def job(args):
    root_name, arm = args
    out = OUT / f"{root_name}_{arm}.json"
    if out.exists():
        return out.name, "cached"
    from evolution.evaluation import evaluate_policy_source
    from experiments.study_e_20261005 import launch as E

    study, index = root_name[0], int(root_name[1:])
    selection = json.loads((F_RUNS / root_name / "SELECTION.json").read_text())
    pick = selection["validated"][arm]
    source, sha = pick["source"], pick["source_sha256"]
    sealed_path = F_RUNS / root_name / "final" / f"{sha}.json"
    if not sealed_path.exists():
        sealed_path = E_RUNS / root_name / "final" / f"{sha}.json"
    sealed = json.loads(sealed_path.read_text())
    sink = []
    install_sampler(sink)
    task = E.final_task(E.SCIENTIFIC, study, index)
    try:
        evaluate_policy_source(source, task, proposal_seq=0, retain_episode_ledgers=False)
    except Exception as exc:  # invalid on this grid: keep the sealed verdict, no trace
        if sealed.get("valid", True):
            raise
        out.write_text(json.dumps({"valid": False, "error": str(exc)}))
        return out.name, "invalid"
    instances = sealed["trace"]["evaluation"]["instances"]
    if len(instances) != len(sink):
        raise RuntimeError(f"{root_name} {arm}: {len(sink)} episodes, sealed {len(instances)}")
    traces = []
    for inst, run in zip(instances, sink, strict=True):
        sealed_e = inst["episode"]["energy"]
        device, cloud, network = run["final"]
        for a, b in ((device, sealed_e["device_j"]), (cloud, sealed_e["cloud_j"]),
                     (network, sealed_e["network_j"])):
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6):
                raise RuntimeError(f"{root_name} {arm}: re-run energy {a} != sealed {b}")
        traces.append({"seed": inst["seed"], "bundle_id": inst["bundle_id"],
                       "samples": run["samples"], "final": run["final"],
                       "idle_floor_j": inst["idle_floor_j"]})
    out.write_text(json.dumps({"valid": True, "root": root_name, "arm": arm,
                               "source_sha256": sha, "feasible": sealed["trace"]["feasible"],
                               "normalized_energy": sealed["trace"]["normalized_energy"],
                               "step_s": STEP_S, "traces": traces}))
    return out.name, "ok"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    OUT.mkdir(exist_ok=True)
    jobs = [(root.name, arm) for root in sorted(F_RUNS.iterdir()) for arm in ARMS]
    # interleave scenarios so partial results cover all three
    jobs.sort(key=lambda j: (int(j[0][1:]), j[0][0], j[1]))
    with Pool(workers, maxtasksperchild=1) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name} {status}", flush=True)


if __name__ == "__main__":
    main()
