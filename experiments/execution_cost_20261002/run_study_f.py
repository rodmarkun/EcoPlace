"""Execution cost of Study F selected policies (same procedure as run.py and run_study_c.py).

Committed in experiments/scientific_gates_20260920/DECISIONS_20260924.json: after final
selection, measure every selected policy and fixed reference on an otherwise idle, identified
machine under the unchanged 5 ms gate (scorer p50/p99), plus memory-independent cost (fuel per
decision) and code size. Also times the rebalancing proposer on the same pinned cluster.
No study outcome is changed; this only characterizes the programs.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import placementscript as ps

from evolution.risk_gates.gates import DECISION_LATENCY_BUDGET_MS, gate_scorer_latency

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
RUNS = ROOT / "experiments/study_f_20261007/runs"
N_NODES, N_PODS = 200, 60


def pinned_cluster():
    """Identical shape to gate_scorer_latency's pinned cluster."""
    nodes = [{"id": i, "tier": i % 3, "cpu_capacity": 4000 + 1000 * (i % 4),
              "cpu_allocated": 300 * (i % 5), "mem_capacity": 8192,
              "mem_allocated": 512 * (i % 6), "power_idle_w": 2.0 + i,
              "power_max_w": 8.0 + 2.0 * i, "suspended": False, "suspendable": i % 2 == 0,
              "latency_to_ms": [abs(i - j) * 2.5 for j in range(N_NODES)]}
             for i in range(N_NODES)]
    pods = [{"id": p, "service": p % 4, "app": 0, "cpu_request": 100 + 50 * (p % 3),
             "mem_request": 128, "stateful": p % 5 == 0, "has_pdb": p % 7 == 0,
             "image_mb": 50, "current_node": -1 if p % 4 == 3 else p % (N_NODES - 1)}
            for p in range(N_PODS)]
    graph = {"src": list(range(N_PODS - 1)), "dst": [e + 1 for e in range(N_PODS - 1)],
             "rate": [0.5 + e for e in range(N_PODS - 1)], "payload_kb": [1.0] * (N_PODS - 1)}
    return nodes, pods, graph


def corpus():
    """Distinct Study F selected sources (Sol search champion; validated Sol, GP, DeepSeek)."""
    sources = {}
    for root in sorted(RUNS.iterdir()):
        path = root / "SELECTION.json"
        if not path.exists():
            continue
        selection = json.loads(path.read_text())
        picks = {"sol_champion": selection["arms"]["sol"]["champion"],
                 **{f"{arm}_validated": rec for arm, rec in selection["validated"].items()}}
        for label, rec in picks.items():
            if rec is None:
                continue
            entry = sources.setdefault(rec["source_sha256"], {"source": rec["source"], "uses": []})
            entry["uses"].append(f"{root.name}:{label}")
    return sources


def fuel(policy, pod, nodes, ctx):
    """Fuel per decision; a bounded-runtime trap is recorded as an execution-cost outcome."""
    try:
        return policy.score_batch_cost(pod, nodes, ctx)
    except ps.PlacementScriptError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def proposer_timing(policy, nodes, pods, graph, repeats=100, trials=3):
    state = {"nodes": nodes, "pods": pods}
    try:
        policy.propose_moves(state, graph, 3)
    except ps.PlacementScriptError as exc:
        # A bounded runtime stops the call: recorded as an execution-cost outcome.
        return {"p50_ms": None, "p99_ms": None, "error": f"{type(exc).__name__}: {exc}"}
    samples = []
    for _ in range(trials * repeats):
        start = time.perf_counter()
        policy.propose_moves(state, graph, 3)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {"p50_ms": samples[len(samples) // 2],
            "p99_ms": samples[min(len(samples) - 1, (len(samples) * 99) // 100)]}


def main():
    nodes, pods, graph = pinned_cluster()
    ctx = {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}
    host = {
        "machine": platform.machine(), "platform": platform.platform(),
        "python": platform.python_version(),
        "model": subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True,
                                text=True).stdout.strip(),
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
        "load_before": os.getloadavg(),
    }
    rows = []
    for sha, entry in sorted(corpus().items()):
        policy = ps.Policy(entry["source"])
        try:
            gate = gate_scorer_latency(policy, seed=20261002)
            stats, passed, failures = gate.stats, gate.passed, gate.failures
        except ps.PlacementScriptError as exc:  # bounded-runtime trap: recorded, not fatal
            stats, passed, failures = {}, False, [f"{type(exc).__name__}: {exc}"]
        row = {
            "source_sha256": sha,
            "uses": entry["uses"],
            "scorer_p50_ms": stats.get("p50_ms"),
            "scorer_p99_ms": stats.get("p99_ms"),
            "gate_passed": passed,
            "gate_failures": failures,
            "fuel_per_decision": fuel(policy, pods[3], nodes, ctx),
            "proposer": proposer_timing(policy, nodes, pods, graph),
            "source_bytes": len(entry["source"].encode()),
            "source_lines": sum(1 for line in entry["source"].splitlines()
                                if line.strip() and not line.strip().startswith("#")),
            "ast_subtrees": len(ps.enumerate_subtrees(entry["source"])),
        }
        rows.append(row)
        print(json.dumps({"sha": sha[:10], "uses": len(entry["uses"]),
                          "p99": row["scorer_p99_ms"]}), flush=True)
    host["load_after"] = os.getloadavg()
    out = {"host": host, "budget_ms": DECISION_LATENCY_BUDGET_MS, "rows": rows,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (HERE / "RESULTS_STUDY_F.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
