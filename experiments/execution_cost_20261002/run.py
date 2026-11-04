"""Execution cost of selected policies and fixed references (post-study measurement).

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
RUNS = ROOT / "experiments/corrected_ab_20260925/runs"
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
    """Distinct sources with every (study, root, label) that selected or fixed them."""
    sources = {}
    for root in sorted(RUNS.iterdir()):
        labels_path = root / "final/LABELS.json"
        if not labels_path.exists():
            continue
        for sha, labels in json.loads(labels_path.read_text())["labels"].items():
            entry = sources.setdefault(sha, {"source": (root / "final" / f"{sha}.ps").read_text(),
                                             "uses": []})
            entry["uses"].extend(f"{root.name}:{label}" for label in labels)
    return sources


def proposer_timing(policy, nodes, pods, graph, repeats=100, trials=3):
    state = {"nodes": nodes, "pods": pods}
    policy.propose_moves(state, graph, 3)
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
        gate = gate_scorer_latency(policy, seed=20261002)
        row = {
            "source_sha256": sha,
            "uses": entry["uses"],
            "scorer_p50_ms": gate.stats.get("p50_ms"),
            "scorer_p99_ms": gate.stats.get("p99_ms"),
            "gate_passed": gate.passed,
            "gate_failures": gate.failures,
            "fuel_per_decision": policy.score_batch_cost(pods[3], nodes, ctx),
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
    (HERE / "RESULTS.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
