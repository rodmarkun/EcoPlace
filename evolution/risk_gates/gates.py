"""Property-based risk-reduction gates (TODO 1.6; C2/C8 scope).

The reusable component the promotion pipeline (7.6) and the evolution harness (Phase 5)
call before any candidate acts. Scope is exactly what the DSL + delegation design leaves
reachable (C8): the gates check

- **scorer totality** — `score` returns a finite float on every generated state (no
  trap/fuel/oom/host failure on the host-guaranteed input domain, spec §4.1);
- **scorer determinism** — bit-identical outputs on repeated evaluation;
- **batch/scalar equivalence** — `score_batch` is bitwise the map of `score` over the
  candidates, and is invariant under candidate permutation (C2: per-node Score, no
  cross-candidate state);
- **decision-latency budget** — measured `score_batch` on a pinned synthetic cluster
  against the ≤5 ms scheduling-cycle budget (plan §3.5); gated on **p99** (tail), p50
  recorded as a diagnostic; measured, not assumed;
- **proposer budget compliance** — never more than `budget` actions;
- **proposer exclusions (C8)** — stateful / PDB-protected pods are never moved; move
  destinations exist; a move onto a suspended node must be preceded by `wake(node)`;
- **proposer totality + determinism** on the same domain.

Every property is checked both over sampled clusters (hypothesis, spec §4.1 domain — see
`strategies.py`) and over the deterministic corner clusters (`corner_clusters()`:
1-node, 0-pod, every-field-at-its-bound, wide).

Feasibility invariants (capacity, affinity, volumes, …) are *not* checked here: they are
out of the evolved code's reach by construction (native Filter plugins + execution-time
re-validation, C2/C3). These are risk-reduction gates, not guarantees (C2 wording).

Seeds are explicit per docs/SEEDS.md — there is no default; derive one with
`analysis.tracking.derive_seed(root_seed, "risk-gates", policy_name, gate)`.
"""

from __future__ import annotations

import statistics
import struct
import time
from dataclasses import dataclass, field

import hypothesis
import placementscript as ps
from hypothesis import HealthCheck, given, settings

from .strategies import clusters, corner_clusters

DECISION_LATENCY_BUDGET_MS = 5.0
_MAX_REPORTED_FAILURES = 8


@dataclass
class GateResult:
    """Outcome of one gate."""

    gate: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def merge_failure(self, msg: str) -> None:
        self.passed = False
        if len(self.failures) < _MAX_REPORTED_FAILURES:
            self.failures.append(msg)


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def _settings(max_examples: int):
    return settings(
        max_examples=max_examples,
        database=None,  # no cross-run state: reproducibility comes from the seed
        deadline=None,  # latency is gated explicitly, not per-example
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )


def _check_scorer(policy: ps.Policy, c: dict, result: GateResult, tag: str = "") -> None:
    """Totality, bitwise determinism, batch=scalar map, and permutation invariance of
    `score` on one cluster."""
    pod, nodes, ctx = c["pending_pod"], c["nodes"], c["ctx"]
    try:
        scores = policy.score_batch(pod, nodes, ctx)
        again = policy.score_batch(pod, nodes, ctx)
    except ps.PlacementScriptError as e:
        result.merge_failure(f"scorer not total{tag}: {type(e).__name__}: {e}")
        return
    for s in scores:
        if s != s or s in (float("inf"), float("-inf")):  # pragma: no cover
            result.merge_failure(f"non-finite score {s!r}{tag}")
    if [_bits(s) for s in scores] != [_bits(s) for s in again]:
        result.merge_failure(f"scorer nondeterministic{tag}: outputs differ bitwise")
    # Batch = map(scalar): one candidate must never observe another's evaluation (C2).
    try:
        singles = [policy.score(pod, n, ctx) for n in nodes]
    except ps.PlacementScriptError as e:  # pragma: no cover
        result.merge_failure(f"scalar score failed where batch succeeded{tag}: {e}")
        return
    if [_bits(s) for s in scores] != [_bits(s) for s in singles]:
        result.merge_failure(f"batch/scalar divergence{tag}: score_batch != map(score)")
    # Candidate-permutation invariance: reversed batch gives reversed scores, bitwise.
    try:
        rev = policy.score_batch(pod, list(reversed(nodes)), ctx)
    except ps.PlacementScriptError as e:  # pragma: no cover
        result.merge_failure(f"scorer failed on permuted batch{tag}: {e}")
        return
    if [_bits(s) for s in rev] != [_bits(s) for s in reversed(scores)]:
        result.merge_failure(f"candidate-order dependence{tag}: permuted batch diverges")


def gate_scorer_properties(
    policy: ps.Policy, *, seed: int, max_examples: int = 120
) -> GateResult:
    """Totality + bitwise determinism + batch/scalar + permutation invariance of
    `score` over generated cluster states and the deterministic domain corners."""
    result = GateResult(gate="scorer_properties", passed=True)
    examples = 0

    @_settings(max_examples)
    @hypothesis.seed(seed)
    @given(clusters())
    def run(c):
        nonlocal examples
        examples += 1
        _check_scorer(policy, c, result)

    run()
    for name, c in corner_clusters().items():
        _check_scorer(policy, c, result, tag=f" [corner: {name}]")
    result.stats["examples"] = examples
    return result


def gate_scorer_latency(
    policy: ps.Policy,
    *,
    seed: int,
    budget_ms: float = DECISION_LATENCY_BUDGET_MS,
    n_nodes: int = 200,
    n_pods: int = 60,
    repeats: int = 300,
    trials: int = 3,
) -> GateResult:
    """Measured `score_batch` wall-clock on a pinned synthetic cluster (one scheduling
    cycle: one pod against `n_nodes` feasible nodes). Gates on **p99 ≤ budget** (the
    scheduling cycle is a tail-latency requirement, not a median one); p50 recorded as
    a diagnostic. The gate statistic is the **pooled p99 over all samples of all
    `trials` trials** (default 3 × 300 samples; the sorted sample at zero-based index
    floor(0.99 * N), not the maximum). Environment interference can inflate the
    measured tail, but finite sampling can also underestimate the population p99.
    This empirical statistic is not an upper confidence bound or a deployment
    guarantee. Per-trial p99s are diagnostics; selecting their minimum would hide
    observed variation. The pinned shape makes runs comparable; `seed` is recorded for the
    tracking store even though the cluster is deterministic."""
    nodes = [
        {
            "id": i,
            "tier": i % 3,
            "cpu_capacity": 4000 + 1000 * (i % 4),
            "cpu_allocated": 300 * (i % 5),
            "mem_capacity": 8192,
            "mem_allocated": 512 * (i % 6),
            "power_idle_w": 2.0 + i,
            "power_max_w": 8.0 + 2.0 * i,
            "suspended": False,
            "suspendable": i % 2 == 0,
            "latency_to_ms": [abs(i - j) * 2.5 for j in range(n_nodes)],
        }
        for i in range(n_nodes)
    ]
    pods = [
        {
            "id": p,
            "service": p % 4,
            "app": 0,
            "cpu_request": 100 + 50 * (p % 3),
            "mem_request": 128,
            "stateful": p % 5 == 0,
            "has_pdb": p % 7 == 0,
            "image_mb": 50,
            "current_node": -1 if p % 4 == 3 else p % (n_nodes - 1),
        }
        for p in range(n_pods)
    ]
    graph = {
        "src": list(range(n_pods - 1)),
        "dst": [e + 1 for e in range(n_pods - 1)],
        "rate": [0.5 + e for e in range(n_pods - 1)],
        "payload_kb": [1.0] * (n_pods - 1),
    }
    ctx = {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}

    result = GateResult(gate="scorer_latency", passed=True)
    trial_p99s: list[float] = []
    pooled: list[float] = []
    try:
        policy.score_batch(pods[3], nodes, ctx)  # warm-up
        for _ in range(trials):
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                policy.score_batch(pods[3], nodes, ctx)
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            trial_p99s.append(samples[min(len(samples) - 1, (len(samples) * 99) // 100)])
            pooled.extend(samples)
    except ps.PlacementScriptError as e:
        result.merge_failure(f"scorer failed on latency cluster: {type(e).__name__}: {e}")
        return result
    pooled.sort()
    p50 = statistics.median(pooled)
    p99 = pooled[min(len(pooled) - 1, (len(pooled) * 99) // 100)]
    result.stats = {
        "seed": seed,
        "n_nodes": n_nodes,
        "repeats": repeats,
        "trials": trials,
        "p50_ms": p50,
        "p99_ms": p99,
        "trial_p99s_ms": trial_p99s,
        "budget_ms": budget_ms,
    }
    if p99 > budget_ms:
        result.merge_failure(
            f"decision latency p99 {p99:.3f}ms exceeds budget {budget_ms}ms "
            f"(pooled over {trials}x{repeats} samples; p50 {p50:.3f}ms, "
            f"{n_nodes} nodes/cycle)"
        )
    return result


def _check_proposal(c: dict, moves: list[dict], result: GateResult, tag: str = "") -> None:
    n_nodes = len(c["nodes"])
    if len(moves) > c["budget"]:
        result.merge_failure(
            f"budget violation{tag}: {len(moves)} actions > budget {c['budget']}"
        )
    woken: set[int] = set()
    for m in moves:
        if m["kind"] not in (0, 1):
            result.merge_failure(f"unknown action kind {m['kind']}{tag}")
            continue
        if not (0 <= m["dest"] < n_nodes):
            result.merge_failure(f"action names unknown node {m['dest']}{tag}")
            continue
        if m["kind"] == 1:
            woken.add(m["dest"])
            continue
        if not (0 <= m["pod"] < len(c["pods"])):
            result.merge_failure(f"move names unknown pod {m['pod']}{tag}")
            continue
        pod = c["pods"][m["pod"]]
        if pod["stateful"]:
            result.merge_failure(f"C8 violation{tag}: moved stateful pod {m['pod']}")
        if pod["has_pdb"]:
            result.merge_failure(f"C8 violation{tag}: moved PDB-protected pod {m['pod']}")
        if c["nodes"][m["dest"]]["suspended"] and m["dest"] not in woken:
            result.merge_failure(
                f"move to suspended node {m['dest']} without a preceding wake(){tag}"
            )


def _check_proposer(policy: ps.Policy, c: dict, result: GateResult, tag: str = "") -> None:
    try:
        moves = policy.propose_moves(c["state"], c["graph"], c["budget"])
        again = policy.propose_moves(c["state"], c["graph"], c["budget"])
    except ps.PlacementScriptError as e:
        result.merge_failure(f"proposer not total{tag}: {type(e).__name__}: {e}")
        return
    if moves != again:
        result.merge_failure(f"proposer nondeterministic{tag}: proposals differ")
    _check_proposal(c, moves, result, tag)


def gate_proposer_properties(
    policy: ps.Policy, *, seed: int, max_examples: int = 100
) -> GateResult:
    """Totality, determinism, budget compliance, and C8 exclusions of `propose_moves`
    over generated cluster states and the deterministic domain corners."""
    result = GateResult(gate="proposer_properties", passed=True)
    examples = 0

    @_settings(max_examples)
    @hypothesis.seed(seed)
    @given(clusters())
    def run(c):
        nonlocal examples
        examples += 1
        _check_proposer(policy, c, result)

    run()
    for name, c in corner_clusters().items():
        _check_proposer(policy, c, result, tag=f" [corner: {name}]")
    result.stats["examples"] = examples
    return result


def run_risk_gates(source: str, *, seed: int) -> dict[str, GateResult]:
    """Run every gate on a PolicyBundle source. Compile/bundle failures are themselves a
    gate outcome (class name preserved for evolution feedback). Returns {gate: result};
    the bundle passes iff all results pass."""
    try:
        policy = ps.Policy(source)
    except ps.PlacementScriptError as e:
        return {
            "load": GateResult(
                gate="load",
                passed=False,
                failures=[f"{type(e).__name__}: {e}"],
            )
        }
    return {
        "load": GateResult(gate="load", passed=True, stats={"name": policy.name}),
        "scorer_properties": gate_scorer_properties(policy, seed=seed),
        "scorer_latency": gate_scorer_latency(policy, seed=seed),
        "proposer_properties": gate_proposer_properties(policy, seed=seed),
    }
