"""Property-based risk-reduction gates for PolicyBundles (TODO 1.6).

Usage (the promotion pipeline and Phase-5 harness import this):

    from analysis.tracking import derive_seed
    from evolution.risk_gates import run_risk_gates

    results = run_risk_gates(source, seed=derive_seed(root_seed, "risk-gates", name))
    ok = all(r.passed for r in results.values())
"""

from .gates import (
    DECISION_LATENCY_BUDGET_MS,
    GateResult,
    gate_proposer_properties,
    gate_scorer_latency,
    gate_scorer_properties,
    run_risk_gates,
)
from .strategies import clusters

__all__ = [
    "DECISION_LATENCY_BUDGET_MS",
    "GateResult",
    "clusters",
    "gate_proposer_properties",
    "gate_scorer_latency",
    "gate_scorer_properties",
    "run_risk_gates",
]
