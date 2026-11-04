"""TODO 1.6: the property-based risk gates must (a) pass every seed policy and
(b) catch policies that violate each gated property — a gate that cannot fail is not a
gate. Seeds follow docs/SEEDS.md (explicit, derived; never wall-clock)."""

from pathlib import Path

import pytest

pytest.importorskip("placementscript")

from analysis.tracking import derive_seed  # noqa: E402
from evolution.risk_gates import run_risk_gates  # noqa: E402

POLICIES = Path(__file__).resolve().parent.parent / "runtime" / "policies"
SEED_NAMES = ["least_allocated", "most_allocated", "latency_greedy", "energy_greedy"]
ROOT_SEED = 20260713  # test-suite root seed (docs/SEEDS.md)

BUNDLE_SKELETON = """POLICY_NAME: str = "{name}"
POLICY_VERSION: str = "0.0.1"

def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:
{score_body}

def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:
{propose_body}
"""


def bundle(name: str, score_body: str, propose_body: str) -> str:
    return BUNDLE_SKELETON.format(
        name=name, score_body=score_body, propose_body=propose_body
    )


RETURN_EMPTY = "    out: list[Move] = []\n    return out"


class TestSeedPoliciesPassAllGates:
    @pytest.mark.parametrize("name", SEED_NAMES)
    def test_seed_policy(self, name):
        source = (POLICIES / f"{name}.ps").read_text()
        results = run_risk_gates(
            source, seed=derive_seed(ROOT_SEED, "risk-gates", name)
        )
        failing = {g: r.failures for g, r in results.items() if not r.passed}
        assert not failing, failing
        lat = results["scorer_latency"].stats
        print(
            f"\n{name}: latency p50={lat['p50_ms']:.3f}ms p99={lat['p99_ms']:.3f}ms "
            f"(budget {lat['budget_ms']}ms, {lat['n_nodes']} nodes/cycle)"
        )


class TestGatesCatchViolations:
    def test_non_total_scorer_caught(self):
        # Traps whenever the node has zero allocated CPU (division produces non-finite).
        src = bundle(
            "trapper",
            "    return 1.0 / float(node.cpu_allocated)",
            RETURN_EMPTY,
        )
        results = run_risk_gates(src, seed=derive_seed(ROOT_SEED, "risk-gates", "trapper"))
        assert not results["scorer_properties"].passed
        assert any("not total" in f for f in results["scorer_properties"].failures)

    def test_budget_violation_caught(self):
        src = bundle(
            "budget_buster",
            "    return 0.0",
            "    out: list[Move] = []\n"
            "    for i in range(budget + 1):\n"
            "        out.append(move_pod(0, 0))\n"
            "    return out",
        )
        results = run_risk_gates(
            src, seed=derive_seed(ROOT_SEED, "risk-gates", "budget_buster")
        )
        assert not results["proposer_properties"].passed
        assert any("budget violation" in f for f in results["proposer_properties"].failures)

    def test_stateful_exclusion_violation_caught(self):
        # Moves every pod, including stateful/PDB-protected ones (C8 violation).
        src = bundle(
            "c8_violator",
            "    return 0.0",
            "    out: list[Move] = []\n"
            "    moved = 0\n"
            "    for pod in state.pods:\n"
            "        if moved < budget:\n"
            "            out.append(move_pod(pod.id, 0))\n"
            "            moved += 1\n"
            "    return out",
        )
        results = run_risk_gates(
            src, seed=derive_seed(ROOT_SEED, "risk-gates", "c8_violator")
        )
        assert not results["proposer_properties"].passed
        assert any("C8 violation" in f for f in results["proposer_properties"].failures)

    def test_wake_ordering_violation_caught(self):
        # Proposes a move onto node 0 even when node 0 is suspended, without wake().
        src = bundle(
            "sleepwalker",
            "    return 0.0",
            "    out: list[Move] = []\n"
            "    for pod in state.pods:\n"
            "        ok = len(out) < budget and not pod.stateful and not pod.has_pdb\n"
            "        if ok and state.nodes[0].suspended:\n"
            "            out.append(move_pod(pod.id, 0))\n"
            "    return out",
        )
        results = run_risk_gates(
            src, seed=derive_seed(ROOT_SEED, "risk-gates", "sleepwalker")
        )
        assert not results["proposer_properties"].passed
        assert any("without a preceding wake" in f for f in results["proposer_properties"].failures)

    def test_latency_budget_violation_caught(self):
        # Burns fuel proportional to cluster size^2 — blows the 5 ms cycle budget.
        src = bundle(
            "slowpoke",
            "    acc = 0.0\n"
            "    for i in range(40):\n"
            "        for node2 in ctx.nodes:\n"
            "            for other in ctx.nodes:\n"
            "                acc += node2.latency_to_ms[other.id] * 0.001\n"
            "    return acc",
            RETURN_EMPTY,
        )
        results = run_risk_gates(
            src, seed=derive_seed(ROOT_SEED, "risk-gates", "slowpoke")
        )
        lat = results["scorer_latency"]
        assert not lat.passed, lat.stats
        assert any("exceeds budget" in f for f in lat.failures) or any(
            "failed on latency cluster" in f for f in lat.failures
        )

    def test_broken_bundle_reports_load_failure_with_class(self):
        results = run_risk_gates(
            "def score(pod: Pod) -> float:\n    return 0.0\n",
            seed=derive_seed(ROOT_SEED, "risk-gates", "broken"),
        )
        assert set(results) == {"load"}
        assert not results["load"].passed
        assert "TypeCheckError" in results["load"].failures[0]

    def test_determinism_gate_runs(self):
        # PlacementScript has no nondeterminism to inject (that is the point); this
        # pins that the determinism check executes and passes on a stateless policy.
        src = bundle("det", "    return float(node.cpu_allocated)", RETURN_EMPTY)
        results = run_risk_gates(src, seed=derive_seed(ROOT_SEED, "risk-gates", "det"))
        assert results["scorer_properties"].passed
        assert results["proposer_properties"].passed
