"""TODO 1.3 verification: the PyO3 native embedding, exercised from Python.

Covers: compile + bundle validation, the structured error taxonomy (evolution feedback
branches on it), batch scoring, determinism, -0.0 normalization, budget knobs, and a
measured (never claimed, C9) decision-latency ballpark printed for the report.
"""

import statistics
import struct
import time
from pathlib import Path

import pytest

ps = pytest.importorskip(
    "placementscript",
    reason=(
        "build the wheel first: "
        "uv run maturin develop -m runtime/crates/placementscript-py/Cargo.toml"
    ),
)

POLICIES = Path(__file__).resolve().parent.parent / "runtime" / "policies"
SEEDS = ["least_allocated", "most_allocated", "latency_greedy", "energy_greedy"]


def make_node(i: int, n: int, suspended: bool = False) -> dict:
    return {
        "id": i,
        "tier": i % 3,
        "cpu_capacity": 4000 + 1000 * (i % 4),
        "cpu_allocated": 300 * (i % 5),
        "mem_capacity": 8192,
        "mem_allocated": 512 * (i % 6),
        "power_idle_w": 2.0 + i,
        "power_max_w": 8.0 + 2.0 * i,
        "suspended": suspended,
        "suspendable": i % 2 == 0,
        "latency_to_ms": [abs(i - j) * 2.5 for j in range(n)],
    }


def make_pod(p: int, n_nodes: int) -> dict:
    return {
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


def make_cluster(n_nodes: int = 6, n_pods: int = 14):
    nodes = [make_node(i, n_nodes, suspended=(i == n_nodes - 1)) for i in range(n_nodes)]
    pods = [make_pod(p, n_nodes) for p in range(n_pods)]
    graph = {
        "src": list(range(n_pods - 1)),
        "dst": [e + 1 for e in range(n_pods - 1)],
        "rate": [0.5 + e for e in range(n_pods - 1)],
        "payload_kb": [1.0] * (n_pods - 1),
    }
    ctx = {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}
    state = {"nodes": nodes, "pods": pods}
    return nodes, pods, graph, ctx, state


def load_policy(name: str) -> "ps.Policy":
    return ps.Policy((POLICIES / f"{name}.ps").read_text())


class TestBundleLoading:
    def test_seed_policies_load_with_metadata(self):
        for name in SEEDS:
            pol = load_policy(name)
            assert pol.name == name
            assert pol.version == "1.0.0"

    def test_missing_metadata_is_typecheck_error(self):
        src = (POLICIES / "least_allocated.ps").read_text()
        src = src.replace('POLICY_NAME: str = "least_allocated"\n', "")
        with pytest.raises(ps.TypeCheckError, match="POLICY_NAME"):
            ps.Policy(src)


class TestErrorTaxonomy:
    """Each spec §9 class must surface as its own exception type."""

    def test_parse_error(self):
        with pytest.raises(ps.ParseError, match="finite"):
            ps.compile_check("def f() -> float:\n    return 1e999\n")

    def test_type_error(self):
        with pytest.raises(ps.TypeCheckError):
            ps.compile_check("def f(a: int, b: float) -> float:\n    return a + b\n")

    def test_trap_fuel_oom_distinguished(self):
        nodes, pods, _, ctx, _ = make_cluster()
        trap_pol = ps.Policy(
            'POLICY_NAME: str = "t"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    return 1.0 / (float(node.cpu_allocated) * 0.0)\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n"
            "    for i in range(budget):\n"
            "        out.append(move_pod(0, 0))\n"
            "    return out\n"
        )
        with pytest.raises(ps.TrapError, match="non-finite"):
            trap_pol.score(pods[1], nodes[0], ctx)

        loop_pol = ps.Policy(
            'POLICY_NAME: str = "t"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    acc = 0.0\n"
            "    for i in range(100000000):\n"
            "        acc += 1.0\n"
            "    return acc\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n"
            "    return out\n"
        )
        with pytest.raises(ps.FuelExhausted):
            loop_pol.score(pods[1], nodes[0], ctx)

        oom_pol = ps.Policy(
            'POLICY_NAME: str = "t"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    xs: list[int] = []\n"
            "    for i in range(100000000):\n"
            "        xs.append(i)\n"
            "    return 0.0\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n"
            "    return out\n"
        )
        with pytest.raises(ps.OutOfMemory):
            oom_pol.score(pods[1], nodes[0], ctx, fuel=10_000_000_000)

    def test_host_error_for_malformed_state(self):
        pol = load_policy("least_allocated")
        nodes, pods, _, ctx, _ = make_cluster()
        bad_node = dict(nodes[0])
        del bad_node["cpu_capacity"]
        with pytest.raises(ps.HostError, match="cpu_capacity"):
            pol.score(pods[0], bad_node, ctx)
        bad_node = dict(nodes[0], power_idle_w=float("nan"))
        with pytest.raises(ps.HostError, match="finite"):
            pol.score(pods[0], bad_node, ctx)


class TestScoring:
    def test_score_batch_matches_single_scores(self):
        nodes, pods, _, ctx, _ = make_cluster()
        for name in SEEDS:
            pol = load_policy(name)
            batch = pol.score_batch(pods[3], nodes, ctx)
            singles = [pol.score(pods[3], n, ctx) for n in nodes]
            assert batch == singles, name
            assert all(isinstance(s, float) for s in batch)

    def test_determinism_bitwise(self):
        nodes, pods, _, ctx, _ = make_cluster()
        for name in SEEDS:
            pol = load_policy(name)
            a = pol.score_batch(pods[3], nodes, ctx)
            b = pol.score_batch(pods[3], nodes, ctx)
            assert [struct.pack("<d", x) for x in a] == [struct.pack("<d", x) for x in b]

    def test_negative_zero_normalized(self):
        pol = ps.Policy(
            'POLICY_NAME: str = "t"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    return -0.0\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n"
            "    return out\n"
        )
        nodes, pods, _, ctx, _ = make_cluster()
        v = pol.score(pods[0], nodes[0], ctx)
        assert struct.pack("<d", v) == struct.pack("<d", 0.0)

    def test_fuel_budget_knob(self):
        pol = load_policy("latency_greedy")
        nodes, pods, _, ctx, _ = make_cluster()
        with pytest.raises(ps.FuelExhausted):
            pol.score(pods[3], nodes[0], ctx, fuel=10)


class TestProposer:
    def test_budget_and_exclusions(self):
        _, _, graph, _, state = make_cluster()
        excluded = {
            p["id"] for p in state["pods"] if p["stateful"] or p["has_pdb"]
        }
        for name in SEEDS:
            pol = load_policy(name)
            for budget in (0, 1, 3, 100):
                moves = pol.propose_moves(state, graph, budget)
                assert len(moves) <= budget, (name, budget)
                for m in moves:
                    assert m["kind"] in (0, 1)
                    if m["kind"] == 0:
                        assert m["pod"] not in excluded, name

    def test_wake_precedes_moves_to_suspended(self):
        # energy_greedy wakes a node only when no awake destination exists.
        nodes = [make_node(i, 3, suspended=(i > 0)) for i in range(3)]
        nodes[0]["suspendable"] = True
        nodes[0]["cpu_allocated"] = 100  # underutilized awake suspendable -> drain
        pods = [
            dict(make_pod(p, 3), stateful=False, has_pdb=False, current_node=0)
            for p in range(3)
        ]
        graph = {"src": [], "dst": [], "rate": [], "payload_kb": []}
        state = {"nodes": nodes, "pods": pods}
        pol = load_policy("energy_greedy")
        moves = pol.propose_moves(state, graph, 5)
        if any(m["kind"] == 0 and nodes[m["dest"]]["suspended"] for m in moves):
            woken = set()
            for m in moves:
                if m["kind"] == 1:
                    woken.add(m["dest"])
                elif nodes[m["dest"]]["suspended"]:
                    assert m["dest"] in woken, "move to suspended node without prior wake"


class TestLatencyBallpark:
    """Measured, not claimed (C9). Prints the numbers the report cites; the assert is a
    loose regression bound, not the ≤5 ms budget claim (the risk gate owns that check
    against its pinned cluster shape)."""

    def test_native_score_batch_latency(self):
        n_nodes = 200
        nodes, pods, _, ctx, _ = make_cluster(n_nodes=n_nodes, n_pods=60)
        for name in SEEDS:
            pol = load_policy(name)
            pol.score_batch(pods[3], nodes, ctx)  # warm-up
            samples = []
            for _ in range(30):
                t0 = time.perf_counter()
                out = pol.score_batch(pods[3], nodes, ctx)
                samples.append((time.perf_counter() - t0) * 1000.0)
                assert len(out) == n_nodes
            p50 = statistics.median(samples)
            print(
                f"\nnative score_batch [{name}]: {n_nodes} nodes/call "
                f"p50={p50:.3f}ms (~{1000 * p50 / n_nodes:.1f}µs/node)"
            )
            assert p50 < 100.0, f"{name}: implausibly slow, investigate"


class TestAgainstDirectExecution:
    def test_scores_are_plausible(self):
        nodes, pods, _, ctx, _ = make_cluster()
        pol = load_policy("least_allocated")
        scores = pol.score_batch(pods[3], nodes, ctx)
        # Spreading policy: emptier node (id 0, alloc 0) beats fuller node (id 3).
        assert scores[0] > scores[3]
        pol = load_policy("most_allocated")
        scores = pol.score_batch(pods[3], nodes, ctx)
        assert scores[3] > scores[0]


class TestBoundaryValidation:
    """Round-1 review R1.4/R1.6: exact numeric extraction, semantic validation, and the
    shared host tie-break."""

    def test_int_for_float_field_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        bad = dict(nodes[0])
        bad["power_idle_w"] = 3  # int where float is required: no implicit coercion
        with pytest.raises(ps.HostError, match="expected float"):
            policy.score(pods[3], bad, ctx)

    def test_bool_for_float_field_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        bad = dict(nodes[0])
        bad["power_idle_w"] = True
        with pytest.raises(ps.HostError, match="expected float"):
            policy.score(pods[3], bad, ctx)

    def test_oversized_int_is_host_error_not_overflow_error(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        bad = dict(pods[3])
        bad["cpu_request"] = 2**80  # would previously leak a raw OverflowError
        with pytest.raises(ps.HostError, match="i64"):
            policy.score(bad, nodes[0], ctx)

    def test_non_dense_node_ids_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        ctx = dict(ctx)
        ctx["nodes"] = [dict(nodes[0], id=5)] + list(nodes[1:])
        with pytest.raises(ps.HostError, match="dense"):
            policy.score_batch(pods[3], nodes, ctx)

    def test_ragged_graph_columns_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        ctx = dict(ctx)
        ctx["graph"] = dict(graph, src=list(graph["src"]) + [0])
        with pytest.raises(ps.HostError, match="columns"):
            policy.score_batch(pods[3], nodes, ctx)

    def test_wrong_latency_vector_length_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        bad = [dict(n) for n in nodes]
        bad[2]["latency_to_ms"] = bad[2]["latency_to_ms"][:-1]
        ctx = dict(ctx, nodes=bad)
        with pytest.raises(ps.HostError, match="latency_to_ms"):
            policy.score_batch(pods[3], bad, ctx)

    def test_negative_budget_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        with pytest.raises(ps.HostError, match="budget"):
            policy.propose_moves(state, graph, -1)

    def test_oversized_budget_is_host_error_not_overflow_error(self):
        # Round-2 review (R1.4 partial): budget was auto-extracted by PyO3 and
        # leaked a raw builtins.OverflowError for 2**80.
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        with pytest.raises(ps.HostError, match="budget.*i64"):
            policy.propose_moves(state, graph, 2**80)

    def test_bool_budget_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        with pytest.raises(ps.HostError, match="budget.*bool"):
            policy.propose_moves(state, graph, True)

    def test_limit_knobs_are_host_errors_not_overflow_errors(self):
        # Round-2 review (R1.4 partial): the fuel/memory knobs had the same
        # raw-OverflowError exposure on every entry point.
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        with pytest.raises(ps.HostError, match="fuel.*u64"):
            policy.score(pods[3], nodes[0], ctx, fuel=2**80)
        with pytest.raises(ps.HostError, match="fuel.*u64"):
            policy.score(pods[3], nodes[0], ctx, fuel=-1)
        with pytest.raises(ps.HostError, match="memory.*bool"):
            policy.score_batch(pods[3], nodes, ctx, memory=True)
        with pytest.raises(ps.HostError, match="memory.*u64"):
            policy.select_node(pods[3], nodes, ctx, memory=2**80)
        with pytest.raises(ps.HostError, match="fuel.*u64"):
            policy.propose_moves(state, graph, 1, fuel=-1)
        # Valid explicit knobs still work.
        assert isinstance(policy.score(pods[3], nodes[0], ctx, fuel=10_000_000), float)

    def test_out_of_domain_resource_rejected(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        bad = [dict(n) for n in nodes]
        bad[0]["cpu_capacity"] = 2**41  # over the spec 4.1 domain bound
        ctx = dict(ctx, nodes=bad)
        with pytest.raises(ps.HostError, match="4.1"):
            policy.score_batch(pods[3], bad, ctx)

    def test_select_node_tie_break_lowest_id(self):
        source = (
            'POLICY_NAME: str = "const"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    return 1.0\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n    return out\n"
        )
        policy = ps.Policy(source)
        nodes, pods, graph, ctx, state = make_cluster()
        shuffled = [nodes[3], nodes[1], nodes[4], nodes[2]]
        best, scores = policy.select_node(pods[3], shuffled, ctx)
        assert best == 1  # equal scores: lowest node id wins, order-independent
        assert scores == [1.0, 1.0, 1.0, 1.0]

    def test_select_node_max_score_wins(self):
        policy = load_policy("least_allocated")
        nodes, pods, graph, ctx, state = make_cluster()
        best, scores = policy.select_node(pods[3], nodes, ctx)
        expected = min(i for i in range(len(nodes)) if scores[i] == max(scores))
        assert best == nodes[expected]["id"]
