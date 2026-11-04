"""Hypothesis strategies generating cluster states over the spec §4.1 host input domain.

These define the *input domain* over which the risk gates (C8-scoped) quantify. Field
values span the **full host domain** the boundary validators enforce
(`bundle::validate_score_inputs` / `validate_propose_inputs`, spec §4.1): resource
integers up to 2^40 (allocation may exceed capacity — the domain does not forbid it),
watts up to 1e6, latencies up to 1e7 ms, rates/payloads up to 1e9, `now_ms` up to 2^53,
tier 0–15, service/app up to 2^32−1. All index conventions (dense ids, equal-length
columns, valid cross-references) are maintained — anything else is rejected at the host
boundary as a host error before the policy runs, so it is outside the totality claim.

Cluster *shapes* are sampled small (≤ GEN_MAX_NODES/PODS/EDGES per example — exhaustive
size coverage is impractical per candidate; the domain allows up to 512 nodes /
16 384 pods / 65 536 edges) and the degenerate/extreme shapes are covered
deterministically by `corner_clusters()`: a 1-node cluster, a 0-pod cluster, an
every-field-at-its-domain-bound cluster, and a wide 128-node cluster.
"""

from hypothesis import strategies as st

# Generator caps on cluster shape (per-example cost control, not the host domain).
GEN_MAX_NODES = 16
GEN_MAX_PODS = 32
GEN_MAX_EDGES = 64

# Spec §4.1 host input domain bounds (mirrors placementscript-core `bundle::domain`).
MAX_NODES = 512
MAX_PODS = 16_384
MAX_EDGES = 65_536
MAX_BUDGET = 16_384
MAX_RESOURCE = 1 << 40
MAX_TIER = 15
MAX_GROUP_ID = 2**32 - 1
MAX_WATTS = 1e6
MAX_LATENCY_MS = 1e7
MAX_RATE = 1e9
MAX_PAYLOAD_KB = 1e9
MAX_NOW_MS = 2**53

_RESOURCE = st.integers(min_value=0, max_value=MAX_RESOURCE)
_GROUP = st.integers(min_value=0, max_value=MAX_GROUP_ID)
_WATTS = st.floats(min_value=0.0, max_value=MAX_WATTS, allow_nan=False, allow_infinity=False)
_LATENCY = st.floats(min_value=0.0, max_value=MAX_LATENCY_MS, allow_nan=False, allow_infinity=False)
_RATE = st.floats(min_value=0.0, max_value=MAX_RATE, allow_nan=False, allow_infinity=False)
_PAYLOAD = st.floats(min_value=0.0, max_value=MAX_PAYLOAD_KB, allow_nan=False, allow_infinity=False)


@st.composite
def node_states(draw, node_id: int, n_nodes: int) -> dict:
    return {
        "id": node_id,
        "tier": draw(st.integers(min_value=0, max_value=MAX_TIER)),
        # Allocation is *not* bounded by capacity in the host domain (§4.1): an
        # overcommitted node is a legal input and policies must stay total on it.
        "cpu_capacity": draw(_RESOURCE),
        "cpu_allocated": draw(_RESOURCE),
        "mem_capacity": draw(_RESOURCE),
        "mem_allocated": draw(_RESOURCE),
        # power_max is likewise not required to exceed power_idle.
        "power_idle_w": draw(_WATTS),
        "power_max_w": draw(_WATTS),
        "suspended": draw(st.booleans()),
        "suspendable": draw(st.booleans()),
        "latency_to_ms": [0.0 if j == node_id else draw(_LATENCY) for j in range(n_nodes)],
    }


@st.composite
def pods(draw, pod_id: int, n_nodes: int) -> dict:
    return {
        "id": pod_id,
        "service": draw(_GROUP),
        "app": draw(_GROUP),
        "cpu_request": draw(_RESOURCE),
        "mem_request": draw(_RESOURCE),
        "stateful": draw(st.booleans()),
        "has_pdb": draw(st.booleans()),
        "image_mb": draw(_RESOURCE),
        "current_node": draw(st.integers(min_value=-1, max_value=n_nodes - 1)),
    }


def _assemble(nodes: list, pod_list: list, graph: dict, now_ms: int, budget: int) -> dict:
    ctx = {"nodes": nodes, "pods": pod_list, "graph": graph, "now_ms": now_ms}
    pending = pod_list[0] if pod_list else {
        "id": 0,
        "service": 0,
        "app": 0,
        "cpu_request": 1,
        "mem_request": 1,
        "stateful": False,
        "has_pdb": False,
        "image_mb": 1,
        "current_node": -1,
    }
    return {
        "nodes": nodes,
        "pods": pod_list,
        "graph": graph,
        "ctx": ctx,
        "state": {"nodes": nodes, "pods": pod_list},
        "pending_pod": pending,
        "budget": budget,
    }


@st.composite
def clusters(draw) -> dict:
    """A full, consistent cluster: nodes, pods, call graph, ctx, state, and a pending
    pod to score. Ids are dense and index-aligned (spec §4 conventions); the pod count
    may be 0 (legal per §4.1)."""
    n_nodes = draw(st.integers(min_value=1, max_value=GEN_MAX_NODES))
    n_pods = draw(st.integers(min_value=0, max_value=GEN_MAX_PODS))
    nodes = [draw(node_states(i, n_nodes)) for i in range(n_nodes)]
    pod_list = [draw(pods(p, n_nodes)) for p in range(n_pods)]
    n_edges = draw(st.integers(min_value=0, max_value=GEN_MAX_EDGES)) if n_pods else 0
    pod_ids = st.integers(min_value=0, max_value=max(n_pods - 1, 0))
    graph = {
        "src": [draw(pod_ids) for _ in range(n_edges)],
        "dst": [draw(pod_ids) for _ in range(n_edges)],
        "rate": [draw(_RATE) for _ in range(n_edges)],
        "payload_kb": [draw(_PAYLOAD) for _ in range(n_edges)],
    }
    pending = draw(st.sampled_from(pod_list)) if pod_list else None
    out = _assemble(
        nodes,
        pod_list,
        graph,
        now_ms=draw(st.integers(min_value=0, max_value=MAX_NOW_MS)),
        budget=draw(st.integers(min_value=0, max_value=MAX_BUDGET)),
    )
    if pending is not None:
        out["pending_pod"] = pending
    return out


def _plain_node(i: int, n_nodes: int, **over) -> dict:
    d = {
        "id": i,
        "tier": 0,
        "cpu_capacity": 4000,
        "cpu_allocated": 1000,
        "mem_capacity": 8192,
        "mem_allocated": 2048,
        "power_idle_w": 10.0,
        "power_max_w": 100.0,
        "suspended": False,
        "suspendable": True,
        "latency_to_ms": [0.0 if j == i else 1.0 for j in range(n_nodes)],
    }
    d.update(over)
    return d


def _plain_pod(p: int, current_node: int, **over) -> dict:
    d = {
        "id": p,
        "service": p,
        "app": 0,
        "cpu_request": 100,
        "mem_request": 128,
        "stateful": False,
        "has_pdb": False,
        "image_mb": 50,
        "current_node": current_node,
    }
    d.update(over)
    return d


def corner_clusters() -> dict[str, dict]:
    """Deterministic domain-corner clusters, checked explicitly by every gate run
    (spec §4.1): the extremes property sampling is unlikely to hit."""
    empty_graph = {"src": [], "dst": [], "rate": [], "payload_kb": []}

    one_node = _assemble(
        [_plain_node(0, 1)], [_plain_pod(0, 0)], empty_graph, now_ms=0, budget=0
    )

    zero_pods = _assemble(
        [_plain_node(i, 3) for i in range(3)], [], empty_graph, now_ms=0, budget=MAX_BUDGET
    )

    hi, lo = MAX_RESOURCE, 0
    max_fields_nodes = [
        _plain_node(
            0,
            2,
            tier=MAX_TIER,
            cpu_capacity=hi,
            cpu_allocated=hi,  # fully (over)committed
            mem_capacity=hi,
            mem_allocated=hi,
            power_idle_w=MAX_WATTS,
            power_max_w=0.0,  # max < idle is a legal input
            suspended=True,
            latency_to_ms=[0.0, MAX_LATENCY_MS],
        ),
        _plain_node(
            1,
            2,
            cpu_capacity=lo,
            cpu_allocated=lo,
            mem_capacity=lo,
            mem_allocated=lo,
            power_idle_w=0.0,
            power_max_w=MAX_WATTS,
            latency_to_ms=[MAX_LATENCY_MS, 0.0],
        ),
    ]
    max_fields_pods = [
        _plain_pod(
            0,
            1,
            service=MAX_GROUP_ID,
            app=MAX_GROUP_ID,
            cpu_request=hi,
            mem_request=hi,
            image_mb=hi,
            stateful=True,
            has_pdb=True,
        ),
        _plain_pod(1, -1, cpu_request=0, mem_request=0, image_mb=0),
    ]
    max_fields = _assemble(
        max_fields_nodes,
        max_fields_pods,
        {
            "src": [0, 1],
            "dst": [1, 0],
            "rate": [MAX_RATE, 0.0],
            "payload_kb": [MAX_PAYLOAD_KB, 0.0],
        },
        now_ms=MAX_NOW_MS,
        budget=MAX_BUDGET,
    )

    n = 128  # wide cluster (larger shapes than the sampler's per-example cap)
    wide = _assemble(
        [_plain_node(i, n) for i in range(n)],
        [_plain_pod(p, p % n) for p in range(64)],
        {
            "src": list(range(63)),
            "dst": [p + 1 for p in range(63)],
            "rate": [1.0] * 63,
            "payload_kb": [1.0] * 63,
        },
        now_ms=0,
        budget=64,
    )

    return {"one_node": one_node, "zero_pods": zero_pods, "max_fields": max_fields, "wide": wide}
