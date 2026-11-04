# ruff: noqa: F821
# root807_marginal_watts_headroom — the root-800000807 LLM champion's EVOLVE block, verbatim,
# scored as a fixed hand-written policy on all 20 confirmatory fresh grids (checklist 5.4, v3).
# If it is feasible and ~0.57-0.58 everywhere, the rule generalises beyond the root that found it.
POLICY_NAME: str = "root807_marginal_watts_headroom"
POLICY_VERSION: str = "1.0.0"

def incremental_power(pod: Pod, node: NodeState) -> float:
    cpu_cap = float(max(node.cpu_capacity, 1))
    mem_cap = float(max(node.mem_capacity, 1))
    cpu_after = float(node.cpu_allocated + pod.cpu_request) / cpu_cap
    mem_after = float(node.mem_allocated + pod.mem_request) / mem_cap
    peak = max(cpu_after, mem_after)
    dynamic = node.power_max_w - node.power_idle_w
    return node.power_idle_w + dynamic * peak

def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:
    if node.suspended:
        return -1000000.0
    cpu_cap = float(max(node.cpu_capacity, 1))
    mem_cap = float(max(node.mem_capacity, 1))
    cpu_after = float(node.cpu_allocated + pod.cpu_request) / cpu_cap
    mem_after = float(node.mem_allocated + pod.mem_request) / mem_cap
    if max(cpu_after, mem_after) > 1.0:
        return -1000000.0
    cost = incremental_power(pod, node)
    headroom = 1.0 - max(cpu_after, mem_after)
    return -cost + headroom * 20.0

def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:
    out: list[Move] = []
    return out
