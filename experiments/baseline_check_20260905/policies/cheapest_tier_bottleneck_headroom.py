# ruff: noqa: F821
# cheapest_tier_bottleneck_headroom — hand-written check baseline (2026-09-05, checklist 5.4, v2).
#
# v1 (cheapest tier, then least-allocated by the *mean* free fraction) was infeasible on all
# 20 fresh grids (worst SLO 0.00-0.04). The root-807 champion spreads on the *bottleneck*
# resource, 1 - max(cpu_after, mem_after), weighted 20x. This variant keeps the tier rule and
# adopts that headroom term, with no power model beyond the tier ordering.
POLICY_NAME: str = "cheapest_tier_bottleneck_headroom"
POLICY_VERSION: str = "1.0.0"

def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:
    if node.suspended:
        return -1000000.0
    cpu_cap = float(max(node.cpu_capacity, 1))
    mem_cap = float(max(node.mem_capacity, 1))
    cpu_after = float(node.cpu_allocated + pod.cpu_request) / cpu_cap
    mem_after = float(node.mem_allocated + pod.mem_request) / mem_cap
    peak = max(cpu_after, mem_after)
    if peak > 1.0:
        return -1000000.0
    headroom = 1.0 - peak
    return -node.power_idle_w + headroom * 20.0

def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:
    out: list[Move] = []
    return out
