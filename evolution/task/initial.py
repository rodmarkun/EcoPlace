# ruff: noqa: F821
# least_allocated — Phase-5 seed policy.  Identity stays outside EVOLVE-BLOCKs;
# Shinka operators may change only the policy implementation below.
POLICY_NAME: str = "least_allocated"
POLICY_VERSION: str = "1.0.0"

# EVOLVE-BLOCK-START
def free_fraction_after(pod: Pod, node: NodeState) -> float:
    cpu_cap = float(max(node.cpu_capacity, 1))
    mem_cap = float(max(node.mem_capacity, 1))
    cpu_free = cpu_cap - float(node.cpu_allocated + pod.cpu_request)
    mem_free = mem_cap - float(node.mem_allocated + pod.mem_request)
    cpu_frac = max(cpu_free, 0.0) / cpu_cap
    mem_frac = max(mem_free, 0.0) / mem_cap
    return (cpu_frac + mem_frac) / 2.0


def cpu_util(node: NodeState) -> float:
    return float(node.cpu_allocated) / float(max(node.cpu_capacity, 1))


def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:
    return free_fraction_after(pod, node)


def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:
    out: list[Move] = []
    if budget <= 0 or len(state.nodes) < 2:
        return out
    hi = -1
    hi_u = -1.0
    lo = -1
    lo_u = 2.0
    for node in state.nodes:
        if not node.suspended:
            u = cpu_util(node)
            if u > hi_u:
                hi_u = u
                hi = node.id
            if u < lo_u:
                lo_u = u
                lo = node.id
    if hi < 0 or lo < 0 or hi == lo or hi_u - lo_u < 0.25:
        return out
    moved = 0
    for pod in state.pods:
        if moved < budget and pod.current_node == hi:
            if not pod.stateful and not pod.has_pdb:
                out.append(move_pod(pod.id, lo))
                moved += 1
    return out
# EVOLVE-BLOCK-END
