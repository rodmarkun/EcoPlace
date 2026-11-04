"""Seeded topology families with machine-enforced evidence labels (TODO 4.2)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.tracking import spawn_rng
from sim.scenarios.sources import STRUCTURE, SourceRegistry
from sim.spec import ClusterSpec, NodeClassSpec, NodeSpec

HIERARCHICAL_LABEL = "hierarchical-tiered synthetic scenario"
ZOO_LABEL = "empirically grounded network-structure generalization"
BA_LABEL = "synthetic stress test"
NO_EXTERNAL_VALIDITY = "no placement external-validity claim"
ROLES = ("cloud-core", "regional-fog", "access-edge", "device-edge")
_TOPOLOGY_TAG = "scenario-topology-v1:"
_SCALE_TAG = "scenario-scale-policy-v1:"
_SCALE_AUTHORITY = object()
_TOPOLOGY_AUTHORITY = object()
G0_DATA_PATH = Path(__file__).resolve().parent / "data" / "g0_capacity_table.json"
PRODUCTION_G0_FILE_SHA256: str | None = None
DEVELOPMENT_ELIBACKBONE_STRUCTURE_SHA256 = (
    "6ada03e16cd9fb77c2df5faa4f4432c87bd38b93c56c2d28ddac302ef38c0462"
)
DEVELOPMENT_ELIBACKBONE_UPSTREAM_BLOB = "783ec61087205a3f5546a94ecff9f41bee1dc91e"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ScalePolicy:
    g0_table_id: str
    g0_table_sha256: str
    confirmatory_max_nodes: int
    exploratory_max_nodes: int
    g0_reserve_fraction: float
    g0_status: str
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            "g0_table_id": self.g0_table_id,
            "g0_table_sha256": self.g0_table_sha256,
            "confirmatory_max_nodes": self.confirmatory_max_nodes,
            "exploratory_max_nodes": self.exploratory_max_nodes,
            "g0_reserve_fraction": self.g0_reserve_fraction,
            "g0_status": self.g0_status,
        }

    def verify(self) -> None:
        if self._authority is not _SCALE_AUTHORITY:
            raise ValueError("ScalePolicy must be loaded from exact G0 capacity-table bytes")
        if not isinstance(self.g0_table_id, str) or not self.g0_table_id:
            raise ValueError("ScalePolicy needs a G0 capacity-table id")
        if len(self.g0_table_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.g0_table_sha256
        ):
            raise ValueError("ScalePolicy needs the frozen G0 table SHA-256")
        if (
            not isinstance(self.confirmatory_max_nodes, int)
            or isinstance(self.confirmatory_max_nodes, bool)
            or not isinstance(self.exploratory_max_nodes, int)
            or isinstance(self.exploratory_max_nodes, bool)
            or not 1 <= self.confirmatory_max_nodes <= self.exploratory_max_nodes <= 512
        ):
            raise ValueError("ScalePolicy requires 1 <= confirmatory <= exploratory <= 512")
        if (
            self.g0_status != "PASS"
            or not isinstance(self.g0_reserve_fraction, int | float)
            or isinstance(self.g0_reserve_fraction, bool)
            or not math.isfinite(self.g0_reserve_fraction)
            or self.g0_reserve_fraction < 0.3
        ):
            raise ValueError("confirmatory scale requires a passing G0 table with >=30% reserve")
        expected = hashlib.sha256((_SCALE_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("ScalePolicy seal mismatch")


def _load_scale_policy_bytes(g0_table_payload: bytes) -> ScalePolicy:
    """Recompute scale eligibility from an exact literal capacity-table export."""
    if not isinstance(g0_table_payload, bytes) or not g0_table_payload:
        raise ValueError("G0 capacity table must be non-empty exact bytes")
    try:
        table = json.loads(g0_table_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("G0 capacity table must be UTF-8 JSON") from exc
    required = {
        "schema",
        "table_id",
        "ceilings",
        "scale_trials",
    }
    if set(table) != required or table["schema"] != "g0-capacity-table-v1":
        raise ValueError("G0 capacity scale export has unknown schema/fields")
    ceilings = table["ceilings"]
    trials = table["scale_trials"]
    if not isinstance(ceilings, list) or not ceilings or not isinstance(trials, list) or not trials:
        raise ValueError("G0 capacity table needs non-empty ceilings and scale trials")
    names: set[str] = set()
    reserves: list[float] = []
    for ceiling in ceilings:
        if not isinstance(ceiling, dict) or set(ceiling) != {
            "name",
            "capacity",
            "projected_use",
        }:
            raise ValueError("G0 ceiling rows have unknown fields")
        name = ceiling["name"]
        capacity = ceiling["capacity"]
        projected = ceiling["projected_use"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(capacity, int | float)
            or isinstance(capacity, bool)
            or not math.isfinite(capacity)
            or capacity <= 0
            or not isinstance(projected, int | float)
            or isinstance(projected, bool)
            or not math.isfinite(projected)
            or projected < 0
        ):
            raise ValueError("invalid G0 ceiling row")
        names.add(name)
        reserves.append((capacity - projected) / capacity)
    eligible_nodes: list[int] = []
    for trial in trials:
        if not isinstance(trial, dict) or set(trial) != {
            "nodes",
            "p90_runtime_s",
            "runtime_budget_s",
        }:
            raise ValueError("G0 scale-trial rows have unknown fields")
        nodes = trial["nodes"]
        runtime = trial["p90_runtime_s"]
        budget = trial["runtime_budget_s"]
        if (
            not isinstance(nodes, int)
            or isinstance(nodes, bool)
            or not 1 <= nodes <= 512
            or not isinstance(runtime, int | float)
            or isinstance(runtime, bool)
            or not math.isfinite(runtime)
            or runtime <= 0
            or not isinstance(budget, int | float)
            or isinstance(budget, bool)
            or not math.isfinite(budget)
            or budget <= 0
        ):
            raise ValueError("invalid G0 scale-trial row")
        if runtime <= budget:
            eligible_nodes.append(nodes)
    reserve = min(reserves)
    status = "PASS" if reserve >= 0.3 and eligible_nodes else "FAIL"
    if status != "PASS":
        raise ValueError("G0 table does not establish >=30% reserve and a feasible scale")
    payload = {
        "g0_table_id": table["table_id"],
        "g0_table_sha256": hashlib.sha256(g0_table_payload).hexdigest(),
        "confirmatory_max_nodes": max(eligible_nodes),
        "exploratory_max_nodes": 512,
        "g0_reserve_fraction": reserve,
        "g0_status": status,
    }
    seal = hashlib.sha256((_SCALE_TAG + _canonical(payload)).encode()).hexdigest()
    policy = ScalePolicy(**payload, seal=seal, _authority=_SCALE_AUTHORITY)
    policy.verify()
    return policy


def load_scale_policy() -> ScalePolicy:
    """Load only the registered production G0 table; fail closed until G0 exists."""
    if PRODUCTION_G0_FILE_SHA256 is None or not G0_DATA_PATH.is_file():
        raise ValueError("registered production G0 capacity table is unavailable")
    payload = G0_DATA_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PRODUCTION_G0_FILE_SHA256:
        raise ValueError("registered production G0 capacity table hash mismatch")
    return _load_scale_policy_bytes(payload)


def _load_scale_policy_fixture_bytes(g0_table_payload: bytes) -> ScalePolicy:
    """Test-only parser for synthetic numerical capacity-table fixtures."""
    return _load_scale_policy_bytes(g0_table_payload)


@dataclass(frozen=True)
class RoleBinding:
    role: str
    runtime_tier: int
    node_class: NodeClassSpec
    cpu_capacity: int
    mem_capacity: int
    synthetic_capacity: bool = True

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if self.runtime_tier not in (0, 1, 2):
            raise ValueError("runtime tier must be 0, 1, or 2")
        if not isinstance(self.cpu_capacity, int) or self.cpu_capacity <= 0:
            raise ValueError("role cpu_capacity must be a positive integer")
        if not isinstance(self.mem_capacity, int) or self.mem_capacity <= 0:
            raise ValueError("role mem_capacity must be a positive integer")
        if not self.synthetic_capacity:
            raise ValueError("Phase-4 generated topology capacities must be labeled synthetic")


@dataclass(frozen=True)
class TopologyArtifact:
    schema: str
    family: str
    evidence_label: str
    external_validity_statement: str
    seed: int
    node_roles: tuple[str, ...]
    edges: tuple[tuple[int, int, float], ...]
    source_id: str | None
    source_content_sha256: str | None
    transform: str
    scale_class: str
    scale_policy_sha256: str
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("digest", "_authority")
        }

    def verify(self, policy: ScalePolicy) -> None:
        if self._authority is not _TOPOLOGY_AUTHORITY:
            raise ValueError("TopologyArtifact must be issued by a registered family generator")
        policy.verify()
        if self.schema != "scenario-topology-v1":
            raise ValueError("unknown topology artifact schema")
        if self.scale_policy_sha256 != policy.seal:
            raise ValueError("topology artifact was generated under a different ScalePolicy")
        if not self.node_roles or any(role not in ROLES for role in self.node_roles):
            raise ValueError("topology node roles are missing or invalid")
        n = len(self.node_roles)
        cap = (
            policy.confirmatory_max_nodes
            if self.scale_class == "confirmatory"
            else policy.exploratory_max_nodes
        )
        if self.scale_class not in ("confirmatory", "exploratory") or n > cap:
            raise ValueError(f"{self.scale_class} topology of {n} nodes exceeds cap {cap}")
        expected_label = {
            "hierarchical-tiered": HIERARCHICAL_LABEL,
            "topology-zoo-derived": ZOO_LABEL,
            "barabasi-albert": BA_LABEL,
        }.get(self.family)
        if expected_label is None or self.evidence_label != expected_label:
            raise ValueError("topology evidence label does not match its family")
        if self.family == "topology-zoo-derived":
            if self.source_id != "topology-zoo" or self.source_content_sha256 is None:
                raise ValueError("Topology-Zoo-derived artifacts require exact source provenance")
            if len(self.source_content_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in self.source_content_sha256
            ):
                raise ValueError("Topology Zoo source hash must be a lowercase SHA-256")
        elif self.source_id is not None or self.source_content_sha256 is not None:
            raise ValueError("synthetic topology families may not advertise empirical sources")
        if self.external_validity_statement != NO_EXTERNAL_VALIDITY:
            raise ValueError("generated topologies may not claim placement external validity")
        seen: set[tuple[int, int]] = set()
        adjacency = [set() for _ in range(n)]
        for u, v, latency in self.edges:
            if not 0 <= u < v < n or (u, v) in seen:
                raise ValueError("topology edges must be unique canonical undirected pairs")
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("topology edge latency must be finite and > 0")
            seen.add((u, v))
            adjacency[u].add(v)
            adjacency[v].add(u)
        reached, frontier = {0}, [0]
        while frontier:
            current = frontier.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        if len(reached) != n:
            raise ValueError("topology graph must be connected")
        expected = hashlib.sha256(
            (_TOPOLOGY_TAG + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("topology artifact digest mismatch")


def _seal_topology(payload: dict[str, object], policy: ScalePolicy) -> TopologyArtifact:
    digest = hashlib.sha256((_TOPOLOGY_TAG + _canonical(payload)).encode()).hexdigest()
    artifact = TopologyArtifact(
        **payload, digest=digest, _authority=_TOPOLOGY_AUTHORITY  # type: ignore[arg-type]
    )
    artifact.verify(policy)
    return artifact


def _scale_class(n: int, policy: ScalePolicy, requested: str) -> str:
    policy.verify()
    if requested not in ("confirmatory", "exploratory"):
        raise ValueError("scale_class must be confirmatory or exploratory")
    cap = (
        policy.confirmatory_max_nodes
        if requested == "confirmatory"
        else policy.exploratory_max_nodes
    )
    if n > cap:
        raise ValueError(f"requested {requested} topology size {n} exceeds G0 cap {cap}")
    return requested


def hierarchical_topology(
    role_counts: tuple[int, int, int, int],
    seed: int,
    policy: ScalePolicy,
    scale_class: str = "confirmatory",
) -> TopologyArtifact:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    if len(role_counts) != 4 or any(not isinstance(v, int) or v <= 0 for v in role_counts):
        raise ValueError("hierarchical topology needs four positive role counts")
    node_roles = tuple(
        role for role, count in zip(ROLES, role_counts, strict=True) for _ in range(count)
    )
    n = len(node_roles)
    rng = spawn_rng(seed, "scenario", "topology", "hierarchical")
    offsets = np.cumsum((0, *role_counts))
    edges: set[tuple[int, int, float]] = set()
    # Connect each role level to the preceding level, then add a chain inside the
    # cloud core. Parent selection is seeded and stable.
    for level in range(1, 4):
        parents = list(range(int(offsets[level - 1]), int(offsets[level])))
        for child in range(int(offsets[level]), int(offsets[level + 1])):
            parent = parents[int(rng.integers(0, len(parents)))]
            latency = (1.0, 4.0, 12.0)[level - 1]
            edges.add((min(parent, child), max(parent, child), latency))
    for node in range(1, role_counts[0]):
        edges.add((node - 1, node, 0.5))
    payload = {
        "schema": "scenario-topology-v1",
        "family": "hierarchical-tiered",
        "evidence_label": HIERARCHICAL_LABEL,
        "external_validity_statement": NO_EXTERNAL_VALIDITY,
        "seed": seed,
        "node_roles": node_roles,
        "edges": tuple(sorted(edges)),
        "source_id": None,
        "source_content_sha256": None,
        "transform": (
            "four structural roles mapped later to three runtime tiers; "
            "synthetic links/tiers/capacities"
        ),
        "scale_class": _scale_class(n, policy, scale_class),
        "scale_policy_sha256": policy.seal,
    }
    return _seal_topology(payload, policy)


def barabasi_albert_topology(
    n: int,
    m: int,
    seed: int,
    policy: ScalePolicy,
    scale_class: str = "confirmatory",
) -> TopologyArtifact:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    if not 1 <= m < n:
        raise ValueError("Barabasi-Albert requires 1 <= m < n")
    rng = spawn_rng(seed, "scenario", "topology", "barabasi-albert")
    edges: set[tuple[int, int, float]] = set()
    degree = [0] * n
    # Seed clique of m+1 nodes.
    for u in range(m + 1):
        for v in range(u + 1, m + 1):
            edges.add((u, v, 5.0))
            degree[u] += 1
            degree[v] += 1
    for node in range(m + 1, n):
        weights = np.asarray(degree[:node], dtype=float)
        probabilities = weights / weights.sum()
        targets = rng.choice(node, size=m, replace=False, p=probabilities)
        for target in sorted(int(value) for value in targets):
            edges.add((target, node, 5.0))
            degree[target] += 1
            degree[node] += 1
    # Synthetic tier assignment from graph degree quartiles, explicitly not evidence.
    order = sorted(range(n), key=lambda node: (-degree[node], node))
    roles = ["device-edge"] * n
    cuts = (max(1, n // 10), max(2, n // 3), max(3, 2 * n // 3))
    for rank, node in enumerate(order):
        roles[node] = (
            "cloud-core"
            if rank < cuts[0]
            else "regional-fog"
            if rank < cuts[1]
            else "access-edge"
            if rank < cuts[2]
            else "device-edge"
        )
    payload = {
        "schema": "scenario-topology-v1",
        "family": "barabasi-albert",
        "evidence_label": BA_LABEL,
        "external_validity_statement": NO_EXTERNAL_VALIDITY,
        "seed": seed,
        "node_roles": tuple(roles),
        "edges": tuple(sorted(edges)),
        "source_id": None,
        "source_content_sha256": None,
        "transform": (
            "seeded preferential attachment; degree-derived synthetic roles and "
            "synthetic capacities"
        ),
        "scale_class": _scale_class(n, policy, scale_class),
        "scale_policy_sha256": policy.seal,
    }
    return _seal_topology(payload, policy)


def parse_undirected_graphml(
    graphml: bytes,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Parse canonical undirected simple structure without attaching an evidence label."""
    root = ET.fromstring(graphml)
    graph = next((element for element in root.iter() if element.tag.endswith("graph")), None)
    if graph is None or graph.attrib.get("edgedefault", "undirected") != "undirected":
        raise ValueError("GraphML input must contain one undirected graph")
    raw_ids = tuple(
        sorted(element.attrib["id"] for element in graph if element.tag.endswith("node"))
    )
    if len(raw_ids) != len(set(raw_ids)) or not raw_ids:
        raise ValueError("GraphML node ids must be non-empty and unique")
    ids = {raw: index for index, raw in enumerate(raw_ids)}
    pairs: set[tuple[int, int]] = set()
    for element in graph:
        if not element.tag.endswith("edge"):
            continue
        try:
            u, v = ids[element.attrib["source"]], ids[element.attrib["target"]]
        except KeyError as exc:
            raise ValueError("GraphML edge references an unknown node") from exc
        if u == v:
            raise ValueError("GraphML self loops are not admitted")
        pair = (min(u, v), max(u, v))
        if pair in pairs:
            raise ValueError("GraphML parallel edges are not admitted")
        pairs.add(pair)
    return raw_ids, tuple(sorted(pairs))


def topology_zoo_from_graphml(
    graphml: bytes,
    source_content_sha256: str,
    registry: SourceRegistry,
    seed: int,
    policy: ScalePolicy,
    scale_class: str = "confirmatory",
    synthetic_link_latency_ms: float = 5.0,
) -> TopologyArtifact:
    if hashlib.sha256(graphml).hexdigest() != source_content_sha256:
        raise ValueError("Topology Zoo GraphML content hash mismatch")
    registry.require_payload("topology-zoo", STRUCTURE, source_content_sha256)
    return _topology_zoo_artifact(
        graphml,
        source_content_sha256,
        seed,
        policy,
        scale_class,
        synthetic_link_latency_ms,
        (
            "operator-published adjacency only; degree-derived tiers, capacities, "
            "and link latency are synthetic"
        ),
    )


def topology_zoo_development_from_graphml(
    graphml: bytes,
    seed: int,
    policy: ScalePolicy,
    synthetic_link_latency_ms: float = 5.0,
) -> TopologyArtifact:
    """Issue the one pinned, structure-only EliBackbone development derivative.

    This does not alter or bypass the production source registry: the ordinary
    :func:`topology_zoo_from_graphml` remains fail-closed while that registry entry is
    unavailable.  The allowed digest is module-owned, the output is exploratory, and
    the T10 label continues to disclose synthetic tiers/capacities/link latencies and
    no placement external-validity claim.
    """
    digest = hashlib.sha256(graphml).hexdigest()
    if digest != DEVELOPMENT_ELIBACKBONE_STRUCTURE_SHA256:
        raise ValueError("unregistered development Topology Zoo structure derivative")
    return _topology_zoo_artifact(
        graphml,
        digest,
        seed,
        policy,
        "exploratory",
        synthetic_link_latency_ms,
        (
            "EliBackbone structure-only derivative of upstream Git blob "
            f"{DEVELOPMENT_ELIBACKBONE_UPSTREAM_BLOB}; degree-derived tiers, "
            "capacities, and link latency are synthetic; development-only"
        ),
    )


def _topology_zoo_artifact(
    graphml: bytes,
    source_content_sha256: str,
    seed: int,
    policy: ScalePolicy,
    scale_class: str,
    synthetic_link_latency_ms: float,
    transform: str,
) -> TopologyArtifact:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    if not math.isfinite(synthetic_link_latency_ms) or synthetic_link_latency_ms <= 0:
        raise ValueError("synthetic link latency must be finite and > 0")
    raw_ids, parsed_pairs = parse_undirected_graphml(graphml)
    pairs = set(parsed_pairs)
    degree = [0] * len(raw_ids)
    for u, v in pairs:
        degree[u] += 1
        degree[v] += 1
    order = sorted(range(len(raw_ids)), key=lambda node: (-degree[node], raw_ids[node]))
    roles = ["device-edge"] * len(raw_ids)
    n = len(raw_ids)
    for rank, node in enumerate(order):
        roles[node] = ROLES[min(3, (4 * rank) // n)]
    payload = {
        "schema": "scenario-topology-v1",
        "family": "topology-zoo-derived",
        "evidence_label": ZOO_LABEL,
        "external_validity_statement": NO_EXTERNAL_VALIDITY,
        "seed": seed,
        "node_roles": tuple(roles),
        "edges": tuple((u, v, float(synthetic_link_latency_ms)) for u, v in sorted(pairs)),
        "source_id": "topology-zoo",
        "source_content_sha256": source_content_sha256,
        "transform": transform,
        "scale_class": _scale_class(n, policy, scale_class),
        "scale_policy_sha256": policy.seal,
    }
    return _seal_topology(payload, policy)


def materialize_cluster(
    artifact: TopologyArtifact,
    policy: ScalePolicy,
    bindings: tuple[RoleBinding, ...],
) -> ClusterSpec:
    artifact.verify(policy)
    by_role = {binding.role: binding for binding in bindings}
    if set(by_role) != set(ROLES) or len(bindings) != len(ROLES):
        raise ValueError("bindings must map every structural role exactly once")
    classes = tuple(binding.node_class for binding in bindings)
    if len({cls.name for cls in classes}) != len(classes):
        raise ValueError("role bindings require distinct node-class names")
    nodes = tuple(
        NodeSpec(
            id=node_id,
            node_class=by_role[role].node_class.name,
            tier=by_role[role].runtime_tier,
            cpu_capacity=by_role[role].cpu_capacity,
            mem_capacity=by_role[role].mem_capacity,
        )
        for node_id, role in enumerate(artifact.node_roles)
    )
    n = len(nodes)
    distance = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        distance[i][i] = 0.0
    for u, v, latency in artifact.edges:
        distance[u][v] = distance[v][u] = latency
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return ClusterSpec(
        classes=classes,
        nodes=nodes,
        latency_ms=tuple(tuple(row) for row in distance),
    )
