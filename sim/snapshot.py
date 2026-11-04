"""Snapshot/reset API + frozen-snapshot library export (TODO 2.6).

Two complementary mechanisms, chosen deliberately (documented in ``sim/README.md``):

1. **Declarative placement snapshots** (this module): the complete placement-relevant
   state at an instant — a spec-§4-shaped ``ClusterCtx`` over truth (nodes with live
   allocations and power states, dense-remapped pods, the pod-level call graph) plus the
   scheduler-view overlay (assumed reservations from in-flight bindings folded into node
   allocations and pod placements) and the pending-pod list in **scheduler-queue order**
   (activeQ order first, then backoff pods by expiry) with per-pod eligibility times.
   These are what L0 consumes (plan §3.3: "analytical replay on frozen simulation
   snapshots") and what the frozen-snapshot library exports. Snapshot payloads are
   **deep-frozen** (dicts become mapping proxies, lists become tuples); ``to_dict()``
   returns a fully-plain deep copy for serialization/scoring.
2. **Replay-based full-state restore** (``sim.engine.Simulation.start``/``advance``):
   episodes are bitwise-deterministically replayable within the pinned environment, so
   the *complete* simulator state at time t — including in-flight request coroutines,
   which no declarative serialization can honestly capture in Python — is reconstructed
   exactly by replaying to t under the same (config, seed). Boundary semantics are
   documented in ``sim.engine``: state "at t" = after events < t, before events at t.

A ``SnapshotLibrary`` bundles the episode config with its snapshots, serializes to
canonical JSON with an **embedded whole-payload digest** verified on every load (config,
snapshots, and metadata are all covered — not just the config hash), and ``load`` also
accepts the externally recorded digest for frozen-library pinning.
"""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sim.runtime import (
    BINDING,
    PENDING,
    PULLING,
    READY,
    STARTING,
    TERMINATING,
)
from sim.spec import (
    MAX_GRAPH_EDGES,
    MAX_GROUP_ID,
    MAX_LATENCY_MS,
    MAX_NODES,
    MAX_NOW_MS,
    MAX_PAYLOAD_KB,
    MAX_PODS,
    MAX_RATE,
    MAX_RESOURCE,
    MAX_WATTS,
    EpisodeConfig,
    config_sha256,
    episode_config_from_json,
    episode_config_to_json,
)

if TYPE_CHECKING:
    from sim.runtime import ClusterRuntime
    from sim.scheduler import SchedulerEmulator


def _deep_freeze(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return types.MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list | tuple):
        return tuple(_deep_freeze(x) for x in obj)
    return obj


def _deep_thaw(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _deep_thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [_deep_thaw(x) for x in obj]
    return obj


def _host_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _host_float(name: str, value: Any, maximum: float) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= maximum:
        raise ValueError(f"{name} must be a float in [0, {maximum:g}]")
    return value


@dataclass(frozen=True)
class Snapshot:
    """Frozen placement-relevant state at one instant of one episode.

    - ``ctx``: ClusterCtx shape (nodes, pods, graph, now_ms), truth **plus** the
      scheduler's assumed reservations (in-flight bindings are committed work: their
      resources are allocated and their pods placed on the reserved node).
    - ``pending_dense_ids``: PENDING pods only (never BINDING/backoff-misclassified), in
      scheduler-queue order — these are the decisions L0 replays.
    - ``pending_eligible_s``: per pending pod, the earliest time it can next be
      scheduled (now for activeQ pods; backoff expiry for backed-off pods).
    - ``binding_dense_ids``: pods whose binding is in flight (already overlaid onto
      ``ctx`` allocations; listed so consumers can tell them from bound pods).
    """

    taken_at_s: float
    ctx: Any  # deep-frozen ClusterCtx mapping
    pending_dense_ids: tuple[int, ...]
    pending_eligible_s: tuple[float, ...] = ()
    binding_dense_ids: tuple[int, ...] = ()
    pod_phases: tuple[str, ...] = ()  # aligned with ctx.pods; READY alone serves
    # Physical power state stays outside the PlacementScript ABI. ``ctx.suspended``
    # means placement-unavailable (power OR planned cordon); L0 must not price an
    # awake cordoned node at suspended power. Empty is the legacy fallback.
    power_suspended_by_node: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.taken_at_s, int | float)
            or isinstance(self.taken_at_s, bool)
            or not math.isfinite(float(self.taken_at_s))
            or self.taken_at_s < 0
        ):
            raise ValueError("taken_at_s must be a finite nonnegative number")
        object.__setattr__(self, "ctx", _deep_freeze(_deep_thaw(self.ctx)))
        object.__setattr__(self, "pending_dense_ids", tuple(self.pending_dense_ids))
        object.__setattr__(self, "pending_eligible_s", tuple(self.pending_eligible_s))
        object.__setattr__(self, "binding_dense_ids", tuple(self.binding_dense_ids))
        object.__setattr__(self, "pod_phases", tuple(self.pod_phases))
        object.__setattr__(self, "power_suspended_by_node", tuple(self.power_suspended_by_node))
        required_ctx = {"nodes", "pods", "graph", "now_ms"}
        if not isinstance(self.ctx, Mapping) or not required_ctx <= set(self.ctx):
            raise ValueError(f"ctx must contain {sorted(required_ctx)}")
        nodes, pods, graph = self.ctx["nodes"], self.ctx["pods"], self.ctx["graph"]
        if self.power_suspended_by_node and (
            len(self.power_suspended_by_node) != len(nodes)
            or any(type(value) is not bool for value in self.power_suspended_by_node)
        ):
            raise ValueError("power_suspended_by_node must be bool and align with nodes")
        if not isinstance(nodes, tuple) or not isinstance(pods, tuple):
            raise ValueError("ctx nodes/pods must be sequences")
        if not 1 <= len(nodes) <= MAX_NODES:
            raise ValueError(f"ctx nodes must contain 1..{MAX_NODES} entries")
        if len(pods) > MAX_PODS:
            raise ValueError(f"ctx pods exceed the host-domain maximum {MAX_PODS}")
        now_ms = self.ctx["now_ms"]
        if (
            not isinstance(now_ms, int)
            or isinstance(now_ms, bool)
            or not 0 <= now_ms <= MAX_NOW_MS
            or now_ms != int(float(self.taken_at_s) * 1000.0)
        ):
            raise ValueError("ctx.now_ms must match taken_at_s")
        if [node.get("id") for node in nodes] != list(range(len(nodes))):
            raise ValueError("ctx node ids must be dense and ordered")
        if [pod.get("id") for pod in pods] != list(range(len(pods))):
            raise ValueError("ctx pod ids must be dense and ordered")
        for node in nodes:
            if len(node.get("latency_to_ms", ())) != len(nodes):
                raise ValueError("each ctx node needs one latency per node")
            _host_int("node.tier", node.get("tier"), 0, 15)
            _host_int("node.cpu_capacity", node.get("cpu_capacity"), 0, MAX_RESOURCE)
            _host_int("node.cpu_allocated", node.get("cpu_allocated"), 0, MAX_RESOURCE)
            _host_int("node.mem_capacity", node.get("mem_capacity"), 0, MAX_RESOURCE)
            _host_int("node.mem_allocated", node.get("mem_allocated"), 0, MAX_RESOURCE)
            # Scheduler-view snapshots may legitimately exceed capacity while an
            # external truth allocation and a stale assumed reservation overlap. The
            # spec caps capacity at MAX_RESOURCE/2, so the exact sum still fits the
            # host field; preserving it keeps NodeResourcesFit unschedulability honest.
            idle = _host_float("node.power_idle_w", node.get("power_idle_w"), MAX_WATTS)
            maximum = _host_float("node.power_max_w", node.get("power_max_w"), MAX_WATTS)
            if maximum < idle:
                raise ValueError("ctx node power_max_w must be >= power_idle_w")
            if not isinstance(node.get("suspended"), bool) or not isinstance(
                node.get("suspendable"), bool
            ):
                raise ValueError("ctx node suspended/suspendable must be bool")
            for latency in node["latency_to_ms"]:
                _host_float("node.latency_to_ms", latency, MAX_LATENCY_MS)
        for pod in pods:
            current = pod.get("current_node")
            if (
                not isinstance(current, int)
                or isinstance(current, bool)
                or not (-1 <= current < len(nodes))
            ):
                raise ValueError("ctx pod current_node must be -1 or a valid node id")
            _host_int("pod.service", pod.get("service"), 0, MAX_GROUP_ID)
            _host_int("pod.app", pod.get("app"), 0, MAX_GROUP_ID)
            for field_name in ("cpu_request", "mem_request", "image_mb"):
                _host_int(f"pod.{field_name}", pod.get(field_name), 0, MAX_RESOURCE)
            if not isinstance(pod.get("stateful"), bool) or not isinstance(
                pod.get("has_pdb"), bool
            ):
                raise ValueError("ctx pod stateful/has_pdb must be bool")
        if len(self.pod_phases) != len(pods):
            raise ValueError("pod_phases must align with ctx pods")
        allowed_phases = {PENDING, BINDING, PULLING, STARTING, READY, TERMINATING}
        if any(phase not in allowed_phases for phase in self.pod_phases):
            raise ValueError("pod_phases contains an unknown lifecycle phase")
        for name, ids in (
            ("pending_dense_ids", self.pending_dense_ids),
            ("binding_dense_ids", self.binding_dense_ids),
        ):
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in ids):
                raise ValueError(f"{name} must contain nonnegative integer ids")
        pending = set(self.pending_dense_ids)
        binding = set(self.binding_dense_ids)
        if len(pending) != len(self.pending_dense_ids) or len(binding) != len(
            self.binding_dense_ids
        ):
            raise ValueError("pending/binding ids must be unique")
        if pending & binding:
            raise ValueError("pending and binding ids must be disjoint")
        if any(dense >= len(pods) for dense in pending | binding):
            raise ValueError("pending/binding id is outside ctx pods")
        if any(self.pod_phases[dense] != PENDING for dense in pending):
            raise ValueError("pending ids must refer to PENDING lifecycle phases")
        if any(self.pod_phases[dense] != BINDING for dense in binding):
            raise ValueError("binding ids must refer to BINDING lifecycle phases")
        if any(pods[dense]["current_node"] != -1 for dense in pending):
            raise ValueError("pending pods must be unplaced")
        if any(pods[dense]["current_node"] < 0 for dense in binding):
            raise ValueError("binding pods must carry their assumed placement")
        if any(
            not isinstance(v, int | float)
            or isinstance(v, bool)
            or not math.isfinite(float(v))
            or v < 0
            for v in self.pending_eligible_s
        ):
            raise ValueError("pending_eligible_s must contain finite nonnegative times")
        if len(self.pending_eligible_s) != len(self.pending_dense_ids):
            raise ValueError("pending_eligible_s must align with pending_dense_ids")
        if any(v < self.taken_at_s for v in self.pending_eligible_s):
            raise ValueError("pending eligibility cannot precede the snapshot")
        if not isinstance(graph, Mapping) or set(graph) != {
            "src",
            "dst",
            "rate",
            "payload_kb",
        }:
            raise ValueError("ctx graph has an invalid shape")
        edge_count = len(graph["src"])
        if edge_count > MAX_GRAPH_EDGES:
            raise ValueError(f"ctx graph exceeds the host-domain maximum {MAX_GRAPH_EDGES}")
        if any(len(graph[key]) != edge_count for key in ("dst", "rate", "payload_kb")):
            raise ValueError("ctx graph columns must have equal length")
        if any(
            not isinstance(dense, int) or isinstance(dense, bool) or not 0 <= dense < len(pods)
            for dense in (*graph["src"], *graph["dst"])
        ):
            raise ValueError("ctx graph endpoints must be valid dense pod ids")
        for value in graph["rate"]:
            _host_float("ctx graph rate", value, MAX_RATE)
        for value in graph["payload_kb"]:
            _host_float("ctx graph payload_kb", value, MAX_PAYLOAD_KB)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "taken_at_s": self.taken_at_s,
            "ctx": _deep_thaw(self.ctx),
            "pending_dense_ids": list(self.pending_dense_ids),
            "pending_eligible_s": list(self.pending_eligible_s),
            "binding_dense_ids": list(self.binding_dense_ids),
            "pod_phases": list(self.pod_phases),
        }
        if self.power_suspended_by_node:
            out["power_suspended_by_node"] = list(self.power_suspended_by_node)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Snapshot:
        return Snapshot(
            taken_at_s=d["taken_at_s"],
            ctx=d["ctx"],
            pending_dense_ids=tuple(d["pending_dense_ids"]),
            pending_eligible_s=tuple(d.get("pending_eligible_s", ())),
            binding_dense_ids=tuple(d.get("binding_dense_ids", ())),
            pod_phases=tuple(d["pod_phases"]),
            power_suspended_by_node=tuple(d.get("power_suspended_by_node", ())),
        )


def take_snapshot(
    cluster: ClusterRuntime,
    config: EpisodeConfig,
    now_s: float,
    scheduler: SchedulerEmulator | None = None,
) -> Snapshot:
    """Capture the scheduler-relevant state: truth (API-server view) overlaid with the
    scheduler's assumed reservations, and the pending queue in scheduler order.

    Without ``scheduler`` (bare-truth capture), BINDING pods appear unplaced and are
    excluded from ``pending`` — they belong to neither the pending decisions nor the
    committed placement; callers wanting scheduler-complete snapshots must pass the
    emulator (the engine always does).
    """
    from sim.scheduler import build_ctx  # local import: avoid a module cycle

    # Pods still awaiting a placement decision are projected into the call-graph
    # expansion (sim.scheduler._pod_level_graph) exactly as the DES scoring path
    # projects the single pod in its cycle. L0 narrows this per-pod when it replays a
    # decision, so a graph-aware scorer is not handed a constant view on either rung.
    prospective = tuple(
        p for p in sorted(cluster.live_pods(), key=lambda p: p.uid) if p.phase == PENDING
    )
    ctx, uid_to_dense, _ = build_ctx(
        cluster,
        config.workload,
        cache=None,
        now_s=now_s,
        cutoff_s=config.cutoff_s,
        prospective=prospective,
    )
    ctx_nodes = {n["id"]: n for n in ctx["nodes"]}
    binding_dense: list[int] = []
    if scheduler is not None:
        # Overlay assumed reservations: an in-flight binding holds its resources.
        for uid, (node_id, cpu, mem) in sorted(scheduler.cache.assumed.items()):
            if uid not in uid_to_dense:
                continue  # pod terminated while its unreserve is still in flight
            dense = uid_to_dense[uid]
            ctx["pods"][dense]["current_node"] = node_id
            ctx_nodes[node_id]["cpu_allocated"] += cpu
            ctx_nodes[node_id]["mem_allocated"] += mem
            binding_dense.append(dense)
        pending_uids = scheduler.pending_in_queue_order()
        pending = tuple(uid_to_dense[uid] for uid, _ in pending_uids if uid in uid_to_dense)
        eligible = tuple(t for uid, t in pending_uids if uid in uid_to_dense)
    else:
        pending = tuple(
            uid_to_dense[p.uid]
            for p in sorted(cluster.live_pods(), key=lambda p: p.uid)
            if p.phase == PENDING
        )
        eligible = tuple(now_s for _ in pending)
        # Bare truth has no assumed-placement overlay. Lifecycle remains visible in
        # pod_phases, but binding_dense_ids is reserved for BINDING pods whose
        # scheduler reservation has actually been folded into ctx.
        binding_dense = []
    phases = [""] * len(uid_to_dense)
    for pod in cluster.live_pods():
        phases[uid_to_dense[pod.uid]] = pod.phase
    physical_suspended = tuple(not node.awake for node in cluster.nodes)
    placement_unavailable = tuple(node["suspended"] for node in ctx["nodes"])
    return Snapshot(
        taken_at_s=now_s,
        ctx=ctx,
        pending_dense_ids=pending,
        pending_eligible_s=eligible,
        binding_dense_ids=tuple(binding_dense),
        pod_phases=tuple(phases),
        power_suspended_by_node=(
            physical_suspended if physical_suspended != placement_unavailable else ()
        ),
    )


def initial_snapshot(config: EpisodeConfig) -> Snapshot:
    """The cold decision problem at t=0: nothing bound, every initial replica pending in
    creation order — the canonical policy-discriminating L0 instance."""
    spec = config.cluster
    n = len(spec.nodes)
    nodes = []
    for node in spec.nodes:
        cls = spec.node_class_of(node.id)
        nodes.append(
            {
                "id": node.id,
                "tier": node.tier,
                "cpu_capacity": node.cpu_capacity,
                "cpu_allocated": 0,
                "mem_capacity": node.mem_capacity,
                "mem_allocated": 0,
                "power_idle_w": float(cls.power_idle_w),
                "power_max_w": float(cls.power_max_w),
                "suspended": node.initially_suspended,
                "suspendable": cls.suspendable,
                "latency_to_ms": [float(spec.latency_ms[node.id][j]) for j in range(n)],
            }
        )
    pods = []
    for service in config.workload.services:
        if service.activation_s > 0.0:
            continue
        for _ in range(service.initial_replicas):
            pods.append(
                {
                    "id": len(pods),
                    "service": service.id,
                    "app": service.app,
                    "cpu_request": service.cpu_request,
                    "mem_request": service.mem_request,
                    "stateful": service.stateful,
                    "has_pdb": service.pdb_protected,
                    "image_mb": service.image_mb,
                    "current_node": -1,
                }
            )
    ctx = {
        "nodes": nodes,
        "pods": pods,
        "graph": {"src": [], "dst": [], "rate": [], "payload_kb": []},
        "now_ms": 0,
    }
    return Snapshot(
        taken_at_s=0.0,
        ctx=ctx,
        pending_dense_ids=tuple(range(len(pods))),
        pending_eligible_s=tuple(0.0 for _ in pods),
        pod_phases=tuple(PENDING for _ in pods),
    )


@dataclass(frozen=True)
class SnapshotLibrary:
    """A frozen set of snapshots + the config they came from (feeds L0)."""

    config: EpisodeConfig
    snapshots: tuple[Snapshot, ...]
    meta: Any = field(default_factory=dict)  # deep-frozen provenance: seeds, episode ids

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        if not self.snapshots:
            raise ValueError("snapshot library must not be empty")
        if any(not isinstance(snapshot, Snapshot) for snapshot in self.snapshots):
            raise ValueError("snapshots must contain Snapshot objects")
        object.__setattr__(self, "meta", _deep_freeze(_deep_thaw(self.meta)))

    def _payload(self) -> dict[str, Any]:
        return {
            "config": json.loads(episode_config_to_json(self.config)),
            "config_sha256": config_sha256(self.config),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "meta": _deep_thaw(self.meta),
        }

    def to_json(self) -> str:
        """Canonical serialization: the payload wrapped with its own sha256 digest.
        ``load`` verifies the digest over the whole payload — config, snapshots, and
        metadata are all tamper-evident, not just the config hash."""
        payload_text = json.dumps(self._payload(), sort_keys=True, indent=1)
        digest = hashlib.sha256(payload_text.encode()).hexdigest()
        return json.dumps(
            {"payload_sha256": digest, "payload": json.loads(payload_text)},
            sort_keys=True,
            indent=1,
        )

    def digest(self) -> str:
        """Content digest of the full serialized library (record it when freezing)."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def save(self, path: str | Path) -> str:
        """Write the library; returns the content digest (record it when freezing)."""
        text = self.to_json()
        Path(path).write_text(text + "\n")
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def load(path: str | Path, expected_digest: str | None = None) -> SnapshotLibrary:
        """Load and verify. The embedded payload digest is always checked (any edit to
        config, snapshots, or metadata fails the load); pass ``expected_digest`` (the
        value recorded by ``save``) to additionally pin the exact frozen artifact."""
        text = Path(path).read_text()
        if expected_digest is not None:
            actual = hashlib.sha256(text.rstrip("\n").encode()).hexdigest()
            if actual != expected_digest:
                raise ValueError(
                    f"{path}: library digest mismatch (expected {expected_digest[:12]}, "
                    f"got {actual[:12]}) — not the frozen artifact"
                )
        d = json.loads(text)
        if set(d) != {"payload_sha256", "payload"}:
            raise ValueError(f"{path}: not a snapshot library file")
        payload_text = json.dumps(d["payload"], sort_keys=True, indent=1)
        if hashlib.sha256(payload_text.encode()).hexdigest() != d["payload_sha256"]:
            raise ValueError(f"{path}: payload digest mismatch — library corrupted or edited")
        p = d["payload"]
        config = episode_config_from_json(json.dumps(p["config"]))
        if config_sha256(config) != p["config_sha256"]:
            raise ValueError(f"{path}: config hash mismatch — library corrupted or edited")
        return SnapshotLibrary(
            config=config,
            snapshots=tuple(Snapshot.from_dict(s) for s in p["snapshots"]),
            meta=p.get("meta", {}),
        )


def export_library(
    config: EpisodeConfig, snapshots: list[Snapshot], meta: dict[str, Any]
) -> SnapshotLibrary:
    """Assemble a library from episode-produced snapshots. ``meta`` must record
    provenance (root seeds, scorer used to generate the episode, episode digests)."""
    if not snapshots:
        raise ValueError("refusing to export an empty snapshot library")
    return SnapshotLibrary(config=config, snapshots=tuple(snapshots), meta=meta)
