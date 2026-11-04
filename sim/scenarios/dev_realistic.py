"""Three sealed, quarantined calibration-informed development scenarios.

These fixtures are development workload probes, never acceptance or confirmatory data:

``dev-alibaba-hier``
    Alibaba calibration-informed workload on a 24-node T-class hierarchical topology.
``dev-alibaba-zoo``
    The same workload family on the 20-node EliBackbone Topology Zoo adjacency.  Only
    adjacency is retained; T10 tiers, capacities and link latencies are synthetic.
``dev-medium-hot``
    The existing ``dev-medium`` cluster with a hotter calibrated shape and an explicit
    synthetic flash interval that the enabled HPA can absorb.

Every source, topology transform, builder seed, evaluation seed, exact config byte,
typed workload closure and generator byte is bound by one per-scenario manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from analysis.tracking import derive_seed
from sim.scenarios.alibaba_workload import (
    BUILDER_VERSION,
    CALIBRATION_SHA256,
    CALL_GRAPH_SAMPLE_SHA256,
    AlibabaWorkloadBuildConfig,
    CalibrationInformedWorkload,
    build_calibration_informed_workload,
)
from sim.scenarios.artifacts import (
    seal_calibration_informed_synthetic_workload,
    workload_model_to_json,
)
from sim.scenarios.dev_medium import medium_cluster
from sim.scenarios.sources import load_source_registry
from sim.scenarios.topology import (
    DEVELOPMENT_ELIBACKBONE_STRUCTURE_SHA256,
    DEVELOPMENT_ELIBACKBONE_UPSTREAM_BLOB,
    RoleBinding,
    TopologyArtifact,
    _load_scale_policy_fixture_bytes,
    hierarchical_topology,
    materialize_cluster,
    topology_zoo_development_from_graphml,
)
from sim.spec import (
    ClusterSpec,
    EpisodeConfig,
    HpaParams,
    NodeClassSpec,
    NodeSpec,
    PowerParams,
    RebalancerParams,
    SchedulerParams,
    config_sha256,
    det,
    load_episode_config,
    save_episode_config,
)

HERE = Path(__file__).resolve().parent
DEV_DIR = HERE / "dev"
ELIBACKBONE_PATH = DEV_DIR / "elibackbone_structure.graphml"
DATA_ROLE = "development-only-quarantined"
MANIFEST_TAG = "dev-realistic-scenario-manifest-v1:"
ROOT_SEED = 20260817
HORIZON_S = 65.0
ARRIVAL_CUTOFF_S = 53.0

DEV_ALIBABA_HIER_ID = "dev-alibaba-hier"
DEV_ALIBABA_ZOO_ID = "dev-alibaba-zoo"
DEV_MEDIUM_HOT_ID = "dev-medium-hot"

DEV_ALIBABA_HIER_PATH = DEV_DIR / "alibaba_hier.json"
DEV_ALIBABA_ZOO_PATH = DEV_DIR / "alibaba_zoo.json"
DEV_MEDIUM_HOT_PATH = DEV_DIR / "medium_hot.json"

DEV_ALIBABA_HIER_SEEDS = (1000003, 1000033, 1000037)
DEV_ALIBABA_ZOO_SEEDS = (1000039, 1000081, 1000099)
DEV_MEDIUM_HOT_SEEDS = (1000117, 1000121, 1000133)

_DEVELOPMENT_SCALE_TABLE = json.dumps(
    {
        "schema": "g0-capacity-table-v1",
        "table_id": "dev-realistic-quarantined-scale-table-v1",
        "ceilings": [
            {"name": "developer-wall-s", "capacity": 100.0, "projected_use": 50.0}
        ],
        "scale_trials": [
            {"nodes": 32, "p90_runtime_s": 3.0, "runtime_budget_s": 4.0}
        ],
    },
    sort_keys=True,
).encode()


@dataclass(frozen=True)
class DevelopmentScenarioRegistration:
    scenario_id: str
    path: Path
    manifest_path: Path
    evaluation_seeds: tuple[int, ...]
    topology_kind: str
    topology_seed: int
    permutation_seed: int
    workload: AlibabaWorkloadBuildConfig


def _seed(scenario_id: str, component: str) -> int:
    return derive_seed(ROOT_SEED, "sim", "scenarios", scenario_id, component)


REGISTRATIONS = (
    DevelopmentScenarioRegistration(
        DEV_ALIBABA_HIER_ID,
        DEV_ALIBABA_HIER_PATH,
        DEV_DIR / "alibaba_hier.manifest.json",
        DEV_ALIBABA_HIER_SEEDS,
        "hierarchical-tiered",
        _seed(DEV_ALIBABA_HIER_ID, "topology"),
        _seed(DEV_ALIBABA_HIER_ID, "node-permutation"),
        AlibabaWorkloadBuildConfig(
            seed=_seed(DEV_ALIBABA_HIER_ID, "workload-builder"),
            arrival_rate_scale=0.30,
        ),
    ),
    DevelopmentScenarioRegistration(
        DEV_ALIBABA_ZOO_ID,
        DEV_ALIBABA_ZOO_PATH,
        DEV_DIR / "alibaba_zoo.manifest.json",
        DEV_ALIBABA_ZOO_SEEDS,
        "topology-zoo-derived",
        _seed(DEV_ALIBABA_ZOO_ID, "topology"),
        _seed(DEV_ALIBABA_ZOO_ID, "node-permutation"),
        AlibabaWorkloadBuildConfig(
            seed=_seed(DEV_ALIBABA_ZOO_ID, "workload-builder"),
            arrival_rate_scale=0.30,
        ),
    ),
    DevelopmentScenarioRegistration(
        DEV_MEDIUM_HOT_ID,
        DEV_MEDIUM_HOT_PATH,
        DEV_DIR / "medium_hot.manifest.json",
        DEV_MEDIUM_HOT_SEEDS,
        "dev-medium-cluster-reuse",
        _seed(DEV_MEDIUM_HOT_ID, "topology-reuse"),
        _seed(DEV_MEDIUM_HOT_ID, "node-permutation-unused"),
        AlibabaWorkloadBuildConfig(
            seed=_seed(DEV_MEDIUM_HOT_ID, "workload-builder"),
            arrival_rate_scale=0.48,
            flash_start_s=25.0,
            flash_end_s=38.0,
            flash_multiplier=1.65,
        ),
    ),
)
_BY_ID = {item.scenario_id: item for item in REGISTRATIONS}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scale_policy():
    # The topology generators require a scale authority.  This literal table is sealed
    # into every manifest and is development-only; load_scale_policy() remains closed.
    return _load_scale_policy_fixture_bytes(_DEVELOPMENT_SCALE_TABLE)


def _bindings() -> tuple[RoleBinding, ...]:
    return (
        RoleBinding(
            "cloud-core",
            0,
            NodeClassSpec("dev-cloud-core", 110.0, 340.0),
            3600,
            8192,
        ),
        RoleBinding(
            "regional-fog",
            1,
            NodeClassSpec(
                "dev-regional-fog",
                60.0,
                200.0,
                suspend_energy_j=1655.0,
                suspend_latency_s=15.0,
                wake_energy_j=23683.0,
                wake_latency_s=182.0,
                suspended_power_w=8.0,
            ),
            16000,
            12288,
        ),
        RoleBinding(
            "access-edge",
            2,
            NodeClassSpec("dev-access-edge", 2.2, 5.4),
            4000,
            4096,
        ),
        RoleBinding(
            "device-edge",
            2,
            NodeClassSpec("dev-device-edge", 1.8, 4.4),
            4000,
            2048,
        ),
    )


def _permute_cluster(cluster: ClusterSpec, seed: int) -> ClusterSpec:
    order = sorted(
        range(len(cluster.nodes)),
        key=lambda node_id: hashlib.sha256(
            f"{seed}:dev-topology-node-permutation-v1:{node_id}".encode()
        ).digest(),
    )
    nodes = tuple(
        NodeSpec(
            new_id,
            cluster.nodes[old_id].node_class,
            cluster.nodes[old_id].tier,
            cluster.nodes[old_id].cpu_capacity,
            cluster.nodes[old_id].mem_capacity,
            cluster.nodes[old_id].initially_suspended,
        )
        for new_id, old_id in enumerate(order)
    )
    latency = tuple(
        tuple(cluster.latency_ms[old_i][old_j] for old_j in order) for old_i in order
    )
    return ClusterSpec(cluster.classes, nodes, latency)


def _generated_cluster(
    registration: DevelopmentScenarioRegistration,
) -> tuple[ClusterSpec, TopologyArtifact | None]:
    if registration.topology_kind == "dev-medium-cluster-reuse":
        return medium_cluster().cluster, None
    policy = _scale_policy()
    if registration.topology_kind == "hierarchical-tiered":
        artifact = hierarchical_topology(
            (2, 4, 8, 10), registration.topology_seed, policy, scale_class="exploratory"
        )
    elif registration.topology_kind == "topology-zoo-derived":
        artifact = topology_zoo_development_from_graphml(
            ELIBACKBONE_PATH.read_bytes(),
            registration.topology_seed,
            policy,
            synthetic_link_latency_ms=7.0,
        )
    else:  # pragma: no cover - registrations are module-owned constants
        raise ValueError(f"unknown development topology kind {registration.topology_kind}")
    cluster = materialize_cluster(artifact, policy, _bindings())
    return _permute_cluster(cluster, registration.permutation_seed), artifact


def _episode(
    cluster: ClusterSpec,
    build: CalibrationInformedWorkload,
    *,
    hot: bool,
) -> EpisodeConfig:
    return EpisodeConfig(
        cluster=cluster,
        workload=build.workload,
        horizon_s=HORIZON_S,
        arrival_cutoff_s=ARRIVAL_CUTOFF_S,
        scheduler=SchedulerParams(bind_latency=det(0.01)),
        hpa=HpaParams(
            enabled=True,
            sync_period_s=5.0 if hot else 7.0,
            target_utilization=0.48 if hot else 0.55,
        ),
        rebalancer=RebalancerParams(
            enabled=True,
            window_s=18.0,
            budget_per_window=3,
            unavailability_precharge_s=1.0,
            max_total_bytes_mb=384.0,
        ),
        power=PowerParams(suspend_after_idle_s=8.0),
        snapshot_times_s=(15.0, 31.0, 49.0),
        snapshot_cycles=(0, 1, 2, 3, 4, 5, 6, 7),
    )


def build_registered_scenario(
    scenario_id: str,
) -> tuple[EpisodeConfig, CalibrationInformedWorkload, TopologyArtifact | None]:
    """Pure deterministic builder for one module-owned scenario registration."""
    try:
        registration = _BY_ID[scenario_id]
    except KeyError as exc:
        raise KeyError(f"unknown realistic development scenario {scenario_id!r}") from exc
    registry = load_source_registry()
    workload = build_calibration_informed_workload(registration.workload, registry)
    cluster, topology = _generated_cluster(registration)
    config = _episode(
        cluster,
        workload,
        hot=scenario_id == DEV_MEDIUM_HOT_ID,
    )
    return config, workload, topology


def _topology_record(
    registration: DevelopmentScenarioRegistration,
    artifact: TopologyArtifact | None,
) -> dict[str, object]:
    if artifact is None:
        return {
            "family": "dev-medium-cluster-reuse",
            "evidence_label": "existing quarantined synthetic development cluster",
            "digest": config_sha256(medium_cluster()),
            "external_validity_statement": "no placement external-validity claim",
        }
    record: dict[str, object] = {
        "family": artifact.family,
        "evidence_label": artifact.evidence_label,
        "digest": artifact.digest,
        "external_validity_statement": artifact.external_validity_statement,
        "seed": artifact.seed,
        "transform": artifact.transform,
        "scale_class": artifact.scale_class,
        "scale_policy_sha256": artifact.scale_policy_sha256,
    }
    if artifact.family == "topology-zoo-derived":
        record["source"] = {
            "retained_path": str(ELIBACKBONE_PATH.relative_to(HERE.parents[1])),
            "retained_sha256": DEVELOPMENT_ELIBACKBONE_STRUCTURE_SHA256,
            "upstream_repository": "mroughan/InternetTopologyZoo",
            "upstream_path": "graphml/EliBackbone.graphml",
            "upstream_git_blob": DEVELOPMENT_ELIBACKBONE_UPSTREAM_BLOB,
            "scope": "adjacency only; tiers/capacities/link latencies synthetic",
        }
    return record


def _manifest_payload(registration: DevelopmentScenarioRegistration) -> dict[str, object]:
    config, build, topology = build_registered_scenario(registration.scenario_id)
    registry = load_source_registry()
    workload_model = seal_calibration_informed_synthetic_workload(
        config, registry, build=build
    )
    return {
        "schema": "evolved-continuum-dev-realistic-scenario-manifest-v1",
        "data_role": DATA_ROLE,
        "scenario_id": registration.scenario_id,
        "evaluation_seeds": list(registration.evaluation_seeds),
        "generator_root_seed": ROOT_SEED,
        "component_seeds": {
            "topology": registration.topology_seed,
            "node_permutation": registration.permutation_seed,
            "workload_builder": registration.workload.seed,
        },
        "horizon_s": HORIZON_S,
        "arrival_cutoff_s": ARRIVAL_CUTOFF_S,
        "generated_by": "sim/scenarios/dev_realistic.py (deterministic)",
        "generator_sha256": _sha256_file(Path(__file__)),
        "workload_builder": {
            "version": BUILDER_VERSION,
            "module_sha256": _sha256_file(HERE / "alibaba_workload.py"),
            "build_sha256": build.digest,
            "calibration_export_sha256": CALIBRATION_SHA256,
            "call_graph_export_sha256": CALL_GRAPH_SAMPLE_SHA256,
            "provenance_label": workload_model.provenance_label,
            "closure_sha256": workload_model.digest,
            "closure_json": workload_model_to_json(workload_model),
        },
        "topology": _topology_record(registration, topology),
        "scale_table_sha256": hashlib.sha256(_DEVELOPMENT_SCALE_TABLE).hexdigest(),
        "file": {
            "path": str(registration.path.relative_to(HERE.parents[1])),
            "sha256": _sha256_file(registration.path),
        },
        "config_sha256": config_sha256(load_episode_config(registration.path)),
        "quarantine_rule": (
            "development-only: never acceptance evidence, never public/private split, "
            "never face-validity, never confirmatory"
        ),
    }


def _seal(payload: dict[str, object]) -> str:
    return hashlib.sha256((MANIFEST_TAG + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )).encode()).hexdigest()


def verify_realistic_manifest(scenario_id: str) -> dict[str, object]:
    """Rebuild the scenario and every closure; reject any manifest/tree drift."""
    registration = _BY_ID[scenario_id]
    if not registration.manifest_path.is_file():
        raise FileNotFoundError(f"missing scenario manifest: {registration.manifest_path}")
    recorded = json.loads(registration.manifest_path.read_text())
    if not isinstance(recorded, dict):
        raise ValueError("development scenario manifest must be an object")
    payload = {key: value for key, value in recorded.items() if key != "seal"}
    if recorded.get("seal") != _seal(payload):
        raise ValueError("development scenario manifest seal mismatch")
    expected = _manifest_payload(registration)
    if payload != expected:
        differing = sorted(
            key for key in set(payload) | set(expected) if payload.get(key) != expected.get(key)
        )
        raise ValueError(f"development scenario manifest drifted from tree: {differing}")
    return recorded


def load_dev_alibaba_hier() -> EpisodeConfig:
    verify_realistic_manifest(DEV_ALIBABA_HIER_ID)
    return load_episode_config(DEV_ALIBABA_HIER_PATH)


def load_dev_alibaba_zoo() -> EpisodeConfig:
    verify_realistic_manifest(DEV_ALIBABA_ZOO_ID)
    return load_episode_config(DEV_ALIBABA_ZOO_PATH)


def load_dev_medium_hot() -> EpisodeConfig:
    verify_realistic_manifest(DEV_MEDIUM_HOT_ID)
    return load_episode_config(DEV_MEDIUM_HOT_PATH)


def main() -> None:
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    for registration in REGISTRATIONS:
        config, _build, _topology = build_registered_scenario(registration.scenario_id)
        save_episode_config(config, registration.path)
        payload = _manifest_payload(registration)
        registration.manifest_path.write_text(
            json.dumps({**payload, "seal": _seal(payload)}, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {registration.path} and {registration.manifest_path}")


if __name__ == "__main__":
    main()
