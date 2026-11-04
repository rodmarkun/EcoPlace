"""Registration, topology, load, replay, and richness contracts for new dev fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolution.evaluation import load_task_evaluation_config
from evolution.ladder import load_ladder_config
from sim.actuation import PlacementScriptProposer
from sim.engine import run_episode
from sim.scenarios.dev_medium import DEV_MEDIUM_CLUSTER_SEEDS, medium_cluster
from sim.scenarios.dev_realistic import (
    DEV_ALIBABA_HIER_ID,
    DEV_ALIBABA_HIER_SEEDS,
    DEV_ALIBABA_ZOO_ID,
    DEV_ALIBABA_ZOO_SEEDS,
    DEV_MEDIUM_HOT_ID,
    DEV_MEDIUM_HOT_SEEDS,
    REGISTRATIONS,
    build_registered_scenario,
    verify_realistic_manifest,
)
from sim.scheduler import PlacementScriptScorer
from sim.spec import config_sha256, load_episode_config

ROOT = Path(__file__).resolve().parent.parent
TASK_CONFIGS = {
    DEV_ALIBABA_HIER_ID: ROOT / "evolution" / "task" / "config_dev_alibaba_hier.json",
    DEV_ALIBABA_ZOO_ID: ROOT / "evolution" / "task" / "config_dev_alibaba_zoo.json",
    DEV_MEDIUM_HOT_ID: ROOT / "evolution" / "task" / "config_dev_medium_hot.json",
}
RICHNESS_DIR = ROOT / "experiments" / "dev_scenarios_realistic_20260809"


@pytest.mark.parametrize("registration", REGISTRATIONS)
def test_manifest_rebuilds_every_config_topology_and_workload_closure(registration):
    manifest = verify_realistic_manifest(registration.scenario_id)
    config = load_episode_config(registration.path)

    assert manifest["data_role"] == "development-only-quarantined"
    assert manifest["config_sha256"] == config_sha256(config)
    assert manifest["file"]["sha256"] == hashlib.sha256(
        registration.path.read_bytes()
    ).hexdigest()
    assert manifest["workload_builder"]["provenance_label"] == (
        "calibration-informed-synthetic-workload"
    )
    assert json.loads(manifest["workload_builder"]["closure_json"])[
        "provenance_label"
    ] == "calibration-informed-synthetic-workload"


def test_scenario_ids_seeds_and_task_configs_are_explicit_and_quarantined():
    matrices = (DEV_ALIBABA_HIER_SEEDS, DEV_ALIBABA_ZOO_SEEDS, DEV_MEDIUM_HOT_SEEDS)
    all_new = {seed for matrix in matrices for seed in matrix}

    assert len(all_new) == sum(map(len, matrices))
    assert not all_new & set(DEV_MEDIUM_CLUSTER_SEEDS)
    assert all(isinstance(seed, int) and seed > 0 for seed in all_new)
    for scenario_id, path in TASK_CONFIGS.items():
        raw = json.loads(path.read_text())
        task = load_task_evaluation_config(path)
        ladder = load_ladder_config(path)
        assert raw["data_role"] == "development-only"
        assert task.scenarios[0].scenario_id == scenario_id
        assert ladder.l1.scenario_ids == (scenario_id,)
        assert not {"acceptance", "private", "holdout"} & {
            part.lower() for part in Path(raw["scenarios"][0]["path"]).parts
        }


def test_hierarchical_and_zoo_topology_labels_and_shapes_are_exact():
    hier, _, hier_topology = build_registered_scenario(DEV_ALIBABA_HIER_ID)
    zoo, _, zoo_topology = build_registered_scenario(DEV_ALIBABA_ZOO_ID)

    assert hier_topology is not None and hier_topology.family == "hierarchical-tiered"
    assert hier_topology.evidence_label == "hierarchical-tiered synthetic scenario"
    assert len(hier.cluster.nodes) == 24
    assert zoo_topology is not None and zoo_topology.family == "topology-zoo-derived"
    assert zoo_topology.evidence_label == (
        "empirically grounded network-structure generalization"
    )
    assert zoo_topology.scale_class == "exploratory"
    assert len(zoo.cluster.nodes) == 20
    assert len(zoo_topology.edges) == 30
    assert all(topology.external_validity_statement == "no placement external-validity claim"
               for topology in (hier_topology, zoo_topology))
    assert {node.tier for node in hier.cluster.nodes} == {0, 1, 2}
    assert {node.tier for node in zoo.cluster.nodes} == {0, 1, 2}


def test_hot_scenario_reuses_medium_cluster_and_adds_absorbable_flash_load():
    hier, _, _ = build_registered_scenario(DEV_ALIBABA_HIER_ID)
    hot, _, _ = build_registered_scenario(DEV_MEDIUM_HOT_ID)

    assert hot.cluster == medium_cluster().cluster
    assert hot.hpa.enabled and hot.hpa.sync_period_s == 5.0
    assert hot.power.suspend_after_idle_s == 8.0
    assert sum(t.arrival.integrated_rate(hot.cutoff_s) for t in hot.workload.templates) > (
        1.5 * sum(t.arrival.integrated_rate(hier.cutoff_s) for t in hier.workload.templates)
    )
    for template in hot.workload.templates:
        assert template.arrival.rate_at(26.0) > template.arrival.rate_at(24.0)
        assert template.arrival.rate_at(39.0) < template.arrival.rate_at(37.0)


@pytest.mark.parametrize("registration", REGISTRATIONS)
def test_one_registered_seed_replays_bitwise(registration):
    import placementscript as ps

    config = load_episode_config(registration.path)
    policy = ps.Policy((ROOT / "runtime" / "policies" / "kube_least_allocated.ps").read_text())
    first = run_episode(
        config,
        registration.evaluation_seeds[0],
        PlacementScriptScorer(policy),
        PlacementScriptProposer(policy),
    )
    second = run_episode(
        config,
        registration.evaluation_seeds[0],
        PlacementScriptScorer(policy),
        PlacementScriptProposer(policy),
    )
    assert first.digest() == second.digest()


@pytest.mark.parametrize("registration", REGISTRATIONS)
def test_sealed_richness_evidence_is_current_and_passes(registration):
    slug = registration.scenario_id.removeprefix("dev-").replace("-", "_")
    path = RICHNESS_DIR / f"richness_{slug}.json"
    sealed = json.loads(path.read_text())
    payload = {
        key: value for key, value in sealed.items() if key not in {"seal", "timings_not_sealed"}
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    assert sealed["seal"] == hashlib.sha256(
        ("dev-realistic-scenario-richness-v1:" + canonical).encode()
    ).hexdigest()
    assert sealed["scenario_manifest"] == verify_realistic_manifest(
        registration.scenario_id
    )
    assert sealed["all_checks_pass"]
    assert sealed["summary"]["baseline_distinct_feasible"] >= 4
    assert sealed["summary"]["mutant_distinct_feasible"] >= 12
    assert sealed["summary"]["mutants_strictly_between_baselines"] >= 1
    assert sealed["summary"]["mutants_valid"] >= 60
    assert sealed["timings_not_sealed"]["median_l2_wall_within_budget"]
