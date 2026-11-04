"""``dev-medium-cluster`` quarantined development fixture: seal, registry, determinism.

The fixture exists to give the Phase-5 evolution loop a landscape with gradient; these
tests lock the contract it has to keep to be usable as *replayable development data*:
sealed bytes, an explicit seed matrix, deterministic replay, a task config that the
Phase-5 loader accepts, and strict separation from acceptance/frozen artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sim.scenarios.dev_medium as dev_medium
from evolution.evaluation import load_task_evaluation_config
from evolution.ladder import load_ladder_config
from sim.actuation import PlacementScriptProposer
from sim.engine import run_episode
from sim.scenarios import (
    DEV_MEDIUM_CLUSTER_ID,
    DEV_MEDIUM_CLUSTER_SEEDS,
    load_dev_medium_cluster,
    medium_cluster,
    verify_dev_medium_manifest,
)
from sim.scheduler import PlacementScriptScorer
from sim.spec import TIER_CLOUD, TIER_EDGE, TIER_FOG, config_sha256, load_episode_config

REPO = Path(__file__).resolve().parent.parent
TASK_CONFIG = REPO / "evolution" / "task" / "config_dev_medium.json"
EXISTING_TASK_CONFIG = REPO / "evolution" / "task" / "config.json"
FROZEN_FIXTURE_MANIFEST = REPO / "sim" / "fixtures" / "MANIFEST.json"


# ------------------------------------------------------------------ seal / registry
def test_manifest_seal_matches_fixture_and_generator_bytes():
    recorded = verify_dev_medium_manifest()

    assert recorded["scenario_id"] == DEV_MEDIUM_CLUSTER_ID == "dev-medium-cluster"
    assert recorded["data_role"] == "development-only-quarantined"
    assert len(recorded["seal"]) == 64
    fixture = dev_medium.DEV_MEDIUM_CLUSTER_PATH
    assert recorded["files"][fixture.name] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    generator = Path(dev_medium.__file__)
    assert recorded["generator_sha256"] == hashlib.sha256(generator.read_bytes()).hexdigest()
    assert recorded["config_sha256"] == config_sha256(load_episode_config(fixture))


def test_manifest_verification_fails_closed_on_a_tampered_payload(tmp_path, monkeypatch):
    recorded = json.loads(dev_medium.MANIFEST_PATH.read_text())
    tampered = dict(recorded)
    tampered["evaluation_seeds"] = [1, 2, 3]
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(dev_medium, "MANIFEST_PATH", path)

    with pytest.raises(ValueError, match="seal mismatch"):
        dev_medium.verify_manifest()


def test_manifest_verification_fails_closed_when_the_seal_is_resealed_over_drift(
    tmp_path, monkeypatch
):
    """Re-sealing an edited payload is still rejected: the tree is the authority."""
    recorded = json.loads(dev_medium.MANIFEST_PATH.read_text())
    payload = {key: value for key, value in recorded.items() if key != "seal"}
    payload["evaluation_seeds"] = [1, 2, 3]
    path = tmp_path / "MANIFEST.json"
    path.write_text(
        json.dumps({**payload, "seal": dev_medium._seal(payload)}, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(dev_medium, "MANIFEST_PATH", path)

    with pytest.raises(ValueError, match="drifted from the tree"):
        dev_medium.verify_manifest()


def test_missing_manifest_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_medium, "MANIFEST_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        dev_medium.verify_manifest()


def test_package_registry_exposes_the_scenario():
    assert DEV_MEDIUM_CLUSTER_ID in json.dumps(verify_dev_medium_manifest())
    assert load_dev_medium_cluster().horizon_s == medium_cluster().horizon_s
    assert config_sha256(load_dev_medium_cluster()) == config_sha256(medium_cluster())


def test_generator_is_a_pure_function_of_the_module():
    assert config_sha256(medium_cluster()) == config_sha256(medium_cluster())


# ------------------------------------------------------------------ quarantine rules
def test_fixture_is_not_registered_in_the_p1_frozen_fixture_manifest():
    """``sim/fixtures/**`` is frozen at P1; this development fixture must stay outside."""
    frozen = json.loads(FROZEN_FIXTURE_MANIFEST.read_text())
    assert not any("medium_cluster" in name for name in frozen["files"])
    assert dev_medium.DEV_DIR.resolve() != (REPO / "sim" / "fixtures" / "dev").resolve()


def test_seed_matrix_is_explicit_unique_and_disjoint_from_the_existing_dev_grid():
    assert len(DEV_MEDIUM_CLUSTER_SEEDS) == len(set(DEV_MEDIUM_CLUSTER_SEEDS))
    assert 3 <= len(DEV_MEDIUM_CLUSTER_SEEDS) <= 5
    assert all(isinstance(seed, int) and seed > 0 for seed in DEV_MEDIUM_CLUSTER_SEEDS)
    existing = {
        seed
        for scenario in json.loads(EXISTING_TASK_CONFIG.read_text())["scenarios"]
        for seed in scenario["seeds"]
    }
    assert not existing & set(DEV_MEDIUM_CLUSTER_SEEDS)
    frozen = json.loads(FROZEN_FIXTURE_MANIFEST.read_text())
    assert not set(frozen["acceptance_seed_matrix"]) & set(DEV_MEDIUM_CLUSTER_SEEDS)


def test_task_config_variant_is_development_only_and_leaves_the_existing_one_untouched():
    raw = json.loads(TASK_CONFIG.read_text())
    assert raw["data_role"] == "development-only"
    assert [scenario["id"] for scenario in raw["scenarios"]] == [DEV_MEDIUM_CLUSTER_ID]
    assert raw["scenarios"][0]["seeds"] == list(DEV_MEDIUM_CLUSTER_SEEDS)
    path = raw["scenarios"][0]["path"]
    assert not {"acceptance", "private", "holdout"} & set(Path(path).parts)

    existing = json.loads(EXISTING_TASK_CONFIG.read_text())
    assert [scenario["id"] for scenario in existing["scenarios"]] == ["dev-small-cluster"]


def test_task_config_loads_through_the_phase5_loaders():
    task = load_task_evaluation_config(TASK_CONFIG)
    ladder = load_ladder_config(TASK_CONFIG)

    assert task.bundle_ids == ("dev-1", "dev-2", "dev-3")
    assert task.instance_keys == tuple(
        (DEV_MEDIUM_CLUSTER_ID, seed) for seed in DEV_MEDIUM_CLUSTER_SEEDS
    )
    assert set(task.constraints.slo_targets) == {
        template.name for template in medium_cluster().workload.templates
    }
    assert ladder.l1.scenario_ids == (DEV_MEDIUM_CLUSTER_ID,)
    # The L1 rung shortens the episode proportionally; the shortened cutoff still has to
    # sit after the registered arrival-rate step or L1 would reject the config outright.
    config = medium_cluster()
    shortened_cutoff = min(
        config.cutoff_s, ladder.l1.horizon_s * config.cutoff_s / config.horizon_s
    )
    steps = [
        at for template in config.workload.templates for at, _ in template.arrival.rate_schedule
    ]
    assert steps and max(steps) < shortened_cutoff


# ---------------------------------------------------------------------- fixture shape
def test_cluster_spans_three_tiers_with_the_registered_node_budget():
    cluster = medium_cluster().cluster
    tiers = [node.tier for node in cluster.nodes]

    assert 12 <= len(cluster.nodes) <= 24
    assert tiers.count(TIER_CLOUD) >= 2
    assert tiers.count(TIER_FOG) >= 2
    assert tiers.count(TIER_EDGE) >= 8
    # Node ids must not be grouped by tier: a stable-id scorer would then reproduce the
    # tier preference order for free (see the module docstring).
    assert tiers != sorted(tiers)
    assert len({node.mem_capacity for node in cluster.nodes}) >= 6


def test_latency_matrix_is_square_symmetric_and_zero_diagonal():
    cluster = medium_cluster().cluster
    n = len(cluster.nodes)
    matrix = cluster.latency_ms

    assert len(matrix) == n and all(len(row) == n for row in matrix)
    for i in range(n):
        assert matrix[i][i] == 0.0
        for j in range(n):
            assert matrix[i][j] == matrix[j][i]
            assert (i == j) or matrix[i][j] > 0.0


def test_workload_has_a_multi_service_multi_replica_call_graph():
    workload = medium_cluster().workload

    assert len(workload.services) >= 4
    assert sum(service.initial_replicas for service in workload.services) >= 10
    assert all(service.max_replicas > service.initial_replicas for service in workload.services)
    assert len(workload.templates) >= 2
    assert all(template.slo_latency_s is not None for template in workload.templates)
    assert {backend.name for backend in workload.backends} == {"db", "kv"}
    # Non-trivial graph: at least one template must be more than one hop deep.
    depths = []
    for template in workload.templates:
        stack = [(template.root, 1)]
        deepest = 1
        while stack:
            call, depth = stack.pop()
            deepest = max(deepest, depth)
            stack.extend((edge.child, depth + 1) for edge in call.children)
        depths.append(deepest)
    assert max(depths) >= 3


def test_controllers_that_make_placement_consequential_are_enabled():
    config = medium_cluster()

    assert config.hpa.enabled
    assert config.rebalancer.enabled  # otherwise propose_moves is dead code
    assert config.power.suspend_after_idle_s is not None
    assert config.snapshot_times_s and config.snapshot_cycles  # L0 needs both
    assert config.cutoff_s < config.horizon_s  # terminating with a drain window
    steps = [
        at for template in config.workload.templates for at, _ in template.arrival.rate_schedule
    ]
    assert steps and all(at < config.cutoff_s for at in steps)


# ------------------------------------------------------------------- replayability
@pytest.mark.parametrize("seed", DEV_MEDIUM_CLUSTER_SEEDS)
def test_episode_replay_is_bitwise_deterministic_for_every_registered_seed(seed):
    import placementscript as ps

    config = load_dev_medium_cluster()
    policy = ps.Policy((REPO / "runtime" / "policies" / "kube_least_allocated.ps").read_text())
    scorer = PlacementScriptScorer(policy)
    proposer = PlacementScriptProposer(policy)
    first = run_episode(config, seed, scorer, proposer)
    second = run_episode(config, seed, scorer, proposer)

    assert first.digest() == second.digest()
    assert first.config_sha256 == config_sha256(config)
    assert first.seed == seed


def test_registered_seeds_do_not_collapse_onto_one_realization():
    import placementscript as ps

    config = load_dev_medium_cluster()
    policy = ps.Policy((REPO / "runtime" / "policies" / "kube_least_allocated.ps").read_text())
    digests = {
        seed: run_episode(
            config, seed, PlacementScriptScorer(policy), PlacementScriptProposer(policy)
        ).digest()
        for seed in DEV_MEDIUM_CLUSTER_SEEDS
    }

    assert len(set(digests.values())) == len(DEV_MEDIUM_CLUSTER_SEEDS)


# --------------------------------------------------------------- richness evidence
def test_sealed_richness_evidence_is_current_and_recomputes():
    """The quarantined richness artifact must describe *this* fixture, not a stale one."""
    payload_path = (
        REPO / "experiments" / "dev_scenario_richness_20260809" / "richness.json"
    )
    sealed = json.loads(payload_path.read_text())
    payload = {
        key: value
        for key, value in sealed.items()
        if key not in {"seal", "timings_not_sealed"}
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    recomputed = hashlib.sha256(("dev-scenario-richness-seal-v1:" + canonical).encode()).hexdigest()

    assert recomputed == sealed["seal"]
    assert sealed["all_checks_pass"]
    assert sealed["scenario_id"] == DEV_MEDIUM_CLUSTER_ID
    assert sealed["evaluation_seeds"] == list(DEV_MEDIUM_CLUSTER_SEEDS)
    assert sealed["scenario_manifest"] == verify_dev_medium_manifest()
    assert (
        sealed["task_config_sha256"] == hashlib.sha256(TASK_CONFIG.read_bytes()).hexdigest()
    )
    # The headline claims this fixture exists for.
    assert sealed["summary"]["baseline_distinct_feasible"] >= 4
    assert sealed["summary"]["overall_distinct_feasible"] >= 15
    assert sealed["summary"]["mutants_strictly_between_baselines"] >= 1
    assert sealed["summary"]["mutants_valid"] >= 60
    assert sealed["timings_not_sealed"]["median_l2_wall_within_budget"]
