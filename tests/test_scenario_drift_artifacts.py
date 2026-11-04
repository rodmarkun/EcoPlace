"""Temporal drift, onboarding, exact-arrival, split, and churn tests."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest
from sim_helpers import fanout_config, flat_scorer_fn, single_service_config, two_edge_cluster

import sim.scenarios.artifacts as artifacts_module
from sim.engine import Simulation, run_episode
from sim.scenarios.artifacts import (
    PILOT,
    PRIVATE,
    PUBLIC,
    P2ConfirmationGate,
    _load_p2_registration_receipt_fixture_bytes,
    build_split_manifest,
    freeze_p2_access_protocol,
    load_scenario_artifact,
    opened_private_scenario_from_json,
    p2_access_receipt_from_json,
    p2_access_receipt_to_json,
    p2_protocol_from_json,
    p2_protocol_to_json,
    private_scenario_commitment,
    run_private_scenario_artifact,
    run_scenario_artifact,
    save_scenario_artifact,
    seal_calibration_informed_synthetic_workload,
    seal_empirical_workload,
    seal_scenario,
    seal_synthetic_workload,
    split_manifest_from_json,
    split_manifest_to_json,
    structural_network_byte_ceiling,
)
from sim.scenarios.drift import (
    OnboardingActivation,
    apply_drift,
    category_prior_call_graph,
    churn_schedule,
    diurnal_changes,
    flash_crowd_changes,
    gradual_mix_shift_changes,
    make_drift_plan,
)
from sim.scenarios.sources import load_source_registry
from sim.scenarios.topology import (
    RoleBinding,
    _load_scale_policy_fixture_bytes,
    hierarchical_topology,
    materialize_cluster,
)
from sim.scheduler import CallableScorer, build_ctx
from sim.spec import ArrivalSpec, HpaParams, NodeClassSpec, expo


def _topology_inputs(seed: int):
    policy = _load_scale_policy_fixture_bytes(
        json.dumps(
            {
                "schema": "g0-capacity-table-v1",
                "table_id": "g0-test-table",
                "ceilings": [
                    {"name": "cpu-hours", "capacity": 100.0, "projected_use": 70.0}
                ],
                "scale_trials": [
                    {
                        "nodes": 8,
                        "p90_runtime_s": 7.0,
                        "runtime_budget_s": 10.0,
                    }
                ],
            },
            sort_keys=True,
        ).encode()
    )
    topology = hierarchical_topology((1, 1, 1, 1), seed, policy)
    bindings = tuple(
        RoleBinding(
            role=role,
            runtime_tier=tier,
            node_class=NodeClassSpec(
                name=f"{role}-{seed}", power_idle_w=2.0, power_max_w=5.0
            ),
            cpu_capacity=4000,
            mem_capacity=8192,
        )
        for role, tier in zip(
            ("cloud-core", "regional-fog", "access-edge", "device-edge"),
            (0, 1, 2, 2),
            strict=True,
        )
    )
    return policy, topology, bindings, materialize_cluster(topology, policy, bindings)


def test_cumulative_intensity_is_partition_invariant_and_supports_zero_gaps():
    plain = ArrivalSpec("deterministic", 2.0)
    split = ArrivalSpec("deterministic", 2.0, rate_schedule=((1.0, 2.0),))
    assert [plain.time_at_integrated_rate(i, 3.0) for i in range(1, 7)] == [
        split.time_at_integrated_rate(i, 3.0) for i in range(1, 7)
    ]
    gap = ArrivalSpec("deterministic", 2.0, rate_schedule=((1.0, 0.0), (2.0, 2.0)))
    assert gap.time_at_integrated_rate(3.0, 4.0) == pytest.approx(2.5)
    assert gap.rate_at(1.0) == 0.0
    with pytest.raises(ValueError, match=">= 0"):
        ArrivalSpec("poisson", 1.0, rate_schedule=((1.0, -1.0),))


def test_flash_feasibility_churn_and_category_prior_are_seeded():
    config = single_service_config(
        two_edge_cluster(),
        rate=2.0,
        horizon_s=20.0,
        hpa=HpaParams(enabled=True, target_utilization=0.6),
    )
    changes = flash_crowd_changes(
        config,
        template_id=0,
        rise_s=4.0,
        fall_s=8.0,
        peak_rate=20.0,
        recovery_rate=2.0,
    )
    assert changes[0].cause == "pre-scaled-hpa-feasible-flash-crowd"
    with pytest.raises(ValueError, match="not HPA-feasible"):
        flash_crowd_changes(
            config,
            template_id=0,
            rise_s=4.0,
            fall_s=8.0,
            peak_rate=1000.0,
            recovery_rate=2.0,
        )
    blocking = fanout_config(horizon_s=20.0, rate=1.0, hpa=HpaParams(enabled=True))
    blocking = dataclasses.replace(
        blocking,
        workload=dataclasses.replace(
            blocking.workload,
            services=(
                dataclasses.replace(
                    blocking.workload.services[0], service_time=expo(0.001)
                ),
                dataclasses.replace(
                    blocking.workload.services[1], service_time=expo(1.0)
                ),
                blocking.workload.services[2],
            ),
        ),
    )
    with pytest.raises(ValueError, match="blocking-slot occupancy"):
        flash_crowd_changes(
            blocking,
            template_id=0,
            rise_s=4.0,
            fall_s=8.0,
            peak_rate=2.0,
            recovery_rate=1.0,
        )
    churn = churn_schedule(node_count=2, down_count=1, down_s=5.0, recovery_s=10.0, seed=77)
    assert churn == churn_schedule(node_count=2, down_count=1, down_s=5.0, recovery_s=10.0, seed=77)
    graph, graph_hash = category_prior_call_graph((0,), "retail", 5)
    assert graph.service == 0 and len(graph_hash) == 64
    diurnal = diurnal_changes(
        template_id=0,
        start_s=1.0,
        end_s=9.0,
        base_rate=10.0,
        amplitude=0.3,
        steps=4,
    )
    assert len(diurnal) == 5 and all(change.rate_per_s > 0 for change in diurnal)
    assert diurnal[0].at_s == 1.0 and diurnal[-1].at_s == 9.0
    assert diurnal[0].rate_per_s == pytest.approx(diurnal[-1].rate_per_s)
    shift = gradual_mix_shift_changes(
        start_s=2.0,
        end_s=8.0,
        total_rate=20.0,
        template_ids=(0, 1),
        start_fractions=(0.8, 0.2),
        target_fractions=(0.3, 0.7),
        steps=3,
    )
    for at_s in sorted({change.at_s for change in shift}):
        assert sum(change.rate_per_s for change in shift if change.at_s == at_s) == pytest.approx(
            20.0
        )


def test_onboarding_has_no_pods_before_activation_and_churn_is_replayed():
    base = single_service_config(two_edge_cluster(), rate=1.0, horizon_s=12.0)
    base = dataclasses.replace(
        base,
        workload=dataclasses.replace(
            base.workload,
            templates=(
                dataclasses.replace(
                    base.workload.templates[0],
                    arrival=dataclasses.replace(base.workload.templates[0].arrival, start_s=5.0),
                ),
            ),
        ),
    )
    _, graph_hash = category_prior_call_graph((0,), "retail", 5)
    plan = make_drift_plan(
        seed=10,
        node_churn=churn_schedule(node_count=2, down_count=1, down_s=7.0, recovery_s=9.0, seed=8),
        onboarding=(OnboardingActivation(5.0, 0, (0,), "retail", 5, graph_hash),),
    )
    with pytest.raises(ValueError, match="must not start before"):
        apply_drift(single_service_config(two_edge_cluster()), plan)
    config = apply_drift(base, plan)
    scorer = CallableScorer("flat", flat_scorer_fn)
    simulation = Simulation(config, 44, scorer)
    simulation.start()
    simulation.advance(5.0)
    assert simulation.cluster.pods == {}
    simulation.advance(5.0001)
    assert len(simulation.cluster.pods) == 1
    simulation.advance(7.0001)
    snapshot = simulation.snapshot_now()
    down_node = plan.node_churn[0].node_id
    assert snapshot.ctx["nodes"][down_node]["suspended"] is True
    assert snapshot.power_suspended_by_node[down_node] is False

    full = run_episode(config, 44, scorer)
    assert [event["schedulable"] for event in full.churn_events] == [False, True]
    assert full.digest() == run_episode(config, 44, scorer).digest()


def test_multiservice_onboarding_activates_the_whole_application_atomically():
    base = fanout_config(horizon_s=12.0, rate=1.0)
    graph, graph_hash = category_prior_call_graph((0, 1, 2), "retail", 55)
    base = dataclasses.replace(
        base,
        workload=dataclasses.replace(
            base.workload,
            templates=(
                dataclasses.replace(
                    base.workload.templates[0],
                    root=graph,
                    arrival=dataclasses.replace(
                        base.workload.templates[0].arrival, start_s=5.0
                    ),
                ),
            ),
        ),
    )
    plan = make_drift_plan(
        seed=56,
        onboarding=(
            OnboardingActivation(5.0, 0, (0, 1, 2), "retail", 55, graph_hash),
        ),
    )
    config = apply_drift(base, plan)
    simulation = Simulation(config, 57, CallableScorer("flat", flat_scorer_fn))
    simulation.start()
    simulation.advance(5.0)
    assert simulation.cluster.pods == {}
    simulation.advance(5.0001)
    assert {pod.service.id for pod in simulation.cluster.pods.values()} == {0, 1, 2}
    assert len(simulation.cluster.pods) == sum(
        service.initial_replicas for service in config.workload.services
    )


def test_scenario_artifact_realizes_arrivals_bounds_bytes_and_separates_splits(
    tmp_path, monkeypatch
):
    registry = load_source_registry()
    policy, topology, bindings, cluster = _topology_inputs(11)
    config = single_service_config(cluster, rate=2.0, horizon_s=4.0)
    drift = make_drift_plan(seed=31)
    workload_model = seal_synthetic_workload(
        config, registry, generator_version="test-synthetic-workload-v1"
    )
    with pytest.raises(ValueError, match="empirical workload sealing is unavailable"):
        seal_empirical_workload(config=config, registry=registry)
    with pytest.raises(ValueError, match="typed builder"):
        seal_calibration_informed_synthetic_workload(config=config, registry=registry)
    artifact = seal_scenario(
        scenario_id="pilot-001",
        visibility=PILOT,
        topology=topology,
        scale_policy=policy,
        topology_bindings=bindings,
        drift_plan=drift,
        workload_model=workload_model,
        source_registry=registry,
        episode_seed=12,
        generator_version="phase4-v1",
        config=config,
    )
    realized = artifact.verify()
    assert artifact.workload_provenance_label == "synthetic-scenario-assumption"
    assert realized.workload.templates[0].arrival.arrivals_realized
    scorer = CallableScorer("flat", flat_scorer_fn)
    result = run_episode(realized, 12, scorer)
    expected = realized.workload.templates[0].arrival.realized_times_s
    assert tuple(record.arrival_s for record in result.records) == expected
    assert result.energy.network_bytes <= artifact.network_byte_ceiling
    mismatched_config = dataclasses.replace(
        config,
        workload=dataclasses.replace(
            config.workload,
            templates=(
                dataclasses.replace(
                    config.workload.templates[0],
                    arrival=dataclasses.replace(
                        config.workload.templates[0].arrival, rate_per_s=3.0
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="exact EpisodeConfig workload"):
        seal_scenario(
            scenario_id="pilot-mismatched-workload",
            visibility=PILOT,
            topology=topology,
            scale_policy=policy,
            topology_bindings=bindings,
            drift_plan=drift,
            workload_model=workload_model,
            source_registry=registry,
            episode_seed=13,
            generator_version="phase4-v1",
            config=mismatched_config,
        )

    public = dataclasses.replace(
        artifact,
        scenario_id="public-001",
        visibility=PUBLIC,
    )
    # Re-seal instead of accepting a field-edited copy.
    with pytest.raises(ValueError, match="digest mismatch"):
        public.verify()
    private_policy, private_topology, private_bindings, private_cluster = _topology_inputs(99)
    private_config = single_service_config(private_cluster, rate=2.0, horizon_s=4.0)
    private = seal_scenario(
        scenario_id="private-001",
        visibility=PRIVATE,
        topology=private_topology,
        scale_policy=private_policy,
        topology_bindings=private_bindings,
        drift_plan=make_drift_plan(seed=199),
        workload_model=seal_synthetic_workload(
            private_config, registry, generator_version="test-synthetic-workload-v1"
        ),
        source_registry=registry,
        episode_seed=299,
        generator_version="phase4-v1",
        config=private_config,
    )
    manifest = build_split_manifest((), (private_scenario_commitment(private),), (artifact,))
    manifest.verify()
    manifest_text = split_manifest_to_json(manifest)
    assert split_manifest_from_json(manifest_text).digest == manifest.digest
    manifest_path = tmp_path / "scenario-split.json"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from sim.scenarios.artifacts import split_manifest_from_json; "
                "import sys; print(split_manifest_from_json("
                "Path(sys.argv[1]).read_text()).digest)"
            ),
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert manifest_process.stdout.strip() == manifest.digest
    assert manifest.private_commitments[0].artifact_sha256 is None
    with pytest.raises(PermissionError, match="P2 access receipt"):
        run_scenario_artifact(private, scorer)
    artifact_path = tmp_path / "pilot-scenario.json"
    save_scenario_artifact(artifact, artifact_path)
    assert load_scenario_artifact(artifact_path).digest == artifact.digest
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from sim.scenarios.artifacts import load_scenario_artifact; "
                "import sys; print(load_scenario_artifact(Path(sys.argv[1])).digest)"
            ),
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert process.stdout.strip() == artifact.digest
    private_path = tmp_path / "private-scenario.json"
    save_scenario_artifact(private, private_path)
    with pytest.raises(PermissionError, match="P2ConfirmationGate"):
        load_scenario_artifact(private_path)
    p2_receipt = json.dumps(
        {
            "schema": "scenario-p2-registration-receipt-v1",
            "receipt_id": "external-p2-test-receipt-001",
            "registration_id": "P2-test",
            "authority_url": "https://osf.example.invalid/registrations/p2-test",
            "frozen_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "registered_at_utc": "2026-07-17T10:00:00+00:00",
            "split_manifest_sha256": manifest.digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    p2_registration = _load_p2_registration_receipt_fixture_bytes(p2_receipt, manifest)
    p2 = freeze_p2_access_protocol(
        registration_receipt=p2_registration, manifest=manifest
    )
    with pytest.raises(ValueError, match="external registration receipt"):
        dataclasses.replace(p2, _authority=object()).verify(manifest)
    p2_text = p2_protocol_to_json(p2, manifest)
    assert p2_protocol_from_json(
        p2_text,
        registration_receipt=p2_registration,
        manifest=manifest,
    ).digest == p2.digest
    p2_marker = tmp_path / "p2-opened.json"
    monkeypatch.setattr(
        artifacts_module, "require_frozen_clean_checkout", lambda commit: commit
    )
    opened = P2ConfirmationGate(p2, manifest, p2_marker).open_private(
        private.scenario_id,
        lambda: private_path.read_text(),
    )
    receipt_text = p2_access_receipt_to_json(
        opened.receipt,
        protocol=p2,
        manifest=manifest,
        artifact=opened.artifact,
        marker=p2_marker,
    )
    assert p2_access_receipt_from_json(
        receipt_text,
        protocol=p2,
        manifest=manifest,
        artifact=opened.artifact,
        marker=p2_marker,
    ).digest == opened.receipt.digest
    assert opened_private_scenario_from_json(
        private_path.read_text(),
        receipt_text,
        protocol=p2,
        manifest=manifest,
        marker=p2_marker,
    ).artifact.digest == private.digest
    p2_receipt_path = tmp_path / "p2-registration.json"
    p2_receipt_path.write_bytes(p2_receipt)
    p2_protocol_path = tmp_path / "p2-protocol.json"
    p2_protocol_path.write_text(p2_text, encoding="utf-8")
    access_receipt_path = tmp_path / "p2-access-receipt.json"
    access_receipt_path.write_text(receipt_text, encoding="utf-8")
    reload_process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "import sim.scenarios.artifacts as a; "
                "a.require_frozen_clean_checkout=lambda commit: commit; "
                "m=a.split_manifest_from_json(Path(sys.argv[1]).read_text()); "
                "r=a._load_p2_registration_receipt_fixture_bytes("
                "Path(sys.argv[2]).read_bytes(),m); "
                "p=a.p2_protocol_from_json(Path(sys.argv[3]).read_text(),"
                "registration_receipt=r,manifest=m); "
                "o=a.opened_private_scenario_from_json("
                "Path(sys.argv[4]).read_text(),Path(sys.argv[5]).read_text(),"
                "protocol=p,manifest=m,marker=Path(sys.argv[6])); "
                "print(o.artifact.digest)"
            ),
            str(manifest_path),
            str(p2_receipt_path),
            str(p2_protocol_path),
            str(private_path),
            str(access_receipt_path),
            str(p2_marker),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert reload_process.stdout.strip() == private.digest
    confirmed = run_private_scenario_artifact(
        opened, p2, manifest, p2_marker, scorer
    )
    assert confirmed.records
    with pytest.raises(ValueError, match="disjoint"):
        duplicate = seal_scenario(
            scenario_id="pilot-001",
            visibility=PRIVATE,
            topology=private_topology,
            scale_policy=private_policy,
            topology_bindings=private_bindings,
            drift_plan=make_drift_plan(seed=199),
            workload_model=seal_synthetic_workload(
                private_config, registry, generator_version="test-synthetic-workload-v1"
            ),
            source_registry=registry,
            episode_seed=300,
            generator_version="phase4-v1",
            config=private_config,
        )
        build_split_manifest((), (private_scenario_commitment(duplicate),), (artifact,))
    with pytest.raises(ValueError, match="disjoint"):
        reused_episode = seal_scenario(
            scenario_id="private-reused-episode",
            visibility=PRIVATE,
            topology=private_topology,
            scale_policy=private_policy,
            topology_bindings=private_bindings,
            drift_plan=make_drift_plan(seed=199),
            workload_model=seal_synthetic_workload(
                private_config, registry, generator_version="test-synthetic-workload-v1"
            ),
            source_registry=registry,
            episode_seed=12,
            generator_version="phase4-v1",
            config=private_config,
        )
        build_split_manifest(
            (), (private_scenario_commitment(reused_episode),), (artifact,)
        )


def test_cutoff_is_half_open_and_campaign_runner_enforces_artifact_seed():
    registry = load_source_registry()
    policy, topology, bindings, cluster = _topology_inputs(401)
    config = single_service_config(cluster, rate=2.0, horizon_s=4.0)
    config = dataclasses.replace(
        config,
        workload=dataclasses.replace(
            config.workload,
            templates=(
                dataclasses.replace(
                    config.workload.templates[0],
                    arrival=ArrivalSpec("deterministic", 2.0),
                ),
            ),
        ),
    )
    artifact = seal_scenario(
        scenario_id="pilot-half-open",
        visibility=PILOT,
        topology=topology,
        scale_policy=policy,
        topology_bindings=bindings,
        drift_plan=make_drift_plan(seed=402),
        workload_model=seal_synthetic_workload(
            config, registry, generator_version="test-synthetic-workload-v1"
        ),
        source_registry=registry,
        episode_seed=403,
        generator_version="phase4-v1",
        config=config,
    )
    sealed = artifact.verify()
    assert sealed.workload.templates[0].arrival.realized_times_s == (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    )
    scorer = CallableScorer("flat", flat_scorer_fn)
    result = run_scenario_artifact(artifact, scorer)
    assert len(result.records) == 7
    assert result.digest() == run_episode(sealed, artifact.episode_seed, scorer).digest()
    assert result.digest() != run_episode(sealed, artifact.episode_seed + 1, scorer).digest()
    graph_config = fanout_config(horizon_s=12.0, rate=1.0)
    simulation = Simulation(graph_config, artifact.episode_seed, scorer)
    simulation.start()
    simulation.advance(graph_config.cutoff_s)
    before_ctx, _, _ = build_ctx(
        simulation.cluster,
        graph_config.workload,
        None,
        graph_config.cutoff_s - 1e-6,
        graph_config.cutoff_s,
    )
    cutoff_ctx, _, _ = build_ctx(
        simulation.cluster,
        graph_config.workload,
        None,
        graph_config.cutoff_s,
        graph_config.cutoff_s,
    )
    assert before_ctx["graph"]["rate"]
    assert all(rate == 0.0 for rate in cutoff_ctx["graph"]["rate"])
    with pytest.raises(ValueError, match="strictly before cutoff"):
        dataclasses.replace(
            sealed,
            workload=dataclasses.replace(
                sealed.workload,
                templates=(
                    dataclasses.replace(
                        sealed.workload.templates[0],
                        arrival=dataclasses.replace(
                            sealed.workload.templates[0].arrival,
                            realized_times_s=(4.0,),
                            arrivals_realized=True,
                        ),
                    ),
                ),
            ),
        )


def test_network_byte_ceiling_covers_concurrent_initial_image_pulls():
    _, _, _, cluster = _topology_inputs(501)
    config = single_service_config(
        cluster,
        rate=1.0,
        horizon_s=3.0,
        service_kwargs={"initial_replicas": 3, "max_replicas": 3, "image_mb": 10},
    )
    config = dataclasses.replace(
        config,
        workload=dataclasses.replace(
            config.workload,
            templates=(
                dataclasses.replace(
                    config.workload.templates[0],
                    arrival=dataclasses.replace(
                        config.workload.templates[0].arrival,
                        realized_times_s=(),
                        arrivals_realized=True,
                    ),
                ),
            ),
        ),
    )
    bound = structural_network_byte_ceiling(config)
    result = run_episode(config, 502, CallableScorer("flat", flat_scorer_fn))
    assert result.energy.network_bytes <= bound
