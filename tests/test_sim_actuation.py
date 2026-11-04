"""Actuation-twin tests (TODO 2.3): budgeted rebalancer + action gateway, HPA, node
power-state model."""

from __future__ import annotations

from dataclasses import replace

import placementscript as ps
import pytest
from sim_helpers import (
    POLICIES,
    fanout_config,
    flat_scorer_fn,
    mixed_cluster,
    simple_service,
    single_service_config,
    two_edge_cluster,
)

from sim import (
    CallableProposer,
    CallableScorer,
    HpaParams,
    PlacementScriptProposer,
    RebalancerParams,
    run_episode,
)
from sim.runtime import HostError, SimulationError
from sim.spec import PowerParams

SEED = 909


def flat():
    return CallableScorer("flat", flat_scorer_fn)


def rebalancer_config(**rebalancer_kwargs):
    """4 pods flat-packed onto node 0 of two nodes; the proposer then acts at 10 s."""
    defaults = dict(enabled=True, window_s=10.0, budget_per_window=4)
    defaults.update(rebalancer_kwargs)
    return single_service_config(
        two_edge_cluster(),
        horizon_s=25.0,
        rate=2.0,
        service_kwargs={"replicas": 4, "cpu_request": 400},
        rebalancer=RebalancerParams(**defaults),
    )


class TestRebalancerGateway:
    def test_moves_execute_and_are_capped_by_budget(self):
        def move_two(state, graph, budget):
            del graph
            pods_on_0 = [p for p in state["pods"] if p["current_node"] == 0]
            return [
                {"kind": 0, "pod": p["id"], "dest": 1} for p in pods_on_0[: budget + 3]
            ]

        result = run_episode(
            rebalancer_config(budget_per_window=2),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("mover", move_two),
        )
        led = result.ledger
        assert led["rebalancer_contract_violations"] >= 1  # over-budget list truncated
        assert 0 < led["rebalancer_moves_executed"] <= 2 * 3  # <= budget x windows
        moved = [p for p in result.pods if p.origin == "rebalancer"]
        assert moved and all(p.node_id in (1, None) for p in moved)

    def test_stateful_and_pdb_pods_are_rejected(self):
        config = rebalancer_config()
        svc = replace(
            config.workload.services[0], stateful=True, pdb_protected=True
        )
        config = replace(config, workload=replace(config.workload, services=(svc,)))

        def move_all(state, graph, budget):
            del graph
            return [
                {"kind": 0, "pod": p["id"], "dest": 1}
                for p in state["pods"]
                if p["current_node"] == 0
            ][:budget]

        result = run_episode(
            config, seed=SEED, scorer=flat(), proposer=CallableProposer("m", move_all)
        )
        led = result.ledger
        assert led["rebalancer_moves_executed"] == 0
        assert led["rebalancer_actions_rejected"] > 0
        reasons = {r["reason"] for r in result.rebalancer["rejections"]}
        assert "move: stateful pod excluded" in reasons

    def test_move_to_suspended_without_wake_rejected_with_wake_executes(self):
        config = single_service_config(
            mixed_cluster(suspended_third=True),
            horizon_s=30.0,
            rate=2.0,
            service_kwargs={"replicas": 2, "cpu_request": 400},
            rebalancer=RebalancerParams(enabled=True, window_s=10.0, budget_per_window=4),
        )

        calls = {"n": 0}

        def proposer(state, graph, budget):
            del graph, budget
            calls["n"] += 1
            movable = [
                p
                for p in state["pods"]
                if p["current_node"] >= 0 and not p["stateful"] and not p["has_pdb"]
            ]
            if calls["n"] == 1:  # no wake first: must be rejected
                return [{"kind": 0, "pod": movable[0]["id"], "dest": 2}]
            if calls["n"] == 2:  # wake then move: must execute
                return [
                    {"kind": 1, "pod": -1, "dest": 2},
                    {"kind": 0, "pod": movable[0]["id"], "dest": 2},
                ]
            return []

        result = run_episode(
            config, seed=SEED, scorer=flat(), proposer=CallableProposer("w", proposer)
        )
        led = result.ledger
        reasons = [r["reason"] for r in result.rebalancer["rejections"]]
        assert "move: destination suspended without prior wake" in reasons
        assert led["rebalancer_wakes_executed"] == 1
        assert led["rebalancer_moves_executed"] == 1
        moved = [p for p in result.pods if p.origin == "rebalancer"]
        assert any(p.node_id == 2 for p in moved)
        # The wake transition consumed published energy on node 2.
        assert result.energy.transition_j_per_node[2] > 0.0

    def test_cumulative_disruption_caps_stop_moves(self):
        def shuffle(state, graph, budget):
            del graph
            out = []
            for p in state["pods"]:
                if p["current_node"] >= 0 and len(out) < budget:
                    dest = 1 - p["current_node"]
                    out.append({"kind": 0, "pod": p["id"], "dest": dest})
            return out

        result = run_episode(
            rebalancer_config(max_total_moves=3),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("s", shuffle),
        )
        assert result.rebalancer["moves"] <= 3
        reasons = {r["reason"] for r in result.rebalancer["rejections"]}
        assert "move: cumulative move cap reached" in reasons

    def test_bytes_cap_is_enforced_before_execution(self):
        """R1-3: a known-oversized migration must be rejected at admission, never
        admitted and charged post-hoc."""

        def move_one(state, graph, budget):
            del graph, budget
            for p in state["pods"]:
                if p["current_node"] == 0:
                    return [{"kind": 0, "pod": p["id"], "dest": 1}]
            return []

        # The service image is 10 MiB; cap the episode at 1 MiB: nothing may move.
        result = run_episode(
            rebalancer_config(max_total_bytes_mb=1.0),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("m1", move_one),
        )
        assert result.rebalancer["moves"] == 0
        assert result.rebalancer["bytes_mb"] == 0.0
        reasons = {r["reason"] for r in result.rebalancer["rejections"]}
        assert "move: cumulative bytes cap would be exceeded" in reasons

    def test_duplicate_moves_for_one_pod_rejected_not_crashing(self):
        """R1-3 regression: two identical accepted actions for the same pod used to
        crash with 'unbind of unbound pod'; the gateway must reject the duplicate."""

        def duplicate(state, graph, budget):
            del graph, budget
            for p in state["pods"]:
                if p["current_node"] == 0:
                    action = {"kind": 0, "pod": p["id"], "dest": 1}
                    return [action, dict(action)]
            return []

        result = run_episode(
            rebalancer_config(),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("dup", duplicate),
        )
        reasons = [r["reason"] for r in result.rebalancer["rejections"]]
        assert "move: duplicate pod in one proposal" in reasons
        # Episode completed; conservation verified by run_episode itself.

    def test_window_moves_reserve_destination_capacity_transactionally(self):
        """R1-3: multiple moves accepted in one window must respect the destination's
        remaining capacity jointly (shadow reservations), not just individually."""
        # Node 1 has 4000m; each pod requests 1500m -> at most 2 can move there.
        config = single_service_config(
            two_edge_cluster(),
            horizon_s=25.0,
            rate=2.0,
            service_kwargs={"replicas": 4, "cpu_request": 1500, "cpu_demand": 100},
            rebalancer=RebalancerParams(enabled=True, window_s=10.0, budget_per_window=4),
        )

        def move_all(state, graph, budget):
            del graph
            return [
                {"kind": 0, "pod": p["id"], "dest": 1}
                for p in state["pods"]
                if p["current_node"] == 0
            ][:budget]

        result = run_episode(
            config, seed=SEED, scorer=flat(), proposer=CallableProposer("ma", move_all)
        )
        reasons = [r["reason"] for r in result.rebalancer["rejections"]]
        assert "move: destination capacity would be exceeded" in reasons
        # Never more pods bound to node 1 than its capacity allows at once (the
        # episode-level invariant is enforced by ClusterRuntime.bind; reaching the end
        # without SimulationError is the proof).

    def test_destination_reservation_persists_across_windows(self):
        """R2-5: an open drain reserves destination capacity across later windows."""
        from sim_helpers import EDGE_CLASS

        from sim import ArrivalSpec, ClusterSpec, NodeSpec, det

        cluster = ClusterSpec(
            classes=(EDGE_CLASS,),
            nodes=(
                NodeSpec(0, "edge", 2, 4000, 8192),
                NodeSpec(1, "edge", 2, 600, 8192),
            ),
            latency_ms=((0.0, 1.0), (1.0, 0.0)),
        )
        config = single_service_config(
            cluster,
            horizon_s=6.0,
            rate=10.0,
            service_kwargs={
                "replicas": 2,
                "cpu_request": 600,
                "cpu_demand": 0,
                "service_time": det(100.0),
                "image_mb": 0,
            },
            rebalancer=RebalancerParams(
                enabled=True, window_s=2.0, budget_per_window=1
            ),
        )
        template = replace(
            config.workload.templates[0],
            arrival=ArrivalSpec("deterministic", 10.0),
        )
        config = replace(
            config, workload=replace(config.workload, templates=(template,))
        )

        windows = {"count": 0}

        def move_ready(state, graph, budget):
            del graph, budget
            windows["count"] += 1
            target = windows["count"] - 1
            candidates = [pod for pod in state["pods"] if pod["current_node"] == 0]
            if target < len(candidates):
                return [{"kind": 0, "pod": candidates[target]["id"], "dest": 1}]
            return []

        result = run_episode(
            config,
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("cross-window", move_ready),
        )
        assert result.rebalancer["moves"] == 1
        assert "move: destination capacity would be exceeded" in {
            rejection["reason"] for rejection in result.rebalancer["rejections"]
        }

    def test_open_disruption_is_right_censored_and_marked_infeasible(self):
        """R2-5: horizon-open drains accrue unavailability and cannot hide a cap miss."""
        from sim import ArrivalSpec, det

        config = single_service_config(
            two_edge_cluster(),
            horizon_s=10.0,
            rate=2.0,
            service_kwargs={
                "replicas": 1,
                "service_time": det(100.0),
                "cpu_demand": 0,
                "image_mb": 0,
            },
            rebalancer=RebalancerParams(
                enabled=True,
                window_s=2.0,
                budget_per_window=1,
                max_total_unavailability_s=5.0,
            ),
        )
        template = replace(
            config.workload.templates[0],
            arrival=ArrivalSpec("deterministic", 2.0),
        )
        config = replace(
            config, workload=replace(config.workload, templates=(template,))
        )

        def move_once(state, graph, budget):
            del graph, budget
            for pod in state["pods"]:
                if pod["current_node"] == 0:
                    return [{"kind": 0, "pod": pod["id"], "dest": 1}]
            return []

        result = run_episode(
            config,
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("censored", move_once),
        )
        assert result.rebalancer["open_moves_censored"] == 1
        assert result.rebalancer["unavailability_s"] == pytest.approx(8.0)
        assert result.rebalancer["budget_exceeded"] is True

    def test_proposer_crash_is_counted_never_fatal(self):
        def boom(state, graph, budget):
            raise RuntimeError("proposer exploded")

        result = run_episode(
            rebalancer_config(),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("boom", boom),
        )
        assert result.ledger["rebalancer_contract_violations"] >= 1
        assert result.ledger["rebalancer_moves_executed"] == 0

    def test_proposer_host_failure_propagates_and_none_is_policy_fault(self):
        def host_fault(state, graph, budget):
            raise HostError("host view unavailable")

        with pytest.raises(HostError, match="host view unavailable"):
            run_episode(
                rebalancer_config(),
                seed=SEED,
                scorer=flat(),
                proposer=CallableProposer("host", host_fault),
            )

        def malformed(state, graph, budget):
            return None

        result = run_episode(
            rebalancer_config(),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("none", malformed),
        )
        assert result.ledger["rebalancer_contract_violations"] > 0
        assert result.rebalancer["moves"] == 0

    def test_migration_bytes_account_image_cache(self):
        def move_one(state, graph, budget):
            del graph, budget
            for p in state["pods"]:
                if p["current_node"] == 0:
                    return [{"kind": 0, "pod": p["id"], "dest": 1}]
            return []

        result = run_episode(
            rebalancer_config(max_total_moves=1),
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("m1", move_one),
        )
        # Destination had no replica of this service -> image pulled -> bytes counted.
        assert result.rebalancer["moves"] == 1
        assert result.rebalancer["bytes_mb"] >= 10  # image_mb of the service
        assert result.rebalancer["unavailability_s"] > 0.0

    def test_fallback_path_accounts_state_transfer_bytes(self):
        """R2-5/R1-11: destination-race fallback cannot erase migration traffic."""
        from sim_helpers import EDGE_CLASS

        from sim import ArrivalSpec, ClusterSpec, NodeSpec, Simulation
        from sim.energy import MIB

        cluster = ClusterSpec(
            (EDGE_CLASS,),
            (
                NodeSpec(0, "edge", 2, 1000, 8192),
                NodeSpec(1, "edge", 2, 500, 8192),
            ),
            ((0.0, 1.0), (1.0, 0.0)),
        )
        config = single_service_config(
            cluster,
            horizon_s=5.0,
            rate=1.0,
            service_kwargs={
                "replicas": 1,
                "cpu_request": 500,
                "state_transfer_mb": 50.0,
            },
            rebalancer=RebalancerParams(
                enabled=True, window_s=100.0, budget_per_window=1
            ),
        )
        template = replace(
            config.workload.templates[0],
            arrival=ArrivalSpec("deterministic", 1.0, start_s=100.0),
        )
        config = replace(
            config, workload=replace(config.workload, templates=(template,))
        )

        def move(state, graph, budget):
            del graph, budget
            pod = next(p for p in state["pods"] if p["current_node"] == 0)
            return [{"kind": 0, "pod": pod["id"], "dest": 1}]

        sim = Simulation(
            config,
            SEED,
            flat(),
            proposer=CallableProposer("fallback", move),
        )
        sim.start()
        sim.advance(1.0)
        network_before_move = sim.meter._bytes
        victim = next(iter(sim.app.services[0].replicas.values())).pod
        victim.outstanding = 1  # hold the admitted move in its drain
        sim.rebalancer._window()
        move_record = sim.rebalancer._open_moves[victim.uid]
        for _ in range(10):
            if victim.drained is not None:
                break
            sim.env.step()  # start _move; it installs victim.drained and waits
        assert victim.drained is not None
        blocker = sim.cluster.create_pod(
            victim.service,
            "hpa",
            rng_identity=("hpa", 99, 0),
        )
        sim.cluster.bind(blocker, 1, notify=True)  # consumes the admitted destination
        victim.outstanding = 0
        victim.drained.succeed()
        sim.advance(5.0)
        sim.rebalancer.finalize(5.0)
        report = sim.meter.finish(5.0)
        assert sim.rebalancer.total_bytes_mb == pytest.approx(50.0)
        assert report.network_bytes - network_before_move == pytest.approx(50.0 * MIB)
        replacement = move_record["replacement"]
        assert replacement.ready_s is not None
        assert sim.rebalancer.total_unavailability_s == pytest.approx(
            replacement.ready_s - move_record["evict_start"], abs=1e-12
        )
        # This path used to be observed by a 0.5 s poll and overcounted short gaps.
        assert sim.rebalancer.total_unavailability_s < 0.5

    def test_placementscript_proposer_path(self):
        policy = ps.Policy((POLICIES / "least_allocated.ps").read_text())
        result = run_episode(
            rebalancer_config(),
            seed=SEED,
            scorer=flat(),
            proposer=PlacementScriptProposer(policy),
        )
        # flat packing put everything on node 0 -> imbalance > 0.25 -> policy moves.
        assert result.ledger["rebalancer_moves_executed"] > 0


class TestHpa:
    def test_scale_up_under_load_and_down_after(self):
        config = fanout_config(horizon_s=120.0, rate=25.0)
        config = replace(
            config,
            hpa=HpaParams(
                enabled=True,
                sync_period_s=5.0,
                target_utilization=0.5,
                scale_down_stabilization_s=20.0,
            ),
            arrival_cutoff_s=60.0,
        )
        result = run_episode(config, seed=SEED, scorer=flat())
        led = result.ledger
        assert led["hpa_scale_ups"] > 0
        assert led["hpa_scale_downs"] > 0
        hpa_pods = [p for p in result.pods if p.origin == "hpa"]
        assert hpa_pods
        ups = [a for a in result.hpa_actions if a["action"] == "up"]
        downs = [a for a in result.hpa_actions if a["action"] == "down"]
        assert ups and downs
        # The loaded service scales up during load, and back down after the arrival
        # cutoff (60 s) once the stabilization window drains.
        loaded = ups[0]["service"]
        assert min(a["t"] for a in ups) < 60.0
        loaded_downs = [a for a in downs if a["service"] == loaded]
        assert loaded_downs and max(a["t"] for a in loaded_downs) > 60.0

    def test_replica_bounds_respected(self):
        config = fanout_config(horizon_s=90.0, rate=40.0)
        config = replace(
            config, hpa=HpaParams(enabled=True, sync_period_s=5.0, target_utilization=0.3)
        )
        result = run_episode(config, seed=SEED, scorer=flat())
        # Live replicas per service never exceed max_replicas.
        for sid, spec in enumerate(config.workload.services):
            live = [
                p
                for p in result.pods
                if p.service == sid and p.terminated_s is None
            ]
            assert len(live) <= spec.max_replicas

    def test_replica_count_never_exceeds_max_at_any_time(self):
        """R1-13: the max-replica bound holds over the whole episode timeline, not just
        in the final state — reconstructed from pod creation/termination events."""
        config = fanout_config(horizon_s=90.0, rate=40.0)
        config = replace(
            config, hpa=HpaParams(enabled=True, sync_period_s=5.0, target_utilization=0.3)
        )
        result = run_episode(config, seed=SEED, scorer=flat())
        for sid, spec in enumerate(config.workload.services):
            events = []  # (time, +1/-1); termination before creation at equal times
            for p in result.pods:
                if p.service != sid:
                    continue
                events.append((p.created_s, 1))
                if p.terminated_s is not None:
                    events.append((p.terminated_s, -1))
            count, peak = 0, 0
            for _t, delta in sorted(events, key=lambda e: (e[0], e[1])):
                count += delta
                peak = max(peak, count)
            assert peak <= spec.max_replicas, f"service {sid} peaked at {peak}"

    def test_hpa_and_rebalancer_same_instant_do_not_double_select(self):
        """R1-3 race regression: HPA sync and the rebalancer window fire at the same
        instants under load with scale-downs; victim locking is synchronous, so no pod
        can be terminated twice (an unbind of an unbound pod would raise)."""

        def churn(state, graph, budget):
            del graph
            out = []
            for p in state["pods"]:
                if p["current_node"] >= 0 and len(out) < budget:
                    out.append(
                        {"kind": 0, "pod": p["id"], "dest": 1 - p["current_node"]}
                    )
            return out

        config = single_service_config(
            two_edge_cluster(),
            horizon_s=60.0,
            rate=30.0,
            service_kwargs={"replicas": 2, "cpu_request": 400},
            hpa=HpaParams(
                enabled=True,
                sync_period_s=10.0,
                target_utilization=0.5,
                scale_down_stabilization_s=10.0,
            ),
            rebalancer=RebalancerParams(enabled=True, window_s=10.0, budget_per_window=2),
            arrival_cutoff_s=30.0,
        )
        result = run_episode(
            config, seed=SEED, scorer=flat(), proposer=CallableProposer("churn", churn)
        )
        # Both controllers acted at overlapping instants; conservation verified by
        # run_episode. Sanity: something actually happened on both sides.
        assert result.ledger["rebalancer_moves_executed"] > 0
        assert result.ledger["hpa_scale_ups"] + result.ledger["hpa_scale_downs"] > 0

    def test_draining_replica_counts_toward_hard_max(self):
        """R2-10: a slow HPA drain cannot be hidden by creating above max_replicas."""
        from sim import (
            ArrivalSpec,
            BackendSpec,
            CallSpec,
            ClusterSpec,
            EpisodeConfig,
            NodeClassSpec,
            NodeSpec,
            SchedulerParams,
            ServiceSpec,
            TemplateSpec,
            WorkloadSpec,
            det,
        )

        cls = NodeClassSpec("n", 1.0, 2.0, image_pull_mib_per_s=1000.0)
        cluster = ClusterSpec(
            (cls,),
            (NodeSpec(0, "n", 2, 10_000, 10_000),),
            ((0.0,),),
        )
        service = ServiceSpec(
            0,
            "s",
            0,
            100,
            10,
            det(0.1),
            100,
            concurrency=64,
            backend="db",
            image_mb=0,
            initial_replicas=2,
            min_replicas=1,
            max_replicas=2,
        )
        low = TemplateSpec(
            0, "low", CallSpec(0), ArrivalSpec("deterministic", 0.5)
        )
        high = TemplateSpec(
            1, "high", CallSpec(0), ArrivalSpec("deterministic", 100.0, start_s=5.0)
        )
        config = EpisodeConfig(
            cluster,
            WorkloadSpec((service,), (low, high), (BackendSpec("db", ((1.0, 100.0),)),)),
            11.0,
            arrival_cutoff_s=10.5,
            scheduler=SchedulerParams(bind_latency=det(0.0)),
            hpa=HpaParams(
                enabled=True,
                sync_period_s=5.0,
                target_utilization=0.5,
                scale_down_stabilization_s=0.0,
            ),
        )
        result = run_episode(config, seed=1, scorer=flat())
        events = []
        for pod in result.pods:
            events.append((pod.created_s, 1))
            if pod.terminated_s is not None:
                events.append((pod.terminated_s, -1))
        live = peak = 0
        for _at, delta in sorted(events, key=lambda event: (event[0], event[1])):
            live += delta
            peak = max(peak, live)
        assert peak <= service.max_replicas


class TestPowerStateModel:
    def test_auto_suspend_of_idle_empty_node(self):
        # 1 replica on 2 nodes: node 1 stays empty and must suspend after 5 s idle.
        config = single_service_config(
            two_edge_cluster(),
            horizon_s=30.0,
            rate=1.0,
            service_kwargs={"replicas": 1},
            power=PowerParams(suspend_after_idle_s=5.0),
        )
        result = run_episode(config, seed=SEED, scorer=flat())
        # Suspend pulse accounted; suspended draw thereafter (device energy < idle
        # power x horizon for node 1).
        assert result.energy.transition_j_per_node[1] > 0.0
        idle_only = 3.0 * 30.0  # idle W x horizon
        assert result.energy.device_j_per_node[1] < idle_only

    def test_non_suspendable_class_never_suspends(self):
        from sim_helpers import PLAIN_EDGE

        from sim.spec import ClusterSpec, NodeSpec

        cluster = ClusterSpec(
            classes=(PLAIN_EDGE,),
            nodes=(NodeSpec(0, "plain-edge", 2, 4000, 8192),
                   NodeSpec(1, "plain-edge", 2, 4000, 8192)),
            latency_ms=((0.0, 5.0), (5.0, 0.0)),
        )
        config = single_service_config(
            cluster,
            horizon_s=30.0,
            rate=1.0,
            service_kwargs={"replicas": 1},
            power=PowerParams(suspend_after_idle_s=5.0),
        )
        result = run_episode(config, seed=SEED, scorer=flat())
        assert result.energy.transition_j_per_node == (0.0, 0.0)
        # Node 1 burns full idle power for the whole horizon.
        assert result.energy.device_j_per_node[1] == pytest.approx(2.0 * 30.0)

    def test_suspend_of_occupied_node_is_an_invariant_violation(self):
        import simpy
        from sim_helpers import cheap_energy

        from sim.energy import EnergyMeter
        from sim.runtime import ClusterRuntime, Ledger

        env = simpy.Environment()
        config = single_service_config(two_edge_cluster(), horizon_s=10.0)
        cluster = ClusterRuntime(
            env, config.cluster, EnergyMeter(config.cluster, cheap_energy()), Ledger()
        )
        pod = cluster.create_pod(simple_service(), origin="initial")
        cluster.bind(pod, 0, notify=False)
        with pytest.raises(SimulationError, match="non-idle"):
            cluster.suspend(0)

    def test_wake_prices_published_transition_energy(self):
        config = single_service_config(
            mixed_cluster(suspended_third=True),
            horizon_s=20.0,
            rate=1.0,
            service_kwargs={"replicas": 1},
            rebalancer=RebalancerParams(enabled=True, window_s=5.0, budget_per_window=1),
        )

        def wake_only_once(state, graph, budget):
            del graph, budget
            if any(n["suspended"] for n in state["nodes"]):
                return [{"kind": 1, "pod": -1, "dest": 2}]
            return []

        result = run_episode(
            config,
            seed=SEED,
            scorer=flat(),
            proposer=CallableProposer("wake", wake_only_once),
        )
        assert result.ledger["rebalancer_wakes_executed"] == 1
        # wake_energy_j = 8 J (EDGE_CLASS published value).
        assert result.energy.transition_j_per_node[2] == pytest.approx(8.0)
