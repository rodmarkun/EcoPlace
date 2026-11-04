"""DES verification suite, part 1 (TODO 2.5, S6): event ordering, simultaneous-event tie
rules, conservation invariants, RNG stream independence, deterministic seeded replay,
pathological overload, warm-up-bias analysis support, and seed discipline."""

from __future__ import annotations

from dataclasses import replace

import pytest
import simpy
from sim_helpers import (
    DEV_FIXTURES,
    POLICIES,
    fanout_config,
    flat_scorer_fn,
    spread_scorer_fn,
)

from sim import (
    CallableScorer,
    PlacementScriptScorer,
    load_episode_config,
    run_episode,
)
from sim.engine import Simulation
from sim.runtime import Ledger, SimulationError

SEED = 424242


def flat():
    return CallableScorer("flat", flat_scorer_fn)


def spread():
    return CallableScorer("spread", spread_scorer_fn)


class TestEventOrdering:
    def test_simultaneous_events_execute_in_scheduling_order(self):
        """The documented tie rule: equal-time events run in the order they were
        scheduled (SimPy heap key (time, priority, insertion_count))."""
        env = simpy.Environment()
        order = []

        def proc(tag, delay):
            yield env.timeout(delay)
            order.append(tag)

        # Three events all at t=1.0, created in this order.
        env.process(proc("a", 1.0))
        env.process(proc("b", 1.0))
        env.process(proc("c", 1.0))
        # And one created later but scheduled earlier.
        env.process(proc("early", 0.5))
        env.run()
        assert order == ["early", "a", "b", "c"]

    def test_time_never_goes_backwards_in_episode(self):
        result = run_episode(fanout_config(horizon_s=20.0), seed=SEED, scorer=spread())
        times = [d["t"] for d in result.decision_log]
        assert times == sorted(times)
        for r in result.records:
            assert r.arrival_s >= 0.0
            if r.terminal:
                assert r.end_s >= r.arrival_s


class TestConservation:
    def test_busy_episode_conserves_requests_pods_subcalls(self):
        """run_episode calls Ledger.verify() internally; re-check the identities on the
        exported dict so a silent verify() regression cannot pass."""
        result = run_episode(fanout_config(horizon_s=40.0), seed=SEED, scorer=spread())
        led = result.ledger
        terminal = (
            led["requests_completed"]
            + led["requests_timeout"]
            + led["requests_dropped_no_replica"]
            + led["requests_dropped_queue_full"]
        )
        assert terminal + led["requests_in_flight_at_end"] == led["requests_injected"]
        assert led["requests_injected"] > 100
        assert (
            led["subcalls_finished"] + led["subcalls_in_flight_at_end"]
            == led["subcalls_started"]
        )
        assert led["pods_terminated"] + led["pods_live_at_end"] == led["pods_created"]
        assert (
            led["unreserves"]
            == led["bind_failures_injected"] + led["bind_failures_validation"]
        )

    def test_ledger_verify_raises_on_violation(self):
        ledger = Ledger()
        ledger.requests_injected = 5
        ledger.requests_completed = 3
        ledger.requests_in_flight_at_end = 1  # 3 + 1 != 5
        with pytest.raises(SimulationError, match="request conservation"):
            ledger.verify()

    def test_ledger_state_machine_rejects_invalid_transitions(self):
        """R1-2: the ledger tracks independent live-ID state; unknown or duplicate
        transitions raise at the moment they happen — lost work can never be repaired
        into a passing balance."""
        ledger = Ledger()
        rid = ledger.request_injected()
        ledger.request_terminal(rid, "completed")
        with pytest.raises(SimulationError, match="twice or never"):
            ledger.request_terminal(rid, "completed")  # double-termination
        with pytest.raises(SimulationError, match="twice or never"):
            ledger.request_terminal(999, "timeout")  # never injected
        with pytest.raises(SimulationError, match="unknown request terminal"):
            ledger.request_terminal(ledger.request_injected(), "vanished")
        cid = ledger.subcall_started()
        ledger.subcall_finished(cid)
        with pytest.raises(SimulationError, match="twice or never"):
            ledger.subcall_finished(cid)
        ledger.pod_created(0)
        with pytest.raises(SimulationError, match="created twice"):
            ledger.pod_created(0)
        ledger.pod_terminated(0)
        with pytest.raises(SimulationError, match="twice or never"):
            ledger.pod_terminated(0)

    def test_lost_work_is_not_verifiable(self):
        """R1-2 regression: a bind success without a reservation must be impossible,
        and a hand-built ledger claiming one must fail verify()."""
        ledger = Ledger()
        with pytest.raises(SimulationError, match="without a reservation"):
            ledger.bind_confirmed(7)
        forged = Ledger(binds_succeeded=1)  # no reserve ever happened
        forged.close()
        with pytest.raises(SimulationError, match="reserve/bind/unreserve"):
            forged.verify()

    @pytest.mark.parametrize(
        "forged",
        [
            Ledger(requests_completed=1, requests_timeout=-1),
            Ledger(reserves=1, binds_succeeded=1),
            Ledger(scheduling_cycles=-1),
        ],
    )
    def test_forged_or_negative_histories_never_verify(self, forged):
        forged.close()
        with pytest.raises(SimulationError):
            forged.verify()

    def test_reservation_transitions_are_strict(self):
        ledger = Ledger()
        ledger.reserved(1, 0)
        with pytest.raises(SimulationError, match="reserved twice"):
            ledger.reserved(1, 1)
        ledger.bind_confirmed(1)
        with pytest.raises(SimulationError, match="without a reservation"):
            ledger.unreserved(1, injected_failure=True)

    def test_binding_failures_conserve(self):
        config = fanout_config(horizon_s=40.0)
        config = replace(
            config, scheduler=replace(config.scheduler, bind_failure_prob=0.5)
        )
        result = run_episode(config, seed=SEED, scorer=spread())
        led = result.ledger
        assert led["bind_failures_injected"] > 0
        assert led["reserves"] == led["binds_succeeded"] + led["unreserves"]
        # Every pod that ever bound is bound exactly once at a time.
        assert led["pods_created"] >= led["binds_succeeded"] - led["pods_terminated"]


class TestOverload:
    def test_pathological_overload_is_accounted_never_lost(self):
        config = load_episode_config(DEV_FIXTURES / "overload.json")
        result = run_episode(config, seed=SEED, scorer=flat())
        led = result.ledger
        # Far more offered than served; drops must appear and books must close.
        assert led["requests_injected"] > 1000
        assert led["requests_dropped_queue_full"] > 0
        terminal = (
            led["requests_completed"]
            + led["requests_timeout"]
            + led["requests_dropped_no_replica"]
            + led["requests_dropped_queue_full"]
        )
        assert terminal + led["requests_in_flight_at_end"] == led["requests_injected"]
        # Energy stays finite and positive.
        assert 0.0 < result.energy.device_j < float("inf")

    def test_cross_node_burst_is_admitted_at_destination_queue(self):
        """R2-1: calls in network transit cannot bypass the child's queue bound."""
        from sim import (
            ArrivalSpec,
            CallEdgeSpec,
            CallSpec,
            ClusterSpec,
            EpisodeConfig,
            NodeClassSpec,
            NodeSpec,
            ServiceSpec,
            TemplateSpec,
            WorkloadSpec,
            det,
        )

        cls = NodeClassSpec("n", 1.0, 2.0)
        cluster = ClusterSpec(
            classes=(cls,),
            nodes=(
                NodeSpec(0, "n", 2, 100_000, 100_000),
                NodeSpec(1, "n", 2, 100_000, 100_000),
            ),
            latency_ms=((0.0, 1000.0), (1000.0, 0.0)),
        )
        root = ServiceSpec(
            0, "root", 0, 1, 1, det(0.0), 0,
            concurrency=100, image_mb=0,
        )
        child = ServiceSpec(
            1, "child", 0, 1, 1, det(100.0), 0,
            concurrency=1, queue_capacity=1, image_mb=0,
        )
        template = TemplateSpec(
            0,
            "burst",
            CallSpec(0, children=(CallEdgeSpec(CallSpec(1)),)),
            ArrivalSpec("deterministic", 50.0),
        )
        config = EpisodeConfig(
            cluster,
            WorkloadSpec((root, child), (template,)),
            horizon_s=5.0,
            arrival_cutoff_s=1.0,
        )

        def pin_service(pod, node, ctx):
            del ctx
            return 1.0 if node["id"] == pod["service"] else 0.0

        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("pin", pin_service)
        )
        assert result.ledger["requests_dropped_queue_full"] > 30
        # At the horizon the child can have at most one call in service and one queued.
        assert result.ledger["subcalls_in_flight_at_end"] <= 2


class TestDeterministicReplay:
    def test_same_seed_bitwise_identical(self):
        config = fanout_config(horizon_s=30.0)
        a = run_episode(config, seed=SEED, scorer=spread())
        b = run_episode(config, seed=SEED, scorer=spread())
        assert a.digest() == b.digest()
        assert a.to_canonical_dict() == b.to_canonical_dict()

    def test_different_seed_differs(self):
        config = fanout_config(horizon_s=30.0)
        a = run_episode(config, seed=SEED, scorer=spread())
        b = run_episode(config, seed=SEED + 1, scorer=spread())
        assert a.digest() != b.digest()

    def test_native_policy_path_replays_bitwise(self):
        import placementscript as ps

        scorer = PlacementScriptScorer(
            ps.Policy((POLICIES / "least_allocated.ps").read_text())
        )
        config = load_episode_config(DEV_FIXTURES / "small_cluster.json")
        a = run_episode(config, seed=SEED, scorer=scorer)
        b = run_episode(config, seed=SEED, scorer=scorer)
        assert a.digest() == b.digest()


class TestRngStreamIndependence:
    """docs/SEEDS.md: component streams are independent — consuming one stream must not
    perturb another. The environment (arrivals) must be identical under CRN when only
    the policy or an unrelated component changes."""

    def test_arrivals_identical_across_scorers(self):
        config = fanout_config(horizon_s=25.0)
        a = run_episode(config, seed=SEED, scorer=spread())
        b = run_episode(config, seed=SEED, scorer=flat())
        arrivals_a = sorted(r.arrival_s for r in a.records)
        arrivals_b = sorted(r.arrival_s for r in b.records)
        assert arrivals_a == arrivals_b  # bitwise: same floats

    def test_arrivals_identical_across_lb_policies(self):
        config = fanout_config(horizon_s=25.0)
        a = run_episode(config, seed=SEED, scorer=spread(), lb_policy="round_robin")
        b = run_episode(config, seed=SEED, scorer=spread(), lb_policy="random")
        assert sorted(r.arrival_s for r in a.records) == sorted(
            r.arrival_s for r in b.records
        )

    def test_arrivals_unaffected_by_binding_stream(self):
        """Turning on injected bind failures consumes the binding stream differently;
        arrival times (a different stream) must not move."""
        config = fanout_config(horizon_s=25.0)
        noisy = replace(
            config, scheduler=replace(config.scheduler, bind_failure_prob=0.3)
        )
        a = run_episode(config, seed=SEED, scorer=spread())
        b = run_episode(noisy, seed=SEED, scorer=spread())
        assert sorted(r.arrival_s for r in a.records) == sorted(
            r.arrival_s for r in b.records
        )

    def test_per_request_service_draws_identical_across_placements(self):
        """R1-7: environmental draws are keyed by request identity, so the k-th request
        of a template receives the *same* service-time variate no matter where the
        replicas are placed. Observable witness: with a zero-latency matrix, ample
        capacity, and one replica per node vs. both on one node, each request's
        end-to-end latency (== its service time at negligible load) is bitwise equal
        across the two placements."""
        from sim_helpers import simple_service

        from sim import (
            ArrivalSpec,
            CallSpec,
            ClusterSpec,
            EpisodeConfig,
            NodeClassSpec,
            NodeSpec,
            TemplateSpec,
            WorkloadSpec,
        )

        cls = NodeClassSpec(name="n", power_idle_w=1.0, power_max_w=2.0)
        cluster = ClusterSpec(
            classes=(cls,),
            nodes=(NodeSpec(0, "n", 2, 100000, 100000), NodeSpec(1, "n", 2, 100000, 100000)),
            latency_ms=((0.0, 0.0), (0.0, 0.0)),
        )
        svc = simple_service(mean_s=0.02, replicas=2, cpu_demand=0, concurrency=64)
        template = TemplateSpec(
            id=0, name="t", root=CallSpec(service=0), arrival=ArrivalSpec("poisson", 2.0)
        )
        config = EpisodeConfig(
            cluster=cluster,
            workload=WorkloadSpec(services=(svc,), templates=(template,)),
            horizon_s=60.0,
            arrival_cutoff_s=50.0,
        )
        pack = run_episode(config, seed=SEED, scorer=flat())  # both replicas node 0
        spread_r = run_episode(config, seed=SEED, scorer=spread())  # one per node
        # Different placements actually happened.
        nodes_pack = {p.node_id for p in pack.pods if p.node_id is not None}
        nodes_spread = {p.node_id for p in spread_r.pods if p.node_id is not None}
        assert nodes_pack == {0} and nodes_spread == {0, 1}
        lat_a = [
            (r.rid, r.latency_s) for r in pack.records if r.status == "completed"
        ]
        lat_b = [
            (r.rid, r.latency_s) for r in spread_r.records if r.status == "completed"
        ]
        assert len(lat_a) > 50
        assert lat_a == lat_b  # bitwise: same variate to the same request

    def test_hpa_pod_draws_ignore_interleaved_rebalancer_pods(self):
        """R2-2: controller-origin semantic pod identities preserve CRN pairing."""
        from sim_helpers import cheap_energy, single_service_config, two_edge_cluster

        from analysis.tracking import spawn_rng
        from sim.energy import EnergyMeter
        from sim.runtime import ClusterRuntime

        config = single_service_config(two_edge_cluster(), horizon_s=10.0)

        def identities(with_move):
            env = simpy.Environment()
            ledger = Ledger()
            cluster = ClusterRuntime(
                env,
                config.cluster,
                EnergyMeter(config.cluster, cheap_energy()),
                ledger,
            )
            service = config.workload.services[0]
            cluster.create_pod(
                service, "initial", rng_identity=("initial", 0)
            )
            if with_move:
                cluster.create_pod(
                    service,
                    "rebalancer",
                    rng_identity=("rebalancer", "initial", 0, "move", 0),
                )
            return [
                cluster.create_pod(
                    service, "hpa", rng_identity=("hpa", 0, index)
                ).rng_identity
                for index in range(3)
            ]

        plain = identities(False)
        moved = identities(True)
        assert plain == moved
        draws_plain = [
            spawn_rng(SEED, "sim", "cold-start", 0, *identity).random()
            for identity in plain
        ]
        draws_moved = [
            spawn_rng(SEED, "sim", "cold-start", 0, *identity).random()
            for identity in moved
        ]
        assert draws_plain == draws_moved


class TestWarmupSupport:
    def test_windowed_stats_exclude_warmup(self):
        config = fanout_config(horizon_s=40.0)
        result = run_episode(config, seed=SEED, scorer=spread())
        full = result.latency_stats(0.0)
        steady = result.latency_stats(10.0)
        assert steady["count"] < full["count"]
        # Early requests (cold starts, image pulls) are excluded from the window.
        early = [r for r in result.records if r.arrival_s < 10.0]
        assert full["count"] - steady["count"] == float(
            sum(1 for r in early if r.status == "completed")
        )

    def test_empty_window_reports_zero_count(self):
        config = fanout_config(horizon_s=20.0)
        result = run_episode(config, seed=SEED, scorer=spread())
        assert result.latency_stats(19.99, 20.0)["count"] in (0.0, 1.0, 2.0)


class TestSeedDiscipline:
    def test_seed_is_required(self):
        config = fanout_config(horizon_s=10.0)
        with pytest.raises(TypeError):
            Simulation(config, scorer=spread())  # no seed

    def test_bool_seed_rejected(self):
        config = fanout_config(horizon_s=10.0)
        with pytest.raises(TypeError, match="seed"):
            Simulation(config, True, spread())

    def test_no_entropy_default_constructions_in_experiment_code(self):
        """Grep enforcement across sim/, baselines/, and evolution/ (R5-5)."""
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        experiment_dirs = tuple(root / name for name in ("sim", "baselines", "evolution"))
        banned = [
            r"default_rng\(\s*\)",
            r"np\.random\.seed",
            r"random\.Random\(\s*\)",
            r"\brandom\.seed\(\s*\)",
            r"time\.time\(\)",
            r"datetime\.now",
            r"uuid4",
            r"os\.urandom",
            r"seed\s*(?::\s*[\w|\[\] ]+)?=\s*(?:None|\d)",  # defaulted seed params
        ]
        offenders = []
        for path in sorted(
            path
            for source_dir in experiment_dirs
            for path in source_dir.rglob("*.py")
        ):
            if path.name in ("run_gate.py", "holdout_gate.py", "fetch_specpower.py"):
                # The one-shot gate drivers record tz-aware execution timestamps for
                # bookkeeping only, and the source-pinning tool records its access
                # date (docs/SEEDS.md non-goal: operational timestamps never feed
                # back into results; fetch_specpower.py is build-time tooling the
                # simulator never imports).
                continue
            text = path.read_text()
            for pattern in banned:
                for m in re.finditer(pattern, text):
                    offenders.append(
                        f"{path.relative_to(root).as_posix()}: {m.group(0)!r}"
                    )
        assert not offenders, offenders
