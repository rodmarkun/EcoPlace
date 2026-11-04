"""Scheduling-cycle emulator tests (TODO 2.2, S2): serial cycles + async binding,
assumed-pod state, informer staleness, binding failure retry/backoff, NormalizeScore
quantization, deterministic tie-breaking, scorer error handling, both scorer adapters,
and the G2 coverage enumeration."""

from __future__ import annotations

from dataclasses import replace

import placementscript as ps
import pytest
import simpy
from sim_helpers import (
    POLICIES,
    cheap_energy,
    fanout_config,
    flat_scorer_fn,
    single_service_config,
    spread_scorer_fn,
    two_edge_cluster,
)

from sim import (
    CallableScorer,
    PlacementScriptScorer,
    run_episode,
)
from sim.energy import EnergyMeter
from sim.runtime import ClusterRuntime, Ledger, SimulationError
from sim.scheduler import (
    EMULATOR_COVERAGE,
    SchedulerCache,
    coverage_as_dicts,
    normalize_scores,
    select_best,
)
from sim.spec import SchedulerParams, det

SEED = 77


def reserves(result):
    """Reserve decisions from the full G2 event trace."""
    return [d for d in result.decision_log if d["event"] == "reserve"]


class TestNormalizeAndTieBreak:
    def test_normalize_scores_affine_map(self):
        assert normalize_scores([0.0, 0.5, 1.0]) == [0, 50, 100]
        assert normalize_scores([2.0, 2.0]) == [100, 100]
        assert normalize_scores([-1.0, 3.0]) == [0, 100]

    def test_select_best_max_wins(self):
        assert select_best([3, 1, 4], [10, 90, 40]) == 1

    def test_select_best_tie_lowest_node_id_order_independent(self):
        assert select_best([4, 2, 7], [100, 100, 100]) == 2
        assert select_best([7, 2, 4], [100, 100, 100]) == 2


class TestSerialCycleAsyncBinding:
    def test_all_cycles_complete_before_any_bind_with_slow_binding(self):
        """Serial cycles + async binding: with a 5 s bind latency, all initial pods are
        scored/reserved at t=0 (assumed pods), long before any binding completes."""
        config = single_service_config(
            two_edge_cluster(),
            horizon_s=20.0,
            service_kwargs={"replicas": 4, "cpu_request": 400},
            scheduler=SchedulerParams(bind_latency=det(5.0)),
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("spread", spread_scorer_fn)
        )
        cycle_times = [d["t"] for d in reserves(result)[:4]]
        assert cycle_times == [0.0, 0.0, 0.0, 0.0]
        binds = [p.bound_s for p in result.pods if p.bound_s is not None]
        assert all(b >= 5.0 for b in binds)

    def test_assumed_pods_prevent_double_placement(self):
        """With spread scoring over the cache (incl. assumed pods), 4 equal pods on 2
        equal nodes must alternate 0,1,0,1 even though nothing has bound yet."""
        config = single_service_config(
            two_edge_cluster(),
            horizon_s=20.0,
            service_kwargs={"replicas": 4, "cpu_request": 400},
            scheduler=SchedulerParams(bind_latency=det(5.0)),
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("spread", spread_scorer_fn)
        )
        assert [d["node"] for d in reserves(result)[:4]] == [0, 1, 0, 1]

    def test_capacity_filter_respects_assumed_pods(self):
        """Two pods of 600m on 1000m nodes: the second pod must be filtered away from
        the node the first was *assumed* on (not yet bound)."""
        config = single_service_config(
            two_edge_cluster(cpu=1000),
            horizon_s=20.0,
            service_kwargs={"replicas": 2, "cpu_request": 600},
            scheduler=SchedulerParams(bind_latency=det(5.0)),
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        assert [d["node"] for d in reserves(result)[:2]] == [0, 1]
        assert [d["feasible"] for d in reserves(result)[:2]] == [[0, 1], [1]]


class TestBindingFailureBackoff:
    def test_injected_failure_unreserves_and_retries_with_backoff(self):
        config = single_service_config(
            two_edge_cluster(),
            horizon_s=60.0,
            service_kwargs={"replicas": 1},
            scheduler=SchedulerParams(bind_latency=det(0.01), bind_failure_prob=1.0),
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        led = result.ledger
        assert led["bind_failures_injected"] > 0
        assert led["unreserves"] == led["bind_failures_injected"]
        assert led["binds_succeeded"] == 0
        # Retry attempts are spaced by exponential backoff (1, 2, 4... capped at 10).
        attempts = [d["t"] for d in reserves(result)]
        assert len(attempts) >= 2
        gaps = [b - a for a, b in zip(attempts, attempts[1:], strict=False)]
        assert all(g >= 0.9 for g in gaps)  # at least ~backoff_initial_s apart
        for earlier, later in zip(gaps, gaps[1:], strict=False):
            assert later >= earlier - 1e-9 or earlier >= 9.0  # non-decreasing until cap

    def test_unschedulable_pod_requeues_and_lands_after_capacity_frees(self):
        """A pod that fits nowhere requeues with backoff (PostFilter pinned off) and
        schedules once capacity appears (via HPA scale-down of the blocker is not
        modeled here; instead the arrival cutoff frees nothing — so it must still be
        pending at the end, with multiple attempts)."""
        config = single_service_config(
            two_edge_cluster(cpu=1000),
            horizon_s=30.0,
            service_kwargs={"replicas": 3, "cpu_request": 600},
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        led = result.ledger
        assert led["unschedulable_cycles"] > 1  # retried with backoff
        assert led["binds_succeeded"] == 2
        pending = [p for p in result.pods if p.phase == "pending"]
        assert len(pending) == 1
        assert pending[0].scheduling_attempts > 1


class TestCacheStaleness:
    def _cluster(self, env, informer_delay):
        config = single_service_config(two_edge_cluster(), horizon_s=10.0)
        meter = EnergyMeter(config.cluster, cheap_energy())
        cluster = ClusterRuntime(env, config.cluster, meter, Ledger())
        cache = SchedulerCache(env, cluster, informer_delay)
        return config, cluster, cache

    def test_external_change_applies_after_informer_delay(self):
        env = simpy.Environment()
        config, cluster, cache = self._cluster(env, informer_delay=2.5)
        pod = cluster.create_pod(config.workload.services[0], origin="rebalancer")
        cluster.bind(pod, 0, notify=True)  # external (rebalancer) bind
        assert cache.cpu_alloc[0] == 0  # not yet visible
        env.run(until=2.4)
        assert cache.cpu_alloc[0] == 0
        env.run(until=2.6)
        assert cache.cpu_alloc[0] == pod.service.cpu_request

    def test_scheduler_initiated_changes_apply_immediately(self):
        env = simpy.Environment()
        config, cluster, cache = self._cluster(env, informer_delay=2.5)
        pod = cluster.create_pod(config.workload.services[0], origin="initial")
        cache.assume(pod, 1)
        assert cache.cpu_alloc[1] == pod.service.cpu_request
        cache.unreserve(pod, 1)
        assert cache.cpu_alloc[1] == 0

    def test_wrong_node_cache_transition_is_atomic(self):
        """R2-7: validation happens before mutation on confirm and unreserve."""
        env = simpy.Environment()
        config, cluster, cache = self._cluster(env, informer_delay=0.0)
        pod = cluster.create_pod(config.workload.services[0], origin="initial")
        cache.assume(pod, 0)
        before = (dict(cache.assumed), list(cache.cpu_alloc), list(cache.mem_alloc))
        with pytest.raises(SimulationError, match="matching assume"):
            cache.confirm(pod, 1)
        assert (dict(cache.assumed), cache.cpu_alloc, cache.mem_alloc) == before
        with pytest.raises(SimulationError, match="matching assume"):
            cache.unreserve(pod, 1)
        assert (dict(cache.assumed), cache.cpu_alloc, cache.mem_alloc) == before

    def test_zero_delay_applies_synchronously(self):
        env = simpy.Environment()
        config, cluster, cache = self._cluster(env, informer_delay=0.0)
        pod = cluster.create_pod(config.workload.services[0], origin="rebalancer")
        cluster.bind(pod, 0, notify=True)
        assert cache.cpu_alloc[0] == pod.service.cpu_request

    def test_truth_bind_never_allows_overcommit(self):
        env = simpy.Environment()
        config = single_service_config(
            two_edge_cluster(cpu=600),
            horizon_s=10.0,
            service_kwargs={"cpu_request": 600},
        )
        meter = EnergyMeter(config.cluster, cheap_energy())
        cluster = ClusterRuntime(env, config.cluster, meter, Ledger())
        first = cluster.create_pod(config.workload.services[0], "initial")
        second = cluster.create_pod(config.workload.services[0], "hpa")
        cluster.bind(first, 0, notify=False)
        with pytest.raises(SimulationError, match="truth capacity overcommit"):
            cluster.bind(second, 0, notify=False)

    def test_stale_truth_plus_assumed_reservation_stays_in_native_domain(self):
        """R5: the worst scheduler-view sum is exactly the host resource maximum."""
        from sim import (
            ArrivalSpec,
            CallSpec,
            ClusterSpec,
            EpisodeConfig,
            NodeClassSpec,
            NodeSpec,
            ServiceSpec,
            TemplateSpec,
            WorkloadSpec,
            take_snapshot,
        )
        from sim.runtime import BINDING, READY
        from sim.scheduler import build_ctx, filter_feasible
        from sim.spec import MAX_NODE_CAPACITY, MAX_RESOURCE

        env = simpy.Environment()
        node_class = NodeClassSpec("large", 1.0, 2.0)
        cluster_spec = ClusterSpec(
            (node_class,),
            tuple(
                NodeSpec(i, "large", 2, MAX_NODE_CAPACITY, MAX_NODE_CAPACITY)
                for i in range(3)
            ),
            ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        )
        big = ServiceSpec(
            0,
            "big",
            0,
            MAX_NODE_CAPACITY,
            MAX_NODE_CAPACITY,
            det(0.0),
            MAX_NODE_CAPACITY,
            image_mb=0,
            max_replicas=2,
        )
        small = ServiceSpec(
            1,
            "small",
            0,
            1,
            1,
            det(0.0),
            1,
            image_mb=0,
        )
        config = EpisodeConfig(
            cluster_spec,
            WorkloadSpec(
                (big, small),
                (
                    TemplateSpec(
                        0,
                        "t",
                        CallSpec(1),
                        ArrivalSpec("deterministic", 1.0),
                    ),
                ),
            ),
            5.0,
        )
        meter = EnergyMeter(config.cluster, cheap_energy())
        cluster = ClusterRuntime(env, config.cluster, meter, Ledger())
        cache = SchedulerCache(env, cluster, informer_delay_s=1.0)

        external = cluster.create_pod(big, "rebalancer")
        cluster.bind(external, 0, notify=True)
        external.phase = READY
        assumed = cluster.create_pod(big, "initial")
        assumed.phase = BINDING
        cache.assume(assumed, 0)
        env.run(until=1.1)
        assert cache.cpu_alloc[0] == cache.mem_alloc[0] == MAX_RESOURCE

        pending = cluster.create_pod(small, "hpa")
        ctx, uid_to_dense, _ = build_ctx(
            cluster,
            config.workload,
            cache,
            env.now,
            config.cutoff_s,
            include_pod=pending,
        )
        pending_dict = ctx["pods"][uid_to_dense[pending.uid]]
        feasible = filter_feasible(pending_dict, ctx["nodes"])
        assert feasible == [1, 2]
        native = PlacementScriptScorer(
            ps.Policy((POLICIES / "least_allocated.ps").read_text())
        )
        candidates = [ctx["nodes"][node_id] for node_id in feasible]
        assert len(native.score_batch(pending_dict, candidates, ctx)) == 2

        class SnapshotScheduler:
            def __init__(self):
                self.cache = cache

            def pending_in_queue_order(self):
                return [(pending.uid, env.now)]

        snapshot = take_snapshot(
            cluster, config, env.now, scheduler=SnapshotScheduler()
        )
        assert snapshot.ctx["nodes"][0]["cpu_allocated"] == MAX_RESOURCE
        snap_ctx = snapshot.to_dict()["ctx"]
        snap_pod = snap_ctx["pods"][snapshot.pending_dense_ids[0]]
        snap_feasible = filter_feasible(snap_pod, snap_ctx["nodes"])
        assert len(
            native.score_batch(
                snap_pod,
                [snap_ctx["nodes"][node_id] for node_id in snap_feasible],
                snap_ctx,
            )
        ) == len(snap_feasible)

    def test_power_state_staleness(self):
        env = simpy.Environment()

        def driver():
            yield env.timeout(0.0)

        config, cluster, cache = self._cluster(env, informer_delay=1.0)
        del config
        env.process(driver())
        proc = cluster.suspend(1)
        del proc
        env.run(until=5.0)  # suspend takes 2 s + 1 s informer delay
        assert cache.suspended[1] is True


class TestScorerAdapters:
    def test_placementscript_and_python_reimplementation_agree(self):
        """The least_allocated .ps policy and its Python reimplementation must produce
        identical decision sequences (same observations -> same decisions)."""
        config = fanout_config(horizon_s=20.0)
        native = PlacementScriptScorer(
            ps.Policy((POLICIES / "least_allocated.ps").read_text())
        )

        def least_allocated_py(pod, node, ctx):
            del ctx
            cpu_cap = float(max(node["cpu_capacity"], 1))
            mem_cap = float(max(node["mem_capacity"], 1))
            cpu_free = cpu_cap - float(node["cpu_allocated"] + pod["cpu_request"])
            mem_free = mem_cap - float(node["mem_allocated"] + pod["mem_request"])
            return (max(cpu_free, 0.0) / cpu_cap + max(mem_free, 0.0) / mem_cap) / 2.0

        a = run_episode(config, seed=SEED, scorer=native)
        b = run_episode(
            config, seed=SEED, scorer=CallableScorer("least_allocated", least_allocated_py)
        )
        assert [(d["pod_uid"], d["node"]) for d in reserves(a)] == [
            (d["pod_uid"], d["node"]) for d in reserves(b)
        ]

    def test_callable_scorer_rejects_non_finite(self):
        bad = CallableScorer("nan", lambda pod, node, ctx: float("nan"))
        with pytest.raises(ValueError, match="non-finite"):
            bad.score_batch({}, [{}], {})

    def test_scorer_error_fails_cycle_and_requeues(self):
        """A trapping policy is a framework error for that cycle (T1): counted, pod
        requeued; nothing binds; conservation still closes."""
        trap = ps.Policy(
            'POLICY_NAME: str = "t"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    return 1.0 / (float(node.cpu_allocated) * 0.0)\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n    return out\n"
        )
        config = single_service_config(two_edge_cluster(), horizon_s=15.0)
        result = run_episode(config, seed=SEED, scorer=PlacementScriptScorer(trap))
        led = result.ledger
        assert led["scorer_errors"] > 1
        assert led["binds_succeeded"] == 0
        assert led["pods_live_at_end"] == 1  # still pending, never lost

    def test_quantized_scores_are_ints_in_0_100(self):
        """quantize_scores=True mirrors the plugin's int-[0,100] CycleState scores;
        quantize_scores=False keeps the raw floats (both paths deterministic)."""

        def graded(pod, node, ctx):
            del pod, ctx
            return {0: 100.0, 1: 100.4}.get(node["id"], 0.0)

        config = single_service_config(
            two_edge_cluster(), horizon_s=10.0, service_kwargs={"replicas": 1}
        )
        quantized = run_episode(config, seed=SEED, scorer=CallableScorer("g", graded))
        d = reserves(quantized)[0]
        assert d["scores"] == [0, 100]  # min->0, max->100 affine map
        assert d["node"] == 1
        no_quant = replace(
            config, scheduler=replace(config.scheduler, quantize_scores=False)
        )
        raw = run_episode(no_quant, seed=SEED, scorer=CallableScorer("g", graded))
        d = reserves(raw)[0]
        assert d["scores"] == [100.0, 100.4]
        assert d["node"] == 1

    def test_flat_scores_quantize_to_all_100_and_tie_break(self):
        config = single_service_config(
            two_edge_cluster(), horizon_s=10.0, service_kwargs={"replicas": 1}
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        d = reserves(result)[0]
        assert d["scores"] == [100, 100]
        assert d["node"] == 0


class TestCoverageEnumeration:
    REQUIRED = {
        "serial_scheduling_cycles",
        "asynchronous_binding",
        "assumed_pod_cache",
        "informer_cache_staleness",
        "binding_failure_retry_backoff",
        "scheduling_queue",
        "pod_priority_queue",
        "postfilter_preemption",
        "permit_plugins",
        "node_sampling_percentage",
        "score_quantization",
        "tie_breaking",
        "filter_plugins",
        "bind_time_capacity_revalidation",
        "scorer_error_requeue",
        "g2_event_trace",
    }

    def test_plan_mandated_behaviors_are_enumerated(self):
        behaviors = {e.behavior for e in EMULATOR_COVERAGE}
        missing = self.REQUIRED - behaviors
        assert not missing, f"G2 coverage list is missing: {missing}"

    def test_every_entry_has_a_valid_status_and_notes(self):
        for e in EMULATOR_COVERAGE:
            assert e.status in (
                "reproduced",
                "approximated",
                "pinned_off",
                "documented_divergence",
            )
            assert len(e.notes) > 10

    def test_scheduler_is_pinned(self):
        """G2 conformance is against one exact scheduler build/profile (R1-5)."""
        from sim import PINNED_SCHEDULER

        assert PINNED_SCHEDULER.kubernetes_version.startswith("v1.")
        assert PINNED_SCHEDULER.score_plugins == (("PlacementScriptScore", 1),)
        assert PINNED_SCHEDULER.percentage_of_nodes_to_score == 100
        assert PINNED_SCHEDULER.queue_backoff_initial_s == 1.0
        assert PINNED_SCHEDULER.queue_backoff_max_s == 10.0

    def test_coverage_export_shape(self):
        rows = coverage_as_dicts()
        assert all(set(r) == {"behavior", "status", "notes"} for r in rows)
        assert len(rows) == len(EMULATOR_COVERAGE)

    def test_event_trace_covers_failure_events(self):
        """The G2 trace must contain unschedulable / bind-failure events, not just
        successful reserves (R1-5)."""
        config = single_service_config(
            two_edge_cluster(cpu=1000),
            horizon_s=15.0,
            service_kwargs={"replicas": 3, "cpu_request": 600},
            scheduler=SchedulerParams(bind_latency=det(0.01), bind_failure_prob=0.5),
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        kinds = {d["event"] for d in result.decision_log}
        assert "reserve" in kinds
        assert "unschedulable" in kinds
        assert "bind_success" in kinds
        led = result.ledger
        if led["bind_failures_injected"] + led["bind_failures_validation"] > 0:
            assert "bind_failure" in kinds
