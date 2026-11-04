"""DES verification suite, part 2 (TODO 2.5, S6): hand-solvable micro-scenarios checked
against closed forms in CI.

- M/M/1: mean sojourn W = 1/(mu - lambda), plus Little's law on the same run.
- Feed-forward Jackson network (A ->(p) B, forward handoff): E[T] = W_A + p * W_B with
  W_i = 1/(mu_i - lambda_i), lambda_B = p * lambda (Burke/Jackson).
- Two-node placement cases with enumerable optima: L0 closed-form energy against hand
  arithmetic, policy argmin against brute-force enumeration, and exact emulator decision
  sequences.

Tolerances: the DES checks are statistical with pinned seeds — observed relative error
across seeds is under 2% (see PHASE2 notes); the assertion bound is 4%. The L0/emulator
checks are exact.
"""

from __future__ import annotations

import math

import placementscript as ps
import pytest
from sim_helpers import DEV_FIXTURES, POLICIES, flat_scorer_fn

from sim import (
    ArrivalSpec,
    BoundaryWeights,
    CallableScorer,
    CallSpec,
    ClusterSpec,
    EpisodeConfig,
    L0Evaluator,
    NodeClassSpec,
    NodeSpec,
    PlacementScriptScorer,
    ServiceSpec,
    SnapshotLibrary,
    TemplateSpec,
    WorkloadSpec,
    brute_force_optimum,
    initial_snapshot,
    load_episode_config,
    run_episode,
)
from sim.spec import det

SEED = 20260716
WARMUP_S = 50.0
TOLERANCE = 0.04


class ForcedScorer:
    """Scores 1.0 for the forced node, 0.0 otherwise — drives hand-checkable
    assignments through the same shared placement pipeline as any real policy."""

    name = "py:forced"

    def __init__(self, choices):
        self._choices = choices

    def score_batch(self, pod, nodes, ctx):
        del ctx
        want = self._choices[pod["id"]]
        return [1.0 if n["id"] == want else 0.0 for n in nodes]


class TestMM1:
    @pytest.mark.parametrize("seed", [SEED, SEED + 1, SEED + 2])
    def test_mean_sojourn_matches_closed_form(self, seed):
        """Multi-seed statistical acceptance (R1-13), with end-of-horizon
        right-censoring excluded: only requests arriving well before the horizon enter
        the estimate (in-flight-at-end requests would otherwise truncate the sample
        toward short sojourns)."""
        config = load_episode_config(DEV_FIXTURES / "mm1.json")
        # Fixture parameters (pinned): lambda = 8/s, E[S] = 0.08s -> mu = 12.5, rho = 0.64.
        lam, mu = 8.0, 12.5
        w_closed = 1.0 / (mu - lam)
        result = run_episode(
            config, seed=seed, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        window_end = config.horizon_s - 50.0
        stats = result.latency_stats(WARMUP_S, window_end)
        # Censoring check: every request arriving in the window must be terminal.
        in_flight_in_window = [
            r
            for r in result.records
            if r.status == "in_flight" and WARMUP_S <= r.arrival_s < window_end
        ]
        assert not in_flight_in_window, "window still right-censored; widen the margin"
        assert stats["count"] > 15000
        assert abs(stats["mean"] - w_closed) / w_closed < TOLERANCE

    def test_littles_law_with_independently_sampled_occupancy(self):
        """Little's law with a genuinely independent L (R1-13): N(t) is *sampled* from
        the simulator's live-request state by a monitor process, not derived from the
        same sojourn records that define W. L_sampled ~= lambda_eff * W_records."""
        from sim import Simulation

        config = load_episode_config(DEV_FIXTURES / "mm1.json")
        window = (WARMUP_S, config.horizon_s - 50.0)
        sim = Simulation(config, SEED, CallableScorer("flat", flat_scorer_fn))
        samples: list[int] = []

        def monitor():
            while True:
                yield sim.env.timeout(0.5)
                if window[0] <= sim.env.now < window[1]:
                    samples.append(len(sim.ledger._live_requests))

        sim.env.process(monitor())
        result = sim.run()
        done = [
            r
            for r in result.records
            if r.status == "completed" and window[0] <= r.arrival_s < window[1]
        ]
        duration = window[1] - window[0]
        lam_eff = len(done) / duration
        w_obs = sum(r.latency_s for r in done) / len(done)
        l_sampled = sum(samples) / len(samples)
        assert len(samples) > 1000
        assert abs(l_sampled - lam_eff * w_obs) / (lam_eff * w_obs) < TOLERANCE

    def test_no_losses_below_saturation(self):
        config = load_episode_config(DEV_FIXTURES / "mm1.json")
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        led = result.ledger
        assert led["requests_timeout"] == 0
        assert led["requests_dropped_queue_full"] == 0
        # Drops can only happen before the replica is ready (startup window).
        assert all(
            r.arrival_s < WARMUP_S
            for r in result.records
            if r.status == "dropped_no_replica"
        )


class TestJacksonNetwork:
    def test_mean_end_to_end_matches_closed_form(self):
        """Pooled over three pinned seeds (R1-13: never a single-seed acceptance), with
        right-censoring excluded via the window end."""
        config = load_episode_config(DEV_FIXTURES / "jackson.json")
        # Fixture (pinned): lambda = 5/s; A: E[S]=0.08 (mu 12.5); B: E[S]=0.1 (mu 10);
        # p(A->B) = 0.6 -> lambda_B = 3.
        lam, mu_a, mu_b, p = 5.0, 12.5, 10.0, 0.6
        t_closed = 1.0 / (mu_a - lam) + p * (1.0 / (mu_b - p * lam))
        window_end = config.horizon_s - 50.0
        total, count = 0.0, 0
        for seed in (SEED, SEED + 1, SEED + 2):
            result = run_episode(
                config, seed=seed, scorer=CallableScorer("flat", flat_scorer_fn)
            )
            stats = result.latency_stats(WARMUP_S, window_end)
            total += stats["mean"] * stats["count"]
            count += int(stats["count"])
        assert count > 30000
        pooled_mean = total / count
        assert abs(pooled_mean - t_closed) / t_closed < TOLERANCE

    def test_branch_visit_fraction(self):
        """The routing stream must hit B for ~p of requests (binomial, n>10k)."""
        config = load_episode_config(DEV_FIXTURES / "jackson.json")
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        led = result.ledger
        visited_b = led["subcalls_started"]
        n = led["requests_completed"] + led["requests_in_flight_at_end"]
        frac = visited_b / n
        assert abs(frac - 0.6) < 0.02


def _two_node_config() -> EpisodeConfig:
    """Two heterogeneous edge nodes, hand-checkable energy arithmetic.

    node 0: idle 2 W, max 10 W (steep curve); node 1: idle 5 W, max 6 W (flat curve);
    capacity 1000 millicores each. Service: E[S]=0.05s deterministic, demand 500,
    request 400 -> at most 2 pods per node. lambda = 4/s per template instance.
    """
    frugal_idle = NodeClassSpec(name="steep", power_idle_w=2.0, power_max_w=10.0)
    flat_curve = NodeClassSpec(name="flat", power_idle_w=5.0, power_max_w=6.0)
    cluster = ClusterSpec(
        classes=(frugal_idle, flat_curve),
        nodes=(NodeSpec(0, "steep", 2, 1000, 4096), NodeSpec(1, "flat", 2, 1000, 4096)),
        latency_ms=((0.0, 10.0), (10.0, 0.0)),
    )
    svc = ServiceSpec(
        id=0,
        name="s",
        app=0,
        cpu_request=400,
        mem_request=128,
        service_time=det(0.05),
        cpu_demand=500,
        image_mb=1,
        initial_replicas=2,
        max_replicas=4,
    )
    template = TemplateSpec(
        id=0, name="t", root=CallSpec(service=0), arrival=ArrivalSpec("poisson", 4.0)
    )
    return EpisodeConfig(
        cluster=cluster,
        workload=WorkloadSpec(services=(svc,), templates=(template,)),
        horizon_s=100.0,
    )


class TestTwoNodeEnumerableOptimum:
    def test_l0_energy_matches_hand_arithmetic(self):
        """Both replicas forced onto node 0: busy = (lambda/2) * E[S] * demand = 50
        millicores per replica -> u = 100/1000 = 0.1; P = 2 + (10-2)*0.1 = 2.8 W.
        Node 1 idles at 5 W. Horizon 100 s -> device = 280 + 500 = 780 J. No cloud
        nodes, single-service template -> cloud = network = 0."""
        config = _two_node_config()
        snap = initial_snapshot(config)
        library = SnapshotLibrary(config=config, snapshots=[snap], meta={})
        evaluator = L0Evaluator(library, BoundaryWeights())
        forced = ForcedScorer({0: 0, 1: 0})
        result = evaluator.evaluate(forced)
        assert result.valid and result.feasible
        r = result.per_snapshot[0]
        assert math.isclose(r.device_j, 780.0, rel_tol=1e-12)
        assert r.cloud_j == 0.0 and r.network_j == 0.0
        assert math.isclose(r.closed_boundary_j, 780.0, rel_tol=1e-12)
        # Idle floor: (2 + 5) W * 100 s = 700 J -> normalized = 780/700.
        assert math.isclose(r.normalized_energy, 780.0 / 700.0, rel_tol=1e-12)
        # M/M/1-style estimate with deterministic service: rho = 2*0.05/1 = 0.1 per
        # replica -> W = 0.05/(1-0.1).
        assert math.isclose(
            r.template_latency_est_s[0], 0.05 / 0.9, rel_tol=1e-12
        )

    def test_l0_energy_matches_hand_arithmetic_split(self):
        """One replica per node: u = 0.05 each; node0 P = 2.4, node1 P = 5.05;
        device = 240 + 505 = 745 J."""
        config = _two_node_config()
        snap = initial_snapshot(config)
        library = SnapshotLibrary(config=config, snapshots=[snap], meta={})
        result = L0Evaluator(library, BoundaryWeights()).evaluate(
            ForcedScorer({0: 0, 1: 1})
        )
        r = result.per_snapshot[0]
        assert math.isclose(r.device_j, 745.0, rel_tol=1e-12)

    def test_brute_force_optimum_is_packing_onto_steep_node(self):
        """Enumerable optimum: {0,0} -> 780 J, {0,1}/{1,0} -> 745 J, {1,1} -> 200+510
        = 710 J (node1 busy u=0.1 -> 5+1*0.1=5.1 W; node0 idle 2 W). Optimum packs
        both replicas onto the flat-curve node 1."""
        config = _two_node_config()
        snap = initial_snapshot(config)
        best_j, best_assign = brute_force_optimum(config, snap, BoundaryWeights())
        assert math.isclose(best_j, 710.0, rel_tol=1e-12)
        assert best_assign == [(0, 1), (1, 1)]

    def test_energy_greedy_policy_finds_the_enumerated_optimum(self):
        config = _two_node_config()
        snap = initial_snapshot(config)
        library = SnapshotLibrary(config=config, snapshots=[snap], meta={})
        evaluator = L0Evaluator(library, BoundaryWeights())
        scorer = PlacementScriptScorer(
            ps.Policy((POLICIES / "energy_greedy.ps").read_text())
        )
        result = evaluator.evaluate(scorer)
        assert result.valid and result.feasible
        best_j, _ = brute_force_optimum(config, snap, BoundaryWeights())
        assert math.isclose(
            result.per_snapshot[0].closed_boundary_j, best_j, rel_tol=1e-12
        )

    def test_emulator_decision_sequence_is_enumerable(self):
        """least_allocated spreads: with assumed-pod accounting the second pod must go
        to the other node even though the first binding is still in flight — the exact
        decision sequence is [0, 1] (ties -> lowest node id)."""
        config = _two_node_config()
        scorer = PlacementScriptScorer(
            ps.Policy((POLICIES / "least_allocated.ps").read_text())
        )
        result = run_episode(config, seed=SEED, scorer=scorer)
        reserves = [d for d in result.decision_log if d["event"] == "reserve"]
        first_two = [d["node"] for d in reserves[:2]]
        assert first_two == [0, 1]

    def test_emulator_flat_scores_tie_break_lowest_id_with_capacity(self):
        """Flat scorer: every pod goes to the lowest feasible node id; capacity
        (2 pods x 400m on a 1000m node) forces the third onto node 1."""
        config = _two_node_config()
        svc = config.workload.services[0]
        from dataclasses import replace

        config = replace(
            config,
            workload=replace(
                config.workload, services=(replace(svc, initial_replicas=3),)
            ),
        )
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        reserves = [d for d in result.decision_log if d["event"] == "reserve"]
        nodes = [d["node"] for d in reserves[:3]]
        assert nodes == [0, 0, 1]
