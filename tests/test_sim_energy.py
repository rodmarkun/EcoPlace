"""Closed modeled energy boundary tests (TODO 2.4, C16/S5, §4.1/§4.3): three categories
never silently summed, hand-computed meter accounting, idle-floor and ceiling math,
provenance records, versioned coefficients."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from sim_helpers import (
    CLOUD_CLASS,
    EDGE_CLASS,
    fanout_config,
    flat_scorer_fn,
    mixed_cluster,
    single_service_config,
    two_edge_cluster,
)

from sim import (
    BoundaryBounds,
    BoundaryWeights,
    CallableScorer,
    ceiling_j,
    idle_floor_j,
    run_episode,
)
from sim.energy import EnergyMeter
from sim.spec import (
    CallEdgeSpec,
    ClusterSpec,
    EnergyCoefficients,
    NodeClassSpec,
    NodeSpec,
)

SEED = 5150


def small_cluster() -> ClusterSpec:
    return mixed_cluster()


class TestNeverSilentlySummed:
    def test_report_has_no_total_attribute(self):
        result = run_episode(
            fanout_config(horizon_s=15.0),
            seed=SEED,
            scorer=CallableScorer("flat", flat_scorer_fn),
        )
        report = result.energy
        for name in ("total", "total_j", "sum", "energy_j"):
            assert not hasattr(report, name)

    def test_closed_boundary_requires_explicit_weights_and_brands_composite(self):
        result = run_episode(
            fanout_config(horizon_s=15.0),
            seed=SEED,
            scorer=CallableScorer("flat", flat_scorer_fn),
        )
        report = result.energy
        with pytest.raises(TypeError):
            report.closed_boundary_j()  # weights are not optional
        value, record = report.closed_boundary_j(BoundaryWeights())
        assert math.isclose(
            value, report.device_j + report.cloud_j + report.network_j, rel_tol=1e-12
        )
        assert record.boundary == "closed-composite"
        assert record.source_status == "modeled-composite"
        assert record.coefficients_sha256 == report.coefficients.sha256()

    def test_per_category_provenance(self):
        result = run_episode(
            fanout_config(horizon_s=15.0),
            seed=SEED,
            scorer=CallableScorer("flat", flat_scorer_fn),
        )
        for boundary, status in (
            ("device", "modeled"),
            ("cloud-allocated", "allocated-point-estimate"),
            ("network-allocated", "allocated-point-estimate"),
        ):
            record = result.energy.provenance(boundary)
            assert record.boundary == boundary
            assert record.source_status == status
            assert record.excluded  # every record states exclusions


class TestMeterHandArithmetic:
    def _meter(self, cluster: ClusterSpec) -> EnergyMeter:
        return EnergyMeter(cluster, EnergyCoefficients(version="test-1"))

    def test_idle_then_busy_device_energy(self):
        cluster = mixed_cluster()
        meter = self._meter(cluster)
        # Node 1 (edge: idle 3 W, max 8 W, cap 4000): idle for 10 s, then 2000 mc busy
        # (u=0.5 -> P = 3 + 5*0.5 = 5.5 W) for 20 s, then idle again until t=50.
        meter.set_busy_millicores(1, 2000.0, 10.0)
        meter.set_busy_millicores(1, 0.0, 30.0)
        report = meter.finish(50.0)
        expected = 3.0 * 10 + 5.5 * 20 + 3.0 * 20
        assert report.device_j_per_node[1] == pytest.approx(expected, rel=1e-12)

    def test_cloud_nodes_produce_no_device_energy(self):
        cluster = mixed_cluster()
        meter = self._meter(cluster)
        meter.set_busy_millicores(0, 4000.0, 5.0)  # node 0 is tier-0 cloud
        report = meter.finish(50.0)
        assert report.device_j_per_node[0] == 0.0

    def test_cloud_vcpu_hours_exact(self):
        cluster = mixed_cluster()
        meter = self._meter(cluster)
        meter.set_cloud_alloc_millicores(2000.0, 0.0)  # 2 vCPU from t=0
        meter.set_cloud_alloc_millicores(500.0, 1800.0)  # 0.5 vCPU for the second half
        report = meter.finish(3600.0)
        expected_hours = 2.0 * 0.5 + 0.5 * 0.5
        assert report.cloud_vcpu_hours == pytest.approx(expected_hours, rel=1e-12)
        assert report.cloud_j == pytest.approx(
            expected_hours * report.coefficients.cloud_j_per_vcpu_hour, rel=1e-12
        )

    def test_suspend_transition_and_suspended_draw(self):
        cluster = mixed_cluster()
        meter = self._meter(cluster)
        # Node 2 (edge): awake-idle 10 s, transition 2 s (idle power during window +
        # 5 J pulse), suspended at 0.5 W until t=100.
        meter.set_state(2, "transition", 10.0)
        meter.add_transition_pulse(2, 5.0)
        meter.set_state(2, "suspended", 12.0)
        report = meter.finish(100.0)
        expected = 3.0 * 10 + 3.0 * 2 + 0.5 * 88 + 5.0
        assert report.device_j_per_node[2] == pytest.approx(expected, rel=1e-12)
        assert report.transition_j_per_node[2] == 5.0

    def test_network_bytes_times_coefficient(self):
        cluster = mixed_cluster()
        meter = self._meter(cluster)
        meter.add_bytes(1e9)
        report = meter.finish(1.0)
        assert report.network_j == pytest.approx(
            1e9 * report.coefficients.network_j_per_byte, rel=1e-12
        )

    def test_time_going_backwards_raises(self):
        meter = self._meter(mixed_cluster())
        meter.set_busy_millicores(1, 100.0, 5.0)
        with pytest.raises(ValueError, match="backwards"):
            meter.set_busy_millicores(1, 0.0, 4.0)

    def test_meter_rejects_negative_or_nonfinite_accounting_inputs(self):
        """R4-2 exact counterexample: the meter must never seal negative energy or
        bytes, and every public numeric input is checked before accumulation."""
        meter = self._meter(mixed_cluster())
        for call in (
            lambda: meter.add_transition_pulse(1, -1.0),
            lambda: meter.add_transition_pulse(1, float("nan")),
            lambda: meter.add_bytes(-1.0),
            lambda: meter.add_bytes(float("inf")),
            lambda: meter.set_cloud_alloc_millicores(-1.0, 0.0),
            lambda: meter.set_busy_millicores(1, -1.0, 0.0),
            lambda: meter.set_busy_millicores(1, 5000.0, 0.0),
            lambda: meter.set_state(1, "awake", float("nan")),
        ):
            with pytest.raises(ValueError):
                call()
        with pytest.raises(ValueError, match="positive"):
            meter.finish(0.0)


class TestPowerCurves:
    def test_linear_default_curve(self):
        assert EDGE_CLASS.power_w(0.0) == 3.0
        assert EDGE_CLASS.power_w(1.0) == 8.0
        assert EDGE_CLASS.power_w(0.5) == 5.5
        assert EDGE_CLASS.power_w(2.0) == 8.0  # clamped

    def test_point_curve_interpolation(self):
        cls = NodeClassSpec(
            name="specpower-like",
            power_idle_w=10.0,
            power_max_w=100.0,
            power_curve_points=((0.0, 10.0), (0.5, 40.0), (1.0, 100.0)),
        )
        assert cls.power_w(0.25) == pytest.approx(25.0)
        assert cls.power_w(0.75) == pytest.approx(70.0)
        assert cls.max_curve_power_w() == 100.0


class TestFloorAndCeiling:
    WEIGHTS = BoundaryWeights()

    def test_idle_floor_hand_computed(self):
        cluster = mixed_cluster()  # cloud (excluded) + 2 edge nodes idling at 3 W
        coeff = EnergyCoefficients(version="t")
        floor = idle_floor_j(cluster, coeff, 100.0, self.WEIGHTS)
        assert floor == pytest.approx(2 * 3.0 * 100.0, rel=1e-12)

    def test_idle_floor_counts_suspended_default_at_suspended_draw(self):
        cluster = mixed_cluster(suspended_third=True)
        coeff = EnergyCoefficients(version="t")
        floor = idle_floor_j(cluster, coeff, 100.0, self.WEIGHTS)
        assert floor == pytest.approx((3.0 + 0.5) * 100.0, rel=1e-12)

    def test_idle_floor_positivity_enforced(self):
        cloud_only = ClusterSpec(
            classes=(CLOUD_CLASS,),
            nodes=(NodeSpec(0, "cloud", 0, 8000, 16384),),
            latency_ms=((0.0,),),
        )
        coeff = EnergyCoefficients(version="t", cloud_floor_vcpu=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            idle_floor_j(cloud_only, coeff, 100.0, self.WEIGHTS)
        # A declared cloud allocation floor makes the instance evaluable again (V1).
        coeff = EnergyCoefficients(version="t", cloud_floor_vcpu=1.0)
        assert idle_floor_j(cloud_only, coeff, 3600.0, self.WEIGHTS) == pytest.approx(
            coeff.cloud_j_per_vcpu_hour
        )

    def test_ceiling_hand_computed_and_dominates(self):
        cluster = mixed_cluster()
        coeff = EnergyCoefficients(version="t")
        horizon = 60.0
        bounds = BoundaryBounds(max_transitions_per_node=4, network_bytes_bound=1e8)
        ceiling = ceiling_j(cluster, coeff, horizon, self.WEIGHTS, bounds)
        # Hand arithmetic: 2 edge nodes at max 8 W x 60 s + 4 transitions x max(5,8) J
        # each; cloud: 8 vCPU x (60/3600) h x coeff; network: 1e8 B x coeff.
        expected = (
            2 * (8.0 * 60.0 + 4 * 8.0)
            + 8.0 * (60.0 / 3600.0) * coeff.cloud_j_per_vcpu_hour
            + 1e8 * coeff.network_j_per_byte
        )
        assert ceiling == pytest.approx(expected, rel=1e-12)

    def test_ceiling_bounds_a_real_stressed_episode(self):
        config = fanout_config(horizon_s=30.0, rate=30.0)
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        value, _ = result.energy.closed_boundary_j(self.WEIGHTS)
        bounds = BoundaryBounds(
            max_transitions_per_node=0,
            network_bytes_bound=result.energy.network_bytes,
        )
        ceiling = ceiling_j(
            config.cluster, config.energy, config.horizon_s, self.WEIGHTS, bounds
        )
        assert value <= ceiling
        floor = idle_floor_j(config.cluster, config.energy, config.horizon_s, self.WEIGHTS)
        assert value >= floor > 0.0

    def test_ceiling_dominates_with_a_priori_bounds(self):
        """Non-circular dominance (R1-13): the bounds come from the *configuration*,
        never from the observed trace. Deterministic arrivals, no retries, known
        payloads and image sizes -> the byte bound is computable a priori; no
        suspendable class -> zero transitions a priori."""
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
            expo,
        )

        cls = NodeClassSpec(name="pe", power_idle_w=2.0, power_max_w=6.0)
        cluster = ClusterSpec(
            classes=(cls,),
            nodes=(NodeSpec(0, "pe", 2, 4000, 8192), NodeSpec(1, "pe", 2, 4000, 8192)),
            latency_ms=((0.0, 5.0), (5.0, 0.0)),
        )
        front = ServiceSpec(
            id=0, name="f", app=0, cpu_request=500, mem_request=256,
            service_time=expo(0.02), cpu_demand=500, image_mb=10,
        )
        back = ServiceSpec(
            id=1, name="b", app=0, cpu_request=500, mem_request=256,
            service_time=expo(0.02), cpu_demand=500, image_mb=10,
        )
        rate, horizon = 5.0, 30.0
        template = TemplateSpec(
            id=0,
            name="t",
            root=CallSpec(
                service=0,
                children=(CallEdgeSpec(child=CallSpec(service=1)),),
            ),
            arrival=ArrivalSpec("deterministic", rate),
        )
        config = EpisodeConfig(
            cluster=cluster,
            workload=WorkloadSpec(services=(front, back), templates=(template,)),
            horizon_s=horizon,
        )
        # A-priori byte bound: <= rate*horizon requests, each moving <= req+resp KB
        # cross-node, plus both images pulled once per node.
        max_requests = math.ceil(rate * horizon) + 1
        edge = template.root.children[0]
        payload_bytes = max_requests * (edge.request_kb + edge.response_kb) * 1000.0
        image_bytes = 2 * (front.image_mb + back.image_mb) * 2**20
        bounds = BoundaryBounds(
            max_transitions_per_node=0,  # no suspendable class in the cluster
            network_bytes_bound=payload_bytes + image_bytes,
        )
        ceiling = ceiling_j(cluster, config.energy, horizon, self.WEIGHTS, bounds)
        for seed in (SEED, SEED + 1, SEED + 2):
            result = run_episode(
                config, seed=seed, scorer=CallableScorer("flat", flat_scorer_fn)
            )
            value, _ = result.energy.closed_boundary_j(self.WEIGHTS)
            assert result.energy.network_bytes <= bounds.network_bytes_bound
            assert value <= ceiling

    def test_suspended_draw_above_idle_rejected(self):
        """R1-4: suspended power above the awake floor breaks ceiling dominance and is
        rejected at the spec boundary."""
        with pytest.raises(ValueError, match="suspended_power_w"):
            NodeClassSpec(
                name="bad",
                power_idle_w=2.0,
                power_max_w=3.0,
                suspend_energy_j=1.0,
                suspend_latency_s=1.0,
                wake_energy_j=1.0,
                wake_latency_s=1.0,
                suspended_power_w=100.0,
            )

    def test_non_finite_and_negative_energy_inputs_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            NodeClassSpec(name="n", power_idle_w=float("nan"), power_max_w=1.0)
        with pytest.raises(ValueError, match=">= 0"):
            NodeClassSpec(name="n", power_idle_w=-1.0, power_max_w=1.0)
        with pytest.raises(ValueError):
            EnergyCoefficients(network_j_per_byte=float("inf"))
        with pytest.raises(ValueError):
            EnergyCoefficients(cloud_j_per_vcpu_hour=-1.0)
        with pytest.raises(ValueError):
            BoundaryWeights(device=float("nan"))
        with pytest.raises(ValueError):
            BoundaryWeights(cloud=-0.5)

    def test_cloud_floor_is_metered_floored_and_ceilinged_consistently(self):
        """R1-11: the baseline cloud allocation enters the metered category, the idle
        floor, and the ceiling identically — floor <= actual <= ceiling holds within
        the cloud category."""
        coeff = EnergyCoefficients(version="t", cloud_floor_vcpu=2.0)
        config = fanout_config(horizon_s=30.0, rate=2.0)
        config = replace(config, energy=coeff)
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        floor_cloud_j = 2.0 * (30.0 / 3600.0) * coeff.cloud_j_per_vcpu_hour
        assert result.energy.cloud_j >= floor_cloud_j - 1e-9
        bounds = BoundaryBounds(max_transitions_per_node=8, network_bytes_bound=1e9)
        ceiling = ceiling_j(config.cluster, coeff, 30.0, self.WEIGHTS, bounds)
        value, _ = result.energy.closed_boundary_j(self.WEIGHTS)
        assert value <= ceiling

    def test_migration_state_transfer_bytes_enter_network_energy(self):
        """R1-11: rebalancer state-transfer bytes are policy-caused traffic and must be
        metered in the network category."""
        from sim import CallableProposer, RebalancerParams

        base = single_service_config(
            two_edge_cluster(),
            horizon_s=25.0,
            rate=2.0,
            service_kwargs={"replicas": 2, "cpu_request": 400, "state_transfer_mb": 50.0},
            rebalancer=RebalancerParams(enabled=True, window_s=10.0, budget_per_window=1),
        )

        def move_one(state, graph, budget):
            del graph, budget
            for p in state["pods"]:
                if p["current_node"] == 0:
                    return [{"kind": 0, "pod": p["id"], "dest": 1}]
            return []

        def stay(state, graph, budget):
            return []

        moved = run_episode(
            base,
            seed=SEED,
            scorer=CallableScorer("flat", flat_scorer_fn),
            proposer=CallableProposer("m", move_one),
        )
        idle = run_episode(
            base,
            seed=SEED,
            scorer=CallableScorer("flat", flat_scorer_fn),
            proposer=CallableProposer("s", stay),
        )
        assert moved.rebalancer["moves"] >= 1
        extra = moved.energy.network_bytes - idle.energy.network_bytes
        assert extra >= 50.0 * 2**20  # at least the state transfer itself

    def test_power_state_dominance_property(self):
        """R1-4 property: for every legal node class and utilization, the power drawn
        in ANY state (awake at any utilization, suspended, transition window = idle) is
        bounded by max_curve_power_w() — the ceiling's per-node basis."""
        from hypothesis import given
        from hypothesis import strategies as st

        watts = st.floats(min_value=0.0, max_value=1e4, allow_nan=False)

        @given(
            idle=watts,
            span=st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
            suspended_frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            curve_watts=st.lists(watts, min_size=0, max_size=4),
            us=st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=1,
                max_size=8,
            ),
        )
        def check(idle, span, suspended_frac, curve_watts, us):
            pmax = idle + span
            curve = None
            if curve_watts:
                n = len(curve_watts) + 1
                utils = [i / (n - 1) for i in range(n)]
                curve = tuple(zip(utils, [idle, *curve_watts], strict=True))
            cls = NodeClassSpec(
                name="h",
                power_idle_w=idle,
                power_max_w=pmax,
                power_curve_points=curve,
                suspended_power_w=idle * suspended_frac,
            )
            bound = cls.max_curve_power_w()
            assert cls.suspended_power_w <= bound + 1e-9
            assert cls.power_idle_w <= bound + 1e-9  # transition-window draw
            for u in us:
                assert cls.power_w(u) <= bound + 1e-9

        check()

    def test_bounds_are_required_and_validated(self):
        with pytest.raises(TypeError):
            ceiling_j(mixed_cluster(), EnergyCoefficients(), 10.0, self.WEIGHTS)
        with pytest.raises(ValueError):
            BoundaryBounds(max_transitions_per_node=-1, network_bytes_bound=0.0)
        with pytest.raises(ValueError):
            BoundaryBounds(max_transitions_per_node=0, network_bytes_bound=float("nan"))


class TestCoefficientVersioning:
    def test_sha_changes_with_any_value(self):
        a = EnergyCoefficients(version="v1")
        b = EnergyCoefficients(version="v1", network_j_per_byte=2e-8)
        c = EnergyCoefficients(version="v2")
        assert a.sha256() != b.sha256()
        assert a.sha256() != c.sha256()
        assert a.sha256() == EnergyCoefficients(version="v1").sha256()


class TestEpisodeAccountingIntegration:
    def test_image_pull_bytes_toggle(self):
        from dataclasses import replace

        config = fanout_config(horizon_s=15.0, rate=2.0)
        on = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        config_off = replace(
            config, energy=replace(config.energy, include_image_pull_bytes=False)
        )
        off = run_episode(
            config_off, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        assert on.energy.network_bytes > off.energy.network_bytes

    def test_same_node_calls_produce_no_network_bytes(self):
        """All replicas forced onto one node -> zero cross-node call bytes (image pulls
        disabled to isolate)."""
        from dataclasses import replace

        config = fanout_config(horizon_s=15.0, rate=2.0)
        config = replace(
            config, energy=replace(config.energy, include_image_pull_bytes=False)
        )
        result = run_episode(
            config,
            seed=SEED,
            scorer=CallableScorer("pack0", lambda p, n, c: 1.0 if n["id"] == 0 else 0.0),
        )
        placed = {p.node_id for p in result.pods if p.node_id is not None}
        assert placed == {0}
        assert result.energy.network_bytes == 0.0

    def test_suspended_by_default_node_burns_suspended_power(self):
        config = fanout_config(horizon_s=20.0, rate=1.0)
        from dataclasses import replace

        config = replace(config, cluster=mixed_cluster(suspended_third=True))
        result = run_episode(
            config, seed=SEED, scorer=CallableScorer("flat", flat_scorer_fn)
        )
        assert result.energy.device_j_per_node[2] == pytest.approx(0.5 * 20.0)


class TestBoundaryCertificate:
    """R1-6/R2-1 (W2): the common-boundary identity — I, M, and every evaluated
    episode verified against one certified (cluster, coefficients, weights, horizon,
    bounds); sealed certificates; verified-energy tokens as the only regret currency."""

    def _certify(self, bounds=None, horizon=100.0, scenario_id="s1"):
        from sim.energy import certify_boundary

        return certify_boundary(
            mixed_cluster(),
            EnergyCoefficients(),
            horizon,
            BoundaryWeights(),
            bounds or BoundaryBounds(max_transitions_per_node=2, network_bytes_bound=1e6),
            scenario_id=scenario_id,
            seed=SEED,
            bundle_id="dev-1",
        )

    def _metered_report(self, horizon=100.0, bytes_sent=0.0, transitions=0):
        cluster = mixed_cluster()
        meter = EnergyMeter(cluster, EnergyCoefficients())
        if bytes_sent:
            meter.add_bytes(bytes_sent)
        for _ in range(transitions):
            meter.set_state(1, "transition", 0.0)
            meter.add_transition_pulse(1, 5.0)
            meter.set_state(1, "awake", 0.0)
        return meter.finish(horizon)

    def test_certificate_binds_floor_and_ceiling_to_one_boundary(self):
        cert = self._certify()
        assert cert.idle_floor_j > 0
        assert cert.ceiling_j > cert.idle_floor_j
        assert cert.cluster_sha256 != cert.coefficients_sha256
        assert len(cert.cluster_sha256) == 64

    def test_verify_covers_accepts_a_bounded_episode(self):
        cert = self._certify()
        report = self._metered_report(bytes_sent=1000.0, transitions=1)
        token = cert.verify_covers(report)
        token.verify_seal()
        assert 0 < token.energy_j <= cert.ceiling_j
        assert token.cluster_sha256 == cert.cluster_sha256
        assert token.idle_floor_j == cert.idle_floor_j

    def test_verify_rejects_more_transitions_than_certified(self):
        cert = self._certify(
            bounds=BoundaryBounds(max_transitions_per_node=1, network_bytes_bound=1e6)
        )
        with pytest.raises(ValueError, match="transition bound"):
            cert.verify_covers(self._metered_report(transitions=2))

    def test_verify_rejects_more_bytes_than_certified(self):
        cert = self._certify(
            bounds=BoundaryBounds(max_transitions_per_node=2, network_bytes_bound=100.0)
        )
        with pytest.raises(ValueError, match="bytes"):
            cert.verify_covers(self._metered_report(bytes_sent=1e5))

    def test_verify_rejects_foreign_coefficients(self):
        cert = self._certify()
        cluster = mixed_cluster()
        other = EnergyCoefficients(version="dev-1", cloud_j_per_vcpu_hour=999.0)
        report = EnergyMeter(cluster, other).finish(100.0)
        with pytest.raises(ValueError, match="different coefficients"):
            cert.verify_covers(report)

    def test_verify_rejects_foreign_cluster_report(self):
        """R2-1 adversarial regression (the reviewer's demonstrated bypass): a
        report metered on a *different cluster* previously passed verify_covers.
        It must now be rejected on the cluster digest."""
        base = mixed_cluster()
        foreign = replace(
            base,
            nodes=base.nodes[:2],
            latency_ms=tuple(row[:2] for row in base.latency_ms[:2]),
        )
        cert = self._certify()
        report = EnergyMeter(foreign, EnergyCoefficients()).finish(100.0)
        with pytest.raises(ValueError, match="different cluster"):
            cert.verify_covers(report)

    def test_verify_rejects_horizon_mismatch(self):
        """R2-1: a report over a different horizon is not the certified boundary."""
        cert = self._certify(horizon=100.0)
        with pytest.raises(ValueError, match="horizon"):
            cert.verify_covers(self._metered_report(horizon=50.0))

    def test_hand_built_certificate_rejected(self):
        """R2-1 adversarial regression: the public dataclass previously accepted
        hand-built certificates; construction now requires the certify_boundary
        seal, and field tampering breaks it."""
        from sim.energy import BoundaryCertificate

        with pytest.raises(ValueError, match="seal"):
            BoundaryCertificate(
                scenario_id="s1", seed=SEED, bundle_id="dev-1", horizon_s=100.0,
                weights=BoundaryWeights(),
                bounds=BoundaryBounds(max_transitions_per_node=2, network_bytes_bound=1e6),
                cluster_sha256="c" * 64, coefficients_sha256="d" * 64,
                idle_floor_j=1.0, ceiling_j=2.0,
            )
        cert = self._certify()
        inflated = replace(cert, ceiling_j=cert.ceiling_j * 100.0)  # keeps stale seal
        with pytest.raises(ValueError, match="seal mismatch"):
            inflated.verify_covers(self._metered_report())

    def test_penalty_dominates_any_verified_episode(self):
        """W2 end-to-end: every episode that passes verify_covers has regret
        strictly below the certificate's penalty, for any verified reference."""
        cert = self._certify()
        token = cert.verify_covers(self._metered_report(bytes_sent=5e5, transitions=2))
        for ref_report in (
            self._metered_report(),
            self._metered_report(bytes_sent=5e5, transitions=2),
            self._metered_report(bytes_sent=1e6),
        ):
            ref = cert.verify_covers(ref_report)
            assert cert.feasible_regret(token, ref) < cert.penalty_regret(ref)

    def test_penalty_refuses_foreign_boundary_reference(self):
        """R2-1: the reference must carry a token of *this* certificate — a token
        verified under another instance boundary is refused."""
        cert = self._certify()
        other_cert = self._certify(scenario_id="s2")
        foreign_ref = other_cert.verify_covers(self._metered_report())
        with pytest.raises(ValueError, match="reference"):
            cert.penalty_regret(foreign_ref)

    def test_forged_token_rejected(self):
        """R2-1: a token whose energy was edited after verification fails its seal
        at every consumption point."""
        cert = self._certify()
        token = cert.verify_covers(self._metered_report())
        forged = replace(token, energy_j=token.energy_j * 0.5)
        with pytest.raises(ValueError, match="seal mismatch"):
            cert.penalty_regret(forged)
        with pytest.raises(ValueError, match="seal"):
            replace(token, seal="")

    def test_certificate_requires_positive_finite_pieces(self):
        from sim.energy import certify_boundary

        with pytest.raises(ValueError, match="horizon"):
            certify_boundary(
                mixed_cluster(), EnergyCoefficients(), 0.0, BoundaryWeights(),
                BoundaryBounds(max_transitions_per_node=1, network_bytes_bound=0.0),
                scenario_id="s", seed=SEED, bundle_id="b",
            )

    def test_hand_assembled_report_never_mints_a_token(self):
        """R3-1 adversarial regression (the reviewer's demonstrated counterexample):
        an EnergyReport assembled from bare values — arbitrary energy, correct
        cluster digest, empty transition counts — previously produced a valid
        token. Reports now carry a meter seal: direct construction is rejected, and
        a field-edited metered report fails at verification."""
        from sim.energy import EnergyReport, cluster_sha256

        cluster = mixed_cluster()
        coeff = EnergyCoefficients()
        with pytest.raises(ValueError, match="seal"):
            EnergyReport(
                device_j=1.0, cloud_j=0.0, network_j=0.0,
                device_j_per_node=(1.0, *((0.0,) * (len(cluster.nodes) - 1))),
                transition_j_per_node=(0.0,) * len(cluster.nodes),
                cloud_vcpu_hours=0.0, network_bytes=0.0, coefficients=coeff,
                cluster_sha256=cluster_sha256(cluster), horizon_s=100.0,
                transition_counts=(0,) * len(cluster.nodes),
            )
        cert = self._certify()
        report = self._metered_report()
        tampered = replace(report)
        object.__setattr__(tampered, "device_j", report.device_j * 0.01)  # keeps old seal
        with pytest.raises(ValueError, match="does not equal|seal mismatch"):
            cert.verify_covers(tampered)

    def test_report_accounting_identities_are_validated(self):
        """R4-2: even a syntactically complete report cannot claim category totals
        inconsistent with the underlying meter bases."""
        report = self._metered_report(bytes_sent=1000.0, transitions=1)
        with pytest.raises(ValueError, match="network_j does not match"):
            replace(report, network_j=report.network_j + 1.0)
        with pytest.raises(ValueError, match="device_j does not equal"):
            replace(report, device_j=report.device_j + 1.0)

    def test_edited_report_rejected_by_every_reporting_consumer(self):
        """R4-2 exact bypass: report consumers and episode archives re-check the
        meter seal; verify_covers is not the only protected path."""
        report = self._metered_report()
        object.__setattr__(report, "device_j", report.device_j * 0.5)
        for consume in (
            lambda: report.closed_boundary_j(BoundaryWeights()),
            lambda: report.provenance("device"),
            report.all_provenance,
        ):
            with pytest.raises(ValueError, match="seal mismatch"):
                consume()

        result = run_episode(
            fanout_config(horizon_s=15.0),
            seed=SEED,
            scorer=CallableScorer("flat", flat_scorer_fn),
        )
        object.__setattr__(result.energy, "network_bytes", result.energy.network_bytes + 1.0)
        with pytest.raises(ValueError, match="seal mismatch"):
            result.to_canonical_dict()

    def test_report_rejects_incomplete_transition_counts(self):
        """R3-1: per-node vectors must cover the same node set — a report claiming
        fewer transition counts than device nodes cannot exist."""
        report = self._metered_report()
        with pytest.raises(ValueError, match="same node set"):
            replace(report, transition_counts=report.transition_counts[:-1])

    def test_token_rejected_across_certificates_with_identical_floor_and_ceiling(self):
        """R3-1 adversarial regression (the reviewer's demonstrated counterexample):
        two certificates over the same cluster/coefficients/horizon whose derived I
        and M coincide but whose certified *bounds* differ are distinct boundaries —
        a token minted under one was previously accepted by the other."""
        from sim.energy import certify_boundary
        from sim.spec import ClusterSpec, NodeClassSpec, NodeSpec

        # No suspendable class -> the transition bound does not enter M, so the two
        # certificates have identical I and M but different certified bounds.
        cls = NodeClassSpec(name="plain", power_idle_w=10.0, power_max_w=50.0)
        cluster = ClusterSpec(
            classes=(cls,),
            nodes=(NodeSpec(id=0, node_class="plain", tier=1,
                            cpu_capacity=1000, mem_capacity=1024),),
            latency_ms=((0.0,),),
        )
        coeff = EnergyCoefficients()

        def cert_with(transition_bound):
            return certify_boundary(
                cluster, coeff, 100.0, BoundaryWeights(),
                BoundaryBounds(
                    max_transitions_per_node=transition_bound, network_bytes_bound=1e6
                ),
                scenario_id="s1", seed=SEED, bundle_id="dev-1",
            )

        cert_a, cert_b = cert_with(0), cert_with(5)
        assert cert_a.idle_floor_j == cert_b.idle_floor_j
        assert cert_a.ceiling_j == cert_b.ceiling_j  # I and M coincide...
        report = EnergyMeter(cluster, coeff).finish(100.0)
        token_b = cert_b.verify_covers(report)
        with pytest.raises(ValueError, match="different certificate"):
            cert_a.penalty_regret(token_b)  # ...but the boundaries are distinct


class TestProvenanceEndToEnd:
    """R2-2 sign-off item: the composite branding and per-category source-record
    filtering, exercised end-to-end from a pinned bundle through a metered report."""

    def test_bundle_metered_records_brand_and_filter(self):
        from sim.energy.bundles import build_development_bundles
        from sim.energy.sources import load_pinned_sources

        bundle = build_development_bundles(load_pinned_sources())[0]
        coeffs = bundle.energy_coefficients()
        report = EnergyMeter(mixed_cluster(), coeffs).finish(10.0)
        value, composite = report.closed_boundary_j(BoundaryWeights())
        assert composite.bundle_set == f"development:{bundle.bundle_id}"
        assert "single-config" not in composite.bundle_set  # stale branding gone
        assert {r.entry_id for r in composite.source_records} == {
            r.entry_id for r in bundle.source_records
        }
        kinds = {
            b: {r.kind for r in rec.source_records}
            for b, rec in report.all_provenance().items()
        }
        assert kinds["device"] <= {"machine", "transition"}
        assert "machine" in kinds["device"]
        assert kinds["cloud-allocated"] == {"cloud-allocation"}
        assert kinds["network-allocated"] == {"network-allocation"}
        # every record carries the structured reference fields (R2-2)
        for rec in report.all_provenance().values():
            for r in rec.source_records:
                assert r.url or r.doi
                assert r.version_or_date_accessed
