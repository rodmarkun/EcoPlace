"""Fitness-aggregation tests (TODO 3.4): V2 shortfall formulas, aggregation axes,
totality/determinism of the order (W3), V1 feasible fitness, and the W2 penalty
dominance property — asserted, not assumed. Outcomes are built through the genuine
verified-token path (R2-1): certify_boundary → verify_covers → InstanceOutcome."""

from __future__ import annotations

import dataclasses
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sim.energy.boundary import (
    BoundaryBounds,
    BoundaryWeights,
    EnergyMeter,
    VerifiedInstanceEnergy,
    certify_boundary,
)
from sim.energy.fitness import (
    CandidateFitness,
    ConstraintSpec,
    InstanceOutcome,
    feasible_regret,
    ge_shortfall,
    le_shortfall,
    penalty_dominates,
    penalty_regret,
    primitive_shortfalls,
    rank_candidates,
)
from sim.energy.fitness import (
    evaluate_candidate as _evaluate_candidate,
)
from sim.spec import ClusterSpec, EnergyCoefficients, NodeClassSpec, NodeSpec

SPEC = ConstraintSpec(
    slo_targets={"app-a": 0.95, "app-b": 0.90},
    goodput_target=0.90,
    drop_cap=0.02,
    timeout_cap=0.02,
    disruption_caps={"moves": 10.0, "unavailability_s": 0.0, "bytes_mb": 500.0},
    zero_cap_scales={"unavailability_s": 60.0},
)

BUNDLES = ("dev-1", "dev-2")
KEYS = (("s1", 1),)  # the frozen scenario x seed grid used by the single-instance tests

HORIZON_S = 2.0  # power of two -> (x / H) * H round-trips exactly in floats


def evaluate_candidate(spec, outcomes, **kwargs):
    """Supply the explicit campaign certificate registry for aggregation tests.

    Production callers cannot derive this registry from candidate outcomes; it is
    frozen before evaluation. These tests use genuine tokens and synthesize their
    expected seals, while the R4-1 regression calls the production function directly.
    """
    bundle_ids = kwargs["development_bundle_ids"]
    instance_keys = kwargs["instance_keys"]
    seals = {
        (scenario, seed, bundle): "0" * 64
        for scenario, seed in instance_keys
        for bundle in bundle_ids
    }
    for item in outcomes:
        key = (item.scenario_id, item.seed, item.bundle_id)
        if key in seals:
            seals[key] = item.verified.certificate_seal
    kwargs.setdefault("expected_certificate_seals", seals)
    return _evaluate_candidate(spec, outcomes, **kwargs)


def make_token(
    scenario: str,
    seed: int,
    bundle: str,
    energy: float,
    floor: float,
    ceiling: float,
) -> VerifiedInstanceEnergy:
    """A genuine verified token (R2-1/R3-1): a one-fog-node instance whose idle
    floor is exactly ``floor`` and ceiling exactly ``ceiling`` under device-only
    weights, certified by certify_boundary; the episode is **actually metered** by
    the real EnergyMeter (reports are sealed — hand-assembled reports no longer
    exist) and passed through verify_covers. To exercise energy above the ceiling
    (physically impossible via the bounded power curve), the over-ceiling branch
    meters a suspendable class and injects one transition pulse within the
    certified transition bound — the excess energy is real metered energy, and
    verify_covers must reject it on the E <= M check."""
    over_ceiling = energy > ceiling
    cls = NodeClassSpec(
        name="test-class",
        power_idle_w=floor / HORIZON_S,
        power_max_w=ceiling / HORIZON_S,
        **(
            dict(
                suspend_energy_j=1.0, suspend_latency_s=1.0,
                wake_energy_j=1.0, wake_latency_s=1.0, suspended_power_w=0.0,
            )
            if over_ceiling
            else {}
        ),
    )
    cluster = ClusterSpec(
        classes=(cls,),
        nodes=(
            NodeSpec(id=0, node_class="test-class", tier=1, cpu_capacity=1000, mem_capacity=1024),
        ),
        latency_ms=((0.0,),),
    )
    coeff = EnergyCoefficients(cloud_j_per_vcpu_hour=0.0, network_j_per_byte=0.0)
    weights = BoundaryWeights(device=1.0, cloud=0.0, network=0.0)
    bounds = BoundaryBounds(
        max_transitions_per_node=1 if over_ceiling else 0, network_bytes_bound=0.0
    )
    cert = certify_boundary(
        cluster, coeff, HORIZON_S, weights, bounds,
        scenario_id=scenario, seed=seed, bundle_id=bundle,
    )
    meter = EnergyMeter(cluster, coeff)
    if over_ceiling:
        meter.add_transition_pulse(0, energy - floor)  # metered pulse beyond M
    else:
        utilization = (energy - floor) / (ceiling - floor)
        meter.set_busy_millicores(0, utilization * 1000.0, 0.0)
    return cert.verify_covers(meter.finish(HORIZON_S))


def outcome(
    scenario="s1",
    seed=1,
    bundle="dev-1",
    slo_a=1.0,
    slo_b=1.0,
    goodput=1.0,
    drop=0.0,
    timeout=0.0,
    moves=0.0,
    unavail=0.0,
    bytes_mb=0.0,
    energy=2000.0,
    floor=1000.0,
    ceiling=10_000.0,
) -> InstanceOutcome:
    return InstanceOutcome(
        verified=make_token(scenario, seed, bundle, energy, floor, ceiling),
        slo_achieved={"app-a": slo_a, "app-b": slo_b},
        goodput=goodput,
        drop_rate=drop,
        timeout_rate=timeout,
        disruption={"moves": moves, "unavailability_s": unavail, "bytes_mb": bytes_mb},
    )


def full_grid(**overrides_per_bundle):
    """One scenario x seed instance covering both bundles."""
    return [
        outcome(bundle=b, **overrides_per_bundle.get(b, {}))
        for b in BUNDLES
    ]


class TestShortfallFormulas:
    def test_ge_shortfall_zero_when_met(self):
        assert ge_shortfall(0.9, 0.95) == 0.0
        assert ge_shortfall(0.9, 0.9) == 0.0

    def test_ge_shortfall_normalized(self):
        assert ge_shortfall(0.9, 0.45) == pytest.approx(0.5)

    def test_ge_requires_positive_target(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ge_shortfall(0.0, 0.5)

    def test_le_shortfall_normalized(self):
        assert le_shortfall(0.02, 0.03) == pytest.approx(0.5)
        assert le_shortfall(0.02, 0.01) == 0.0

    def test_le_zero_cap_uses_registered_scale(self):
        assert le_shortfall(0.0, 30.0, zero_cap_scale=60.0) == pytest.approx(0.5)

    def test_le_zero_cap_without_scale_is_an_error(self):
        with pytest.raises(ValueError, match="scaling constant"):
            le_shortfall(0.0, 1.0)

    def test_spec_rejects_unscaled_zero_cap(self):
        with pytest.raises(ValueError, match="zero_cap_scales"):
            ConstraintSpec(
                slo_targets={"a": 0.9},
                goodput_target=0.9,
                drop_cap=0.0,
                timeout_cap=0.02,
                disruption_caps={"moves": 1.0},
            )


class TestAggregationAxes:
    def test_hand_computed_vector(self):
        """Two scenarios x one seed x two bundles with known primitive shortfalls."""
        outcomes = [
            # instance (s1, 1): app-a violated in dev-2 only (max across bundles picks it)
            outcome(scenario="s1", bundle="dev-1", slo_a=0.95),
            outcome(scenario="s1", bundle="dev-2", slo_a=0.475),  # shortfall 0.5
            # instance (s2, 1): drop violated in both bundles, worse in dev-1
            outcome(scenario="s2", bundle="dev-1", drop=0.04),  # shortfall 1.0
            outcome(scenario="s2", bundle="dev-2", drop=0.03),  # shortfall 0.5
        ]
        fit = evaluate_candidate(
            SPEC,
            outcomes,
            development_bundle_ids=BUNDLES,
            instance_keys=(("s1", 1), ("s2", 1)),
            proposal_seq=7,
        )
        assert not fit.feasible
        # SLO category: instance s1 -> max over bundles = 0.5; s2 -> 0; mean = 0.25
        assert fit.infeasibility_vector[0] == pytest.approx(0.25)
        # goodput untouched
        assert fit.infeasibility_vector[1] == 0.0
        # drop/timeout: s1 -> 0; s2 -> max(bundle max drop 1.0, timeout 0) = 1.0; mean 0.5
        assert fit.infeasibility_vector[2] == pytest.approx(0.5)
        assert fit.infeasibility_vector[3] == 0.0
        # summed primitive (before within-category reduction), W3 literal *total sum*
        # over the frozen grid: s1: 0.5; s2: 1.0 -> 1.5 (not the mean)
        assert fit.summed_primitive_shortfall == pytest.approx(1.5)
        assert fit.normalized_energy is None

    def test_max_within_category_across_apps(self):
        outcomes = full_grid(**{"dev-1": {"slo_a": 0.475, "slo_b": 0.45}})
        fit = evaluate_candidate(SPEC, outcomes, development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=0)
        # app-a shortfall 0.5, app-b 0.5 -> category max 0.5; but summed counts both
        assert fit.infeasibility_vector[0] == pytest.approx(0.5)
        assert fit.summed_primitive_shortfall == pytest.approx(1.0)

    def test_feasibility_quantifies_over_every_bundle(self):
        with pytest.raises(ValueError, match="missing bundles"):
            evaluate_candidate(
                SPEC,
                [outcome(bundle="dev-1")],
                development_bundle_ids=BUNDLES,
            instance_keys=KEYS,
                proposal_seq=0,
            )

    def test_duplicate_instance_bundle_rejected(self):
        with pytest.raises(ValueError, match="duplicate outcome"):
            evaluate_candidate(
                SPEC,
                [outcome(), outcome()],
                development_bundle_ids=BUNDLES,
            instance_keys=KEYS,
                proposal_seq=0,
            )

    def test_partial_grid_rejected(self):
        # Candidate evaluated on one instance while the frozen grid names two (R1-7).
        with pytest.raises(ValueError, match="not evaluated on frozen instances"):
            evaluate_candidate(
                SPEC,
                full_grid(),
                development_bundle_ids=BUNDLES,
                instance_keys=(("s1", 1), ("s2", 1)),
                proposal_seq=0,
            )

    def test_outcome_outside_grid_rejected(self):
        with pytest.raises(ValueError, match="outside the frozen grid"):
            evaluate_candidate(
                SPEC,
                full_grid() + [outcome(scenario="rogue", bundle="dev-1")],
                development_bundle_ids=BUNDLES,
                instance_keys=KEYS,
                proposal_seq=0,
            )

    def test_duplicate_instance_keys_rejected(self):
        with pytest.raises(ValueError, match="duplicate-free"):
            evaluate_candidate(
                SPEC,
                full_grid(),
                development_bundle_ids=BUNDLES,
                instance_keys=(("s1", 1), ("s1", 1)),
                proposal_seq=0,
            )

    def test_feasible_fitness_is_mean_normalized_energy(self):
        outcomes = [
            outcome(bundle="dev-1", energy=1500.0, floor=1000.0),
            outcome(bundle="dev-2", energy=2500.0, floor=1000.0),
        ]
        fit = evaluate_candidate(SPEC, outcomes, development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=0)
        assert fit.feasible
        assert fit.normalized_energy == pytest.approx(2.0)

    def test_infeasible_below_feasible_always(self):
        feasible = evaluate_candidate(
            SPEC, full_grid(), development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=5
        )
        infeasible = evaluate_candidate(
            SPEC,
            full_grid(**{"dev-1": {"goodput": 0.1}}),
            development_bundle_ids=BUNDLES,
            instance_keys=KEYS,
            proposal_seq=1,
        )
        assert feasible.sort_key() < infeasible.sort_key()


class TestOrderTotalityDeterminism:
    @staticmethod
    def _random_fitness(rng: random.Random, seq: int) -> CandidateFitness:
        overrides = {}
        for b in BUNDLES:
            overrides[b] = {
                "slo_a": rng.choice([1.0, 0.95, 0.6, 0.3]),
                "goodput": rng.choice([1.0, 0.85, 0.5]),
                "drop": rng.choice([0.0, 0.01, 0.05]),
                "moves": rng.choice([0.0, 5.0, 20.0]),
                "energy": rng.uniform(1000.0, 9000.0),
            }
        return evaluate_candidate(
            SPEC, full_grid(**overrides), development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=seq
        )

    def test_order_is_total_and_deterministic(self):
        rng = random.Random(20260716)
        fits = [self._random_fitness(rng, seq) for seq in range(50)]
        keys = [f.sort_key() for f in fits]
        # totality: every pair strictly comparable and antisymmetric
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert (keys[i] < keys[j]) != (keys[j] < keys[i])
        # determinism: ranking invariant under input permutation
        base = [f.proposal_seq for f in rank_candidates(fits)]
        for shuffle_seed in range(5):
            shuffled = list(fits)
            random.Random(shuffle_seed).shuffle(shuffled)
            assert [f.proposal_seq for f in rank_candidates(shuffled)] == base

    def test_proposal_seq_breaks_exact_ties(self):
        a = evaluate_candidate(SPEC, full_grid(), development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=2)
        b = evaluate_candidate(SPEC, full_grid(), development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=9)
        assert a.sort_key() < b.sort_key()

    def test_duplicate_proposal_seq_rejected_in_ranking(self):
        # W3: exact ties must fall to the canonical sequence number; duplicates would
        # silently retain caller (asynchronous completion) order (R1-7).
        a = evaluate_candidate(
            SPEC, full_grid(), development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=3
        )
        b = evaluate_candidate(
            SPEC, full_grid(), development_bundle_ids=BUNDLES,
            instance_keys=KEYS, proposal_seq=3
        )
        with pytest.raises(ValueError, match="duplicate proposal_seq"):
            rank_candidates([a, b])

    def test_summed_primitive_breaks_vector_ties(self):
        # Same 4-vector (category max 0.5 in SLO), different primitive sums.
        one_app = evaluate_candidate(
            SPEC,
            full_grid(**{"dev-1": {"slo_a": 0.475}}),
            development_bundle_ids=BUNDLES,
            instance_keys=KEYS,
            proposal_seq=9,
        )
        two_apps = evaluate_candidate(
            SPEC,
            full_grid(**{"dev-1": {"slo_a": 0.475, "slo_b": 0.45}}),
            development_bundle_ids=BUNDLES,
            instance_keys=KEYS,
            proposal_seq=1,
        )
        assert one_app.infeasibility_vector == two_apps.infeasibility_vector
        assert one_app.sort_key() < two_apps.sort_key()  # despite the larger seq

    @given(st.integers(0, 2**32), st.integers(0, 2**32))
    @settings(max_examples=25, deadline=None)
    def test_hypothesis_pairwise_totality(self, s1, s2):
        f1 = self._random_fitness(random.Random(s1), 1)
        f2 = self._random_fitness(random.Random(s2), 2)
        assert (f1.sort_key() < f2.sort_key()) or (f2.sort_key() < f1.sort_key())


class TestPenaltyDominance:
    def test_dominance_over_seeded_random_instances(self):
        """P(instance) strictly exceeds any feasible regret for every energy within
        the ceiling — including E = M exactly and E < reference."""
        rng = random.Random(42)
        for _ in range(500):
            floor = rng.uniform(1e-3, 1e6)
            ceiling = floor * rng.uniform(1.0, 1e4)
            ref = rng.uniform(floor * 0.5, ceiling)  # any reference energy
            for energy in (0.0, floor, ref, ceiling, rng.uniform(0.0, ceiling)):
                assert penalty_dominates(energy, ceiling, ref, floor)
                assert feasible_regret(energy, ref, floor) < penalty_regret(
                    ceiling, ref, floor
                )

    def test_boundary_case_energy_equals_ceiling_is_strict(self):
        assert feasible_regret(5000.0, 3000.0, 100.0) < penalty_regret(5000.0, 3000.0, 100.0)
        assert penalty_regret(5000.0, 3000.0, 100.0) == pytest.approx(21.0)

    def test_penalty_requires_finite_ceiling(self):
        with pytest.raises(ValueError, match="finite"):
            penalty_regret(float("inf"), 1.0, 1.0)

    def test_penalty_requires_positive_floor(self):
        with pytest.raises(ValueError, match="strictly positive"):
            penalty_regret(10.0, 1.0, 0.0)

    def test_precondition_violation_raises(self):
        with pytest.raises(ValueError, match="W2 precondition"):
            penalty_dominates(11.0, 10.0, 1.0, 1.0)

    def test_outcome_construction_asserts_w2_precondition(self):
        with pytest.raises(ValueError, match="ceiling"):
            outcome(energy=20_000.0, ceiling=10_000.0)

    def test_outcome_requires_positive_floor(self):
        with pytest.raises(ValueError, match="strictly positive"):
            outcome(floor=0.0)


class TestPrimitiveShortfalls:
    def test_missing_app_rejected(self):
        o = outcome()
        broken = dataclasses.replace(o, slo_achieved={"app-a": 1.0})
        with pytest.raises(ValueError, match="missing SLO"):
            primitive_shortfalls(SPEC, broken)

    def test_missing_disruption_component_rejected(self):
        o = outcome()
        broken = dataclasses.replace(o, disruption={"moves": 0.0})
        with pytest.raises(ValueError, match="missing disruption"):
            primitive_shortfalls(SPEC, broken)

    def test_component_enumeration_is_deterministic(self):
        assert SPEC.component_names() == (
            ("slo", "app-a"),
            ("slo", "app-b"),
            ("goodput", "goodput"),
            ("drop_timeout", "drop"),
            ("drop_timeout", "timeout"),
            ("disruption", "bytes_mb"),
            ("disruption", "moves"),
            ("disruption", "unavailability_s"),
        )


class TestTotalityGuards:
    def test_nan_metrics_rejected_at_outcome_construction(self):
        for field, bad in (
            ("goodput", float("nan")),
            ("drop_rate", float("inf")),
        ):
            with pytest.raises(ValueError, match="finite"):
                outcome(**{field.replace("_rate", ""): bad}
                        if field == "drop_rate" else {field: bad})

    def test_nan_slo_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            outcome(slo_a=float("nan"))

    def test_nan_disruption_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            outcome(moves=float("nan"))

    def test_shortfall_functions_reject_nonfinite_inputs(self):
        with pytest.raises(ValueError, match="finite"):
            ge_shortfall(0.9, float("nan"))
        with pytest.raises(ValueError, match="finite"):
            le_shortfall(0.02, float("inf"))


class TestVerifiedTokenEnforcement:
    """R2-1/R3-1 adversarial regressions in the fitness layer: outcomes accept
    only sealed VerifiedInstanceEnergy tokens, and forged tokens fail at both
    construction and evaluation."""

    def test_bare_floats_rejected(self):
        with pytest.raises(ValueError, match="VerifiedInstanceEnergy"):
            InstanceOutcome(
                verified=1234.5,  # type: ignore[arg-type]
                slo_achieved={"app-a": 1.0, "app-b": 1.0},
                goodput=1.0,
                drop_rate=0.0,
                timeout_rate=0.0,
                disruption={"moves": 0.0, "unavailability_s": 0.0, "bytes_mb": 0.0},
            )

    def test_forged_token_rejected_at_outcome_construction(self):
        token = make_token("s1", 1, "dev-1", 2000.0, 1000.0, 10_000.0)
        forged = dataclasses.replace(token, energy_j=1.0)  # claim less energy
        with pytest.raises(ValueError, match="seal mismatch"):
            InstanceOutcome(
                verified=forged,
                slo_achieved={"app-a": 1.0, "app-b": 1.0},
                goodput=1.0,
                drop_rate=0.0,
                timeout_rate=0.0,
                disruption={"moves": 0.0, "unavailability_s": 0.0, "bytes_mb": 0.0},
            )

    def test_sealless_token_rejected(self):
        token = make_token("s1", 1, "dev-1", 2000.0, 1000.0, 10_000.0)
        with pytest.raises(ValueError, match="seal"):
            dataclasses.replace(token, seal="")

    def test_forged_token_rejected_at_evaluation(self):
        """Even a token swapped in after outcome construction (frozen-dataclass
        setattr side channel) is caught by evaluate_candidate's per-outcome
        re-verification."""
        o = outcome()
        forged = dataclasses.replace(o.verified, energy_j=0.0)
        object.__setattr__(o, "verified", forged)
        with pytest.raises(ValueError, match="seal mismatch"):
            evaluate_candidate(
                SPEC,
                [o, outcome(bundle="dev-2")],
                development_bundle_ids=BUNDLES,
                instance_keys=KEYS,
                proposal_seq=0,
            )

    def test_valid_token_from_unregistered_certificate_rejected_by_fitness(self):
        """R4-1 exact counterexample: coincident I/M does not authorize a token."""
        cls = NodeClassSpec(name="plain", power_idle_w=10.0, power_max_w=50.0)
        cluster = ClusterSpec(
            classes=(cls,),
            nodes=(NodeSpec(0, "plain", 1, 1000, 1024),),
            latency_ms=((0.0,),),
        )
        coeff = EnergyCoefficients()

        def cert(bound):
            return certify_boundary(
                cluster,
                coeff,
                100.0,
                BoundaryWeights(),
                BoundaryBounds(max_transitions_per_node=bound, network_bytes_bound=0.0),
                scenario_id="s1",
                seed=1,
                bundle_id="dev-1",
            )

        registered, foreign = cert(0), cert(5)
        assert registered.idle_floor_j == foreign.idle_floor_j
        assert registered.ceiling_j == foreign.ceiling_j
        foreign_outcome = InstanceOutcome(
            verified=foreign.verify_covers(EnergyMeter(cluster, coeff).finish(100.0)),
            slo_achieved={"app-a": 1.0, "app-b": 1.0},
            goodput=1.0,
            drop_rate=0.0,
            timeout_rate=0.0,
            disruption={"moves": 0.0, "unavailability_s": 0.0, "bytes_mb": 0.0},
        )
        with pytest.raises(ValueError, match="unregistered boundary certificate"):
            _evaluate_candidate(
                SPEC,
                [foreign_outcome],
                development_bundle_ids=("dev-1",),
                instance_keys=KEYS,
                expected_certificate_seals={("s1", 1, "dev-1"): registered.seal},
                proposal_seq=0,
            )


class TestSpecAndMetricDomains:
    """R3-4 adversarial regressions: NaN caps must raise (never soften to zero
    shortfall), post-construction mutation of caller dicts must not reach the
    validated spec/outcome, and rates/satisfactions must live in [0, 1]."""

    def test_nan_cap_raises_instead_of_zero_shortfall(self):
        with pytest.raises(ValueError, match="finite"):
            le_shortfall(float("nan"), 1.0)
        with pytest.raises(ValueError, match="finite"):
            ge_shortfall(float("nan"), 1.0)
        with pytest.raises(ValueError, match="finite"):
            le_shortfall(0.0, 1.0, zero_cap_scale=float("nan"))

    def test_spec_is_immune_to_later_dict_mutation(self):
        slo = {"app": 0.9}
        caps = {"moves": 1.0}
        spec = ConstraintSpec(
            slo_targets=slo, goodput_target=0.9, drop_cap=0.1, timeout_cap=0.1,
            disruption_caps=caps,
        )
        slo["app"] = float("nan")
        caps["moves"] = float("nan")
        assert spec.slo_targets["app"] == 0.9
        assert spec.disruption_caps["moves"] == 1.0
        with pytest.raises(TypeError):
            spec.slo_targets["app"] = 0.5  # type: ignore[index]

    def test_outcome_is_immune_to_later_dict_mutation(self):
        slo = {"app-a": 1.0, "app-b": 1.0}
        o = outcome()
        object.__getattribute__(o, "slo_achieved")
        mutated = dict(slo)
        del mutated
        with pytest.raises(TypeError):
            o.slo_achieved["app-a"] = float("nan")  # type: ignore[index]

    def test_out_of_domain_rates_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            outcome(goodput=1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            outcome(drop=-0.1)
        with pytest.raises(ValueError, match=">= 0"):
            outcome(moves=-1.0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ConstraintSpec(
                slo_targets={"a": 1.5}, goodput_target=0.9, drop_cap=0.1,
                timeout_cap=0.1, disruption_caps={"moves": 1.0},
            )

    def test_mapping_keys_must_be_nonempty_strings_from_construction(self):
        """R4-7: mixed keys fail at construction, never later during sorting."""
        for bad_targets in ({"ok": 0.9, 1: 0.9}, {"": 0.9}):
            with pytest.raises(ValueError, match="non-empty strings"):
                ConstraintSpec(
                    slo_targets=bad_targets,  # type: ignore[arg-type]
                    goodput_target=0.9,
                    drop_cap=0.1,
                    timeout_cap=0.1,
                    disruption_caps={"moves": 1.0},
                )
        with pytest.raises(ValueError, match="non-empty strings"):
            dataclasses.replace(outcome(), disruption={"moves": 0.0, 7: 0.0})
