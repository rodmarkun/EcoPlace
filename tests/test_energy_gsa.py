"""Stratified GSA harness tests (TODO 3.5): analytic toy models with known
sensitivities, determinism under fixed seeds, dependency preservation of the swap
analysis, Shapley effects on models with closed-form values, and the G0 cost item."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.energy.gsa import (
    AnalysisEnvelope,
    GsaComputeCost,
    MorrisResult,
    NuisanceParameter,
    SobolResult,
    Stratum,
    analysis_postflight_receipt,
    analysis_run_from_bytes,
    analysis_run_to_bytes,
    archive_analysis,
    between_stratum_swap,
    confirmatory_status,
    exact_shapley_effects,
    load_analysis_archive,
    morris_screen,
    nonconfirmatory_between_stratum_swap,
    nonconfirmatory_estimate_compute_cost,
    nonconfirmatory_morris_screen,
    nonconfirmatory_sobol_indices,
    nonconfirmatory_stratified_gsa,
    registered_shapley_effects,
    sobol_indices,
    stratified_gsa,
    verify_analysis_envelope,
)

# Toy model with known analytic sensitivities: Y = 2*x1 + 1*x2 + 0.5*x3 on U(0,1)^3.
# Var(Y) = (4 + 1 + 0.25)/12; S1_i = ST_i = a_i^2 / 5.25.
COEFFS = {"x1": 2.0, "x2": 1.0, "x3": 0.5}
ANALYTIC_S1 = {n: a * a / 5.25 for n, a in COEFFS.items()}
TEST_INDEPENDENCE = (
    "non-confirmatory analytic-test assumption: all synthetic unit inputs are independent"
)


def linear_model(stratum_id: str, params) -> float:
    return sum(COEFFS[k] * v for k, v in params.items())


def registered_python_model(function, **parameters):
    """Attach the explicit semantic-state contract required by production paths."""

    def descriptor():
        return {
            "kind": "python-callable",
            "parameters": parameters,
            "policy_path": None,
            "policy_sha256": None,
            "bundle_path": None,
            "bundle_sha256": None,
        }

    function.analysis_model_descriptor = descriptor
    return function


registered_python_model(linear_model, coefficients=COEFFS)


def registered_sum_model(_stratum_id: str, params) -> float:
    return sum(params.values())


registered_python_model(registered_sum_model, operation="sum-parameters")


def _registered_shapley_run(bundles):
    from sim.energy.gsa import stratum_for_bundle

    def model(params):
        return params["x"]

    def marginal(rng, n):
        return rng.normal(size=(n, 1))

    def conditional(rng, fixed, free):
        return {name: float(rng.normal()) for name in free}

    registered_python_model(model, selected_parameter="x")
    registered_python_model(marginal, distribution="standard-normal")
    registered_python_model(
        conditional,
        distribution="independent-standard-normal",
    )
    return registered_shapley_effects(
        model,
        ("x",),
        marginal,
        conditional,
        n_outer=2,
        n_inner=2,
        seed=20260729,
        strata=[stratum_for_bundle(bundle) for bundle in bundles],
        bundles=bundles,
    )


def _receipt_candidate_bytes(
    run,
    *,
    false_sampler=None,
    registration_tamper=None,
):
    """Rehash every public layer after making a run receipt-pending/tampered."""
    from sim.energy import gsa as gsa_module
    from sim.energy.registration import canonical_json_sha256

    artifact = json.loads(analysis_run_to_bytes(run.envelope, run.results))
    envelope = artifact["envelope"]
    envelope["execution_classification"] = "receipt_pending"
    if false_sampler is not None:
        sampler = envelope["estimators"][0]["parameters"][false_sampler]
        sampler["source_file_sha256"] = "f" * 64
        sampler["registration_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in sampler.items()
                if key != "registration_sha256"
            }
        )
    if registration_tamper is not None:
        location, changes = registration_tamper
        if location == "model_registration":
            registration = envelope["model_registration"]
        else:
            registration = envelope["estimators"][0]["parameters"][location]
        registration.update(changes)
        registration["registration_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in registration.items()
                if key != "registration_sha256"
            }
        )

    model_registration = gsa_module._model_registration_from_payload(
        envelope["model_registration"]
    )
    estimator_parameters = tuple(
        (estimator["estimator"], estimator["parameters"])
        for estimator in envelope["estimators"]
    )
    closure = gsa_module._source_closure_sha256(
        model_registration,
        estimator_parameters,
    )
    envelope["source_closure_sha256"] = closure
    envelope["producer_identity"]["source_closure_sha256"] = closure
    for estimator in envelope["estimators"]:
        estimator["source_closure_sha256"] = closure
    envelope["payload_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in envelope.items()
            if key != "payload_sha256"
        }
    )
    artifact["payload_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "payload_sha256"
        }
    )
    encoded = (
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    assert analysis_run_from_bytes(encoded).envelope_verified
    return encoded


def _simulate_clean_repository(
    monkeypatch,
    archive_data,
    *,
    stale_source_path=None,
):
    from sim.energy import gsa as gsa_module

    archive_relative = "sim/energy/data/receipt-candidate.json"
    original_artifact_bytes = gsa_module._repository_artifact_bytes

    def repository_artifact_bytes(relative_path):
        data = original_artifact_bytes(relative_path)
        if relative_path == stale_source_path:
            return data + b"\n# simulated committed sampler source change\n"
        return data

    def committed_blob(relative_path, _commit):
        if relative_path == archive_relative:
            return archive_data
        return repository_artifact_bytes(relative_path)

    monkeypatch.setattr(
        gsa_module,
        "_repository_state",
        lambda: gsa_module._RepositoryState("a" * 40, ()),
    )
    monkeypatch.setattr(
        gsa_module,
        "_archive_repository_path",
        lambda _path: archive_relative,
    )
    monkeypatch.setattr(
        gsa_module,
        "_repository_artifact_bytes",
        repository_artifact_bytes,
    )
    monkeypatch.setattr(gsa_module, "_committed_blob", committed_blob)


def unit_stratum(stratum_id="tuple-A", lo=0.0, hi=1.0) -> Stratum:
    return Stratum(
        stratum_id=stratum_id,
        nuisances=tuple(
            NuisanceParameter(
                n,
                lo,
                hi,
                conditional_basis="test: unit range",
                within_stratum_independence=TEST_INDEPENDENCE,
            )
            for n in ("x1", "x2", "x3")
        ),
    )


class TestMorris:
    def test_ranking_matches_analytic_coefficients(self):
        res = nonconfirmatory_morris_screen(
            linear_model, unit_stratum(), n_trajectories=20, seed=20260716
        )
        assert res.ranking() == ("x1", "x2", "x3")
        # linear model: mu* equals |a_i| * delta-normalized effect; exact proportions
        assert res.mu_star["x1"] == pytest.approx(2 * res.mu_star["x2"], rel=1e-6)
        assert res.mu_star["x2"] == pytest.approx(2 * res.mu_star["x3"], rel=1e-6)

    def test_deterministic_under_fixed_seed(self):
        a = nonconfirmatory_morris_screen(
            linear_model, unit_stratum(), n_trajectories=8, seed=7
        )
        b = nonconfirmatory_morris_screen(
            linear_model, unit_stratum(), n_trajectories=8, seed=7
        )
        assert a == b
        c = nonconfirmatory_morris_screen(
            linear_model, unit_stratum(), n_trajectories=8, seed=8
        )
        assert a != c

    def test_evaluation_count(self):
        res = nonconfirmatory_morris_screen(
            linear_model, unit_stratum(), n_trajectories=10, seed=1
        )
        assert res.n_evaluations == 10 * (3 + 1)


class TestSobol:
    def test_first_order_matches_analytic(self):
        res = nonconfirmatory_sobol_indices(
            linear_model, unit_stratum(), n_base=512, seed=20260716
        )
        for name, expected in ANALYTIC_S1.items():
            assert res.first_order[name] == pytest.approx(expected, abs=0.05)
            assert res.total_order[name] == pytest.approx(expected, abs=0.05)

    def test_requires_power_of_two(self):
        with pytest.raises(ValueError, match="power of two"):
            nonconfirmatory_sobol_indices(
                linear_model, unit_stratum(), n_base=500, seed=1
            )

    def test_deterministic_under_fixed_seed(self):
        a = nonconfirmatory_sobol_indices(
            linear_model, unit_stratum(), n_base=64, seed=3
        )
        b = nonconfirmatory_sobol_indices(
            linear_model, unit_stratum(), n_base=64, seed=3
        )
        assert a == b


class TestStratifiedRun:
    def test_runs_per_stratum_and_keeps_them_separate(self):
        strata = [unit_stratum("tuple-A"), unit_stratum("tuple-B", lo=10.0, hi=11.0)]
        out = nonconfirmatory_stratified_gsa(
            linear_model, strata, n_trajectories=8, n_base=64, seed=20260716
        )
        assert set(out) == {"tuple-A", "tuple-B"}
        # different conditional ranges -> different samples, same model structure
        assert out["tuple-A"][0] != out["tuple-B"][0]

    def test_mismatched_nuisance_names_rejected(self):
        s2 = Stratum(
            stratum_id="tuple-B",
            nuisances=(
                NuisanceParameter(
                    "other",
                    0,
                    1,
                    conditional_basis="test basis",
                    within_stratum_independence=TEST_INDEPENDENCE,
                ),
            ),
        )
        with pytest.raises(ValueError, match="same nuisance parameter names"):
            nonconfirmatory_stratified_gsa(
                linear_model,
                [unit_stratum(), s2],
                n_trajectories=4,
                n_base=8,
                seed=1,
            )

    def test_seed_is_required_everywhere(self):
        """SEEDS.md: omission must be a TypeError, not a silent default."""
        for fn in (
            morris_screen,
            sobol_indices,
            stratified_gsa,
            between_stratum_swap,
            exact_shapley_effects,
            registered_shapley_effects,
        ):
            param = inspect.signature(fn).parameters["seed"]
            assert param.default is inspect.Parameter.empty
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
        with pytest.raises(TypeError):
            nonconfirmatory_morris_screen(
                linear_model, unit_stratum(), n_trajectories=4
            )  # no seed


class TestSwapAnalysis:
    def test_swap_preserves_conditional_ranges(self):
        """No stratum is ever evaluated outside its own published spread."""
        seen: dict[str, list[float]] = {"lo": [], "hi": []}

        def spy(stratum_id, params):
            for v in params.values():
                seen[stratum_id].append(v)
            return sum(params.values())

        strata = [unit_stratum("lo", 0.0, 1.0), unit_stratum("hi", 100.0, 101.0)]
        res = nonconfirmatory_between_stratum_swap(
            spy, strata, n_points=32, seed=20260716
        )
        assert all(0.0 <= v <= 1.0 for v in seen["lo"])
        assert all(100.0 <= v <= 101.0 for v in seen["hi"])
        assert res.ranking == ("lo", "hi")
        assert res.n_evaluations == 64

    def test_swap_is_paired_across_strata(self):
        """Common random numbers: identical unit draws mapped into each stratum."""
        captured: dict[str, list] = {"a": [], "b": []}

        def spy(stratum_id, params):
            captured[stratum_id].append(tuple(params.values()))
            return 0.0 if stratum_id == "a" else 1.0

        strata = [unit_stratum("a", 0.0, 1.0), unit_stratum("b", 0.0, 2.0)]
        nonconfirmatory_between_stratum_swap(spy, strata, n_points=16, seed=5)
        for pa, pb in zip(captured["a"], captured["b"], strict=True):
            for va, vb in zip(pa, pb, strict=True):
                assert vb == pytest.approx(2 * va)  # same unit draw, rank-matched

    def test_sign_consistency_flag(self):
        res = nonconfirmatory_between_stratum_swap(
            lambda s, p: 1.0 if s == "a" else -1.0,
            [unit_stratum("a"), unit_stratum("b")],
            n_points=4,
            seed=1,
        )
        assert not res.sign_consistent
        assert res.max_pairwise_mean_gap == pytest.approx(2.0)


class TestShapleyEffects:
    def test_independent_linear_gaussian_matches_analytic(self):
        """Y = 2*X1 + X2, X ~ N(0, I): Shapley_i = a_i^2 / (a1^2 + a2^2) = 0.8 / 0.2."""
        names = ["x1", "x2"]

        def model(params):
            return 2.0 * params["x1"] + params["x2"]

        def marginal(rng, n):
            return rng.normal(size=(n, 2))

        def conditional(rng, fixed, free):
            return {f: float(rng.normal()) for f in free}  # independence

        sh = exact_shapley_effects(
            model, names, marginal, conditional, n_outer=64, n_inner=64, seed=20260716
        )
        assert sh["x1"] == pytest.approx(0.8, abs=0.06)
        assert sh["x2"] == pytest.approx(0.2, abs=0.06)
        assert sh["x1"] + sh["x2"] == pytest.approx(1.0, abs=0.06)

    def test_dependent_symmetric_gaussian_splits_evenly(self):
        """Y = X1 + X2 with corr rho: Shapley effects are 1/2 each by symmetry —
        the dependent-input case a standard Sobol decomposition cannot serve."""
        rho = 0.7
        names = ["x1", "x2"]

        def model(params):
            return params["x1"] + params["x2"]

        def marginal(rng, n):
            cov = np.array([[1.0, rho], [rho, 1.0]])
            return rng.multivariate_normal(np.zeros(2), cov, size=n)

        def conditional(rng, fixed, free):
            (other,) = fixed.values()
            return {f: float(rng.normal(rho * other, np.sqrt(1 - rho**2))) for f in free}

        sh = exact_shapley_effects(
            model, names, marginal, conditional, n_outer=64, n_inner=64, seed=20260716
        )
        assert sh["x1"] == pytest.approx(0.5, abs=0.07)
        assert sh["x2"] == pytest.approx(0.5, abs=0.07)

    def test_deterministic_under_fixed_seed(self):
        def model(params):
            return params["x1"]

        def marginal(rng, n):
            return rng.normal(size=(n, 1))

        def conditional(rng, fixed, free):
            return {f: float(rng.normal()) for f in free}

        kw = dict(n_outer=8, n_inner=8)
        a = exact_shapley_effects(model, ["x1"], marginal, conditional, seed=4, **kw)
        b = exact_shapley_effects(model, ["x1"], marginal, conditional, seed=4, **kw)
        assert a == b

    def test_nan_during_conditional_sampling_is_rejected(self):
        """R7-4: a finite variance baseline cannot authorize later NaN outputs."""
        calls = {"count": 0}

        def model(params):
            calls["count"] += 1
            if calls["count"] > 512:
                return float("nan")
            return params["x"] + params["y"]

        def marginal(rng, n):
            return rng.normal(size=(n, 2))

        def conditional(rng, fixed, free):
            return {name: float(rng.normal()) for name in free}

        with pytest.raises(ValueError, match="non-finite output"):
            exact_shapley_effects(
                model,
                ("x", "y"),
                marginal,
                conditional,
                n_outer=2,
                n_inner=2,
                seed=20260729,
            )
        assert calls["count"] > 512

    def test_vector_model_output_is_rejected(self):
        """R7-4: every Shapley model call has the scalar contract used by Stratum."""

        def marginal(rng, n):
            return rng.normal(size=(n, 2))

        with pytest.raises(ValueError, match="one scalar output"):
            exact_shapley_effects(
                lambda params: np.array([params["x"], params["y"]]),
                ("x", "y"),
                marginal,
                lambda rng, fixed, free: {
                    name: float(rng.normal()) for name in free
                },
                n_outer=2,
                n_inner=2,
                seed=20260729,
            )


class TestComputeCost:
    def test_counts_and_g0_line_item(self):
        strata = [unit_stratum("a"), unit_stratum("b")]
        cost = nonconfirmatory_estimate_compute_cost(
            strata,
            n_trajectories=20,
            n_base=512,
            swap_points=100,
            seconds_per_eval=0.01,
            shapley=(3, 32, 32),
        )
        assert cost.morris_evals == 2 * 20 * 4
        assert cost.sobol_evals == 2 * 512 * 5
        assert cost.swap_evals == 200
        # R1-10: the exact cached-coalition count, not the naive d!*d*n_o*n_i figure
        assert cost.shapley_evals == max(32 * 32, 512) + (2**3 - 2) * 32 * 32
        assert cost.total_evals == (
            cost.morris_evals + cost.sobol_evals + cost.swap_evals + cost.shapley_evals
        )
        line = cost.as_g0_line_item()
        assert line.startswith("| stratified-GSA")
        assert "30% reserve" in line
        assert f"{cost.total_evals} model evals" in line

    def test_requires_measured_eval_cost(self):
        with pytest.raises(ValueError, match="measured"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], n_trajectories=1, n_base=8, swap_points=1,
                seconds_per_eval=0.0,
            )

    def test_estimator_input_validation_shared_with_cost_table(self):
        """R2-9 adversarial regression: the cost estimator previously priced
        designs the estimators would refuse (negative counts, non-power-of-two
        Sobol bases, non-finite eval times, non-integral Shapley loops)."""
        base = dict(n_trajectories=2, n_base=8, swap_points=4, seconds_per_eval=1.0)
        with pytest.raises(ValueError, match="n_trajectories"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], **{**base, "n_trajectories": -5}
            )
        with pytest.raises(ValueError, match="power of two"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], **{**base, "n_base": 500}
            )
        with pytest.raises(ValueError, match="swap_points"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], **{**base, "swap_points": 0}
            )
        with pytest.raises(ValueError, match="finite"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], **{**base, "seconds_per_eval": float("nan")}
            )
        with pytest.raises(ValueError, match="finite"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], **{**base, "seconds_per_eval": float("inf")}
            )
        with pytest.raises(ValueError, match="int"):
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], **base, shapley=(3, 4.5, 4)
            )

    def test_estimators_reject_what_the_cost_table_rejects(self):
        """R2-9: the same validators guard the estimators themselves."""
        from sim.energy.gsa import shapley_eval_count

        with pytest.raises(ValueError, match="n_trajectories"):
            nonconfirmatory_morris_screen(
                linear_model, unit_stratum(), n_trajectories=0, seed=1
            )
        with pytest.raises(ValueError, match="power of two"):
            nonconfirmatory_sobol_indices(
                linear_model, unit_stratum(), n_base=12, seed=1
            )
        with pytest.raises(ValueError, match="n_points"):
            nonconfirmatory_between_stratum_swap(
                linear_model, [unit_stratum()], n_points=0, seed=1
            )
        with pytest.raises(ValueError, match="int"):
            shapley_eval_count(3, 4.0, 4)
        with pytest.raises(ValueError, match="int"):
            exact_shapley_effects(
                lambda p: 0.0, ["x"],
                lambda rng, n: rng.random((n, 1)),
                lambda rng, fixed, free: {f: 0.0 for f in free},
                n_outer=4.5, n_inner=4, seed=1,
            )

    def test_cost_dataclass_is_exact(self):
        assert isinstance(
            nonconfirmatory_estimate_compute_cost(
                [unit_stratum()], n_trajectories=2, n_base=8, swap_points=4,
                seconds_per_eval=1.0,
            ),
            GsaComputeCost,
        )


class TestShapleyEvalCount:
    def test_formula_matches_actual_model_calls(self):
        """R1-10: the G0 line item must report the estimator's true cost."""
        from sim.energy.gsa import shapley_eval_count

        calls = {"n": 0}

        def counting_model(params):
            calls["n"] += 1
            return 2.0 * params["a"] + params["b"] + 0.5 * params["c"]

        names = ("a", "b", "c")

        def marginal(rng, n):
            return rng.random((n, 3))

        def conditional(rng, fixed, free):
            return {f: float(rng.random()) for f in free}

        n_outer, n_inner = 4, 4
        exact_shapley_effects(
            counting_model, names, marginal, conditional,
            n_outer=n_outer, n_inner=n_inner, seed=20260718,
        )
        assert calls["n"] == shapley_eval_count(3, n_outer, n_inner)

    def test_count_rejects_bad_inputs(self):
        from sim.energy.gsa import shapley_eval_count

        with pytest.raises(ValueError, match="positive int"):
            shapley_eval_count(0, 4, 4)
        with pytest.raises(ValueError, match=">= 2"):
            shapley_eval_count(3, 1, 4)


class TestBundleBinding:
    """R1-9: strata bound to sealed bundle manifests; registered distributions."""

    @pytest.fixture(scope="class")
    def bundles(self):
        from sim.energy.bundles import build_development_bundles
        from sim.energy.sources import load_pinned_sources

        return build_development_bundles(load_pinned_sources())

    def test_stratum_for_bundle_binds_digest_and_sweeps(self, bundles):
        from sim.energy.gsa import stratum_for_bundle, validate_strata_bound

        strata = [stratum_for_bundle(b) for b in bundles]
        validate_strata_bound(strata, bundles)  # must not raise
        s0, b0 = strata[0], bundles[0]
        assert s0.stratum_id == b0.bundle_id
        assert s0.bundle_manifest_sha256 == b0.manifest_sha256()
        by_name = {p.name: p for p in s0.nuisances}
        assert (by_name["cloud_j_per_vcpu_hour"].low,
                by_name["cloud_j_per_vcpu_hour"].high) == b0.cloud_swept_j_per_vcpu_hour
        assert (by_name["network_j_per_byte"].low,
                by_name["network_j_per_byte"].high) == b0.network_swept_j_per_byte
        assert b0.cloud_entry_id in by_name["cloud_j_per_vcpu_hour"].conditional_basis

    def test_unbound_stratum_rejected_by_design_validator(self, bundles):
        from sim.energy.gsa import validate_strata_bound

        free = Stratum(
            stratum_id=bundles[0].bundle_id,  # right id, but no digest and no seal
            nuisances=unit_stratum().nuisances,
        )
        with pytest.raises(ValueError, match="unregistered|non-confirmatory"):
            validate_strata_bound([free], [bundles[0]])

    def test_production_entrypoints_require_registered_bundles(self, bundles):
        """R5-2 exact bypass: omitting the P2 bundle list is never a production mode."""
        from sim.energy.gsa import stratum_for_bundle

        free = unit_stratum("unregistered-free-form")
        with pytest.raises(ValueError, match="unregistered|non-confirmatory"):
            free.verify_binding()
        with pytest.raises(TypeError, match="bundles"):
            stratified_gsa(
                linear_model,
                [free],
                n_trajectories=2,
                n_base=8,
                seed=1,
            )
        with pytest.raises(TypeError, match="bundles"):
            between_stratum_swap(linear_model, [free], n_points=2, seed=1)

        bound = [stratum_for_bundle(bundle) for bundle in bundles]
        run = stratified_gsa(
            registered_sum_model,
            bound,
            n_trajectories=2,
            n_base=8,
            seed=1,
            bundles=bundles,
        )
        assert run.envelope_verified
        assert run.execution_classification == "development_only"
        assert verify_analysis_envelope(run.envelope, run.results)
        assert set(run.by_stratum()) == {bundle.bundle_id for bundle in bundles}

    def test_v6_independence_attestation_is_required_and_sealed(self, bundles):
        """Conditional-range prose cannot stand in for the V6 independence premise."""
        with pytest.raises(TypeError, match="within_stratum_independence"):
            NuisanceParameter("x", 0.0, 1.0, conditional_basis="range only")
        with pytest.raises(ValueError, match="independence"):
            NuisanceParameter(
                "x",
                0.0,
                1.0,
                conditional_basis="range basis",
                within_stratum_independence="",
            )
        with pytest.raises(ValueError, match="within_stratum_independence.*str"):
            NuisanceParameter(
                "x",
                0.0,
                1.0,
                conditional_basis="range basis",
                within_stratum_independence=True,
            )
        with pytest.raises(ValueError, match="conditional_basis.*str"):
            NuisanceParameter(
                "x",
                0.0,
                1.0,
                conditional_basis=True,
                within_stratum_independence=TEST_INDEPENDENCE,
            )
        with pytest.raises(ValueError, match="non-whitespace"):
            NuisanceParameter(
                "x",
                0.0,
                1.0,
                conditional_basis="published range",
                within_stratum_independence="   ",
            )

        from sim.energy.gsa import strata_design_sha256, stratum_for_bundle

        honest = stratum_for_bundle(bundles[0])
        assert all(
            nuisance.within_stratum_independence
            for nuisance in honest.nuisances
        )
        assert all(
            "within_stratum_independence" in nuisance.payload()
            for nuisance in honest.nuisances
        )
        assert len(strata_design_sha256([honest])) == 64

    def test_nan_model_outputs_fail_before_analysis_or_results(self, bundles):
        """R5-3 exact probes: Morris, Sobol, production GSA, and swap all reject NaN."""
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        stratum = strata[0]

        def nan_model(_stratum_id, _params):
            return float("nan")

        registered_python_model(nan_model, output="nan-rejection-probe")
        with pytest.raises(ValueError, match="non-finite output"):
            morris_screen(nan_model, stratum, n_trajectories=2, seed=1)
        with pytest.raises(ValueError, match="non-finite output"):
            sobol_indices(nan_model, stratum, n_base=8, seed=1)
        with pytest.raises(ValueError, match="non-finite output"):
            stratified_gsa(
                nan_model,
                strata,
                n_trajectories=2,
                n_base=8,
                seed=1,
                bundles=bundles,
            )
        with pytest.raises(ValueError, match="non-finite output"):
            between_stratum_swap(
                nan_model,
                strata,
                n_points=2,
                seed=1,
                bundles=bundles,
            )

    def test_subset_of_registered_bundles_rejected(self, bundles):
        """R2-3 adversarial regression (the reviewer's demonstrated bypass): a
        one-of-three strata subset previously validated against the full bundle
        list; the design now requires an exact bijection."""
        from sim.energy.gsa import (
            estimate_compute_cost,
            stratum_for_bundle,
            validate_strata_bound,
        )

        strata = [stratum_for_bundle(b) for b in bundles]
        assert len(strata) >= 2
        with pytest.raises(ValueError, match="exact bijection"):
            validate_strata_bound(strata[:1], bundles)
        with pytest.raises(ValueError, match="exact bijection"):
            validate_strata_bound(strata, bundles[:1])

        calls = {"count": 0}

        def model(_stratum_id, params):
            calls["count"] += 1
            return sum(params.values())

        # R6-1 exact reviewer probe: equality between two caller-selected
        # One-of-three subsets cannot mint a complete registered envelope.
        with pytest.raises(ValueError, match="full canonical development design"):
            stratified_gsa(
                model,
                strata[:1],
                n_trajectories=2,
                n_base=8,
                seed=20260729,
                bundles=bundles[:1],
            )
        assert calls["count"] == 0
        # Standalone bound analyses remain useful, but cannot be confused with a
        # full-design registered run.
        assert not hasattr(
            morris_screen(model, strata[0], n_trajectories=2, seed=20260729),
            "confirmatory",
        )
        assert not hasattr(
            sobol_indices(model, strata[0], n_base=8, seed=20260729),
            "confirmatory",
        )
        with pytest.raises(ValueError, match="full canonical development design"):
            estimate_compute_cost(
                strata[:1],
                n_trajectories=2,
                n_base=8,
                swap_points=2,
                seconds_per_eval=0.01,
            )

    def test_digest_mismatch_rejected(self, bundles):
        import dataclasses

        from sim.energy.gsa import stratum_for_bundle, validate_strata_bound

        s = stratum_for_bundle(bundles[0])
        forged = dataclasses.replace(s, bundle_manifest_sha256="f" * 64)
        with pytest.raises(ValueError, match="seal mismatch|does not match"):
            validate_strata_bound([forged], [bundles[0]])

    def test_invented_bounds_with_retained_digest_rejected(self, bundles):
        """R2-3 adversarial regression (the reviewer's demonstrated bypass): a
        stratum with invented bounds/basis that *retains the correct manifest
        digest* — and even carries a self-computed valid seal (the scheme is
        public) — must fail the design validator's re-derivation check."""
        from sim.energy.gsa import _stratum_seal, stratum_for_bundle, validate_strata_bound

        honest = stratum_for_bundle(bundles[0])
        invented = tuple(
            NuisanceParameter(
                name=p.name,
                low=p.low,
                high=p.high * 100.0,  # invented spread
                conditional_basis="invented: not any published spread",
                within_stratum_independence=p.within_stratum_independence,
                distribution=p.distribution,
            )
            for p in honest.nuisances
        )
        payload = {
            "stratum_id": honest.stratum_id,
            "nuisances": [p.payload() for p in invented],
            "bundle_manifest_sha256": honest.bundle_manifest_sha256,  # retained digest
        }
        forged = Stratum(
            stratum_id=honest.stratum_id,
            nuisances=invented,
            bundle_manifest_sha256=honest.bundle_manifest_sha256,
            seal=_stratum_seal(payload),  # adversary computes a valid seal
        )
        forged.verify_binding()  # the seal alone does not catch invention...
        full_strata = [
            forged,
            *(stratum_for_bundle(bundle) for bundle in bundles[1:]),
        ]
        with pytest.raises(ValueError, match="not re-derivable"):
            validate_strata_bound(full_strata, bundles)  # ...re-derivation does

    def test_tampered_bound_stratum_never_sampled(self, bundles):
        """R2-3: sampling entry points verify the stratum seal — a bound stratum
        whose bounds were replaced after construction fails before any draw."""
        import dataclasses

        from sim.energy.gsa import stratum_for_bundle

        s = stratum_for_bundle(bundles[0])
        tampered = dataclasses.replace(s, nuisances=unit_stratum().nuisances)
        with pytest.raises(ValueError, match="seal mismatch"):
            morris_screen(linear_model, tampered, n_trajectories=2, seed=1)
        with pytest.raises(ValueError, match="seal mismatch"):
            sobol_indices(linear_model, tampered, n_base=8, seed=1)
        with pytest.raises(ValueError, match="seal mismatch"):
            between_stratum_swap(
                linear_model, [tampered], n_points=2, seed=1, bundles=[bundles[0]]
            )

    def test_entry_points_validate_binding_against_bundle_list(self, bundles):
        """The pre-registered design path: passing bundles= enforces the exact
        bijection before sampling (R2-3)."""
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(b) for b in bundles]

        def model(stratum_id, params):
            return sum(params.values())

        registered_python_model(model, operation="sum-parameters")
        with pytest.raises(ValueError, match="exact bijection"):
            between_stratum_swap(model, strata[:1], n_points=2, seed=1, bundles=bundles)
        run = between_stratum_swap(model, strata, n_points=2, seed=1, bundles=bundles)
        assert run.envelope_verified
        assert run.execution_classification == "development_only"
        assert analysis_run_from_bytes(
            analysis_run_to_bytes(run.envelope, run.results)
        ).envelope_verified
        res = run.results["between-stratum-swap"]
        assert set(res.per_stratum_mean) == {b.bundle_id for b in bundles}

    def test_registered_shapley_path_returns_verifying_envelope(self, bundles):
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(bundle) for bundle in bundles]

        def model(params):
            return params["x"]

        def marginal(rng, n):
            return rng.normal(size=(n, 1))

        def conditional(rng, fixed, free):
            return {name: float(rng.normal()) for name in free}

        registered_python_model(model, selected_parameter="x")
        registered_python_model(marginal, distribution="standard-normal")
        registered_python_model(
            conditional,
            distribution="independent-standard-normal",
        )
        run = registered_shapley_effects(
            model,
            ("x",),
            marginal,
            conditional,
            n_outer=2,
            n_inner=2,
            seed=20260729,
            strata=strata,
            bundles=bundles,
        )
        assert run.envelope_verified
        assert run.execution_classification == "development_only"
        result = run.results["registered-shapley-effects"]
        assert result.effects["x"] == pytest.approx(1.0)
        assert analysis_run_from_bytes(
            analysis_run_to_bytes(run.envelope, run.results)
        ).envelope_verified

    def test_full_registered_run_with_string_attestations(self, bundles):
        """R6-2 acceptance probe: the complete canonical design still runs when
        every V6 premise is actual, meaningful string prose."""
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        assert all(
            isinstance(nuisance.within_stratum_independence, str)
            and len(nuisance.within_stratum_independence) >= 8
            for stratum in strata
            for nuisance in stratum.nuisances
        )
        run = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        assert run.envelope_verified
        assert run.execution_classification == "development_only"
        assert verify_analysis_envelope(run.envelope, run.results)
        assert set(run.by_stratum()) == {
            bundle.bundle_id for bundle in bundles
        }

    def test_caller_invented_extra_nuisance_is_not_registered(self, bundles):
        """R7-2: replaying one's own extra definition cannot extend the P2 design."""
        from sim.energy.gsa import stratum_for_bundle

        invented = NuisanceParameter(
            "invented_extra",
            1.0,
            1.2,
            conditional_basis="caller-invented nuisance bounds",
            within_stratum_independence=(
                "caller assertion: independent of the sealed allocation nuisances"
            ),
        )
        strata = [
            stratum_for_bundle(bundle, extra_nuisances=(invented,))
            for bundle in bundles
        ]
        calls = {"count": 0}

        def model(_stratum_id, params):
            calls["count"] += 1
            return sum(params.values())

        with pytest.raises(ValueError, match="sealed registered design|design_sha256"):
            stratified_gsa(
                model,
                strata,
                n_trajectories=2,
                n_base=8,
                seed=20260729,
                bundles=bundles,
            )
        assert calls["count"] == 0

    def test_registered_design_hash_is_the_freeze_point(self, bundles):
        from sim.energy.gsa import strata_design_sha256, stratum_for_bundle
        from sim.energy.registration import load_energy_analysis_registration

        registration = load_energy_analysis_registration()
        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        assert strata_design_sha256(strata) == registration.design_sha256
        assert registration.bundle_manifests("development") == tuple(
            (bundle.bundle_id, bundle.manifest_sha256()) for bundle in bundles
        )

    def test_registered_design_payload_tampering_is_rejected(self, tmp_path):
        """The design hash is load-bearing, not informational metadata."""
        from sim.energy import registration as registration_module

        raw = json.loads(registration_module._REGISTRATION_PATH.read_text())
        raw["gsa_design"]["payload"]["strata"][0]["nuisances"][0]["high"] += 1.0
        tampered = tmp_path / "energy_analysis_registration.json"
        tampered.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="design registration seal mismatch"):
            registration_module._load_energy_analysis_registration(tampered)

    def test_authority_is_only_the_complete_envelope(self, bundles):
        """R8-1/R8-2: no issuer is reachable and no result is authoritative alone."""
        import copy

        from sim.energy import gsa as gsa_module
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        run = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        assert run.envelope_verified
        assert run.execution_classification == "development_only"
        assert not hasattr(run, "confirmatory")
        assert verify_analysis_envelope(run.envelope, run.results)
        assert not hasattr(gsa_module, "_issue_confirmatory_result")
        assert not hasattr(gsa_module, "_has_result_authority")

        morris = run.results[f"{bundles[0].bundle_id}/morris"]
        assert not hasattr(morris, "_issuer")
        assert not hasattr(morris, "confirmatory")
        assert not hasattr(morris, "verify_seal")
        assert "design_sha256" not in morris.payload()
        assert "payload_sha256" not in morris.payload()

        with pytest.raises(TypeError):
            morris.mu_star[next(iter(morris.mu_star))] = 0.0
        with pytest.raises(TypeError):
            run.results["invented"] = morris

        # A byte-identical copy has no object authority and, when separated from
        # the complete set, cannot produce an envelope-verification verdict.
        copied = copy.copy(morris)
        assert copied == morris
        with pytest.raises(ValueError, match="incomplete"):
            verify_analysis_envelope(
                run.envelope,
                {copied.result_id(): copied},
            )

        development = nonconfirmatory_morris_screen(
            linear_model,
            unit_stratum(),
            n_trajectories=4,
            seed=20260729,
        )
        assert not hasattr(development, "confirmatory")
        with pytest.raises(TypeError):
            dataclasses.replace(development, confirmatory=True)

    def test_replay_tampering_and_subclasses_fail_envelope_verification(
        self, bundles
    ):
        """R8-1: changed constructor replay and caller subclasses cannot verify."""
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        run = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        result_id = f"{bundles[0].bundle_id}/morris"
        morris = run.results[result_id]
        changed = dict(morris.mu_star)
        first = next(iter(changed))
        changed[first] += 1.0
        replayed = MorrisResult(
            stratum_id=morris.stratum_id,
            mu_star=changed,
            sigma=morris.sigma,
            n_evaluations=morris.n_evaluations,
            seed=morris.seed,
        )
        replay_results = dict(run.results)
        replay_results[result_id] = replayed
        with pytest.raises(ValueError, match="payload hash"):
            verify_analysis_envelope(run.envelope, replay_results)

        class CallerMorrisResult(MorrisResult):
            pass

        subclass = CallerMorrisResult(
            stratum_id=morris.stratum_id,
            mu_star=morris.mu_star,
            sigma=morris.sigma,
            n_evaluations=morris.n_evaluations,
            seed=morris.seed,
        )
        subclass_results = dict(run.results)
        subclass_results[result_id] = subclass
        with pytest.raises(TypeError, match="exact production result types"):
            verify_analysis_envelope(run.envelope, subclass_results)

    def test_foreign_hand_built_envelope_fails_registered_design(self, bundles):
        """A self-hashed envelope cannot substitute a foreign design registration."""
        from sim.energy.gsa import stratum_for_bundle
        from sim.energy.registration import canonical_json_sha256

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        run = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        result_id = f"{bundles[0].bundle_id}/morris"
        genuine = run.results[result_id]
        hand_built = MorrisResult(
            stratum_id=genuine.stratum_id,
            mu_star={name: 123.0 for name in genuine.mu_star},
            sigma={name: 1.0 for name in genuine.sigma},
            n_evaluations=genuine.n_evaluations,
            seed=genuine.seed,
        )
        hand_built_results = dict(run.results)
        hand_built_results[result_id] = hand_built
        hand_built_manifest = tuple(
            dataclasses.replace(
                binding,
                payload_sha256=(
                    canonical_json_sha256(hand_built.payload())
                    if binding.result_id == result_id
                    else binding.payload_sha256
                ),
            )
            for binding in run.envelope.result_manifest
        )
        foreign_unsigned = dataclasses.replace(
            run.envelope,
            design_sha256="f" * 64,
            result_manifest=hand_built_manifest,
            payload_sha256="0" * 64,
        )
        foreign = dataclasses.replace(
            foreign_unsigned,
            payload_sha256=canonical_json_sha256(foreign_unsigned.payload()),
        )
        assert isinstance(foreign, AnalysisEnvelope)
        with pytest.raises(ValueError, match="registered design"):
            verify_analysis_envelope(foreign, hand_built_results)

    def test_hand_built_self_consistent_archive_has_no_eligibility(
        self, bundles, tmp_path
    ):
        """R9-1: public hashes prove consistency, never production issuance."""
        from sim.energy.gsa import stratum_for_bundle
        from sim.energy.registration import canonical_json_sha256

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        genuine = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        hand_built_results = {}
        for result_id, result in genuine.results.items():
            if type(result) is MorrisResult:
                forged = dataclasses.replace(
                    result,
                    mu_star={name: 123456.0 for name in result.mu_star},
                    sigma={name: -98765.0 for name in result.sigma},
                )
            elif type(result) is SobolResult:
                forged = dataclasses.replace(
                    result,
                    first_order={name: 123456.0 for name in result.first_order},
                    total_order={name: -98765.0 for name in result.total_order},
                )
            else:  # pragma: no cover - this run has exactly Morris/Sobol results
                raise AssertionError(type(result))
            hand_built_results[result_id] = forged
        hand_built_manifest = tuple(
            dataclasses.replace(
                binding,
                payload_sha256=canonical_json_sha256(
                    hand_built_results[binding.result_id].payload()
                ),
            )
            for binding in genuine.envelope.result_manifest
        )
        unsigned = dataclasses.replace(
            genuine.envelope,
            result_manifest=hand_built_manifest,
            payload_sha256="0" * 64,
        )
        hand_built_envelope = dataclasses.replace(
            unsigned,
            payload_sha256=canonical_json_sha256(unsigned.payload()),
        )

        assert verify_analysis_envelope(hand_built_envelope, hand_built_results)
        archive_path = tmp_path / "hand-built.json"
        archive_analysis(archive_path, hand_built_envelope, hand_built_results)
        loaded = load_analysis_archive(archive_path)
        assert loaded.envelope_verified
        assert loaded.execution_classification == "development_only"
        assert not hasattr(loaded, "confirmatory")
        assert not confirmatory_status(archive_path, None)

    def test_model_registration_is_content_and_state_bound(self, bundles):
        """R9-2: opposite semantic state changes the model and envelope hashes."""
        from sim.energy.gsa import stratum_for_bundle

        class SignedModel:
            def __init__(self, sign):
                self.sign = sign

            def analysis_model_descriptor(self):
                return {
                    "kind": "python-callable",
                    "parameters": {"sign": self.sign},
                    "policy_path": None,
                    "policy_sha256": None,
                    "bundle_path": None,
                    "bundle_sha256": None,
                }

            def __call__(self, _stratum_id, parameters):
                return self.sign * sum(parameters.values())

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        positive = stratified_gsa(
            SignedModel(1),
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        negative = stratified_gsa(
            SignedModel(-1),
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )

        assert positive.envelope.model_registration.registration_sha256 != (
            negative.envelope.model_registration.registration_sha256
        )
        assert positive.envelope.payload_sha256 != negative.envelope.payload_sha256
        assert all(
            estimator.source_closure_sha256
            == positive.envelope.source_closure_sha256
            for estimator in positive.envelope.estimators
        )
        assert positive.envelope.producer_identity["source_closure_sha256"] == (
            positive.envelope.source_closure_sha256
        )

    def test_production_model_without_identity_contract_is_rejected(self, bundles):
        from sim.energy.gsa import stratum_for_bundle

        calls = {"count": 0}

        def unregistered_model(_stratum_id, parameters):
            calls["count"] += 1
            return sum(parameters.values())

        with pytest.raises(TypeError, match="analysis_model_descriptor"):
            stratified_gsa(
                unregistered_model,
                [stratum_for_bundle(bundle) for bundle in bundles],
                n_trajectories=2,
                n_base=8,
                seed=20260729,
                bundles=bundles,
            )
        assert calls["count"] == 0

    def test_postflight_receipt_is_separate_and_dirty_state_fails_closed(
        self, bundles, tmp_path, monkeypatch
    ):
        """Simulate repository evidence; never mutate the real repository or commit."""
        from sim.energy import gsa as gsa_module
        from sim.energy.gsa import stratum_for_bundle
        from sim.energy.registration import canonical_json_sha256

        run = stratified_gsa(
            registered_sum_model,
            [stratum_for_bundle(bundle) for bundle in bundles],
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        unsigned = dataclasses.replace(
            run.envelope,
            execution_classification="receipt_pending",
            payload_sha256="0" * 64,
        )
        candidate_envelope = dataclasses.replace(
            unsigned,
            payload_sha256=canonical_json_sha256(unsigned.payload()),
        )
        archive_path = tmp_path / "receipt-candidate.json"
        archive_path.write_bytes(
            analysis_run_to_bytes(candidate_envelope, run.results)
        )

        monkeypatch.setattr(
            gsa_module,
            "_repository_state",
            lambda: gsa_module._RepositoryState("a" * 40, ()),
        )
        monkeypatch.setattr(
            gsa_module,
            "_archive_repository_path",
            lambda _path: "sim/energy/data/receipt-candidate.json",
        )
        monkeypatch.setattr(
            gsa_module,
            "_require_repository_bindings",
            lambda **_kwargs: None,
        )
        receipt_path = tmp_path / "analysis-postflight.json"
        receipt = analysis_postflight_receipt(archive_path, receipt_path)
        assert receipt.status == "eligible"
        assert confirmatory_status(archive_path, receipt_path)
        assert analysis_postflight_receipt(archive_path, receipt_path) == receipt

        mismatch = json.loads(receipt_path.read_text())
        mismatch["source_closure_sha256"] = "f" * 64
        mismatch["receipt_sha256"] = canonical_json_sha256(
            {key: value for key, value in mismatch.items() if key != "receipt_sha256"}
        )
        mismatch_path = tmp_path / "closure-mismatch.json"
        mismatch_path.write_text(
            json.dumps(mismatch, sort_keys=True, separators=(",", ":")) + "\n"
        )
        assert not confirmatory_status(archive_path, mismatch_path)

        monkeypatch.setattr(
            gsa_module,
            "_repository_state",
            lambda: gsa_module._RepositoryState(
                "a" * 40,
                (" M sim/energy/gsa.py",),
            ),
        )
        with pytest.raises(RuntimeError, match="dirty repository"):
            analysis_postflight_receipt(
                archive_path,
                tmp_path / "must-not-exist.json",
            )
        assert not (tmp_path / "must-not-exist.json").exists()

    def test_false_marginal_sampler_source_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        """R10-1 exact probe: all public hashes are recomputed after tampering."""
        archive_data = _receipt_candidate_bytes(
            _registered_shapley_run(bundles),
            false_sampler="marginal_sampler",
        )
        archive_path = tmp_path / "false-marginal.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "false-marginal-receipt.json"

        with pytest.raises(
            ValueError,
            match=r"marginal_sampler source_file differs",
        ):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_false_conditional_sampler_source_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        archive_data = _receipt_candidate_bytes(
            _registered_shapley_run(bundles),
            false_sampler="conditional_sampler",
        )
        archive_path = tmp_path / "false-conditional.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "false-conditional-receipt.json"

        with pytest.raises(
            ValueError,
            match=r"conditional_sampler source_file differs",
        ):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_false_marginal_callable_source_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        """R11-1 exact probe: all public hashes are recomputed after tampering."""
        archive_data = _receipt_candidate_bytes(
            _registered_shapley_run(bundles),
            registration_tamper=(
                "marginal_sampler",
                {"callable_source_sha256": "f" * 64},
            ),
        )
        archive_path = tmp_path / "false-marginal-callable.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "false-marginal-callable-receipt.json"

        with pytest.raises(
            ValueError,
            match=r"marginal_sampler callable_source differs",
        ):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_false_primary_callable_source_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        from sim.energy.gsa import stratum_for_bundle

        run = stratified_gsa(
            registered_sum_model,
            [stratum_for_bundle(bundle) for bundle in bundles],
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        archive_data = _receipt_candidate_bytes(
            run,
            registration_tamper=(
                "model_registration",
                {"callable_source_sha256": "f" * 64},
            ),
        )
        archive_path = tmp_path / "false-primary-callable.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "false-primary-callable-receipt.json"

        with pytest.raises(
            ValueError,
            match=r"model_registration callable_source differs",
        ):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_sampler_qualname_for_different_callable_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        run = _registered_shapley_run(bundles)
        marginal = run.envelope.estimators[0].parameters["marginal_sampler"]
        conditional = run.envelope.estimators[0].parameters["conditional_sampler"]
        assert marginal["source_path"] == conditional["source_path"]
        assert marginal["qualname"] != conditional["qualname"]
        archive_data = _receipt_candidate_bytes(
            run,
            registration_tamper=(
                "marginal_sampler",
                {
                    "module": conditional["module"],
                    "qualname": conditional["qualname"],
                },
            ),
        )
        archive_path = tmp_path / "different-sampler-qualname.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "different-sampler-qualname-receipt.json"

        with pytest.raises(ValueError, match=r"marginal_sampler callable_source differs"):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_callable_missing_from_pinned_source_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        run = _registered_shapley_run(bundles)
        marginal = run.envelope.estimators[0].parameters["marginal_sampler"]
        missing_qualname = (
            marginal["qualname"].rsplit(".", 1)[0] + ".missing_sampler"
        )
        archive_data = _receipt_candidate_bytes(
            run,
            registration_tamper=(
                "marginal_sampler",
                {"qualname": missing_qualname},
            ),
        )
        archive_path = tmp_path / "missing-callable.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "missing-callable-receipt.json"

        with pytest.raises(
            ValueError,
            match=r"marginal_sampler callable cannot be located",
        ):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_sampler_source_change_after_execution_refuses_receipt_and_status(
        self, bundles, tmp_path, monkeypatch
    ):
        run = _registered_shapley_run(bundles)
        archive_data = _receipt_candidate_bytes(run)
        archive_path = tmp_path / "stale-sampler.json"
        archive_path.write_bytes(archive_data)
        marginal = run.envelope.estimators[0].parameters["marginal_sampler"]
        _simulate_clean_repository(
            monkeypatch,
            archive_data,
            stale_source_path=marginal["source_path"],
        )
        receipt_path = tmp_path / "stale-sampler-receipt.json"

        with pytest.raises(ValueError, match=r"source_file differs"):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_genuine_registered_shapley_run_passes_simulated_clean_postflight(
        self, bundles, tmp_path, monkeypatch
    ):
        run = _registered_shapley_run(bundles)
        archive_data = _receipt_candidate_bytes(run)
        archive_path = tmp_path / "genuine-shapley.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / "genuine-shapley-receipt.json"

        receipt = analysis_postflight_receipt(archive_path, receipt_path)
        assert receipt.status == "eligible"
        assert confirmatory_status(archive_path, receipt_path)
        assert analysis_postflight_receipt(archive_path, receipt_path) == receipt

    @pytest.mark.parametrize(
        "false_artifact",
        ("policy", "bundle"),
    )
    def test_false_placementscript_artifact_hash_refuses_receipt(
        self, bundles, tmp_path, monkeypatch, false_artifact
    ):
        from sim.energy import gsa as gsa_module
        from sim.energy.gsa import stratum_for_bundle

        policy_path = "sim/energy/data/energy_analysis_registration.json"
        bundle_path = "sim/energy/README.md"

        def placementscript_model(_stratum_id, parameters):
            return sum(parameters.values())

        def descriptor():
            policy_sha256 = hashlib.sha256(
                (gsa_module._REPOSITORY_ROOT / policy_path).read_bytes()
            ).hexdigest()
            bundle_sha256 = hashlib.sha256(
                (gsa_module._REPOSITORY_ROOT / bundle_path).read_bytes()
            ).hexdigest()
            if false_artifact == "policy":
                policy_sha256 = "f" * 64
            else:
                bundle_sha256 = "f" * 64
            return {
                "kind": "placementscript",
                "parameters": {"operation": "sum-parameters"},
                "policy_path": policy_path,
                "policy_sha256": policy_sha256,
                "bundle_path": bundle_path,
                "bundle_sha256": bundle_sha256,
            }

        placementscript_model.analysis_model_descriptor = descriptor
        run = stratified_gsa(
            placementscript_model,
            [stratum_for_bundle(bundle) for bundle in bundles],
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        archive_data = _receipt_candidate_bytes(run)
        archive_path = tmp_path / f"false-placementscript-{false_artifact}.json"
        archive_path.write_bytes(archive_data)
        _simulate_clean_repository(monkeypatch, archive_data)
        receipt_path = tmp_path / f"{false_artifact}-receipt.json"

        with pytest.raises(
            ValueError,
            match=rf"placementscript_{false_artifact} differs",
        ):
            analysis_postflight_receipt(archive_path, receipt_path)
        assert not receipt_path.exists()
        assert not confirmatory_status(archive_path, receipt_path)

    def test_archive_round_trip_is_fresh_process_verifiable(self, bundles, tmp_path):
        """R8-2: durable canonical JSON survives process boundaries and is load-bearing."""
        from sim.energy.gsa import stratum_for_bundle

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        run = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        serialized = analysis_run_to_bytes(run.envelope, run.results)
        restored = analysis_run_from_bytes(serialized)
        assert restored.envelope_verified
        assert restored.execution_classification == "development_only"
        assert restored.results == run.results

        archive_path = tmp_path / "e4-gsa-analysis.json"
        assert archive_analysis(
            archive_path, run.envelope, run.results
        ) == archive_path
        assert load_analysis_archive(archive_path).envelope_verified
        with pytest.raises(FileExistsError, match="already exists"):
            archive_analysis(archive_path, run.envelope, run.results)

        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "from sim.energy.gsa import load_analysis_archive, "
                    "verify_analysis_envelope; import sys; "
                    "run=load_analysis_archive(sys.argv[1]); "
                    "assert verify_analysis_envelope(run.envelope, run.results); "
                    "print(len(run.results))"
                ),
                str(archive_path),
            ],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "6"

    def test_serialized_tampering_and_omission_are_rejected(
        self, bundles, tmp_path
    ):
        from sim.energy.gsa import stratum_for_bundle
        from sim.energy.registration import canonical_json_sha256

        strata = [stratum_for_bundle(bundle) for bundle in bundles]
        run = stratified_gsa(
            registered_sum_model,
            strata,
            n_trajectories=2,
            n_base=8,
            seed=20260729,
            bundles=bundles,
        )
        artifact = json.loads(analysis_run_to_bytes(run.envelope, run.results))
        artifact["results"][0]["statistics"]["mu_star"][
            next(iter(artifact["results"][0]["statistics"]["mu_star"]))
        ] += 1.0
        artifact["payload_sha256"] = canonical_json_sha256(
            {key: value for key, value in artifact.items() if key != "payload_sha256"}
        )
        with pytest.raises(ValueError, match="payload hash"):
            analysis_run_from_bytes(json.dumps(artifact).encode())

        partial = json.loads(analysis_run_to_bytes(run.envelope, run.results))
        partial["results"].pop()
        partial["payload_sha256"] = canonical_json_sha256(
            {key: value for key, value in partial.items() if key != "payload_sha256"}
        )
        with pytest.raises(ValueError, match="incomplete"):
            analysis_run_from_bytes(json.dumps(partial).encode())

        partial_results = dict(tuple(run.results.items())[:-1])
        with pytest.raises(ValueError, match="incomplete"):
            verify_analysis_envelope(
                run.envelope,
                partial_results,
            )
        refused_path = tmp_path / "partial-must-not-exist.json"
        with pytest.raises(ValueError, match="incomplete"):
            archive_analysis(refused_path, run.envelope, partial_results)
        assert not refused_path.exists()

        extra = dict(run.results)
        extra["foreign"] = next(iter(run.results.values()))
        with pytest.raises(ValueError, match="extras"):
            verify_analysis_envelope(run.envelope, extra)

    def test_builder_monkeypatch_cannot_authorize_subset(
        self, bundles, monkeypatch
    ):
        """R7-1: builder substitution is irrelevant to sealed-design authority."""
        from sim.energy import bundles as bundles_module
        from sim.energy.gsa import stratum_for_bundle

        monkeypatch.setattr(
            bundles_module,
            "build_development_bundles",
            lambda *_args, **_kwargs: bundles[:1],
        )
        calls = {"count": 0}

        def model(_stratum_id, params):
            calls["count"] += 1
            return sum(params.values())

        with pytest.raises(ValueError, match="full canonical development design"):
            stratified_gsa(
                model,
                [stratum_for_bundle(bundles[0])],
                n_trajectories=2,
                n_base=8,
                seed=20260729,
                bundles=bundles[:1],
            )
        assert calls["count"] == 0

    def test_design_sha256_pins_full_payload(self, bundles):
        """strata_design_sha256 must move when any conditional definition moves."""
        from sim.energy.gsa import strata_design_sha256, stratum_for_bundle

        extra = NuisanceParameter(
            "idle_fraction",
            0.1,
            0.9,
            conditional_basis="published idle spread (test)",
            within_stratum_independence=(
                "P2 V6 test assumption: independent of allocation nuisances "
                "within each registered bundle"
            ),
        )
        a = strata_design_sha256([stratum_for_bundle(b) for b in bundles])
        b_ = strata_design_sha256(
            [stratum_for_bundle(b, extra_nuisances=(extra,)) for b in bundles]
        )
        assert a != b_
        assert len(a) == 64

    def test_tampered_bundle_refuses_stratum_construction(self, bundles):
        import dataclasses

        from sim.energy.gsa import stratum_for_bundle

        tampered = dataclasses.replace(
            bundles[0], cloud_j_per_vcpu_hour=bundles[0].cloud_j_per_vcpu_hour * 1.01
        )
        with pytest.raises(ValueError, match="seal mismatch"):
            stratum_for_bundle(tampered)


class TestRegisteredDistributions:
    def test_loguniform_ppf_endpoints_and_median(self):
        p = NuisanceParameter(
            "x",
            1.0,
            100.0,
            conditional_basis="test basis",
            within_stratum_independence=TEST_INDEPENDENCE,
            distribution="loguniform",
        )
        assert p.ppf(0.0) == pytest.approx(1.0)
        assert p.ppf(1.0) == pytest.approx(100.0)
        assert p.ppf(0.5) == pytest.approx(10.0)  # geometric median, not 50.5

    def test_loguniform_requires_positive_low(self):
        with pytest.raises(ValueError, match="loguniform requires low > 0"):
            NuisanceParameter(
                "x",
                0.0,
                1.0,
                conditional_basis="test basis",
                within_stratum_independence=TEST_INDEPENDENCE,
                distribution="loguniform",
            )

    def test_unknown_distribution_rejected(self):
        with pytest.raises(ValueError, match="unknown distribution"):
            NuisanceParameter(
                "x",
                0.0,
                1.0,
                conditional_basis="test basis",
                within_stratum_independence=TEST_INDEPENDENCE,
                distribution="beta",
            )

    def test_swap_honors_registered_distribution(self):
        """The swap analysis must map common quantiles through each stratum's
        registered inverse CDF, not an affine range map (R1-9)."""

        def identity_model(stratum_id: str, params) -> float:
            return params["x"]

        uni = Stratum(
            "uni",
            (
                NuisanceParameter(
                    "x",
                    1.0,
                    100.0,
                    conditional_basis="test basis",
                    within_stratum_independence=TEST_INDEPENDENCE,
                ),
            ),
        )
        logu = Stratum(
            "logu",
            (
                NuisanceParameter(
                    "x",
                    1.0,
                    100.0,
                    conditional_basis="test basis",
                    within_stratum_independence=TEST_INDEPENDENCE,
                    distribution="loguniform",
                ),
            ),
        )
        res = nonconfirmatory_between_stratum_swap(
            identity_model, [uni, logu], n_points=400, seed=7
        )
        mean_uni = res.per_stratum_mean["uni"]
        mean_logu = res.per_stratum_mean["logu"]
        # E[U(1,100)] = 50.5; E[LogU(1,100)] = 99/ln(100) ~ 21.5 — must differ.
        assert mean_uni == pytest.approx(50.5, rel=0.1)
        assert mean_logu == pytest.approx(99.0 / np.log(100.0), rel=0.15)
