"""Dependency-preserving stratified GSA harness (TODO 3.5; plan §4.2, U4/V6).

Machine tuples enter the sensitivity design as **grouped/categorical factors**: an
analysis swaps whole machines, never one machine's power against another's capacity.
Continuous nuisance parameters (idle fraction, transition energy, allocation
coefficients, service-time inflation as an interference proxy) vary **conditionally
within each tuple's published spread** — the joint input space is therefore
dependent, and ordinary Morris/Sobol estimators are not automatically valid over it
(U4). The design implemented here is **stratified**:

- :func:`morris_screen` and :func:`sobol_indices` run **within one stratum** (one
  machine tuple / bundle), where the continuous nuisances are conditionally
  independent only under the explicit V6 attestations frozen in P2. Grouping
  machines removes tuple dependence but does **not** by itself make the nuisances
  independent: every :class:`NuisanceParameter` therefore carries both its
  conditional basis and a required ``within_stratum_independence`` attestation,
  and both fields enter the sealed design digest.
- :func:`between_stratum_swap` reports between-tuple variation **separately**, as a
  categorical swap analysis: the same unit-cube quantile draws are mapped into each
  stratum's own conditional distribution through its inverse CDF (rank-matched,
  dependency-preserving) and the model output is compared across strata.
- Strata are **bound to bundles** (R1-9/R2-3): :func:`stratum_for_bundle` derives the
  allocation-coefficient nuisances from a sealed bundle's published swept ranges,
  records the bundle's manifest digest, and **seals the full canonical stratum
  payload** (every bound/basis/independence attestation/distribution); every
  sampling entry point re-verifies the seal. :func:`validate_strata_bound` loads
  the separately stored, content-sealed registration and requires exact
  bundle/stratum identities plus equality with its full ``design_sha256``;
  runtime builders and caller-supplied bundles are never authority.
  :func:`strata_design_sha256` is the P2 freeze-point digest. The production
  :func:`stratified_gsa` and :func:`between_stratum_swap` entry points require
  ``bundles=`` and cannot accept free-form strata. Free-form strata remain
  constructable only for the explicitly named ``nonconfirmatory_*`` analytic
  harnesses. Standalone single-stratum Morris/Sobol calls are likewise
  non-confirmatory. Individual results carry immutable scientific data but no
  issuing capability or confirmatory flag. Production runners instead return one
  durable :class:`AnalysisRun`: its canonical-JSON :class:`AnalysisEnvelope` binds
  the registered design/bundles, content-bearing model and estimator identities,
  registered parameters and seeds, and the exact complete result manifest.
  :func:`verify_analysis_envelope` proves internal consistency only. Production
  eligibility is a separate postflight decision made by
  :func:`analysis_postflight_receipt` and checked by :func:`confirmatory_status`.
- Cross-stratum joint quantification, when needed, uses a **named dependent-input
  method**: :func:`exact_shapley_effects` implements the exact-permutation Shapley
  estimator of Song, Nelson & Staum (SIAM/ASA JUQ 2016, DOI 10.1137/15M1048070),
  taking caller-supplied marginal and conditional samplers (the dependence structure
  is the caller's declared model, not an implicit assumption). A standard Sobol
  decomposition over the joint dependent space is deliberately not offered. Every
  Shapley model call, including conditional sampling, uses the same scalar/finite
  output contract as the stratified estimators.

Sampling space: Morris/Sobol matrices are drawn in **unit-quantile space** ([0, 1]
per nuisance) and mapped through each parameter's registered inverse CDF
(:meth:`NuisanceParameter.ppf`) before model evaluation. For uniform marginals this
is numerically identical to sampling the bounds directly (SALib normalizes by the
bounds when computing elementary effects), and it makes non-uniform registered
distributions (V6: P2 enumerates them) valid rather than silently mis-sampled;
Sobol indices are invariant under each input's own quantile reparameterization.

Seeds: every entry point requires an explicit root seed (docs/SEEDS.md); per-stratum
and per-method streams are derived with ``analysis.tracking.derive_seed``.

Compute cost: :func:`estimate_compute_cost` returns the exact evaluation counts and a
formatted **G0 capacity-table line item** (plan §8; the G0 table is the arm-level
capacity screen with a ≥30% reserve per ceiling).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim.energy.bundles import EnergyModelBundle
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

from analysis.tracking import derive_seed
from sim.energy.registration import (
    EnergyAnalysisRegistration,
    canonical_json_sha256,
    load_energy_analysis_registration,
)

#: model(stratum_id, {nuisance name: value}) -> scalar output (e.g. the headline
#: comparison statistic re-computed under those nuisances).
StratifiedModel = Callable[[str, Mapping[str, float]], float]


# ------------------------------------------------------------------ shared validators
# R2-9: one set of runtime input validators shared by the estimators *and* the
# compute-cost calculator, so every design the G0 table prices is executable and
# the "exact cost" claim holds for every accepted input.
def _pos_int(name: str, v: object, minimum: int = 1) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return v


def _power_of_two(name: str, v: object) -> int:
    out = _pos_int(name, v, minimum=2)
    if out & (out - 1) != 0:
        raise ValueError(f"{name} must be a power of two (Sobol' sequence)")
    return out


def _shapley_loops(n_outer: object, n_inner: object) -> tuple[int, int]:
    return (_pos_int("n_outer", n_outer, minimum=2), _pos_int("n_inner", n_inner, minimum=2))


def _require_finite_values(name: str, values: Mapping[str, float]) -> None:
    bad = sorted(key for key, value in values.items() if not math.isfinite(value))
    if bad:
        raise ValueError(f"{name} contains non-finite values for {bad}")


def _scalar_finite_model_output(value: object, context: str) -> float:
    """The one-scalar, finite model-output contract shared by every estimator."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: model must return one scalar output") from exc
    if array.ndim != 0:
        raise ValueError(f"{context}: model must return one scalar output")
    output = float(array)
    if not math.isfinite(output):
        raise ValueError(
            f"{context}: model produced a non-finite output; "
            "sensitivity analysis is undefined"
        )
    return output


_MIN_V6_STATEMENT_CHARS = 8


def _require_text(name: str, value: object, *, minimum: int = 1) -> str:
    """Return canonical stripped text, rejecting truthy non-strings and prose-free
    placeholders. V6 basis/attestation fields use a small minimum meaningful
    length; identifiers use only the non-empty-string rule."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str")
    stripped = value.strip()
    if len(stripped) < minimum:
        raise ValueError(
            f"{name} must contain at least {minimum} non-whitespace characters"
        )
    return stripped


# ------------------------------------------------------------------ strata
_DISTRIBUTIONS = ("uniform", "loguniform")
_STRATUM_SEAL_TAG = "gsa-stratum-seal-v1:"


def _stratum_seal(payload: dict) -> str:
    return hashlib.sha256(
        (
            _STRATUM_SEAL_TAG
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ).encode()
    ).hexdigest()


def _stratum_payload(
    stratum_id: str,
    nuisances: tuple[NuisanceParameter, ...],
    bundle_manifest_sha256: str | None,
) -> dict:
    """The single canonical stratum-payload builder (used by ``Stratum.payload`` and
    ``stratum_for_bundle`` — one definition, so seal payloads cannot drift)."""
    return {
        "stratum_id": stratum_id,
        "nuisances": [p.payload() for p in nuisances],
        "bundle_manifest_sha256": bundle_manifest_sha256,
    }


@dataclass(frozen=True)
class NuisanceParameter:
    """One continuous nuisance, bounded by the tuple's published spread.

    ``conditional_basis`` records where the bounds come from (e.g. "published swept
    range of allocation entry ccf-aws-cpu" or "transition energy ± published
    measurement spread"). ``within_stratum_independence`` is the separate, explicit
    V6 attestation stating the conditional-independence premise for this nuisance;
    prose about a range is never treated as that premise. P2 seals both strings.
    ``distribution`` is the registered conditional distribution family within the
    spread; sampling flows through :meth:`ppf`, so a non-uniform registration is
    honored rather than silently replaced by an affine map (R1-9).
    """

    name: str
    low: float
    high: float
    conditional_basis: str
    within_stratum_independence: str
    distribution: str = "uniform"

    def __post_init__(self) -> None:
        name = _require_text("nuisance parameter name", self.name)
        conditional_basis = _require_text(
            f"{name}: conditional_basis",
            self.conditional_basis,
            minimum=_MIN_V6_STATEMENT_CHARS,
        )
        independence = _require_text(
            f"{name}: within_stratum_independence attestation",
            self.within_stratum_independence,
            minimum=_MIN_V6_STATEMENT_CHARS,
        )
        distribution = _require_text(f"{name}: distribution", self.distribution)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "conditional_basis", conditional_basis)
        object.__setattr__(self, "within_stratum_independence", independence)
        object.__setattr__(self, "distribution", distribution)
        if (
            isinstance(self.low, bool)
            or isinstance(self.high, bool)
            or not isinstance(self.low, int | float)
            or not isinstance(self.high, int | float)
            or not math.isfinite(self.low)
            or not math.isfinite(self.high)
            or self.low >= self.high
        ):
            raise ValueError(f"{self.name}: need finite low < high")
        if self.distribution not in _DISTRIBUTIONS:
            raise ValueError(
                f"{self.name}: unknown distribution {self.distribution!r} "
                f"(registered families: {_DISTRIBUTIONS})"
            )
        if self.distribution == "loguniform" and self.low <= 0:
            raise ValueError(f"{self.name}: loguniform requires low > 0")

    def ppf(self, q: np.ndarray | float) -> np.ndarray | float:
        """Inverse CDF of the registered conditional distribution on [low, high]."""
        if self.distribution == "uniform":
            return self.low + q * (self.high - self.low)
        # loguniform
        return self.low * (self.high / self.low) ** q

    def payload(self) -> dict:
        """Canonical registration payload: every field (R2-3 — the P2 pin covers the
        full conditional definition, not just a name)."""
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "conditional_basis": self.conditional_basis,
            "within_stratum_independence": self.within_stratum_independence,
            "distribution": self.distribution,
        }


@dataclass(frozen=True)
class Stratum:
    """One machine-tuple stratum: the categorical factor level plus the conditional
    distributions of its continuous nuisances within their published spreads (the
    distribution family per nuisance is part of the P2 registration).

    ``bundle_manifest_sha256`` binds the stratum to the sealed bundle it was derived
    from (:func:`stratum_for_bundle`, the only constructor that seals a bound
    stratum — R2-3); ``None`` marks a free-form stratum that is accepted only by the
    explicitly non-confirmatory harnesses. The seal covers the **whole canonical
    payload** — every nuisance's bounds, basis, independence attestation, and
    distribution — so a bound stratum whose contents were altered after
    construction fails at every registered sampling entry point."""

    stratum_id: str
    nuisances: tuple[NuisanceParameter, ...]
    bundle_manifest_sha256: str | None = None
    seal: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stratum_id", _require_text("stratum_id", self.stratum_id)
        )
        names = [p.name for p in self.nuisances]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{self.stratum_id}: nuisance names must be non-empty and unique")
        if self.bundle_manifest_sha256 is not None and not self.seal:
            raise ValueError(
                f"{self.stratum_id}: bundle-bound strata carry a seal — construct "
                "them through stratum_for_bundle() (R2-3)"
            )

    def payload(self) -> dict:
        """The canonical stratum payload (R2-3): stratum id, every nuisance's full
        conditional definition, and the bundle digest. This is what the seal covers
        and what :func:`strata_design_sha256` pins for P2."""
        return _stratum_payload(self.stratum_id, self.nuisances, self.bundle_manifest_sha256)

    def verify_binding(self) -> None:
        """Require a bundle-bound seal; unregistered strata always fail closed."""
        if self.bundle_manifest_sha256 is None:
            raise ValueError(
                f"{self.stratum_id}: unregistered/free-form stratum is "
                "development-only and cannot enter a registered GSA path"
            )
        if self.seal != _stratum_seal(self.payload()):
            raise ValueError(
                f"{self.stratum_id}: stratum seal mismatch — the bound stratum's "
                "distributions/bounds/basis/independence attestations were changed "
                "after construction (R2-3/V6)"
            )

    def verify_nonconfirmatory(self) -> None:
        """Require an honestly unbound stratum for analytic development harnesses."""
        if self.bundle_manifest_sha256 is not None or self.seal:
            raise ValueError(
                f"{self.stratum_id}: bound strata must use the registered production path"
            )

    def problem(self) -> dict:
        """SALib problem in unit-quantile space (see the module docstring)."""
        return {
            "num_vars": len(self.nuisances),
            "names": [p.name for p in self.nuisances],
            "bounds": [[0.0, 1.0] for _ in self.nuisances],
        }

    def _to_values(self, q_row: np.ndarray) -> dict[str, float]:
        return {p.name: float(p.ppf(q)) for p, q in zip(self.nuisances, q_row, strict=True)}

    def _evaluate(self, model: StratifiedModel, q_matrix: np.ndarray) -> np.ndarray:
        outputs = np.array(
            [
                _scalar_finite_model_output(
                    model(self.stratum_id, self._to_values(row)),
                    self.stratum_id,
                )
                for row in q_matrix
            ],
            dtype=float,
        )
        if outputs.shape != (q_matrix.shape[0],):
            raise ValueError(f"{self.stratum_id}: model must return one scalar per design row")
        return outputs


# ------------------------------------------------------------------ durable analysis provenance
_RESULT_SCHEMA = "energy-gsa-result-v2"
_MODEL_REGISTRATION_SCHEMA = "energy-analysis-model-registration-v2"
_SOURCE_CLOSURE_SCHEMA = "energy-analysis-source-closure-v2"
_ENVELOPE_SCHEMA = "energy-analysis-envelope-v3"
_ARCHIVE_SCHEMA = "energy-analysis-archive-v3"
_POSTFLIGHT_SCHEMA = "energy-analysis-postflight-receipt-v2"
_DEVELOPMENT_ONLY = "development_only"
_RECEIPT_PENDING = "receipt_pending"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REGISTRATION_RELATIVE_PATH = Path(
    "sim/energy/data/energy_analysis_registration.json"
)
_ANALYSIS_STRATIFIED = "stratified-gsa"
_ANALYSIS_SWAP = "between-stratum-swap"
_ANALYSIS_SHAPLEY = "registered-shapley-effects"
_ESTIMATOR_IMPLEMENTATIONS = {
    "morris": "sim.energy.gsa:morris-unit-quantile-v1",
    "sobol": "sim.energy.gsa:sobol-unit-quantile-first-total-v1",
    "between-stratum-swap": "sim.energy.gsa:rank-matched-categorical-swap-v1",
    "shapley": "sim.energy.gsa:exact-permutation-dependent-input-shapley-v1",
}
_PRODUCTION_ENTRYPOINTS = {
    _ANALYSIS_STRATIFIED: "sim.energy.gsa.stratified_gsa",
    _ANALYSIS_SWAP: "sim.energy.gsa.between_stratum_swap",
    _ANALYSIS_SHAPLEY: "sim.energy.gsa.registered_shapley_effects",
}


@dataclass(frozen=True, kw_only=True)
class _GsaResult:
    """Immutable scientific result with deliberately no in-process authority.

    Membership in a verified :class:`AnalysisEnvelope` proves completeness and
    internal consistency, never production eligibility.
    """

    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("result seed must be an explicit int")

    def _estimator(self) -> str:
        raise NotImplementedError

    def _statistics_payload(self) -> dict[str, object]:
        raise NotImplementedError

    def result_id(self) -> str:
        raise NotImplementedError

    def payload(self) -> dict[str, object]:
        """Canonical, serialization-safe scientific payload."""
        return {
            "schema": _RESULT_SCHEMA,
            "estimator": self._estimator(),
            "seed": self.seed,
            "statistics": self._statistics_payload(),
        }


@dataclass(frozen=True)
class ModelRegistration:
    """Content-bearing identity for one analysis model.

    The callable supplies the semantic descriptor through the
    ``analysis_model_descriptor()`` contract. This class adds independently
    inspected callable source and whole-file hashes, then seals the canonical
    descriptor. A postflight receipt additionally requires ``source_path`` to be
    the exact committed blob at the receipt's repository commit.
    """

    kind: str
    module: str
    qualname: str
    source_path: str
    callable_source_sha256: str
    source_file_sha256: str
    parameters: Mapping[str, object]
    policy_path: str | None
    policy_sha256: str | None
    bundle_path: str | None
    bundle_sha256: str | None
    registration_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"python-callable", "placementscript"}:
            raise ValueError("model registration kind must be python-callable or placementscript")
        for name in ("module", "qualname", "source_path"):
            _require_text(f"model registration {name}", getattr(self, name))
        source_path = Path(self.source_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("model registration source_path must be repository-relative")
        for name in (
            "callable_source_sha256",
            "source_file_sha256",
            "registration_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"model registration {name} must be a SHA-256 digest")
        parameters = _freeze_envelope_json(dict(self.parameters))
        # Canonical serialization is part of the model identity contract. This
        # rejects NaN, unsupported objects, and ambiguous caller state now.
        canonical_json_sha256(_thaw_envelope_json(parameters))
        object.__setattr__(self, "parameters", parameters)
        if self.kind == "python-callable":
            if any(
                value is not None
                for value in (
                    self.policy_path,
                    self.policy_sha256,
                    self.bundle_path,
                    self.bundle_sha256,
                )
            ):
                raise ValueError(
                    "python-callable model registrations cannot claim policy/bundle artifacts"
                )
        else:
            for name in ("policy_path", "bundle_path"):
                value = getattr(self, name)
                _require_text(f"PlacementScript {name}", value)
                artifact_path = Path(value)
                if artifact_path.is_absolute() or ".." in artifact_path.parts:
                    raise ValueError(
                        f"PlacementScript {name} must be repository-relative"
                    )
            if not _is_sha256(self.policy_sha256) or not _is_sha256(
                self.bundle_sha256
            ):
                raise ValueError(
                    "PlacementScript model registrations require policy_sha256 "
                    "and bundle_sha256"
                )
        if canonical_json_sha256(self.payload()) != self.registration_sha256:
            raise ValueError("model registration canonical hash mismatch")

    def payload(self) -> dict[str, object]:
        return {
            "schema": _MODEL_REGISTRATION_SCHEMA,
            "kind": self.kind,
            "module": self.module,
            "qualname": self.qualname,
            "source_path": self.source_path,
            "callable_source_sha256": self.callable_source_sha256,
            "source_file_sha256": self.source_file_sha256,
            "parameters": _thaw_envelope_json(self.parameters),
            "policy_path": self.policy_path,
            "policy_sha256": self.policy_sha256,
            "bundle_path": self.bundle_path,
            "bundle_sha256": self.bundle_sha256,
        }

    def artifact(self) -> dict[str, object]:
        return {**self.payload(), "registration_sha256": self.registration_sha256}


@dataclass(frozen=True)
class EstimatorRegistration:
    """One estimator implementation plus all parameters that determined its output."""

    estimator: str
    implementation: str
    source_closure_sha256: str
    seed: int
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            self.estimator not in _ESTIMATOR_IMPLEMENTATIONS
            or self.implementation != _ESTIMATOR_IMPLEMENTATIONS[self.estimator]
        ):
            raise ValueError("unregistered GSA estimator identity")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("estimator seed must be an explicit int")
        if not _is_sha256(self.source_closure_sha256):
            raise ValueError("estimator source_closure_sha256 must be a SHA-256 digest")
        object.__setattr__(
            self,
            "parameters",
            _freeze_envelope_json(dict(self.parameters)),
        )

    def payload(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "implementation": self.implementation,
            "source_closure_sha256": self.source_closure_sha256,
            "seed": self.seed,
            "parameters": _thaw_envelope_json(self.parameters),
        }


@dataclass(frozen=True)
class ResultBinding:
    """Canonical manifest entry for one required result payload."""

    result_id: str
    result_type: str
    payload_sha256: str

    def __post_init__(self) -> None:
        _require_text("result_id", self.result_id)
        _require_text("result_type", self.result_type)
        if not _is_sha256(self.payload_sha256):
            raise ValueError("result payload_sha256 must be a SHA-256 digest")

    def payload(self) -> dict[str, str]:
        return {
            "result_id": self.result_id,
            "result_type": self.result_type,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, kw_only=True)
class AnalysisEnvelope:
    """Durable run-level provenance envelope.

    ``payload_sha256`` seals every other field. The result manifest is complete by
    construction and is independently checked against the registered design and
    estimator parameters by :func:`verify_analysis_envelope`. Neither the envelope
    nor its self-hash is production authority.
    """

    analysis: str
    design_sha256: str
    registration_payload_sha256: str
    bundle_manifests: tuple[tuple[str, str], ...]
    model_registration: ModelRegistration
    source_closure_sha256: str
    producer_identity: Mapping[str, str]
    execution_classification: str
    estimators: tuple[EstimatorRegistration, ...]
    result_manifest: tuple[ResultBinding, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.analysis not in _PRODUCTION_ENTRYPOINTS:
            raise ValueError("unknown production analysis identity")
        if type(self.model_registration) is not ModelRegistration:
            raise TypeError("analysis envelope requires the exact ModelRegistration type")
        if not _is_sha256(self.design_sha256):
            raise ValueError("analysis envelope design_sha256 must be a SHA-256 digest")
        if not _is_sha256(self.registration_payload_sha256):
            raise ValueError(
                "analysis envelope registration_payload_sha256 must be a SHA-256 digest"
            )
        if not _is_sha256(self.source_closure_sha256):
            raise ValueError(
                "analysis envelope source_closure_sha256 must be a SHA-256 digest"
            )
        if not _is_sha256(self.payload_sha256):
            raise ValueError("analysis envelope payload_sha256 must be a SHA-256 digest")
        if self.execution_classification not in {
            _DEVELOPMENT_ONLY,
            _RECEIPT_PENDING,
        }:
            raise ValueError("unknown analysis execution classification")
        object.__setattr__(self, "bundle_manifests", tuple(self.bundle_manifests))
        object.__setattr__(self, "estimators", tuple(self.estimators))
        object.__setattr__(self, "result_manifest", tuple(self.result_manifest))
        producer_identity = dict(self.producer_identity)
        if producer_identity != {
            "entrypoint": _PRODUCTION_ENTRYPOINTS[self.analysis],
            "source_closure_sha256": self.source_closure_sha256,
        }:
            raise ValueError("analysis envelope producer identity mismatch")
        object.__setattr__(
            self,
            "producer_identity",
            types.MappingProxyType(producer_identity),
        )

    def payload(self) -> dict[str, object]:
        """Every field covered by ``payload_sha256`` (the digest excludes itself)."""
        return {
            "schema": _ENVELOPE_SCHEMA,
            "analysis": self.analysis,
            "design_sha256": self.design_sha256,
            "registration_payload_sha256": self.registration_payload_sha256,
            "bundles": [
                {"bundle_id": bundle_id, "manifest_sha256": manifest}
                for bundle_id, manifest in self.bundle_manifests
            ],
            "model_registration": self.model_registration.artifact(),
            "source_closure_sha256": self.source_closure_sha256,
            "producer_identity": dict(self.producer_identity),
            "execution_classification": self.execution_classification,
            "estimators": [estimator.payload() for estimator in self.estimators],
            "results": [binding.payload() for binding in self.result_manifest],
        }

    def artifact(self) -> dict[str, object]:
        """JSON envelope including its canonical payload digest."""
        return {**self.payload(), "payload_sha256": self.payload_sha256}


@dataclass(frozen=True)
class AnalysisRun:
    """Results plus their internally verifiable run envelope."""

    envelope: AnalysisEnvelope
    results: Mapping[str, _GsaResult]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "results",
            types.MappingProxyType(dict(self.results)),
        )

    @property
    def envelope_verified(self) -> bool:
        """Whether the complete envelope is internally consistent."""
        try:
            return verify_analysis_envelope(self.envelope, self.results)
        except (TypeError, ValueError):
            return False

    @property
    def execution_classification(self) -> str:
        """The non-authorizing run label preserved through archive round trips."""
        return self.envelope.execution_classification

    def by_stratum(self) -> Mapping[str, tuple[MorrisResult, SobolResult]]:
        """Compatibility view for an internally verified stratified run."""
        verify_analysis_envelope(self.envelope, self.results)
        if self.envelope.analysis != _ANALYSIS_STRATIFIED:
            raise ValueError("by_stratum() requires a stratified GSA run")
        ordered: dict[str, tuple[MorrisResult, SobolResult]] = {}
        for bundle_id, _ in self.envelope.bundle_manifests:
            morris = self.results[f"{bundle_id}/morris"]
            sobol = self.results[f"{bundle_id}/sobol"]
            if type(morris) is not MorrisResult or type(sobol) is not SobolResult:
                raise TypeError("stratified result has an unexpected exact type")
            ordered[bundle_id] = (morris, sobol)
        return types.MappingProxyType(ordered)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze_envelope_json(value: object) -> object:
    if isinstance(value, Mapping):
        return types.MappingProxyType(
            {str(key): _freeze_envelope_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_envelope_json(item) for item in value)
    return value


def _thaw_envelope_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_envelope_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_envelope_json(item) for item in value]
    return value


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(_REPOSITORY_ROOT.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError(
            "production model source must be a file inside the repository"
        ) from exc


def _callable_source_target(function: object) -> object:
    if inspect.isfunction(function) or inspect.ismethod(function):
        return function
    if not callable(function):
        raise TypeError("production analysis model must be callable")
    return type(function).__call__


def _callable_source_span(
    source_file_bytes: bytes,
    *,
    module: str,
    qualname: str,
    source_path: str,
) -> bytes:
    """Extract one callable's canonical source lines from repository file bytes."""
    module_parts = module.split(".")
    if any(not part.isidentifier() for part in module_parts):
        raise ValueError("callable module is not a valid dotted module name")
    path = Path(source_path)
    if path.suffix != ".py":
        raise ValueError("callable source file must be Python")
    if path.name == "__init__.py":
        expected_module_parts: list[str] = []
        package_path = path.parent
    else:
        expected_module_parts = [path.stem]
        package_path = path.parent
    while (
        package_path != Path(".")
        and (_REPOSITORY_ROOT / package_path / "__init__.py").is_file()
    ):
        expected_module_parts.insert(0, package_path.name)
        package_path = package_path.parent
    if module_parts != expected_module_parts:
        raise ValueError("callable module does not identify its source file")

    parts = [part for part in qualname.split(".") if part != "<locals>"]
    if not parts or any(not part.isidentifier() for part in parts):
        raise ValueError("callable qualname is not statically locatable")
    try:
        tree = ast.parse(source_file_bytes, filename=source_path)
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError("callable source file is not valid Python") from exc

    definition_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def definitions_in_scope(scope: ast.AST):
        def visit(node: ast.AST):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, definition_types):
                    yield child
                else:
                    yield from visit(child)

        yield from visit(scope)

    scope: ast.AST = tree
    target: ast.AST | None = None
    for part in parts:
        matches = [
            definition
            for definition in definitions_in_scope(scope)
            if definition.name == part
        ]
        if len(matches) != 1:
            raise ValueError("callable qualname is missing or ambiguous in source file")
        target = matches[0]
        scope = target
    if isinstance(target, ast.ClassDef):
        matches = [
            definition
            for definition in definitions_in_scope(target)
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
            and definition.name == "__call__"
        ]
        if len(matches) != 1:
            raise ValueError("callable class has no unique __call__ source span")
        target = matches[0]
    if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("callable qualname does not identify executable source")

    first_line = min(
        (target.lineno, *(decorator.lineno for decorator in target.decorator_list))
    )
    if target.end_lineno is None:
        raise ValueError("callable source span has no end position")
    lines = source_file_bytes.splitlines(keepends=True)
    return b"".join(lines[first_line - 1 : target.end_lineno])


def _model_registration(model: object) -> ModelRegistration:
    """Apply the production model identity contract and inspect executable content."""
    provider = getattr(model, "analysis_model_descriptor", None)
    if not callable(provider):
        raise TypeError(
            "production analysis model must expose analysis_model_descriptor()"
        )
    declared = provider()
    if not isinstance(declared, Mapping):
        raise TypeError("analysis_model_descriptor() must return a mapping")
    if set(declared) != {
        "kind",
        "parameters",
        "policy_path",
        "policy_sha256",
        "bundle_path",
        "bundle_sha256",
    }:
        raise ValueError(
            "analysis model descriptor must contain exactly kind, parameters, "
            "policy_path, policy_sha256, bundle_path, and bundle_sha256"
        )
    parameters = declared["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("analysis model descriptor parameters must be a mapping")
    source_target = _callable_source_target(model)
    try:
        source_file_raw = inspect.getsourcefile(source_target)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "production model executable source must be inspectable"
        ) from exc
    if source_file_raw is None:
        raise ValueError("production model executable source file is unavailable")
    source_file = Path(source_file_raw)
    module = getattr(model, "__module__", None)
    qualname = getattr(model, "__qualname__", None)
    if not isinstance(module, str) or not module:
        module = type(model).__module__
    if not isinstance(qualname, str) or not qualname:
        qualname = type(model).__qualname__
    source_path = _repository_relative(source_file)
    source_file_bytes = source_file.read_bytes()
    try:
        callable_source = _callable_source_span(
            source_file_bytes,
            module=module,
            qualname=qualname,
            source_path=source_path,
        )
    except ValueError as exc:
        raise ValueError(
            "production model executable source must be statically locatable"
        ) from exc
    fields = {
        "kind": declared["kind"],
        "module": module,
        "qualname": qualname,
        "source_path": source_path,
        "callable_source_sha256": hashlib.sha256(callable_source).hexdigest(),
        "source_file_sha256": hashlib.sha256(source_file_bytes).hexdigest(),
        "parameters": dict(parameters),
        "policy_path": declared["policy_path"],
        "policy_sha256": declared["policy_sha256"],
        "bundle_path": declared["bundle_path"],
        "bundle_sha256": declared["bundle_sha256"],
    }
    unsigned = {
        "schema": _MODEL_REGISTRATION_SCHEMA,
        **fields,
    }
    return ModelRegistration(
        **fields,
        registration_sha256=canonical_json_sha256(unsigned),
    )


def _sim_energy_source_records() -> tuple[dict[str, str], ...]:
    energy_root = _REPOSITORY_ROOT / "sim" / "energy"
    paths = sorted(
        (
            path
            for path in energy_root.rglob("*.py")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(_REPOSITORY_ROOT).as_posix(),
    )
    if not paths:
        raise ValueError("sim/energy source closure is empty")
    return tuple(
        {
            "path": path.relative_to(_REPOSITORY_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    )


@dataclass(frozen=True)
class _RepositoryArtifactBinding:
    """One repository file whose content is claimed by an executable identity."""

    role: str
    path: str
    sha256: str

    def payload(self) -> dict[str, str]:
        return {"role": self.role, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class _ExecutableRegistrationBinding:
    """One executable registration and every repository artifact it references."""

    location: str
    registration: ModelRegistration
    artifacts: tuple[_RepositoryArtifactBinding, ...]

    def closure_payload(self) -> dict[str, object]:
        return {
            "location": self.location,
            "registration_sha256": self.registration.registration_sha256,
            "repository_artifacts": [
                artifact.payload() for artifact in self.artifacts
            ],
        }


def _registration_repository_artifacts(
    registration: ModelRegistration,
) -> tuple[_RepositoryArtifactBinding, ...]:
    artifacts = [
        _RepositoryArtifactBinding(
            role="source_file",
            path=registration.source_path,
            sha256=registration.source_file_sha256,
        )
    ]
    if registration.kind == "placementscript":
        if registration.policy_path is None or registration.bundle_path is None:
            raise ValueError(
                "PlacementScript registration lacks repository artifact paths"
            )
        if registration.policy_sha256 is None or registration.bundle_sha256 is None:
            raise ValueError(
                "PlacementScript registration lacks repository artifact hashes"
            )
        artifacts.extend(
            (
                _RepositoryArtifactBinding(
                    role="placementscript_policy",
                    path=registration.policy_path,
                    sha256=registration.policy_sha256,
                ),
                _RepositoryArtifactBinding(
                    role="placementscript_bundle",
                    path=registration.bundle_path,
                    sha256=registration.bundle_sha256,
                ),
            )
        )
    return tuple(artifacts)


def _canonical_executable_registrations(
    model_registration: ModelRegistration,
    estimator_parameter_payloads: Sequence[
        tuple[str, Mapping[str, object]]
    ] = (),
) -> tuple[_ExecutableRegistrationBinding, ...]:
    """Enumerate every executable registration represented by an envelope.

    The primary model is first. Nested registration artifacts are then discovered
    recursively in estimator parameters in canonical estimator/key/index order.
    Any registration-bearing mapping with an unknown schema fails closed instead
    of silently falling outside the repository receipt traversal.
    """
    if type(model_registration) is not ModelRegistration:
        raise TypeError("primary executable registration must use the exact type")

    bindings = [
        _ExecutableRegistrationBinding(
            location="model_registration",
            registration=model_registration,
            artifacts=_registration_repository_artifacts(model_registration),
        )
    ]
    estimator_names = [name for name, _ in estimator_parameter_payloads]
    if (
        any(not isinstance(name, str) or not name for name in estimator_names)
        or len(estimator_names) != len(set(estimator_names))
    ):
        raise ValueError("estimator parameter payload identities must be unique")

    def visit(location: str, value: object) -> None:
        if isinstance(value, Mapping):
            schema = value.get("schema")
            if schema == _MODEL_REGISTRATION_SCHEMA:
                registration = _model_registration_from_payload(value)
                bindings.append(
                    _ExecutableRegistrationBinding(
                        location=location,
                        registration=registration,
                        artifacts=_registration_repository_artifacts(registration),
                    )
                )
                return
            if "registration_sha256" in value:
                raise ValueError(
                    f"{location}: unsupported executable registration schema"
                )
            for key in sorted(value):
                visit(f"{location}.{key}", value[key])
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                visit(f"{location}[{index}]", item)

    visit("model_registration.parameters", model_registration.parameters)
    for estimator_name, parameters in sorted(
        estimator_parameter_payloads,
        key=lambda item: item[0],
    ):
        if not isinstance(parameters, Mapping):
            raise TypeError("estimator parameters must be mappings")
        visit(f"estimators.{estimator_name}.parameters", parameters)
    return tuple(bindings)


def _estimator_parameter_payloads(
    estimators: Sequence[EstimatorRegistration],
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    return tuple(
        (estimator.estimator, estimator.parameters) for estimator in estimators
    )


def _source_closure_sha256(
    model: ModelRegistration,
    estimator_parameter_payloads: Sequence[
        tuple[str, Mapping[str, object]]
    ] = (),
) -> str:
    executable_registrations = _canonical_executable_registrations(
        model,
        estimator_parameter_payloads,
    )
    return canonical_json_sha256(
        {
            "schema": _SOURCE_CLOSURE_SCHEMA,
            "sim_energy_sources": list(_sim_energy_source_records()),
            "executable_registrations": [
                binding.closure_payload()
                for binding in executable_registrations
            ],
        }
    )


def _working_tree_is_clean() -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return not completed.stdout.strip()


def _execution_classification() -> str:
    return _RECEIPT_PENDING if _working_tree_is_clean() else _DEVELOPMENT_ONLY


def _receipt_timestamp_utc() -> str:
    """Operational receipt time; never used by sampling or result computation."""
    clock = datetime
    return clock.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class MorrisResult(_GsaResult):
    stratum_id: str
    mu_star: Mapping[str, float]
    sigma: Mapping[str, float]
    n_evaluations: int

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "mu_star", types.MappingProxyType(dict(self.mu_star)))
        object.__setattr__(self, "sigma", types.MappingProxyType(dict(self.sigma)))
        _require_finite_values("Morris mu_star", self.mu_star)
        _require_finite_values("Morris sigma", self.sigma)
        if set(self.mu_star) != set(self.sigma) or not self.mu_star:
            raise ValueError("Morris result mappings must cover the same non-empty parameter set")
        _pos_int("n_evaluations", self.n_evaluations)

    def _estimator(self) -> str:
        return "morris"

    def result_id(self) -> str:
        return f"{self.stratum_id}/morris"

    def _statistics_payload(self) -> dict[str, object]:
        return {
            "stratum_id": self.stratum_id,
            "mu_star": dict(self.mu_star),
            "sigma": dict(self.sigma),
            "n_evaluations": self.n_evaluations,
        }

    def ranking(self) -> tuple[str, ...]:
        """Parameters by descending mu*; ties broken by name (deterministic)."""
        return tuple(sorted(self.mu_star, key=lambda k: (-self.mu_star[k], k)))


def _morris_screen(
    model: StratifiedModel,
    stratum: Stratum,
    *,
    n_trajectories: int,
    seed: int,
    num_levels: int,
    bound: bool,
) -> MorrisResult:
    _pos_int("n_trajectories", n_trajectories)
    _pos_int("num_levels", num_levels, minimum=2)
    if bound:
        stratum.verify_binding()
    else:
        stratum.verify_nonconfirmatory()
    problem = stratum.problem()
    s = derive_seed(seed, "gsa", "morris", stratum.stratum_id)
    x = morris_sample.sample(problem, N=n_trajectories, num_levels=num_levels, seed=s)
    y = stratum._evaluate(model, x)
    res = morris_analyze.analyze(
        problem, x, y, num_levels=num_levels, seed=derive_seed(s, "analyze")
    )
    names = problem["names"]
    mu_star = {n: float(v) for n, v in zip(names, res["mu_star"], strict=True)}
    sigma = {n: float(v) for n, v in zip(names, res["sigma"], strict=True)}
    # SALib can itself emit NaN for undefined designs (e.g. constant output).
    # Validate before constructing an archivable result, in addition to the
    # pre-analysis finite model-output check in Stratum._evaluate (R5-3).
    _require_finite_values("Morris mu_star", mu_star)
    _require_finite_values("Morris sigma", sigma)
    return MorrisResult(
        stratum_id=stratum.stratum_id,
        mu_star=mu_star,
        sigma=sigma,
        n_evaluations=int(x.shape[0]),
        seed=seed,
    )


def morris_screen(
    model: StratifiedModel,
    stratum: Stratum,
    *,
    n_trajectories: int,
    seed: int,
    num_levels: int = 4,
) -> MorrisResult:
    """Standalone Morris screening for a sealed bundle-bound stratum.

    A single-stratum call cannot establish completeness of the registered
    three-bundle design, so its result is explicitly development-only. Use
    :func:`stratified_gsa` to obtain a complete registered envelope.
    """
    return _morris_screen(
        model,
        stratum,
        n_trajectories=n_trajectories,
        seed=seed,
        num_levels=num_levels,
        bound=True,
    )


def nonconfirmatory_morris_screen(
    model: StratifiedModel,
    stratum: Stratum,
    *,
    n_trajectories: int,
    seed: int,
    num_levels: int = 4,
) -> MorrisResult:
    """Analytic/test-only Morris harness; rejects bundle-bound production strata."""
    return _morris_screen(
        model,
        stratum,
        n_trajectories=n_trajectories,
        seed=seed,
        num_levels=num_levels,
        bound=False,
    )


@dataclass(frozen=True)
class SobolResult(_GsaResult):
    stratum_id: str
    first_order: Mapping[str, float]
    total_order: Mapping[str, float]
    n_evaluations: int

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "first_order", types.MappingProxyType(dict(self.first_order))
        )
        object.__setattr__(
            self, "total_order", types.MappingProxyType(dict(self.total_order))
        )
        _require_finite_values("Sobol first_order", self.first_order)
        _require_finite_values("Sobol total_order", self.total_order)
        if set(self.first_order) != set(self.total_order) or not self.first_order:
            raise ValueError("Sobol result mappings must cover the same non-empty parameter set")
        _pos_int("n_evaluations", self.n_evaluations)

    def _estimator(self) -> str:
        return "sobol"

    def result_id(self) -> str:
        return f"{self.stratum_id}/sobol"

    def _statistics_payload(self) -> dict[str, object]:
        return {
            "stratum_id": self.stratum_id,
            "first_order": dict(self.first_order),
            "total_order": dict(self.total_order),
            "n_evaluations": self.n_evaluations,
        }


def _sobol_indices(
    model: StratifiedModel,
    stratum: Stratum,
    *,
    n_base: int,
    seed: int,
    calc_second_order: bool,
    bound: bool,
) -> SobolResult:
    _power_of_two("n_base", n_base)
    if bound:
        stratum.verify_binding()
    else:
        stratum.verify_nonconfirmatory()
    problem = stratum.problem()
    s = derive_seed(seed, "gsa", "sobol", stratum.stratum_id)
    x = sobol_sample.sample(problem, N=n_base, calc_second_order=calc_second_order, seed=s)
    y = stratum._evaluate(model, x)
    res = sobol_analyze.analyze(
        problem,
        y,
        calc_second_order=calc_second_order,
        seed=derive_seed(s, "analyze"),
    )
    names = problem["names"]
    first_order = {n: float(v) for n, v in zip(names, res["S1"], strict=True)}
    total_order = {n: float(v) for n, v in zip(names, res["ST"], strict=True)}
    _require_finite_values("Sobol first_order", first_order)
    _require_finite_values("Sobol total_order", total_order)
    return SobolResult(
        stratum_id=stratum.stratum_id,
        first_order=first_order,
        total_order=total_order,
        n_evaluations=int(x.shape[0]),
        seed=seed,
    )


def sobol_indices(
    model: StratifiedModel,
    stratum: Stratum,
    *,
    n_base: int,
    seed: int,
    calc_second_order: bool = False,
) -> SobolResult:
    """Standalone Sobol indices for a sealed bundle-bound stratum.

    The result is development-only because one stratum cannot prove completeness
    of the registered design. Use :func:`stratified_gsa` for complete output.
    """
    return _sobol_indices(
        model,
        stratum,
        n_base=n_base,
        seed=seed,
        calc_second_order=calc_second_order,
        bound=True,
    )


def nonconfirmatory_sobol_indices(
    model: StratifiedModel,
    stratum: Stratum,
    *,
    n_base: int,
    seed: int,
    calc_second_order: bool = False,
) -> SobolResult:
    """Analytic/test-only Sobol harness; rejects bundle-bound production strata."""
    return _sobol_indices(
        model,
        stratum,
        n_base=n_base,
        seed=seed,
        calc_second_order=calc_second_order,
        bound=False,
    )


def stratified_gsa(
    model: StratifiedModel,
    strata: Sequence[Stratum],
    *,
    n_trajectories: int,
    n_base: int,
    seed: int,
    bundles: Sequence[EnergyModelBundle],
) -> AnalysisRun:
    """Run Morris + Sobol independently within every stratum (the pre-registered
    §4.2 design); between-stratum effects are *not* mixed in — use
    :func:`between_stratum_swap` for those.

    ``bundles`` is mandatory checked evidence: both it and the strata must be exact
    bijections with the separately loaded sealed registration, with exact manifest
    and full-design hashes before any sampling (R5-2/R2-3/R6/R7).
    Free-form analytic work uses :func:`nonconfirmatory_stratified_gsa`.

    The return value carries one envelope over all ``2 * len(strata)`` results;
    individual results contain no issuer, seal, or confirmatory field."""
    registration = validate_strata_bound(strata, bundles)
    model_registration = _model_registration(model)
    source_closure_sha256 = _source_closure_sha256(model_registration)
    _pos_int("n_trajectories", n_trajectories)
    _power_of_two("n_base", n_base)
    out: dict[str, _GsaResult] = {}
    for stratum in strata:
        morris = _morris_screen(
            model,
            stratum,
            n_trajectories=n_trajectories,
            seed=seed,
            num_levels=4,
            bound=True,
        )
        sobol = _sobol_indices(
            model,
            stratum,
            n_base=n_base,
            seed=seed,
            calc_second_order=False,
            bound=True,
        )
        out[morris.result_id()] = morris
        out[sobol.result_id()] = sobol
    estimators = (
        EstimatorRegistration(
            estimator="morris",
            implementation=_ESTIMATOR_IMPLEMENTATIONS["morris"],
            source_closure_sha256=source_closure_sha256,
            seed=seed,
            parameters={"n_trajectories": n_trajectories, "num_levels": 4},
        ),
        EstimatorRegistration(
            estimator="sobol",
            implementation=_ESTIMATOR_IMPLEMENTATIONS["sobol"],
            source_closure_sha256=source_closure_sha256,
            seed=seed,
            parameters={"n_base": n_base, "calc_second_order": False},
        ),
    )
    return _complete_analysis_run(
        analysis=_ANALYSIS_STRATIFIED,
        registration=registration,
        model_registration=model_registration,
        source_closure_sha256=source_closure_sha256,
        estimators=estimators,
        results=out,
    )


def nonconfirmatory_stratified_gsa(
    model: StratifiedModel,
    strata: Sequence[Stratum],
    *,
    n_trajectories: int,
    n_base: int,
    seed: int,
) -> dict[str, tuple[MorrisResult, SobolResult]]:
    """Free-form analytic/test harness, structurally isolated from production."""
    _validate_nonconfirmatory_strata(strata)
    out: dict[str, tuple[MorrisResult, SobolResult]] = {}
    for stratum in strata:
        out[stratum.stratum_id] = (
            nonconfirmatory_morris_screen(
                model, stratum, n_trajectories=n_trajectories, seed=seed
            ),
            nonconfirmatory_sobol_indices(model, stratum, n_base=n_base, seed=seed),
        )
    return out


# ------------------------------------------------------------------ between strata
@dataclass(frozen=True)
class SwapResult(_GsaResult):
    """Categorical swap analysis: the headline statistic re-computed per machine
    tuple under rank-matched nuisance draws. Reported separately from the
    within-stratum indices (plan §4.2)."""

    per_stratum_outputs: Mapping[str, tuple[float, ...]]
    per_stratum_mean: Mapping[str, float]
    ranking: tuple[str, ...]  # strata by ascending mean output (ties by id)
    max_pairwise_mean_gap: float
    sign_consistent: bool  # True iff every stratum's mean output has the same sign
    n_evaluations: int

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "per_stratum_outputs",
            types.MappingProxyType(
                {
                    stratum_id: tuple(values)
                    for stratum_id, values in self.per_stratum_outputs.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "per_stratum_mean",
            types.MappingProxyType(dict(self.per_stratum_mean)),
        )
        for stratum_id, values in self.per_stratum_outputs.items():
            if not values or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{stratum_id}: swap outputs must be non-empty and finite")
        _require_finite_values("swap per_stratum_mean", self.per_stratum_mean)
        if not math.isfinite(self.max_pairwise_mean_gap):
            raise ValueError("swap max_pairwise_mean_gap must be finite")
        _pos_int("n_evaluations", self.n_evaluations)
        if not isinstance(self.sign_consistent, bool):
            raise ValueError("swap sign_consistent flag must be a bool")

    def _estimator(self) -> str:
        return "between-stratum-swap"

    def result_id(self) -> str:
        return "between-stratum-swap"

    def _statistics_payload(self) -> dict[str, object]:
        return {
            "per_stratum_outputs": {
                stratum_id: list(values)
                for stratum_id, values in self.per_stratum_outputs.items()
            },
            "per_stratum_mean": dict(self.per_stratum_mean),
            "ranking": list(self.ranking),
            "max_pairwise_mean_gap": self.max_pairwise_mean_gap,
            "sign_consistent": self.sign_consistent,
            "n_evaluations": self.n_evaluations,
        }


def _between_stratum_swap(
    model: StratifiedModel,
    strata: Sequence[Stratum],
    *,
    n_points: int,
    seed: int,
) -> SwapResult:
    _pos_int("n_points", n_points)
    names = [p.name for p in strata[0].nuisances]
    rng = np.random.default_rng(derive_seed(seed, "gsa", "swap"))
    unit = rng.random((n_points, len(names)))
    outputs: dict[str, tuple[float, ...]] = {}
    for stratum in strata:
        outputs[stratum.stratum_id] = tuple(float(v) for v in stratum._evaluate(model, unit))
    means = {sid: math.fsum(v) / len(v) for sid, v in outputs.items()}
    ranking = tuple(sorted(means, key=lambda k: (means[k], k)))
    vals = [means[sid] for sid in ranking]
    return SwapResult(
        per_stratum_outputs=outputs,
        per_stratum_mean=means,
        ranking=ranking,
        max_pairwise_mean_gap=vals[-1] - vals[0] if vals else 0.0,
        sign_consistent=all(v > 0 for v in vals) or all(v < 0 for v in vals),
        n_evaluations=n_points * len(strata),
        seed=seed,
    )


def between_stratum_swap(
    model: StratifiedModel,
    strata: Sequence[Stratum],
    *,
    n_points: int,
    seed: int,
    bundles: Sequence[EnergyModelBundle],
) -> AnalysisRun:
    """Swap registered whole-machine tuples under matched nuisance conditions.

    ``bundles`` is mandatory checked evidence and registration-anchored exact-bijection
    validation runs before sampling. Common unit-cube quantiles are mapped through
    each registered conditional distribution, preserving paired rank matching
    (R1-9/R2-3/R5-2/R6).
    """
    registration = validate_strata_bound(strata, bundles)
    model_registration = _model_registration(model)
    source_closure_sha256 = _source_closure_sha256(model_registration)
    _pos_int("n_points", n_points)
    result = _between_stratum_swap(
        model, strata, n_points=n_points, seed=seed
    )
    estimator = EstimatorRegistration(
        estimator="between-stratum-swap",
        implementation=_ESTIMATOR_IMPLEMENTATIONS["between-stratum-swap"],
        source_closure_sha256=source_closure_sha256,
        seed=seed,
        parameters={"n_points": n_points, "rank_matching": "common-unit-quantiles"},
    )
    return _complete_analysis_run(
        analysis=_ANALYSIS_SWAP,
        registration=registration,
        model_registration=model_registration,
        source_closure_sha256=source_closure_sha256,
        estimators=(estimator,),
        results={result.result_id(): result},
    )


def nonconfirmatory_between_stratum_swap(
    model: StratifiedModel,
    strata: Sequence[Stratum],
    *,
    n_points: int,
    seed: int,
) -> SwapResult:
    """Free-form analytic/test swap harness; never reachable through production."""
    _validate_nonconfirmatory_strata(strata)
    return _between_stratum_swap(model, strata, n_points=n_points, seed=seed)


def _validate_strata(strata: Sequence[Stratum]) -> None:
    if not strata:
        raise ValueError("need at least one stratum")
    ids = [s.stratum_id for s in strata]
    if len(ids) != len(set(ids)):
        raise ValueError("stratum ids must be unique")
    name_sets = {tuple(p.name for p in s.nuisances) for s in strata}
    if len(name_sets) != 1:
        raise ValueError(
            "all strata must expose the same nuisance parameter names (the conditional "
            "ranges differ per stratum; the parameter set does not)"
        )
    for s in strata:
        s.verify_binding()  # R2-3: tampered bound strata never reach sampling


def _validate_nonconfirmatory_strata(strata: Sequence[Stratum]) -> None:
    if not strata:
        raise ValueError("need at least one stratum")
    ids = [s.stratum_id for s in strata]
    if len(ids) != len(set(ids)):
        raise ValueError("stratum ids must be unique")
    name_sets = {tuple(p.name for p in s.nuisances) for s in strata}
    if len(name_sets) != 1:
        raise ValueError(
            "all strata must expose the same nuisance parameter names (the conditional "
            "ranges differ per stratum; the parameter set does not)"
        )
    for s in strata:
        s.verify_nonconfirmatory()


# ------------------------------------------------------------------ bundle binding
def stratum_for_bundle(
    bundle: "EnergyModelBundle",  # noqa: UP037 - forward ref, imported lazily below
    extra_nuisances: tuple[NuisanceParameter, ...] = (),
) -> Stratum:
    """Build a machine-tuple stratum bound to a **sealed** bundle (R1-9).

    The allocation-coefficient nuisances are derived from the bundle's published
    swept ranges (their ``conditional_basis`` cites the pinned entry ids); the
    stratum records the bundle's manifest digest and is **sealed over its full
    canonical payload** (R2-3), so the P2 registration and
    :func:`validate_strata_bound` can verify the binding and any post-construction
    edit fails at sampling. ``extra_nuisances`` (idle fraction, transition energy,
    service-time inflation, ...) must carry their own published
    ``conditional_basis`` and explicit V6 independence attestation. They remain
    useful for development, but enter a registered run only if their complete
    definitions already appear in the sealed design registration; caller replay is
    never registration authority (R7-2).
    """
    bundle.verify_seal()
    cloud_lo, cloud_hi = bundle.cloud_swept_j_per_vcpu_hour
    net_lo, net_hi = bundle.network_swept_j_per_byte
    allocation_independence = (
        "P2 V6 assumption: conditionally independent of every other continuous "
        f"nuisance within sealed bundle stratum {bundle.bundle_id}; this is a "
        "registered modeling assumption, not implied by tuple grouping"
    )
    nuisances = (
        NuisanceParameter(
            name="cloud_j_per_vcpu_hour",
            low=cloud_lo,
            high=cloud_hi,
            conditional_basis=(
                f"published swept range of allocation entry {bundle.cloud_entry_id} "
                f"(bundle {bundle.bundle_id})"
            ),
            within_stratum_independence=allocation_independence,
        ),
        NuisanceParameter(
            name="network_j_per_byte",
            low=net_lo,
            high=net_hi,
            conditional_basis=(
                f"published swept range of allocation entry {bundle.network_entry_id} "
                f"(bundle {bundle.bundle_id})"
            ),
            within_stratum_independence=allocation_independence,
        ),
        *extra_nuisances,
    )
    digest = bundle.manifest_sha256()
    return Stratum(
        stratum_id=bundle.bundle_id,
        nuisances=nuisances,
        bundle_manifest_sha256=digest,
        seal=_stratum_seal(_stratum_payload(bundle.bundle_id, nuisances, digest)),
    )


def validate_strata_bound(
    strata: Sequence[Stratum],
    bundles: Sequence[EnergyModelBundle],
) -> EnergyAnalysisRegistration:
    """Sealed-registration check for the complete registered design.

    The caller's ``bundles`` are evidence; authority comes from the separately
    loaded, canonical-JSON-sealed artifact. Runtime builder output is never used to
    decide completeness (R7-1), and the supplied strata's recomputed full design
    hash must equal the registered ``design_sha256`` (R7-2).

    1. **Registered completeness**: supplied bundle and stratum ids must each be an
       exact bijection with the sealed development identities — no subsets, extras,
       or free-form strata.
    2. **Seal**: every stratum's seal verifies over its full canonical payload.
    3. **Bundle identity**: every supplied bundle validates against the canonical
       source registry and exactly matches its registered manifest hash.
    4. **Design identity**: every complete stratum payload and the recomputed
       ``strata_design_sha256`` equal the registered artifact. Caller-replayed
       extras are not registration authority and hard-fail.
    """
    _validate_strata(strata)

    registration = load_energy_analysis_registration()
    registration.verify_seal()
    from sim.energy.bundles import (
        default_capacity_mapping,
        validate_bundle,
    )
    from sim.energy.sources import load_pinned_sources

    registry = load_pinned_sources()
    mapping = default_capacity_mapping(registry)
    registered_manifests = dict(registration.bundle_manifests("development"))
    expected_ids = set(registered_manifests)

    stratum_ids = {s.stratum_id for s in strata}
    supplied_bundle_ids = [b.bundle_id for b in bundles]
    bundle_ids = set(supplied_bundle_ids)
    if len(supplied_bundle_ids) != len(bundle_ids):
        raise ValueError("bundle ids must be unique")
    if bundle_ids != expected_ids:
        raise ValueError(
            "bundles must be an exact bijection with the full canonical development "
            f"design (R2-3/R6): expected {sorted(expected_ids)}, got "
            f"{sorted(bundle_ids)}"
        )
    if stratum_ids != expected_ids:
        raise ValueError(
            "strata must be an exact bijection with the full canonical development "
            f"design (R2-3/R6): expected {sorted(expected_ids)}, got "
            f"{sorted(stratum_ids)}"
        )

    for supplied in bundles:
        validate_bundle(supplied, registry, mapping)
        if supplied.manifest_sha256() != registered_manifests[supplied.bundle_id]:
            raise ValueError(
                f"{supplied.bundle_id}: supplied bundle differs from the canonical "
                "sealed development manifest"
            )

    registered_strata = {
        stratum["stratum_id"]: stratum
        for stratum in registration.strata_payloads()
    }
    for s in strata:
        if s.bundle_manifest_sha256 is None:
            raise ValueError(f"{s.stratum_id}: stratum is not bound to a bundle manifest")
        if s.bundle_manifest_sha256 != registered_manifests[s.stratum_id]:
            raise ValueError(
                f"{s.stratum_id}: stratum digest does not match the canonical "
                "registered bundle manifest"
            )
        if canonical_json_sha256(s.payload()) != canonical_json_sha256(
            registered_strata[s.stratum_id]
        ):
            raise ValueError(
                f"{s.stratum_id}: stratum is not re-derivable from the sealed "
                "registered design — "
                "conditional bounds/basis/independence attestations/distributions "
                "or the nuisance set diverge (R2-3/R7-2/V6)"
            )

    if strata_design_sha256(strata) != registration.design_sha256:
        raise ValueError(
            "strata design_sha256 does not match the sealed P2 design registration; "
            "caller-selected or caller-extended designs are development-only"
        )
    return registration


def _strata_design_payload(strata: Sequence[Stratum]) -> dict[str, object]:
    ordered = sorted(strata, key=lambda stratum: stratum.stratum_id)
    return {
        "schema": "gsa-strata-design-registration-v1",
        "development_bundles": [
            {
                "bundle_id": stratum.stratum_id,
                "manifest_sha256": stratum.bundle_manifest_sha256,
            }
            for stratum in ordered
        ],
        "strata": [stratum.payload() for stratum in ordered],
    }


def strata_design_sha256(strata: Sequence[Stratum]) -> str:
    """Digest the complete canonical design artifact.

    The payload includes the schema, ordered development bundle ids and manifest
    hashes, and every stratum's complete nuisance definitions. This is the exact P2
    freeze point loaded by registered-path validation.
    """
    _validate_strata(strata)
    return canonical_json_sha256(_strata_design_payload(strata))


# ------------------------------------------------------------------ Shapley effects
def exact_shapley_effects(
    model: Callable[[Mapping[str, float]], float],
    names: Sequence[str],
    marginal_sampler: Callable[[np.random.Generator, int], np.ndarray],
    conditional_sampler: Callable[
        [np.random.Generator, Mapping[str, float], Sequence[str]], Mapping[str, float]
    ],
    *,
    n_outer: int,
    n_inner: int,
    seed: int,
) -> Mapping[str, float]:
    """Exact-permutation Shapley effects (Song, Nelson & Staum 2016) for dependent
    inputs — the named cross-stratum method of plan §4.2.

    The caller supplies the dependence structure explicitly:

    - ``marginal_sampler(rng, n)`` → array (n, d) of joint draws in ``names`` order;
    - ``conditional_sampler(rng, fixed, free_names)`` → one draw of ``free_names``
      conditional on the ``fixed`` assignment.

    Cost function c(J) = E[Var(Y | X_{J̄})], estimated with an outer loop over
    conditioning draws and an inner conditional-variance loop; the Shapley effect of
    input i is the permutation-averaged incremental cost, normalized by Var(Y).
    Exact permutation enumeration with **coalition-cost caching**: c(∅) and c(full)
    are free, every other coalition (2^d − 2 of them) is evaluated once at
    n_outer·n_inner model calls, plus the max(n_outer·n_inner, 512) variance
    baseline — :func:`shapley_eval_count` is the exact model-call count (R1-10).
    Use only for small d.
    """
    d = len(names)
    if d < 1 or len(set(names)) != d:
        raise ValueError("names must be non-empty and unique")
    n_outer, n_inner = _shapley_loops(n_outer, n_inner)
    rng = np.random.default_rng(derive_seed(seed, "gsa", "shapley"))

    joint = marginal_sampler(rng, max(n_outer * n_inner, 512))
    y_all = np.array(
        [
            _scalar_finite_model_output(
                model(dict(zip(names, row, strict=True))),
                "Shapley variance baseline",
            )
            for row in joint
        ],
        dtype=float,
    )
    var_y = float(np.var(y_all, ddof=1))
    if not math.isfinite(var_y):
        raise ValueError("model output variance is non-finite; Shapley effects undefined")
    if var_y <= 0:
        raise ValueError("model output variance is zero; Shapley effects undefined")

    def cost(coalition: tuple[str, ...]) -> float:
        """c(J) = E[Var(Y | X_{J̄})] for J = coalition (variance explained residually
        by the coalition once its complement is fixed). c(∅)=0, c(all)=Var(Y)."""
        if not coalition:
            return 0.0
        if len(coalition) == d:
            return var_y
        complement = [n for n in names if n not in coalition]
        cond_vars = []
        outer_draws = marginal_sampler(rng, n_outer)
        for row in outer_draws:
            fixed = {n: v for n, v in zip(names, row, strict=True) if n in complement}
            ys = []
            for _ in range(n_inner):
                free = conditional_sampler(rng, fixed, list(coalition))
                full = dict(fixed)
                full.update(free)
                ys.append(
                    _scalar_finite_model_output(
                        model(full),
                        "Shapley conditional sampling",
                    )
                )
            conditional_variance = float(
                np.var(np.asarray(ys, dtype=float), ddof=1)
            )
            if not math.isfinite(conditional_variance):
                raise ValueError(
                    "Shapley conditional variance is non-finite; effects are undefined"
                )
            cond_vars.append(conditional_variance)
        result = float(np.mean(cond_vars))
        if not math.isfinite(result):
            raise ValueError("Shapley coalition cost is non-finite; effects are undefined")
        return result

    cost_cache: dict[frozenset[str], float] = {}

    def cached_cost(coalition: tuple[str, ...]) -> float:
        key = frozenset(coalition)
        if key not in cost_cache:
            cost_cache[key] = cost(tuple(sorted(coalition)))
        return cost_cache[key]

    shapley = dict.fromkeys(names, 0.0)
    perms = list(permutations(names))
    for perm in perms:
        prefix: tuple[str, ...] = ()
        prev_cost = 0.0
        for name in perm:
            new = prefix + (name,)
            c = cached_cost(new)
            shapley[name] += c - prev_cost
            prefix, prev_cost = new, c
    effects = {n: shapley[n] / (len(perms) * var_y) for n in names}
    _require_finite_values("Shapley effects", effects)
    return types.MappingProxyType(effects)


@dataclass(frozen=True)
class ShapleyResult(_GsaResult):
    """Registered dependent-input Shapley result for the canonical design."""

    effects: Mapping[str, float]
    n_evaluations: int

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "effects",
            types.MappingProxyType(dict(self.effects)),
        )
        _require_finite_values("Shapley effects", self.effects)
        if not self.effects:
            raise ValueError("Shapley effects must cover a non-empty input set")
        _pos_int("n_evaluations", self.n_evaluations)

    def _estimator(self) -> str:
        return "shapley"

    def result_id(self) -> str:
        return "registered-shapley-effects"

    def _statistics_payload(self) -> dict[str, object]:
        return {
            "effects": dict(self.effects),
            "n_evaluations": self.n_evaluations,
        }


def registered_shapley_effects(
    model: Callable[[Mapping[str, float]], float],
    names: Sequence[str],
    marginal_sampler: Callable[[np.random.Generator, int], np.ndarray],
    conditional_sampler: Callable[
        [np.random.Generator, Mapping[str, float], Sequence[str]], Mapping[str, float]
    ],
    *,
    n_outer: int,
    n_inner: int,
    seed: int,
    strata: Sequence[Stratum],
    bundles: Sequence[EnergyModelBundle],
) -> AnalysisRun:
    """Production Shapley path bound to the canonical registered energy design.

    :func:`exact_shapley_effects` remains the explicitly analytic estimator. This
    wrapper is the archivable production boundary: it first validates the complete
    registered strata/bundle design, then returns a one-result envelope recording
    the model and sampler identities plus every loop parameter and the explicit
    seed.
    """
    registration = validate_strata_bound(strata, bundles)
    model_registration = _model_registration(model)
    marginal_registration = _model_registration(marginal_sampler)
    conditional_registration = _model_registration(conditional_sampler)
    name_tuple = tuple(names)
    n_outer, n_inner = _shapley_loops(n_outer, n_inner)
    estimator_parameters = {
        "names": list(name_tuple),
        "n_outer": n_outer,
        "n_inner": n_inner,
        "marginal_sampler": marginal_registration.artifact(),
        "conditional_sampler": conditional_registration.artifact(),
    }
    source_closure_sha256 = _source_closure_sha256(
        model_registration,
        (("shapley", estimator_parameters),),
    )
    effects = exact_shapley_effects(
        model,
        name_tuple,
        marginal_sampler,
        conditional_sampler,
        n_outer=n_outer,
        n_inner=n_inner,
        seed=seed,
    )
    result = ShapleyResult(
        effects=effects,
        n_evaluations=shapley_eval_count(len(name_tuple), n_outer, n_inner),
        seed=seed,
    )
    estimator = EstimatorRegistration(
        estimator="shapley",
        implementation=_ESTIMATOR_IMPLEMENTATIONS["shapley"],
        source_closure_sha256=source_closure_sha256,
        seed=seed,
        parameters=estimator_parameters,
    )
    return _complete_analysis_run(
        analysis=_ANALYSIS_SHAPLEY,
        registration=registration,
        model_registration=model_registration,
        source_closure_sha256=source_closure_sha256,
        estimators=(estimator,),
        results={result.result_id(): result},
    )


# ------------------------------------------------------------------ compute cost
def shapley_eval_count(d: int, n_outer: int, n_inner: int) -> int:
    """Exact model-call count of :func:`exact_shapley_effects` (R1-10): the variance
    baseline ``max(n_outer·n_inner, 512)`` plus one ``n_outer·n_inner`` block per
    cached coalition — c(∅) and c(full) are free, leaving ``2^d − 2`` coalitions.
    (The naive d!·d·n_outer·n_inner figure ignores the coalition cache and
    overstates the cost.) Shares the estimator's own input validators (R2-9), so
    every priced design is executable."""
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("d must be a positive int")
    n_outer, n_inner = _shapley_loops(n_outer, n_inner)
    return max(n_outer * n_inner, 512) + (2**d - 2) * n_outer * n_inner


@dataclass(frozen=True)
class GsaComputeCost:
    """Exact evaluation counts for the pre-registered design + a G0 line item."""

    n_strata: int
    n_nuisances: int
    morris_evals: int  # per stratum: N * (k + 1)
    sobol_evals: int  # per stratum: N * (k + 2) [first/total order only]
    swap_evals: int  # n_points * n_strata (shared across strata by CRN)
    shapley_evals: int  # shapley_eval_count(...); 0 unless the cross-stratum path is on
    total_evals: int
    seconds_per_eval: float
    total_hours: float

    def as_g0_line_item(self) -> str:
        """One row for the G0 arm-level capacity table (plan §8): item, evaluation
        count, wall-clock estimate at the measured per-evaluation cost, and the
        30%-reserve figure G0 checks against its ceiling."""
        reserve = self.total_hours * 1.3
        return (
            f"| stratified-GSA ({self.n_strata} strata x {self.n_nuisances} nuisances; "
            f"Morris {self.morris_evals} + Sobol {self.sobol_evals} + swap "
            f"{self.swap_evals} + Shapley {self.shapley_evals} evals) "
            f"| {self.total_evals} model evals "
            f"| {self.total_hours:.2f} h @ {self.seconds_per_eval:.3g} s/eval "
            f"| {reserve:.2f} h incl. 30% reserve |"
        )


def _estimate_compute_cost(
    strata: Sequence[Stratum],
    *,
    n_trajectories: int,
    n_base: int,
    swap_points: int,
    seconds_per_eval: float,
    shapley: tuple[int, int, int] | None = None,  # (d, n_outer, n_inner) if enabled
    registered: bool,
) -> GsaComputeCost:
    if registered:
        _validate_strata(strata)
    else:
        _validate_nonconfirmatory_strata(strata)
    _pos_int("n_trajectories", n_trajectories)
    _power_of_two("n_base", n_base)
    _pos_int("swap_points", swap_points)
    if (
        isinstance(seconds_per_eval, bool)
        or not isinstance(seconds_per_eval, int | float)
        or not math.isfinite(seconds_per_eval)
        or seconds_per_eval <= 0
    ):
        raise ValueError("seconds_per_eval must be a finite number > 0 (measured, not guessed)")
    k = len(strata[0].nuisances)
    morris = len(strata) * n_trajectories * (k + 1)
    sobol = len(strata) * n_base * (k + 2)
    swap = swap_points * len(strata)
    shap = 0
    if shapley is not None:
        d, n_outer, n_inner = shapley
        shap = shapley_eval_count(d, n_outer, n_inner)
    total = morris + sobol + swap + shap
    return GsaComputeCost(
        n_strata=len(strata),
        n_nuisances=k,
        morris_evals=morris,
        sobol_evals=sobol,
        swap_evals=swap,
        shapley_evals=shap,
        total_evals=total,
        seconds_per_eval=seconds_per_eval,
        total_hours=total * seconds_per_eval / 3600.0,
    )


def estimate_compute_cost(
    strata: Sequence[Stratum],
    *,
    n_trajectories: int,
    n_base: int,
    swap_points: int,
    seconds_per_eval: float,
    shapley: tuple[int, int, int] | None = None,
) -> GsaComputeCost:
    """G0 cost for the full canonical registered design.

    The same sealed ``design_sha256`` used by registered execution is checked
    here; runtime builder substitution cannot reprice a caller subset as the
    registered G0 design.
    """
    registration = load_energy_analysis_registration()
    registration.verify_seal()
    if strata_design_sha256(strata) != registration.design_sha256:
        raise ValueError(
            "cost design must equal the full canonical development design "
            "registered by design_sha256"
        )
    return _estimate_compute_cost(
        strata,
        n_trajectories=n_trajectories,
        n_base=n_base,
        swap_points=swap_points,
        seconds_per_eval=seconds_per_eval,
        shapley=shapley,
        registered=True,
    )


def nonconfirmatory_estimate_compute_cost(
    strata: Sequence[Stratum],
    *,
    n_trajectories: int,
    n_base: int,
    swap_points: int,
    seconds_per_eval: float,
    shapley: tuple[int, int, int] | None = None,
) -> GsaComputeCost:
    """Analytic/test-only cost harness for explicitly unregistered strata."""
    return _estimate_compute_cost(
        strata,
        n_trajectories=n_trajectories,
        n_base=n_base,
        swap_points=swap_points,
        seconds_per_eval=seconds_per_eval,
        shapley=shapley,
        registered=False,
    )


# ------------------------------------------------------------------ envelope verification/archive
def _result_type_name(result: _GsaResult) -> str:
    exact_names = {
        MorrisResult: "MorrisResult",
        SobolResult: "SobolResult",
        SwapResult: "SwapResult",
        ShapleyResult: "ShapleyResult",
    }
    result_type = type(result)
    if result_type not in exact_names:
        raise TypeError(
            "analysis envelope verification requires exact production result types; "
            "subclasses and foreign result objects are rejected"
        )
    return exact_names[result_type]


def _expected_result_ids(
    analysis: str,
    bundle_manifests: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    if analysis == _ANALYSIS_STRATIFIED:
        return tuple(
            result_id
            for bundle_id, _ in bundle_manifests
            for result_id in (f"{bundle_id}/morris", f"{bundle_id}/sobol")
        )
    if analysis == _ANALYSIS_SWAP:
        return ("between-stratum-swap",)
    if analysis == _ANALYSIS_SHAPLEY:
        return ("registered-shapley-effects",)
    raise ValueError("unknown production analysis identity")


def _complete_analysis_run(
    *,
    analysis: str,
    registration: EnergyAnalysisRegistration,
    model_registration: ModelRegistration,
    source_closure_sha256: str,
    estimators: tuple[EstimatorRegistration, ...],
    results: Mapping[str, _GsaResult],
) -> AnalysisRun:
    """Build and immediately verify one internally consistent production-path run."""
    registration.verify_seal()
    if _source_closure_sha256(
        model_registration,
        _estimator_parameter_payloads(estimators),
    ) != source_closure_sha256:
        raise ValueError("analysis source closure drifted during execution")
    if any(
        estimator.source_closure_sha256 != source_closure_sha256
        for estimator in estimators
    ):
        raise ValueError("estimator identity is not bound to the analysis source closure")
    bundle_manifests = registration.bundle_manifests("development")
    expected_ids = _expected_result_ids(analysis, bundle_manifests)
    if set(results) != set(expected_ids):
        raise ValueError(
            "production runner did not produce its exact complete result set"
        )
    ordered_results = {result_id: results[result_id] for result_id in expected_ids}
    result_manifest = tuple(
        ResultBinding(
            result_id=result_id,
            result_type=_result_type_name(result),
            payload_sha256=canonical_json_sha256(result.payload()),
        )
        for result_id, result in ordered_results.items()
    )
    envelope_fields = {
        "analysis": analysis,
        "design_sha256": registration.design_sha256,
        "registration_payload_sha256": registration.split_payload_sha256,
        "bundle_manifests": bundle_manifests,
        "model_registration": model_registration,
        "source_closure_sha256": source_closure_sha256,
        "producer_identity": {
            "entrypoint": _PRODUCTION_ENTRYPOINTS[analysis],
            "source_closure_sha256": source_closure_sha256,
        },
        "execution_classification": _execution_classification(),
        "estimators": estimators,
        "result_manifest": result_manifest,
    }
    unsigned = {
        "schema": _ENVELOPE_SCHEMA,
        "analysis": analysis,
        "design_sha256": registration.design_sha256,
        "registration_payload_sha256": registration.split_payload_sha256,
        "bundles": [
            {"bundle_id": bundle_id, "manifest_sha256": manifest}
            for bundle_id, manifest in bundle_manifests
        ],
        "model_registration": model_registration.artifact(),
        "source_closure_sha256": source_closure_sha256,
        "producer_identity": dict(envelope_fields["producer_identity"]),
        "execution_classification": envelope_fields["execution_classification"],
        "estimators": [estimator.payload() for estimator in estimators],
        "results": [binding.payload() for binding in result_manifest],
    }
    envelope = AnalysisEnvelope(
        **envelope_fields,
        payload_sha256=canonical_json_sha256(unsigned),
    )
    run = AnalysisRun(envelope=envelope, results=ordered_results)
    verify_analysis_envelope(run.envelope, run.results)
    return run


def _registered_nuisance_names(
    registration: EnergyAnalysisRegistration,
) -> dict[str, tuple[str, ...]]:
    return {
        stratum["stratum_id"]: tuple(
            nuisance["name"] for nuisance in stratum["nuisances"]
        )
        for stratum in registration.strata_payloads()
    }


def _estimator_map(
    envelope: AnalysisEnvelope,
) -> dict[str, EstimatorRegistration]:
    if any(type(estimator) is not EstimatorRegistration for estimator in envelope.estimators):
        raise TypeError("analysis envelope estimator registrations must use exact types")
    estimator_map = {
        estimator.estimator: estimator for estimator in envelope.estimators
    }
    if len(estimator_map) != len(envelope.estimators):
        raise ValueError("analysis envelope estimator identities must be unique")
    expected = {
        _ANALYSIS_STRATIFIED: {"morris", "sobol"},
        _ANALYSIS_SWAP: {"between-stratum-swap"},
        _ANALYSIS_SHAPLEY: {"shapley"},
    }[envelope.analysis]
    if set(estimator_map) != expected:
        raise ValueError(
            "analysis envelope estimator set is incomplete or contains extras"
        )
    return estimator_map


def _require_exact_parameters(
    estimator: EstimatorRegistration,
    names: set[str],
) -> Mapping[str, object]:
    if set(estimator.parameters) != names:
        raise ValueError(
            f"{estimator.estimator} envelope parameters must be exactly {sorted(names)}"
        )
    return estimator.parameters


def _verify_stratified_results(
    results: Mapping[str, _GsaResult],
    estimator_map: Mapping[str, EstimatorRegistration],
    nuisance_names: Mapping[str, tuple[str, ...]],
) -> None:
    morris_spec = estimator_map["morris"]
    morris_parameters = _require_exact_parameters(
        morris_spec, {"n_trajectories", "num_levels"}
    )
    n_trajectories = _pos_int(
        "envelope n_trajectories", morris_parameters["n_trajectories"]
    )
    num_levels = _pos_int(
        "envelope Morris num_levels",
        morris_parameters["num_levels"],
        minimum=2,
    )
    if num_levels != 4:
        raise ValueError("production Morris num_levels must equal the registered value 4")

    sobol_spec = estimator_map["sobol"]
    sobol_parameters = _require_exact_parameters(
        sobol_spec, {"n_base", "calc_second_order"}
    )
    n_base = _power_of_two("envelope n_base", sobol_parameters["n_base"])
    if sobol_parameters["calc_second_order"] is not False:
        raise ValueError("production Sobol calc_second_order must be false")

    for stratum_id, names in nuisance_names.items():
        morris = results[f"{stratum_id}/morris"]
        sobol = results[f"{stratum_id}/sobol"]
        if type(morris) is not MorrisResult or type(sobol) is not SobolResult:
            raise TypeError(
                "stratified envelope requires exact MorrisResult and SobolResult types"
            )
        if morris.stratum_id != stratum_id or sobol.stratum_id != stratum_id:
            raise ValueError("result stratum identity differs from its manifest key")
        if morris.seed != morris_spec.seed or sobol.seed != sobol_spec.seed:
            raise ValueError("result seed differs from its registered estimator seed")
        if set(morris.mu_star) != set(names) or set(morris.sigma) != set(names):
            raise ValueError("Morris result does not cover the registered nuisance set")
        if (
            set(sobol.first_order) != set(names)
            or set(sobol.total_order) != set(names)
        ):
            raise ValueError("Sobol result does not cover the registered nuisance set")
        if morris.n_evaluations != n_trajectories * (len(names) + 1):
            raise ValueError("Morris evaluation count differs from envelope parameters")
        if sobol.n_evaluations != n_base * (len(names) + 2):
            raise ValueError("Sobol evaluation count differs from envelope parameters")


def _verify_swap_result(
    result: _GsaResult,
    estimator: EstimatorRegistration,
    registered_ids: tuple[str, ...],
) -> None:
    if type(result) is not SwapResult:
        raise TypeError("swap envelope requires the exact SwapResult type")
    parameters = _require_exact_parameters(
        estimator, {"n_points", "rank_matching"}
    )
    n_points = _pos_int("envelope n_points", parameters["n_points"])
    if parameters["rank_matching"] != "common-unit-quantiles":
        raise ValueError("swap envelope has an unregistered rank-matching method")
    if result.seed != estimator.seed:
        raise ValueError("swap result seed differs from its registered estimator seed")
    if set(result.per_stratum_outputs) != set(registered_ids):
        raise ValueError("swap outputs do not cover the complete registered strata")
    if set(result.per_stratum_mean) != set(registered_ids):
        raise ValueError("swap means do not cover the complete registered strata")
    for stratum_id in registered_ids:
        values = result.per_stratum_outputs[stratum_id]
        if len(values) != n_points:
            raise ValueError("swap output count differs from envelope n_points")
        expected_mean = math.fsum(values) / n_points
        if result.per_stratum_mean[stratum_id] != expected_mean:
            raise ValueError("swap mean does not match its archived outputs")
    expected_ranking = tuple(
        sorted(
            registered_ids,
            key=lambda stratum_id: (
                result.per_stratum_mean[stratum_id],
                stratum_id,
            ),
        )
    )
    means = [result.per_stratum_mean[stratum_id] for stratum_id in expected_ranking]
    expected_gap = means[-1] - means[0]
    expected_sign = all(value > 0 for value in means) or all(
        value < 0 for value in means
    )
    if result.ranking != expected_ranking:
        raise ValueError("swap ranking does not match the archived means")
    if result.max_pairwise_mean_gap != expected_gap:
        raise ValueError("swap pairwise gap does not match the archived means")
    if result.sign_consistent is not expected_sign:
        raise ValueError("swap sign-consistency verdict does not match the archived means")
    if result.n_evaluations != n_points * len(registered_ids):
        raise ValueError("swap evaluation count differs from envelope parameters")


def _verify_shapley_result(
    result: _GsaResult,
    estimator: EstimatorRegistration,
) -> None:
    if type(result) is not ShapleyResult:
        raise TypeError("Shapley envelope requires the exact ShapleyResult type")
    parameters = _require_exact_parameters(
        estimator,
        {
            "names",
            "n_outer",
            "n_inner",
            "marginal_sampler",
            "conditional_sampler",
        },
    )
    names_raw = parameters["names"]
    if (
        not isinstance(names_raw, tuple)
        or not names_raw
        or any(not isinstance(name, str) or not name for name in names_raw)
        or len(names_raw) != len(set(names_raw))
    ):
        raise ValueError("Shapley envelope names must be unique non-empty strings")
    n_outer, n_inner = _shapley_loops(
        parameters["n_outer"], parameters["n_inner"]
    )
    for sampler_name in ("marginal_sampler", "conditional_sampler"):
        identity = parameters[sampler_name]
        try:
            _model_registration_from_payload(identity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Shapley {sampler_name} content identity is malformed"
            ) from exc
    if result.seed != estimator.seed:
        raise ValueError("Shapley result seed differs from its registered estimator seed")
    if set(result.effects) != set(names_raw):
        raise ValueError("Shapley result does not cover the registered input names")
    if result.n_evaluations != shapley_eval_count(
        len(names_raw), n_outer, n_inner
    ):
        raise ValueError("Shapley evaluation count differs from envelope parameters")


def verify_analysis_envelope(
    envelope: AnalysisEnvelope,
    results: Mapping[str, _GsaResult],
) -> bool:
    """Verify complete archive-envelope internal consistency.

    This deliberately is **not** a production-eligibility verdict. It checks the
    canonical hashes, content identities, repository registration, exact bundle and
    result completeness, exact result types, and estimator parameters/counts.
    :func:`confirmatory_status` is the separate receipt-bound eligibility boundary.
    """
    if type(envelope) is not AnalysisEnvelope:
        raise TypeError("verification requires the exact AnalysisEnvelope type")
    if not isinstance(results, Mapping):
        raise TypeError("analysis results must be a mapping")

    registration = load_energy_analysis_registration()
    registration.verify_seal()
    if envelope.design_sha256 != registration.design_sha256:
        raise ValueError(
            "analysis envelope design_sha256 differs from the registered design"
        )
    if envelope.registration_payload_sha256 != registration.split_payload_sha256:
        raise ValueError(
            "analysis envelope registration digest differs from the registered artifact"
        )
    registered_bundles = registration.bundle_manifests("development")
    if envelope.bundle_manifests != registered_bundles:
        raise ValueError(
            "analysis envelope bundle identities differ from the registered complete set"
        )
    if canonical_json_sha256(envelope.payload()) != envelope.payload_sha256:
        raise ValueError("analysis envelope payload_sha256 mismatch")
    if (
        canonical_json_sha256(envelope.model_registration.payload())
        != envelope.model_registration.registration_sha256
    ):
        raise ValueError("analysis envelope model registration hash mismatch")
    if _source_closure_sha256(
        envelope.model_registration,
        _estimator_parameter_payloads(envelope.estimators),
    ) != (
        envelope.source_closure_sha256
    ):
        raise ValueError("analysis envelope source closure differs from current content")

    expected_ids = _expected_result_ids(envelope.analysis, registered_bundles)
    if set(results) != set(expected_ids) or len(results) != len(expected_ids):
        raise ValueError(
            "analysis result set is incomplete or contains extras; "
            f"expected {list(expected_ids)}, got {sorted(results)}"
        )
    if any(type(binding) is not ResultBinding for binding in envelope.result_manifest):
        raise TypeError("analysis result manifest entries must use exact types")
    manifest_ids = tuple(binding.result_id for binding in envelope.result_manifest)
    if manifest_ids != expected_ids:
        raise ValueError(
            "analysis envelope result manifest is incomplete, extra, or out of canonical order"
        )
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("analysis envelope result ids must be unique")

    for binding in envelope.result_manifest:
        result = results[binding.result_id]
        result_type = _result_type_name(result)
        if result.result_id() != binding.result_id:
            raise ValueError("result payload identity differs from its manifest key")
        if result_type != binding.result_type:
            raise ValueError("result exact type differs from its envelope binding")
        if canonical_json_sha256(result.payload()) != binding.payload_sha256:
            raise ValueError(
                f"{binding.result_id}: result payload hash differs from the envelope"
            )

    estimator_map = _estimator_map(envelope)
    if any(
        estimator.source_closure_sha256 != envelope.source_closure_sha256
        for estimator in estimator_map.values()
    ):
        raise ValueError("estimator identity differs from the envelope source closure")
    nuisance_names = _registered_nuisance_names(registration)
    if envelope.analysis == _ANALYSIS_STRATIFIED:
        _verify_stratified_results(results, estimator_map, nuisance_names)
    elif envelope.analysis == _ANALYSIS_SWAP:
        _verify_swap_result(
            results["between-stratum-swap"],
            estimator_map["between-stratum-swap"],
            tuple(nuisance_names),
        )
    elif envelope.analysis == _ANALYSIS_SHAPLEY:
        _verify_shapley_result(
            results["registered-shapley-effects"],
            estimator_map["shapley"],
        )
    else:  # pragma: no cover - AnalysisEnvelope rejects this at construction
        raise ValueError("unknown production analysis identity")
    return True


def _json_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _json_array(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _parse_bundle_manifests(value: object) -> tuple[tuple[str, str], ...]:
    records = _json_array("envelope.bundles", value)
    bundles: list[tuple[str, str]] = []
    for index, raw in enumerate(records):
        record = _json_mapping(f"envelope.bundles[{index}]", raw)
        if set(record) != {"bundle_id", "manifest_sha256"}:
            raise ValueError("bundle bindings must contain only id and manifest hash")
        bundle_id = record["bundle_id"]
        manifest = record["manifest_sha256"]
        if not isinstance(bundle_id, str) or not bundle_id or not _is_sha256(manifest):
            raise ValueError("bundle binding contains an invalid id or manifest hash")
        bundles.append((bundle_id, manifest))
    if not bundles or len({bundle_id for bundle_id, _ in bundles}) != len(bundles):
        raise ValueError("envelope bundles must be unique and non-empty")
    return tuple(bundles)


def _parse_estimator(raw: object) -> EstimatorRegistration:
    record = _json_mapping("envelope estimator", raw)
    if set(record) != {
        "estimator",
        "implementation",
        "source_closure_sha256",
        "seed",
        "parameters",
    }:
        raise ValueError("estimator registration has unexpected fields")
    parameters = _json_mapping("estimator.parameters", record["parameters"])
    return EstimatorRegistration(
        estimator=record["estimator"],
        implementation=record["implementation"],
        source_closure_sha256=record["source_closure_sha256"],
        seed=record["seed"],
        parameters=parameters,
    )


def _parse_result_binding(raw: object) -> ResultBinding:
    record = _json_mapping("envelope result binding", raw)
    if set(record) != {"result_id", "result_type", "payload_sha256"}:
        raise ValueError("result binding has unexpected fields")
    return ResultBinding(
        result_id=record["result_id"],
        result_type=record["result_type"],
        payload_sha256=record["payload_sha256"],
    )


def _model_registration_from_payload(raw: object) -> ModelRegistration:
    record = _json_mapping("model registration", raw)
    expected = {
        "schema",
        "kind",
        "module",
        "qualname",
        "source_path",
        "callable_source_sha256",
        "source_file_sha256",
        "parameters",
        "policy_path",
        "policy_sha256",
        "bundle_path",
        "bundle_sha256",
        "registration_sha256",
    }
    if set(record) != expected or record.get("schema") != _MODEL_REGISTRATION_SCHEMA:
        raise ValueError("model registration has an unexpected schema or fields")
    parameters = _json_mapping("model registration parameters", record["parameters"])
    return ModelRegistration(
        kind=record["kind"],
        module=record["module"],
        qualname=record["qualname"],
        source_path=record["source_path"],
        callable_source_sha256=record["callable_source_sha256"],
        source_file_sha256=record["source_file_sha256"],
        parameters=parameters,
        policy_path=record["policy_path"],
        policy_sha256=record["policy_sha256"],
        bundle_path=record["bundle_path"],
        bundle_sha256=record["bundle_sha256"],
        registration_sha256=record["registration_sha256"],
    )


def analysis_envelope_from_payload(raw: object) -> AnalysisEnvelope:
    """Strictly reconstruct an envelope from a JSON-decoded object."""
    record = _json_mapping("analysis envelope", raw)
    expected_fields = {
        "schema",
        "analysis",
        "design_sha256",
        "registration_payload_sha256",
        "bundles",
        "model_registration",
        "source_closure_sha256",
        "producer_identity",
        "execution_classification",
        "estimators",
        "results",
        "payload_sha256",
    }
    if set(record) != expected_fields or record.get("schema") != _ENVELOPE_SCHEMA:
        raise ValueError("analysis envelope has an unexpected schema or fields")
    producer_identity = _json_mapping(
        "envelope.producer_identity", record["producer_identity"]
    )
    return AnalysisEnvelope(
        analysis=record["analysis"],
        design_sha256=record["design_sha256"],
        registration_payload_sha256=record["registration_payload_sha256"],
        bundle_manifests=_parse_bundle_manifests(record["bundles"]),
        model_registration=_model_registration_from_payload(
            record["model_registration"]
        ),
        source_closure_sha256=record["source_closure_sha256"],
        producer_identity=producer_identity,
        execution_classification=record["execution_classification"],
        estimators=tuple(
            _parse_estimator(item)
            for item in _json_array("envelope.estimators", record["estimators"])
        ),
        result_manifest=tuple(
            _parse_result_binding(item)
            for item in _json_array("envelope.results", record["results"])
        ),
        payload_sha256=record["payload_sha256"],
    )


def _finite_result_mapping(name: str, raw: object) -> dict[str, float]:
    mapping = _json_mapping(name, raw)
    output: dict[str, float] = {}
    for key, value in mapping.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must map non-empty names to finite numbers")
        output[key] = float(value)
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _result_from_payload(raw: object) -> _GsaResult:
    record = _json_mapping("analysis result", raw)
    if set(record) != {"schema", "estimator", "seed", "statistics"}:
        raise ValueError("analysis result has unexpected fields")
    if record["schema"] != _RESULT_SCHEMA:
        raise ValueError("analysis result has an unexpected schema")
    estimator = record["estimator"]
    seed = record["seed"]
    statistics = _json_mapping("analysis result statistics", record["statistics"])
    if estimator == "morris":
        expected = {"stratum_id", "mu_star", "sigma", "n_evaluations"}
        if set(statistics) != expected:
            raise ValueError("Morris result has unexpected statistic fields")
        return MorrisResult(
            stratum_id=statistics["stratum_id"],
            mu_star=_finite_result_mapping("Morris mu_star", statistics["mu_star"]),
            sigma=_finite_result_mapping("Morris sigma", statistics["sigma"]),
            n_evaluations=statistics["n_evaluations"],
            seed=seed,
        )
    if estimator == "sobol":
        expected = {"stratum_id", "first_order", "total_order", "n_evaluations"}
        if set(statistics) != expected:
            raise ValueError("Sobol result has unexpected statistic fields")
        return SobolResult(
            stratum_id=statistics["stratum_id"],
            first_order=_finite_result_mapping(
                "Sobol first_order", statistics["first_order"]
            ),
            total_order=_finite_result_mapping(
                "Sobol total_order", statistics["total_order"]
            ),
            n_evaluations=statistics["n_evaluations"],
            seed=seed,
        )
    if estimator == "between-stratum-swap":
        expected = {
            "per_stratum_outputs",
            "per_stratum_mean",
            "ranking",
            "max_pairwise_mean_gap",
            "sign_consistent",
            "n_evaluations",
        }
        if set(statistics) != expected:
            raise ValueError("swap result has unexpected statistic fields")
        outputs_raw = _json_mapping(
            "swap per_stratum_outputs", statistics["per_stratum_outputs"]
        )
        outputs = {
            stratum_id: tuple(
                _finite_result_sequence(
                    f"swap outputs for {stratum_id}", values
                )
            )
            for stratum_id, values in outputs_raw.items()
        }
        ranking_raw = _json_array("swap ranking", statistics["ranking"])
        if any(not isinstance(item, str) or not item for item in ranking_raw):
            raise ValueError("swap ranking must contain non-empty strings")
        return SwapResult(
            per_stratum_outputs=outputs,
            per_stratum_mean=_finite_result_mapping(
                "swap per_stratum_mean", statistics["per_stratum_mean"]
            ),
            ranking=tuple(ranking_raw),
            max_pairwise_mean_gap=_finite_result_number(
                "swap max_pairwise_mean_gap",
                statistics["max_pairwise_mean_gap"],
            ),
            sign_consistent=statistics["sign_consistent"],
            n_evaluations=statistics["n_evaluations"],
            seed=seed,
        )
    if estimator == "shapley":
        expected = {"effects", "n_evaluations"}
        if set(statistics) != expected:
            raise ValueError("Shapley result has unexpected statistic fields")
        return ShapleyResult(
            effects=_finite_result_mapping("Shapley effects", statistics["effects"]),
            n_evaluations=statistics["n_evaluations"],
            seed=seed,
        )
    raise ValueError("analysis result has an unregistered estimator identity")


def _finite_result_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _finite_result_sequence(name: str, raw: object) -> tuple[float, ...]:
    values = _json_array(name, raw)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(_finite_result_number(name, value) for value in values)


def analysis_run_to_bytes(
    envelope: AnalysisEnvelope,
    results: Mapping[str, _GsaResult],
) -> bytes:
    """Canonical internally verified archive bytes.

    The archive's public payload hash detects corruption; it does not grant
    production eligibility.
    """
    verify_analysis_envelope(envelope, results)
    ordered_results = [
        results[binding.result_id].payload()
        for binding in envelope.result_manifest
    ]
    payload = {
        "schema": _ARCHIVE_SCHEMA,
        "envelope": envelope.artifact(),
        "results": ordered_results,
    }
    artifact = {
        **payload,
        "payload_sha256": canonical_json_sha256(payload),
    }
    return (
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def analysis_run_from_bytes(data: bytes) -> AnalysisRun:
    """Load and independently re-verify one serialized run's internal consistency."""
    if type(data) is not bytes:
        raise TypeError("analysis archive input must be bytes")
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid analysis archive JSON: {exc}") from exc
    artifact = _json_mapping("analysis archive", raw)
    if set(artifact) != {"schema", "envelope", "results", "payload_sha256"}:
        raise ValueError("analysis archive has unexpected fields")
    if artifact["schema"] != _ARCHIVE_SCHEMA:
        raise ValueError("analysis archive has an unexpected schema")
    payload = {
        "schema": artifact["schema"],
        "envelope": artifact["envelope"],
        "results": artifact["results"],
    }
    if canonical_json_sha256(payload) != artifact["payload_sha256"]:
        raise ValueError("analysis archive payload_sha256 mismatch")
    envelope = analysis_envelope_from_payload(artifact["envelope"])
    result_items = _json_array("analysis archive results", artifact["results"])
    results: dict[str, _GsaResult] = {}
    for raw_result in result_items:
        result = _result_from_payload(raw_result)
        result_id = result.result_id()
        if result_id in results:
            raise ValueError("analysis archive contains duplicate result identities")
        results[result_id] = result
    run = AnalysisRun(envelope=envelope, results=results)
    verify_analysis_envelope(run.envelope, run.results)
    return run


def archive_analysis(
    path: str | os.PathLike[str],
    envelope: AnalysisEnvelope,
    results: Mapping[str, _GsaResult],
) -> Path:
    """E4 internal-consistency archive entry point.

    The complete envelope/result set is verified before any filesystem mutation.
    The canonical artifact is then published atomically without overwriting an
    existing archive.
    """
    archive_path = Path(path)
    data = analysis_run_to_bytes(envelope, results)
    parent = archive_path.parent
    if not parent.is_dir():
        raise ValueError(f"analysis archive parent does not exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, archive_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"analysis archive already exists: {archive_path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def load_analysis_archive(path: str | os.PathLike[str]) -> AnalysisRun:
    """Read and independently verify an E4 archive's internal consistency."""
    archive_path = Path(path)
    try:
        data = archive_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read analysis archive {archive_path}: {exc}") from exc
    return analysis_run_from_bytes(data)


# ------------------------------------------------------------------ postflight eligibility receipt
@dataclass(frozen=True)
class AnalysisPostflightReceipt:
    """Separate durable authority binding committed archive and code state."""

    status: str
    archive_repository_path: str
    archive_payload_sha256: str
    envelope_sha256: str
    design_sha256: str
    registration_payload_sha256: str
    repository_commit: str
    source_closure_sha256: str
    model_registration_sha256: str
    issued_at_utc: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.status != "eligible":
            raise ValueError("analysis postflight receipt status must be eligible")
        relative = Path(self.archive_repository_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not self.archive_repository_path
        ):
            raise ValueError("receipt archive path must be repository-relative")
        for name in (
            "archive_payload_sha256",
            "envelope_sha256",
            "design_sha256",
            "registration_payload_sha256",
            "source_closure_sha256",
            "model_registration_sha256",
            "receipt_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"receipt {name} must be a SHA-256 digest")
        if (
            len(self.repository_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.repository_commit)
        ):
            raise ValueError("receipt repository_commit must be a full SHA-1 object id")
        try:
            timestamp = datetime.fromisoformat(self.issued_at_utc)
        except ValueError as exc:
            raise ValueError("receipt issued_at_utc must be ISO-8601") from exc
        if timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError("receipt issued_at_utc must use UTC")
        if canonical_json_sha256(self.payload()) != self.receipt_sha256:
            raise ValueError("analysis postflight receipt hash mismatch")

    def payload(self) -> dict[str, str]:
        return {
            "schema": _POSTFLIGHT_SCHEMA,
            "status": self.status,
            "archive_repository_path": self.archive_repository_path,
            "archive_payload_sha256": self.archive_payload_sha256,
            "envelope_sha256": self.envelope_sha256,
            "design_sha256": self.design_sha256,
            "registration_payload_sha256": self.registration_payload_sha256,
            "repository_commit": self.repository_commit,
            "source_closure_sha256": self.source_closure_sha256,
            "model_registration_sha256": self.model_registration_sha256,
            "issued_at_utc": self.issued_at_utc,
        }

    def artifact(self) -> dict[str, str]:
        return {**self.payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class _RepositoryState:
    head_commit: str
    dirty_entries: tuple[str, ...]


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _repository_state() -> _RepositoryState:
    status = _git_text("status", "--porcelain=v1", "--untracked-files=all")
    return _RepositoryState(
        head_commit=_git_text("rev-parse", "HEAD"),
        dirty_entries=tuple(line for line in status.splitlines() if line.strip()),
    )


def _require_clean_repository(state: _RepositoryState) -> None:
    if (
        len(state.head_commit) != 40
        or any(character not in "0123456789abcdef" for character in state.head_commit)
    ):
        raise RuntimeError("analysis postflight requires a full pinned repository commit")
    if state.dirty_entries:
        raise RuntimeError(
            "analysis postflight refuses a dirty repository; run remains development_only"
        )


def _committed_blob(relative_path: str, commit: str) -> bytes:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("committed artifact path must be repository-relative")
    try:
        return _git_bytes("show", f"{commit}:{path.as_posix()}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"{path.as_posix()} is not an exact committed blob at {commit}"
        ) from exc


def _require_committed_bytes(
    relative_path: str,
    expected: bytes,
    commit: str,
) -> None:
    if _committed_blob(relative_path, commit) != expected:
        raise ValueError(
            f"{relative_path} differs from its exact blob at repository commit {commit}"
        )


def _repository_artifact_bytes(relative_path: str) -> bytes:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("executable artifact path must be repository-relative")
    candidate = _REPOSITORY_ROOT / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(_REPOSITORY_ROOT.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"executable artifact is not a repository file: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"executable artifact is not a file: {relative_path}")
    return resolved.read_bytes()


def _archive_repository_path(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            _REPOSITORY_ROOT.resolve()
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError(
            "receipt-bound analysis archive must be a file inside the repository"
        ) from exc


def _archive_payload_sha256(data: bytes) -> str:
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("analysis archive is not valid JSON") from exc
    artifact = _json_mapping("analysis archive", raw)
    digest = artifact.get("payload_sha256")
    if not _is_sha256(digest):
        raise ValueError("analysis archive payload_sha256 is missing or malformed")
    return digest


def _require_repository_bindings(
    *,
    archive_relative: str,
    archive_data: bytes,
    run: AnalysisRun,
    commit: str,
) -> None:
    _require_committed_bytes(archive_relative, archive_data, commit)
    registration_path = _REPOSITORY_ROOT / _REGISTRATION_RELATIVE_PATH
    registration_bytes = registration_path.read_bytes()
    _require_committed_bytes(
        _REGISTRATION_RELATIVE_PATH.as_posix(),
        registration_bytes,
        commit,
    )
    estimator_parameters = _estimator_parameter_payloads(
        run.envelope.estimators
    )
    for executable in _canonical_executable_registrations(
        run.envelope.model_registration,
        estimator_parameters,
    ):
        source_file_bytes = None
        for artifact in executable.artifacts:
            artifact_bytes = _repository_artifact_bytes(artifact.path)
            if hashlib.sha256(artifact_bytes).hexdigest() != artifact.sha256:
                raise ValueError(
                    f"{executable.location} {artifact.role} differs from its "
                    "envelope registration"
                )
            _require_committed_bytes(
                artifact.path,
                artifact_bytes,
                commit,
            )
            if artifact.role == "source_file":
                source_file_bytes = artifact_bytes
        if source_file_bytes is None:
            raise ValueError(f"{executable.location} source_file binding is missing")
        registration = executable.registration
        try:
            callable_source = _callable_source_span(
                source_file_bytes,
                module=registration.module,
                qualname=registration.qualname,
                source_path=registration.source_path,
            )
        except ValueError as exc:
            raise ValueError(
                f"{executable.location} callable cannot be located in source_file"
            ) from exc
        if (
            hashlib.sha256(callable_source).hexdigest()
            != registration.callable_source_sha256
        ):
            raise ValueError(
                f"{executable.location} callable_source differs from its "
                "envelope registration"
            )
    if _source_closure_sha256(
        run.envelope.model_registration,
        estimator_parameters,
    ) != (
        run.envelope.source_closure_sha256
    ):
        raise ValueError("current source closure differs from the analysis envelope")


def _receipt_from_bytes(data: bytes) -> AnalysisPostflightReceipt:
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("analysis postflight receipt is not valid JSON") from exc
    record = _json_mapping("analysis postflight receipt", raw)
    expected = {
        "schema",
        "status",
        "archive_repository_path",
        "archive_payload_sha256",
        "envelope_sha256",
        "design_sha256",
        "registration_payload_sha256",
        "repository_commit",
        "source_closure_sha256",
        "model_registration_sha256",
        "issued_at_utc",
        "receipt_sha256",
    }
    if set(record) != expected or record.get("schema") != _POSTFLIGHT_SCHEMA:
        raise ValueError("analysis postflight receipt has unexpected fields or schema")
    receipt = AnalysisPostflightReceipt(
        status=record["status"],
        archive_repository_path=record["archive_repository_path"],
        archive_payload_sha256=record["archive_payload_sha256"],
        envelope_sha256=record["envelope_sha256"],
        design_sha256=record["design_sha256"],
        registration_payload_sha256=record["registration_payload_sha256"],
        repository_commit=record["repository_commit"],
        source_closure_sha256=record["source_closure_sha256"],
        model_registration_sha256=record["model_registration_sha256"],
        issued_at_utc=record["issued_at_utc"],
        receipt_sha256=record["receipt_sha256"],
    )
    canonical = (
        json.dumps(
            receipt.artifact(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if data != canonical:
        raise ValueError("analysis postflight receipt is not canonical JSON")
    return receipt


def _publish_receipt(path: Path, receipt: AnalysisPostflightReceipt) -> None:
    encoded = (
        json.dumps(
            receipt.artifact(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError("existing analysis postflight receipt differs")
        return
    parent = path.parent
    if not parent.is_dir():
        raise ValueError(f"analysis receipt parent does not exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise RuntimeError(
                    "existing analysis postflight receipt differs"
                ) from None
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def analysis_postflight_receipt(
    archive: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
) -> AnalysisPostflightReceipt:
    """Issue the sole production-eligibility receipt.

    The archive, design registration, every executable registration's source and
    referenced content artifacts, and the complete energy source closure must match
    one clean committed repository state. Thus a hand-built archive can cross this
    boundary only by committing its bytes (and any forged code/registration) as
    durable reviewable evidence.
    """
    archive_path = Path(archive)
    receipt_output = Path(receipt_path)
    if receipt_output.exists():
        existing = _receipt_from_bytes(receipt_output.read_bytes())
        if confirmatory_status(archive_path, receipt_output):
            return existing
        raise RuntimeError(
            "existing analysis postflight receipt is not valid for this archive"
        )
    archive_data = archive_path.read_bytes()
    run = analysis_run_from_bytes(archive_data)
    if analysis_run_to_bytes(run.envelope, run.results) != archive_data:
        raise ValueError("analysis postflight requires exact canonical archive bytes")
    if run.execution_classification != _RECEIPT_PENDING:
        raise RuntimeError(
            "development_only analysis archives can never receive a postflight receipt"
        )
    state = _repository_state()
    _require_clean_repository(state)
    archive_relative = _archive_repository_path(archive_path)
    _require_repository_bindings(
        archive_relative=archive_relative,
        archive_data=archive_data,
        run=run,
        commit=state.head_commit,
    )
    fields = {
        "status": "eligible",
        "archive_repository_path": archive_relative,
        "archive_payload_sha256": _archive_payload_sha256(archive_data),
        "envelope_sha256": run.envelope.payload_sha256,
        "design_sha256": run.envelope.design_sha256,
        "registration_payload_sha256": run.envelope.registration_payload_sha256,
        "repository_commit": state.head_commit,
        "source_closure_sha256": run.envelope.source_closure_sha256,
        "model_registration_sha256": (
            run.envelope.model_registration.registration_sha256
        ),
        "issued_at_utc": _receipt_timestamp_utc(),
    }
    receipt = AnalysisPostflightReceipt(
        **fields,
        receipt_sha256=canonical_json_sha256(
            {"schema": _POSTFLIGHT_SCHEMA, **fields}
        ),
    )
    postflight_state = _repository_state()
    _require_clean_repository(postflight_state)
    if postflight_state != state:
        raise RuntimeError("repository state drifted during analysis postflight")
    _require_repository_bindings(
        archive_relative=archive_relative,
        archive_data=archive_data,
        run=run,
        commit=postflight_state.head_commit,
    )
    _publish_receipt(receipt_output, receipt)
    return receipt


def confirmatory_status(
    archive: str | os.PathLike[str],
    receipt: str | os.PathLike[str] | None,
) -> bool:
    """Return eligibility only for an exact receipt-bound committed archive.

    Missing, malformed, stale, copied, development-only, or closure-mismatched
    evidence returns ``False``. The checker never upgrades envelope verification
    alone into eligibility.
    """
    if receipt is None:
        return False
    try:
        archive_path = Path(archive)
        archive_data = archive_path.read_bytes()
        run = analysis_run_from_bytes(archive_data)
        if analysis_run_to_bytes(run.envelope, run.results) != archive_data:
            return False
        if run.execution_classification != _RECEIPT_PENDING:
            return False
        parsed = _receipt_from_bytes(Path(receipt).read_bytes())
        archive_relative = _archive_repository_path(archive_path)
        if parsed.archive_repository_path != archive_relative:
            return False
        if {
            "archive_payload_sha256": _archive_payload_sha256(archive_data),
            "envelope_sha256": run.envelope.payload_sha256,
            "design_sha256": run.envelope.design_sha256,
            "registration_payload_sha256": run.envelope.registration_payload_sha256,
            "source_closure_sha256": run.envelope.source_closure_sha256,
            "model_registration_sha256": (
                run.envelope.model_registration.registration_sha256
            ),
        } != {
            "archive_payload_sha256": parsed.archive_payload_sha256,
            "envelope_sha256": parsed.envelope_sha256,
            "design_sha256": parsed.design_sha256,
            "registration_payload_sha256": parsed.registration_payload_sha256,
            "source_closure_sha256": parsed.source_closure_sha256,
            "model_registration_sha256": parsed.model_registration_sha256,
        }:
            return False
        state = _repository_state()
        if state.head_commit != parsed.repository_commit:
            return False
        _require_repository_bindings(
            archive_relative=archive_relative,
            archive_data=archive_data,
            run=run,
            commit=parsed.repository_commit,
        )
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ):
        return False
    return True
