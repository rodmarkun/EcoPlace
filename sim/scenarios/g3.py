"""P1-frozen G3-cal face-validity harness (TODO 4.5).

G3 supports workload-model credibility only. It is neither DES verification nor energy
validation. Missing publication/trace payloads yield ``UNAVAILABLE`` and can never pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sim.engine import EpisodeResult, run_episode
from sim.scenarios.alibaba import MetricCorpus
from sim.scenarios.artifacts import WorkloadModelClosure
from sim.scenarios.sources import (
    CALIBRATION,
    FACE_VALIDITY,
    SourceRegistry,
    clean_checkout_commit,
    require_frozen_clean_checkout,
)
from sim.scheduler import CallableScorer
from sim.spec import EpisodeConfig, config_sha256

_CURVE_TAG = "published-curve-v1:"
_PROTOCOL_TAG = "g3-protocol-v1:"
_RESULT_TAG = "g3-result-v1:"
SUPPORTS_ONLY = "workload-model credibility only; never energy claims"
_CURVE_AUTHORITY = object()
_RESIDUAL_AUTHORITY = object()
_RESIDUAL_TAG = "g3-source-bound-residual-v1:"
_PREDICTION_TAG = "g3-frozen-prediction-v1:"
_PREDICTION_AUTHORITY = object()
_P1_RECEIPT_AUTHORITY = object()
_PROTOCOL_AUTHORITY = object()
_RESULT_AUTHORITY = object()
_P1_SPEC_TAG = "g3-p1-protocol-spec-v1:"
_G3_ACCESS_AUTHORITY = object()
P1_RECEIPT_PATH = Path(__file__).resolve().parent / "data" / "p1_g3_receipt.json"
PRODUCTION_P1_RECEIPT_SHA256: str | None = None
_PREDICTION_GENERATOR_VERSION = "phase4-runtime-derived-prediction-v1"
_PREDICTION_RUNNER_VERSION = "g3-reference-runner-v1"
_PREDICTION_SCORER_NAME = "py:g3-reference-least-allocated-v1"
_PREDICTION_LB_POLICY = "round_robin"
_PREDICTION_RUN_AUTHORITY = object()
_RT_CENSORING_TREATMENT = "substitute-half-detection-limit-v1"
_RT_QUANTILE_METHOD = "linear-type-7-v1"
_RT_ACTIVITY_RULE = "paired-same-cell-family-mcr-strictly-positive-v1"
_MIN_CURVE_COMPLETIONS_PER_POINT = 30
_MIN_RT_INTERVAL_CELLS = 20
_RT_CALL_POPULATION_RULE = (
    "abstract-child-invocations-as-provider-rpc; selected-service-never-root-v1"
)
_RT_SIM_WINDOW_RULE = (
    "service-call-start; exclude-first-and-final-60s; full-60s-cells-only; "
    "all-selected-calls-terminal-by-horizon; "
    "all-root-window-offers-completed-no-drop-timeout; "
    f"{_RT_CALL_POPULATION_RULE}; min-populated-instance-cells={_MIN_RT_INTERVAL_CELLS}"
)
_CURVE_SIM_RULE = (
    "configured-stationary-total-root-rps; completed-root-linear-type7-p95-ms; "
    "arrival-window-[60s,cutoff); cutoff-at-least-60s-before-horizon; "
    "all-window-offers-completed-no-drop-timeout; "
    f"min-completed-per-coordinate={_MIN_CURVE_COMPLETIONS_PER_POINT}"
)


def _canonical(value: object) -> str:
    def jsonable(item: object) -> object:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return {
                field.name: jsonable(getattr(item, field.name))
                for field in dataclasses.fields(item)
                if field.name != "_authority"
            }
        if isinstance(item, tuple | list):
            return [jsonable(value) for value in item]
        if isinstance(item, dict):
            return {key: jsonable(value) for key, value in item.items()}
        return item

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_hex(name: str, value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class PublishedCurve:
    schema: str
    source_id: str
    paper_sha256: str
    extraction_manifest_sha256: str
    doi_or_url: str
    page: str
    figure_or_table: str
    series_label: str
    x_unit: str
    y_unit: str
    points: tuple[tuple[float, float], ...]
    extraction_method: str
    extraction_tool_version: str
    extraction_code_sha256: str
    point_selection_rule: str
    configuration: str
    unknown_configuration: str
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("seal", "_authority")
        }

    def verify(self) -> None:
        if self._authority is not _CURVE_AUTHORITY:
            raise ValueError("PublishedCurve must be issued from exact publication bytes")
        _sha256_hex("paper_sha256", self.paper_sha256)
        _sha256_hex("extraction_manifest_sha256", self.extraction_manifest_sha256)
        _sha256_hex("extraction_code_sha256", self.extraction_code_sha256)
        for name in (
            "source_id",
            "doi_or_url",
            "page",
            "figure_or_table",
            "series_label",
            "x_unit",
            "y_unit",
            "extraction_method",
            "extraction_tool_version",
            "point_selection_rule",
            "configuration",
            "unknown_configuration",
        ):
            if not getattr(self, name):
                raise ValueError(f"PublishedCurve {name} must be non-empty")
        if self.point_selection_rule != "all-visible-points-in-selected-series":
            raise ValueError("PublishedCurve must retain all visible points in its series")
        if len(self.points) < 2:
            raise ValueError("PublishedCurve needs at least two observed points")
        xs = [x for x, _ in self.points]
        if xs != sorted(set(xs)):
            raise ValueError("PublishedCurve x coordinates must strictly increase")
        if any(not math.isfinite(x) or not math.isfinite(y) or y <= 0 for x, y in self.points):
            raise ValueError("PublishedCurve points must be finite with y > 0")
        expected = hashlib.sha256((_CURVE_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("PublishedCurve seal mismatch")


def make_published_curve(
    *, paper_content: bytes, extraction_manifest_content: bytes
) -> PublishedCurve:
    if not isinstance(paper_content, bytes) or not paper_content:
        raise ValueError("PublishedCurve requires non-empty exact publication bytes")
    if not isinstance(extraction_manifest_content, bytes) or not extraction_manifest_content:
        raise ValueError("PublishedCurve requires exact non-empty extraction-manifest bytes")
    try:
        manifest = json.loads(extraction_manifest_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("curve extraction manifest must be UTF-8 JSON") from exc
    required = {
        "schema",
        "source_id",
        "doi_or_url",
        "page",
        "figure_or_table",
        "series_label",
        "x_unit",
        "y_unit",
        "points",
        "extraction_method",
        "extraction_tool_version",
        "extraction_code_sha256",
        "point_selection_rule",
        "configuration",
        "unknown_configuration",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required
        or manifest["schema"] != "published-curve-extraction-v1"
        or _canonical(manifest).encode() != extraction_manifest_content
    ):
        raise ValueError("curve extraction manifest fields/schema/encoding are not canonical")
    payload = {
        "schema": "published-curve-v1",
        "paper_sha256": hashlib.sha256(paper_content).hexdigest(),
        "extraction_manifest_sha256": hashlib.sha256(
            extraction_manifest_content
        ).hexdigest(),
        **{key: value for key, value in manifest.items() if key != "schema"},
    }
    if "points" in payload:
        payload["points"] = tuple(tuple(point) for point in payload["points"])  # type: ignore[arg-type]
    seal = hashlib.sha256((_CURVE_TAG + _canonical(payload)).encode()).hexdigest()
    curve = PublishedCurve(**payload, seal=seal, _authority=_CURVE_AUTHORITY)  # type: ignore[arg-type]
    curve.verify()
    return curve


def published_curve_selector_sha256(
    *,
    source_id: str,
    page: str,
    figure_or_table: str,
    series_label: str,
    x_unit: str,
    y_unit: str,
    extraction_method: str,
    extraction_tool_version: str,
    extraction_code_sha256: str,
    configuration: str,
    unknown_configuration: str,
    point_selection_rule: str = "all-visible-points-in-selected-series",
) -> str:
    """Hash a P1-freezable selector without reading curve outcomes or coordinates."""
    payload = {
        "kind": "published-curve",
        "source_id": source_id,
        "page": page,
        "figure_or_table": figure_or_table,
        "series_label": series_label,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "extraction_method": extraction_method,
        "extraction_tool_version": extraction_tool_version,
        "extraction_code_sha256": extraction_code_sha256,
        "configuration": configuration,
        "unknown_configuration": unknown_configuration,
        "point_selection_rule": point_selection_rule,
        "simulator_prediction_rule": _CURVE_SIM_RULE,
    }
    if any(not value for value in payload.values()):
        raise ValueError("published-curve selector fields must be non-empty")
    return hashlib.sha256(("g3-selector-v1:" + _canonical(payload)).encode()).hexdigest()


def rt_shape_selector_sha256(*, source_id: str, service: str, metric: str) -> str:
    """Hash the fixed Alibaba instance×60s interval-mean RT estimand selector."""
    if metric.upper() != "PROVIDERRPC_RT":
        raise ValueError("G3 currently supports only the providerRPC RT call population")
    payload = {
        "kind": "rt-shape",
        "source_id": source_id,
        "service": service,
        "metric": metric.upper(),
        "estimand": "service-instance-by-60s-interval-mean-response-time",
        "left_censoring_treatment": _RT_CENSORING_TREATMENT,
        "quantile_method": _RT_QUANTILE_METHOD,
        "activity_rule": _RT_ACTIVITY_RULE,
        "simulator_measurement_rule": _RT_SIM_WINDOW_RULE,
    }
    if not source_id or not service or not metric:
        raise ValueError("RT selector fields must be non-empty")
    return hashlib.sha256(("g3-selector-v1:" + _canonical(payload)).encode()).hexdigest()


def aggregate_instance_interval_mean_rt(
    samples: tuple[tuple[float, str, float], ...], *, interval_s: float = 60.0
) -> tuple[float, ...]:
    """Aggregate simulator request RTs to Alibaba's instance×interval mean estimand.

    Each sample is ``(completion_time_s, service_instance_id, response_time_ms)``.
    Empty instance-interval cells are absent, matching the trace row representation.
    """
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("aggregation interval must be finite and > 0")
    grouped: dict[tuple[int, str], list[float]] = {}
    for at_s, instance, response_ms in samples:
        if (
            not math.isfinite(at_s)
            or at_s < 0
            or not instance
            or not math.isfinite(response_ms)
            or response_ms <= 0
        ):
            raise ValueError("simulator RT samples must have finite time/RT and an instance id")
        grouped.setdefault((math.floor(at_s / interval_s), instance), []).append(response_ms)
    if len(grouped) < 2:
        raise ValueError("RT shape needs at least two populated instance-interval cells")
    return tuple(
        math.fsum(grouped[key]) / len(grouped[key]) for key in sorted(grouped)
    )


@dataclass(frozen=True)
class G3PredictionArtifact:
    schema: str
    source_id: str
    kind: str
    selector_sha256: str
    estimand: str
    curve_points: tuple[tuple[float, float], ...]
    distribution_values_ms: tuple[float, ...]
    effective_sample_counts: tuple[int, ...]
    model_recipe_sha256s: tuple[str, ...]
    calibration_evidence_sha256s: tuple[str, ...]
    episode_config_sha256s: tuple[str, ...]
    episode_seeds: tuple[int, ...]
    source_registry_sha256: str
    episode_result_sha256s: tuple[str, ...]
    runner_version: str
    scorer_name: str
    proposer_name: str | None
    lb_policy: str
    simulation_output_sha256: str
    generator_version: str
    generator_code_sha256: str
    source_checkout_commit: str
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }

    def verify(self) -> None:
        if self._authority is not _PREDICTION_AUTHORITY:
            raise ValueError("G3 prediction must be issued from exact pre-access inputs")
        if self.schema != "g3-frozen-prediction-v1" or self.kind not in (
            "published-curve",
            "rt-shape",
        ):
            raise ValueError("unknown G3 prediction schema/kind")
        _sha256_hex("prediction selector", self.selector_sha256)
        if not self.model_recipe_sha256s or not self.episode_config_sha256s:
            raise ValueError("prediction must bind workload models and EpisodeConfigs")
        if not (
            len(self.model_recipe_sha256s)
            == len(self.episode_config_sha256s)
            == len(self.episode_seeds)
            == len(self.episode_result_sha256s)
        ):
            raise ValueError("prediction run-closure vectors must have equal lengths")
        for model_sha256 in self.model_recipe_sha256s:
            _sha256_hex("prediction model recipe", model_sha256)
        if not self.calibration_evidence_sha256s:
            raise ValueError("G3 prediction must bind calibration evidence artifacts")
        for evidence_sha256 in self.calibration_evidence_sha256s:
            _sha256_hex("prediction calibration evidence", evidence_sha256)
        for episode_sha256 in self.episode_config_sha256s:
            _sha256_hex("prediction EpisodeConfig", episode_sha256)
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.episode_seeds
        ):
            raise TypeError("prediction episode seeds must be explicit integers")
        if not self.effective_sample_counts or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.effective_sample_counts
        ):
            raise TypeError("prediction effective sample counts must be positive integers")
        _sha256_hex("prediction source registry", self.source_registry_sha256)
        if not self.episode_result_sha256s:
            raise ValueError("prediction must cite at least one exact episode result")
        for result_sha256 in self.episode_result_sha256s:
            _sha256_hex("prediction episode result", result_sha256)
        _sha256_hex("prediction simulation output", self.simulation_output_sha256)
        _sha256_hex("prediction generator code", self.generator_code_sha256)
        if len(self.source_checkout_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.source_checkout_commit
        ):
            raise ValueError("prediction source checkout must be a full SHA-1 object id")
        if not self.source_id or not self.estimand or not self.generator_version:
            raise ValueError("prediction source/estimand/generator version must be non-empty")
        if self.generator_version != _PREDICTION_GENERATOR_VERSION:
            raise ValueError(
                "prediction generator version is not the runtime-derived implementation"
            )
        if (
            self.runner_version != _PREDICTION_RUNNER_VERSION
            or self.scorer_name != _PREDICTION_SCORER_NAME
            or self.proposer_name is not None
            or self.lb_policy != _PREDICTION_LB_POLICY
        ):
            raise ValueError("prediction execution contract is not the fixed G3 reference runner")
        if self.generator_code_sha256 != _prediction_generator_code_sha256():
            raise ValueError("prediction generator code differs from the executing implementation")
        if self.kind == "published-curve":
            xs = [x for x, _ in self.curve_points]
            if (
                len(self.curve_points) < 2
                or xs != sorted(set(xs))
                or self.distribution_values_ms
                or len(self.effective_sample_counts) != len(self.curve_points)
                or any(
                    count < _MIN_CURVE_COMPLETIONS_PER_POINT
                    for count in self.effective_sample_counts
                )
                or any(
                    not math.isfinite(x) or not math.isfinite(y) or y <= 0
                    for x, y in self.curve_points
                )
            ):
                raise ValueError("curve prediction must be a finite positive ordered grid")
            if self.estimand != (
                "configured-total-root-offered-rps-to-"
                "completed-root-request-linear-type7-p95-latency-ms;"
                + _CURVE_SIM_RULE
            ):
                raise ValueError("curve prediction estimand is not simulator root p95 latency")
        elif (
            self.curve_points
            or len(self.distribution_values_ms) < _MIN_RT_INTERVAL_CELLS
            or self.effective_sample_counts != (len(self.distribution_values_ms),)
            or any(
                not math.isfinite(value) or value <= 0
                for value in self.distribution_values_ms
            )
            or self.estimand
            != "service-instance-by-60s-interval-mean-response-time;" + _RT_SIM_WINDOW_RULE
        ):
            raise ValueError("RT prediction must contain matching interval-mean values")
        if self.simulation_output_sha256 != _prediction_run_closure_sha256(self):
            raise ValueError("prediction output closure does not match its retained runs")
        expected = hashlib.sha256(
            (_PREDICTION_TAG + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("G3 prediction artifact digest mismatch")


@dataclass(frozen=True)
class G3PredictionRun:
    config: EpisodeConfig
    model: WorkloadModelClosure
    result: EpisodeResult
    runner_version: str
    scorer_name: str
    proposer_name: str | None
    lb_policy: str
    execution_commit: str
    runner_code_sha256: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def __iter__(self):
        yield self.config
        yield self.model
        yield self.result

    def verify(self, registry: SourceRegistry) -> None:
        if self._authority is not _PREDICTION_RUN_AUTHORITY:
            raise ValueError("G3 prediction run must be issued by the fixed reference runner")
        self.model.verify(self.config, registry)
        self.result.verify_runtime_origin()
        if (
            self.runner_version != _PREDICTION_RUNNER_VERSION
            or self.scorer_name != _PREDICTION_SCORER_NAME
            or self.proposer_name is not None
            or self.lb_policy != _PREDICTION_LB_POLICY
            or self.result.scorer != self.scorer_name
            or self.result.proposer is not None
            or self.result.config_sha256 != config_sha256(self.config)
            or self.runner_code_sha256 != _prediction_generator_code_sha256()
        ):
            raise ValueError("G3 prediction run does not match its fixed execution contract")
        if len(self.execution_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.execution_commit
        ):
            raise ValueError("G3 prediction run execution commit must be a full SHA-1")


PredictionRun = G3PredictionRun


def _g3_reference_score(
    pod: dict[str, object], node: dict[str, object], ctx: dict[str, object]
) -> float:
    """Versioned least-allocated score used only for G3 prediction generation."""
    del ctx
    cpu_capacity = float(max(int(node["cpu_capacity"]), 1))
    mem_capacity = float(max(int(node["mem_capacity"]), 1))
    cpu_free = cpu_capacity - float(
        int(node["cpu_allocated"]) + int(pod["cpu_request"])
    )
    mem_free = mem_capacity - float(
        int(node["mem_allocated"]) + int(pod["mem_request"])
    )
    return (
        max(cpu_free, 0.0) / cpu_capacity + max(mem_free, 0.0) / mem_capacity
    ) / 2.0


def run_g3_prediction_episode(
    *,
    config: EpisodeConfig,
    model: WorkloadModelClosure,
    registry: SourceRegistry,
    seed: int,
) -> G3PredictionRun:
    """Run one pre-access episode under the sole supported G3 execution policy."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("G3 prediction seed must be an explicit integer")
    model.verify(config, registry)
    execution_commit = clean_checkout_commit()
    runner_code_sha256 = _prediction_generator_code_sha256()
    scorer = CallableScorer(
        "g3-reference-least-allocated-v1",
        _g3_reference_score,
    )
    result = run_episode(
        config,
        seed,
        scorer,
        proposer=None,
        lb_policy=_PREDICTION_LB_POLICY,
    )
    if clean_checkout_commit() != execution_commit:
        raise ValueError("G3 prediction checkout changed while the episode executed")
    if _prediction_generator_code_sha256() != runner_code_sha256:
        raise ValueError("G3 prediction runner code changed while the episode executed")
    run = G3PredictionRun(
        config=config,
        model=model,
        result=result,
        runner_version=_PREDICTION_RUNNER_VERSION,
        scorer_name=_PREDICTION_SCORER_NAME,
        proposer_name=None,
        lb_policy=_PREDICTION_LB_POLICY,
        execution_commit=execution_commit,
        runner_code_sha256=runner_code_sha256,
        _authority=_PREDICTION_RUN_AUTHORITY,
    )
    run.verify(registry)
    return run


def _prediction_generator_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _prediction_run_closure_sha256(prediction: G3PredictionArtifact) -> str:
    payload = {
        "source_id": prediction.source_id,
        "kind": prediction.kind,
        "selector_sha256": prediction.selector_sha256,
        "estimand": prediction.estimand,
        "curve_points": prediction.curve_points,
        "distribution_values_ms": prediction.distribution_values_ms,
        "effective_sample_counts": prediction.effective_sample_counts,
        "model_recipe_sha256s": prediction.model_recipe_sha256s,
        "calibration_evidence_sha256s": prediction.calibration_evidence_sha256s,
        "episode_config_sha256s": prediction.episode_config_sha256s,
        "episode_seeds": prediction.episode_seeds,
        "source_registry_sha256": prediction.source_registry_sha256,
        "episode_result_sha256s": prediction.episode_result_sha256s,
        "runner_version": prediction.runner_version,
        "scorer_name": prediction.scorer_name,
        "proposer_name": prediction.proposer_name,
        "lb_policy": prediction.lb_policy,
        "generator_version": prediction.generator_version,
        "generator_code_sha256": prediction.generator_code_sha256,
        "source_checkout_commit": prediction.source_checkout_commit,
    }
    encoded = ("g3-runtime-output-closure-v1:" + _canonical(payload)).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validated_runs(
    registry: SourceRegistry, runs: tuple[PredictionRun, ...]
) -> tuple[PredictionRun, ...]:
    registry.verify_seal()
    if not runs:
        raise ValueError("G3 prediction needs at least one simulator run")
    result_digests = tuple(run.result.digest() for run in runs)
    if len(result_digests) != len(set(result_digests)):
        raise ValueError("G3 prediction runs must be distinct retained episode results")
    for run in runs:
        run.verify(registry)
        if run.model.provenance_label != "calibration-evidence-citing-synthetic-assumption":
            raise ValueError(
                "G3 predictions require an explicitly synthetic evidence-citing closure"
            )
    return runs


def _make_runtime_prediction(
    *,
    registry: SourceRegistry,
    source_id: str,
    selector_sha256: str,
    kind: str,
    estimand: str,
    curve_points: tuple[tuple[float, float], ...],
    distribution_values_ms: tuple[float, ...],
    effective_sample_counts: tuple[int, ...],
    runs: tuple[PredictionRun, ...],
) -> G3PredictionArtifact:
    _sha256_hex("prediction selector", selector_sha256)
    if not source_id:
        raise ValueError("prediction source id must be non-empty")
    runs = _validated_runs(registry, runs)
    freeze_commit = clean_checkout_commit()
    if any(
        run.execution_commit != freeze_commit
        or run.runner_code_sha256 != _prediction_generator_code_sha256()
        for run in runs
    ):
        raise ValueError("G3 prediction runs were not executed from the frozen checkout")
    full_payload = {
        "schema": "g3-frozen-prediction-v1",
        "source_id": source_id,
        "kind": kind,
        "selector_sha256": selector_sha256,
        "estimand": estimand,
        "curve_points": curve_points,
        "distribution_values_ms": distribution_values_ms,
        "effective_sample_counts": effective_sample_counts,
        "model_recipe_sha256s": tuple(model.digest for _, model, _ in runs),
        "calibration_evidence_sha256s": tuple(
            sorted(
                {
                    evidence.artifact_sha256
                    for _, model, _ in runs
                    for evidence in model.evidence
                }
            )
        ),
        "episode_config_sha256s": tuple(config_sha256(config) for config, _, _ in runs),
        "episode_seeds": tuple(result.seed for _, _, result in runs),
        "source_registry_sha256": registry.manifest_sha256(),
        "episode_result_sha256s": tuple(result.digest() for _, _, result in runs),
        "runner_version": _PREDICTION_RUNNER_VERSION,
        "scorer_name": _PREDICTION_SCORER_NAME,
        "proposer_name": None,
        "lb_policy": _PREDICTION_LB_POLICY,
        "simulation_output_sha256": "",
        "generator_version": _PREDICTION_GENERATOR_VERSION,
        "generator_code_sha256": _prediction_generator_code_sha256(),
        "source_checkout_commit": freeze_commit,
    }
    provisional = G3PredictionArtifact(
        **full_payload,
        digest="",
        _authority=_PREDICTION_AUTHORITY,  # type: ignore[arg-type]
    )
    full_payload["simulation_output_sha256"] = _prediction_run_closure_sha256(provisional)
    digest = hashlib.sha256(
        (_PREDICTION_TAG + _canonical(full_payload)).encode()
    ).hexdigest()
    prediction = G3PredictionArtifact(
        **full_payload,
        digest=digest,
        _authority=_PREDICTION_AUTHORITY,  # type: ignore[arg-type]
    )
    prediction.verify()
    return prediction


def freeze_curve_prediction(
    *,
    registry: SourceRegistry,
    source_id: str,
    selector_sha256: str,
    x_unit: str,
    y_unit: str,
    runs: tuple[PredictionRun, ...],
) -> G3PredictionArtifact:
    """Derive a curve prediction from sealed ``Simulation.run`` results only."""
    if x_unit != "requests/s" or y_unit != "ms":
        raise ValueError(
            "runtime curve prediction supports only configured requests/s to direct ms"
        )
    validated = _validated_runs(registry, runs)
    points: list[tuple[float, float]] = []
    sample_counts: list[int] = []
    for config, _, result in validated:
        if any(
            template.arrival.arrivals_realized
            or template.arrival.rate_schedule
            or template.arrival.start_s != 0.0
            for template in config.workload.templates
        ):
            raise ValueError(
                "curve prediction x requires non-realized stationary arrivals starting at zero"
            )
        coordinate = math.fsum(
            template.arrival.rate_per_s for template in config.workload.templates
        )
        if config.cutoff_s <= 60.0 or config.cutoff_s > config.horizon_s - 60.0:
            raise ValueError(
                "curve prediction needs a 60s warm-up and at least 60s post-cutoff drain"
            )
        cohort = tuple(
            record
            for record in result.records
            if 60.0 <= record.arrival_s < config.cutoff_s
        )
        if any(not record.terminal for record in cohort):
            raise ValueError("curve measurement cohort is right-censored at the horizon")
        if any(record.status != "completed" for record in cohort):
            raise ValueError(
                "curve measurement requires every offered request to complete successfully"
            )
        latencies_ms = tuple(
            record.latency_s * 1000.0
            for record in cohort
            if record.status == "completed"
        )
        if len(latencies_ms) < _MIN_CURVE_COMPLETIONS_PER_POINT:
            raise ValueError(
                "curve prediction run lacks the minimum completed root-request count"
            )
        points.append((coordinate, _quantile(latencies_ms, 0.95)))
        sample_counts.append(len(latencies_ms))
    return _make_runtime_prediction(
        registry=registry,
        source_id=source_id,
        selector_sha256=selector_sha256,
        kind="published-curve",
        estimand=(
            "configured-total-root-offered-rps-to-"
            "completed-root-request-linear-type7-p95-latency-ms;"
            + _CURVE_SIM_RULE
        ),
        curve_points=tuple(points),
        distribution_values_ms=(),
        effective_sample_counts=tuple(sample_counts),
        runs=runs,
    )


def freeze_rt_prediction(
    *,
    registry: SourceRegistry,
    source_id: str,
    service: str,
    metric: str,
    runs: tuple[PredictionRun, ...],
) -> G3PredictionArtifact:
    """Derive Alibaba's RT estimand from retained service-instance call records."""
    if metric.upper() != "PROVIDERRPC_RT":
        raise ValueError("G3 currently supports only the providerRPC RT call population")
    validated = _validated_runs(registry, runs)
    samples: list[tuple[float, str, float]] = []
    for run_index, (config, _, result) in enumerate(validated):
        if config.horizon_s < 240.0 or config.horizon_s % 60.0 != 0.0:
            raise ValueError(
                "RT prediction requires >=240s horizon aligned to complete 60s cells"
            )
        measurement_start_s = 60.0
        measurement_end_s = config.horizon_s - 60.0
        root_cohort = tuple(
            record
            for record in result.records
            if measurement_start_s <= record.arrival_s < measurement_end_s
        )
        if any(not record.terminal for record in root_cohort):
            raise ValueError("RT measurement root cohort is right-censored at the horizon")
        if any(record.status != "completed" for record in root_cohort):
            raise ValueError(
                "RT prediction requires every root request in the measurement window "
                "to complete successfully"
            )
        service_ids = tuple(item.id for item in config.workload.services if item.name == service)
        if len(service_ids) != 1:
            raise ValueError("RT prediction service name is not unique in EpisodeConfig")
        service_id = service_ids[0]
        if any(template.root.service == service_id for template in config.workload.templates):
            raise ValueError("providerRPC RT service must never be a root/ingress service")
        selected = tuple(
            call
            for call in result.service_calls
            if call.service == service_id
            and measurement_start_s <= call.start_s < measurement_end_s
        )
        if not selected or any(not call.path for call in selected):
            raise ValueError("providerRPC RT samples must be abstract child/provider calls")
        if any(call.end_s is None for call in selected):
            raise ValueError("RT measurement window is right-censored at the horizon")
        samples.extend(
            (
                call.start_s,
                f"run-{run_index}:pod-{call.pod_uid}",
                call.latency_s * 1000.0,
            )
            for call in selected
        )
    distribution = aggregate_instance_interval_mean_rt(tuple(samples))
    return _make_runtime_prediction(
        registry=registry,
        source_id=source_id,
        selector_sha256=rt_shape_selector_sha256(
            source_id=source_id, service=service, metric=metric
        ),
        kind="rt-shape",
        estimand=(
            "service-instance-by-60s-interval-mean-response-time;" + _RT_SIM_WINDOW_RULE
        ),
        curve_points=(),
        distribution_values_ms=distribution,
        effective_sample_counts=(len(distribution),),
        runs=runs,
    )


@dataclass(frozen=True)
class P1RegistrationReceipt:
    schema: str
    receipt_id: str
    registration_id: str
    authority_url: str
    frozen_commit: str
    registered_at_utc: str
    protocol_spec_sha256: str
    receipt_sha256: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify(self) -> None:
        if self._authority is not _P1_RECEIPT_AUTHORITY:
            raise ValueError("P1 receipt must be loaded from exact registration bytes")
        if self.schema != "g3-p1-registration-receipt-v1":
            raise ValueError("unknown P1 registration receipt schema")
        if not self.receipt_id or not self.registration_id or not self.authority_url.startswith(
            "https://"
        ):
            raise ValueError("P1 receipt needs an id, registration id, and HTTPS authority")
        if len(self.frozen_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.frozen_commit
        ):
            raise ValueError("P1 receipt frozen_commit must be a full SHA-1 object id")
        try:
            registered = datetime.fromisoformat(self.registered_at_utc)
        except ValueError as exc:
            raise ValueError("P1 receipt timestamp must be ISO-8601") from exc
        if registered.tzinfo is None or registered.utcoffset() != timedelta(0):
            raise ValueError("P1 receipt timestamp must use the UTC offset")
        _sha256_hex("P1 protocol spec", self.protocol_spec_sha256)
        _sha256_hex("P1 receipt", self.receipt_sha256)


def _parse_p1_registration_receipt(
    receipt_content: bytes, expected_protocol_spec_sha256: str
) -> P1RegistrationReceipt:
    """Load the durable receipt returned by the external preregistration custodian."""
    if not isinstance(receipt_content, bytes) or not receipt_content:
        raise ValueError("P1 registration receipt must be non-empty exact bytes")
    try:
        payload = json.loads(receipt_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("P1 registration receipt must be UTF-8 JSON") from exc
    required = {
        "schema",
        "receipt_id",
        "registration_id",
        "authority_url",
        "frozen_commit",
        "registered_at_utc",
        "protocol_spec_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or _canonical(payload).encode() != receipt_content
        or payload["protocol_spec_sha256"] != expected_protocol_spec_sha256
    ):
        raise ValueError("P1 receipt does not canonically bind the expected protocol spec")
    receipt = P1RegistrationReceipt(
        **payload,
        receipt_sha256=hashlib.sha256(receipt_content).hexdigest(),
        _authority=_P1_RECEIPT_AUTHORITY,
    )
    receipt.verify()
    return receipt


def load_p1_registration_receipt(
    expected_protocol_spec_sha256: str,
) -> P1RegistrationReceipt:
    """Load only the repository-pinned external P1 receipt; fail closed until frozen."""
    if PRODUCTION_P1_RECEIPT_SHA256 is None or not P1_RECEIPT_PATH.is_file():
        raise ValueError("registered production P1 G3 receipt is unavailable")
    content = P1_RECEIPT_PATH.read_bytes()
    if hashlib.sha256(content).hexdigest() != PRODUCTION_P1_RECEIPT_SHA256:
        raise ValueError("registered production P1 G3 receipt hash mismatch")
    return _parse_p1_registration_receipt(content, expected_protocol_spec_sha256)


def _load_p1_registration_receipt_fixture_bytes(
    content: bytes, expected_protocol_spec_sha256: str
) -> P1RegistrationReceipt:
    """Test-only exact-byte loader for a synthetic custodian receipt."""
    return _parse_p1_registration_receipt(content, expected_protocol_spec_sha256)


def g3_protocol_spec_sha256(
    *,
    source_split_sha256: str,
    rt_quantiles: tuple[float, ...],
    curve_log_ratio_abs_band: float,
    rt_log_quantile_abs_band: float,
    evidence_selectors: tuple[tuple[str, str], ...],
    predictions: tuple[G3PredictionArtifact, ...],
) -> str:
    payload = {
        "source_split_sha256": source_split_sha256,
        "curve_residual": "log(predicted/observed)",
        "interpolation": "linear-at-observed-x",
        "extrapolation": "forbidden",
        "missing_data": "UNAVAILABLE-never-pass",
        "rt_quantiles": tuple(rt_quantiles),
        "curve_log_ratio_abs_band": curve_log_ratio_abs_band,
        "rt_log_quantile_abs_band": rt_log_quantile_abs_band,
        "aggregate": "maximum-absolute-residual",
        "consequence": "gate-fail-and-narrow-workload-claims; no retuning on heldout",
        "evidence_selectors": tuple(sorted(evidence_selectors)),
        "prediction_artifacts": tuple(
            sorted((prediction.source_id, prediction.digest) for prediction in predictions)
        ),
    }
    return hashlib.sha256((_P1_SPEC_TAG + _canonical(payload)).encode()).hexdigest()


@dataclass(frozen=True)
class G3Protocol:
    schema: str
    registration_id: str
    registration_receipt_id: str
    registration_authority_url: str
    registration_receipt_sha256: str
    protocol_spec_sha256: str
    frozen_commit: str
    frozen_at_utc: str
    source_split_sha256: str
    curve_residual: str
    interpolation: str
    extrapolation: str
    missing_data: str
    rt_quantiles: tuple[float, ...]
    curve_log_ratio_abs_band: float
    rt_log_quantile_abs_band: float
    aggregate: str
    consequence: str
    evidence_selectors: tuple[tuple[str, str], ...]
    prediction_artifacts: tuple[tuple[str, str], ...]
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }

    def verify(self) -> None:
        if self._authority is not _PROTOCOL_AUTHORITY:
            raise ValueError("G3 protocol must be issued from an external P1 receipt")
        if len(self.frozen_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.frozen_commit
        ):
            raise ValueError("G3 frozen_commit must be a full SHA-1 object id")
        if not self.registration_receipt_id or not self.registration_authority_url.startswith(
            "https://"
        ):
            raise ValueError("G3 protocol must cite its external registration receipt")
        _sha256_hex("G3 registration receipt", self.registration_receipt_sha256)
        _sha256_hex("G3 protocol spec", self.protocol_spec_sha256)
        try:
            frozen_at = datetime.fromisoformat(self.frozen_at_utc)
        except ValueError as exc:
            raise ValueError("G3 frozen_at_utc must be an ISO-8601 timestamp") from exc
        if frozen_at.tzinfo is None or frozen_at.utcoffset() != timedelta(0):
            raise ValueError("G3 frozen_at_utc must use the UTC offset")
        if self.interpolation != "linear-at-observed-x" or self.extrapolation != "forbidden":
            raise ValueError("G3 interpolation/extrapolation contract changed")
        if self.missing_data != "UNAVAILABLE-never-pass":
            raise ValueError("G3 missing-data behavior must be UNAVAILABLE-never-pass")
        if self.curve_residual != "log(predicted/observed)":
            raise ValueError("G3 curve residual definition changed")
        if not self.rt_quantiles or tuple(sorted(set(self.rt_quantiles))) != self.rt_quantiles:
            raise ValueError("G3 RT quantiles must be unique and sorted")
        if any(not 0 < quantile < 1 for quantile in self.rt_quantiles):
            raise ValueError("G3 RT quantiles must be inside (0,1)")
        for band in (self.curve_log_ratio_abs_band, self.rt_log_quantile_abs_band):
            if not math.isfinite(band) or band <= 0:
                raise ValueError("G3 bands must be finite and > 0")
        if self.consequence != "gate-fail-and-narrow-workload-claims; no retuning on heldout":
            raise ValueError("G3 consequence changed")
        if self.evidence_selectors != tuple(sorted(self.evidence_selectors)):
            raise ValueError("G3 evidence selectors must be canonically sorted")
        source_ids = [source_id for source_id, _ in self.evidence_selectors]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("G3 evidence selectors must have unique source ids")
        for source_id, digest in self.evidence_selectors:
            if not source_id:
                raise ValueError("G3 evidence source id must be non-empty")
            _sha256_hex("G3 evidence selector", digest)
        if self.prediction_artifacts != tuple(sorted(self.prediction_artifacts)):
            raise ValueError("G3 prediction artifacts must be canonically sorted")
        prediction_sources = [source_id for source_id, _ in self.prediction_artifacts]
        if len(prediction_sources) != len(set(prediction_sources)):
            raise ValueError("G3 prediction artifacts must have unique source ids")
        for source_id, digest in self.prediction_artifacts:
            if not source_id:
                raise ValueError("G3 prediction source id must be non-empty")
            _sha256_hex("G3 prediction artifact", digest)
        spec_payload = {
            "source_split_sha256": self.source_split_sha256,
            "curve_residual": self.curve_residual,
            "interpolation": self.interpolation,
            "extrapolation": self.extrapolation,
            "missing_data": self.missing_data,
            "rt_quantiles": self.rt_quantiles,
            "curve_log_ratio_abs_band": self.curve_log_ratio_abs_band,
            "rt_log_quantile_abs_band": self.rt_log_quantile_abs_band,
            "aggregate": self.aggregate,
            "consequence": self.consequence,
            "evidence_selectors": self.evidence_selectors,
            "prediction_artifacts": self.prediction_artifacts,
        }
        expected_spec = hashlib.sha256(
            (_P1_SPEC_TAG + _canonical(spec_payload)).encode()
        ).hexdigest()
        if self.protocol_spec_sha256 != expected_spec:
            raise ValueError("G3 protocol does not match its registered scientific spec")
        expected = hashlib.sha256(
            (_PROTOCOL_TAG + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("G3 protocol digest mismatch")


def freeze_protocol(
    *,
    registration_receipt: P1RegistrationReceipt,
    source_split_sha256: str,
    rt_quantiles: tuple[float, ...],
    curve_log_ratio_abs_band: float,
    rt_log_quantile_abs_band: float,
    evidence_selectors: tuple[tuple[str, str], ...] = (),
    predictions: tuple[G3PredictionArtifact, ...] = (),
) -> G3Protocol:
    selector_map = dict(evidence_selectors)
    protocol_spec_sha256 = g3_protocol_spec_sha256(
        source_split_sha256=source_split_sha256,
        rt_quantiles=rt_quantiles,
        curve_log_ratio_abs_band=curve_log_ratio_abs_band,
        rt_log_quantile_abs_band=rt_log_quantile_abs_band,
        evidence_selectors=evidence_selectors,
        predictions=predictions,
    )
    registration_receipt.verify()
    if registration_receipt.protocol_spec_sha256 != protocol_spec_sha256:
        raise ValueError("P1 receipt does not bind this exact G3 protocol spec")
    receipt = registration_receipt
    for prediction in predictions:
        prediction.verify()
        if prediction.source_checkout_commit != receipt.frozen_commit:
            raise ValueError("G3 prediction was not generated from the P1 frozen commit")
        if selector_map.get(prediction.source_id) != prediction.selector_sha256:
            raise ValueError("G3 prediction does not match its evidence selector")
        if prediction.source_registry_sha256 != source_split_sha256:
            raise ValueError("G3 prediction was generated under a different source split")
    payload = {
        "schema": "g3-protocol-v1",
        "registration_id": receipt.registration_id,
        "registration_receipt_id": receipt.receipt_id,
        "registration_authority_url": receipt.authority_url,
        "registration_receipt_sha256": receipt.receipt_sha256,
        "protocol_spec_sha256": protocol_spec_sha256,
        "frozen_commit": receipt.frozen_commit,
        "frozen_at_utc": receipt.registered_at_utc,
        "source_split_sha256": source_split_sha256,
        "curve_residual": "log(predicted/observed)",
        "interpolation": "linear-at-observed-x",
        "extrapolation": "forbidden",
        "missing_data": "UNAVAILABLE-never-pass",
        "rt_quantiles": tuple(rt_quantiles),
        "curve_log_ratio_abs_band": curve_log_ratio_abs_band,
        "rt_log_quantile_abs_band": rt_log_quantile_abs_band,
        "aggregate": "maximum-absolute-residual",
        "consequence": "gate-fail-and-narrow-workload-claims; no retuning on heldout",
        "evidence_selectors": tuple(sorted(evidence_selectors)),
        "prediction_artifacts": tuple(
            sorted((prediction.source_id, prediction.digest) for prediction in predictions)
        ),
    }
    digest = hashlib.sha256((_PROTOCOL_TAG + _canonical(payload)).encode()).hexdigest()
    protocol = G3Protocol(
        **payload,
        digest=digest,
        _authority=_PROTOCOL_AUTHORITY,  # type: ignore[arg-type]
    )
    protocol.verify()
    return protocol


@dataclass(frozen=True)
class Residual:
    source_id: str
    kind: str
    evidence_sha256: str
    selector_sha256: str
    prediction_artifact_sha256: str
    evidence_artifact_sha256: str
    coordinate_closure_sha256: str
    coordinate_count: int
    series: str
    coordinate: float
    observed: float
    predicted: float
    residual: float
    band: float
    within_band: bool
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify(self, protocol: G3Protocol | None = None) -> None:
        if self._authority is not _RESIDUAL_AUTHORITY:
            raise ValueError("G3 residual must be issued by a source-bound evaluator")
        if self.kind not in ("published-curve", "rt-shape"):
            raise ValueError("unknown G3 residual kind")
        _sha256_hex("residual evidence_sha256", self.evidence_sha256)
        _sha256_hex("residual selector_sha256", self.selector_sha256)
        _sha256_hex("residual prediction artifact", self.prediction_artifact_sha256)
        _sha256_hex("residual evidence_artifact_sha256", self.evidence_artifact_sha256)
        _sha256_hex("residual coordinate closure", self.coordinate_closure_sha256)
        if not isinstance(self.coordinate_count, int) or isinstance(self.coordinate_count, bool):
            raise TypeError("G3 residual coordinate count must be an integer")
        if self.coordinate_count <= 0:
            raise ValueError("G3 residual coordinate count must be > 0")
        if not self.source_id or not self.series:
            raise ValueError("G3 residual source and series must be non-empty")
        if any(
            not math.isfinite(value)
            for value in (self.coordinate, self.observed, self.predicted, self.residual, self.band)
        ) or self.observed <= 0 or self.predicted <= 0 or self.band <= 0:
            raise ValueError("G3 residual values must be finite and positive where required")
        expected_residual = math.log(self.predicted / self.observed)
        if not math.isclose(self.residual, expected_residual, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("G3 residual is not log(predicted/observed)")
        expected_within = abs(expected_residual) <= self.band
        if self.within_band is not expected_within:
            raise ValueError("G3 residual within_band flag is inconsistent")
        if protocol is not None:
            expected_band = (
                protocol.curve_log_ratio_abs_band
                if self.kind == "published-curve"
                else protocol.rt_log_quantile_abs_band
            )
            if self.band != expected_band:
                raise ValueError("G3 residual band does not match its frozen protocol")
        payload = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("seal", "_authority")
        }
        expected_seal = hashlib.sha256(
            (_RESIDUAL_TAG + _canonical(payload)).encode()
        ).hexdigest()
        if self.seal != expected_seal:
            raise ValueError("G3 residual seal mismatch")


def _residual(**payload: object) -> Residual:
    seal = hashlib.sha256((_RESIDUAL_TAG + _canonical(payload)).encode()).hexdigest()
    residual = Residual(
        **payload, seal=seal, _authority=_RESIDUAL_AUTHORITY  # type: ignore[arg-type]
    )
    residual.verify()
    return residual


@dataclass(frozen=True)
class G3Result:
    schema: str
    status: str
    protocol_sha256: str
    source_split_sha256: str
    access_receipt_sha256: str | None
    residuals: tuple[Residual, ...]
    unavailable_sources: tuple[str, ...]
    supports: str
    note: str
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }

    def verify(self) -> None:
        if self._authority is not _RESULT_AUTHORITY:
            raise ValueError("G3 result must be issued by contextual evaluation")
        if self.status not in ("PASS", "FAIL", "UNAVAILABLE"):
            raise ValueError("unknown G3 status")
        if self.supports != SUPPORTS_ONLY:
            raise ValueError("G3 claim boundary changed")
        if self.status == "PASS" and (self.unavailable_sources or not self.residuals):
            raise ValueError("G3 cannot pass with missing sources or no residuals")
        if self.status == "PASS" and any(not item.within_band for item in self.residuals):
            raise ValueError("G3 PASS contains an out-of-band residual")
        if self.status == "UNAVAILABLE":
            if self.access_receipt_sha256 is not None:
                raise ValueError("UNAVAILABLE G3 result cannot claim a held-out access receipt")
        elif self.access_receipt_sha256 is None:
            raise ValueError("evaluated G3 result must bind its held-out access receipt")
        else:
            _sha256_hex("G3 result access receipt", self.access_receipt_sha256)
        for item in self.residuals:
            item.verify()
        expected = hashlib.sha256((_RESULT_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("G3 result digest mismatch")


def unavailable_result(protocol: G3Protocol, registry: SourceRegistry) -> G3Result:
    protocol.verify()
    registry.verify_seal()
    if protocol.source_split_sha256 != registry.manifest_sha256():
        raise ValueError("G3 protocol source split does not match registry")
    unavailable = tuple(
        source.source_id
        for source in registry.sources
        if source.role in ("calibration", FACE_VALIDITY)
        and source.kind in ("trace", "published-curve")
        and source.payload_status != "AVAILABLE"
    )
    payload = {
        "schema": "g3-result-v1",
        "status": "UNAVAILABLE",
        "protocol_sha256": protocol.digest,
        "source_split_sha256": registry.manifest_sha256(),
        "access_receipt_sha256": None,
        "residuals": (),
        "unavailable_sources": unavailable,
        "supports": SUPPORTS_ONLY,
        "note": (
            "Required raw traces/published curve artifacts are absent; "
            "no placeholder data generated."
        ),
    }
    digest = hashlib.sha256((_RESULT_TAG + _canonical(payload)).encode()).hexdigest()
    result = G3Result(
        **payload,
        digest=digest,
        _authority=_RESULT_AUTHORITY,  # type: ignore[arg-type]
    )
    result.verify()
    return result


def _interpolate(points: tuple[tuple[float, float], ...], x: float) -> float:
    if x < points[0][0] or x > points[-1][0]:
        raise ValueError("G3 extrapolation is forbidden")
    for px, py in points:
        if px == x:
            return py
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 < x < x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError("prediction grid does not cover observed coordinate")


def _coordinate_closure(kind: str, series: str, coordinates: tuple[float, ...]) -> str:
    return hashlib.sha256(
        (
            "g3-coordinate-closure-v1:"
            + _canonical({"kind": kind, "series": series, "coordinates": coordinates})
        ).encode()
    ).hexdigest()


def _require_single_payload_face_validity(registry: SourceRegistry) -> None:
    """Fail closed while the one-digest-per-source access receipt meets a multi-part source.

    ``G3AccessReceipt.opened_payload_sha256s`` carries exactly one digest per
    face-validity source.  A component-aware face-validity registration (for example a
    future multi-member v2022 acquisition) would silently bind only one of its parser
    inputs, so the receipt schema must be extended before such a source is opened.
    """
    offending = tuple(
        source.source_id
        for source in registry.sources
        if source.role == FACE_VALIDITY and source.payload_components
    )
    if offending:
        raise ValueError(
            "G3 held-out access receipts bind one payload digest per source; "
            f"component-aware face-validity sources {list(offending)} need a "
            "component-aware receipt schema first"
        )


def _require_face_access(
    source_id: str,
    role: str,
    protocol: G3Protocol,
    registry: SourceRegistry,
    access_receipt: G3AccessReceipt | None,
    access_marker: Path | None,
) -> None:
    if role != FACE_VALIDITY:
        return
    if access_receipt is None or access_marker is None:
        raise PermissionError("face-validity evidence cannot be read before the P1 access gate")
    access_receipt.verify(protocol, registry, access_marker)
    if source_id not in access_receipt.face_validity_source_ids:
        raise ValueError("face-validity source is absent from the access receipt")


def curve_residuals(
    observed: PublishedCurve,
    prediction: G3PredictionArtifact,
    protocol: G3Protocol,
    registry: SourceRegistry,
    role: str,
    access_receipt: G3AccessReceipt | None = None,
    access_marker: Path | None = None,
) -> tuple[Residual, ...]:
    observed.verify()
    prediction.verify()
    protocol.verify()
    if role not in (CALIBRATION, FACE_VALIDITY):
        raise ValueError("G3 curve role must be calibration or face-validity")
    _require_face_access(
        observed.source_id, role, protocol, registry, access_receipt, access_marker
    )
    registry.require_payload(observed.source_id, role, observed.paper_sha256)
    selector = published_curve_selector_sha256(
        source_id=observed.source_id,
        page=observed.page,
        figure_or_table=observed.figure_or_table,
        series_label=observed.series_label,
        x_unit=observed.x_unit,
        y_unit=observed.y_unit,
        extraction_method=observed.extraction_method,
        extraction_tool_version=observed.extraction_tool_version,
        extraction_code_sha256=observed.extraction_code_sha256,
        configuration=observed.configuration,
        unknown_configuration=observed.unknown_configuration,
        point_selection_rule=observed.point_selection_rule,
    )
    if dict(protocol.evidence_selectors).get(observed.source_id) != selector:
        raise ValueError("published curve does not match its P1-frozen evidence selector")
    if (
        prediction.kind != "published-curve"
        or prediction.source_id != observed.source_id
        or prediction.selector_sha256 != selector
        or dict(protocol.prediction_artifacts).get(observed.source_id) != prediction.digest
    ):
        raise ValueError("curve prediction was not frozen into the P1 protocol")
    if observed.x_unit != "requests/s" or observed.y_unit != "ms":
        raise ValueError(
            "runtime-derived curves require observed x_unit=requests/s and y_unit=ms"
        )
    predicted_points = prediction.curve_points
    xs = [x for x, _ in predicted_points]
    if xs != sorted(set(xs)) or any(
        not math.isfinite(x) or not math.isfinite(y) or y <= 0 for x, y in predicted_points
    ):
        raise ValueError("predicted curve must be finite, positive, and strictly ordered")
    out = []
    coordinates = tuple(x for x, _ in observed.points)
    coordinate_closure = _coordinate_closure(
        "published-curve", observed.series_label, coordinates
    )
    for x, actual in observed.points:
        predicted = _interpolate(predicted_points, x)
        residual = math.log(predicted / actual)
        band = protocol.curve_log_ratio_abs_band
        out.append(
            _residual(
                source_id=observed.source_id,
                kind="published-curve",
                evidence_sha256=observed.paper_sha256,
                selector_sha256=selector,
                prediction_artifact_sha256=prediction.digest,
                evidence_artifact_sha256=observed.seal,
                coordinate_closure_sha256=coordinate_closure,
                coordinate_count=len(coordinates),
                series=observed.series_label,
                coordinate=x,
                observed=actual,
                predicted=predicted,
                residual=residual,
                band=band,
                within_band=abs(residual) <= band,
            )
        )
    return tuple(out)


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def rt_quantile_residuals(
    *,
    corpus: MetricCorpus,
    registry: SourceRegistry,
    service: str,
    metric: str,
    prediction: G3PredictionArtifact,
    protocol: G3Protocol,
    access_receipt: G3AccessReceipt | None = None,
    access_marker: Path | None = None,
) -> tuple[Residual, ...]:
    """Compare Alibaba and simulator instance×60s interval-mean RT shapes."""
    prediction.verify()
    protocol.verify()
    observed_ms, selection_artifact_sha256 = rt_shape_evidence(
        corpus=corpus,
        registry=registry,
        service=service,
        metric=metric,
        protocol=protocol,
        access_receipt=access_receipt,
        access_marker=access_marker,
    )
    selector = rt_shape_selector_sha256(
        source_id=corpus.source_id, service=service, metric=metric
    )
    if (
        prediction.kind != "rt-shape"
        or prediction.source_id != corpus.source_id
        or prediction.selector_sha256 != selector
        or dict(protocol.prediction_artifacts).get(corpus.source_id) != prediction.digest
    ):
        raise ValueError("RT prediction was not frozen into the P1 protocol")
    predicted_instance_interval_mean_ms = prediction.distribution_values_ms
    for name, values in (
        ("observed", observed_ms),
        ("predicted", predicted_instance_interval_mean_ms),
    ):
        if len(values) < _MIN_RT_INTERVAL_CELLS or any(
            not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError(
                f"{name} RT shape needs at least {_MIN_RT_INTERVAL_CELLS} "
                "finite positive cells"
            )
    observed_median = _quantile(observed_ms, 0.5)
    predicted_median = _quantile(predicted_instance_interval_mean_ms, 0.5)
    coordinate_closure = _coordinate_closure(
        "rt-shape",
        "median-normalized-instance-interval-mean-rt-quantile-shape",
        protocol.rt_quantiles,
    )
    out = []
    for probability in protocol.rt_quantiles:
        observed = _quantile(observed_ms, probability) / observed_median
        predicted = _quantile(predicted_instance_interval_mean_ms, probability) / predicted_median
        residual = math.log(predicted / observed)
        band = protocol.rt_log_quantile_abs_band
        out.append(
            _residual(
                source_id=corpus.source_id,
                kind="rt-shape",
                evidence_sha256=corpus.content_sha256,
                selector_sha256=selector,
                prediction_artifact_sha256=prediction.digest,
                evidence_artifact_sha256=selection_artifact_sha256,
                coordinate_closure_sha256=coordinate_closure,
                coordinate_count=len(protocol.rt_quantiles),
                series="median-normalized-instance-interval-mean-rt-quantile-shape",
                coordinate=probability,
                observed=observed,
                predicted=predicted,
                residual=residual,
                band=band,
                within_band=abs(residual) <= band,
            )
        )
    return tuple(out)


def rt_shape_evidence(
    *,
    corpus: MetricCorpus,
    registry: SourceRegistry,
    service: str,
    metric: str,
    protocol: G3Protocol,
    access_receipt: G3AccessReceipt | None = None,
    access_marker: Path | None = None,
) -> tuple[tuple[float, ...], str]:
    """Open and seal the exact P1-selected interval-mean RT evidence."""
    corpus.verify_seal()
    protocol.verify()
    source = registry.get(corpus.source_id)
    if source.role not in (CALIBRATION, FACE_VALIDITY):
        raise ValueError("G3 RT evidence must be calibration or face-validity data")
    registry.require_payload(corpus.source_id, source.role, corpus.content_sha256)
    _require_face_access(
        corpus.source_id,
        source.role,
        protocol,
        registry,
        access_receipt,
        access_marker,
    )
    selector = rt_shape_selector_sha256(
        source_id=corpus.source_id, service=service, metric=metric
    )
    if dict(protocol.evidence_selectors).get(corpus.source_id) != selector:
        raise ValueError("RT evidence does not match its P1-frozen selector")
    candidates = tuple(
        observation
        for observation in corpus.observations
        if observation.service == service
        and observation.metric == metric.upper()
        and observation.unit == "ms"
        and observation.estimand
        == "service-instance-by-60s-interval-mean-response-time"
        and observation.aggregation_interval_s == 60.0
    )
    cell_keys = tuple(
        (
            observation.timestamp_ms,
            observation.service,
            observation.instance,
            observation.metric,
        )
        for observation in candidates
    )
    if len(cell_keys) != len(set(cell_keys)):
        raise ValueError("G3 RT evidence contains duplicate instance-interval cells")
    if any(
        observation.activity_metric
        != observation.metric.removesuffix("_RT") + "_MCR"
        or observation.activity_value is None
        for observation in candidates
    ):
        raise ValueError("RT evidence requires a unique paired same-cell family MCR")
    selected = tuple(
        observation
        for observation in candidates
        if observation.activity_value is not None and observation.activity_value > 0.0
    )
    observed_values: list[float] = []
    for observation in selected:
        if observation.value > 0 and observation.left_censored_below is None:
            observed_values.append(observation.value)
        elif observation.value == 0 and observation.left_censored_below is not None:
            # P1 selector freezes this deterministic substitution.  The censored mass
            # remains in the empirical distribution instead of being silently dropped.
            observed_values.append(observation.left_censored_below / 2.0)
        else:
            raise ValueError("RT evidence contains zero/unclassified or inconsistent censoring")
    observed_ms = tuple(observed_values)
    if len(observed_ms) < _MIN_RT_INTERVAL_CELLS:
        raise ValueError(
            f"G3 RT evidence needs at least {_MIN_RT_INTERVAL_CELLS} active cells"
        )
    selection_artifact_sha256 = hashlib.sha256(
        (
            "g3-rt-shape-selection-v1:"
            + _canonical(
                {
                    "source_id": corpus.source_id,
                    "source_content_sha256": corpus.content_sha256,
                    "corpus_seal": corpus.seal,
                    "service": service,
                    "metric": metric.upper(),
                    "estimand": "service-instance-by-60s-interval-mean-response-time",
                    "aggregation_interval_s": 60.0,
                    "activity_rule": _RT_ACTIVITY_RULE,
                    "call_population_rule": _RT_CALL_POPULATION_RULE,
                    "minimum_active_cells": _MIN_RT_INTERVAL_CELLS,
                    "candidate_count": len(candidates),
                    "inactive_count": len(candidates) - len(selected),
                    "left_censoring_treatment": _RT_CENSORING_TREATMENT,
                    "quantile_method": _RT_QUANTILE_METHOD,
                    "observed_ms": observed_ms,
                }
            )
        ).encode()
    ).hexdigest()
    return observed_ms, selection_artifact_sha256


def result_from_residuals(
    protocol: G3Protocol,
    registry: SourceRegistry,
    residuals: tuple[Residual, ...],
    access_receipt: G3AccessReceipt,
    access_marker: Path,
) -> G3Result:
    protocol.verify()
    registry.verify_seal()
    access_receipt.verify(protocol, registry, access_marker)
    if protocol.source_split_sha256 != registry.manifest_sha256():
        raise ValueError("G3 protocol source split does not match registry")
    unavailable = tuple(
        source.source_id
        for source in registry.sources
        if source.role in ("calibration", FACE_VALIDITY)
        and source.kind in ("trace", "published-curve")
        and source.payload_status != "AVAILABLE"
    )
    if unavailable:
        raise ValueError(
            f"G3 residual result cannot be formed while sources are unavailable: {unavailable}"
        )
    if not residuals:
        raise ValueError("G3 requires residuals from every registered held-out source")
    required = {
        source.source_id: source
        for source in registry.sources
        if source.role in ("calibration", FACE_VALIDITY)
        and source.kind in ("trace", "published-curve")
    }
    expected_evidence_selectors = dict(protocol.evidence_selectors)
    if set(expected_evidence_selectors) != set(required):
        raise ValueError(
            "G3 protocol must freeze one evidence selector for every registered G3 source"
        )
    expected_predictions = dict(protocol.prediction_artifacts)
    if set(expected_predictions) != set(required):
        raise ValueError(
            "G3 protocol must freeze one pre-access prediction for every G3 source"
        )
    for item in residuals:
        item.verify(protocol)
        try:
            source = required[item.source_id]
        except KeyError as exc:
            raise ValueError(
                f"G3 residual cites an unregistered source {item.source_id!r}"
            ) from exc
        expected_kind = "rt-shape" if source.kind == "trace" else "published-curve"
        if (
            item.kind != expected_kind
            or item.evidence_sha256 not in source.authorized_digests()
            or item.selector_sha256 != expected_evidence_selectors[item.source_id]
            or item.prediction_artifact_sha256 != expected_predictions[item.source_id]
        ):
            raise ValueError("G3 residual is not bound to its exact registered source payload")
    observed = {item.source_id for item in residuals}
    if set(required) != observed:
        raise ValueError("G3 residuals do not cover exactly every registered G3 source")
    rt_coordinates = {
        item.source_id: tuple(
            sorted(
                residual.coordinate
                for residual in residuals
                if residual.source_id == item.source_id
            )
        )
        for item in residuals
        if item.kind == "rt-shape"
    }
    if any(coordinates != protocol.rt_quantiles for coordinates in rt_coordinates.values()):
        raise ValueError("G3 RT residuals do not cover the complete frozen quantile grid")
    groups: dict[tuple[str, str, str], list[Residual]] = {}
    for item in residuals:
        groups.setdefault(
            (item.source_id, item.series, item.evidence_artifact_sha256), []
        ).append(item)
    for (_, series, _), group in groups.items():
        coordinates = tuple(sorted(item.coordinate for item in group))
        closures = {item.coordinate_closure_sha256 for item in group}
        counts = {item.coordinate_count for item in group}
        expected_closure = _coordinate_closure(group[0].kind, series, coordinates)
        if (
            len(closures) != 1
            or len(counts) != 1
            or counts != {len(group)}
            or closures != {expected_closure}
            or len(coordinates) != len(set(coordinates))
        ):
            raise ValueError(
                "G3 residuals omit or duplicate coordinates from an evidence series"
            )
    status = "PASS" if all(item.within_band for item in residuals) else "FAIL"
    payload = {
        "schema": "g3-result-v1",
        "status": status,
        "protocol_sha256": protocol.digest,
        "source_split_sha256": registry.manifest_sha256(),
        "access_receipt_sha256": access_receipt.digest,
        "residuals": residuals,
        "unavailable_sources": (),
        "supports": SUPPORTS_ONLY,
        "note": "P1-frozen residual evaluation; held-out outcomes were not used to refit.",
    }
    digest = hashlib.sha256((_RESULT_TAG + _canonical(payload)).encode()).hexdigest()
    result = G3Result(
        **payload,
        digest=digest,
        _authority=_RESULT_AUTHORITY,  # type: ignore[arg-type]
    )
    result.verify()
    return result


@dataclass(frozen=True)
class G3AccessReceipt:
    schema: str
    protocol_sha256: str
    source_split_sha256: str
    face_validity_source_ids: tuple[str, ...]
    opened_payload_sha256s: tuple[tuple[str, str], ...]
    opened_at_utc: str
    marker_sha256: str
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }

    def verify(self, protocol: G3Protocol, registry: SourceRegistry, marker: Path) -> None:
        if self._authority is not _G3_ACCESS_AUTHORITY:
            raise ValueError("G3 access receipt must be issued by the held-out gate")
        protocol.verify()
        registry.verify_seal()
        require_frozen_clean_checkout(protocol.frozen_commit)
        expected_sources = tuple(
            source.source_id for source in registry.sources if source.role == FACE_VALIDITY
        )
        if (
            self.schema != "g3-access-receipt-v1"
            or self.protocol_sha256 != protocol.digest
            or self.source_split_sha256 != registry.manifest_sha256()
            or self.face_validity_source_ids != expected_sources
        ):
            raise ValueError("G3 access receipt does not match protocol/registry")
        _require_single_payload_face_validity(registry)
        expected_payloads = tuple(
            (source.source_id, source.payload_sha256)
            for source in registry.sources
            if source.role == FACE_VALIDITY
        )
        if self.opened_payload_sha256s != expected_payloads:
            raise ValueError("G3 receipt does not bind every exact face-validity payload")
        try:
            opened_at = datetime.fromisoformat(self.opened_at_utc)
            frozen_at = datetime.fromisoformat(protocol.frozen_at_utc)
        except ValueError as exc:
            raise ValueError("G3 receipt timestamps must be ISO-8601") from exc
        if opened_at.tzinfo is None or opened_at < frozen_at:
            raise ValueError("G3 evidence was opened before its P1 protocol was frozen")
        _sha256_hex("G3 marker SHA-256", self.marker_sha256)
        if (
            not marker.is_file()
            or hashlib.sha256(marker.read_bytes()).hexdigest() != self.marker_sha256
        ):
            raise ValueError("G3 access marker is missing or does not match its receipt")
        marker_payload = json.loads(marker.read_bytes())
        if (
            marker_payload.get("schema") != "g3-access-marker-v1"
            or marker_payload.get("protocol_sha256") != self.protocol_sha256
            or marker_payload.get("source_split_sha256") != self.source_split_sha256
            or tuple(marker_payload.get("face_validity_source_ids", ()))
            != self.face_validity_source_ids
            or marker_payload.get("opened_at_utc") != self.opened_at_utc
        ):
            raise ValueError("G3 marker fields do not match the access receipt")
        expected = hashlib.sha256(
            ("g3-access-receipt-v1:" + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("G3 access receipt digest mismatch")


class G3AccessGate:
    """Atomically mark access, then load the complete held-out battery exactly once."""

    def __init__(self, protocol: G3Protocol, registry: SourceRegistry, marker: Path) -> None:
        protocol.verify()
        registry.verify_seal()
        if protocol.source_split_sha256 != registry.manifest_sha256():
            raise ValueError("G3 protocol/registry split mismatch")
        require_frozen_clean_checkout(protocol.frozen_commit)
        self.protocol = protocol
        self.registry = registry
        self.marker = marker
        self._opened = False

    def open_all_face_validity(
        self, loader: Callable[[str], bytes]
    ) -> tuple[tuple[tuple[str, bytes], ...], G3AccessReceipt]:
        if self._opened:
            raise RuntimeError("G3 face-validity gate is run once per driver instance")
        sources = tuple(
            source for source in self.registry.sources if source.role == FACE_VALIDITY
        )
        if not sources:
            raise ValueError("G3 registry has no face-validity sources")
        if any(source.payload_status != "AVAILABLE" for source in sources):
            raise ValueError("G3 face-validity payloads are not all available")
        _require_single_payload_face_validity(self.registry)
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        opened_at_utc = datetime.fromtimestamp(
            os.fstat(fd).st_ctime_ns / 1_000_000_000, timezone.utc
        ).isoformat()
        if datetime.fromisoformat(opened_at_utc) < datetime.fromisoformat(
            self.protocol.frozen_at_utc
        ):
            os.close(fd)
            raise ValueError("G3 gate creation predates the external P1 receipt")
        marker_payload = {
            "schema": "g3-access-marker-v1",
            "protocol_sha256": self.protocol.digest,
            "source_split_sha256": self.registry.manifest_sha256(),
            "face_validity_source_ids": tuple(source.source_id for source in sources),
            "protocol_frozen_at_utc": self.protocol.frozen_at_utc,
            "opened_at_utc": opened_at_utc,
        }
        marker_bytes = (_canonical(marker_payload) + "\n").encode()
        try:
            os.write(fd, marker_bytes)
        finally:
            os.close(fd)
        self._opened = True
        opened_payloads: list[tuple[str, bytes]] = []
        for source in sources:
            payload = loader(source.source_id)
            if not isinstance(payload, bytes):
                raise TypeError("G3 held-out loader must return exact bytes")
            if hashlib.sha256(payload).hexdigest() != source.payload_sha256:
                raise ValueError(f"G3 held-out payload hash mismatch for {source.source_id}")
            opened_payloads.append((source.source_id, payload))
        receipt_payload = {
            "schema": "g3-access-receipt-v1",
            "protocol_sha256": self.protocol.digest,
            "source_split_sha256": self.registry.manifest_sha256(),
            "face_validity_source_ids": tuple(source.source_id for source in sources),
            "opened_payload_sha256s": tuple(
                (source_id, hashlib.sha256(payload).hexdigest())
                for source_id, payload in opened_payloads
            ),
            "opened_at_utc": opened_at_utc,
            "marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        }
        digest = hashlib.sha256(
            ("g3-access-receipt-v1:" + _canonical(receipt_payload)).encode()
        ).hexdigest()
        receipt = G3AccessReceipt(
            **receipt_payload, digest=digest, _authority=_G3_ACCESS_AUTHORITY
        )
        receipt.verify(self.protocol, self.registry, self.marker)
        return tuple(opened_payloads), receipt


def prediction_to_json(prediction: G3PredictionArtifact) -> str:
    prediction.verify()
    return _canonical({**prediction._payload(), "digest": prediction.digest})


def prediction_from_json(
    text: str,
    *,
    registry: SourceRegistry,
    runs: tuple[PredictionRun, ...],
    service: str | None = None,
    metric: str | None = None,
) -> G3PredictionArtifact:
    """Contextually reload a prediction by re-deriving it from sealed runtime runs."""
    payload = json.loads(text)
    required = {
        field.name
        for field in dataclasses.fields(G3PredictionArtifact)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("G3 prediction JSON fields do not match its schema")
    if payload["kind"] == "published-curve":
        prediction = freeze_curve_prediction(
            registry=registry,
            source_id=payload["source_id"],
            selector_sha256=payload["selector_sha256"],
            x_unit="requests/s",
            y_unit="ms",
            runs=runs,
        )
    elif payload["kind"] == "rt-shape" and service is not None and metric is not None:
        prediction = freeze_rt_prediction(
            registry=registry,
            source_id=payload["source_id"],
            service=service,
            metric=metric,
            runs=runs,
        )
    else:
        raise ValueError("RT prediction reload requires its frozen service and metric")
    if prediction_to_json(prediction) != text:
        raise ValueError("G3 prediction JSON does not match the re-executed run closure")
    return prediction


def protocol_to_json(protocol: G3Protocol) -> str:
    protocol.verify()
    return _canonical({**protocol._payload(), "digest": protocol.digest})


def protocol_from_json(
    text: str,
    *,
    registration_receipt: P1RegistrationReceipt,
    predictions: tuple[G3PredictionArtifact, ...],
) -> G3Protocol:
    """Contextually reload a protocol through its exact external P1 receipt."""
    payload = json.loads(text)
    required = {
        field.name
        for field in dataclasses.fields(G3Protocol)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("G3 protocol JSON fields do not match its schema")
    protocol = freeze_protocol(
        registration_receipt=registration_receipt,
        source_split_sha256=payload["source_split_sha256"],
        rt_quantiles=tuple(payload["rt_quantiles"]),
        curve_log_ratio_abs_band=payload["curve_log_ratio_abs_band"],
        rt_log_quantile_abs_band=payload["rt_log_quantile_abs_band"],
        evidence_selectors=tuple(tuple(item) for item in payload["evidence_selectors"]),
        predictions=predictions,
    )
    if protocol_to_json(protocol) != text:
        raise ValueError("G3 protocol JSON is not canonical")
    return protocol


def access_receipt_to_json(receipt: G3AccessReceipt) -> str:
    return _canonical({**receipt._payload(), "digest": receipt.digest})


def access_receipt_from_json(
    text: str,
    protocol: G3Protocol,
    registry: SourceRegistry,
    marker: Path,
) -> G3AccessReceipt:
    payload = json.loads(text)
    required = {
        field.name
        for field in dataclasses.fields(G3AccessReceipt)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("G3 access-receipt JSON fields do not match its schema")
    for name in ("face_validity_source_ids",):
        payload[name] = tuple(payload[name])
    payload["opened_payload_sha256s"] = tuple(
        tuple(item) for item in payload["opened_payload_sha256s"]
    )
    receipt = G3AccessReceipt(**payload, _authority=_G3_ACCESS_AUTHORITY)
    receipt.verify(protocol, registry, marker)
    if access_receipt_to_json(receipt) != text:
        raise ValueError("G3 access-receipt JSON is not canonical")
    return receipt


def result_to_json(result: G3Result) -> str:
    result.verify()
    return _canonical({**result._payload(), "digest": result.digest})


def result_from_json(
    text: str,
    *,
    protocol: G3Protocol,
    registry: SourceRegistry,
    access_receipt: G3AccessReceipt | None = None,
    marker: Path | None = None,
    expected_residuals: tuple[Residual, ...] | None = None,
) -> G3Result:
    """Contextually reload and fully recompute an authoritative G3 result."""
    payload = json.loads(text)
    required = {
        field.name
        for field in dataclasses.fields(G3Result)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("G3 result JSON fields do not match its schema")
    if payload["status"] == "UNAVAILABLE":
        if expected_residuals:
            raise ValueError("UNAVAILABLE G3 result cannot accept evaluated residuals")
        result = unavailable_result(protocol, registry)
    else:
        if access_receipt is None or marker is None:
            raise ValueError("evaluated G3 result reload requires access receipt and marker")
        if expected_residuals is None:
            raise ValueError(
                "evaluated G3 result reload requires source-evaluator-issued residuals"
            )
        for residual in expected_residuals:
            residual.verify(protocol)
        if payload["residuals"] != json.loads(_canonical(expected_residuals)):
            raise ValueError(
                "G3 result JSON residuals do not match contextual source evaluation"
            )
        result = result_from_residuals(
            protocol, registry, expected_residuals, access_receipt, marker
        )
    if result_to_json(result) != text:
        raise ValueError("G3 result JSON does not match contextual recomputation")
    return result


def export_residuals(result: G3Result, path: Path) -> None:
    path.write_text(result_to_json(result), encoding="utf-8")
