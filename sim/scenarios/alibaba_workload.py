"""Typed, fail-closed Alibaba calibration-informed workload construction.

The retained ingestion exports are deliberately compact: RT-fit and scan payloads retain
enough fields to recompute their seals, while the call-graph sample omits raw row ids and
the root-rate summary omits the full root-service list.  Those two compact exports cannot
be re-issued as :mod:`sim.scenarios.alibaba` authority objects.  This module therefore
uses the exact SHA-256 of both immutable export files as the trust root, recomputes every
scan and RT-fit seal whose full payload is retained, and cross-checks the compact
call-sample/rate seals and their parent links against pinned values.  Any byte, registry,
seal, count, seed, selector, or shape drift fails closed.

The resulting :class:`CalibrationInformedWorkload` is not empirical.  Alibaba RT is a
service-instance x 60-second *interval-mean response-time* estimand, not CPU service
time.  Mapping a selected RT fit into ``ServiceSpec.service_time`` therefore applies an
explicit synthetic attribution factor, and CPU/memory/image/edge payload parameters are
an explicit deterministic development profile.  The calibrated evidence supplies only:

* fitted lognormal distribution shape and relative scale for selected services;
* sampled per-template call-tree structure (blocking fan-out/fan-in);
* the root-rate temporal shape.

That boundary is why the only authorized label is ``calibration-informed``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sim.scenarios.alibaba import CALL_GRAPH_COMPONENT, METRICS_COMPONENT, V2021
from sim.scenarios.sources import CALIBRATION, SourceRegistry, load_source_registry
from sim.spec import (
    ArrivalSpec,
    CallEdgeSpec,
    CallSpec,
    DistributionSpec,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    det,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "experiments" / "alibaba_ingestion_20260809" / "calibration.json"
CALL_GRAPH_SAMPLE_PATH = (
    REPO_ROOT / "experiments" / "alibaba_ingestion_20260809" / "call_graph_sample.json"
)

# Exact-byte trust roots for the already-produced, quarantined calibration exports.
CALIBRATION_SHA256 = "3c72cdc206c39f22ee2a92e0fafe80a8b1819d52d658426a3df67d32665364b3"
CALL_GRAPH_SAMPLE_SHA256 = "5bad0b52a9a0a5176035ecec70ca80a29dd23a4920c1988e7cd7c1b8691fde97"
SOURCE_REGISTRY_SHA256 = "81e1c5d9035da14e3ea0ce6ff351c57b43d2ce86669e3f23c5cb61f3380875e9"
METRICS_SCAN_SEAL = "a1b5d40496c2bf7f120ab245478684be4260a59c9997f194253849eed072d025"
CALL_GRAPH_SCAN_SEAL = "7c9d7bca4594a7489d497a019912f7b7f304fed2aff8ece339ebd108a848e325"
ROOT_RATE_SEAL = "97eef53a72cd3045918d39ed966a887cb34ee873a2fb173d8aacaa85b1bf68f2"
CALL_GRAPH_SAMPLE_SEAL = "751a2f2486960c20eeb35c1e1695d2dcc90c75be78c8caeb3b606303bfc7621d"
ALIBABA_CODE_SHA256 = "88faaed3b330798c9fc76cc7c88650b6da3b564637c2b21cc48d9021f21425f8"
STREAM_CODE_VERSION = (
    "alibaba-complete-member-streaming-scan-v1:"
    "858630db751752bf254e519b34da8948836dd9baeebe40714a53b4b1c0ba2b39"
)
TRACE_SAMPLE_SEED = 6399959053302358801
ROOT_SEED = 20260809
BUILDER_VERSION = "alibaba-calibration-informed-workload-builder-v1"

_SCAN_TAG = "alibaba-v2021-complete-member-scan-v1:"
_FIT_TAG = "alibaba-rt-fit-v1:"
_BUILD_TAG = "alibaba-calibration-informed-workload-v1:"
_BUILD_AUTHORITY = object()
_RT_ESTIMAND = "service-instance-by-60s-interval-mean-response-time"
_RT_ACTIVITY_RULE = "paired-same-cell-family-mcr-strictly-positive-v1"
_RT_CENSORING_RULE = "active-zero-left-censored-below-1ms-v1"
_RT_SEMANTIC = (
    "distribution of service-instance-by-60s-interval mean response times; "
    "not per-request latency and not direct CPU service time"
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
            return [jsonable(entry) for entry in item]
        if isinstance(item, Mapping):
            return {key: jsonable(entry) for key, entry in item.items()}
        return item

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lower_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_exact_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing sealed {label}: {path}")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"sealed {label} exact-byte SHA-256 mismatch")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"sealed {label} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"sealed {label} must contain a JSON object")
    return value


@dataclass(frozen=True, order=True)
class RetainedRtFit:
    service: str
    metric: str
    mu: float
    sigma: float
    positive_count: int
    censored_count: int
    seal: str

    @property
    def p95_s(self) -> float:
        return math.exp(self.mu + 1.6448536269514722 * self.sigma)


@dataclass(frozen=True, order=True)
class RetainedCall:
    trace_id: str
    rpc_id: str
    parent_rpc_id: str | None
    downstream: str | None
    rpc_type: str


@dataclass(frozen=True)
class SealedAlibabaCalibration:
    """Authority-bearing, exact-byte-verified view of the compact retained exports."""

    registry_sha256: str
    metrics_content_sha256: str
    call_graph_content_sha256: str
    metrics_scan_seal: str
    call_graph_scan_seal: str
    root_rate_seal: str
    call_graph_sample_seal: str
    trace_sample_seed: int
    rt_fits: tuple[RetainedRtFit, ...]
    rate_points: tuple[tuple[float, float], ...]
    rate_support_end_s: float
    rate_target_positive_mean_rps: float
    calls: tuple[RetainedCall, ...]
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify_authority(self) -> None:
        if self._authority is not _BUILD_AUTHORITY:
            raise ValueError("Alibaba calibration view must be issued by the sealed loader")
        if len(self.rt_fits) != 3734:
            raise ValueError("Alibaba calibration must retain exactly 3,734 RT fits")
        if len(self.calls) != 29933:
            raise ValueError("Alibaba call-graph sample must retain exactly 29,933 calls")


@dataclass(frozen=True)
class AlibabaWorkloadBuildConfig:
    """Every modelling choice applied after the sealed evidence, including its seed."""

    seed: int
    template_count: int = 3
    max_services: int = 8
    max_calls_per_template: int = 9
    rt_to_service_time_fraction: float = 0.45
    arrival_rate_scale: float = 0.30
    arrival_start_s: float = 4.0
    arrival_support_s: float = 48.0
    flash_start_s: float | None = None
    flash_end_s: float | None = None
    flash_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("Alibaba workload builder seed must be an explicit int")
        for name in ("template_count", "max_services", "max_calls_per_template"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.template_count <= 8:
            raise ValueError("template_count must be in [1, 8]")
        if self.max_services < 3 or self.max_calls_per_template < 4:
            raise ValueError("builder needs >=3 services and >=4 calls per template")
        for name in (
            "rt_to_service_time_fraction",
            "arrival_rate_scale",
            "arrival_start_s",
            "arrival_support_s",
            "flash_multiplier",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and > 0")
        flash = (self.flash_start_s, self.flash_end_s)
        if (flash[0] is None) != (flash[1] is None):
            raise ValueError("flash_start_s and flash_end_s must appear together")
        if flash[0] is not None:
            assert flash[1] is not None
            if not self.arrival_start_s < flash[0] < flash[1] < (
                self.arrival_start_s + self.arrival_support_s
            ):
                raise ValueError("flash interval must be strictly inside arrival support")
            if self.flash_multiplier <= 1.0:
                raise ValueError("a configured flash_multiplier must be > 1")
        elif self.flash_multiplier != 1.0:
            raise ValueError("flash_multiplier requires a flash interval")


@dataclass(frozen=True, order=True)
class SelectedTemplate:
    trace_id: str
    root_rpc_id: str
    call_rpc_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationInformedWorkload:
    """Typed proof that one exact ``WorkloadSpec`` was derived from sealed evidence."""

    config: AlibabaWorkloadBuildConfig
    workload: WorkloadSpec
    selected_templates: tuple[SelectedTemplate, ...]
    selected_fit_seals: tuple[tuple[str, str], ...]
    evidence_seals: tuple[tuple[str, str], ...]
    transform_config: tuple[tuple[str, str], ...]
    code_versions: tuple[str, ...]
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }

    def verify(
        self,
        registry: SourceRegistry,
        *,
        calibration_path: Path = CALIBRATION_PATH,
        call_graph_sample_path: Path = CALL_GRAPH_SAMPLE_PATH,
    ) -> None:
        if self._authority is not _BUILD_AUTHORITY:
            raise ValueError("calibration-informed workload must be issued by the typed builder")
        evidence = load_sealed_alibaba_calibration(
            registry,
            calibration_path=calibration_path,
            call_graph_sample_path=call_graph_sample_path,
        )
        derived = _derive(evidence, self.config)
        if (
            self.workload != derived.workload
            or self.selected_templates != derived.selected_templates
            or self.selected_fit_seals != derived.selected_fit_seals
            or self.evidence_seals != derived.evidence_seals
            or self.transform_config != derived.transform_config
            or self.code_versions != derived.code_versions
        ):
            raise ValueError("calibration-informed workload drifted from sealed evidence")
        expected = hashlib.sha256((_BUILD_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("calibration-informed workload builder seal mismatch")


def _verify_scan(scan: object, expected_seal: str, component: str) -> dict[str, object]:
    if not isinstance(scan, dict):
        raise ValueError(f"{component} scan must be an object")
    extras = (
        {"distinct_instances", "distinct_services", "timestamps_ms", "metric_counts"}
        if component == METRICS_COMPONENT
        else {"identified_root_services", "rpc_type_counts"}
    )
    payload = {
        key: value
        for key, value in scan.items()
        if key not in extras | {"seal"}
    }
    payload["facts"] = sorted(payload["facts"].items())
    payload["rejected_by_reason"] = [list(item) for item in payload["rejected_by_reason"]]
    computed = hashlib.sha256((_SCAN_TAG + _canonical(payload)).encode()).hexdigest()
    if scan.get("seal") != expected_seal or computed != expected_seal:
        raise ValueError(f"{component} complete-member scan seal mismatch")
    if (
        payload.get("schema") != "alibaba-v2021-complete-member-scan-v1"
        or payload.get("source_id") != V2021
        or payload.get("component") != component
        or payload.get("source_registry_sha256") != SOURCE_REGISTRY_SHA256
        or payload.get("truncation") != "none-complete-member-streamed"
        or payload.get("scan_code_version") != STREAM_CODE_VERSION
    ):
        raise ValueError(f"{component} scan typed provenance mismatch")
    return scan


def _rt_fit_payload(
    summary: dict[str, object], calibration: dict[str, object]
) -> dict[str, object]:
    declared = calibration["declared"]
    metrics_scan = calibration["metrics_scan"]
    assert isinstance(declared, dict) and isinstance(metrics_scan, dict)
    service = summary["service"]
    metric = summary["metric"]
    minimum_count = declared["minimum_active_cells"]
    params = summary["params_seconds"]
    if not isinstance(service, str) or not isinstance(metric, str) or not isinstance(params, dict):
        raise ValueError("RT-fit summary has invalid service/metric/params")
    config = {
        "service": service,
        "metric": metric,
        "minimum_count": minimum_count,
        "family": "lognormal",
        "censor_threshold_s": 0.001,
        "estimand": _RT_ESTIMAND,
        "activity_rule": _RT_ACTIVITY_RULE,
        "censoring_rule": _RT_CENSORING_RULE,
    }
    return {
        "schema": "alibaba-rt-fit-v1",
        "source_id": V2021,
        "source_content_sha256": metrics_scan["content_sha256"],
        "source_registry_sha256": SOURCE_REGISTRY_SHA256,
        "service": service,
        "metric": metric,
        "semantic": _RT_SEMANTIC,
        "estimand": _RT_ESTIMAND,
        "activity_rule": _RT_ACTIVITY_RULE,
        "censoring_rule": _RT_CENSORING_RULE,
        "family": "lognormal",
        "params_seconds": (("mu", params["mu"]), ("sigma", params["sigma"])),
        "positive_count": summary["positive_count"],
        "censored_count": summary["censored_count"],
        "rejected_count": sum(count for _, count in metrics_scan["rejected_by_reason"]),
        "row_selection": (
            f"service == {service!r}; metric == {metric!r}; "
            "estimand == service-instance-by-60s-interval-mean-response-time; "
            "unique paired same-cell MCR > 0; positive or documented left-censored zero"
        ),
        "fit_config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
        "fit_code_version": "censored-lognormal-coordinate-search-v1:" + ALIBABA_CODE_SHA256,
    }


def _verify_rt_fits(calibration: dict[str, object]) -> tuple[RetainedRtFit, ...]:
    raw_fits = calibration.get("rt_fits")
    cells = calibration.get("rt_cell_summaries")
    if not isinstance(raw_fits, list) or len(raw_fits) != 3734 or not isinstance(cells, list):
        raise ValueError("calibration must contain 3,734 RT fits and typed cell summaries")
    eligible = {
        (cell["service"], cell["metric"]): cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("eligible") is True
    }
    if len(eligible) != len(raw_fits):
        raise ValueError("eligible RT-cell summaries do not match the RT-fit count")
    out: list[RetainedRtFit] = []
    seen: set[tuple[str, str]] = set()
    for summary in raw_fits:
        if not isinstance(summary, dict):
            raise ValueError("RT-fit summary must be an object")
        payload = _rt_fit_payload(summary, calibration)
        computed = hashlib.sha256((_FIT_TAG + _canonical(payload)).encode()).hexdigest()
        seal = _lower_sha256("RT-fit seal", summary.get("seal"))
        if computed != seal:
            raise ValueError(
                f"RT-fit seal mismatch for {summary.get('service')}:{summary.get('metric')}"
            )
        service = payload["service"]
        metric = payload["metric"]
        assert isinstance(service, str) and isinstance(metric, str)
        key = (service, metric)
        if key in seen or key not in eligible:
            raise ValueError("RT fits must be unique and backed by an eligible cell summary")
        seen.add(key)
        cell = eligible[key]
        if (
            cell["duplicate_cells"] != 0
            or cell["unpaired_cells"] != 0
            or cell["positive_cells"] != summary["positive_count"]
            or cell["censored_cells"] != summary["censored_count"]
            or cell["active_cells"]
            != summary["positive_count"] + summary["censored_count"]
        ):
            raise ValueError("RT-fit cell-selection counts do not close")
        params = summary["params_seconds"]
        assert isinstance(params, dict)
        mu, sigma = params.get("mu"), params.get("sigma")
        if (
            not isinstance(mu, int | float)
            or isinstance(mu, bool)
            or not math.isfinite(mu)
            or not isinstance(sigma, int | float)
            or isinstance(sigma, bool)
            or not math.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("RT-fit lognormal parameters must be finite with sigma > 0")
        out.append(
            RetainedRtFit(
                service,
                metric,
                float(mu),
                float(sigma),
                int(summary["positive_count"]),
                int(summary["censored_count"]),
                seal,
            )
        )
    if tuple(out) != tuple(sorted(out)):
        raise ValueError("RT fits must be canonically sorted by service and metric")
    return tuple(out)


def _verify_rate_shape(calibration: dict[str, object]) -> tuple[tuple[float, float], ...]:
    raw = calibration.get("root_rate_shape")
    if not isinstance(raw, dict) or raw.get("seal") != ROOT_RATE_SEAL:
        raise ValueError("root-rate compact artifact seal mismatch")
    points = tuple(tuple(point) for point in raw.get("points_relative_s", ()))
    if len(points) != 30:
        raise ValueError("root-rate shape must contain 30 complete 60-second cells")
    checked = tuple((float(at), float(rate)) for at, rate in points)
    if [at for at, _ in checked] != [60.0 * index for index in range(30)]:
        raise ValueError("root-rate shape cells must be consecutive and ordered")
    if any(not math.isfinite(rate) or rate <= 0 for _, rate in checked):
        raise ValueError("retained root-rate cells must be finite and positive")
    positive_mean = math.fsum(rate for _, rate in checked) / len(checked)
    if not math.isclose(positive_mean, 50.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("root-rate shape does not close to its synthetic 50 rps level")
    arrival = raw.get("arrival_spec")
    expected_arrival = {
        "kind": "poisson",
        "start_s": 0.0,
        "rate_per_s": checked[0][1],
        "rate_schedule": [list(point) for point in checked[1:]] + [[1800.0, 0.0]],
    }
    if arrival != expected_arrival or raw.get("support_end_relative_s") != 1800.0:
        raise ValueError("root-rate ArrivalSpec summary does not match the sealed points")
    return checked


def _verify_calls(
    calibration: dict[str, object], sample: dict[str, object]
) -> tuple[RetainedCall, ...]:
    meta = calibration.get("call_graph_sample")
    scan = calibration.get("call_graph_scan")
    if not isinstance(meta, dict) or not isinstance(scan, dict):
        raise ValueError("call-graph calibration metadata is missing")
    expected = {
        "seal": CALL_GRAPH_SAMPLE_SEAL,
        "parent_scan_seal": CALL_GRAPH_SCAN_SEAL,
        "selection_method": "streaming-bottom-k-trace-sha256-v1",
        "selection_seed": TRACE_SAMPLE_SEED,
        "requested_trace_count": 1000,
        "calls": 29933,
        "traces_with_retained_calls": 991,
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("call-graph sample metadata/seal chain mismatch")
    if (
        sample.get("seal") != CALL_GRAPH_SAMPLE_SEAL
        or sample.get("parent_scan_seal") != CALL_GRAPH_SCAN_SEAL
        or sample.get("selection_seed") != TRACE_SAMPLE_SEED
        or sample.get("requested_trace_count") != 1000
        or sample.get("content_sha256") != scan.get("content_sha256")
    ):
        raise ValueError("call-graph compact export does not match its sealed metadata")
    raw_calls = sample.get("calls")
    if not isinstance(raw_calls, list) or len(raw_calls) != 29933:
        raise ValueError("call-graph compact export must contain 29,933 calls")
    out: list[RetainedCall] = []
    seen: set[tuple[str, str]] = set()
    previous: tuple[str, str] | None = None
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise ValueError("call-graph entries must be objects")
        trace_id, rpc_id = raw.get("trace_id"), raw.get("rpc_id")
        parent, downstream, rpc_type = (
            raw.get("parent_rpc_id"),
            raw.get("downstream"),
            raw.get("rpc_type"),
        )
        if (
            not isinstance(trace_id, str)
            or not trace_id
            or not isinstance(rpc_id, str)
            or not rpc_id
            or (parent is not None and not isinstance(parent, str))
            or (downstream is not None and not isinstance(downstream, str))
            or not isinstance(rpc_type, str)
            or not rpc_type
        ):
            raise ValueError("call-graph compact entry has invalid typed fields")
        key = (trace_id, rpc_id)
        if key in seen or (previous is not None and key < previous):
            raise ValueError("call-graph sample calls must be unique and canonically sorted")
        seen.add(key)
        previous = key
        out.append(RetainedCall(trace_id, rpc_id, parent, downstream, rpc_type))
    if len({call.trace_id for call in out}) != 991:
        raise ValueError("call-graph sample retained-trace count mismatch")
    return tuple(out)


def load_sealed_alibaba_calibration(
    registry: SourceRegistry | None = None,
    *,
    calibration_path: Path = CALIBRATION_PATH,
    call_graph_sample_path: Path = CALL_GRAPH_SAMPLE_PATH,
) -> SealedAlibabaCalibration:
    """Load and revalidate the exact sealed chunk-0 calibration, fail closed."""
    active_registry = registry if registry is not None else load_source_registry()
    active_registry.verify_seal()
    if active_registry.manifest_sha256() != SOURCE_REGISTRY_SHA256:
        raise ValueError("Alibaba calibration source-registry seal mismatch")
    if _sha256_bytes((Path(__file__).parent / "alibaba.py").read_bytes()) != ALIBABA_CODE_SHA256:
        raise ValueError("Alibaba fit implementation drifted from the sealed calibration")
    calibration = _load_exact_json(calibration_path, CALIBRATION_SHA256, "calibration report")
    sample = _load_exact_json(
        call_graph_sample_path, CALL_GRAPH_SAMPLE_SHA256, "call-graph sample"
    )
    if (
        calibration.get("schema") != "alibaba-calibration-ingestion-v1"
        or calibration.get("corpus_id") != "alibaba-calibration-corpus-v1"
        or calibration.get("root_seed") != ROOT_SEED
        or calibration.get("code_version") != STREAM_CODE_VERSION
        or calibration.get("source_registry_sha256") != SOURCE_REGISTRY_SHA256
    ):
        raise ValueError("Alibaba calibration report provenance mismatch")
    declared = calibration.get("declared")
    if not isinstance(declared, dict) or declared.get("source_id") != V2021:
        raise ValueError("Alibaba calibration declaration is missing")
    source = active_registry.require_role(V2021, CALIBRATION)
    registered = {
        item.component: {
            "component": item.component,
            "parser_input": item.parser_input,
            "sha256": item.sha256,
            "bytes": item.bytes,
            "archive_path": item.archive_path,
            "archive_sha256": item.archive_sha256,
            "archive_bytes": item.archive_bytes,
        }
        for item in source.payload_components
    }
    declared_components = declared.get("components")
    if (
        declared.get("role") != CALIBRATION
        or declared.get("minimum_active_cells") != 20
        or declared.get("trace_sample_seed") != TRACE_SAMPLE_SEED
        or declared.get("sample_traces") != 1000
        or declared.get("truncation") != "none-complete-member-streamed"
        or not isinstance(declared_components, list)
        or {item["component"]: item for item in declared_components} != registered
    ):
        raise ValueError("Alibaba calibration declaration/registry binding mismatch")
    metrics_scan = _verify_scan(calibration.get("metrics_scan"), METRICS_SCAN_SEAL, "metrics")
    call_scan = _verify_scan(
        calibration.get("call_graph_scan"), CALL_GRAPH_SCAN_SEAL, "call_graph"
    )
    metrics_digest = _lower_sha256("metrics content", metrics_scan.get("content_sha256"))
    call_digest = _lower_sha256("call-graph content", call_scan.get("content_sha256"))
    active_registry.require_component(V2021, CALIBRATION, METRICS_COMPONENT, metrics_digest)
    active_registry.require_component(V2021, CALIBRATION, CALL_GRAPH_COMPONENT, call_digest)
    fits = _verify_rt_fits(calibration)
    points = _verify_rate_shape(calibration)
    calls = _verify_calls(calibration, sample)
    evidence = SealedAlibabaCalibration(
        SOURCE_REGISTRY_SHA256,
        metrics_digest,
        call_digest,
        METRICS_SCAN_SEAL,
        CALL_GRAPH_SCAN_SEAL,
        ROOT_RATE_SEAL,
        CALL_GRAPH_SAMPLE_SEAL,
        TRACE_SAMPLE_SEED,
        fits,
        points,
        1800.0,
        50.0,
        calls,
        _BUILD_AUTHORITY,
    )
    evidence.verify_authority()
    return evidence


@dataclass(frozen=True)
class _Candidate:
    trace_id: str
    root_rpc_id: str
    call_rpc_ids: tuple[str, ...]
    calls_by_rpc: dict[str, RetainedCall] = dataclasses.field(compare=False, repr=False)
    children_by_rpc: dict[str, tuple[str, ...]] = dataclasses.field(compare=False, repr=False)

    @property
    def services(self) -> set[str]:
        return {
            call.downstream
            for rpc_id in self.call_rpc_ids
            if (call := self.calls_by_rpc[rpc_id]).downstream is not None
        }


def _rpc_sort_key(rpc_id: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in rpc_id.split("."))
    except ValueError as exc:
        raise ValueError(f"sampled call has malformed rpc id {rpc_id!r}") from exc


def _candidate_templates(
    evidence: SealedAlibabaCalibration, config: AlibabaWorkloadBuildConfig
) -> tuple[_Candidate, ...]:
    fitted_services = {fit.service for fit in evidence.rt_fits}
    by_trace: dict[str, list[RetainedCall]] = defaultdict(list)
    for call in evidence.calls:
        if call.downstream in fitted_services:
            by_trace[call.trace_id].append(call)
    candidates: list[_Candidate] = []
    for trace_id, calls in sorted(by_trace.items()):
        by_rpc = {call.rpc_id: call for call in calls}
        children: dict[str | None, list[str]] = defaultdict(list)
        parent_by_rpc: dict[str, str | None] = {}
        for rpc_id in by_rpc:
            parts = rpc_id.split(".")
            parent = None
            # Contract un-fitted call stages onto the nearest fitted ancestor.  This is
            # a structural transform only; no missing service is invented.
            for count in range(len(parts) - 1, 0, -1):
                candidate = ".".join(parts[:count])
                if candidate in by_rpc:
                    parent = candidate
                    break
            children[parent].append(rpc_id)
            parent_by_rpc[rpc_id] = parent
        for values in children.values():
            values.sort(key=_rpc_sort_key)
        for root in children[None]:
            queue = [(root, 0)]
            selected: list[str] = []
            deepest = 0
            widest = 0
            while queue and len(selected) < config.max_calls_per_template:
                rpc_id, depth = queue.pop(0)
                selected.append(rpc_id)
                deepest = max(deepest, depth)
                direct = children[rpc_id]
                widest = max(widest, len(direct))
                queue.extend((child, depth + 1) for child in direct)
            services = {by_rpc[rpc_id].downstream for rpc_id in selected}
            recursive_path = False
            included = set(selected)
            for rpc_id in selected:
                path_services: set[str | None] = set()
                current: str | None = rpc_id
                while current is not None and current in included:
                    downstream = by_rpc[current].downstream
                    if downstream in path_services:
                        recursive_path = True
                        break
                    path_services.add(downstream)
                    current = parent_by_rpc[current]
                if recursive_path:
                    break
            if (
                len(selected) >= 4
                and deepest >= 2
                and widest >= 2
                and len(services) >= 3
                and not recursive_path
            ):
                candidates.append(
                    _Candidate(
                        trace_id,
                        root,
                        tuple(selected),
                        by_rpc,
                        {key: tuple(value) for key, value in children.items() if key is not None},
                    )
                )
    if not candidates:
        raise ValueError("call-graph sample has no typed multi-service fan-out candidates")
    return tuple(candidates)


def _select_templates(
    evidence: SealedAlibabaCalibration, config: AlibabaWorkloadBuildConfig
) -> tuple[_Candidate, ...]:
    ranked = sorted(
        _candidate_templates(evidence, config),
        key=lambda item: hashlib.sha256(
            f"{config.seed}:alibaba-template-v1:{item.trace_id}:{item.root_rpc_id}".encode()
        ).digest(),
    )
    def choose(
        start: int, selected: tuple[_Candidate, ...], services: frozenset[str]
    ) -> tuple[_Candidate, ...] | None:
        if len(selected) == config.template_count:
            return selected
        for index in range(start, len(ranked)):
            candidate = ranked[index]
            merged = services | candidate.services
            if (
                len(merged) > config.max_services
                or candidate.trace_id in {item.trace_id for item in selected}
                or (selected and not candidate.services - services)
            ):
                continue
            result = choose(index + 1, (*selected, candidate), frozenset(merged))
            if result is not None:
                return result
        return None

    selected = choose(0, (), frozenset())
    if selected is None:
        raise ValueError("seed/config cannot select enough distinct call-graph templates")
    return selected


def _select_fit(
    service: str,
    rpc_types: Counter[str],
    fits_by_service: dict[str, tuple[RetainedRtFit, ...]],
) -> RetainedRtFit:
    families = {
        "http": ("HTTP_RT", "PROVIDERRPC_RT", "CONSUMERRPC_RT", "CONSUMERMQ_RT"),
        "mq": ("CONSUMERMQ_RT", "PROVIDERRPC_RT", "CONSUMERRPC_RT", "HTTP_RT"),
        "rpc": ("PROVIDERRPC_RT", "CONSUMERRPC_RT", "HTTP_RT", "CONSUMERMQ_RT"),
        "mc": ("CONSUMERRPC_RT", "PROVIDERRPC_RT", "CONSUMERMQ_RT", "HTTP_RT"),
        "db": ("CONSUMERRPC_RT", "PROVIDERRPC_RT", "CONSUMERMQ_RT", "HTTP_RT"),
        "userdefined": ("PROVIDERRPC_RT", "CONSUMERRPC_RT", "HTTP_RT", "CONSUMERMQ_RT"),
    }
    dominant = min(rpc_types, key=lambda kind: (-rpc_types[kind], kind))
    preference = families.get(dominant, families["rpc"])
    by_metric = {fit.metric: fit for fit in fits_by_service[service]}
    for metric in preference:
        if metric in by_metric:
            return by_metric[metric]
    return min(fits_by_service[service])


def _resource_profile(service: str) -> tuple[int, int, int, int, int]:
    """Synthetic development resources, deterministically keyed by opaque service id."""
    digest = hashlib.sha256(("alibaba-resource-profile-v1:" + service).encode()).digest()
    cpu_request = 80 + 40 * (digest[0] % 5)
    cpu_demand = int(round(cpu_request * (2.2 + 0.35 * (digest[1] % 5))))
    mem_request = 192 + 96 * (digest[2] % 6)
    image_mb = 12 + 6 * (digest[3] % 7)
    initial_replicas = 2 + (digest[4] % 2)
    return cpu_request, cpu_demand, mem_request, image_mb, initial_replicas


def _edge_payload(rpc_type: str) -> tuple[float, float]:
    return {
        "http": (2.0, 6.0),
        "rpc": (1.5, 2.5),
        "mq": (3.0, 0.5),
        "mc": (0.8, 1.2),
        "db": (1.0, 4.0),
        "userdefined": (1.0, 1.0),
    }.get(rpc_type, (1.0, 1.0))


def _arrival(
    evidence: SealedAlibabaCalibration,
    config: AlibabaWorkloadBuildConfig,
    weight: float,
) -> ArrivalSpec:
    support_scale = config.arrival_support_s / evidence.rate_support_end_s
    points = {
        config.arrival_start_s + at * support_scale: rate * config.arrival_rate_scale * weight
        for at, rate in evidence.rate_points
    }
    support_end = config.arrival_start_s + config.arrival_support_s
    points[support_end] = 0.0
    if config.flash_start_s is not None:
        assert config.flash_end_s is not None

        def base_rate(at: float) -> float:
            prior = max(time for time in points if time <= at)
            return points[prior]

        points[config.flash_start_s] = base_rate(config.flash_start_s)
        points[config.flash_end_s] = base_rate(config.flash_end_s)
    ordered = sorted(points)
    schedule: list[tuple[float, float]] = []
    for at in ordered[1:]:
        rate = points[at]
        if (
            config.flash_start_s is not None
            and config.flash_end_s is not None
            and config.flash_start_s <= at < config.flash_end_s
        ):
            rate *= config.flash_multiplier
        schedule.append((at, rate))
    initial = points[ordered[0]]
    return ArrivalSpec("poisson", initial, start_s=ordered[0], rate_schedule=tuple(schedule))


def _derive(
    evidence: SealedAlibabaCalibration, config: AlibabaWorkloadBuildConfig
) -> CalibrationInformedWorkload:
    evidence.verify_authority()
    selected = _select_templates(evidence, config)
    services = sorted({service for candidate in selected for service in candidate.services})
    service_ids = {service: index for index, service in enumerate(services)}
    rpc_types: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in selected:
        for rpc_id in candidate.call_rpc_ids:
            call = candidate.calls_by_rpc[rpc_id]
            assert call.downstream is not None
            rpc_types[call.downstream][call.rpc_type] += 1
    fits_by_service: dict[str, tuple[RetainedRtFit, ...]] = {}
    for fit in evidence.rt_fits:
        fits_by_service.setdefault(fit.service, ())
        fits_by_service[fit.service] = (*fits_by_service[fit.service], fit)
    selected_fits = {
        service: _select_fit(service, rpc_types[service], fits_by_service) for service in services
    }
    service_specs: list[ServiceSpec] = []
    for service in services:
        fit = selected_fits[service]
        cpu_request, cpu_demand, mem_request, image_mb, initial_replicas = _resource_profile(
            service
        )
        service_specs.append(
            ServiceSpec(
                id=service_ids[service],
                name="ali-" + service[:12],
                app=0,
                cpu_request=cpu_request,
                mem_request=mem_request,
                service_time=DistributionSpec(
                    "lognormal",
                    {
                        "mu": fit.mu + math.log(config.rt_to_service_time_fraction),
                        "sigma": fit.sigma,
                    },
                ),
                cpu_demand=cpu_demand,
                # Sampled traces can revisit one service along a blocking path.  Four
                # server slots prevent that observed recursion from becoming a
                # single-slot simulator deadlock while retaining queueing/HPA pressure.
                concurrency=4,
                cold_start=det(0.15 + 0.05 * (service_ids[service] % 5)),
                image_mb=image_mb,
                initial_replicas=initial_replicas,
                min_replicas=initial_replicas,
                max_replicas=initial_replicas + 4,
                category="calibration-informed-development",
            )
        )

    def call_spec(candidate: _Candidate, rpc_id: str, included: set[str]) -> CallSpec:
        call = candidate.calls_by_rpc[rpc_id]
        assert call.downstream is not None
        edges = []
        for child_id in candidate.children_by_rpc.get(rpc_id, ()):
            if child_id not in included:
                continue
            child = candidate.calls_by_rpc[child_id]
            request_kb, response_kb = _edge_payload(child.rpc_type)
            edges.append(
                CallEdgeSpec(
                    child=call_spec(candidate, child_id, included),
                    probability=1.0,
                    # The retained graph has no timeout/retry policy evidence.  Leaving
                    # both unset avoids inventing failure semantics (and avoids turning
                    # a long-tailed calibrated draw into an artificial retry storm).
                    timeout_s=None,
                    max_retries=0,
                    request_kb=request_kb,
                    response_kb=response_kb,
                )
            )
        return CallSpec(service_ids[call.downstream], tuple(edges), mode="blocking")

    weights_raw = [len(candidate.call_rpc_ids) for candidate in selected]
    weight_total = math.fsum(weights_raw)
    templates: list[TemplateSpec] = []
    for template_id, (candidate, raw_weight) in enumerate(zip(selected, weights_raw, strict=True)):
        included = set(candidate.call_rpc_ids)
        root = call_spec(candidate, candidate.root_rpc_id, included)

        def critical_p95(call: CallSpec) -> float:
            service_name = services[call.service]
            own = selected_fits[service_name].p95_s * config.rt_to_service_time_fraction
            return own + max(
                (critical_p95(edge.child) for edge in call.children), default=0.0
            )

        templates.append(
            TemplateSpec(
                id=template_id,
                name=f"ali-trace-{template_id}",
                root=root,
                arrival=_arrival(evidence, config, raw_weight / weight_total),
                slo_latency_s=max(1.5, 8.0 * critical_p95(root)),
            )
        )
    workload = WorkloadSpec(tuple(service_specs), tuple(templates))
    selected_templates = tuple(
        SelectedTemplate(item.trace_id, item.root_rpc_id, item.call_rpc_ids) for item in selected
    )
    selected_fit_seals = tuple(
        sorted((service, fit.seal) for service, fit in selected_fits.items())
    )
    evidence_seals = (
        ("call-graph-complete-member-scan", evidence.call_graph_scan_seal),
        ("call-graph-sample", evidence.call_graph_sample_seal),
        ("metrics-complete-member-scan", evidence.metrics_scan_seal),
        ("root-rate-shape", evidence.root_rate_seal),
    )
    transform_config = tuple(
        sorted(
            (
                ("arrival_rate_scale", repr(config.arrival_rate_scale)),
                ("arrival_start_s", repr(config.arrival_start_s)),
                ("arrival_support_s", repr(config.arrival_support_s)),
                ("call_graph_transform", "nearest-fitted-ancestor-contraction-bfs-v1"),
                ("flash_end_s", repr(config.flash_end_s)),
                ("flash_multiplier", repr(config.flash_multiplier)),
                ("flash_start_s", repr(config.flash_start_s)),
                ("max_calls_per_template", str(config.max_calls_per_template)),
                ("max_services", str(config.max_services)),
                ("resource_profile", "synthetic-development-resource-profile-v1"),
                ("rt_semantic", _RT_SEMANTIC),
                ("rt_to_service_time_fraction", repr(config.rt_to_service_time_fraction)),
                ("template_count", str(config.template_count)),
            )
        )
    )
    code_versions = (BUILDER_VERSION, STREAM_CODE_VERSION)
    unsealed = CalibrationInformedWorkload(
        config,
        workload,
        selected_templates,
        selected_fit_seals,
        evidence_seals,
        transform_config,
        code_versions,
        "",
        _BUILD_AUTHORITY,
    )
    digest = hashlib.sha256((_BUILD_TAG + _canonical(unsealed._payload())).encode()).hexdigest()
    return dataclasses.replace(unsealed, digest=digest)


def build_calibration_informed_workload(
    config: AlibabaWorkloadBuildConfig,
    registry: SourceRegistry | None = None,
    *,
    calibration_path: Path = CALIBRATION_PATH,
    call_graph_sample_path: Path = CALL_GRAPH_SAMPLE_PATH,
) -> CalibrationInformedWorkload:
    """Derive and seal one exact calibration-informed (never empirical) workload."""
    active_registry = registry if registry is not None else load_source_registry()
    evidence = load_sealed_alibaba_calibration(
        active_registry,
        calibration_path=calibration_path,
        call_graph_sample_path=call_graph_sample_path,
    )
    result = _derive(evidence, config)
    result.verify(
        active_registry,
        calibration_path=calibration_path,
        call_graph_sample_path=call_graph_sample_path,
    )
    return result
