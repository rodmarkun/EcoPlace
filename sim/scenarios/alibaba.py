"""Version-specific Alibaba ingestion and calibration-safe transforms (TODO 4.1).

No parser downloads data.  v2021 supplies calibration observations; v2022 is held-out
shape evidence only.  Alibaba RT is retained as a response-time target—not silently
reinterpreted as CPU service demand, which would double-count queueing/network/children.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sim.scenarios.sources import CALIBRATION, FACE_VALIDITY, SourceRegistry
from sim.spec import ArrivalSpec

V2021 = "alibaba-microservices-v2021"
V2022 = "alibaba-microservices-v2022"
METRICS_COMPONENT = "metrics"
CALL_GRAPH_COMPONENT = "call_graph"
CALLGRAPH_SAMPLE_FRACTION_V2021 = 0.005
_MISSING = {"", "NAN", "(?)"}
_FIT_TAG = "alibaba-rt-fit-v1:"
_RATE_SHAPE_TAG = "alibaba-absolute-root-rate-shape-v1:"
_CORPUS_TAG = "alibaba-ingestion-corpus-v1:"
_CORPUS_AUTHORITY = object()
_FIT_AUTHORITY = object()
_RATE_AUTHORITY = object()
_NO_SAMPLE_SEED = None
_RT_ESTIMAND = "service-instance-by-60s-interval-mean-response-time"
_RT_ACTIVITY_RULE = "paired-same-cell-family-mcr-strictly-positive-v1"
_RT_CENSORING_RULE = "active-zero-left-censored-below-1ms-v1"
_RATE_AGGREGATION = "sum-root-instances-and-services-per-timestamp-v1"
_RATE_SCALING = "arithmetic-mean-over-positive-timestamp-aggregates-v1; zeros-retained"


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
        if isinstance(item, dict):
            return {key: jsonable(entry) for key, entry in item.items()}
        return item

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class IngestionReport:
    accepted: int
    rejected_by_reason: tuple[tuple[str, int], ...]

    @property
    def rejected(self) -> int:
        return sum(count for _, count in self.rejected_by_reason)


@dataclass(frozen=True)
class LogicalCall:
    source_id: str
    row_ids: tuple[int, ...]
    timestamp_ms: int
    trace_id: str
    rpc_id: str
    parent_rpc_id: str | None
    upstream: str | None
    downstream: str | None
    rpc_type: str
    interface: str | None
    upstream_rt_ms: float | None
    provider_rt_ms: float | None
    left_censored_below_ms: float | None
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class CallGraphCorpus:
    source_id: str
    content_sha256: str
    sample_fraction: float | None
    calls: tuple[LogicalCall, ...]
    report: IngestionReport
    selection_method: str
    selection_seed: int | None
    requested_trace_count: int | None
    parent_corpus_seal: str | None
    parent_scan_seal: str | None
    transform_code_version: str
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify_seal(self) -> None:
        if self._authority is not _CORPUS_AUTHORITY:
            raise ValueError("call-graph corpus must be issued by the byte parser")
        parents = [
            seal for seal in (self.parent_corpus_seal, self.parent_scan_seal) if seal is not None
        ]
        if self.selection_seed is None:
            if self.requested_trace_count is not None or parents:
                raise ValueError("source call-graph corpus has inconsistent selection provenance")
        else:
            if (
                isinstance(self.selection_seed, bool)
                or self.requested_trace_count is None
                or self.requested_trace_count <= 0
                or len(parents) != 1
            ):
                raise ValueError("sampled call-graph corpus lacks seed/count/parent provenance")
            if len(parents[0]) != 64:
                raise ValueError("sampled call-graph parent seal is malformed")
        if not self.selection_method or not self.transform_code_version:
            raise ValueError("call-graph selection method/code version must be recorded")
        payload = {
            "source_id": self.source_id,
            "content_sha256": self.content_sha256,
            "sample_fraction": self.sample_fraction,
            "calls": self.calls,
            "report": self.report,
            "selection_method": self.selection_method,
            "selection_seed": self.selection_seed,
            "requested_trace_count": self.requested_trace_count,
            "parent_corpus_seal": self.parent_corpus_seal,
            "parent_scan_seal": self.parent_scan_seal,
            "transform_code_version": self.transform_code_version,
        }
        expected = hashlib.sha256((_CORPUS_TAG + _canonical(payload)).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("call-graph corpus seal mismatch")


@dataclass(frozen=True)
class MetricObservation:
    source_id: str
    row_id: int
    timestamp_ms: int
    service: str
    instance: str
    metric: str
    value: float
    unit: str
    normalized: bool
    estimand: str
    aggregation_interval_s: float
    left_censored_below: float | None = None
    activity_metric: str | None = None
    activity_value: float | None = None


@dataclass(frozen=True)
class MetricCorpus:
    source_id: str
    content_sha256: str
    observations: tuple[MetricObservation, ...]
    report: IngestionReport
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify_seal(self) -> None:
        if self._authority is not _CORPUS_AUTHORITY:
            raise ValueError("metric corpus must be issued by the byte parser")
        payload = {
            "source_id": self.source_id,
            "content_sha256": self.content_sha256,
            "observations": self.observations,
            "report": self.report,
        }
        expected = hashlib.sha256((_CORPUS_TAG + _canonical(payload)).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("metric corpus seal mismatch")


def _call_corpus(**payload: object) -> CallGraphCorpus:
    seal = hashlib.sha256((_CORPUS_TAG + _canonical(payload)).encode()).hexdigest()
    corpus = CallGraphCorpus(**payload, seal=seal, _authority=_CORPUS_AUTHORITY)  # type: ignore[arg-type]
    corpus.verify_seal()
    return corpus


def _metric_corpus(**payload: object) -> MetricCorpus:
    seal = hashlib.sha256((_CORPUS_TAG + _canonical(payload)).encode()).hexdigest()
    corpus = MetricCorpus(**payload, seal=seal, _authority=_CORPUS_AUTHORITY)  # type: ignore[arg-type]
    corpus.verify_seal()
    return corpus


def _attach_rt_activity(
    observations: list[MetricObservation], source_id: str
) -> list[MetricObservation]:
    """Join every RT cell to its same-instance/time/family MCR activity cell."""
    activity: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for observation in observations:
        if observation.metric.endswith("_MCR"):
            activity[
                (
                    observation.timestamp_ms,
                    observation.service,
                    observation.instance,
                    observation.metric,
                )
            ].append(observation.value)
    joined: list[MetricObservation] = []
    for observation in observations:
        if not observation.metric.endswith("_RT"):
            joined.append(observation)
            continue
        activity_metric = observation.metric.removesuffix("_RT") + "_MCR"
        matches = activity.get(
            (
                observation.timestamp_ms,
                observation.service,
                observation.instance,
                activity_metric,
            ),
            [],
        )
        activity_value = matches[0] if len(matches) == 1 else None
        censor_limit = (
            1.0
            if source_id == V2021
            and observation.value == 0.0
            and activity_value is not None
            and activity_value > 0.0
            else None
        )
        joined.append(
            dataclasses.replace(
                observation,
                left_censored_below=censor_limit,
                activity_metric=activity_metric,
                activity_value=activity_value,
            )
        )
    return joined


@dataclass(frozen=True)
class RtFitArtifact:
    schema: str
    source_id: str
    source_content_sha256: str
    source_registry_sha256: str
    service: str
    metric: str
    semantic: str
    estimand: str
    activity_rule: str
    censoring_rule: str
    family: str
    params_seconds: tuple[tuple[str, float], ...]
    positive_count: int
    censored_count: int
    rejected_count: int
    row_selection: str
    fit_config_sha256: str
    fit_code_version: str
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("seal", "_authority")
        }

    def verify_seal(self) -> None:
        if self._authority is not _FIT_AUTHORITY:
            raise ValueError("RT-fit artifact must be issued by the registered fitter")
        if self.schema != "alibaba-rt-fit-v1" or self.source_id != V2021:
            raise ValueError("RT-fit artifact has an invalid schema or source")
        for name, digest in (
            ("source content", self.source_content_sha256),
            ("source registry", self.source_registry_sha256),
            ("fit config", self.fit_config_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"RT-fit {name} must be a lowercase SHA-256 digest")
        if (
            not self.service
            or not self.metric.endswith("_RT")
            or self.semantic
            != (
                "distribution of service-instance-by-60s-interval mean response times; "
                "not per-request latency and not direct CPU service time"
            )
            or self.estimand != _RT_ESTIMAND
            or self.activity_rule != _RT_ACTIVITY_RULE
            or self.censoring_rule != _RT_CENSORING_RULE
            or self.family != "lognormal"
        ):
            raise ValueError("RT-fit artifact has invalid typed estimand semantics")
        if tuple(name for name, _ in self.params_seconds) != ("mu", "sigma"):
            raise ValueError("RT-fit parameters must be ordered lognormal mu/sigma")
        params = dict(self.params_seconds)
        if (
            any(not math.isfinite(value) for value in params.values())
            or params["sigma"] <= 0
        ):
            raise ValueError("RT-fit parameters must be finite with sigma > 0")
        counts = (self.positive_count, self.censored_count, self.rejected_count)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ):
            raise ValueError("RT-fit counts must be nonnegative integers")
        if self.positive_count <= 0 or not self.row_selection or not self.fit_code_version:
            raise ValueError("RT-fit artifact lacks selected positive rows or provenance")
        expected = hashlib.sha256((_FIT_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("RT-fit artifact seal mismatch")


@dataclass(frozen=True)
class AbsoluteRootRateArtifact:
    schema: str
    source_id: str
    source_content_sha256: str
    source_registry_sha256: str
    metric: str
    root_services: tuple[str, ...]
    selection_basis: str
    time_origin_ms: int
    points_relative_s: tuple[tuple[float, float], ...]
    aggregation_interval_s: float
    support_end_relative_s: float
    target_positive_mean_rps: float
    aggregation: str
    scaling: str
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("seal", "_authority")
        }

    def verify_seal(self) -> None:
        if self._authority is not _RATE_AUTHORITY:
            raise ValueError(
                "absolute root-rate artifact must be issued by the registered transform"
            )
        if self.schema != "alibaba-absolute-root-rate-shape-v1" or self.source_id != V2021:
            raise ValueError("absolute root-rate artifact has an invalid schema or source")
        for name, digest in (
            ("source content", self.source_content_sha256),
            ("source registry", self.source_registry_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"absolute root-rate {name} must be a lowercase SHA-256 digest")
        if (
            not self.metric.endswith("_MCR")
            or not self.root_services
            or len(self.root_services) != len(set(self.root_services))
            or self.selection_basis != "identified-entry-root-only"
            or isinstance(self.time_origin_ms, bool)
            or not isinstance(self.time_origin_ms, int)
            or self.time_origin_ms < 0
            or self.aggregation_interval_s != 60.0
            or not math.isfinite(self.support_end_relative_s)
            or not math.isfinite(self.target_positive_mean_rps)
            or self.target_positive_mean_rps <= 0
            or self.aggregation != _RATE_AGGREGATION
            or self.scaling != _RATE_SCALING
        ):
            raise ValueError("absolute root-rate artifact has invalid typed semantics")
        if not self.points_relative_s or not any(rate > 0 for _, rate in self.points_relative_s):
            raise ValueError("absolute root-rate artifact needs at least one positive point")
        times = [at for at, _ in self.points_relative_s]
        if times != sorted(set(times)) or times[0] != 0.0:
            raise ValueError("absolute root-rate points must be unique, ordered, and relative")
        if times != [index * self.aggregation_interval_s for index in range(len(times))]:
            raise ValueError("absolute root-rate points must cover consecutive complete cells")
        if any(
            not math.isfinite(at) or not math.isfinite(rate) or at < 0 or rate < 0
            for at, rate in self.points_relative_s
        ):
            raise ValueError("absolute root-rate points must be finite and nonnegative")
        if self.support_end_relative_s != times[-1] + self.aggregation_interval_s:
            raise ValueError("absolute root-rate support must end after the final complete cell")
        positive = [rate for _, rate in self.points_relative_s if rate > 0]
        if not math.isclose(
            math.fsum(positive) / len(positive),
            self.target_positive_mean_rps,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("absolute root-rate points do not realize the target positive mean")
        expected = hashlib.sha256(
            (_RATE_SHAPE_TAG + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.seal != expected:
            raise ValueError("absolute root-rate artifact seal mismatch")

    def to_arrival_spec(self, *, kind: str, episode_start_s: float = 0.0) -> ArrivalSpec:
        """Convert the relative trace shape into a legal piecewise-rate arrival spec."""
        self.verify_seal()
        first_positive = next(
            index for index, (_, rate) in enumerate(self.points_relative_s) if rate > 0
        )
        relative_start, initial_rate = self.points_relative_s[first_positive]
        start_s = episode_start_s + relative_start
        schedule = tuple(
            (episode_start_s + relative_s, rate)
            for relative_s, rate in self.points_relative_s[first_positive + 1 :]
        ) + ((episode_start_s + self.support_end_relative_s, 0.0),)
        return ArrivalSpec(kind, initial_rate, start_s=start_s, rate_schedule=schedule)


def _reader(payload: bytes) -> csv.DictReader:
    """Return a CSV reader over the exact bytes whose digest enters provenance.

    Production parsers deliberately do not accept a caller-asserted content hash: doing
    so would let an arbitrary fixture masquerade as a registered empirical payload.
    """
    if not isinstance(payload, bytes):
        raise TypeError("empirical CSV payload must be exact bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("empirical CSV payload must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header")
    return reader


def _parent_rpc_id(rpc_id: str) -> str | None:
    parts = rpc_id.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("malformed_rpc_id")
    return None if len(parts) == 1 else ".".join(parts[:-1])


def v2021_call_graph_row_rejection(row: dict[str, str]) -> str | None:
    """Return ``None`` when a v2021 call-graph row is well formed, else its reason."""
    try:
        if not row["traceid"]:
            raise ValueError("missing_trace_id")
        _parent_rpc_id(row["rpcid"])
        int(row["timestamp"])
        rt = float(row["rt"])
        if not math.isfinite(rt):
            raise ValueError("nonfinite_rt")
    except (KeyError, TypeError, ValueError) as exc:
        return str(exc) if str(exc) else "malformed_row"
    return None


def collapse_v2021_call_group(
    trace_id: str, rpc_id: str, rows: list[tuple[int, dict[str, str]]]
) -> tuple[LogicalCall | None, str | None]:
    """Collapse one ``(traceid, rpcid)`` group's signed halves into a logical call.

    Shared verbatim by the whole-payload parser and the streaming scan.  The grouping key
    contains the trace id, so sharding a stream by trace id keeps every group intact.
    """
    positives = [(rid, row, float(row["rt"])) for rid, row in rows if float(row["rt"]) >= 0]
    negatives = [(rid, row, float(row["rt"])) for rid, row in rows if float(row["rt"]) < 0]
    if len(positives) > 1 or len(negatives) > 1:
        return None, "ambiguous_duplicate_half"
    chosen = positives[0] if positives else negatives[0]
    _, row, signed_rt = chosen
    if positives and negatives:
        positive_row, negative_row = positives[0][1], negatives[0][1]

        if (
            positive_row["timestamp"] != negative_row["timestamp"]
            or positive_row["rpctype"].lower() != negative_row["rpctype"].lower()
            or any(
                positive_row[field] not in _MISSING
                and negative_row[field] not in _MISSING
                and positive_row[field] != negative_row[field]
                for field in ("um", "dm", "interface")
            )
        ):
            return None, "inconsistent_remote_halves"
        row = {
            **positive_row,
            **{
                field: (
                    positive_row[field]
                    if positive_row[field] not in _MISSING
                    else negative_row[field]
                )
                for field in ("um", "dm", "interface")
            },
        }
    flags: list[str] = []
    if row["um"] in _MISSING or row["dm"] in _MISSING:
        flags.append("missing_service_identity")
    remote = row["rpctype"].lower() in {"rpc", "http"}
    if remote and (not positives or not negatives):
        flags.append("unmatched_remote_half")
    up_rt = positives[0][2] if positives else None
    provider_rt = abs(negatives[0][2]) if negatives else None
    censored = 1.0 if signed_rt == 0.0 else None
    return (
        LogicalCall(
            source_id=V2021,
            row_ids=tuple(sorted(rid for rid, _ in rows)),
            timestamp_ms=int(row["timestamp"]),
            trace_id=trace_id,
            rpc_id=rpc_id,
            parent_rpc_id=_parent_rpc_id(rpc_id),
            upstream=None if row["um"] in _MISSING else row["um"],
            downstream=None if row["dm"] in _MISSING else row["dm"],
            rpc_type=row["rpctype"].lower(),
            interface=None if row["interface"] in _MISSING else row["interface"],
            upstream_rt_ms=up_rt,
            provider_rt_ms=provider_rt,
            left_censored_below_ms=censored,
            quality_flags=tuple(sorted(flags)),
        ),
        None,
    )


def parse_v2021_call_graph(payload: bytes) -> CallGraphCorpus:
    """Parse the v2021 MSCallGraph dump exactly as published.

    The real dump (MSCallGraph_0.tar.gz, sha256 b5cba70a4064018af096096b4fdb4c81
    161a1e681e1add78930201846ad9a235, fetched 2026-07-21) carries an unnamed
    pandas-style leading index column that the pinned README does not document.
    The contract matches the observed bytes; the index value itself is ignored.
    """
    reader = _reader(payload)
    content_sha256 = hashlib.sha256(payload).hexdigest()
    if set(reader.fieldnames or ()) != V2021_CALL_GRAPH_COLUMNS:
        raise ValueError(
            f"v2021 call-graph columns must be exactly {sorted(V2021_CALL_GRAPH_COLUMNS)}"
        )
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for row_id, row in enumerate(reader, start=2):
        reason = v2021_call_graph_row_rejection(row)
        if reason is not None:
            rejected[reason] += 1
            continue
        grouped[(row["traceid"], row["rpcid"])].append((row_id, row))

    calls: list[LogicalCall] = []
    for (trace_id, rpc_id), rows in sorted(grouped.items()):
        call, reason = collapse_v2021_call_group(trace_id, rpc_id, rows)
        if call is None:
            rejected[reason or "malformed_row"] += len(rows)
            continue
        calls.append(call)
    return _call_corpus(
        source_id=V2021,
        content_sha256=content_sha256,
        sample_fraction=CALLGRAPH_SAMPLE_FRACTION_V2021,
        calls=tuple(calls),
        report=IngestionReport(len(calls), tuple(sorted(rejected.items()))),
        selection_method="source-provided-0.5%-call-graph-sample",
        selection_seed=_NO_SAMPLE_SEED,
        requested_trace_count=None,
        parent_corpus_seal=None,
        parent_scan_seal=None,
        transform_code_version=(
            "v2021-exact-byte-parser-v2:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
    )


def issue_streamed_call_graph_sample(
    *,
    content_sha256: str,
    calls: tuple[LogicalCall, ...],
    report: IngestionReport,
    selection_seed: int,
    requested_trace_count: int,
    parent_scan_seal: str,
    transform_code_version: str,
    selection_method: str = "streaming-bottom-k-trace-sha256-v1",
) -> CallGraphCorpus:
    """Seal a trace-atomic sample taken during a complete-member streaming scan.

    The whole-corpus object never exists on a memory-lean run, so the sample's parentage
    is the sealed complete-member scan rather than a parent corpus.  Sealing stays inside
    this module so the corpus authority is never handed to a caller.
    """
    return _call_corpus(
        source_id=V2021,
        content_sha256=content_sha256,
        sample_fraction=CALLGRAPH_SAMPLE_FRACTION_V2021,
        calls=calls,
        report=report,
        selection_method=selection_method,
        selection_seed=selection_seed,
        requested_trace_count=requested_trace_count,
        parent_corpus_seal=None,
        parent_scan_seal=parent_scan_seal,
        transform_code_version=transform_code_version,
    )


def sample_call_graphs(corpus: CallGraphCorpus, count: int, seed: int) -> CallGraphCorpus:
    """Deterministic bottom-k trace sampling, invariant to input row order."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("sample count must be an integer")
    if count <= 0:
        raise ValueError("sample count must be > 0")
    traces = sorted({call.trace_id for call in corpus.calls})
    corpus.verify_seal()
    if count > len(traces):
        raise ValueError(
            f"sample count {count} exceeds the {len(traces)} available trace ids"
        )
    ranked = sorted(
        traces,
        key=lambda trace: hashlib.sha256(f"{seed}:{trace}".encode()).digest(),
    )
    selected = set(ranked[:count])
    calls = tuple(call for call in corpus.calls if call.trace_id in selected)
    return _call_corpus(
        source_id=corpus.source_id,
        content_sha256=corpus.content_sha256,
        sample_fraction=corpus.sample_fraction,
        calls=calls,
        report=IngestionReport(len(calls), corpus.report.rejected_by_reason),
        selection_method="deterministic-bottom-k-trace-sha256-v1",
        selection_seed=seed,
        requested_trace_count=count,
        parent_corpus_seal=corpus.seal,
        parent_scan_seal=None,
        transform_code_version=(
            "bottom-k-trace-sampler-v1:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
    )


V2021_METRIC_COLUMNS = frozenset({"", "timestamp", "msname", "msinstanceid", "metric", "value"})
V2021_CALL_GRAPH_COLUMNS = frozenset(
    {"", "timestamp", "traceid", "rpcid", "um", "rpctype", "interface", "dm", "rt"}
)


def v2021_metric_row(
    row_id: int, row: dict[str, str]
) -> tuple[MetricObservation | None, str | None]:
    """Validate and type one v2021 metric row, or return its rejection reason.

    Shared verbatim by the whole-payload parser and the streaming scan so the two paths
    cannot drift in acceptance, typing, or rejection taxonomy.
    """
    try:
        timestamp = int(row["timestamp"])
        value = float(row["value"])
        if timestamp < 0 or not math.isfinite(value) or value < 0:
            raise ValueError("invalid_numeric_value")
        metric = row["metric"].upper()
        if not (metric.endswith("_RT") or metric.endswith("_MCR")):
            raise ValueError("unknown_metric")
        if row["msname"] in _MISSING or not row["msinstanceid"]:
            raise ValueError("missing_identity")
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc) or "malformed_row"
    is_rt = metric.endswith("_RT")
    return (
        MetricObservation(
            source_id=V2021,
            row_id=row_id,
            timestamp_ms=timestamp,
            service=row["msname"],
            instance=row["msinstanceid"],
            metric=metric,
            value=value,
            unit="ms" if is_rt else "calls/s",
            normalized=False,
            estimand=(
                "service-instance-by-60s-interval-mean-response-time"
                if is_rt
                else "service-instance-by-60s-interval-mean-call-rate"
            ),
            aggregation_interval_s=60.0,
            left_censored_below=None,
        ),
        None,
    )


def parse_v2021_metrics(payload: bytes) -> MetricCorpus:
    """Parse the v2021 MSRTQps dump exactly as published.

    The real dump (MSRTQps_0.tar.gz, sha256 8feb324ba06a1abaa1639a031720023d7b
    bbcfc3de75cfc46194455ad2d80a1d, fetched 2026-07-21) contradicts the pinned
    README's documented schema in two ways: an unnamed pandas-style leading
    index column, and the metric-name column is ``metric`` (singular), not
    ``metrics``. The contract matches the observed bytes; the index value
    itself is ignored.
    """
    reader = _reader(payload)
    content_sha256 = hashlib.sha256(payload).hexdigest()
    if set(reader.fieldnames or ()) != V2021_METRIC_COLUMNS:
        raise ValueError(
            f"v2021 metric columns must be exactly {sorted(V2021_METRIC_COLUMNS)}"
        )
    observations: list[MetricObservation] = []
    rejected: Counter[str] = Counter()
    for row_id, row in enumerate(reader, start=2):
        observation, reason = v2021_metric_row(row_id, row)
        if observation is None:
            rejected[reason or "malformed_row"] += 1
            continue
        observations.append(observation)
    observations = _attach_rt_activity(observations, V2021)
    return _metric_corpus(
        source_id=V2021,
        content_sha256=content_sha256,
        observations=tuple(observations),
        report=IngestionReport(len(observations), tuple(sorted(rejected.items()))),
    )


def parse_v2022_metrics(payload: bytes) -> MetricCorpus:
    """Parse v2022's wide table for held-out RT/load *shapes* only."""
    reader = _reader(payload)
    content_sha256 = hashlib.sha256(payload).hexdigest()
    fields = tuple(reader.fieldnames or ())
    base = {"timestamp", "msname", "msinstanceid", "nodeid"}
    if not base.issubset(fields):
        raise ValueError(f"v2022 metrics missing base columns {sorted(base - set(fields))}")
    metrics = [field for field in fields if field not in base]
    if not metrics or any(
        not (m.lower().endswith("_rt") or m.lower().endswith("_mcr")) for m in metrics
    ):
        raise ValueError("v2022 metric columns must end in _rt or _mcr")
    observations: list[MetricObservation] = []
    rejected: Counter[str] = Counter()
    for row_id, row in enumerate(reader, start=2):
        try:
            timestamp = int(row["timestamp"])
            if timestamp < 0 or row["msname"] in _MISSING or not row["msinstanceid"]:
                raise ValueError("invalid_base_fields")
        except (KeyError, TypeError, ValueError) as exc:
            rejected[str(exc) or "malformed_row"] += len(metrics)
            continue
        for metric in metrics:
            try:
                value = float(row[metric])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid_numeric_value")
            except (TypeError, ValueError) as exc:
                rejected[str(exc) or "malformed_value"] += 1
                continue
            is_rt = metric.lower().endswith("_rt")
            observations.append(
                MetricObservation(
                    source_id=V2022,
                    row_id=row_id,
                    timestamp_ms=timestamp,
                    service=row["msname"],
                    instance=row["msinstanceid"],
                    metric=metric.upper(),
                    value=value,
                    unit="ms" if is_rt else "normalized-shape",
                    normalized=not is_rt,
                    estimand=(
                        "service-instance-by-60s-interval-mean-response-time"
                        if is_rt
                        else "service-instance-by-60s-interval-normalized-call-rate-shape"
                    ),
                    aggregation_interval_s=60.0,
                    left_censored_below=None,
                )
            )
    observations = _attach_rt_activity(observations, V2022)
    return _metric_corpus(
        source_id=V2022,
        content_sha256=content_sha256,
        observations=tuple(observations),
        report=IngestionReport(len(observations), tuple(sorted(rejected.items()))),
    )


def absolute_root_rate_shape(
    corpus: MetricCorpus,
    registry: SourceRegistry,
    metric: str,
    target_positive_mean_rps: float,
    root_services: tuple[str, ...],
    selection_basis: str,
) -> AbsoluteRootRateArtifact:
    """Capacity-scale a v2021 absolute root-MCR temporal shape to a synthetic mean."""
    corpus.verify_seal()
    registry.require_component(
        corpus.source_id, CALIBRATION, METRICS_COMPONENT, corpus.content_sha256
    )
    if corpus.source_id != V2021:
        raise ValueError("absolute rates are available only from Alibaba v2021")
    if selection_basis != "identified-entry-root-only":
        raise ValueError("absolute offered rates require identified-entry-root-only selection")
    if not root_services or len(root_services) != len(set(root_services)):
        raise ValueError("root_services must be non-empty and unique")
    if not math.isfinite(target_positive_mean_rps) or target_positive_mean_rps <= 0:
        raise ValueError("target_positive_mean_rps must be finite and > 0")
    selected = tuple(
        obs
        for obs in corpus.observations
        if obs.metric == metric.upper()
        and obs.service in root_services
        and obs.unit == "calls/s"
    )
    if not selected or not any(obs.value > 0 for obs in selected):
        raise ValueError("no absolute root-MCR observations with a positive value selected")
    missing_roots = set(root_services) - {obs.service for obs in selected}
    if missing_roots:
        raise ValueError(f"no root-MCR observations for identified roots {sorted(missing_roots)}")
    # MCR is reported per service instance. The offered root process is the sum over
    # selected root instances/services at each timestamp—not a flat mean over rows,
    # which would double-weight timestamps with more reporting instances.
    by_timestamp: dict[int, float] = defaultdict(float)
    for observation in selected:
        by_timestamp[observation.timestamp_ms] += observation.value
    return seal_absolute_root_rate_shape(
        source_id=corpus.source_id,
        source_content_sha256=corpus.content_sha256,
        registry=registry,
        metric=metric,
        target_positive_mean_rps=target_positive_mean_rps,
        root_services=root_services,
        selection_basis=selection_basis,
        by_timestamp_ms=dict(by_timestamp),
    )


def seal_absolute_root_rate_shape(
    *,
    source_id: str,
    source_content_sha256: str,
    registry: SourceRegistry,
    metric: str,
    target_positive_mean_rps: float,
    root_services: tuple[str, ...],
    selection_basis: str,
    by_timestamp_ms: dict[int, float],
) -> AbsoluteRootRateArtifact:
    """Seal a root-rate shape from an already root-selected per-timestamp aggregate."""
    timestamps = sorted(by_timestamp_ms)
    if any(
        right - left != 60_000
        for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("absolute root-MCR shape has a missing 60-second source cell")
    scaled = capacity_scaled_shape(
        tuple(sorted(by_timestamp_ms.items())), target_positive_mean_rps
    )
    origin_ms = scaled[0][0]
    points = tuple(((timestamp - origin_ms) / 1000.0, rate) for timestamp, rate in scaled)
    payload = {
        "schema": "alibaba-absolute-root-rate-shape-v1",
        "source_id": source_id,
        "source_content_sha256": source_content_sha256,
        "source_registry_sha256": registry.manifest_sha256(),
        "metric": metric.upper(),
        "root_services": tuple(root_services),
        "selection_basis": selection_basis,
        "time_origin_ms": origin_ms,
        "points_relative_s": points,
        "aggregation_interval_s": 60.0,
        "support_end_relative_s": points[-1][0] + 60.0,
        "target_positive_mean_rps": target_positive_mean_rps,
        "aggregation": _RATE_AGGREGATION,
        "scaling": _RATE_SCALING,
    }
    seal = hashlib.sha256((_RATE_SHAPE_TAG + _canonical(payload)).encode()).hexdigest()
    artifact = AbsoluteRootRateArtifact(**payload, seal=seal, _authority=_RATE_AUTHORITY)
    artifact.verify_seal()
    return artifact


def capacity_scaled_shape(
    observations: tuple[tuple[int, float], ...], target_positive_mean_rps: float
) -> tuple[tuple[int, float], ...]:
    """Pure registered synthetic transform used after source-role/hash validation."""
    if not math.isfinite(target_positive_mean_rps) or target_positive_mean_rps <= 0:
        raise ValueError("target_positive_mean_rps must be finite and > 0")
    timestamps = [timestamp for timestamp, _ in observations]
    if (
        not observations
        or timestamps != sorted(set(timestamps))
        or any(not math.isfinite(value) or value < 0 for _, value in observations)
        or not any(value > 0 for _, value in observations)
    ):
        raise ValueError(
            "shape observations must have ordered unique timestamps, finite "
            "nonnegative values, and at least one positive value"
        )
    positive = [value for _, value in observations if value > 0]
    mean_positive = math.fsum(positive) / len(positive)
    return tuple(
        (timestamp, target_positive_mean_rps * value / mean_positive)
        for timestamp, value in observations
    )


def normalized_load_shape(corpus: MetricCorpus, registry: SourceRegistry) -> tuple[float, ...]:
    corpus.verify_seal()
    registry.require_component(
        corpus.source_id, FACE_VALIDITY, METRICS_COMPONENT, corpus.content_sha256
    )
    values = tuple(obs.value for obs in corpus.observations if obs.normalized)
    if not values:
        raise ValueError("no normalized load-shape values")
    return values


def _lognormal_censored_fit(
    values_s: list[float], censored: int, threshold_s: float
) -> tuple[float, float]:
    """Coordinate-search MLE for a left-censored lognormal, in log-seconds.

    The negative log-likelihood is evaluated from the three sufficient statistics of the
    uncensored log sample (count, mean, centred sum of squares) rather than by re-summing
    every observation per candidate.  That is the same objective — the centred form is
    used to avoid cancellation — but it makes the search cost independent of the sample
    size, which is what allows the complete-chunk streaming path to fit every eligible
    service without sub-sampling the evidence.
    """
    logs = [math.log(value) for value in values_s]
    if not logs:
        raise ValueError("at least one positive RT observation is required")
    count = len(logs)
    mu = math.fsum(logs) / count
    centred_squares = math.fsum((value - mu) ** 2 for value in logs)
    variance = centred_squares / max(count, 1)
    sigma = max(math.sqrt(variance), 0.05)
    log_sum = math.fsum(logs)
    log_mean = mu
    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    log_threshold = math.log(threshold_s)

    def nll(candidate_mu: float, candidate_sigma: float) -> float:
        if candidate_sigma <= 1e-5:
            return float("inf")
        deviation = centred_squares + count * (log_mean - candidate_mu) ** 2
        out = (
            log_sum
            + count * math.log(candidate_sigma)
            + count * half_log_2pi
            + deviation / (2.0 * candidate_sigma**2)
        )
        if censored:
            z = (log_threshold - candidate_mu) / candidate_sigma
            cdf = max(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 1e-300)
            out -= censored * math.log(cdf)
        return out

    step_mu, step_sigma = 1.0, 0.5
    best = nll(mu, sigma)
    for _ in range(80):
        candidates = []
        for dm in (-step_mu, 0.0, step_mu):
            for ds in (-step_sigma, 0.0, step_sigma):
                cm, cs = mu + dm, max(1e-4, sigma + ds)
                candidates.append((nll(cm, cs), cm, cs))
        score, next_mu, next_sigma = min(candidates)
        if score < best:
            best, mu, sigma = score, next_mu, next_sigma
        else:
            step_mu *= 0.5
            step_sigma *= 0.5
    return mu, sigma


def fit_response_time_targets(
    corpus: MetricCorpus,
    registry: SourceRegistry,
    service: str,
    metric: str,
    minimum_count: int = 20,
) -> RtFitArtifact:
    """Fit a censored lognormal *response-time target* from calibration data only."""
    corpus.verify_seal()
    registry.require_component(
        corpus.source_id, CALIBRATION, METRICS_COMPONENT, corpus.content_sha256
    )
    candidates = [
        obs
        for obs in corpus.observations
        if obs.service == service
        and obs.metric == metric.upper()
        and obs.unit == "ms"
        and obs.estimand == _RT_ESTIMAND
    ]
    cell_keys = [
        (obs.timestamp_ms, obs.service, obs.instance, obs.metric) for obs in candidates
    ]
    if len(cell_keys) != len(set(cell_keys)):
        raise ValueError("RT fit contains duplicate service-instance interval cells")
    if any(obs.activity_value is None for obs in candidates):
        raise ValueError("RT fit requires a unique paired same-cell MCR activity value")
    selected = [
        obs
        for obs in candidates
        if obs.activity_value is not None and obs.activity_value > 0
    ]
    if any(obs.value == 0 and obs.left_censored_below is None for obs in selected):
        raise ValueError("active RT zero lacks a source-documented censoring treatment")
    positive_s = [obs.value / 1000.0 for obs in selected if obs.value > 0]
    censored = sum(obs.left_censored_below is not None for obs in selected)
    if len(selected) < minimum_count:
        raise ValueError(f"RT fit needs at least {minimum_count} selected observations")
    return seal_rt_fit(
        source_id=corpus.source_id,
        source_content_sha256=corpus.content_sha256,
        registry=registry,
        service=service,
        metric=metric,
        minimum_count=minimum_count,
        positive_s=positive_s,
        censored=censored,
        rejected_count=corpus.report.rejected,
    )


def seal_rt_fit(
    *,
    source_id: str,
    source_content_sha256: str,
    registry: SourceRegistry,
    service: str,
    metric: str,
    minimum_count: int,
    positive_s: list[float],
    censored: int,
    rejected_count: int,
) -> RtFitArtifact:
    """Seal one RT fit from already activity-selected observations.

    Callers must have applied the P1 activity rule before calling: unique
    service-instance-interval cells, a unique same-cell same-family MCR partner per RT
    cell, MCR strictly positive, and the registered censoring treatment for active zeros.
    """
    mu, sigma = _lognormal_censored_fit(positive_s, censored, 0.001)
    config = {
        "service": service,
        "metric": metric.upper(),
        "minimum_count": minimum_count,
        "family": "lognormal",
        "censor_threshold_s": 0.001,
        "estimand": _RT_ESTIMAND,
        "activity_rule": _RT_ACTIVITY_RULE,
        "censoring_rule": _RT_CENSORING_RULE,
    }
    payload = {
        "schema": "alibaba-rt-fit-v1",
        "source_id": source_id,
        "source_content_sha256": source_content_sha256,
        "source_registry_sha256": registry.manifest_sha256(),
        "service": service,
        "metric": metric.upper(),
        "semantic": (
            "distribution of service-instance-by-60s-interval mean response times; "
            "not per-request latency and not direct CPU service time"
        ),
        "estimand": _RT_ESTIMAND,
        "activity_rule": _RT_ACTIVITY_RULE,
        "censoring_rule": _RT_CENSORING_RULE,
        "family": "lognormal",
        "params_seconds": (("mu", mu), ("sigma", sigma)),
        "positive_count": len(positive_s),
        "censored_count": censored,
        "rejected_count": rejected_count,
        "row_selection": (
            f"service == {service!r}; metric == {metric.upper()!r}; "
            "estimand == service-instance-by-60s-interval-mean-response-time; "
            "unique paired same-cell MCR > 0; positive or documented left-censored zero"
        ),
        "fit_config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
        "fit_code_version": (
            "censored-lognormal-coordinate-search-v1:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
    }
    seal = hashlib.sha256((_FIT_TAG + _canonical(payload)).encode()).hexdigest()
    artifact = RtFitArtifact(**payload, seal=seal, _authority=_FIT_AUTHORITY)
    artifact.verify_seal()
    return artifact
