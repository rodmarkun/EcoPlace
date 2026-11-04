"""Memory-lean complete-member ingestion for the Alibaba v2021 calibration corpus.

The whole-payload parsers in :mod:`sim.scenarios.alibaba` materialise every row of a
member at once.  On the retained v2021 chunk-0 pair that costs roughly 14-16 GB, which a
16 GB machine cannot afford.  The obvious shortcut — parsing a byte prefix — is
forbidden: the 2026-07-21 pilot showed the dump is not time ordered, so a head slice
silently severs about 80% of the RT-MCR activity pairs that are present in the complete
chunk, and the fitted estimand quietly changes.

This module therefore reads the *complete* member and keeps memory bounded by grouping on
disk instead of in RAM:

* every byte of the member is streamed once, hashed, and decoded through the very same
  :class:`csv.DictReader` contract the whole-payload parsers use;
* each accepted row is appended to one of ``shards`` temporary files chosen by a stable
  digest of its grouping key;
* each shard is then loaded and reduced on its own.

Both grouping keys used here are closed under their shard key.  The RT activity join is
keyed by ``(timestamp, service, instance, metric family)``, so sharding by *service*
keeps every RT cell together with its MCR partner.  Call halves are grouped by
``(traceid, rpcid)``, so sharding by *trace id* keeps every logical call intact.  No pair
is lost, and the row-level acceptance, typing and rejection taxonomy are imported from
the whole-payload parsers rather than re-implemented, so the two paths cannot drift.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import resource
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sim.scenarios.alibaba import (
    CALL_GRAPH_COMPONENT,
    METRICS_COMPONENT,
    V2021,
    V2021_CALL_GRAPH_COLUMNS,
    V2021_METRIC_COLUMNS,
    AbsoluteRootRateArtifact,
    CallGraphCorpus,
    IngestionReport,
    LogicalCall,
    MetricObservation,
    RtFitArtifact,
    collapse_v2021_call_group,
    issue_streamed_call_graph_sample,
    seal_absolute_root_rate_shape,
    seal_rt_fit,
    v2021_call_graph_row_rejection,
    v2021_metric_row,
)
from sim.scenarios.sources import CALIBRATION, SourceRegistry

_SCAN_TAG = "alibaba-v2021-complete-member-scan-v1:"
_SCAN_AUTHORITY = object()
_CORPUS_TAG = "alibaba-ingestion-corpus-v1:"
DEFAULT_SHARDS = 256
_RT_ACTIVITY_RULE = "paired-same-cell-family-mcr-strictly-positive-v1"
_ROOT_SELECTION_BASIS = "identified-entry-root-only"
CODE_VERSION = "alibaba-complete-member-streaming-scan-v1"


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


def code_version() -> str:
    """Version tag binding streamed artifacts to the exact scan implementation."""
    return CODE_VERSION + ":" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _HashingReader(io.RawIOBase):
    """Read-through wrapper that digests and counts every byte handed downstream."""

    def __init__(self, raw: io.BufferedIOBase) -> None:
        self._raw = raw
        self._digest = hashlib.sha256()
        self._bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        chunk = self._raw.read(len(buffer))
        if not chunk:
            return 0
        buffer[: len(chunk)] = chunk
        self._digest.update(chunk)
        self._bytes += len(chunk)
        return len(chunk)

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def byte_count(self) -> int:
        return self._bytes


def _shard_of(key: str, shards: int) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % shards


def stream_tar_member_rows(
    archive: Path, member: str, columns: frozenset[str], label: str
) -> Iterator[tuple[int, dict[str, str], _HashingReader]]:
    """Yield ``(row_id, row, reader)`` for a complete tar member, hashing every byte.

    ``row_id`` matches the whole-payload parsers' audit identity (header is row 1).  The
    reader is yielded so the caller can read the final digest and byte count *after* the
    stream is exhausted; reading them earlier would describe a prefix, not the member.
    """
    with tarfile.open(archive, "r:gz") as tar:
        members = [item for item in tar.getmembers() if item.name == member]
        if len(members) != 1:
            raise ValueError(f"{label} archive must contain exactly one member {member!r}")
        handle = tar.extractfile(members[0])
        if handle is None:
            raise ValueError(f"{label} archive member {member!r} is not a regular file")
        hashing = _HashingReader(handle)
        text = io.TextIOWrapper(
            io.BufferedReader(hashing, buffer_size=1 << 22), encoding="utf-8-sig", newline=""
        )
        reader = csv.DictReader(text)
        if reader.fieldnames is None:
            raise ValueError(f"{label} member has no header")
        if set(reader.fieldnames) != columns:
            raise ValueError(f"{label} columns must be exactly {sorted(columns)}")
        for row_id, row in enumerate(reader, start=2):
            yield row_id, row, hashing


@dataclass(frozen=True)
class RtCellSummary:
    """Per ``(service, metric)`` outcome of applying the P1 activity rule."""

    service: str
    metric: str
    rt_cells: int
    duplicate_cells: int
    unpaired_cells: int
    active_cells: int
    inactive_cells: int
    positive_cells: int
    censored_cells: int
    eligible: bool
    ineligibility: str | None


@dataclass(frozen=True)
class MemberScanArtifact:
    """Sealed record of one complete-member streaming scan."""

    schema: str
    source_id: str
    component: str
    parser_input: str
    content_sha256: str
    content_bytes: int
    source_registry_sha256: str
    rows_read: int
    accepted: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    facts: tuple[tuple[str, str], ...]
    truncation: str
    scan_code_version: str
    shards: int
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("seal", "_authority")
        }

    @property
    def rejected(self) -> int:
        return sum(count for _, count in self.rejected_by_reason)

    def verify_seal(self) -> None:
        if self._authority is not _SCAN_AUTHORITY:
            raise ValueError("member scan artifact must be issued by the streaming scanner")
        if self.schema != "alibaba-v2021-complete-member-scan-v1" or self.source_id != V2021:
            raise ValueError("member scan artifact has an invalid schema or source")
        for name, digest in (
            ("content", self.content_sha256),
            ("source registry", self.source_registry_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"member scan {name} must be a lowercase SHA-256 digest")
        if self.truncation != "none-complete-member-streamed":
            raise ValueError("member scan must certify that the complete member was streamed")
        if self.content_bytes <= 0 or self.rows_read < 0 or self.accepted < 0:
            raise ValueError("member scan counters must be nonnegative")
        if self.accepted > self.rows_read:
            raise ValueError("member scan cannot accept more rows than it read")
        if not self.component or not self.parser_input or not self.scan_code_version:
            raise ValueError("member scan lacks component/parser-input/code provenance")
        expected = hashlib.sha256((_SCAN_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("member scan artifact seal mismatch")


def _seal_scan(**payload: object) -> MemberScanArtifact:
    seal = hashlib.sha256((_SCAN_TAG + _canonical(payload)).encode()).hexdigest()
    artifact = MemberScanArtifact(**payload, seal=seal, _authority=_SCAN_AUTHORITY)  # type: ignore[arg-type]
    artifact.verify_seal()
    return artifact


@dataclass(frozen=True)
class MetricScanResult:
    scan: MemberScanArtifact
    cells: tuple[RtCellSummary, ...]
    fits: tuple[RtFitArtifact, ...]
    services: int
    instances: int
    timestamps: tuple[int, ...]
    metric_counts: tuple[tuple[str, int], ...]
    root_rate_shape: AbsoluteRootRateArtifact | None
    root_services_requested: tuple[str, ...]
    root_services_used: tuple[str, ...]
    root_services_without_observations: tuple[str, ...]


class _ShardWriter:
    """Append-only fan-out of accepted rows to ``shards`` temporary files.

    Rows are written as CSV rather than joined on a separator: service, instance and
    interface identifiers are opaque published strings, and a field containing the
    separator or a newline would otherwise corrupt the shard round-trip silently.
    """

    def __init__(self, directory: Path, shards: int) -> None:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and shards + 32 > soft:
            raise ValueError(
                f"{shards} shards need more open files than the {soft} descriptor limit allows"
            )
        self._paths = [directory / f"shard_{index:04d}.csv" for index in range(shards)]
        self._handles = [path.open("w", encoding="utf-8", newline="") for path in self._paths]
        self._writers = [csv.writer(handle) for handle in self._handles]
        self._shards = shards

    def write(self, key: str, fields: list[str]) -> None:
        self._writers[_shard_of(key, self._shards)].writerow(fields)

    def close(self) -> list[Path]:
        for handle in self._handles:
            handle.close()
        return list(self._paths)


def _read_shard(shard: Path) -> Iterator[list[str]]:
    with shard.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.reader(handle)


def scan_v2021_metrics(
    archive: Path,
    member: str,
    registry: SourceRegistry,
    *,
    workspace: Path,
    minimum_count: int = 20,
    root_services: tuple[str, ...] = (),
    root_metric: str | None = None,
    target_positive_mean_rps: float | None = None,
    shards: int = DEFAULT_SHARDS,
    progress: object = None,
) -> MetricScanResult:
    """Scan the complete metric member and fit every activity-rule-eligible pair.

    The selector is outcome-independent and fixed before any parameter is estimated: a
    ``(service, metric)`` pair is fit when, and only when, it satisfies the registered
    activity rule exactly — no duplicate service-instance-interval RT cell, a unique
    same-cell same-family MCR partner for every RT cell, and at least ``minimum_count``
    cells whose partner is strictly positive.  Nothing about the fitted values enters the
    selection.
    """
    if shards <= 0:
        raise ValueError("shard count must be positive")
    workspace.mkdir(parents=True, exist_ok=True)
    rejected: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    services: set[str] = set()
    instances: set[str] = set()
    timestamps: set[int] = set()
    root_totals: dict[int, float] = defaultdict(float)
    # Roots are identified in the call-graph member (itself an upstream 0.5% sample), so
    # a root can legitimately report no row of the chosen metric family in the metric
    # member.  Such roots are dropped from the aggregate, and the drop is reported rather
    # than silently folded into the sealed selection.
    roots_observed: set[str] = set()
    want_roots = set(root_services)
    root_metric_upper = root_metric.upper() if root_metric else None
    rows_read = 0
    accepted = 0

    with tempfile.TemporaryDirectory(dir=workspace, prefix="metric-shards-") as raw:
        writer = _ShardWriter(Path(raw), shards)
        reader_ref: _HashingReader | None = None
        for row_id, row, hashing in stream_tar_member_rows(
            archive, member, V2021_METRIC_COLUMNS, "v2021 metric"
        ):
            reader_ref = hashing
            rows_read += 1
            observation, reason = v2021_metric_row(row_id, row)
            if observation is None:
                rejected[reason or "malformed_row"] += 1
                continue
            accepted += 1
            metric_counts[observation.metric] += 1
            services.add(observation.service)
            instances.add(observation.instance)
            timestamps.add(observation.timestamp_ms)
            if (
                root_metric_upper is not None
                and observation.metric == root_metric_upper
                and observation.service in want_roots
            ):
                root_totals[observation.timestamp_ms] += observation.value
                roots_observed.add(observation.service)
            writer.write(
                observation.service,
                [
                    str(observation.row_id),
                    str(observation.timestamp_ms),
                    observation.service,
                    observation.instance,
                    observation.metric,
                    repr(observation.value),
                ],
            )
            if progress is not None and rows_read % 1_000_000 == 0:
                print(f"  metrics: {rows_read:,} rows read", flush=True)
        if reader_ref is None:
            raise ValueError("v2021 metric member contained no data rows")
        content_sha256 = reader_ref.sha256
        content_bytes = reader_ref.byte_count
        shard_paths = writer.close()

        registry.require_component(V2021, CALIBRATION, METRICS_COMPONENT, content_sha256)

        cells: list[RtCellSummary] = []
        fit_inputs: list[tuple[str, str, list[float], int]] = []
        for shard in shard_paths:
            cells.extend(_reduce_metric_shard(shard, minimum_count, fit_inputs))
            os.unlink(shard)

    facts = {
        "distinct_services": str(len(services)),
        "distinct_instances": str(len(instances)),
        "distinct_timestamps": str(len(timestamps)),
        "first_timestamp_ms": str(min(timestamps)) if timestamps else "",
        "last_timestamp_ms": str(max(timestamps)) if timestamps else "",
        "rt_cells": str(sum(item.rt_cells for item in cells)),
        "rt_cells_duplicate": str(sum(item.duplicate_cells for item in cells)),
        "rt_cells_unpaired": str(sum(item.unpaired_cells for item in cells)),
        "rt_cells_active": str(sum(item.active_cells for item in cells)),
        "rt_cells_inactive": str(sum(item.inactive_cells for item in cells)),
        "rt_cells_positive": str(sum(item.positive_cells for item in cells)),
        "rt_cells_censored": str(sum(item.censored_cells for item in cells)),
        "eligible_service_metric_pairs": str(sum(1 for item in cells if item.eligible)),
        "activity_rule": _RT_ACTIVITY_RULE,
        "minimum_active_cells": str(minimum_count),
        "root_services_requested": str(len(want_roots)),
        "root_services_used": str(len(roots_observed)),
        "root_metric": root_metric_upper or "",
    }
    scan = _seal_scan(
        schema="alibaba-v2021-complete-member-scan-v1",
        source_id=V2021,
        component=METRICS_COMPONENT,
        parser_input=member,
        content_sha256=content_sha256,
        content_bytes=content_bytes,
        source_registry_sha256=registry.manifest_sha256(),
        rows_read=rows_read,
        accepted=accepted,
        rejected_by_reason=tuple(sorted(rejected.items())),
        facts=tuple(sorted(facts.items())),
        truncation="none-complete-member-streamed",
        scan_code_version=code_version(),
        shards=shards,
    )

    fits = tuple(
        seal_rt_fit(
            source_id=V2021,
            source_content_sha256=content_sha256,
            registry=registry,
            service=service,
            metric=metric,
            minimum_count=minimum_count,
            positive_s=positive_s,
            censored=censored,
            rejected_count=scan.rejected,
        )
        for service, metric, positive_s, censored in sorted(
            fit_inputs, key=lambda item: (item[0], item[1])
        )
    )

    shape = None
    if root_metric_upper is not None and target_positive_mean_rps is not None:
        if not root_totals:
            raise ValueError("no absolute root-MCR observations with the selected roots")
        shape = seal_absolute_root_rate_shape(
            source_id=V2021,
            source_content_sha256=content_sha256,
            registry=registry,
            metric=root_metric_upper,
            target_positive_mean_rps=target_positive_mean_rps,
            root_services=tuple(sorted(roots_observed)),
            selection_basis=_ROOT_SELECTION_BASIS,
            by_timestamp_ms=dict(root_totals),
        )

    return MetricScanResult(
        scan=scan,
        cells=tuple(sorted(cells, key=lambda item: (item.service, item.metric))),
        fits=fits,
        services=len(services),
        instances=len(instances),
        timestamps=tuple(sorted(timestamps)),
        metric_counts=tuple(sorted(metric_counts.items())),
        root_rate_shape=shape,
        root_services_requested=tuple(sorted(want_roots)),
        root_services_used=tuple(sorted(roots_observed)),
        root_services_without_observations=tuple(sorted(want_roots - roots_observed)),
    )


def _reduce_metric_shard(
    shard: Path, minimum_count: int, fit_inputs: list[tuple[str, str, list[float], int]]
) -> list[RtCellSummary]:
    """Apply the activity join to one service-closed shard and summarise every pair."""
    activity: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    rt_rows: list[MetricObservation] = []
    for record in _read_shard(shard):
        row_id, timestamp, service, instance, metric, value = record
        observation = MetricObservation(
            source_id=V2021,
            row_id=int(row_id),
            timestamp_ms=int(timestamp),
            service=service,
            instance=instance,
            metric=metric,
            value=float(value),
            unit="ms" if metric.endswith("_RT") else "calls/s",
            normalized=False,
            estimand=(
                "service-instance-by-60s-interval-mean-response-time"
                if metric.endswith("_RT")
                else "service-instance-by-60s-interval-mean-call-rate"
            ),
            aggregation_interval_s=60.0,
        )
        if metric.endswith("_MCR"):
            activity[(observation.timestamp_ms, service, instance, metric)].append(
                observation.value
            )
        else:
            rt_rows.append(observation)

    grouped: dict[tuple[str, str], list[MetricObservation]] = defaultdict(list)
    for observation in rt_rows:
        grouped[(observation.service, observation.metric)].append(observation)

    summaries: list[RtCellSummary] = []
    for (service, metric), observations in grouped.items():
        keys = [
            (item.timestamp_ms, item.service, item.instance, item.metric) for item in observations
        ]
        duplicates = len(keys) - len(set(keys))
        activity_metric = metric.removesuffix("_RT") + "_MCR"
        unpaired = 0
        positive_s: list[float] = []
        censored = 0
        active = 0
        inactive = 0
        for item in observations:
            matches = activity.get(
                (item.timestamp_ms, item.service, item.instance, activity_metric), []
            )
            if len(matches) != 1:
                unpaired += 1
                continue
            if matches[0] <= 0:
                inactive += 1
                continue
            active += 1
            if item.value > 0:
                positive_s.append(item.value / 1000.0)
            else:
                censored += 1
        ineligibility = None
        if duplicates:
            ineligibility = "duplicate-service-instance-interval-cells"
        elif unpaired:
            ineligibility = "rt-cell-without-unique-same-cell-mcr-partner"
        elif active < minimum_count:
            ineligibility = f"fewer-than-{minimum_count}-active-cells"
        elif not positive_s:
            ineligibility = "no-positive-active-rt-observation"
        eligible = ineligibility is None
        if eligible:
            fit_inputs.append((service, metric, positive_s, censored))
        summaries.append(
            RtCellSummary(
                service=service,
                metric=metric,
                rt_cells=len(observations),
                duplicate_cells=duplicates,
                unpaired_cells=unpaired,
                active_cells=active,
                inactive_cells=inactive,
                positive_cells=len(positive_s),
                censored_cells=censored,
                eligible=eligible,
                ineligibility=ineligibility,
            )
        )
    return summaries


@dataclass(frozen=True)
class CallGraphScanResult:
    scan: MemberScanArtifact
    sample: CallGraphCorpus
    distinct_traces: int
    logical_calls: int
    rpc_type_counts: tuple[tuple[str, int], ...]
    root_services: tuple[str, ...]


def scan_v2021_call_graph(
    archive: Path,
    member: str,
    registry: SourceRegistry,
    *,
    workspace: Path,
    sample_traces: int,
    seed: int,
    shards: int = DEFAULT_SHARDS,
    progress: object = None,
) -> CallGraphScanResult:
    """Scan the complete call-graph member and emit a seeded trace-atomic sample.

    Sampling remains exactly the registered deterministic bottom-k rule over trace ids:
    identical ranking function, identical seed semantics, and still trace-atomic — a
    selected trace keeps every one of its calls.  Only the *materialisation* differs: the
    complete corpus is never held in memory, so the sample's provenance cites the sealed
    complete-member scan instead of a parent corpus object.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    if not isinstance(sample_traces, int) or isinstance(sample_traces, bool):
        raise TypeError("sample count must be an integer")
    if sample_traces <= 0:
        raise ValueError("sample count must be > 0")
    workspace.mkdir(parents=True, exist_ok=True)

    rejected: Counter[str] = Counter()
    rpc_types: Counter[str] = Counter()
    rows_read = 0
    accepted_rows = 0
    logical_calls = 0
    distinct_traces = 0
    root_services: set[str] = set()
    ranked: dict[str, bytes] = {}

    with tempfile.TemporaryDirectory(dir=workspace, prefix="call-shards-") as raw:
        writer = _ShardWriter(Path(raw), shards)
        reader_ref: _HashingReader | None = None
        fields = ("timestamp", "traceid", "rpcid", "um", "rpctype", "interface", "dm", "rt")
        for row_id, row, hashing in stream_tar_member_rows(
            archive, member, V2021_CALL_GRAPH_COLUMNS, "v2021 call-graph"
        ):
            reader_ref = hashing
            rows_read += 1
            reason = v2021_call_graph_row_rejection(row)
            if reason is not None:
                rejected[reason] += 1
                continue
            accepted_rows += 1
            trace_id = row["traceid"]
            if trace_id not in ranked:
                ranked[trace_id] = hashlib.sha256(f"{seed}:{trace_id}".encode()).digest()
            writer.write(trace_id, [str(row_id)] + [row[name] for name in fields])
            if progress is not None and rows_read % 1_000_000 == 0:
                print(f"  call graph: {rows_read:,} rows read", flush=True)
        if reader_ref is None:
            raise ValueError("v2021 call-graph member contained no data rows")
        content_sha256 = reader_ref.sha256
        content_bytes = reader_ref.byte_count
        shard_paths = writer.close()

        registry.require_component(V2021, CALIBRATION, CALL_GRAPH_COMPONENT, content_sha256)

        distinct_traces = len(ranked)
        if sample_traces > distinct_traces:
            raise ValueError(
                f"sample count {sample_traces} exceeds the {distinct_traces} available trace ids"
            )
        selected = set(sorted(ranked, key=ranked.__getitem__)[:sample_traces])
        ranked.clear()

        calls: list[LogicalCall] = []
        for shard in shard_paths:
            shard_calls, shard_rejections, shard_types, shard_roots = _reduce_call_shard(
                shard, selected, fields
            )
            logical_calls += shard_calls[0]
            calls.extend(shard_calls[1])
            rejected.update(shard_rejections)
            rpc_types.update(shard_types)
            root_services.update(shard_roots)
            os.unlink(shard)

    facts = {
        "distinct_trace_ids": str(distinct_traces),
        "accepted_rows": str(accepted_rows),
        "logical_calls": str(logical_calls),
        "identified_root_services": str(len(root_services)),
        "rpc_types": ",".join(f"{name}={count}" for name, count in sorted(rpc_types.items())),
        "sample_seed": str(seed),
        "sample_traces": str(sample_traces),
    }
    scan = _seal_scan(
        schema="alibaba-v2021-complete-member-scan-v1",
        source_id=V2021,
        component=CALL_GRAPH_COMPONENT,
        parser_input=member,
        content_sha256=content_sha256,
        content_bytes=content_bytes,
        source_registry_sha256=registry.manifest_sha256(),
        rows_read=rows_read,
        accepted=logical_calls,
        rejected_by_reason=tuple(sorted(rejected.items())),
        facts=tuple(sorted(facts.items())),
        truncation="none-complete-member-streamed",
        scan_code_version=code_version(),
        shards=shards,
    )

    ordered = tuple(sorted(calls, key=lambda call: (call.trace_id, call.rpc_id)))
    sample = issue_streamed_call_graph_sample(
        content_sha256=content_sha256,
        calls=ordered,
        report=IngestionReport(len(ordered), tuple(sorted(rejected.items()))),
        selection_seed=seed,
        requested_trace_count=sample_traces,
        parent_scan_seal=scan.seal,
        transform_code_version="streaming-bottom-k-trace-sampler-v1:" + code_version(),
    )

    return CallGraphScanResult(
        scan=scan,
        sample=sample,
        distinct_traces=distinct_traces,
        logical_calls=logical_calls,
        rpc_type_counts=tuple(sorted(rpc_types.items())),
        root_services=tuple(sorted(root_services)),
    )


def _reduce_call_shard(
    shard: Path, selected: set[str], fields: tuple[str, ...]
) -> tuple[tuple[int, list[LogicalCall]], Counter[str], Counter[str], set[str]]:
    """Collapse one trace-closed shard; retain only calls of the sampled traces."""
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for record in _read_shard(shard):
        row = dict(zip(fields, record[1:], strict=True))
        grouped[(row["traceid"], row["rpcid"])].append((int(record[0]), row))

    rejections: Counter[str] = Counter()
    rpc_types: Counter[str] = Counter()
    roots: set[str] = set()
    retained: list[LogicalCall] = []
    total = 0
    for (trace_id, rpc_id), rows in sorted(grouped.items()):
        call, reason = collapse_v2021_call_group(trace_id, rpc_id, rows)
        if call is None:
            rejections[reason or "malformed_row"] += len(rows)
            continue
        total += 1
        rpc_types[call.rpc_type] += 1
        if call.parent_rpc_id is None and call.downstream is not None:
            roots.add(call.downstream)
        if trace_id in selected:
            retained.append(call)
    return (total, retained), rejections, rpc_types, roots


def summarise_fit(fit: RtFitArtifact) -> dict[str, float]:
    """Report the fitted lognormal in interpretable seconds without re-deriving it."""
    params = dict(fit.params_seconds)
    mu, sigma = params["mu"], params["sigma"]
    return {
        "mu_log_s": mu,
        "sigma_log_s": sigma,
        "median_s": math.exp(mu),
        "mean_s": math.exp(mu + sigma**2 / 2.0),
        "p95_s": math.exp(mu + 1.6448536269514722 * sigma),
    }
