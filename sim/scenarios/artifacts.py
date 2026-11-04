"""Sealed, replayable scenario artifacts and public/private/pilot split controls."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sim.arrivals import episode_arrival_times
from sim.scenarios.alibaba import (
    AbsoluteRootRateArtifact,
    CallGraphCorpus,
    RtFitArtifact,
)
from sim.scenarios.drift import DriftPlan, assert_drift_applied
from sim.scenarios.sources import CALIBRATION, SourceRegistry, require_frozen_clean_checkout
from sim.scenarios.topology import (
    RoleBinding,
    ScalePolicy,
    TopologyArtifact,
    materialize_cluster,
)
from sim.spec import (
    CallSpec,
    EpisodeConfig,
    WorkloadSpec,
    config_sha256,
    episode_config_from_json,
    episode_config_to_json,
)

PUBLIC = "public-tuning"
PRIVATE = "private-confirmatory"
PILOT = "pilot-quarantined"
VISIBILITIES = (PUBLIC, PRIVATE, PILOT)
IN_DISTRIBUTION = "in-distribution"
HELD_OUT_FAMILY = "held-out-family"
SYNTHETIC_STRESS = "synthetic-stress"
EVIDENCE_ROLES = (IN_DISTRIBUTION, HELD_OUT_FAMILY, SYNTHETIC_STRESS)
_ARTIFACT_TAG = "replayable-scenario-v1:"
_SPLIT_TAG = "scenario-split-manifest-v1:"
_RECIPE_TAG = "scenario-recipe-v1:"
_SCENARIO_SEEDS_TAG = "scenario-generator-seeds-v1:"
_WORKLOAD_TAG = "scenario-workload-model-v1:"
_ARTIFACT_AUTHORITY = object()
_SPLIT_AUTHORITY = object()
_WORKLOAD_AUTHORITY = object()
_DESCRIPTOR_AUTHORITY = object()
_P2_TAG = "scenario-p2-access-v1:"
_P2_PROTOCOL_AUTHORITY = object()
_P2_RECEIPT_AUTHORITY = object()
_P2_REGISTRATION_AUTHORITY = object()
P2_RECEIPT_PATH = Path(__file__).resolve().parent / "data" / "p2_scenario_receipt.json"
PRODUCTION_P2_RECEIPT_SHA256: str | None = None
MIB = 1024.0 * 1024.0


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


def seed_commitment(seed: int) -> str:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an explicit int")
    # Deliberately independent of scenario id: the same generator seed reused in two
    # visibility groups must yield the same commitment and fail split disjointness.
    return hashlib.sha256(f"scenario-seed-v1:{seed}".encode()).hexdigest()


def _sha256_hex(name: str, value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _max_call_bytes(call: CallSpec, visits: int = 1) -> float:
    total = 0.0
    for edge in call.children:
        attempts = edge.max_retries + 1
        payload_kb = edge.request_kb + (edge.response_kb if call.mode == "blocking" else 0.0)
        child_visits = visits * attempts
        total += visits * attempts * payload_kb * 1000.0
        total += _max_call_bytes(edge.child, child_visits)
    return total


def structural_network_byte_ceiling(config: EpisodeConfig) -> float:
    """Finite policy-independent byte bound for a realized Phase-4 episode.

    It assumes every probabilistic edge fires, every bounded retry is used, every
    possible initial/onboarding/HPA pod start performs its own image pull (even when
    concurrent pulls have not populated the cache yet), and the registered migration
    byte cap is exhausted. Planned cordon/uncordon causes no replacement.
    """
    if any(not template.arrival.arrivals_realized for template in config.workload.templates):
        raise ValueError("finite scenario byte ceiling requires all arrivals to be realized")
    request_bytes = math.fsum(
        len(template.arrival.realized_times_s) * _max_call_bytes(template.root)
        for template in config.workload.templates
    )
    image_bytes = 0.0
    if config.energy.include_image_pull_bytes:
        hpa_syncs = (
            math.ceil(config.horizon_s / config.hpa.sync_period_s)
            if config.hpa.enabled
            else 0
        )
        image_bytes = math.fsum(
            service.image_mb
            * (service.initial_replicas + hpa_syncs * service.max_replicas)
            * MIB
            for service in config.workload.services
        )
    migration_bytes = (
        0.0
        if config.rebalancer.max_total_bytes_mb is None
        else config.rebalancer.max_total_bytes_mb * MIB
    )
    if config.rebalancer.enabled and config.rebalancer.max_total_bytes_mb is None:
        raise ValueError(
            "finite scenario byte ceiling requires max_total_bytes_mb when rebalancing"
        )
    return request_bytes + image_bytes + migration_bytes


def _workload_sha256(config: EpisodeConfig) -> str:
    payload = json.loads(episode_config_to_json(config))["workload"]
    # Arrival realizations are generated only after the model closure is sealed; the
    # scenario artifact separately closes their exact times/counts and episode seed.
    for template in payload["templates"]:
        template["arrival"].pop("realized_times_s", None)
        template["arrival"].pop("arrivals_realized", None)
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class WorkloadEvidenceRef:
    kind: str
    source_id: str
    source_content_sha256: str
    artifact_sha256: str
    selection: str


@dataclass(frozen=True)
class WorkloadModelClosure:
    """Exact provenance closure for the workload embedded in a scenario.

    A closure is either explicitly synthetic or is issued only from sealed Alibaba
    transform artifacts.  In both cases it binds the exact post-drift ``WorkloadSpec``
    that scenario sealing later realizes.
    """

    schema: str
    provenance_label: str
    workload_sha256: str
    source_registry_sha256: str
    evidence: tuple[WorkloadEvidenceRef, ...]
    transform_config: tuple[tuple[str, str], ...]
    component_seed_commitments: tuple[tuple[str, str], ...]
    code_versions: tuple[str, ...]
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }

    def verify_closed(self, config: EpisodeConfig, source_registry_sha256: str) -> None:
        if self._authority is not _WORKLOAD_AUTHORITY:
            raise ValueError("WorkloadModelClosure must be issued by a workload sealer")
        if self.schema != "scenario-workload-model-v1":
            raise ValueError("unknown workload-model closure schema")
        if self.provenance_label not in (
            "synthetic-scenario-assumption",
            "calibration-evidence-citing-synthetic-assumption",
            "calibration-informed-synthetic-workload",
        ):
            raise ValueError("unknown workload provenance label")
        _sha256_hex("workload source registry", source_registry_sha256)
        if self.source_registry_sha256 != source_registry_sha256:
            raise ValueError("workload closure source registry mismatch")
        if self.workload_sha256 != _workload_sha256(config):
            raise ValueError("workload closure does not match the exact EpisodeConfig workload")
        if self.evidence != tuple(sorted(self.evidence)):
            raise ValueError("workload evidence must be canonically sorted")
        if self.provenance_label == "synthetic-scenario-assumption" and self.evidence:
            raise ValueError("synthetic workload closure cannot cite empirical evidence")
        if (
            self.provenance_label
            in (
                "calibration-evidence-citing-synthetic-assumption",
                "calibration-informed-synthetic-workload",
            )
            and not self.evidence
        ):
            raise ValueError("evidence-citing workload needs sealed calibration artifacts")
        if self.provenance_label == "calibration-informed-synthetic-workload":
            kinds = {item.kind for item in self.evidence}
            required = {
                "absolute-root-rate-shape",
                "call-graph-sample",
                "interval-mean-rt-fit",
            }
            if not required <= kinds:
                raise ValueError("calibration-informed workload lacks its typed evidence kinds")
            if not any(key == "typed_builder_sha256" for key, _ in self.transform_config):
                raise ValueError("calibration-informed workload lacks its typed builder seal")
        for item in self.evidence:
            _sha256_hex("workload evidence source payload", item.source_content_sha256)
            _sha256_hex("workload evidence artifact", item.artifact_sha256)
            if not item.kind or not item.selection:
                raise ValueError("workload evidence kind/selection must be non-empty")
        for sequence, name in (
            (self.transform_config, "workload transform config"),
            (self.component_seed_commitments, "workload component seeds"),
        ):
            if sequence != tuple(sorted(sequence)):
                raise ValueError(f"{name} must be canonically sorted")
            keys = [key for key, _ in sequence]
            if len(keys) != len(set(keys)) or any(not key for key in keys):
                raise ValueError(f"{name} namespaces must be non-empty and unique")
        for _, commitment in self.component_seed_commitments:
            _sha256_hex("workload component seed commitment", commitment)
        if not self.code_versions or any(not version for version in self.code_versions):
            raise ValueError("workload closure needs non-empty transform code versions")
        expected = hashlib.sha256(
            (_WORKLOAD_TAG + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("workload-model closure digest mismatch")

    def verify(self, config: EpisodeConfig, registry: SourceRegistry) -> None:
        registry.verify_seal()
        self.verify_closed(config, registry.manifest_sha256())
        for item in self.evidence:
            registry.require_payload(item.source_id, CALIBRATION, item.source_content_sha256)


def _seal_workload_payload(payload: dict[str, object]) -> WorkloadModelClosure:
    digest = hashlib.sha256((_WORKLOAD_TAG + _canonical(payload)).encode()).hexdigest()
    closure = WorkloadModelClosure(
        **payload, digest=digest, _authority=_WORKLOAD_AUTHORITY  # type: ignore[arg-type]
    )
    return closure


def workload_model_to_json(model: WorkloadModelClosure) -> str:
    return _canonical({**model._payload(), "digest": model.digest})


def workload_model_from_json(
    text: str, config: EpisodeConfig, source_registry_sha256: str
) -> WorkloadModelClosure:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("workload-model closure is not valid JSON") from exc
    expected_fields = {
        field.name
        for field in dataclasses.fields(WorkloadModelClosure)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("workload-model closure fields do not match the schema exactly")
    payload["evidence"] = tuple(WorkloadEvidenceRef(**item) for item in payload["evidence"])
    for name in ("transform_config", "component_seed_commitments"):
        payload[name] = tuple(tuple(item) for item in payload[name])
    payload["code_versions"] = tuple(payload["code_versions"])
    model = WorkloadModelClosure(**payload, _authority=_WORKLOAD_AUTHORITY)
    model.verify_closed(config, source_registry_sha256)
    if workload_model_to_json(model) != text:
        raise ValueError("workload-model closure JSON is not canonical")
    return model


def seal_synthetic_workload(
    config: EpisodeConfig,
    registry: SourceRegistry,
    *,
    generator_version: str,
    component_seeds: tuple[tuple[str, int], ...] = (),
    transform_config: tuple[tuple[str, str], ...] = (),
) -> WorkloadModelClosure:
    """Declare a generated workload as synthetic and close its exact bytes/seeds."""
    if not generator_version:
        raise ValueError("synthetic workload generator version must be non-empty")
    payload = {
        "schema": "scenario-workload-model-v1",
        "provenance_label": "synthetic-scenario-assumption",
        "workload_sha256": _workload_sha256(config),
        "source_registry_sha256": registry.manifest_sha256(),
        "evidence": (),
        "transform_config": tuple(sorted(transform_config)),
        "component_seed_commitments": tuple(
            sorted((namespace, seed_commitment(seed)) for namespace, seed in component_seeds)
        ),
        "code_versions": (generator_version,),
    }
    closure = _seal_workload_payload(payload)
    closure.verify(config, registry)
    return closure


def seal_evidence_citing_synthetic_workload(
    config: EpisodeConfig,
    registry: SourceRegistry,
    *,
    call_graph_samples: tuple[CallGraphCorpus, ...] = (),
    rt_fits: tuple[RtFitArtifact, ...] = (),
    rate_shapes: tuple[AbsoluteRootRateArtifact, ...] = (),
    transform_config: tuple[tuple[str, str], ...],
    component_seeds: tuple[tuple[str, int], ...] = (),
    code_versions: tuple[str, ...],
) -> WorkloadModelClosure:
    """Close a synthetic workload that cites—but is not claimed derived from—evidence."""
    evidence: list[WorkloadEvidenceRef] = []
    for sample in call_graph_samples:
        sample.verify_seal()
        if sample.selection_seed is None or sample.requested_trace_count is None:
            raise ValueError("empirical call-graph input must be an explicitly seeded sample")
        evidence.append(
            WorkloadEvidenceRef(
                "call-graph-sample",
                sample.source_id,
                sample.content_sha256,
                sample.seal,
                f"{sample.selection_method}; count={sample.requested_trace_count}",
            )
        )
    for fit in rt_fits:
        fit.verify_seal()
        evidence.append(
            WorkloadEvidenceRef(
                "interval-mean-rt-fit",
                fit.source_id,
                fit.source_content_sha256,
                fit.seal,
                fit.row_selection,
            )
        )
    for shape in rate_shapes:
        shape.verify_seal()
        evidence.append(
            WorkloadEvidenceRef(
                "absolute-root-rate-shape",
                shape.source_id,
                shape.source_content_sha256,
                shape.seal,
                shape.selection_basis,
            )
        )
    payload = {
        "schema": "scenario-workload-model-v1",
        "provenance_label": "calibration-evidence-citing-synthetic-assumption",
        "workload_sha256": _workload_sha256(config),
        "source_registry_sha256": registry.manifest_sha256(),
        "evidence": tuple(sorted(evidence)),
        "transform_config": tuple(sorted(transform_config)),
        "component_seed_commitments": tuple(
            sorted((namespace, seed_commitment(seed)) for namespace, seed in component_seeds)
        ),
        "code_versions": tuple(code_versions),
    }
    closure = _seal_workload_payload(payload)
    closure.verify(config, registry)
    return closure


def seal_calibration_informed_synthetic_workload(
    config: EpisodeConfig,
    registry: SourceRegistry,
    *,
    build: object | None = None,
) -> WorkloadModelClosure:
    """Seal only a workload re-derived by the authority-bearing Alibaba builder."""
    # Local imports avoid making the generic artifact schema depend on one source family.
    from sim.scenarios.alibaba import CALL_GRAPH_COMPONENT, METRICS_COMPONENT, V2021
    from sim.scenarios.alibaba_workload import (
        CALL_GRAPH_SAMPLE_SEAL,
        CALL_GRAPH_SCAN_SEAL,
        METRICS_SCAN_SEAL,
        ROOT_RATE_SEAL,
        TRACE_SAMPLE_SEED,
        CalibrationInformedWorkload,
    )

    if build is None:
        raise ValueError(
            "calibration-informed workload sealing remains fail-closed without "
            "an authority-bearing typed builder result"
        )
    if not isinstance(build, CalibrationInformedWorkload):
        raise TypeError("calibration-informed sealing requires a typed builder result")
    build.verify(registry)
    if config.workload != build.workload:
        raise ValueError("EpisodeConfig workload was not produced by the cited typed builder")
    source = registry.require_role(V2021, CALIBRATION)
    metrics_digest = source.component(METRICS_COMPONENT).sha256
    call_digest = source.component(CALL_GRAPH_COMPONENT).sha256
    if metrics_digest is None or call_digest is None:
        raise ValueError("Alibaba calibration components are unavailable")
    evidence = [
        WorkloadEvidenceRef(
            "complete-member-scan",
            V2021,
            call_digest,
            CALL_GRAPH_SCAN_SEAL,
            "call_graph; none-complete-member-streamed",
        ),
        WorkloadEvidenceRef(
            "call-graph-sample",
            V2021,
            call_digest,
            CALL_GRAPH_SAMPLE_SEAL,
            f"streaming-bottom-k-trace-sha256-v1; count=1000; seed={TRACE_SAMPLE_SEED}",
        ),
        WorkloadEvidenceRef(
            "complete-member-scan",
            V2021,
            metrics_digest,
            METRICS_SCAN_SEAL,
            "metrics; none-complete-member-streamed",
        ),
        WorkloadEvidenceRef(
            "absolute-root-rate-shape",
            V2021,
            metrics_digest,
            ROOT_RATE_SEAL,
            "identified-entry-root-only; synthetic target positive mean=50 rps",
        ),
    ]
    evidence.extend(
        WorkloadEvidenceRef(
            "interval-mean-rt-fit",
            V2021,
            metrics_digest,
            seal,
            f"selected fitted service={service}; interval-mean response-time proxy",
        )
        for service, seal in build.selected_fit_seals
    )
    payload = {
        "schema": "scenario-workload-model-v1",
        "provenance_label": "calibration-informed-synthetic-workload",
        "workload_sha256": _workload_sha256(config),
        "source_registry_sha256": registry.manifest_sha256(),
        "evidence": tuple(sorted(evidence)),
        "transform_config": tuple(
            sorted((*build.transform_config, ("typed_builder_sha256", build.digest)))
        ),
        "component_seed_commitments": tuple(
            sorted(
                (
                    ("builder-selection", seed_commitment(build.config.seed)),
                    ("call-graph-trace-sample", seed_commitment(TRACE_SAMPLE_SEED)),
                )
            )
        ),
        "code_versions": build.code_versions,
    }
    closure = _seal_workload_payload(payload)
    closure.verify(config, registry)
    return closure


def seal_empirical_workload(*args: object, **kwargs: object) -> WorkloadModelClosure:
    """Fail closed until a typed builder can derive WorkloadSpec from raw evidence."""
    del args, kwargs
    raise ValueError(
        "empirical workload sealing is unavailable: use the explicitly synthetic "
        "evidence-citing closure until a typed deterministic builder exists"
    )


@dataclass(frozen=True)
class ScenarioArtifact:
    schema: str
    scenario_id: str
    visibility: str
    topology_evidence_role: str
    topology_artifact_sha256: str
    topology_evidence_label: str
    scenario_seed_commitment: str
    generator_seed_commitments: tuple[tuple[str, str], ...]
    episode_seed: int
    episode_seed_commitment: str
    recipe_sha256: str
    generator_version: str
    source_registry_sha256: str
    workload_model_sha256: str
    workload_provenance_label: str
    workload_model_json: str
    drift_sha256: str
    episode_config_json: str
    episode_config_sha256: str
    offered_counts: tuple[tuple[int, int], ...]
    network_byte_ceiling: float
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("digest", "_authority")
        }

    def verify(self) -> EpisodeConfig:
        if self._authority is not _ARTIFACT_AUTHORITY:
            raise ValueError("ScenarioArtifact must be issued by seal_scenario()")
        if self.schema != "replayable-scenario-v1":
            raise ValueError("unknown scenario artifact schema")
        if self.visibility not in VISIBILITIES:
            raise ValueError(f"scenario visibility must be one of {VISIBILITIES}")
        if self.topology_evidence_role not in EVIDENCE_ROLES:
            raise ValueError(f"topology evidence role must be one of {EVIDENCE_ROLES}")
        expected_role = {
            "hierarchical-tiered synthetic scenario": IN_DISTRIBUTION,
            "empirically grounded network-structure generalization": HELD_OUT_FAMILY,
            "synthetic stress test": SYNTHETIC_STRESS,
        }.get(self.topology_evidence_label)
        if self.topology_evidence_role != expected_role:
            raise ValueError("scenario topology label/evidence role mismatch")
        if self.workload_provenance_label not in (
            "synthetic-scenario-assumption",
            "calibration-evidence-citing-synthetic-assumption",
            "calibration-informed-synthetic-workload",
        ):
            raise ValueError("scenario workload provenance label is invalid")
        for name in (
            "topology_artifact_sha256",
            "scenario_seed_commitment",
            "episode_seed_commitment",
            "recipe_sha256",
            "source_registry_sha256",
            "workload_model_sha256",
            "drift_sha256",
            "episode_config_sha256",
            "digest",
        ):
            _sha256_hex(name, getattr(self, name))
        if self.generator_seed_commitments != tuple(sorted(self.generator_seed_commitments)):
            raise ValueError("scenario generator seed commitments must be canonically sorted")
        namespaces = [namespace for namespace, _ in self.generator_seed_commitments]
        if len(namespaces) != len(set(namespaces)) or any(
            not namespace for namespace in namespaces
        ):
            raise ValueError("scenario generator seed namespaces must be non-empty and unique")
        for _, commitment in self.generator_seed_commitments:
            _sha256_hex("generator component seed commitment", commitment)
        expected_scenario_commitment = hashlib.sha256(
            (_SCENARIO_SEEDS_TAG + _canonical(self.generator_seed_commitments)).encode()
        ).hexdigest()
        if self.scenario_seed_commitment != expected_scenario_commitment:
            raise ValueError("scenario aggregate generator-seed commitment mismatch")
        if self.episode_seed_commitment != seed_commitment(self.episode_seed):
            raise ValueError("scenario episode-seed commitment mismatch")
        config = episode_config_from_json(self.episode_config_json)
        if episode_config_to_json(config) != self.episode_config_json:
            raise ValueError("scenario EpisodeConfig JSON is not exact canonical encoding")
        if config_sha256(config) != self.episode_config_sha256:
            raise ValueError("scenario EpisodeConfig hash mismatch")
        workload_model = workload_model_from_json(
            self.workload_model_json, config, self.source_registry_sha256
        )
        if (
            workload_model.digest != self.workload_model_sha256
            or workload_model.provenance_label != self.workload_provenance_label
        ):
            raise ValueError("scenario workload-model closure mismatch")
        counts = tuple(
            (template.id, len(template.arrival.realized_times_s))
            for template in config.workload.templates
        )
        if counts != self.offered_counts:
            raise ValueError("scenario offered-count closure mismatch")
        bound = structural_network_byte_ceiling(config)
        if bound != self.network_byte_ceiling:
            raise ValueError("scenario network-byte ceiling mismatch")
        expected = hashlib.sha256(
            (_ARTIFACT_TAG + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("scenario artifact digest mismatch")
        return config


def seal_scenario(
    *,
    scenario_id: str,
    visibility: str,
    topology: TopologyArtifact,
    scale_policy: ScalePolicy,
    topology_bindings: tuple[RoleBinding, ...],
    drift_plan: DriftPlan,
    workload_model: WorkloadModelClosure,
    source_registry: SourceRegistry,
    episode_seed: int,
    generator_version: str,
    config: EpisodeConfig,
) -> ScenarioArtifact:
    if not scenario_id or not generator_version:
        raise ValueError("scenario id and generator version must be non-empty")
    topology.verify(scale_policy)
    drift_plan.verify()
    source_registry.verify_seal()
    assert_drift_applied(config, drift_plan)
    workload_model.verify(config, source_registry)
    if config.cluster != materialize_cluster(topology, scale_policy, topology_bindings):
        raise ValueError("scenario config cluster does not match its topology artifact/bindings")
    topology_evidence_role = {
        "hierarchical-tiered": IN_DISTRIBUTION,
        "topology-zoo-derived": HELD_OUT_FAMILY,
        "barabasi-albert": SYNTHETIC_STRESS,
    }[topology.family]
    source_registry_sha256 = source_registry.manifest_sha256()
    base_config_sha256 = config_sha256(config)
    generator_seed_commitments = tuple(
        sorted(
            (
                ("topology", seed_commitment(topology.seed)),
                ("drift-root", seed_commitment(drift_plan.seed)),
                *(
                    (namespace, seed_commitment(seed))
                    for namespace, seed in drift_plan.component_seeds
                ),
                *workload_model.component_seed_commitments,
            )
        )
    )
    scenario_seed_sha256 = hashlib.sha256(
        (_SCENARIO_SEEDS_TAG + _canonical(generator_seed_commitments)).encode()
    ).hexdigest()
    recipe = {
        "generator_version": generator_version,
        "topology_artifact_sha256": topology.digest,
        "scale_policy_sha256": scale_policy.seal,
        "drift_sha256": drift_plan.digest,
        "source_registry_sha256": source_registry_sha256,
        "workload_model_sha256": workload_model.digest,
        "base_episode_config_sha256": base_config_sha256,
        "scenario_seed_material_sha256": scenario_seed_sha256,
    }
    recipe_sha256 = hashlib.sha256((_RECIPE_TAG + _canonical(recipe)).encode()).hexdigest()
    templates = tuple(
        replace(
            template,
            arrival=replace(
                template.arrival,
                realized_times_s=episode_arrival_times(
                    template.arrival, config.cutoff_s, episode_seed, template.id
                ),
                arrivals_realized=True,
            ),
        )
        for template in config.workload.templates
    )
    realized_config = replace(
        config,
        workload=WorkloadSpec(
            services=config.workload.services,
            templates=templates,
            backends=config.workload.backends,
        ),
    )
    config_json = episode_config_to_json(realized_config)
    offered_counts = tuple(
        (template.id, len(template.arrival.realized_times_s)) for template in templates
    )
    payload = {
        "schema": "replayable-scenario-v1",
        "scenario_id": scenario_id,
        "visibility": visibility,
        "topology_evidence_role": topology_evidence_role,
        "topology_artifact_sha256": topology.digest,
        "topology_evidence_label": topology.evidence_label,
        "scenario_seed_commitment": scenario_seed_sha256,
        "generator_seed_commitments": generator_seed_commitments,
        "episode_seed": episode_seed,
        "episode_seed_commitment": seed_commitment(episode_seed),
        "recipe_sha256": recipe_sha256,
        "generator_version": generator_version,
        "source_registry_sha256": source_registry_sha256,
        "workload_model_sha256": workload_model.digest,
        "workload_provenance_label": workload_model.provenance_label,
        "workload_model_json": workload_model_to_json(workload_model),
        "drift_sha256": drift_plan.digest,
        "episode_config_json": config_json,
        "episode_config_sha256": config_sha256(realized_config),
        "offered_counts": offered_counts,
        "network_byte_ceiling": structural_network_byte_ceiling(realized_config),
    }
    digest = hashlib.sha256((_ARTIFACT_TAG + _canonical(payload)).encode()).hexdigest()
    artifact = ScenarioArtifact(
        **payload, digest=digest, _authority=_ARTIFACT_AUTHORITY  # type: ignore[arg-type]
    )
    artifact.verify()
    return artifact


def run_scenario_artifact(
    artifact: ScenarioArtifact,
    scorer: Any,
    proposer: Any | None = None,
    lb_policy: str = "round_robin",
) -> Any:
    """Development runner for public/pilot artifacts, always using the sealed seed."""
    from sim.engine import run_episode

    config = artifact.verify()
    if artifact.visibility == PRIVATE:
        raise PermissionError(
            "private-confirmatory scenarios require run_private_scenario_artifact() "
            "with a P2 access receipt"
        )
    return run_episode(
        config,
        seed=artifact.episode_seed,
        scorer=scorer,
        proposer=proposer,
        lb_policy=lb_policy,
    )


def scenario_artifact_to_json(artifact: ScenarioArtifact) -> str:
    """Canonical durable representation; authority tokens are never serialized."""
    artifact.verify()
    return _canonical({**artifact._payload(), "digest": artifact.digest})


def scenario_artifact_from_json(text: str) -> ScenarioArtifact:
    """Load a public/pilot artifact; private payloads require ``P2ConfirmationGate``."""
    return _scenario_artifact_from_json(text, allow_private=False)


def _scenario_artifact_from_json(text: str, *, allow_private: bool) -> ScenarioArtifact:
    """Recompute every durable closure and reissue authority under a role gate."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("scenario artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("scenario artifact JSON must be an object")
    expected_fields = {
        field.name for field in dataclasses.fields(ScenarioArtifact) if field.name != "_authority"
    }
    if set(payload) != expected_fields:
        raise ValueError("scenario artifact JSON fields do not match the schema exactly")
    for name in ("generator_seed_commitments", "offered_counts"):
        payload[name] = tuple(tuple(item) for item in payload[name])
    artifact = ScenarioArtifact(**payload, _authority=_ARTIFACT_AUTHORITY)
    artifact.verify()
    if artifact.visibility == PRIVATE and not allow_private:
        raise PermissionError("private scenario loading requires P2ConfirmationGate")
    if scenario_artifact_to_json(artifact) != text:
        raise ValueError("scenario artifact JSON is not canonical")
    return artifact


def save_scenario_artifact(artifact: ScenarioArtifact, path: Path) -> None:
    path.write_text(scenario_artifact_to_json(artifact), encoding="utf-8")


def load_scenario_artifact(path: Path) -> ScenarioArtifact:
    return scenario_artifact_from_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ScenarioDescriptor:
    scenario_id: str
    visibility: str
    topology_evidence_role: str
    seed_commitment: str
    generator_seed_commitments: tuple[tuple[str, str], ...]
    episode_seed_commitment: str
    recipe_sha256: str
    artifact_sha256: str | None
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "_authority"
        }

    def verify(self) -> None:
        if self._authority is not _DESCRIPTOR_AUTHORITY:
            raise ValueError("ScenarioDescriptor must be derived from a sealed artifact")
        if self.visibility not in VISIBILITIES or self.topology_evidence_role not in EVIDENCE_ROLES:
            raise ValueError("scenario descriptor has an invalid role")
        _sha256_hex("scenario seed commitment", self.seed_commitment)
        _sha256_hex("episode seed commitment", self.episode_seed_commitment)
        _sha256_hex("scenario recipe", self.recipe_sha256)
        if self.artifact_sha256 is not None:
            _sha256_hex("scenario artifact", self.artifact_sha256)
        if self.generator_seed_commitments != tuple(sorted(self.generator_seed_commitments)):
            raise ValueError("descriptor generator seeds must be canonically sorted")
        for namespace, commitment in self.generator_seed_commitments:
            if not namespace:
                raise ValueError("descriptor generator seed namespace is empty")
            _sha256_hex("descriptor generator seed commitment", commitment)


def _descriptor_from_artifact(
    artifact: ScenarioArtifact, *, disclose_hash: bool
) -> ScenarioDescriptor:
    artifact.verify()
    descriptor = ScenarioDescriptor(
        artifact.scenario_id,
        artifact.visibility,
        artifact.topology_evidence_role,
        artifact.scenario_seed_commitment,
        artifact.generator_seed_commitments,
        artifact.episode_seed_commitment,
        artifact.recipe_sha256,
        artifact.digest if disclose_hash else None,
        _authority=_DESCRIPTOR_AUTHORITY,
    )
    descriptor.verify()
    return descriptor


def private_scenario_commitment(artifact: ScenarioArtifact) -> ScenarioDescriptor:
    """Prepare a non-payload commitment in the separate private-generation step."""
    if artifact.visibility != PRIVATE:
        raise ValueError("private commitment requires a private-confirmatory artifact")
    return _descriptor_from_artifact(artifact, disclose_hash=False)


@dataclass(frozen=True)
class SplitManifest:
    schema: str
    public: tuple[ScenarioDescriptor, ...]
    private_commitments: tuple[ScenarioDescriptor, ...]
    pilot: tuple[ScenarioDescriptor, ...]
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "public": [item._payload() for item in self.public],
            "private_commitments": [item._payload() for item in self.private_commitments],
            "pilot": [item._payload() for item in self.pilot],
        }

    def verify(self) -> None:
        if self._authority is not _SPLIT_AUTHORITY:
            raise ValueError("SplitManifest must be issued by build_split_manifest()")
        if self.schema != "scenario-split-manifest-v1":
            raise ValueError("unknown scenario split manifest schema")
        groups = (self.public, self.private_commitments, self.pilot)
        expected_visibility = (PUBLIC, PRIVATE, PILOT)
        all_ids: list[str] = []
        all_commitments: list[str] = []
        all_episode_commitments: list[str] = []
        for group, visibility in zip(groups, expected_visibility, strict=True):
            if tuple(sorted(group, key=lambda item: item.scenario_id)) != group:
                raise ValueError("split groups must be sorted by scenario id")
            for item in group:
                item.verify()
                if item.visibility != visibility:
                    raise ValueError("split descriptor is in the wrong visibility group")
                if visibility == PRIVATE and item.artifact_sha256 is not None:
                    raise ValueError("normal split manifest must not expose private payload hashes")
                if visibility != PRIVATE and item.artifact_sha256 is None:
                    raise ValueError("public/pilot descriptors require full artifact hashes")
                all_ids.append(item.scenario_id)
                all_commitments.append(item.seed_commitment)
                all_episode_commitments.append(item.episode_seed_commitment)
        if (
            len(all_ids) != len(set(all_ids))
            or len(all_commitments) != len(set(all_commitments))
            or len(all_episode_commitments) != len(set(all_episode_commitments))
        ):
            raise ValueError(
                "scenario ids, generator-seed commitments, and episode-seed commitments "
                "must be disjoint across splits"
            )
        recipe_sets = [{item.recipe_sha256 for item in group} for group in groups]
        if any(
            recipe_sets[left] & recipe_sets[right]
            for left in range(len(recipe_sets))
            for right in range(left + 1, len(recipe_sets))
        ):
            raise ValueError("scenario recipes must be disjoint across visibility splits")
        component_seed_sets = [
            {commitment for item in group for _, commitment in item.generator_seed_commitments}
            for group in groups
        ]
        if any(
            component_seed_sets[left] & component_seed_sets[right]
            for left in range(len(component_seed_sets))
            for right in range(left + 1, len(component_seed_sets))
        ):
            raise ValueError("generator component seeds must be disjoint across splits")
        expected = hashlib.sha256((_SPLIT_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("scenario split manifest digest mismatch")


def build_split_manifest(
    public: tuple[ScenarioArtifact, ...],
    private_commitments: tuple[ScenarioDescriptor, ...],
    pilot: tuple[ScenarioArtifact, ...],
) -> SplitManifest:
    for item in private_commitments:
        item.verify()
        if item.visibility != PRIVATE or item.artifact_sha256 is not None:
            raise ValueError("normal split construction accepts private commitments only")

    payload = {
        "schema": "scenario-split-manifest-v1",
        "public": tuple(
            sorted(
                (_descriptor_from_artifact(item, disclose_hash=True) for item in public),
                key=lambda item: item.scenario_id,
            )
        ),
        "private_commitments": tuple(
            sorted(private_commitments, key=lambda item: item.scenario_id)
        ),
        "pilot": tuple(
            sorted(
                (_descriptor_from_artifact(item, disclose_hash=True) for item in pilot),
                key=lambda item: item.scenario_id,
            )
        ),
    }
    canonical_payload = {
        "schema": payload["schema"],
        "public": [item._payload() for item in payload["public"]],
        "private_commitments": [
            item._payload() for item in payload["private_commitments"]
        ],
        "pilot": [item._payload() for item in payload["pilot"]],
    }
    digest = hashlib.sha256((_SPLIT_TAG + _canonical(canonical_payload)).encode()).hexdigest()
    manifest = SplitManifest(
        **payload, digest=digest, _authority=_SPLIT_AUTHORITY  # type: ignore[arg-type]
    )
    manifest.verify()
    return manifest


def split_manifest_to_json(manifest: SplitManifest) -> str:
    """Canonical durable split manifest; descriptor authority is never serialized."""
    manifest.verify()
    return _canonical({**manifest._payload(), "digest": manifest.digest})


def split_manifest_from_json(text: str) -> SplitManifest:
    """Reload and fully revalidate a canonical split manifest."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("split manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "public",
        "private_commitments",
        "pilot",
        "digest",
    }:
        raise ValueError("split manifest JSON fields do not match the schema exactly")
    descriptor_fields = {
        field.name
        for field in dataclasses.fields(ScenarioDescriptor)
        if field.name != "_authority"
    }

    def parse_group(name: str) -> tuple[ScenarioDescriptor, ...]:
        raw = payload[name]
        if not isinstance(raw, list):
            raise ValueError(f"split manifest {name} must be a JSON list")
        descriptors: list[ScenarioDescriptor] = []
        for item in raw:
            if not isinstance(item, dict) or set(item) != descriptor_fields:
                raise ValueError("split descriptor JSON fields do not match the schema exactly")
            parsed = dict(item)
            parsed["generator_seed_commitments"] = tuple(
                tuple(pair) for pair in parsed["generator_seed_commitments"]
            )
            descriptor = ScenarioDescriptor(
                **parsed, _authority=_DESCRIPTOR_AUTHORITY  # type: ignore[arg-type]
            )
            descriptor.verify()
            descriptors.append(descriptor)
        return tuple(descriptors)

    manifest = SplitManifest(
        schema=payload["schema"],
        public=parse_group("public"),
        private_commitments=parse_group("private_commitments"),
        pilot=parse_group("pilot"),
        digest=payload["digest"],
        _authority=_SPLIT_AUTHORITY,
    )
    manifest.verify()
    if split_manifest_to_json(manifest) != text:
        raise ValueError("split manifest JSON is not canonical")
    return manifest


@dataclass(frozen=True)
class P2RegistrationReceipt:
    schema: str
    receipt_id: str
    registration_id: str
    authority_url: str
    frozen_commit: str
    registered_at_utc: str
    split_manifest_sha256: str
    receipt_sha256: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify(self) -> None:
        if self._authority is not _P2_REGISTRATION_AUTHORITY:
            raise ValueError("P2 registration receipt must be loaded from exact bytes")
        if self.schema != "scenario-p2-registration-receipt-v1":
            raise ValueError("unknown P2 registration receipt schema")
        if not self.receipt_id or not self.registration_id or not self.authority_url.startswith(
            "https://"
        ):
            raise ValueError("P2 receipt needs ids and an HTTPS registration authority")
        if len(self.frozen_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.frozen_commit
        ):
            raise ValueError("P2 receipt frozen_commit must be a full SHA-1 object id")
        registered = datetime.fromisoformat(self.registered_at_utc)
        if registered.tzinfo is None or registered.utcoffset() != timedelta(0):
            raise ValueError("P2 registration timestamp must use the UTC offset")
        _sha256_hex("P2 split manifest", self.split_manifest_sha256)
        _sha256_hex("P2 registration receipt", self.receipt_sha256)


def _parse_p2_registration_receipt(
    receipt_content: bytes, manifest: SplitManifest
) -> P2RegistrationReceipt:
    manifest.verify()
    try:
        payload = json.loads(receipt_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("P2 registration receipt must be exact UTF-8 JSON bytes") from exc
    required = {
        "schema",
        "receipt_id",
        "registration_id",
        "authority_url",
        "frozen_commit",
        "registered_at_utc",
        "split_manifest_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or _canonical(payload).encode() != receipt_content
        or payload["split_manifest_sha256"] != manifest.digest
    ):
        raise ValueError("P2 receipt does not canonically bind this split manifest")
    receipt = P2RegistrationReceipt(
        **payload,
        receipt_sha256=hashlib.sha256(receipt_content).hexdigest(),
        _authority=_P2_REGISTRATION_AUTHORITY,
    )
    receipt.verify()
    return receipt


def load_p2_registration_receipt(manifest: SplitManifest) -> P2RegistrationReceipt:
    """Load only the repository-pinned external P2 receipt; fail closed until frozen."""
    if PRODUCTION_P2_RECEIPT_SHA256 is None or not P2_RECEIPT_PATH.is_file():
        raise ValueError("registered production P2 scenario receipt is unavailable")
    content = P2_RECEIPT_PATH.read_bytes()
    if hashlib.sha256(content).hexdigest() != PRODUCTION_P2_RECEIPT_SHA256:
        raise ValueError("registered production P2 scenario receipt hash mismatch")
    return _parse_p2_registration_receipt(content, manifest)


def _load_p2_registration_receipt_fixture_bytes(
    content: bytes, manifest: SplitManifest
) -> P2RegistrationReceipt:
    """Test-only exact-byte loader for a synthetic custodian receipt."""
    return _parse_p2_registration_receipt(content, manifest)


@dataclass(frozen=True)
class P2AccessProtocol:
    schema: str
    registration_receipt_id: str
    registration_authority_url: str
    registration_receipt_sha256: str
    registration_id: str
    frozen_commit: str
    frozen_at_utc: str
    split_manifest_sha256: str
    digest: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify(self, manifest: SplitManifest) -> None:
        if self._authority is not _P2_PROTOCOL_AUTHORITY:
            raise ValueError("P2 protocol must be issued from an external registration receipt")
        manifest.verify()
        if self.schema != "scenario-p2-access-v1" or not self.registration_id:
            raise ValueError("invalid P2 access protocol")
        if not self.registration_receipt_id or not self.registration_authority_url.startswith(
            "https://"
        ):
            raise ValueError("P2 protocol must cite its external registration receipt")
        _sha256_hex("P2 registration receipt", self.registration_receipt_sha256)
        if len(self.frozen_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.frozen_commit
        ):
            raise ValueError("P2 frozen_commit must be a full SHA-1 object id")
        try:
            frozen_at = datetime.fromisoformat(self.frozen_at_utc)
        except ValueError as exc:
            raise ValueError("P2 frozen_at_utc must be an ISO-8601 timestamp") from exc
        if frozen_at.tzinfo is None or frozen_at.utcoffset() != timedelta(0):
            raise ValueError("P2 frozen_at_utc must use the UTC offset")
        if self.split_manifest_sha256 != manifest.digest:
            raise ValueError("P2 protocol does not bind this split manifest")
        payload = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("digest", "_authority")
        }
        expected = hashlib.sha256((_P2_TAG + _canonical(payload)).encode()).hexdigest()
        if self.digest != expected:
            raise ValueError("P2 access protocol digest mismatch")


def freeze_p2_access_protocol(
    *,
    registration_receipt: P2RegistrationReceipt,
    manifest: SplitManifest,
) -> P2AccessProtocol:
    registration_receipt.verify()
    if registration_receipt.split_manifest_sha256 != manifest.digest:
        raise ValueError("P2 receipt does not bind this split manifest")
    receipt = registration_receipt
    payload = {
        "schema": "scenario-p2-access-v1",
        "registration_receipt_id": receipt.receipt_id,
        "registration_authority_url": receipt.authority_url,
        "registration_receipt_sha256": receipt.receipt_sha256,
        "registration_id": receipt.registration_id,
        "frozen_commit": receipt.frozen_commit,
        "frozen_at_utc": receipt.registered_at_utc,
        "split_manifest_sha256": manifest.digest,
    }
    digest = hashlib.sha256((_P2_TAG + _canonical(payload)).encode()).hexdigest()
    protocol = P2AccessProtocol(
        **payload,
        digest=digest,
        _authority=_P2_PROTOCOL_AUTHORITY,  # type: ignore[arg-type]
    )
    protocol.verify(manifest)
    return protocol


def p2_protocol_to_json(protocol: P2AccessProtocol, manifest: SplitManifest) -> str:
    """Canonical durable P2 protocol representation."""
    protocol.verify(manifest)
    return _canonical(
        {
            field.name: getattr(protocol, field.name)
            for field in dataclasses.fields(protocol)
            if field.name != "_authority"
        }
    )


def p2_protocol_from_json(
    text: str,
    *,
    registration_receipt: P2RegistrationReceipt,
    manifest: SplitManifest,
) -> P2AccessProtocol:
    """Reload only the protocol exactly derivable from its external receipt."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("P2 protocol is not valid JSON") from exc
    expected_fields = {
        field.name
        for field in dataclasses.fields(P2AccessProtocol)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("P2 protocol JSON fields do not match the schema exactly")
    expected = freeze_p2_access_protocol(
        registration_receipt=registration_receipt,
        manifest=manifest,
    )
    expected_payload = {
        field.name: getattr(expected, field.name)
        for field in dataclasses.fields(expected)
        if field.name != "_authority"
    }
    if payload != expected_payload:
        raise ValueError("P2 protocol JSON is not derivable from the supplied receipt")
    if p2_protocol_to_json(expected, manifest) != text:
        raise ValueError("P2 protocol JSON is not canonical")
    return expected


@dataclass(frozen=True)
class P2AccessReceipt:
    schema: str
    protocol_sha256: str
    scenario_id: str
    artifact_sha256: str
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

    def verify(
        self,
        protocol: P2AccessProtocol,
        manifest: SplitManifest,
        artifact: ScenarioArtifact,
        marker: Path,
    ) -> None:
        if self._authority is not _P2_RECEIPT_AUTHORITY:
            raise ValueError("P2 receipt must be issued by P2ConfirmationGate")
        protocol.verify(manifest)
        require_frozen_clean_checkout(protocol.frozen_commit)
        artifact.verify()
        if (
            self.schema != "scenario-p2-access-receipt-v1"
            or artifact.visibility != PRIVATE
            or self.protocol_sha256 != protocol.digest
            or self.scenario_id != artifact.scenario_id
            or self.artifact_sha256 != artifact.digest
        ):
            raise ValueError("P2 receipt does not match protocol/private artifact")
        _sha256_hex("P2 marker SHA-256", self.marker_sha256)
        if (
            not marker.is_file()
            or hashlib.sha256(marker.read_bytes()).hexdigest() != self.marker_sha256
        ):
            raise ValueError("P2 access marker is missing or mismatched")
        marker_payload = json.loads(marker.read_bytes())
        if (
            marker_payload.get("schema") != "scenario-p2-access-marker-v1"
            or marker_payload.get("protocol_sha256") != self.protocol_sha256
            or marker_payload.get("scenario_id") != self.scenario_id
            or marker_payload.get("opened_at_utc") != self.opened_at_utc
        ):
            raise ValueError("P2 marker fields do not match the access receipt")
        expected = hashlib.sha256(
            ("scenario-p2-access-receipt-v1:" + _canonical(self._payload())).encode()
        ).hexdigest()
        if self.digest != expected:
            raise ValueError("P2 access receipt digest mismatch")


def p2_access_receipt_to_json(
    receipt: P2AccessReceipt,
    *,
    protocol: P2AccessProtocol,
    manifest: SplitManifest,
    artifact: ScenarioArtifact,
    marker: Path,
) -> str:
    """Canonical receipt export after rechecking the protected access context."""
    receipt.verify(protocol, manifest, artifact, marker)
    return _canonical({**receipt._payload(), "digest": receipt.digest})


def p2_access_receipt_from_json(
    text: str,
    *,
    protocol: P2AccessProtocol,
    manifest: SplitManifest,
    artifact: ScenarioArtifact,
    marker: Path,
) -> P2AccessReceipt:
    """Reload a receipt only when its private artifact and durable marker still verify."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("P2 access receipt is not valid JSON") from exc
    expected_fields = {
        field.name
        for field in dataclasses.fields(P2AccessReceipt)
        if field.name != "_authority"
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("P2 access receipt JSON fields do not match the schema exactly")
    receipt = P2AccessReceipt(
        **payload, _authority=_P2_RECEIPT_AUTHORITY  # type: ignore[arg-type]
    )
    receipt.verify(protocol, manifest, artifact, marker)
    if p2_access_receipt_to_json(
        receipt,
        protocol=protocol,
        manifest=manifest,
        artifact=artifact,
        marker=marker,
    ) != text:
        raise ValueError("P2 access receipt JSON is not canonical")
    return receipt


@dataclass(frozen=True)
class OpenedPrivateScenario:
    artifact: ScenarioArtifact
    receipt: P2AccessReceipt
    _authority: object = dataclasses.field(repr=False, compare=False)

    def verify(
        self, protocol: P2AccessProtocol, manifest: SplitManifest, marker: Path
    ) -> ScenarioArtifact:
        if self._authority is not _P2_RECEIPT_AUTHORITY:
            raise ValueError("private scenario must be opened by P2ConfirmationGate")
        self.receipt.verify(protocol, manifest, self.artifact, marker)
        return self.artifact


def opened_private_scenario_from_json(
    artifact_text: str,
    receipt_text: str,
    *,
    protocol: P2AccessProtocol,
    manifest: SplitManifest,
    marker: Path,
) -> OpenedPrivateScenario:
    """Contextually reload an already-opened private scenario in a fresh process."""
    protocol.verify(manifest)
    require_frozen_clean_checkout(protocol.frozen_commit)
    artifact = _scenario_artifact_from_json(artifact_text, allow_private=True)
    matches = [
        item
        for item in manifest.private_commitments
        if item.scenario_id == artifact.scenario_id
    ]
    if (
        len(matches) != 1
        or private_scenario_commitment(artifact)._payload() != matches[0]._payload()
    ):
        raise ValueError("reloaded private artifact does not match its split commitment")
    receipt = p2_access_receipt_from_json(
        receipt_text,
        protocol=protocol,
        manifest=manifest,
        artifact=artifact,
        marker=marker,
    )
    opened = OpenedPrivateScenario(
        artifact,
        receipt,
        _authority=_P2_RECEIPT_AUTHORITY,
    )
    opened.verify(protocol, manifest, marker)
    return opened


class P2ConfirmationGate:
    """Open one private artifact only after a durable P2-bound marker is created."""

    def __init__(
        self,
        protocol: P2AccessProtocol,
        manifest: SplitManifest,
        marker: Path,
    ) -> None:
        protocol.verify(manifest)
        require_frozen_clean_checkout(protocol.frozen_commit)
        self.protocol = protocol
        self.manifest = manifest
        self.marker = marker

    def open_private(
        self, scenario_id: str, loader: Callable[[], str]
    ) -> OpenedPrivateScenario:
        matches = [
            item for item in self.manifest.private_commitments if item.scenario_id == scenario_id
        ]
        if len(matches) != 1:
            raise ValueError("private scenario is not uniquely committed in the split manifest")
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        opened_at_utc = datetime.fromtimestamp(
            os.fstat(fd).st_ctime_ns / 1_000_000_000, timezone.utc
        ).isoformat()
        if datetime.fromisoformat(opened_at_utc) < datetime.fromisoformat(
            self.protocol.frozen_at_utc
        ):
            os.close(fd)
            raise ValueError("P2 gate creation predates the frozen P2 protocol")
        marker_payload = {
            "schema": "scenario-p2-access-marker-v1",
            "protocol_sha256": self.protocol.digest,
            "scenario_id": scenario_id,
            "opened_at_utc": opened_at_utc,
        }
        marker_bytes = (_canonical(marker_payload) + "\n").encode()
        try:
            os.write(fd, marker_bytes)
        finally:
            os.close(fd)
        artifact = _scenario_artifact_from_json(loader(), allow_private=True)
        if private_scenario_commitment(artifact)._payload() != matches[0]._payload():
            raise ValueError("opened private artifact does not match its frozen commitment")
        receipt_payload = {
            "schema": "scenario-p2-access-receipt-v1",
            "protocol_sha256": self.protocol.digest,
            "scenario_id": scenario_id,
            "artifact_sha256": artifact.digest,
            "opened_at_utc": opened_at_utc,
            "marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        }
        digest = hashlib.sha256(
            ("scenario-p2-access-receipt-v1:" + _canonical(receipt_payload)).encode()
        ).hexdigest()
        receipt = P2AccessReceipt(
            **receipt_payload,
            digest=digest,
            _authority=_P2_RECEIPT_AUTHORITY,  # type: ignore[arg-type]
        )
        receipt.verify(self.protocol, self.manifest, artifact, self.marker)
        return OpenedPrivateScenario(artifact, receipt, _authority=_P2_RECEIPT_AUTHORITY)


def run_private_scenario_artifact(
    opened: OpenedPrivateScenario,
    protocol: P2AccessProtocol,
    manifest: SplitManifest,
    marker: Path,
    scorer: Any,
    proposer: Any | None = None,
    lb_policy: str = "round_robin",
) -> Any:
    """Confirmation-only runner; public/pilot artifacts use the normal runner."""
    from sim.engine import run_episode

    artifact = opened.verify(protocol, manifest, marker)
    return run_episode(
        artifact.verify(),
        seed=artifact.episode_seed,
        scorer=scorer,
        proposer=proposer,
        lb_policy=lb_policy,
    )
