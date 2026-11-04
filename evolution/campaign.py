"""Resumable, append-only Phase-5 campaign runner (TODO 5.8)."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import placementscript as ps
from shinka.core import ProposalOffspring
from shinka.core.proposal_operator import (
    ProposalOperator,
    invoke_proposal_operator,
)
from shinka.database import Program

from analysis.tracking import derive_seed
from sim.energy import CandidateFitness

from .evaluation import (
    OMITTED_EPISODE_LEDGERS,
    ScenarioGrid,
    TaskEvaluationConfig,
    candidate_failure_metadata,
    compact_episode_ledgers,
    policy_identity_for_source,
)
from .feedback import FeedbackProfile, generate_feedback
from .islands import (
    ArchiveAssessment,
    IslandsConfig,
    assess_for_archive,
    measure_decision_cost,
)
from .ladder import (
    BehavioralNoveltyConfig,
    LadderConfig,
    LadderResult,
    evaluate_through_ladder,
)
from .llm_operator import ValidationRepair
from .operators import (
    AblationProbeInterleaveOperator,
    SharedNumericInterleaveOperator,
    validate_scoped_edit,
)
from .profiles import MATCHED_PROFILE, EvolutionProfile, evolution_profile

CAMPAIGN_CONFIG_SCHEMA = "evolved-continuum-campaigns-v3"
ARCHIVE_SCHEMA = "evolved-continuum-campaign-archive-v4"
AUTHORIZATION_SCHEMA = "evolved-continuum-confirmatory-authorization-v2"
CONFIRMATORY_PREREQUISITES = {
    "G0": "PASS",
    "G-K8s": "PASS",
    "G-ablate": "PASS",
    "DES-verification": "PASS",
    "G2": "PASS",
    "G3-cal": "PASS",
    "G-ladder": "PASS",
    "P0-eng": "FROZEN",
    "P1": "FROZEN",
    "P2": "FROZEN",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class CampaignDefinition:
    campaign_id: str
    root_seed: int
    evaluation_replicates: int

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must be non-empty")
        if (
            not isinstance(self.root_seed, int)
            or isinstance(self.root_seed, bool)
            or not 0 <= self.root_seed < 2**63
        ):
            raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
        if (
            not isinstance(self.evaluation_replicates, int)
            or isinstance(self.evaluation_replicates, bool)
            or self.evaluation_replicates < 1
        ):
            raise ValueError("evaluation_replicates must be a positive integer")


@dataclass(frozen=True)
class EvaluatorWorkBudget:
    registration_status: str
    registration_source: str
    max_microwork: int
    rung_microwork: dict[str, int]
    checkpoints_microwork: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.registration_status not in {"development-provisional", "p2-frozen"}:
            raise ValueError("unknown evaluator-work registration status")
        if not self.registration_source:
            raise ValueError("evaluator-work registration source is required")
        if (
            not isinstance(self.max_microwork, int)
            or isinstance(self.max_microwork, bool)
            or self.max_microwork < 1
        ):
            raise ValueError("max_microwork must be a positive integer")
        if set(self.rung_microwork) != {"L0", "L1", "L2"} or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.rung_microwork.values()
        ):
            raise ValueError("rung_microwork needs positive integer L0/L1/L2 weights")
        if (
            len(self.checkpoints_microwork) < 2
            or self.checkpoints_microwork[0] != 0
            or self.checkpoints_microwork[-1] != self.max_microwork
            or tuple(sorted(set(self.checkpoints_microwork))) != self.checkpoints_microwork
        ):
            raise ValueError("work checkpoints must be unique, sorted, and span 0 to budget")
        if self.max_candidate_microwork > self.max_microwork:
            raise ValueError("work budget cannot fit one full-ladder candidate")

    @property
    def max_candidate_microwork(self) -> int:
        return sum(self.rung_microwork.values())

    @property
    def confirmatory_ready(self) -> bool:
        return self.registration_status == "p2-frozen"

    def work_for_trace(self, trace: tuple[dict[str, Any], ...]) -> int:
        work = 0
        for record in trace:
            rung = record["rung"]
            if rung in self.rung_microwork:
                work += self.rung_microwork[rung]
            elif rung == "epsilon-audit" and record.get("selected"):
                work += self.rung_microwork[record["target_rung"]]
        return work


@dataclass(frozen=True)
class CampaignPlan:
    status: str
    frozen_tag: str | None
    confirmatory_authorization_sha256: str | None
    candidate_budget: int
    population_per_island: int
    max_operator_attempts: int
    shared_numeric_probability: float
    llm_proposal_modes: tuple[str, ...]
    evaluator_work: EvaluatorWorkBudget
    feedback_profile: str
    evolution_profile: str
    arms: tuple[str, ...]
    campaigns: tuple[CampaignDefinition, ...]
    proposals_per_call: int = 1
    ablation_probe_probability: float = 0.0
    ablation_prompt_hint_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in {"development-provisional", "p2-frozen"}:
            raise ValueError("unknown campaign-plan status")
        if self.status == "development-provisional" and self.frozen_tag is not None:
            raise ValueError("a development campaign plan cannot claim a frozen tag")
        if (
            self.status == "development-provisional"
            and self.confirmatory_authorization_sha256 is not None
        ):
            raise ValueError("a development plan cannot claim confirmatory authorization")
        if self.status == "p2-frozen" and (
            not isinstance(self.frozen_tag, str)
            or not self.frozen_tag
            or self.frozen_tag.startswith("-")
            or any(char.isspace() for char in self.frozen_tag)
        ):
            raise ValueError("a P2-frozen campaign plan requires its predeclared Git tag")
        if self.status == "p2-frozen" and (
            not isinstance(self.confirmatory_authorization_sha256, str)
            or len(self.confirmatory_authorization_sha256) != 64
            or any(
                char not in "0123456789abcdef" for char in self.confirmatory_authorization_sha256
            )
        ):
            raise ValueError("a P2-frozen campaign plan requires an authorization hash")
        if self.status == "p2-frozen" and not self.evaluator_work.confirmatory_ready:
            raise ValueError("P2 campaigns require P2-frozen evaluator-work weights")
        if self.status == "development-provisional" and self.evaluator_work.confirmatory_ready:
            raise ValueError("a development campaign cannot claim P2 evaluator-work weights")
        for field in ("candidate_budget", "population_per_island", "max_operator_attempts"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive int")
        if (
            not isinstance(self.proposals_per_call, int)
            or isinstance(self.proposals_per_call, bool)
            or self.proposals_per_call < 1
        ):
            raise ValueError("proposals_per_call must be a positive int")
        if (
            not isinstance(self.shared_numeric_probability, (int, float))
            or isinstance(self.shared_numeric_probability, bool)
            or not 0.0 <= self.shared_numeric_probability <= 1.0
        ):
            raise ValueError("shared_numeric_probability must be in [0, 1]")
        for field in (
            "ablation_probe_probability",
            "ablation_prompt_hint_probability",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field} must be in [0, 1]")
        if self.shared_numeric_probability + self.ablation_probe_probability > 1.0:
            raise ValueError("shared numeric and ablation route shares exceed 1")
        if self.status != "development-provisional" and (
            self.ablation_probe_probability > 0.0
            or self.ablation_prompt_hint_probability > 0.0
        ):
            raise ValueError("ablation probes are development-only")
        if self.llm_proposal_modes != ("diff", "rewrite", "crossover"):
            raise ValueError("campaign must register the diff/rewrite/crossover LLM modes")
        profile = evolution_profile(self.evolution_profile)
        if profile.development_only and self.status != "development-provisional":
            raise ValueError("enhanced-llm-dev is development-only and never confirmatory")
        if profile.name != MATCHED_PROFILE and (
            self.feedback_profile != profile.feedback_profile
            or self.shared_numeric_probability != profile.shared_numeric_probability
        ):
            raise ValueError("campaign controls differ from the selected evolution profile")
        if self.proposals_per_call > 1 and (
            profile.numeric_interleave_enabled
            or self.ablation_probe_probability > 0.0
        ):
            raise ValueError(
                "batched LLM proposals require a profile without proposal interleaves"
            )
        if self.arms != ("llm", "gp"):
            raise ValueError("the G-ablate plan has exactly the matched llm/gp arms")
        ids = [campaign.campaign_id for campaign in self.campaigns]
        if len(ids) < 10 or len(ids) != len(set(ids)):
            raise ValueError("the plan requires at least 10 independent campaign ids")
        roots = [campaign.root_seed for campaign in self.campaigns]
        if len(roots) != len(set(roots)):
            raise ValueError("campaign root seeds must be independent and unique")

    @property
    def confirmatory_ready(self) -> bool:
        return self.status == "p2-frozen" and self.evaluator_work.confirmatory_ready

    @property
    def profile(self) -> EvolutionProfile:
        return evolution_profile(self.evolution_profile)


def load_campaign_plan(path: str | Path) -> CampaignPlan:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != CAMPAIGN_CONFIG_SCHEMA:
        raise ValueError(f"campaign schema must be {CAMPAIGN_CONFIG_SCHEMA}")
    return CampaignPlan(
        status=raw["status"],
        frozen_tag=raw.get("frozen_tag"),
        confirmatory_authorization_sha256=raw.get("confirmatory_authorization_sha256"),
        candidate_budget=raw["candidate_budget"],
        population_per_island=raw["population_per_island"],
        max_operator_attempts=raw["max_operator_attempts"],
        shared_numeric_probability=raw["shared_numeric_probability"],
        llm_proposal_modes=tuple(raw["llm_proposal_modes"]),
        evaluator_work=EvaluatorWorkBudget(
            registration_status=raw["evaluator_work"]["registration_status"],
            registration_source=raw["evaluator_work"]["registration_source"],
            max_microwork=raw["evaluator_work"]["max_microwork"],
            rung_microwork=raw["evaluator_work"]["rung_microwork"],
            checkpoints_microwork=tuple(raw["evaluator_work"]["checkpoints_microwork"]),
        ),
        feedback_profile=raw["feedback_profile"],
        evolution_profile=raw.get("evolution_profile", MATCHED_PROFILE),
        arms=tuple(raw["arms"]),
        campaigns=tuple(
            CampaignDefinition(
                campaign_id=item["campaign_id"],
                root_seed=item["root_seed"],
                evaluation_replicates=item["evaluation_replicates"],
            )
            for item in raw["campaigns"]
        ),
        proposals_per_call=raw.get("proposals_per_call", 1),
        ablation_probe_probability=raw.get("ablation_probe_probability", 0.0),
        ablation_prompt_hint_probability=raw.get(
            "ablation_prompt_hint_probability", 0.0
        ),
    )


def verify_confirmatory_authorization(
    plan: CampaignPlan,
    path: str | Path,
    *,
    engine_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the commit-pinned go/no-go envelope before confirmatory execution."""
    from .codex_engine import refuse_codex_confirmatory_engine

    refuse_codex_confirmatory_engine(
        engine_descriptor, boundary="confirmatory authorization"
    )
    if not plan.confirmatory_ready or plan.confirmatory_authorization_sha256 is None:
        raise ValueError("only a P2-frozen campaign plan can authorize confirmation")
    encoded = Path(path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != plan.confirmatory_authorization_sha256:
        raise ValueError("confirmatory authorization differs from the P2 hash pin")
    raw = json.loads(encoded)
    if raw.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError(f"authorization schema must be {AUTHORIZATION_SCHEMA}")
    if raw.get("prerequisites") != CONFIRMATORY_PREREQUISITES:
        raise ValueError("confirmatory prerequisites are incomplete or not all passed/frozen")
    if raw.get("frozen_tag") != plan.frozen_tag:
        raise ValueError("authorization tag differs from the campaign plan")
    origin = raw.get("proposal_endpoint_origin")
    handshake_sha256 = raw.get("proposal_host_handshake_sha256")
    if not isinstance(origin, str) or not origin:
        raise ValueError("authorization must pin the proposal endpoint origin")
    if (
        not isinstance(handshake_sha256, str)
        or len(handshake_sha256) != 64
        or any(char not in "0123456789abcdef" for char in handshake_sha256)
    ):
        raise ValueError("authorization must pin the proposal host handshake")
    return raw


class EventArchive:
    """Canonical hash-chained JSONL; candidate creation is durable before proposal."""

    def __init__(self, root: str | Path, manifest: dict[str, Any]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidate_dir = self.root / "candidates"
        self.candidate_dir.mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.events_path = self.root / "events.jsonl"
        manifest_text = _canonical(manifest) + "\n"
        if self.manifest_path.exists():
            if self.manifest_path.read_text() != manifest_text:
                raise ValueError("archive manifest differs on resume")
        else:
            with self.manifest_path.open("x") as stream:
                stream.write(manifest_text)
                stream.flush()
                os.fsync(stream.fileno())
        self.events = self._read_events()
        self.validate_semantics()

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        encoded = self.events_path.read_bytes()
        if encoded and not encoded.endswith(b"\n"):
            # A killed append can leave a partial final record. It was never a durable
            # event, so discard only that unterminated tail and replay the candidate
            # from its preceding creation event.
            boundary = encoded.rfind(b"\n") + 1
            with self.events_path.open("r+b") as stream:
                stream.truncate(boundary)
                stream.flush()
                os.fsync(stream.fileno())
            encoded = encoded[:boundary]
        events = []
        previous = "0" * 64
        for index, raw_line in enumerate(encoded.splitlines()):
            line = raw_line.decode()
            event = json.loads(line)
            if line != _canonical(event):
                raise ValueError(f"event {index} is not canonically encoded")
            digest = event.pop("event_sha256", None)
            if (
                event.get("event_seq") != index
                or event.get("previous_event_sha256") != previous
                or digest != _sha(event)
            ):
                raise ValueError(f"event-chain verification failed at event {index}")
            event["event_sha256"] = digest
            previous = digest
            events.append(event)
        return events

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "event_seq": len(self.events),
            "previous_event_sha256": (self.events[-1]["event_sha256"] if self.events else "0" * 64),
            "type": event_type,
            "payload": payload,
        }
        event = {**body, "event_sha256": _sha(body)}
        self._validate_events([*self.events, event])
        encoded = _canonical(event) + "\n"
        with self.events_path.open("a") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    def validate_semantics(self) -> None:
        self._validate_events(self.events)

    @staticmethod
    def _validate_events(events: list[dict[str, Any]]) -> None:
        created: dict[int, dict[str, Any]] = {}
        completed: set[int] = set()
        terminal_seen = False
        for index, event in enumerate(events):
            if terminal_seen:
                raise ValueError("campaign terminal event must be the final event")
            if event["type"] == "candidate_created":
                payload = event["payload"]
                seq = payload["proposal_seq"]
                if seq in created or seq != len(created):
                    raise ValueError("candidate creation sequence must be contiguous and unique")
                batch_fields = (
                    payload.get("shared_call_id"),
                    payload.get("batch_index"),
                    payload.get("batch_size"),
                )
                if any(value is not None for value in batch_fields):
                    call_id, batch_index, batch_size = batch_fields
                    if (
                        not isinstance(call_id, str)
                        or not call_id
                        or not isinstance(batch_index, int)
                        or isinstance(batch_index, bool)
                        or not isinstance(batch_size, int)
                        or isinstance(batch_size, bool)
                        or batch_size < 2
                        or not 1 <= batch_index <= batch_size
                    ):
                        raise ValueError("candidate batch creation fields are invalid")
                    if batch_index > 1:
                        previous = created.get(seq - 1)
                        if (
                            previous is None
                            or previous.get("shared_call_id") != call_id
                            or previous.get("batch_index") != batch_index - 1
                            or previous.get("batch_size") != batch_size
                        ):
                            raise ValueError("candidate batch creation is not contiguous")
                    elif any(
                        prior.get("shared_call_id") == call_id for prior in created.values()
                    ):
                        raise ValueError("candidate batch call id must be unique")
                created[seq] = payload
            elif event["type"] == "candidate_completed":
                payload = event["payload"]
                seq = payload["proposal_seq"]
                if seq not in created or seq in completed:
                    raise ValueError("candidate completion lacks one unique creation event")
                for field in (
                    "candidate_id",
                    "island_index",
                    "island_id",
                    "parent_id",
                    "inspiration_ids",
                    "selection_seed",
                    "operator",
                ):
                    if payload[field] != created[seq][field]:
                        raise ValueError(f"candidate completion changed creation field {field}")
                if payload.get("proposal_engine") != created[seq].get("proposal_engine"):
                    raise ValueError("candidate completion changed proposal engine identity")
                for field in ("shared_call_id", "batch_index", "batch_size"):
                    if payload.get(field) != created[seq].get(field):
                        raise ValueError(
                            f"candidate completion changed batch field {field}"
                        )
                completed.add(seq)
            elif event["type"] == "campaign_terminal":
                if set(created) != completed:
                    raise ValueError("campaign cannot terminate with a pending candidate")
                batch_call_ids = {
                    payload.get("shared_call_id")
                    for payload in created.values()
                    if payload.get("shared_call_id") is not None
                }
                for call_id in batch_call_ids:
                    members = [
                        payload
                        for payload in created.values()
                        if payload.get("shared_call_id") == call_id
                    ]
                    declared_size = members[0]["batch_size"]
                    if (
                        len(members) != declared_size
                        or sorted(payload["batch_index"] for payload in members)
                        != list(range(1, declared_size + 1))
                    ):
                        raise ValueError("campaign terminated with an incomplete batch")
                if index != len(events) - 1:
                    raise ValueError("campaign terminal event must be the final event")
                payload = event["payload"]
                if payload.get("status") not in {"complete", "unusable"}:
                    raise ValueError("campaign terminal status is invalid")
                if payload.get("usable_for_confirmatory_analysis") is not False:
                    raise ValueError("campaign terminal cannot self-authorize confirmatory use")
                expected_receipt = (
                    payload.get("run_purpose") == "scientific-full-work"
                    and payload.get("status") == "complete"
                    and payload.get("reason") == "normalized-work-budget-exhausted"
                )
                if payload.get("confirmatory_postflight_receipt_required") is not expected_receipt:
                    raise ValueError("campaign terminal postflight-receipt requirement is invalid")
                budget_policy = payload.get("terminal_budget_policy")
                if budget_policy not in {"carry-to-work-limit", "discard-at-trigger"}:
                    raise ValueError("campaign terminal budget policy is invalid")
                remaining = payload.get("remaining_microwork")
                if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
                    raise ValueError("campaign terminal remaining work is invalid")
                exhausted = payload.get("reason") == "normalized-work-budget-exhausted"
                expected_carry = (
                    remaining if exhausted and budget_policy == "carry-to-work-limit" else 0
                )
                expected_discard = (
                    remaining if exhausted and budget_policy == "discard-at-trigger" else 0
                )
                if (
                    payload.get("budget_carry_microwork") != expected_carry
                    or payload.get("unused_microwork_discarded") != expected_discard
                ):
                    raise ValueError("campaign terminal budget disposition is inconsistent")
                terminal_seen = True
            else:
                raise ValueError(f"unknown archive event type {event['type']!r}")

    def materialize_candidates(self) -> None:
        expected_files: set[str] = set()
        for event in self.events:
            if event["type"] != "candidate_completed":
                continue
            payload = event["payload"]
            source = payload["proposal"].get("source")
            if source is None:
                continue
            path = self.candidate_dir / f"{payload['proposal_seq']:08d}.ps"
            expected_files.add(path.name)
            expected = source
            if path.exists() and path.read_text() != expected:
                raise ValueError(f"candidate file differs from event: {path.name}")
            if not path.exists():
                path.write_text(expected)
        actual_files = {path.name for path in self.candidate_dir.glob("*.ps")}
        if actual_files != expected_files:
            raise ValueError("candidate directory contains files outside the event archive")

    def digest(self) -> str:
        digest = hashlib.sha256()
        paths = [self.manifest_path]
        if self.events_path.exists():
            paths.append(self.events_path)
        proposal_usage = self.root / "proposal-engine-usage.jsonl"
        if proposal_usage.exists():
            paths.append(proposal_usage)
        paths.extend(sorted(self.candidate_dir.glob("*.ps")))
        for path in paths:
            content = path.read_bytes()
            digest.update(path.name.encode())
            digest.update(b"\x00")
            digest.update(str(len(content)).encode())
            digest.update(b"\x00")
            digest.update(content)
        return digest.hexdigest()


EvaluateFn = Callable[[str, TaskEvaluationConfig, LadderConfig, int], LadderResult]
AssessFn = Callable[[str, TaskEvaluationConfig, IslandsConfig], ArchiveAssessment]


def _default_evaluate(
    source: str,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    proposal_seq: int,
) -> LadderResult:
    return evaluate_through_ladder(source, task, ladder, proposal_seq=proposal_seq)


def _default_assess(
    source: str,
    task: TaskEvaluationConfig,
    islands: IslandsConfig,
) -> ArchiveAssessment:
    cost = measure_decision_cost(source, task, islands.latency_calibration)
    return assess_for_archive(source, cost, islands.archive)


def _invalidate_evaluated_proposal(
    proposal: dict[str, Any], failure: dict[str, Any]
) -> None:
    proposal["valid"] = False
    proposal["error"] = f"{failure['failure_category']}: {failure['message']}"
    proposal["metadata"] = {
        **proposal.get("metadata", {}),
        "failure_category": failure["failure_category"],
        "evaluation_invalid_attempts": 1,
        "candidate_failure": failure,
    }


def _fitness_from_dict(raw: dict[str, Any]) -> CandidateFitness:
    return CandidateFitness(
        feasible=raw["feasible"],
        infeasibility_vector=tuple(raw["infeasibility_vector"]),
        summed_primitive_shortfall=raw["summed_primitive_shortfall"],
        normalized_energy=raw["normalized_energy"],
        proposal_seq=raw["proposal_seq"],
        n_instances=raw["n_instances"],
        n_bundles=raw["n_bundles"],
    )


class CampaignRunner:
    def __init__(
        self,
        *,
        definition: CampaignDefinition,
        plan: CampaignPlan,
        arm: str,
        operator: ProposalOperator,
        initial_source: str,
        task: TaskEvaluationConfig,
        ladder: LadderConfig,
        islands: IslandsConfig,
        feedback_profile: FeedbackProfile,
        archive_dir: str | Path,
        provenance: dict[str, Any],
        run_purpose: Literal["scientific-full-work", "determinism-smoke"] = (
            "scientific-full-work"
        ),
        evaluation_seed_mode: Literal["campaign-derived", "presealed-task"] = (
            "campaign-derived"
        ),
        terminal_budget_policy: Literal["carry-to-work-limit", "discard-at-trigger"] = (
            "carry-to-work-limit"
        ),
        evaluate_fn: EvaluateFn = _default_evaluate,
        assess_fn: AssessFn = _default_assess,
        retain_episode_ledgers: bool = False,
    ) -> None:
        if arm not in plan.arms:
            raise ValueError(f"arm must be one of {plan.arms}")
        if definition not in plan.campaigns:
            raise ValueError("campaign definition is not a member of the campaign plan")
        if feedback_profile.name != plan.feedback_profile:
            raise ValueError("feedback profile differs from campaign plan")
        if run_purpose not in {"scientific-full-work", "determinism-smoke"}:
            raise ValueError("unknown campaign run purpose")
        if evaluation_seed_mode not in {"campaign-derived", "presealed-task"}:
            raise ValueError("unknown campaign evaluation-seed mode")
        if terminal_budget_policy not in {"carry-to-work-limit", "discard-at-trigger"}:
            raise ValueError("unknown campaign terminal-budget policy")
        ps.Policy(initial_source)
        self.definition = definition
        self.plan = plan
        self.arm = arm
        self.operator = operator
        if arm == "llm" and plan.proposals_per_call > 1:
            if not callable(getattr(operator, "propose_batch", None)):
                raise ValueError(
                    "LLM proposals_per_call > 1 requires a batch-capable operator"
                )
            configured_batch_size = getattr(operator, "proposals_per_call", None)
            if configured_batch_size != plan.proposals_per_call:
                raise ValueError("operator batch size differs from the campaign plan")
        self.engine_descriptor = getattr(operator, "engine_descriptor", None)
        if (
            isinstance(self.engine_descriptor, dict)
            and self.engine_descriptor.get("development_only") is True
            and plan.status != "development-provisional"
        ):
            raise ValueError("development-only proposal engine cannot run a frozen campaign")
        self.run_purpose = run_purpose
        # A campaign has no consumer for per-episode `records` / `service_calls` /
        # `snapshots`: it consumes fitness, rung decisions and the feedback narrative
        # (which reads `episode.energy`), and the archive writer compacts those ledgers
        # anyway. Retaining them peaked a study-B evaluation at 2.0 GiB and killed a
        # pilot against a 2 GiB guard on 2026-09-17; compacting each instance as it is
        # produced measured 461.7 MiB and 1.58x faster on that same request, with
        # identical fitness, digests, seals and decisions. Pass True only for a caller
        # that reads raw ledgers out of a live result.
        self.retain_episode_ledgers = retain_episode_ledgers
        self.evaluation_seed_mode = evaluation_seed_mode
        self.terminal_budget_policy = terminal_budget_policy
        self.initial_source = initial_source
        self.task = task
        if evaluation_seed_mode == "campaign-derived":
            self.task = replace(
                task,
                scenarios=tuple(
                    ScenarioGrid(
                        scenario_id=scenario.scenario_id,
                        path=scenario.path,
                        seeds=tuple(
                            derive_seed(
                                definition.root_seed,
                                "sim",
                                "scenario",
                                scenario.scenario_id,
                                "replicate",
                                replicate,
                            )
                            for replicate in range(definition.evaluation_replicates)
                        ),
                    )
                    for scenario in task.scenarios
                ),
            )
        self.ladder = ladder
        if evaluation_seed_mode == "campaign-derived":
            self.ladder = replace(
                ladder,
                audit_root_seed=derive_seed(
                    definition.root_seed,
                    "evolution",
                    "ladder-audit",
                ),
            )
        self.islands = islands
        self.feedback_profile = feedback_profile
        self.evaluate_fn = evaluate_fn
        self.assess_fn = assess_fn
        self.behavioral_novelty = (
            BehavioralNoveltyConfig(
                tolerance=plan.profile.behavioral_novelty_tolerance,
                signature_seed=derive_seed(
                    definition.root_seed,
                    "evolution",
                    "behavioral-novelty-signature",
                ),
            )
            if plan.profile.behavioral_novelty_enabled
            else None
        )
        if (
            self.behavioral_novelty is not None
            and evaluate_fn is not _default_evaluate
            and getattr(evaluate_fn, "supports_behavioral_novelty", False) is not True
        ):
            raise ValueError(
                "behavioral novelty requires the authoritative ladder evaluator"
            )
        manifest = {
            "schema": ARCHIVE_SCHEMA,
            "campaign": {
                "campaign_id": definition.campaign_id,
                "root_seed": definition.root_seed,
                "evaluation_seeds": {
                    scenario.scenario_id: list(scenario.seeds) for scenario in self.task.scenarios
                },
                "ladder_audit_seed": self.ladder.audit_root_seed,
                "arm": arm,
                "operator": operator.name,
                "candidate_budget": plan.candidate_budget,
                "population_per_island": plan.population_per_island,
                "max_operator_attempts": plan.max_operator_attempts,
                "shared_numeric_probability": plan.shared_numeric_probability,
                "llm_proposal_modes": list(plan.llm_proposal_modes),
                "feedback_profile": feedback_profile.name,
                "evolution_profile": plan.evolution_profile,
                "evolution_controls": {
                    "numeric_interleave_enabled": plan.profile.numeric_interleave_enabled,
                    "grammar_aware_retry_enabled": plan.profile.grammar_aware_retry_enabled,
                    "prompt_exemplars_enabled": plan.profile.prompt_exemplars_enabled,
                    "behavioral_novelty_enabled": plan.profile.behavioral_novelty_enabled,
                    "behavioral_novelty_tolerance": (
                        plan.profile.behavioral_novelty_tolerance
                    ),
                    "structural_rewrite_bias_enabled": (
                        plan.profile.structural_rewrite_bias_enabled
                    ),
                    "diverse_inspirations_enabled": (
                        plan.profile.diverse_inspirations_enabled
                    ),
                    "simulator_feedback_enabled": plan.profile.simulator_feedback_enabled,
                    "seeded_initial_archive_enabled": (
                        plan.profile.seeded_initial_archive_enabled
                    ),
                },
                "behavioral_novelty_signature_seed": (
                    None
                    if self.behavioral_novelty is None
                    else self.behavioral_novelty.signature_seed
                ),
                "plan_status": plan.status,
                "run_purpose": run_purpose,
                "evaluation_seed_mode": evaluation_seed_mode,
                "terminal_budget_policy": terminal_budget_policy,
                "evaluator_work": {
                    "registration_status": plan.evaluator_work.registration_status,
                    "registration_source": plan.evaluator_work.registration_source,
                    "max_microwork": plan.evaluator_work.max_microwork,
                    "rung_microwork": plan.evaluator_work.rung_microwork,
                    "checkpoints_microwork": list(plan.evaluator_work.checkpoints_microwork),
                    "candidate_safety_ceiling": plan.candidate_budget,
                },
            },
            "initial_source_sha256": hashlib.sha256(initial_source.encode()).hexdigest(),
            "island_ids": [profile.island_id for profile in islands.profiles],
            "provenance": provenance,
        }
        if (
            plan.ablation_probe_probability > 0.0
            or plan.ablation_prompt_hint_probability > 0.0
        ):
            manifest["campaign"]["ablation_controls"] = {
                "ablation_probe_probability": plan.ablation_probe_probability,
                "ablation_prompt_hint_probability": (
                    plan.ablation_prompt_hint_probability
                ),
            }
        if plan.proposals_per_call > 1:
            manifest["campaign"]["proposals_per_call"] = plan.proposals_per_call
        if self.engine_descriptor is not None:
            manifest["campaign"]["proposal_engine"] = self.engine_descriptor
        self.archive = EventArchive(archive_dir, manifest)
        self._restore_cost_ledger()
        self._restore_usage_ledger()
        self._verify_work_ledger()

    def _restore_cost_ledger(self) -> None:
        ledger = getattr(self.operator, "cost_ledger", None)
        if ledger is None:
            return
        spent = 0
        for event in self.archive.events:
            if event["type"] != "candidate_completed":
                continue
            for attempt in event["payload"]["proposal"].get("attempts", []):
                spent += attempt.get("metadata", {}).get("api_costs_microusd", 0)
        if ledger.spent_microusd not in {0, spent} or ledger.reserved_microusd != 0:
            raise ValueError("cost ledger state differs from resumable archive")
        ledger.spent_microusd = spent
        ledger._check()

    def _restore_usage_ledger(self) -> None:
        ledger = getattr(self.operator, "usage_ledger", None)
        if ledger is None:
            return
        bind_journal = getattr(ledger, "bind_journal", None)
        if callable(bind_journal):
            bind_journal(self.archive.root / "proposal-engine-usage.jsonl")
        deltas = []
        for event in self.archive.events:
            if event["type"] != "candidate_completed":
                continue
            for attempt in event["payload"]["proposal"].get("attempts", []):
                raw_response = attempt.get("metadata", {}).get("raw_response")
                if not isinstance(raw_response, dict):
                    continue
                for cli_attempt in raw_response.get("attempts", []):
                    if isinstance(cli_attempt, dict) and isinstance(
                        cli_attempt.get("accounting_delta"), dict
                    ):
                        deltas.append(cli_attempt["accounting_delta"])
        ledger.restore(deltas)

    def _spent_microwork(self) -> int:
        return sum(
            event["payload"]["evaluator_work"]["candidate_microwork"]
            for event in self.archive.events
            if event["type"] == "candidate_completed"
        )

    # Per-episode ledgers that scale with the *horizon*, not with anything the archive is
    # for. On a 420 s episode `service_calls` alone is a 74,754-element list: one archived
    # candidate carried 22 MB and a 19-candidate arm reached 489 MB, which is the same
    # class of failure as the 2026-08 four-gigabyte arm results (that fix stopped arm
    # results inlining episode data; the campaign archive still did). Each instance keeps
    # its `episode_digest` and `certificate_seal`, so the full episode remains pinned --
    # what is dropped is a copy of it, not evidence of it.
    _OMITTED_EPISODE_LEDGERS = OMITTED_EPISODE_LEDGERS

    @classmethod
    def _compact_episode_ledgers(cls, value: Any) -> Any:
        """Recursively replace bulk per-episode ledgers with their lengths."""
        return compact_episode_ledgers(value)

    def _work_payload(self, result: LadderResult | None) -> dict[str, Any]:
        candidate = 0 if result is None else self.plan.evaluator_work.work_for_trace(result.trace)
        cumulative = self._spent_microwork() + candidate
        return {
            "candidate_microwork": candidate,
            "cumulative_microwork": cumulative,
            "candidate_normalized_work": candidate / 1_000_000,
            "cumulative_normalized_work": cumulative / 1_000_000,
        }

    def _verify_work_ledger(self) -> None:
        cumulative = 0
        for event in self.archive.events:
            if event["type"] != "candidate_completed":
                continue
            payload = event["payload"]
            result = payload["evaluation"]
            expected = (
                0
                if result is None
                else self.plan.evaluator_work.work_for_trace(tuple(result["trace"]))
            )
            recorded = payload["evaluator_work"]
            cumulative += expected
            if (
                recorded["candidate_microwork"] != expected
                or recorded["cumulative_microwork"] != cumulative
            ):
                raise ValueError("archive evaluator-work ledger is inconsistent")
        if cumulative > self.plan.evaluator_work.max_microwork:
            raise ValueError("archive exceeds the registered evaluator-work budget")

    def _created(self) -> dict[int, dict[str, Any]]:
        return {
            event["payload"]["proposal_seq"]: event["payload"]
            for event in self.archive.events
            if event["type"] == "candidate_created"
        }

    def _completed(self) -> dict[int, dict[str, Any]]:
        return {
            event["payload"]["proposal_seq"]: event["payload"]
            for event in self.archive.events
            if event["type"] == "candidate_completed"
        }

    def _terminal(self) -> dict[str, Any] | None:
        terminal = [
            event["payload"]
            for event in self.archive.events
            if event["type"] == "campaign_terminal"
        ]
        if len(terminal) > 1:
            raise ValueError("archive contains multiple campaign terminal events")
        return terminal[0] if terminal else None

    def _best_incumbent_id(self) -> str | None:
        eligible = [
            payload
            for payload in self._completed().values()
            if payload["decision"].get("eligible")
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda payload: _fitness_from_dict(
                payload["evaluation"]["evaluation"]["fitness"]
            ).sort_key(),
        )["candidate_id"]

    def _terminate(self, *, reason: str, status: str) -> dict[str, Any]:
        spent = self._spent_microwork()
        maximum = self.plan.evaluator_work.max_microwork
        remaining = maximum - spent
        requires_postflight_receipt = (
            self.run_purpose == "scientific-full-work"
            and status == "complete"
            and reason == "normalized-work-budget-exhausted"
        )
        exhausted = reason == "normalized-work-budget-exhausted"
        carry = (
            remaining
            if exhausted and self.terminal_budget_policy == "carry-to-work-limit"
            else 0
        )
        discarded = (
            remaining
            if exhausted and self.terminal_budget_policy == "discard-at-trigger"
            else 0
        )
        payload = {
            "status": status,
            "reason": reason,
            "run_purpose": self.run_purpose,
            "completed_candidates": len(self._completed()),
            "candidate_safety_ceiling": self.plan.candidate_budget,
            "actual_microwork": spent,
            "registered_max_microwork": maximum,
            "remaining_microwork": remaining,
            "terminal_budget_policy": self.terminal_budget_policy,
            "budget_carry_microwork": carry,
            "unused_microwork_discarded": discarded,
            "effective_terminal_microwork": (
                maximum if exhausted else spent
            ),
            "carried_incumbent_id": self._best_incumbent_id(),
            # A campaign event cannot attest to state checked only after it is
            # durably appended. The CLI's separate postflight receipt is the sole
            # confirmatory-usability authority.
            "usable_for_confirmatory_analysis": False,
            "confirmatory_postflight_receipt_required": requires_postflight_receipt,
        }
        if self.engine_descriptor is not None:
            payload["proposal_engine"] = self.engine_descriptor
            usage_ledger = getattr(self.operator, "usage_ledger", None)
            payload["proposal_engine_accounting"] = (
                None if usage_ledger is None else usage_ledger.to_dict()
            )
        self.archive.append("campaign_terminal", payload)
        return payload

    def _pools(self) -> dict[int, list[str]]:
        pools = {index: [] for index in range(len(self.islands.profiles))}
        for event in self.archive.events:
            if event["type"] != "candidate_completed":
                continue
            decision = event["payload"]["decision"]
            island = event["payload"]["island_index"]
            if decision.get("entered_survivor"):
                pools[island].append(event["payload"]["candidate_id"])
            evicted = decision.get("evicted_candidate_id")
            if evicted is not None:
                pools[island].remove(evicted)
        return pools

    def _by_id(self) -> dict[str, dict[str, Any]]:
        return {payload["candidate_id"]: payload for payload in self._completed().values()}

    def _known_l1_behavior_signatures(self) -> frozenset[str]:
        signatures = set()
        for payload in self._completed().values():
            evaluation = payload.get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            for record in evaluation.get("trace", []):
                signature = record.get("behavior_signature")
                if isinstance(signature, dict) and isinstance(signature.get("sha256"), str):
                    signatures.add(signature["sha256"])
        return frozenset(signatures)

    def _recent_rejected_attempts(self, limit: int = 3) -> tuple[str, ...]:
        reasons: list[str] = []

        def add(reason: str) -> bool:
            compact = " ".join(reason.split())[:600]
            if compact and compact not in reasons:
                reasons.append(compact)
            return len(reasons) == limit

        for seq in sorted(self._completed(), reverse=True):
            completed = self._completed()[seq]
            evaluation = completed.get("evaluation")
            if isinstance(evaluation, dict):
                failure = evaluation.get("candidate_failure")
                if isinstance(failure, dict):
                    category = failure.get("failure_category", "candidate-invalid")
                    message = failure.get("message", "no failure message archived")
                    if add(f"{category}: {message}"):
                        return tuple(reasons)
                for trace in reversed(evaluation.get("trace", [])):
                    decision = trace.get("decision")
                    if decision == "reject":
                        result = trace.get("result", {})
                        if add(
                            f"{trace.get('rung', 'ladder')} reject: "
                            f"feasible={result.get('feasible')}, "
                            f"energy={result.get('normalized_energy')}, "
                            "shortfall="
                            f"{result.get('summed_primitive_shortfall')}"
                        ):
                            return tuple(reasons)
                    elif decision == "inert-not-promoted" and add(
                        "behavioral-novelty: inert-not-promoted"
                    ):
                        return tuple(reasons)
            proposal = completed["proposal"]
            for attempt in reversed(proposal.get("attempts", [])):
                if attempt.get("valid") is False and isinstance(attempt.get("error"), str):
                    if add(attempt["error"]):
                        return tuple(reasons)
        return tuple(reasons)

    def _program(self, candidate_id: str) -> Program:
        payload = self._by_id()[candidate_id]
        evaluation = payload.get("evaluation")
        public = evaluation["evaluation"] if evaluation is not None else {}
        return Program(
            id=candidate_id,
            code=payload["proposal"]["source"],
            language="python",
            parent_id=payload["parent_id"],
            island_idx=payload["island_index"],
            generation=payload["proposal_seq"],
            timestamp=0.0,
            public_metrics=public or {},
            text_feedback=payload.get("feedback_text", ""),
            correct=evaluation is not None,
            metadata=payload["proposal"].get("metadata", {}),
        )

    def _choose_parent(
        self, candidates: list[str], by_id: dict[str, Any], rng: random.Random
    ) -> str:
        """Pick the parent for this proposal from an island's survivor pool.

        The historical rule is uniform choice, which is what
        ``parent_selection_tournament_k == 1`` reproduces byte-for-byte. Uniform choice
        applies no selection pressure at all: with a work budget that admits ~42
        candidates across the islands, a pool barely fills once, essentially everything
        proposed survives, and the "evolutionary" loop degenerates into parallel random
        restarts of depth ~10 — which is a large part of why no proposal operator could
        separate from random mutation on the 2026-08 campaigns.

        With ``k > 1`` this becomes a k-way tournament on the Phase-3 lexicographic
        order (the same total order survivor selection uses), which is the cheapest
        standard way to restore pressure without touching the archive or the ladder. It
        runs in the shared runner, so the LLM and GP arms receive it identically.
        """
        k = self.plan.profile.parent_selection_tournament_k
        if k <= 1 or len(candidates) <= 1:
            return rng.choice(candidates)

        def sort_key(candidate_id: str) -> tuple:
            payload = by_id[candidate_id]
            evaluation = payload.get("evaluation")
            if evaluation is None:
                # Not yet evaluated: rank strictly last, deterministically.
                return (2, payload["proposal_seq"])
            return _fitness_from_dict(evaluation["evaluation"]["fitness"]).sort_key()

        drawn = [rng.choice(candidates) for _ in range(min(k, len(candidates)))]
        return min(drawn, key=lambda candidate_id: (sort_key(candidate_id), candidate_id))

    def _selection(self, proposal_seq: int, island: int) -> tuple[str, list[str], int]:
        pools = self._pools()
        candidates = list(pools[island])
        if not candidates:
            candidates = [item for pool in pools.values() for item in pool]
        if not candidates:
            candidates = [self._completed()[0]["candidate_id"]]
        selection_seed = derive_seed(
            self.definition.root_seed,
            "evolution",
            "selection",
            proposal_seq,
            island,
        )
        rng = random.Random(selection_seed)
        by_id = self._by_id()
        parent_id = self._choose_parent(sorted(candidates), by_id, rng)
        if self.plan.profile.diverse_inspirations_enabled:
            inspiration_candidates = sorted(
                {
                    candidate_id
                    for pool in pools.values()
                    for candidate_id in pool
                    if candidate_id != parent_id
                }
            )
            representatives: dict[str, str] = {}
            for candidate_id in inspiration_candidates:
                payload = by_id[candidate_id]
                signature = next(
                    (
                        record["behavior_signature"]["sha256"]
                        for record in payload["evaluation"]["trace"]
                        if isinstance(record.get("behavior_signature"), dict)
                    ),
                    hashlib.sha256(payload["proposal"]["source"].encode()).hexdigest(),
                )
                previous = representatives.get(signature)
                if previous is None or _fitness_from_dict(
                    payload["evaluation"]["evaluation"]["fitness"]
                ).sort_key() < _fitness_from_dict(
                    by_id[previous]["evaluation"]["evaluation"]["fitness"]
                ).sort_key():
                    representatives[signature] = candidate_id
            ranked_diverse = sorted(
                representatives.values(),
                key=lambda cid: _fitness_from_dict(
                    by_id[cid]["evaluation"]["evaluation"]["fitness"]
                ).sort_key(),
            )
            return (
                parent_id,
                ranked_diverse[: self.plan.profile.max_inspirations],
                selection_seed,
            )
        ranked = sorted(
            (cid for cid in candidates if cid != parent_id),
            key=lambda cid: _fitness_from_dict(
                by_id[cid]["evaluation"]["evaluation"]["fitness"]
            ).sort_key(),
        )
        return parent_id, ranked[:2], selection_seed

    def _allocate(self, proposal_seq: int) -> dict[str, Any]:
        island = proposal_seq % len(self.islands.profiles)
        parent_id, inspirations, selection_seed = self._selection(proposal_seq, island)
        payload = {
            "proposal_seq": proposal_seq,
            "candidate_id": (f"{self.definition.campaign_id}:{self.arm}:{proposal_seq:08d}"),
            "island_index": island,
            "island_id": self.islands.profiles[island].island_id,
            "parent_id": parent_id,
            "inspiration_ids": inspirations,
            "selection_seed": selection_seed,
            "operator": self.operator.name,
        }
        if self.engine_descriptor is not None:
            payload["proposal_engine"] = self.engine_descriptor
        self.archive.append("candidate_created", payload)
        return payload

    def _allocate_batch(
        self,
        proposal_seq: int,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """Durably allocate one sibling batch before its shared model dispatch."""
        if batch_size < 2:
            raise ValueError("batch allocation requires at least two candidates")
        prior_calls = {
            event["payload"].get("shared_call_id")
            for event in self.archive.events
            if event["type"] == "candidate_created"
            and event["payload"].get("shared_call_id") is not None
        }
        call_seq = len(prior_calls)
        island = (call_seq + 1) % len(self.islands.profiles)
        parent_id, inspirations, selection_seed = self._selection(proposal_seq, island)
        shared_call_id = (
            f"{self.definition.campaign_id}:{self.arm}:call:{call_seq:08d}"
        )
        allocated = []
        for offset in range(batch_size):
            seq = proposal_seq + offset
            payload = {
                "proposal_seq": seq,
                "candidate_id": (
                    f"{self.definition.campaign_id}:{self.arm}:{seq:08d}"
                ),
                "island_index": island,
                "island_id": self.islands.profiles[island].island_id,
                "parent_id": parent_id,
                "inspiration_ids": inspirations,
                "selection_seed": selection_seed,
                "operator": self.operator.name,
                "shared_call_id": shared_call_id,
                "batch_index": offset + 1,
                "batch_size": batch_size,
            }
            if self.engine_descriptor is not None:
                payload["proposal_engine"] = self.engine_descriptor
            self.archive.append("candidate_created", payload)
            allocated.append(payload)
        return allocated

    def _finish_batch_allocation(
        self,
        created_batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Finish a batch whose durable creation sequence was interrupted mid-append."""
        first = created_batch[0]
        expected_size = first["batch_size"]
        if len(created_batch) > expected_size:
            raise ValueError("archive batch contains more candidates than declared")
        for offset in range(len(created_batch), expected_size):
            seq = first["proposal_seq"] + offset
            payload = {
                **{
                    key: first[key]
                    for key in (
                        "island_index",
                        "island_id",
                        "parent_id",
                        "inspiration_ids",
                        "selection_seed",
                        "operator",
                        "shared_call_id",
                        "batch_size",
                    )
                },
                "proposal_seq": seq,
                "candidate_id": (
                    f"{self.definition.campaign_id}:{self.arm}:{seq:08d}"
                ),
                "batch_index": offset + 1,
            }
            if self.engine_descriptor is not None:
                payload["proposal_engine"] = self.engine_descriptor
            self.archive.append("candidate_created", payload)
            created_batch.append(payload)
        return created_batch

    def _proposal(
        self,
        created: dict[str, Any],
        *,
        initial_offspring: ProposalOffspring | None = None,
    ) -> dict[str, Any]:
        seq = created["proposal_seq"]
        if seq == 0:
            return {
                "valid": True,
                "source": self.initial_source,
                "error": None,
                "metadata": {"operator": "initial_seed"},
                "attempts": [],
            }
        parent = self._program(created["parent_id"])
        inspirations = tuple(self._program(cid) for cid in created["inspiration_ids"])
        attempts = []
        repair: ValidationRepair | None = None
        repair_mode: str | None = None
        for attempt in range(self.plan.max_operator_attempts):
            route_seed = derive_seed(
                self.definition.root_seed,
                "evolution",
                "proposal",
                "shared-route",
                seq,
                attempt,
            )
            numeric_seed = derive_seed(
                self.definition.root_seed,
                "evolution",
                "proposal",
                "shared-numeric",
                seq,
                attempt,
            )
            primary_seed = derive_seed(
                self.definition.root_seed,
                "evolution",
                self.arm,
                "primary-proposal",
                seq,
                attempt,
            )
            if attempt == 0 and initial_offspring is not None:
                offspring = initial_offspring
            elif isinstance(self.operator, AblationProbeInterleaveOperator):
                ablation_seed = derive_seed(
                    self.definition.root_seed,
                    "evolution",
                    "proposal",
                    "ablation-probe",
                    seq,
                    attempt,
                )
                offspring = asyncio.run(
                    self.operator.propose_seeded(
                        parent,
                        inspirations,
                        route_seed=route_seed,
                        ablation_seed=ablation_seed,
                        numeric_seed=numeric_seed,
                        primary_seed=primary_seed,
                        primary_mode=repair_mode,
                        primary_repair=repair,
                    )
                )
            elif isinstance(self.operator, SharedNumericInterleaveOperator):
                offspring = asyncio.run(
                    self.operator.propose_seeded(
                        parent,
                        inspirations,
                        route_seed=route_seed,
                        numeric_seed=numeric_seed,
                        primary_seed=primary_seed,
                        primary_mode=repair_mode,
                        primary_repair=repair,
                    )
                )
            elif repair is not None:
                repair_method = getattr(self.operator, "propose_with_repair", None)
                if not callable(repair_method):
                    raise ValueError("operator does not support configured grammar-aware retry")
                offspring = repair_method(
                    parent,
                    inspirations,
                    random.Random(primary_seed),
                    mode=repair_mode,
                    repair=repair,
                )
            else:
                offspring = asyncio.run(
                    invoke_proposal_operator(
                        self.operator,
                        parent,
                        inspirations,
                        random.Random(primary_seed),
                    )
                )
            record = {
                "attempt": attempt,
                "shared_route_seed": route_seed,
                "shared_numeric_seed": numeric_seed,
                "arm_primary_seed": primary_seed,
                "valid": offspring.valid,
                "error": offspring.error,
                "metadata": offspring.metadata,
                "repair_of_attempt": (attempt - 1 if repair is not None else None),
            }
            if isinstance(self.operator, AblationProbeInterleaveOperator):
                record["ablation_probe_seed"] = ablation_seed
            attempts.append(record)
            campaign_stop_reason = offspring.metadata.get("campaign_stop_reason")
            if campaign_stop_reason is not None:
                return {
                    "valid": False,
                    "source": None,
                    "error": offspring.error,
                    "metadata": {
                        "campaign_stop_reason": campaign_stop_reason,
                        "engine_descriptor": offspring.metadata.get(
                            "engine_descriptor", self.engine_descriptor
                        ),
                    },
                    "attempts": attempts,
                }
            if not offspring.valid:
                failure = offspring.metadata.get("validation_failure")
                if (
                    self.plan.profile.grammar_aware_retry_enabled
                    and isinstance(failure, dict)
                    and isinstance(failure.get("error"), str)
                    and isinstance(failure.get("offending_excerpt"), str)
                ):
                    # A repair is an *optimization* over a blind retry, so a repair
                    # payload the constructor rejects must degrade to the blind retry
                    # rather than kill the campaign. `ValidationRepair` caps the error at
                    # 2000 characters; a model whose validation errors run longer than
                    # that (GLM's do, where the Codex engine's never did) otherwise takes
                    # the whole arm down mid-run with "repair error must be non-empty and
                    # bounded" -- 43 minutes and 29 archived candidates lost on the first
                    # root that hit it.
                    try:
                        repair = ValidationRepair(
                            error=failure["error"],
                            offending_excerpt=failure["offending_excerpt"],
                        )
                    except ValueError as exc:
                        record["repair_unavailable"] = str(exc)
                        repair = None
                        repair_mode = None
                        continue
                    proposal_mode = offspring.metadata.get("proposal_mode")
                    if not isinstance(proposal_mode, str):
                        raise ValueError("validation failure omitted its proposal mode")
                    repair_mode = proposal_mode
                else:
                    repair = None
                    repair_mode = None
                continue
            try:
                ps.Policy(offspring.source)
                validate_scoped_edit(parent.code, offspring.source)
            except ps.PlacementScriptError as exc:
                record["valid"] = False
                record["error"] = f"runner validation: {exc}"
                continue
            except ValueError as exc:
                record["valid"] = False
                record["error"] = f"runner immutable-boundary validation: {exc}"
                continue
            return {
                "valid": True,
                "source": offspring.source,
                "error": None,
                "metadata": offspring.metadata,
                "attempts": attempts,
            }
        return {
            "valid": False,
            "source": None,
            "error": "operator exhausted the shared invalid-offspring retry budget",
            "metadata": {},
            "attempts": attempts,
        }

    def _decision(
        self,
        created: dict[str, Any],
        result: LadderResult | None,
        assessment: ArchiveAssessment | None,
    ) -> dict[str, Any]:
        if (
            result is None
            or not result.promoted_to_l2
            or result.evaluation is None
            or assessment is None
        ):
            return {
                "eligible": False,
                "entered_survivor": False,
                "evicted_candidate_id": None,
                "survivor_key": None,
                "elite_view": {
                    "eligible": False,
                    "entered": False,
                    "evicted_candidate_id": None,
                    "key": None,
                },
            }
        fitness = result.evaluation.fitness
        island = created["island_index"]
        pool = self._pools()[island]
        by_id = self._by_id()
        entries = [
            (
                cid,
                _fitness_from_dict(by_id[cid]["evaluation"]["evaluation"]["fitness"]),
            )
            for cid in pool
        ]
        entries.append((created["candidate_id"], fitness))
        ranked = sorted(entries, key=lambda item: item[1].sort_key())
        survivors = ranked[: self.plan.population_per_island]
        survivor_ids = {candidate_id for candidate_id, _ in survivors}
        entered = created["candidate_id"] in survivor_ids
        evicted = next((cid for cid in pool if cid not in survivor_ids), None)
        scientific_key = fitness.sort_key()
        elite_key = (scientific_key[0], assessment.total_penalty, *scientific_key[1:])
        prior_elite_entries = []
        for candidate_id, payload in by_id.items():
            if payload["island_index"] != island or not payload["decision"].get("eligible"):
                continue
            prior_fitness = _fitness_from_dict(payload["evaluation"]["evaluation"]["fitness"])
            prior_scientific = prior_fitness.sort_key()
            prior_elite_entries.append(
                (
                    candidate_id,
                    (
                        prior_scientific[0],
                        payload["archive_assessment"]["total_penalty"],
                        *prior_scientific[1:],
                    ),
                )
            )
        previous_elites = {
            candidate_id
            for candidate_id, _key in sorted(prior_elite_entries, key=lambda item: item[1])[
                : self.plan.population_per_island
            ]
        }
        new_elites = {
            candidate_id
            for candidate_id, _key in sorted(
                [*prior_elite_entries, (created["candidate_id"], elite_key)],
                key=lambda item: item[1],
            )[: self.plan.population_per_island]
        }
        return {
            "eligible": True,
            "entered_survivor": entered,
            "evicted_candidate_id": evicted,
            "survivor_key": list(fitness.sort_key()),
            "elite_view": {
                "eligible": True,
                "entered": created["candidate_id"] in new_elites,
                "evicted_candidate_id": next(iter(previous_elites - new_elites), None),
                "key": list(elite_key),
            },
        }

    def _complete(
        self,
        created: dict[str, Any],
        *,
        proposal: dict[str, Any] | None = None,
    ) -> str | None:
        if proposal is None:
            proposal = self._proposal(created)
        result = None
        assessment = None
        feedback = None
        if proposal["valid"]:
            profile = self.islands.profiles[created["island_index"]]
            island_task = replace(
                self.task,
                constraints=profile.constraints(self.task.constraints),
            )
            if self.behavioral_novelty is None:
                result = (
                    evaluate_through_ladder(
                        proposal["source"],
                        island_task,
                        self.ladder,
                        proposal_seq=created["proposal_seq"],
                        retain_episode_ledgers=self.retain_episode_ledgers,
                    )
                    if self.evaluate_fn is _default_evaluate
                    else self.evaluate_fn(
                        proposal["source"], island_task, self.ladder, created["proposal_seq"]
                    )
                )
            else:
                known = self._known_l1_behavior_signatures()
                if self.evaluate_fn is _default_evaluate:
                    result = evaluate_through_ladder(
                        proposal["source"],
                        island_task,
                        self.ladder,
                        proposal_seq=created["proposal_seq"],
                        behavioral_novelty=self.behavioral_novelty,
                        known_l1_signatures=known,
                        retain_episode_ledgers=self.retain_episode_ledgers,
                    )
                else:
                    # A caller-supplied evaluator owns its own retention policy: it may
                    # be an out-of-process guard or a replay, and its signature is not
                    # ours to extend.
                    result = self.evaluate_fn(
                        proposal["source"],
                        island_task,
                        self.ladder,
                        created["proposal_seq"],
                        behavioral_novelty=self.behavioral_novelty,
                        known_l1_signatures=known,
                    )
            expected_identity = policy_identity_for_source(proposal["source"])
            if (
                result.policy_name != expected_identity["name"]
                or result.policy_version != expected_identity["version"]
                or result.policy_identity_sha256
                != expected_identity["identity_sha256"]
                or result.policy_artifact_sha256
                != expected_identity["artifact_sha256"]
            ):
                raise ValueError("campaign evaluator returned a mismatched policy identity")
            if result.candidate_failure is not None:
                _invalidate_evaluated_proposal(proposal, result.candidate_failure)
            else:
                try:
                    assessment = self.assess_fn(
                        proposal["source"], island_task, self.islands
                    )
                except ps.PlacementScriptError as exc:
                    failure = candidate_failure_metadata(
                        exc,
                        stage="archive-assessment",
                    )
                    result = replace(
                        result,
                        promoted_to_l2=False,
                        evaluation=None,
                        candidate_failure=failure,
                    )
                    _invalidate_evaluated_proposal(proposal, failure)
                else:
                    feedback = generate_feedback(
                        result,
                        self.feedback_profile,
                        constraints=(
                            island_task.constraints
                            if self.plan.profile.simulator_feedback_enabled
                            else None
                        ),
                        recent_rejections=(
                            self._recent_rejected_attempts()
                            if self.plan.profile.simulator_feedback_enabled
                            else ()
                        ),
                    )
        decision = self._decision(created, result, assessment)
        payload = {
            **created,
            "proposal": proposal,
            "evaluation": (
                None if result is None else self._compact_episode_ledgers(result.to_dict())
            ),
            "archive_assessment": None if assessment is None else assessment.to_dict(),
            "feedback": None if feedback is None else feedback.to_dict(),
            "feedback_text": "" if feedback is None else feedback.render_text(),
            "decision": decision,
            "evaluator_work": self._work_payload(result),
        }
        self.archive.append("candidate_completed", payload)
        self.archive.materialize_candidates()
        return proposal.get("metadata", {}).get("campaign_stop_reason")

    def _dispatch_batch(
        self,
        created_batch: list[dict[str, Any]],
    ) -> tuple[ProposalOffspring, ...]:
        first = created_batch[0]
        parent = self._program(first["parent_id"])
        inspirations = tuple(
            self._program(candidate_id) for candidate_id in first["inspiration_ids"]
        )
        for created in created_batch:
            if any(
                created[field] != first[field]
                for field in (
                    "island_index",
                    "island_id",
                    "parent_id",
                    "inspiration_ids",
                    "selection_seed",
                    "operator",
                    "shared_call_id",
                    "batch_size",
                )
            ):
                raise ValueError("batched candidate creation fields are inconsistent")
        candidate_attempt_seeds = tuple(
            derive_seed(
                self.definition.root_seed,
                "evolution",
                self.arm,
                "primary-proposal",
                created["proposal_seq"],
                0,
            )
            for created in created_batch
        )
        if len(set(candidate_attempt_seeds)) != len(candidate_attempt_seeds):
            raise ValueError("batched candidate attempt seeds collided")
        batch_method = getattr(self.operator, "propose_batch", None)
        if not callable(batch_method):
            raise ValueError("configured batched proposals require a batch-capable operator")
        batch_rng_seed = derive_seed(
            self.definition.root_seed,
            "evolution",
            self.arm,
            "batched-proposal-call",
            first["shared_call_id"],
        )
        offspring = batch_method(
            parent,
            inspirations,
            random.Random(batch_rng_seed),
            proposals_per_call=len(created_batch),
            shared_call_id=first["shared_call_id"],
            candidate_attempt_seeds=candidate_attempt_seeds,
        )
        if not isinstance(offspring, tuple) or len(offspring) != len(created_batch):
            raise ValueError("batch operator returned a different number of candidate slots")
        return offspring

    def _replay_batch(
        self,
        created_batch: list[dict[str, Any]],
    ) -> tuple[ProposalOffspring, ...] | None:
        """Recover pending siblings from the owner's once-only archived call evidence."""
        owner_seq = created_batch[0]["proposal_seq"]
        owner = self._completed().get(owner_seq)
        if owner is None:
            return None
        attempts = owner["proposal"].get("attempts", [])
        if not attempts:
            raise ValueError("completed batch owner has no proposal attempt evidence")
        owner_offspring = ProposalOffspring(
            source=attempts[0].get("metadata", {}).get("batch_owner_initial_source"),
            error=(
                attempts[0].get("error")
                or "already-completed batch owner replay placeholder"
            ),
            metadata=attempts[0]["metadata"],
        )
        replay = owner_offspring.metadata.get("batch_replay")
        if not isinstance(replay, list) or len(replay) != len(created_batch) - 1:
            raise ValueError("completed batch owner has invalid replay evidence")
        siblings = []
        for record in replay:
            if not isinstance(record, dict) or set(record) != {
                "source",
                "error",
                "metadata",
            }:
                raise ValueError("batch replay outcome is malformed")
            siblings.append(
                ProposalOffspring(
                    source=record["source"],
                    error=record["error"],
                    metadata=record["metadata"],
                )
            )
        return (owner_offspring, *siblings)

    def _complete_batch(
        self,
        created_batch: list[dict[str, Any]],
    ) -> str | None:
        outcomes = self._replay_batch(created_batch)
        if outcomes is None:
            outcomes = self._dispatch_batch(created_batch)
        completed = self._completed()
        stop_reason = None
        for created, offspring in zip(created_batch, outcomes, strict=True):
            if created["proposal_seq"] in completed:
                continue
            proposal = self._proposal(created, initial_offspring=offspring)
            candidate_stop = self._complete(created, proposal=proposal)
            if stop_reason is None and candidate_stop is not None:
                stop_reason = candidate_stop
        return stop_reason

    def run(self) -> str:
        lease_path = self.archive.root / "campaign-process.lock"
        with lease_path.open("a+") as lease:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "campaign archive already has an active process; concurrent resume "
                    f"refused before dispatch: {self.archive.root}"
                ) from exc
            try:
                return self._run_with_lease()
            finally:
                usage_ledger = getattr(self.operator, "usage_ledger", None)
                release_journal_lease = getattr(
                    usage_ledger, "release_journal_lease", None
                )
                if callable(release_journal_lease):
                    release_journal_lease()
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)

    def _run_with_lease(self) -> str:
        terminal = self._terminal()
        if terminal is not None:
            # A crash can occur after the durable terminal append but before the
            # derived candidate view is refreshed. Resume repairs that view first.
            self.archive.materialize_candidates()
            if terminal["status"] == "unusable":
                raise RuntimeError(
                    f"campaign archive is unusable: {terminal['reason']}"
                )
            return self.archive.digest()
        created = self._created()
        completed = self._completed()
        if not created:
            initial = {
                "proposal_seq": 0,
                "candidate_id": f"{self.definition.campaign_id}:{self.arm}:00000000",
                "island_index": 0,
                "island_id": self.islands.profiles[0].island_id,
                "parent_id": None,
                "inspiration_ids": [],
                "selection_seed": None,
                "operator": "initial_seed",
            }
            if self.engine_descriptor is not None:
                initial["proposal_engine"] = self.engine_descriptor
            self.archive.append("candidate_created", initial)
            created[0] = initial
        while True:
            created = self._created()
            completed = self._completed()
            pending = sorted(set(created) - set(completed))
            if pending:
                first_pending = created[pending[0]]
                shared_call_id = first_pending.get("shared_call_id")
                if shared_call_id is None:
                    campaign_stop_reason = self._complete(first_pending)
                else:
                    created_batch = sorted(
                        (
                            payload
                            for payload in created.values()
                            if payload.get("shared_call_id") == shared_call_id
                        ),
                        key=lambda payload: payload["batch_index"],
                    )
                    created_batch = self._finish_batch_allocation(created_batch)
                    campaign_stop_reason = self._complete_batch(created_batch)
                if campaign_stop_reason is not None:
                    self._terminate(reason=campaign_stop_reason, status="unusable")
                    raise RuntimeError(
                        f"campaign proposal-engine ceiling reached: {campaign_stop_reason}"
                    )
                continue
            if (
                self._spent_microwork() + self.plan.evaluator_work.max_candidate_microwork
                > self.plan.evaluator_work.max_microwork
            ):
                self._terminate(
                    reason="normalized-work-budget-exhausted",
                    status="complete",
                )
                break
            if len(completed) >= self.plan.candidate_budget:
                if self.run_purpose == "determinism-smoke":
                    self._terminate(
                        reason="determinism-smoke-candidate-target",
                        status="complete",
                    )
                    break
                self._terminate(
                    reason="candidate-safety-ceiling-reached",
                    status="unusable",
                )
                raise RuntimeError(
                    "candidate safety ceiling reached before normalized work exhaustion"
                )
            next_seq = max(created, default=-1) + 1
            remaining_candidates = self.plan.candidate_budget - len(completed)
            remaining_work_slots = (
                self.plan.evaluator_work.max_microwork - self._spent_microwork()
            ) // self.plan.evaluator_work.max_candidate_microwork
            batch_size = min(
                self.plan.proposals_per_call,
                remaining_candidates,
                remaining_work_slots,
            )
            batch_method = getattr(self.operator, "propose_batch", None)
            if self.arm == "llm" and batch_size > 1 and callable(batch_method):
                self._allocate_batch(next_seq, batch_size)
            else:
                self._allocate(next_seq)
        self.archive.materialize_candidates()
        return self.archive.digest()
