"""P1-gated, paired bitwise-determinism acceptance harness (TODO 5.9)."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shinka.core.proposal_operator import ProposalOperator

from .campaign import CampaignDefinition, CampaignRunner, load_campaign_plan
from .evaluation import TaskEvaluationConfig, load_task_evaluation_config
from .feedback import MATCHED_PAIR_PROFILE, FeedbackProfile, load_feedback_profiles
from .gp_operator import PlacementScriptGPOperator
from .islands import ArchiveAssessment, IslandsConfig, load_islands_config
from .ladder import LadderConfig, LadderResult, load_ladder_config
from .llm_operator import OpenWeightLLMOperator, VLLMCompletionClient
from .model import (
    ApiCostLedger,
    ModelTrack,
    host_handshake_sha256,
    load_model_track,
    verify_host_handshake,
    verify_weight_directory,
)
from .operators import SharedNumericInterleaveOperator
from .run_campaign import (
    CANONICAL_INPUTS,
    _endpoint_origin,
    _file_sha256,
    _git_state,
    _require_external_archive_path,
    _resolve_frozen_tag,
    _tagged_blob,
    _verify_canonical_tagged_inputs,
)

ROOT = Path(__file__).parents[1]
PROTOCOL_SCHEMA = "evolved-continuum-g-ablate-protocol-v3"
REPORT_SCHEMA = "evolved-continuum-g-ablate-report-v2"
ACCEPTANCE_RULE = "two-fresh-runs-per-arm-have-identical-complete-archive-bytes"

EvaluateFn = Callable[[str, TaskEvaluationConfig, LadderConfig, int], LadderResult]
AssessFn = Callable[[str, TaskEvaluationConfig, IslandsConfig], ArchiveAssessment]
OperatorFactory = Callable[[], ProposalOperator]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AblationProtocol:
    status: str
    frozen_tag: str | None
    proposal_endpoint_origin: str | None
    proposal_host_handshake_sha256: str | None
    campaign_id: str
    candidate_budget: int
    min_valid_primary_offspring_per_arm: int
    min_valid_shared_numeric_offspring_per_arm: int
    repetitions: int
    arms: tuple[str, ...]
    feedback_profile: str
    acceptance_rule: str

    def __post_init__(self) -> None:
        if self.status not in {"development-provisional", "p1-frozen"}:
            raise ValueError("unknown G-ablate protocol status")
        if self.status == "development-provisional" and self.frozen_tag is not None:
            raise ValueError("a development G-ablate protocol cannot claim a frozen tag")
        if self.status == "development-provisional" and (
            self.proposal_endpoint_origin is not None
            or self.proposal_host_handshake_sha256 is not None
        ):
            raise ValueError("a development G-ablate protocol cannot claim proposal-host pins")
        if self.status == "p1-frozen" and (
            not isinstance(self.frozen_tag, str)
            or not self.frozen_tag
            or self.frozen_tag.startswith("-")
            or any(char.isspace() for char in self.frozen_tag)
        ):
            raise ValueError("a P1-frozen G-ablate protocol requires its predeclared Git tag")
        if self.status == "p1-frozen" and (
            not isinstance(self.proposal_endpoint_origin, str)
            or not self.proposal_endpoint_origin
            or not isinstance(self.proposal_host_handshake_sha256, str)
            or len(self.proposal_host_handshake_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.proposal_host_handshake_sha256
            )
        ):
            raise ValueError("a P1-frozen G-ablate protocol requires exact proposal-host pins")
        if not self.campaign_id:
            raise ValueError("G-ablate campaign_id is required")
        if (
            not isinstance(self.candidate_budget, int)
            or isinstance(self.candidate_budget, bool)
            or self.candidate_budget < 2
        ):
            raise ValueError("G-ablate needs the seed and at least one offspring")
        for field in (
            "min_valid_primary_offspring_per_arm",
            "min_valid_shared_numeric_offspring_per_arm",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            1
            + self.min_valid_primary_offspring_per_arm
            + self.min_valid_shared_numeric_offspring_per_arm
            > self.candidate_budget
        ):
            raise ValueError("G-ablate budget cannot exercise both registered proposal routes")
        if self.repetitions != 2:
            raise ValueError("G-ablate requires exactly two fresh runs per arm")
        if self.arms != ("llm", "gp"):
            raise ValueError("G-ablate has exactly the matched llm/gp arms")
        if self.feedback_profile != MATCHED_PAIR_PROFILE:
            raise ValueError("G-ablate must disable unmatched simulator feedback")
        if self.acceptance_rule != ACCEPTANCE_RULE:
            raise ValueError("G-ablate acceptance rule differs from the registered rule")

    @property
    def acceptance_ready(self) -> bool:
        return self.status == "p1-frozen"


def load_protocol(path: str | Path) -> AblationProtocol:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"G-ablate protocol schema must be {PROTOCOL_SCHEMA}")
    return AblationProtocol(
        status=raw["status"],
        frozen_tag=raw.get("frozen_tag"),
        proposal_endpoint_origin=raw.get("proposal_endpoint_origin"),
        proposal_host_handshake_sha256=raw.get("proposal_host_handshake_sha256"),
        campaign_id=raw["campaign_id"],
        candidate_budget=raw["candidate_budget"],
        min_valid_primary_offspring_per_arm=raw[
            "min_valid_primary_offspring_per_arm"
        ],
        min_valid_shared_numeric_offspring_per_arm=raw[
            "min_valid_shared_numeric_offspring_per_arm"
        ],
        repetitions=raw["repetitions"],
        arms=tuple(raw["arms"]),
        feedback_profile=raw["feedback_profile"],
        acceptance_rule=raw["acceptance_rule"],
    )


def _archive_inventory(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return files


def _archive_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


@dataclass(frozen=True)
class _ComparisonResult:
    definition: CampaignDefinition
    root: Path
    passed: bool
    arm_reports: dict[str, Any]


def _smoke_coverage(
    runner: CampaignRunner, protocol: AblationProtocol, arm: str
) -> dict[str, Any]:
    completed = runner._completed()
    terminal = runner._terminal()
    valid_primary = []
    valid_numeric = []
    valid_llm_records = []
    for seq, payload in completed.items():
        if seq == 0 or not payload["proposal"]["valid"]:
            continue
        metadata = payload["proposal"]["metadata"]
        if metadata.get("route") == "primary":
            valid_primary.append(payload["candidate_id"])
            if arm == "llm" and (
                isinstance(metadata.get("raw_response"), dict)
                and metadata.get("cost_basis")
                in {"host-reported", "conservative-reservation"}
                and isinstance(metadata.get("api_costs_microusd"), int)
                and not isinstance(metadata.get("api_costs_microusd"), bool)
                and metadata["api_costs_microusd"] >= 0
            ):
                valid_llm_records.append(payload["candidate_id"])
        elif metadata.get("route") == "shared_numeric":
            valid_numeric.append(payload["candidate_id"])
    terminal_ok = (
        terminal is not None
        and terminal["status"] == "complete"
        and terminal["reason"] == "determinism-smoke-candidate-target"
        and terminal["completed_candidates"] == protocol.candidate_budget
    )
    passed = (
        len(completed) == protocol.candidate_budget
        and terminal_ok
        and len(valid_primary) >= protocol.min_valid_primary_offspring_per_arm
        and len(valid_numeric) >= protocol.min_valid_shared_numeric_offspring_per_arm
        and (arm != "llm" or len(valid_llm_records) >= 1)
    )
    return {
        "passed": passed,
        "completed_candidates": len(completed),
        "terminal_ok": terminal_ok,
        "valid_primary_candidate_ids": valid_primary,
        "valid_shared_numeric_candidate_ids": valid_numeric,
        "valid_llm_completion_cost_candidate_ids": valid_llm_records,
    }


def _compare_archives(
    *,
    protocol: AblationProtocol,
    campaign_plan_path: str | Path,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    islands: IslandsConfig,
    feedback_profile: FeedbackProfile,
    initial_source: str,
    operator_factories: dict[str, OperatorFactory],
    archive_root: str | Path,
    provenance: dict[str, Any],
    evaluate_fn: EvaluateFn | None = None,
    assess_fn: AssessFn | None = None,
) -> _ComparisonResult:
    """Run two fresh archives per arm and compare their complete byte inventories."""
    plan = load_campaign_plan(campaign_plan_path)
    if protocol.candidate_budget > plan.candidate_budget:
        raise ValueError("smoke candidate budget exceeds the campaign budget")
    if protocol.arms != plan.arms or feedback_profile.name != protocol.feedback_profile:
        raise ValueError("gate arms/feedback differ from the paired campaign plan")
    definition = next(
        (item for item in plan.campaigns if item.campaign_id == protocol.campaign_id),
        None,
    )
    if definition is None:
        raise ValueError("G-ablate campaign_id is absent from the campaign plan")
    if set(operator_factories) != set(protocol.arms):
        raise ValueError("operator factories must cover exactly both registered arms")
    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=False)
    smoke_plan = dataclasses.replace(plan, candidate_budget=protocol.candidate_budget)
    arm_reports: dict[str, Any] = {}
    for arm in protocol.arms:
        repetitions = []
        byte_snapshots = []
        for repeat in range(protocol.repetitions):
            archive = root / arm / f"run-{repeat + 1}"
            kwargs: dict[str, Any] = {}
            if evaluate_fn is not None:
                kwargs["evaluate_fn"] = evaluate_fn
            if assess_fn is not None:
                kwargs["assess_fn"] = assess_fn
            runner = CampaignRunner(
                definition=definition,
                plan=smoke_plan,
                arm=arm,
                operator=SharedNumericInterleaveOperator(
                    operator_factories[arm](),
                    probability=plan.shared_numeric_probability,
                ),
                initial_source=initial_source,
                task=task,
                ladder=ladder,
                islands=islands,
                feedback_profile=feedback_profile,
                archive_dir=archive,
                provenance=provenance,
                run_purpose="determinism-smoke",
                **kwargs,
            )
            archive_sha256 = runner.run()
            coverage = _smoke_coverage(runner, protocol, arm)
            byte_snapshots.append(_archive_bytes(archive))
            repetitions.append(
                {
                    "archive_sha256": archive_sha256,
                    "files": _archive_inventory(archive),
                    "coverage": coverage,
                }
            )
        identical = byte_snapshots[0] == byte_snapshots[1]
        arm_reports[arm] = {
            "bitwise_identical": identical,
            "coverage_passed": all(run["coverage"]["passed"] for run in repetitions),
            "runs": repetitions,
        }
    passed = all(
        report["bitwise_identical"] and report["coverage_passed"]
        for report in arm_reports.values()
    )
    return _ComparisonResult(
        definition=definition,
        root=root,
        passed=passed,
        arm_reports=arm_reports,
    )


def _write_report(root: Path, report: dict[str, Any]) -> None:
    report_path = root / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(report) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_gate(
    *,
    protocol: AblationProtocol,
    campaign_plan_path: str | Path,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    islands: IslandsConfig,
    feedback_profile: FeedbackProfile,
    initial_source: str,
    operator_factories: dict[str, OperatorFactory],
    archive_root: str | Path,
    provenance: dict[str, Any],
    evaluate_fn: EvaluateFn | None = None,
    assess_fn: AssessFn | None = None,
) -> dict[str, Any]:
    """Development-only injected harness; structurally incapable of emitting PASS."""
    if protocol.acceptance_ready:
        raise ValueError("the injected run_gate API is development-only")
    comparison = _compare_archives(
        protocol=protocol,
        campaign_plan_path=campaign_plan_path,
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback_profile,
        initial_source=initial_source,
        operator_factories=operator_factories,
        archive_root=archive_root,
        provenance=provenance,
        evaluate_fn=evaluate_fn,
        assess_fn=assess_fn,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "DEVELOPMENT_" + ("PASS" if comparison.passed else "FAIL"),
        "acceptance_evidence": False,
        "protocol": dataclasses.asdict(protocol),
        "campaign_root_row": dataclasses.asdict(comparison.definition),
        "provenance": provenance,
        "arms": comparison.arm_reports,
    }
    _write_report(comparison.root, report)
    return report


def _verify_gate_tagged_inputs(tag: str) -> dict[str, Path]:
    verified = _verify_canonical_tagged_inputs(CANONICAL_INPUTS, tag=tag)
    extra = {
        "g_ablate_protocol": ROOT / "evolution/g_ablate.json",
        "g_ablate_implementation": ROOT / "evolution/g_ablate.py",
    }
    for name, path in extra.items():
        relative = path.relative_to(ROOT).as_posix()
        if path.read_bytes() != _tagged_blob(ROOT, tag, relative):
            raise ValueError(f"acceptance {name} differs from frozen tag {tag}")
    return {**verified, **extra}


@dataclass(frozen=True)
class _AcceptanceSnapshot:
    files: dict[str, str]
    repository: dict[str, object]
    shinkaevolve_repository: dict[str, object]
    host_handshake: dict[str, Any]
    host_handshake_sha256: str

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _capture_acceptance_snapshot(
    *,
    protocol: AblationProtocol,
    track: ModelTrack,
    client: VLLMCompletionClient,
    weight_directory: str | Path,
) -> _AcceptanceSnapshot:
    """Revalidate every mutable acceptance dependency at one boundary."""
    if protocol.frozen_tag is None:
        raise RuntimeError("G-ablate acceptance protocol has no frozen tag")
    root_state = _git_state(ROOT)
    if (
        root_state["head"] != _resolve_frozen_tag(ROOT, protocol.frozen_tag)
        or root_state["dirty_entries"]
    ):
        raise RuntimeError("G-ablate requires its frozen clean repository commit")
    paths = _verify_gate_tagged_inputs(protocol.frozen_tag)

    lock = json.loads((ROOT / "evolution/shinkaevolve.lock.json").read_text())
    sibling_state = _git_state(ROOT.parent / "ShinkaEvolve")
    if (
        sibling_state["head"] != lock["fork_commit"]
        or sibling_state["tree"] != lock["fork_tree"]
        or sibling_state["dirty_entries"]
    ):
        raise RuntimeError("G-ablate requires the clean pinned ShinkaEvolve fork")

    verify_weight_directory(track, weight_directory)
    handshake = client.fetch_handshake()
    verify_host_handshake(track, handshake)
    handshake_sha256 = host_handshake_sha256(handshake)
    if handshake_sha256 != protocol.proposal_host_handshake_sha256:
        raise ValueError("live proposal handshake differs from the P1 protocol")
    return _AcceptanceSnapshot(
        files={name: _file_sha256(path) for name, path in paths.items()},
        repository=root_state,
        shinkaevolve_repository=sibling_state,
        host_handshake=handshake,
        host_handshake_sha256=handshake_sha256,
    )


def _run_acceptance_gate_in_process(
    *,
    archive_root: str | Path,
    proposal_endpoint: str,
    weight_directory: str | Path,
) -> dict[str, Any]:
    """Fresh-worker implementation; not an injectable public acceptance API."""
    archive = _require_external_archive_path(
        archive_root,
        label="G-ablate acceptance archive",
    )
    protocol_path = ROOT / "evolution/g_ablate.json"
    protocol = load_protocol(protocol_path)
    if not protocol.acceptance_ready or protocol.frozen_tag is None:
        raise RuntimeError("G-ablate acceptance cannot run before its P1 freeze")
    endpoint_origin = _endpoint_origin(proposal_endpoint)
    if endpoint_origin != protocol.proposal_endpoint_origin:
        raise ValueError("proposal endpoint origin differs from the P1 protocol")

    task_path = ROOT / "evolution/task/config.json"
    plan_path = ROOT / "evolution/campaigns.json"
    model_path = ROOT / "evolution/open_model.json"
    initial_path = ROOT / "evolution/task/initial.py"
    task = load_task_evaluation_config(task_path)
    ladder = load_ladder_config(task_path)
    islands = load_islands_config(task_path)
    plan = load_campaign_plan(plan_path)
    feedback = load_feedback_profiles(task_path)[protocol.feedback_profile]
    track = load_model_track(model_path)
    if not ladder.confirmatory_ready or islands.latency_calibration.status != "p1-frozen":
        raise RuntimeError("G-ablate requires the P1-frozen ladder and latency calibration")

    client = VLLMCompletionClient(
        proposal_endpoint,
        max_request_cost_microusd=track.max_call_reservation_microusd,
        revision=track.revision,
        weight_set_sha256=track.weight_set_sha256,
    )
    preflight = _capture_acceptance_snapshot(
        protocol=protocol,
        track=track,
        client=client,
        weight_directory=weight_directory,
    )
    client.bind_handshake(preflight.host_handshake)

    provenance = {
        "files": preflight.files,
        "model": {
            "model_id": track.model_id,
            "revision": track.revision,
            "weight_set_sha256": track.weight_set_sha256,
            "host_handshake_sha256": preflight.host_handshake_sha256,
            "weights_verified_locally": True,
        },
        "proposal_endpoint_origin": endpoint_origin,
        "repository": preflight.repository,
        "shinkaevolve_repository": preflight.shinkaevolve_repository,
        "development_override": False,
    }

    def llm_factory() -> ProposalOperator:
        return OpenWeightLLMOperator(
            track,
            client,
            ApiCostLedger(track.max_api_costs_microusd),
            proposal_modes=plan.llm_proposal_modes,
        )

    comparison = _compare_archives(
        protocol=protocol,
        campaign_plan_path=plan_path,
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        initial_source=initial_path.read_text(),
        operator_factories={"llm": llm_factory, "gp": PlacementScriptGPOperator},
        archive_root=archive,
        provenance=provenance,
    )
    postflight = _capture_acceptance_snapshot(
        protocol=protocol,
        track=track,
        client=client,
        weight_directory=weight_directory,
    )
    if postflight != preflight:
        raise RuntimeError("G-ablate frozen acceptance dependencies drifted during execution")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if comparison.passed else "FAIL",
        "acceptance_evidence": True,
        "protocol": dataclasses.asdict(protocol),
        "campaign_root_row": dataclasses.asdict(comparison.definition),
        "provenance": provenance,
        "preflight": preflight.payload(),
        "postflight": postflight.payload(),
        "arms": comparison.arm_reports,
    }
    _write_report(comparison.root, report)
    return report


def run_acceptance_gate(
    *,
    archive_root: str | Path,
    proposal_endpoint: str,
    weight_directory: str | Path,
) -> dict[str, Any]:
    """Run production acceptance in a fresh isolated interpreter."""
    archive = _require_external_archive_path(
        archive_root,
        label="G-ablate acceptance archive",
    )
    command = [
        sys.executable,
        "-I",
        "-m",
        "evolution.g_ablate",
        "--acceptance-worker",
        "--archive_root",
        str(archive),
        "--proposal_endpoint",
        proposal_endpoint,
        "--weight_directory",
        str(Path(weight_directory).resolve()),
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((archive / "report.json").read_text())
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") not in {"PASS", "FAIL"}
        or not report.get("acceptance_evidence")
    ):
        raise RuntimeError("fresh acceptance worker produced an invalid report")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "evolution/g_ablate.json")
    parser.add_argument("--campaign_config", type=Path, default=ROOT / "evolution/campaigns.json")
    parser.add_argument("--task_config", type=Path, default=ROOT / "evolution/task/config.json")
    parser.add_argument("--model_manifest", type=Path, default=ROOT / "evolution/open_model.json")
    parser.add_argument("--initial", type=Path, default=ROOT / "evolution/task/initial.py")
    parser.add_argument("--proposal_endpoint")
    parser.add_argument("--host_handshake", type=Path)
    parser.add_argument("--weight_directory", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--acceptance-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    endpoint = args.proposal_endpoint or os.environ.get("EVOLVED_CONTINUUM_PROPOSAL_ENDPOINT")
    if not endpoint:
        raise ValueError("G-ablate requires the pinned proposal endpoint")

    if args.acceptance_worker:
        if args.development or args.weight_directory is None:
            raise ValueError("acceptance worker requires production weight verification")
        report = _run_acceptance_gate_in_process(
            archive_root=args.archive_root,
            proposal_endpoint=endpoint,
            weight_directory=args.weight_directory,
        )
        print(_canonical({"status": report["status"]}))
        # The parent process validates the report and maps FAIL to its public exit status.
        return 0

    if not args.development:
        supplied = {
            "protocol": args.protocol.resolve(),
            "campaign_config": args.campaign_config.resolve(),
            "task_config": args.task_config.resolve(),
            "model_manifest": args.model_manifest.resolve(),
            "initial": args.initial.resolve(),
        }
        expected = {
            "protocol": (ROOT / "evolution/g_ablate.json").resolve(),
            "campaign_config": CANONICAL_INPUTS["campaign_config"].resolve(),
            "task_config": CANONICAL_INPUTS["task_config"].resolve(),
            "model_manifest": CANONICAL_INPUTS["model_manifest"].resolve(),
            "initial": CANONICAL_INPUTS["initial"].resolve(),
        }
        if supplied != expected:
            raise ValueError("acceptance CLI permits only canonical repository inputs")
        if args.host_handshake is not None:
            raise ValueError("acceptance reads the handshake only from the live endpoint")
        if args.weight_directory is None:
            raise ValueError("G-ablate acceptance requires --weight_directory verification")
        report = run_acceptance_gate(
            archive_root=args.archive_root,
            proposal_endpoint=endpoint,
            weight_directory=args.weight_directory,
        )
        print(
            _canonical(
                {"report": str(args.archive_root / "report.json"), "status": report["status"]}
            )
        )
        return 0 if report["status"] == "PASS" else 1

    protocol = load_protocol(args.protocol)
    if protocol.acceptance_ready:
        raise ValueError("do not use a development override with the frozen P1 gate")
    task = load_task_evaluation_config(args.task_config)
    ladder = load_ladder_config(args.task_config)
    islands = load_islands_config(args.task_config)
    feedback = load_feedback_profiles(args.task_config)[protocol.feedback_profile]
    plan = load_campaign_plan(args.campaign_config)
    track = load_model_track(args.model_manifest)
    if args.weight_directory is not None:
        verify_weight_directory(track, args.weight_directory)
    endpoint_origin = _endpoint_origin(endpoint)
    client = VLLMCompletionClient(
        endpoint,
        max_request_cost_microusd=track.max_call_reservation_microusd,
        revision=track.revision,
        weight_set_sha256=track.weight_set_sha256,
    )
    handshake = client.fetch_handshake()
    verify_host_handshake(track, handshake)
    client.bind_handshake(handshake)
    if args.host_handshake is not None and json.loads(
        args.host_handshake.read_text()
    ) != handshake:
        raise ValueError("local host receipt differs from the live proposal endpoint")
    root_state = _git_state(ROOT)
    sibling_state = _git_state(ROOT.parent / "ShinkaEvolve")
    lock_path = ROOT / "evolution/shinkaevolve.lock.json"
    paths = {
        "protocol": args.protocol.resolve(),
        "campaign_config": args.campaign_config.resolve(),
        "task_config": args.task_config.resolve(),
        "model_manifest": args.model_manifest.resolve(),
        "initial": args.initial.resolve(),
        "shinkaevolve_lock": lock_path.resolve(),
        "gate_implementation": (ROOT / "evolution/g_ablate.py").resolve(),
        "campaign_implementation": (ROOT / "evolution/campaign.py").resolve(),
        "llm_operator_implementation": (ROOT / "evolution/llm_operator.py").resolve(),
        "gp_operator_implementation": (ROOT / "evolution/gp_operator.py").resolve(),
    }
    if args.host_handshake is not None:
        paths["host_handshake_receipt"] = args.host_handshake.resolve()
    provenance = {
        "files": {name: _file_sha256(path) for name, path in paths.items()},
        "model": {
            "model_id": track.model_id,
            "revision": track.revision,
            "weight_set_sha256": track.weight_set_sha256,
            "host_handshake_sha256": host_handshake_sha256(handshake),
            "weights_verified_locally": args.weight_directory is not None,
        },
        "proposal_endpoint_origin": endpoint_origin,
        "repository": root_state,
        "shinkaevolve_repository": sibling_state,
        "development_override": True,
    }

    def llm_factory() -> ProposalOperator:
        return OpenWeightLLMOperator(
            track,
            client,
            ApiCostLedger(track.max_api_costs_microusd),
            proposal_modes=plan.llm_proposal_modes,
        )

    report = run_gate(
        protocol=protocol,
        campaign_plan_path=args.campaign_config,
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        initial_source=args.initial.read_text(),
        operator_factories={"llm": llm_factory, "gp": PlacementScriptGPOperator},
        archive_root=args.archive_root,
        provenance=provenance,
    )
    print(
        _canonical({"report": str(args.archive_root / "report.json"), "status": report["status"]})
    )
    return 0 if report["status"] == "DEVELOPMENT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
