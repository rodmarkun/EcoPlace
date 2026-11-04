"""Trigger-matched, per-trigger-budgeted online GPHH comparator (TODO 7.6).

The trigger schedule is a first-class sealed artifact: both the continuous LLM arm and
this comparator must consume the same schedule digest.  A single work-budget object is
shared by the named arms, so equality cannot drift through duplicated configuration.
Each online GPHH search warm-starts with the previously deployed tree and discards
unused per-trigger work; work is never pooled across triggers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import placementscript as ps
from deap import gp

from analysis.tracking import derive_seed
from baselines.adapter import PolicyBundleAdapter
from baselines.gphh import (
    ClassicGPHH,
    EvaluateGPHH,
    GPHHConfig,
    GPHHRun,
    gphh_bundle,
    primitive_set,
)
from evolution.campaign import CampaignRunner, EvaluatorWorkBudget
from evolution.evaluation import TaskEvaluationConfig
from evolution.ladder import LadderConfig


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Thaw frozen domain mappings without deepcopying mappingproxy values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _task_payload(task: TaskEvaluationConfig, *, include_seeds: bool) -> dict[str, Any]:
    scenarios = []
    for scenario in task.scenarios:
        path = scenario.path.resolve()
        item: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "path_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if include_seeds:
            item["seeds"] = list(scenario.seeds)
        scenarios.append(item)
    return {
        "scenarios": scenarios,
        "bundle_ids": list(task.bundle_ids),
        "constraints": _jsonable(task.constraints),
        "weights": _jsonable(task.weights),
        "bounds": _jsonable(task.bounds),
    }


@dataclass(frozen=True)
class DriftTriggerConfig:
    registration_status: str
    registration_source: str
    residual_source: str
    energy_allowance: float
    energy_threshold: float
    violation_allowance: float
    violation_threshold: float
    min_observations: int
    cooldown_steps: int

    def __post_init__(self) -> None:
        if self.registration_status not in {"development-provisional", "p2-frozen"}:
            raise ValueError("unknown trigger registration status")
        if not self.registration_source:
            raise ValueError("trigger registration_source is required")
        if self.residual_source != "episode-exogenous-reference-stream":
            raise ValueError("shared triggers require the episode-exogenous-reference-stream")
        for name in (
            "energy_allowance",
            "energy_threshold",
            "violation_allowance",
            "violation_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.energy_threshold <= 0 or self.violation_threshold <= 0:
            raise ValueError("CUSUM thresholds must be positive")
        for name in ("min_observations", "cooldown_steps"):
            value = getattr(self, name)
            minimum = 1 if name == "min_observations" else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}")

    @property
    def confirmatory_ready(self) -> bool:
        return self.registration_status == "p2-frozen"

    @property
    def sha256(self) -> str:
        return _sha(dataclasses.asdict(self))


@dataclass(frozen=True)
class TriggerObservation:
    step: int
    energy_residual: float
    violation_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("trigger observation step must be a non-negative int")
        for name in ("energy_residual", "violation_residual"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TriggerRecord:
    observation: TriggerObservation
    triggered: bool
    trigger_index: int | None
    energy_cusum: float
    violation_cusum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.observation.step,
            "energy_residual": self.observation.energy_residual,
            "violation_residual": self.observation.violation_residual,
            "triggered": self.triggered,
            "trigger_index": self.trigger_index,
            "energy_cusum": self.energy_cusum,
            "violation_cusum": self.violation_cusum,
        }


class CUSUMDriftDetector:
    """Positive-degradation CUSUM with warm-up, reset, and step cooldown."""

    def __init__(self, config: DriftTriggerConfig) -> None:
        self.config = config
        self.energy_cusum = 0.0
        self.violation_cusum = 0.0
        self.observations = 0
        self.last_step: int | None = None
        self.last_trigger_step: int | None = None
        self.trigger_count = 0

    def observe(self, observation: TriggerObservation) -> TriggerRecord:
        if self.last_step is not None and observation.step <= self.last_step:
            raise ValueError("trigger observation steps must be strictly increasing")
        self.last_step = observation.step
        self.observations += 1
        cooling_down = (
            self.last_trigger_step is not None
            and observation.step - self.last_trigger_step <= self.config.cooldown_steps
        )
        if cooling_down:
            self.energy_cusum = 0.0
            self.violation_cusum = 0.0
        else:
            self.energy_cusum = max(
                0.0,
                self.energy_cusum + observation.energy_residual - self.config.energy_allowance,
            )
            self.violation_cusum = max(
                0.0,
                self.violation_cusum
                + observation.violation_residual
                - self.config.violation_allowance,
            )
        triggered = (
            not cooling_down
            and self.observations >= self.config.min_observations
            and (
                self.energy_cusum >= self.config.energy_threshold
                or self.violation_cusum >= self.config.violation_threshold
            )
        )
        trigger_index = None
        captured_energy = self.energy_cusum
        captured_violation = self.violation_cusum
        if triggered:
            trigger_index = self.trigger_count
            self.trigger_count += 1
            self.last_trigger_step = observation.step
            self.energy_cusum = 0.0
            self.violation_cusum = 0.0
        return TriggerRecord(
            observation=observation,
            triggered=triggered,
            trigger_index=trigger_index,
            energy_cusum=captured_energy,
            violation_cusum=captured_violation,
        )


@dataclass(frozen=True)
class TriggerSchedule:
    episode_sha256: str
    config: DriftTriggerConfig
    observations: tuple[TriggerObservation, ...]
    records: tuple[TriggerRecord, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_sha256, str)
            or len(self.episode_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.episode_sha256)
        ):
            raise ValueError("trigger stream episode_sha256 must be a lowercase SHA-256")
        detector = CUSUMDriftDetector(self.config)
        # Records are always replayed from the sealed exogenous observations; callers
        # cannot inject a hand-authored trigger list.
        object.__setattr__(
            self,
            "records",
            tuple(detector.observe(observation) for observation in self.observations),
        )

    @staticmethod
    def build(
        config: DriftTriggerConfig,
        episode_sha256: str,
        observations: tuple[TriggerObservation, ...],
    ) -> TriggerSchedule:
        return TriggerSchedule(episode_sha256, config, observations)

    @property
    def trigger_config_sha256(self) -> str:
        return self.config.sha256

    @property
    def triggers(self) -> tuple[TriggerRecord, ...]:
        return tuple(record for record in self.records if record.triggered)

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict(include_sha=False))

    def to_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-trigger-schedule-v2",
            "episode_sha256": self.episode_sha256,
            "trigger_config_sha256": self.trigger_config_sha256,
            "observations": [dataclasses.asdict(observation) for observation in self.observations],
            "records": [record.to_dict() for record in self.records],
        }
        return {**payload, "sha256": _sha(payload)} if include_sha else payload


@dataclass(frozen=True)
class OnlineAdaptationPlan:
    status: str
    registration_source: str
    arms: tuple[str, ...]
    trigger: DriftTriggerConfig
    per_trigger_work: EvaluatorWorkBudget

    def __post_init__(self) -> None:
        if self.status not in {"development-provisional", "p2-frozen"}:
            raise ValueError("unknown online-adaptation plan status")
        if not self.registration_source:
            raise ValueError("online-adaptation registration_source is required")
        if self.arms != ("llm_continuous", "online_gphh"):
            raise ValueError("online adaptation must name the matched comparator arms")
        expected_status = "p2-frozen" if self.status == "p2-frozen" else "development-provisional"
        if (
            self.trigger.registration_status != expected_status
            or self.per_trigger_work.registration_status != expected_status
        ):
            raise ValueError("plan, trigger, and work registration statuses must agree")

    @property
    def confirmatory_ready(self) -> bool:
        return self.status == "p2-frozen"

    @property
    def work_sha256(self) -> str:
        return _sha(dataclasses.asdict(self.per_trigger_work))

    @property
    def sha256(self) -> str:
        return _sha(
            {
                "status": self.status,
                "registration_source": self.registration_source,
                "arms": self.arms,
                "trigger": dataclasses.asdict(self.trigger),
                "per_trigger_work": dataclasses.asdict(self.per_trigger_work),
            }
        )


def load_online_adaptation_plan(path: str | Path) -> OnlineAdaptationPlan:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != "evolved-continuum-online-adaptation-v1":
        raise ValueError("unknown online-adaptation schema")
    trigger = DriftTriggerConfig(**raw["trigger"])
    work_raw = raw["per_trigger_evaluator_work"]
    work = EvaluatorWorkBudget(
        registration_status=work_raw["registration_status"],
        registration_source=work_raw["registration_source"],
        max_microwork=work_raw["max_microwork"],
        rung_microwork=work_raw["rung_microwork"],
        checkpoints_microwork=tuple(work_raw["checkpoints_microwork"]),
    )
    return OnlineAdaptationPlan(
        status=raw["status"],
        registration_source=raw["registration_source"],
        arms=tuple(raw["arms"]),
        trigger=trigger,
        per_trigger_work=work,
    )


@dataclass(frozen=True)
class OnlineGPHHEvent:
    trigger_index: int
    trigger_step: int
    search_seed: int
    incumbent_before_sha256: str
    incumbent_after_sha256: str
    actual_microwork: int
    unused_microwork_discarded: int
    search: GPHHRun
    task_window_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_index": self.trigger_index,
            "trigger_step": self.trigger_step,
            "search_seed": self.search_seed,
            "incumbent_before_sha256": self.incumbent_before_sha256,
            "incumbent_after_sha256": self.incumbent_after_sha256,
            "actual_microwork": self.actual_microwork,
            "unused_microwork_discarded": self.unused_microwork_discarded,
            "budget_carry_microwork": 0,
            "search": self.search.to_dict(),
            "task_window_sha256": self.task_window_sha256,
        }


@dataclass(frozen=True)
class OnlineGPHHRun:
    root_seed: int
    plan_sha256: str
    trigger_schedule_sha256: str
    trigger_config_sha256: str
    per_trigger_work_sha256: str
    initial_tree: str
    final_tree: gp.PrimitiveTree
    events: tuple[OnlineGPHHEvent, ...]

    @property
    def bundle(self) -> PolicyBundleAdapter:
        return gphh_bundle(self.final_tree)

    def to_dict(self) -> dict[str, Any]:
        final = str(self.final_tree)
        payload = {
            "schema": "evolved-continuum-online-gphh-run-v2",
            "root_seed": self.root_seed,
            "plan_sha256": self.plan_sha256,
            "trigger_schedule_sha256": self.trigger_schedule_sha256,
            "trigger_config_sha256": self.trigger_config_sha256,
            "per_trigger_work_sha256": self.per_trigger_work_sha256,
            "initial_tree": self.initial_tree,
            "initial_tree_sha256": hashlib.sha256(self.initial_tree.encode()).hexdigest(),
            "final_tree": final,
            "final_tree_sha256": hashlib.sha256(final.encode()).hexdigest(),
            "trigger_count": len(self.events),
            "total_actual_microwork": sum(event.actual_microwork for event in self.events),
            "total_budget_carry_microwork": 0,
            "events": [event.to_dict() for event in self.events],
        }
        return {**payload, "sha256": _sha(payload)}


@dataclass(frozen=True)
class OnlineTaskWindow:
    """One episode-bound, strictly pre-trigger evaluation window."""

    task: TaskEvaluationConfig
    episode_sha256: str
    data_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_sha256, str)
            or len(self.episode_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.episode_sha256
            )
        ):
            raise ValueError("episode_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.task, TaskEvaluationConfig):
            raise TypeError("online task window needs a TaskEvaluationConfig")
        if (
            not self.data_steps
            or any(
                not isinstance(step, int) or isinstance(step, bool) or step < 0
                for step in self.data_steps
            )
            or tuple(sorted(set(self.data_steps))) != self.data_steps
        ):
            raise ValueError("online task data_steps must be non-empty, unique, and sorted")

    @property
    def data_start_step(self) -> int:
        return self.data_steps[0]

    @property
    def data_end_step(self) -> int:
        return self.data_steps[-1]

    @property
    def task_sha256(self) -> str:
        return _sha(_task_payload(self.task, include_seeds=True))

    @property
    def task_contract_sha256(self) -> str:
        return _sha(_task_payload(self.task, include_seeds=False))

    @property
    def artifact_sha256(self) -> str:
        return _sha(
            {
                "task_sha256": self.task_sha256,
                "episode_sha256": self.episode_sha256,
                "step_unit": "placement-decision",
                "data_steps": self.data_steps,
            }
        )

    @property
    def sha256(self) -> str:
        return _sha(
            {
                "episode_sha256": self.episode_sha256,
                "step_unit": "placement-decision",
                "data_steps": self.data_steps,
                "task_sha256": self.task_sha256,
                "task_contract_sha256": self.task_contract_sha256,
                "artifact_sha256": self.artifact_sha256,
            }
        )


TaskForTrigger = Callable[[TriggerRecord], OnlineTaskWindow]


class OnlineGPHHComparator:
    def __init__(
        self,
        plan: OnlineAdaptationPlan,
        initial_tree: gp.PrimitiveTree,
        *,
        root_seed: int,
        gphh_config: GPHHConfig | None = None,
        evaluate_fn: EvaluateGPHH | None = None,
    ) -> None:
        if (
            not isinstance(root_seed, int)
            or isinstance(root_seed, bool)
            or not 0 <= root_seed < 2**63
        ):
            raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
        self.plan = plan
        self.root_seed = root_seed
        self.pset = primitive_set()
        self.initial_tree = gp.PrimitiveTree(initial_tree)
        gp.compile(self.initial_tree, self.pset)
        kwargs = {} if evaluate_fn is None else {"evaluate_fn": evaluate_fn}
        self.search = ClassicGPHH(gphh_config, **kwargs)

    def run(
        self,
        schedule: TriggerSchedule,
        task_for_trigger: TaskForTrigger,
        ladder: LadderConfig,
    ) -> OnlineGPHHRun:
        if schedule.trigger_config_sha256 != self.plan.trigger.sha256:
            raise ValueError("trigger schedule does not match the shared trigger plan")
        current = gp.PrimitiveTree(self.initial_tree)
        events = []
        for record in schedule.triggers:
            assert record.trigger_index is not None
            seed = derive_seed(
                self.root_seed,
                "online-gphh",
                schedule.sha256,
                record.trigger_index,
            )
            before = str(current)
            window = task_for_trigger(record)
            if window.episode_sha256 != schedule.episode_sha256:
                raise ValueError("online task window belongs to a different episode")
            if window.data_end_step >= record.observation.step:
                raise ValueError("online task window contains trigger/future data")
            result = self.search.run(
                window.task,
                ladder,
                self.plan.per_trigger_work,
                root_seed=seed,
                initial_trees=(current,),
            )
            if result.spent_microwork > self.plan.per_trigger_work.max_microwork:
                raise ValueError("online GPHH exceeded the per-trigger work budget")
            if result.champion is not None:
                current = gp.PrimitiveTree(result.champion.tree)
            after = str(current)
            events.append(
                OnlineGPHHEvent(
                    trigger_index=record.trigger_index,
                    trigger_step=record.observation.step,
                    search_seed=seed,
                    incumbent_before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                    incumbent_after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                    actual_microwork=result.spent_microwork,
                    unused_microwork_discarded=(
                        self.plan.per_trigger_work.max_microwork - result.spent_microwork
                    ),
                    search=result,
                    task_window_sha256=window.sha256,
                )
            )
        return OnlineGPHHRun(
            root_seed=self.root_seed,
            plan_sha256=self.plan.sha256,
            trigger_schedule_sha256=schedule.sha256,
            trigger_config_sha256=self.plan.trigger.sha256,
            per_trigger_work_sha256=self.plan.work_sha256,
            initial_tree=str(self.initial_tree),
            final_tree=current,
            events=tuple(events),
        )


@dataclass(frozen=True)
class ContinuousLLMResult:
    archive_sha256: str
    artifact_sha256: str
    actual_microwork: int
    unused_microwork_discarded: int
    incumbent_candidate_id: str | None
    promoted: bool
    deployment_status: str
    confirmatory_usable: bool
    source: str = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for digest in (self.archive_sha256, self.artifact_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or digest == "0" * 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("continuous LLM result digests must be non-placeholder SHA-256")
        if (
            not isinstance(self.actual_microwork, int)
            or isinstance(self.actual_microwork, bool)
            or self.actual_microwork <= 0
            or not isinstance(self.unused_microwork_discarded, int)
            or isinstance(self.unused_microwork_discarded, bool)
            or self.unused_microwork_discarded < 0
        ):
            raise ValueError("continuous LLM verified result fields are invalid")
        if self.promoted and self.incumbent_candidate_id is None:
            raise ValueError("continuous LLM promotion lacks its campaign candidate id")
        if self.deployment_status not in {
            "promoted",
            "incumbent-retained-no-better-candidate",
            "incumbent-retained-no-feasible-candidate",
        } or self.promoted != (self.deployment_status == "promoted"):
            raise ValueError("continuous LLM deployment status is inconsistent")
        if not isinstance(self.confirmatory_usable, bool):
            raise ValueError("continuous LLM confirmatory status must be boolean")
        ps.Policy(self.source)
        if hashlib.sha256(self.source.encode()).hexdigest() != self.artifact_sha256:
            raise ValueError("continuous LLM source differs from its artifact digest")

    @property
    def result_sha256(self) -> str:
        return _sha(
            {
                "artifact_sha256": self.artifact_sha256,
                "archive_sha256": self.archive_sha256,
                "actual_microwork": self.actual_microwork,
                "unused_microwork_discarded": self.unused_microwork_discarded,
                "incumbent_candidate_id": self.incumbent_candidate_id,
                "promoted": self.promoted,
                "deployment_status": self.deployment_status,
                "confirmatory_usable": self.confirmatory_usable,
            }
        )


class ContinuousLLMCampaignFactory(Protocol):
    def __call__(
        self,
        record: TriggerRecord,
        window: OnlineTaskWindow,
        work: EvaluatorWorkBudget,
        pairing_provenance: dict[str, Any],
        incumbent_source: str,
        ladder: LadderConfig,
    ) -> CampaignRunner: ...


@dataclass(frozen=True)
class PairedOnlineEvent:
    trigger_index: int
    trigger_step: int
    task_window_sha256: str
    schedule_sha256: str
    work_sha256: str
    llm_actual_microwork: int
    llm_unused_microwork_discarded: int
    llm_result_sha256: str
    llm_incumbent_before_sha256: str
    llm_incumbent_after_sha256: str
    llm_promoted: bool
    llm_deployment_status: str
    llm_confirmatory_usable: bool
    gphh_actual_microwork: int
    gphh_unused_microwork_discarded: int
    gphh_incumbent_after_sha256: str


@dataclass(frozen=True)
class PairedOnlineRun:
    plan_sha256: str
    schedule_sha256: str
    per_trigger_work_sha256: str
    gphh: OnlineGPHHRun
    events: tuple[PairedOnlineEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-paired-online-run-v1",
            "plan_sha256": self.plan_sha256,
            "schedule_sha256": self.schedule_sha256,
            "per_trigger_work_sha256": self.per_trigger_work_sha256,
            "arms": ["llm_continuous", "online_gphh"],
            "budget_carry_microwork": 0,
            "events": [dataclasses.asdict(event) for event in self.events],
            "gphh": self.gphh.to_dict(),
        }
        return {**payload, "sha256": _sha(payload)}


class PairedOnlineController:
    """Dispatch the exact same trigger, task window, and budget object to both arms."""

    def __init__(
        self,
        plan: OnlineAdaptationPlan,
        gphh: OnlineGPHHComparator,
        llm_campaign_factory: ContinuousLLMCampaignFactory,
        *,
        initial_llm_source: str,
    ) -> None:
        if gphh.plan is not plan:
            raise ValueError("paired controller and GPHH must share one plan object")
        self.plan = plan
        self.gphh = gphh
        self.llm_campaign_factory = llm_campaign_factory
        ps.Policy(initial_llm_source)
        self.initial_llm_source = initial_llm_source
        if self.plan.confirmatory_ready:
            raise ValueError(
                "this in-process online adapter is non-confirmatory; the Phase-9 E3 driver "
                "must own P2 authorization and dependency postflight evidence"
            )

    def _run_llm_campaign(
        self,
        record: TriggerRecord,
        window: OnlineTaskWindow,
        schedule: TriggerSchedule,
        incumbent_source: str,
        ladder: LadderConfig,
    ) -> ContinuousLLMResult:
        assert record.trigger_index is not None
        pairing = {
            "schema": "evolved-continuum-online-pairing-v1",
            "schedule_sha256": schedule.sha256,
            "trigger_index": record.trigger_index,
            "trigger_step": record.observation.step,
            "task_window_sha256": window.sha256,
            "task_artifact_sha256": window.artifact_sha256,
            "episode_sha256": schedule.episode_sha256,
            "work_sha256": self.plan.work_sha256,
            "incumbent_before_sha256": hashlib.sha256(incumbent_source.encode()).hexdigest(),
        }
        runner = self.llm_campaign_factory(
            record,
            window,
            self.plan.per_trigger_work,
            pairing,
            incumbent_source,
            ladder,
        )
        if not isinstance(runner, CampaignRunner):
            raise TypeError("continuous LLM factory must return a CampaignRunner")
        if (
            runner.arm != "llm"
            or runner.run_purpose != "scientific-full-work"
            or runner.plan.evaluator_work is not self.plan.per_trigger_work
            or runner.evaluation_seed_mode != "presealed-task"
            or runner.terminal_budget_policy != "discard-at-trigger"
        ):
            raise ValueError(
                "continuous LLM runner changed its arm, purpose, work, seeds, or carry policy"
            )
        if _sha(_task_payload(runner.task, include_seeds=True)) != window.task_sha256:
            raise ValueError("continuous LLM runner changed the sealed task, including seeds")
        if runner.ladder is not ladder:
            raise ValueError("continuous LLM runner changed the shared evaluation ladder")
        manifest = json.loads(runner.archive.manifest_path.read_text())
        if manifest.get("provenance", {}).get("online_pairing") != pairing:
            raise ValueError("continuous LLM campaign manifest lacks exact pairing provenance")
        incumbent_before_sha256 = hashlib.sha256(incumbent_source.encode()).hexdigest()
        if manifest.get("initial_source_sha256") != incumbent_before_sha256:
            raise ValueError("continuous LLM campaign did not warm-start from its incumbent")
        if runner.plan.confirmatory_ready:
            raise ValueError("provisional online adaptation cannot consume a P2 campaign")
        archive_sha256 = runner.run()
        runner.archive.validate_semantics()
        runner.archive.materialize_candidates()
        runner._verify_work_ledger()
        if archive_sha256 != runner.archive.digest():
            raise ValueError("continuous LLM campaign digest changed after verification")
        terminal = runner._terminal()
        if (
            terminal is None
            or terminal.get("status") != "complete"
            or terminal.get("reason") != "normalized-work-budget-exhausted"
        ):
            raise ValueError("continuous LLM campaign did not complete on its work budget")
        remaining = self.plan.per_trigger_work.max_microwork - runner._spent_microwork()
        if (
            terminal.get("budget_carry_microwork") != 0
            or terminal.get("unused_microwork_discarded") != remaining
            or terminal.get("terminal_budget_policy") != "discard-at-trigger"
        ):
            raise ValueError("continuous LLM campaign did not discard its trigger-local remainder")
        actual = runner._spent_microwork()
        if actual <= 0 or actual > self.plan.per_trigger_work.max_microwork:
            raise ValueError("continuous LLM campaign has invalid verified work")
        incumbent_id = terminal.get("carried_incumbent_id")
        completed = runner._by_id()
        if incumbent_id is not None and (
            not isinstance(incumbent_id, str) or incumbent_id not in completed
        ):
            raise ValueError("continuous LLM campaign has an invalid incumbent id")
        incumbent = completed.get(incumbent_id) if isinstance(incumbent_id, str) else None
        evaluation = (
            None if incumbent is None else incumbent.get("evaluation", {}).get("evaluation")
        )
        feasible_incumbent = bool(evaluation and evaluation["fitness"]["feasible"] is True)
        source = incumbent_source
        if feasible_incumbent:
            assert isinstance(incumbent_id, str) and incumbent is not None
            candidate_source = incumbent.get("proposal", {}).get("source")
            if not isinstance(candidate_source, str) or not candidate_source:
                raise ValueError("continuous LLM campaign incumbent lacks executable source")
            source = candidate_source
            artifact_sha256 = hashlib.sha256(source.encode()).hexdigest()
            candidate_path = runner.archive.candidate_dir / f"{incumbent['proposal_seq']:08d}.ps"
            if (
                not candidate_path.is_file()
                or hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                != artifact_sha256
            ):
                raise ValueError("continuous LLM incumbent differs from the archive artifact")
            promoted = artifact_sha256 != incumbent_before_sha256
            if not promoted:
                source = incumbent_source
                artifact_sha256 = incumbent_before_sha256
                deployment_status = "incumbent-retained-no-better-candidate"
            else:
                deployment_status = "promoted"
        else:
            promoted = False
            artifact_sha256 = incumbent_before_sha256
            deployment_status = "incumbent-retained-no-feasible-candidate"
        return ContinuousLLMResult(
            archive_sha256,
            artifact_sha256,
            actual,
            remaining,
            incumbent_id,
            promoted,
            deployment_status,
            False,
            source,
        )

    def run(
        self,
        schedule: TriggerSchedule,
        task_for_trigger: TaskForTrigger,
        ladder: LadderConfig,
    ) -> PairedOnlineRun:
        windows: dict[int, OnlineTaskWindow] = {}
        llm_results: dict[int, ContinuousLLMResult] = {}
        llm_before: dict[int, str] = {}
        current_llm_source = self.initial_llm_source
        for record in schedule.triggers:
            assert record.trigger_index is not None
            window = task_for_trigger(record)
            if (
                window.episode_sha256 != schedule.episode_sha256
                or window.data_end_step >= record.observation.step
            ):
                raise ValueError("paired online window is not strictly pre-trigger in this episode")
            windows[record.trigger_index] = window
            llm_before[record.trigger_index] = hashlib.sha256(
                current_llm_source.encode()
            ).hexdigest()
            result = self._run_llm_campaign(
                record,
                window,
                schedule,
                current_llm_source,
                ladder,
            )
            llm_results[record.trigger_index] = result
            current_llm_source = result.source
        gphh_run = self.gphh.run(
            schedule,
            lambda record: windows[record.trigger_index],  # type: ignore[index]
            ladder,
        )
        if len(gphh_run.events) != len(schedule.triggers):
            raise ValueError("paired arms consumed different trigger counts")
        paired = []
        for record, gphh_event in zip(schedule.triggers, gphh_run.events, strict=True):
            assert record.trigger_index is not None
            window = windows[record.trigger_index]
            llm = llm_results[record.trigger_index]
            llm_work = llm.actual_microwork
            if (
                gphh_event.trigger_index != record.trigger_index
                or gphh_event.task_window_sha256 != window.sha256
            ):
                raise ValueError("paired arm receipts disagree on trigger or task window")
            paired.append(
                PairedOnlineEvent(
                    record.trigger_index,
                    record.observation.step,
                    window.sha256,
                    schedule.sha256,
                    self.plan.work_sha256,
                    llm_work,
                    llm.unused_microwork_discarded,
                    llm.result_sha256,
                    llm_before[record.trigger_index],
                    llm.artifact_sha256,
                    llm.promoted,
                    llm.deployment_status,
                    llm.confirmatory_usable,
                    gphh_event.actual_microwork,
                    gphh_event.unused_microwork_discarded,
                    gphh_event.incumbent_after_sha256,
                )
            )
        return PairedOnlineRun(
            self.plan.sha256,
            schedule.sha256,
            self.plan.work_sha256,
            gphh_run,
            tuple(paired),
        )


__all__ = [
    "CUSUMDriftDetector",
    "DriftTriggerConfig",
    "OnlineAdaptationPlan",
    "OnlineGPHHComparator",
    "OnlineGPHHEvent",
    "OnlineGPHHRun",
    "OnlineTaskWindow",
    "ContinuousLLMCampaignFactory",
    "ContinuousLLMResult",
    "PairedOnlineController",
    "PairedOnlineEvent",
    "PairedOnlineRun",
    "TriggerObservation",
    "TriggerRecord",
    "TriggerSchedule",
    "load_online_adaptation_plan",
]
