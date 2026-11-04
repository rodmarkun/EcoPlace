from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
from deap import gp
from shinka.core import ProposalOffspring

from baselines.gphh import GPHHConfig, primitive_set
from baselines.online_gphh import (
    DriftTriggerConfig,
    OnlineAdaptationPlan,
    OnlineGPHHComparator,
    OnlineTaskWindow,
    PairedOnlineController,
    TriggerObservation,
    TriggerSchedule,
    load_online_adaptation_plan,
)
from evolution.campaign import CampaignRunner, EvaluatorWorkBudget, load_campaign_plan
from evolution.evaluation import (
    PolicyEvaluation,
    load_task_evaluation_config,
    policy_identity_for_source,
)
from evolution.feedback import load_feedback_profiles
from evolution.islands import ArchiveAssessment, DecisionCost, load_islands_config
from evolution.ladder import LadderResult, load_ladder_config
from sim.energy import CandidateFitness

ROOT = Path(__file__).parents[1]
EPISODE = "a" * 64
TASK_CONFIG = ROOT / "evolution" / "task" / "config.json"


class _CloneOperator:
    name = "clone"

    def __call__(self, parent, inspirations, rng):
        del inspirations, rng
        return ProposalOffspring(source=parent.code, metadata={"operator": self.name})


class _MutateOperator:
    name = "mutate"

    def __call__(self, parent, inspirations, rng):
        del inspirations, rng
        source = parent.code.replace(
            "return free_fraction_after(pod, node)",
            "return -free_fraction_after(pod, node)",
        )
        return ProposalOffspring(source=source, metadata={"operator": self.name})


def _trigger() -> DriftTriggerConfig:
    return DriftTriggerConfig(
        registration_status="development-provisional",
        registration_source="test",
        residual_source="episode-exogenous-reference-stream",
        energy_allowance=0.0,
        energy_threshold=1.0,
        violation_allowance=0.0,
        violation_threshold=1.0,
        min_observations=2,
        cooldown_steps=1,
    )


def _schedule() -> TriggerSchedule:
    observations = tuple(
        TriggerObservation(step, energy, violation)
        for step, energy, violation in (
            (0, 0.6, 0.0),
            (1, 0.5, 0.0),
            (2, 4.0, 4.0),
            (3, 0.0, 0.6),
            (4, 0.0, 0.5),
        )
    )
    return TriggerSchedule.build(_trigger(), EPISODE, observations)


def _work() -> EvaluatorWorkBudget:
    return EvaluatorWorkBudget(
        registration_status="development-provisional",
        registration_source="test",
        max_microwork=12,
        rung_microwork={"L0": 1, "L1": 2, "L2": 3},
        checkpoints_microwork=(0, 12),
    )


def _plan() -> OnlineAdaptationPlan:
    return OnlineAdaptationPlan(
        status="development-provisional",
        registration_source="test",
        arms=("llm_continuous", "online_gphh"),
        trigger=_trigger(),
        per_trigger_work=_work(),
    )


def _window(record) -> OnlineTaskWindow:
    return OnlineTaskWindow(
        load_task_evaluation_config(TASK_CONFIG),
        EPISODE,
        tuple(range(record.observation.step)),
    )


def _rejected(bundle, task, ladder, proposal_seq):
    del task, ladder, proposal_seq
    return LadderResult(
        promoted_to_l2=False,
        evaluation=None,
        trace=({"rung": "L0"},),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=bundle.name,
        policy_version=bundle.version,
        policy_identity_sha256=bundle.identity_sha256,
        policy_artifact_sha256=bundle.artifact_sha256,
    )


def _llm_evaluate(source, task, ladder, proposal_seq, *, feasible=True):
    del task, ladder
    identity = policy_identity_for_source(source)
    fitness = CandidateFitness(
        feasible=feasible,
        infeasibility_vector=(0.0, 0.0, 0.0, 0.0) if feasible else (1.0, 0.0, 0.0, 0.0),
        summed_primitive_shortfall=0.0 if feasible else 1.0,
        normalized_energy=1.0 - proposal_seq / 10.0,
        proposal_seq=proposal_seq,
        n_instances=1,
        n_bundles=1,
    )
    evaluation = PolicyEvaluation(
        fitness=fitness,
        instances=(),
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )
    return LadderResult(
        promoted_to_l2=True,
        evaluation=evaluation,
        trace=({"rung": "L0"}, {"rung": "L1"}, {"rung": "L2"}),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name=identity["name"],
        policy_version=identity["version"],
        policy_identity_sha256=identity["identity_sha256"],
        policy_artifact_sha256=identity["artifact_sha256"],
    )


def _llm_assess(source, task, islands):
    del task, islands
    digest = hashlib.sha256(source.encode()).hexdigest()
    return ArchiveAssessment(
        loc=1,
        decision_cost=DecisionCost(
            source_sha256=digest,
            probe_sha256="b" * 64,
            score_batch_fuel=1,
            score_batch_memory_bytes=1,
            calibrated_latency_us=1.0,
            calibration_id="test",
            calibration_status="development-provisional",
        ),
        loc_penalty=0.0,
        decision_latency_penalty=0.0,
        total_penalty=0.0,
    )


def _campaign_factory(
    tmp_path,
    plan,
    seen,
    *,
    feasible=True,
    wrong_provenance=False,
    wrong_ladder=False,
):
    def factory(record, window, work, pairing, incumbent_source, ladder):
        seen.append((record.trigger_index, window, work, incumbent_source))
        assert work is plan.per_trigger_work
        campaign_plan = dataclasses.replace(
            load_campaign_plan(ROOT / "evolution" / "campaigns.json"),
            evaluator_work=work,
            candidate_budget=2,
        )
        feedback = load_feedback_profiles(TASK_CONFIG)[campaign_plan.feedback_profile]
        return CampaignRunner(
            definition=campaign_plan.campaigns[record.trigger_index],
            plan=campaign_plan,
            arm="llm",
            operator=_MutateOperator(),
            initial_source=incumbent_source,
            task=window.task,
            ladder=(
                dataclasses.replace(ladder, audit_root_seed=ladder.audit_root_seed + 1)
                if wrong_ladder
                else ladder
            ),
            islands=load_islands_config(TASK_CONFIG),
            feedback_profile=feedback,
            archive_dir=tmp_path / f"llm-{record.trigger_index}",
            provenance={"online_pairing": {} if wrong_provenance else pairing},
            run_purpose="scientific-full-work",
            evaluation_seed_mode="presealed-task",
            terminal_budget_policy="discard-at-trigger",
            evaluate_fn=lambda *args: _llm_evaluate(*args, feasible=feasible),
            assess_fn=_llm_assess,
        )

    return factory


def _comparator(plan: OnlineAdaptationPlan) -> OnlineGPHHComparator:
    tree = gp.PrimitiveTree.from_string("cpu_free_after", primitive_set())
    return OnlineGPHHComparator(
        plan,
        tree,
        root_seed=91,
        gphh_config=GPHHConfig(population_size=2, generations=1, elite_size=1),
        evaluate_fn=_rejected,
    )


def test_shared_cusum_schedule_is_replayed_and_episode_bound() -> None:
    first = _schedule()
    second = _schedule()
    assert first.to_dict() == second.to_dict()
    assert [record.observation.step for record in first.triggers] == [1, 4]
    assert [record.trigger_index for record in first.triggers] == [0, 1]
    assert first.episode_sha256 == EPISODE
    with pytest.raises(ValueError, match="init=False"):
        dataclasses.replace(first, records=())


def test_online_gphh_uses_pretrigger_windows_and_noncarried_budget() -> None:
    plan = _plan()
    comparator = _comparator(plan)
    first = comparator.run(_schedule(), _window, None)
    second = comparator.run(_schedule(), _window, None)
    assert first.to_dict() == second.to_dict()
    assert [event.actual_microwork for event in first.events] == [3, 3]
    assert [event.unused_microwork_discarded for event in first.events] == [9, 9]
    assert first.to_dict()["total_budget_carry_microwork"] == 0
    assert first.bundle.name == "classic_gphh"


def test_online_gphh_rejects_other_rules_episode_and_future_data() -> None:
    plan = _plan()
    comparator = _comparator(plan)
    other = dataclasses.replace(_trigger(), energy_threshold=2.0)
    schedule = TriggerSchedule.build(
        other,
        EPISODE,
        (TriggerObservation(0, 1.0, 0.0), TriggerObservation(1, 1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="shared trigger plan"):
        comparator.run(schedule, _window, None)
    with pytest.raises(ValueError, match="future"):
        comparator.run(
            _schedule(),
            lambda record: dataclasses.replace(
                _window(record),
                data_steps=tuple(range(record.observation.step + 1)),
            ),
            None,
        )


def test_paired_controller_dispatches_same_window_and_work_to_both_arms(tmp_path) -> None:
    plan = _plan()
    comparator = _comparator(plan)
    seen = []
    initial = (ROOT / "evolution" / "task" / "initial.py").read_text()
    ladder = load_ladder_config(TASK_CONFIG)
    run = PairedOnlineController(
        plan,
        comparator,
        _campaign_factory(tmp_path, plan, seen),
        initial_llm_source=initial,
    ).run(_schedule(), _window, ladder)
    assert len(run.events) == len(seen) == 2
    assert [event.task_window_sha256 for event in run.events] == [
        event.task_window_sha256 for event in run.gphh.events
    ]
    assert all(event.work_sha256 == plan.work_sha256 for event in run.events)
    assert all(event.llm_actual_microwork == 12 for event in run.events)
    assert all(event.llm_unused_microwork_discarded == 0 for event in run.events)
    assert run.events[0].llm_promoted
    assert not run.events[1].llm_promoted
    assert run.events[1].llm_deployment_status == "incumbent-retained-no-better-candidate"
    assert all(not event.llm_confirmatory_usable for event in run.events)
    assert run.events[1].llm_incumbent_before_sha256 == run.events[0].llm_incumbent_after_sha256
    assert run.to_dict()["budget_carry_microwork"] == 0


def test_paired_controller_requires_real_campaign_and_exact_pairing(tmp_path) -> None:
    plan = _plan()
    comparator = _comparator(plan)
    initial = (ROOT / "evolution" / "task" / "initial.py").read_text()
    controller = PairedOnlineController(
        plan,
        comparator,
        lambda *args: object(),
        initial_llm_source=initial,
    )
    ladder = load_ladder_config(TASK_CONFIG)
    with pytest.raises(TypeError, match="CampaignRunner"):
        controller.run(_schedule(), _window, ladder)

    controller = PairedOnlineController(
        plan,
        comparator,
        _campaign_factory(tmp_path, plan, [], wrong_provenance=True),
        initial_llm_source=initial,
    )
    with pytest.raises(ValueError, match="exact pairing provenance"):
        controller.run(_schedule(), _window, ladder)

    controller = PairedOnlineController(
        plan,
        comparator,
        _campaign_factory(tmp_path / "wrong-ladder", plan, [], wrong_ladder=True),
        initial_llm_source=initial,
    )
    with pytest.raises(ValueError, match="shared evaluation ladder"):
        controller.run(_schedule(), _window, ladder)


def test_paired_controller_retains_incumbent_when_no_candidate_is_feasible(tmp_path) -> None:
    plan = _plan()
    initial = (ROOT / "evolution" / "task" / "initial.py").read_text()
    digest = hashlib.sha256(initial.encode()).hexdigest()
    run = PairedOnlineController(
        plan,
        _comparator(plan),
        _campaign_factory(tmp_path, plan, [], feasible=False),
        initial_llm_source=initial,
    ).run(_schedule(), _window, load_ladder_config(TASK_CONFIG))
    assert all(not event.llm_promoted for event in run.events)
    assert all(
        event.llm_deployment_status == "incumbent-retained-no-feasible-candidate"
        for event in run.events
    )
    assert all(event.llm_incumbent_after_sha256 == digest for event in run.events)


def test_in_process_online_controller_refuses_confirmatory_authority() -> None:
    work = dataclasses.replace(_work(), registration_status="p2-frozen")
    trigger = dataclasses.replace(_trigger(), registration_status="p2-frozen")
    plan = dataclasses.replace(_plan(), status="p2-frozen", trigger=trigger, per_trigger_work=work)
    initial = (ROOT / "evolution" / "task" / "initial.py").read_text()
    with pytest.raises(ValueError, match="non-confirmatory"):
        PairedOnlineController(
            plan,
            _comparator(plan),
            lambda *args: object(),
            initial_llm_source=initial,
        )


def test_checked_in_plan_is_single_source_for_both_arms_and_provisional() -> None:
    plan = load_online_adaptation_plan(ROOT / "baselines" / "online_adaptation.json")
    assert plan.arms == ("llm_continuous", "online_gphh")
    assert not plan.confirmatory_ready
    assert plan.per_trigger_work.max_microwork == 10_000_000
    assert len(plan.sha256) == len(plan.work_sha256) == 64
