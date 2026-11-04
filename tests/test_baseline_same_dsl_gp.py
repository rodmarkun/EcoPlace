from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import pytest
from shinka.database import Program

from baselines.same_dsl_gp import SameDSLGPArm, build_same_dsl_gp_runner
from evolution.campaign import load_campaign_plan
from evolution.feedback import load_feedback_profiles
from evolution.gp_operator import PlacementScriptGPOperator

ROOT = Path(__file__).parents[1]


def _program(name: str) -> Program:
    return Program(
        id=name,
        code=(ROOT / "runtime" / "policies" / f"{name}.ps").read_text(),
        language="placementscript",
        correct=True,
    )


def test_same_dsl_entry_is_exact_phase5_operator() -> None:
    arm = SameDSLGPArm()
    exposed = arm.operator()
    direct = PlacementScriptGPOperator()
    assert type(exposed) is type(direct) is PlacementScriptGPOperator
    parent = _program("least_allocated")
    inspirations = (_program("most_allocated"),)
    assert exposed(parent, inspirations, random.Random(91)) == direct(
        parent, inspirations, random.Random(91)
    )


def test_same_dsl_receipt_is_canonical_and_names_shared_path() -> None:
    a = SameDSLGPArm().receipt()
    b = SameDSLGPArm().receipt()
    assert a == b
    assert len(a["sha256"]) == 64
    assert a["operator_class"] == "evolution.gp_operator.PlacementScriptGPOperator"
    assert a["downstream_path"] == "evolution.campaign.CampaignRunner"


def test_same_dsl_profile_requires_matched_feedback_off() -> None:
    plan = load_campaign_plan(ROOT / "evolution" / "campaigns.json")
    profiles = load_feedback_profiles(ROOT / "evolution" / "task" / "config.json")
    arm = SameDSLGPArm()
    arm.validate_matched_profile(plan, profiles["matched-llm-vs-gp"])
    with pytest.raises(ValueError, match="matched-llm-vs-gp"):
        arm.validate_matched_profile(
            dataclasses.replace(plan, feedback_profile="e5-text-feedback-off"),
            profiles["e5-text-feedback-off"],
        )
    with pytest.raises(ValueError, match="disabled"):
        arm.validate_matched_profile(
            plan,
            dataclasses.replace(profiles["matched-llm-vs-gp"], enabled=True),
        )


def test_runner_factory_refuses_operator_replacement() -> None:
    plan = load_campaign_plan(ROOT / "evolution" / "campaigns.json")
    feedback = load_feedback_profiles(ROOT / "evolution" / "task" / "config.json")[
        "matched-llm-vs-gp"
    ]
    with pytest.raises(ValueError, match="cannot replace"):
        build_same_dsl_gp_runner(
            plan=plan,
            feedback_profile=feedback,
            operator=object(),
        )
