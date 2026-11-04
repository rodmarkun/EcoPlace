"""Exact invariants for registered evolution profiles."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from evolution.feedback import load_feedback_profiles
from evolution.profiles import (
    _PROFILES,
    DEEP_SEARCH_FEEDBACK_PROFILE,
    DEEP_SEARCH_PROFILE,
    evolution_profile,
)

ROOT = Path(__file__).parents[1]
TASK_CONFIG = ROOT / "evolution" / "task" / "config_dev_continuum_tight.json"


def test_deep_search_feedback_is_single_variable_ablation() -> None:
    baseline = evolution_profile(DEEP_SEARCH_PROFILE)
    feedback = evolution_profile(DEEP_SEARCH_FEEDBACK_PROFILE)
    differing = {
        field.name
        for field in fields(baseline)
        if getattr(baseline, field.name) != getattr(feedback, field.name)
    }

    # ``name`` is profile identity, while the flag and selected feedback profile
    # together express the single behavioral variable: simulator feedback on/off.
    assert differing == {"feedback_profile", "name", "simulator_feedback_enabled"}
    assert feedback.simulator_feedback_enabled is True
    assert feedback.feedback_profile == "e5-text-feedback-on"

    feedback_profiles = load_feedback_profiles(TASK_CONFIG)
    for profile in (baseline, feedback):
        assert (
            feedback_profiles[profile.feedback_profile].enabled
            is profile.simulator_feedback_enabled
        )


def test_registered_profile_keys_match_profile_names() -> None:
    for key, profile in _PROFILES.items():
        assert key == profile.name
