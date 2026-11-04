from __future__ import annotations

from pathlib import Path

from evolution.feedback import (
    FEEDBACK_ON_PROFILE,
    MATCHED_PAIR_PROFILE,
    FailureSignal,
    FeedbackNarrative,
    generate_feedback,
    load_feedback_profiles,
)
from evolution.ladder import LadderResult

TASK = Path(__file__).parents[1] / "evolution" / "task"


def _l1_reject() -> LadderResult:
    return LadderResult(
        promoted_to_l2=False,
        evaluation=None,
        trace=(
            {
                "rung": "L0",
                "decision": "promote",
                "result": {"valid": True, "feasible": True},
            },
            {
                "rung": "L1",
                "decision": "reject",
                "result": {
                    "fitness_sort_key": [1, 0.2, 0.0, 0.4, 0.0, 0.6, 9],
                    "feasible": False,
                    "normalized_energy": None,
                    "infeasibility_vector": [0.2, 0.0, 0.4, 0.0],
                    "summed_primitive_shortfall": 0.6,
                },
            },
        ),
        budget_registration_status="development-provisional",
        budget_registration_source="test",
        policy_name="test",
        policy_version="1",
        policy_identity_sha256="a" * 64,
        policy_artifact_sha256="b" * 64,
    )


def test_matched_pair_feedback_is_enforced_off():
    profiles = load_feedback_profiles(TASK / "config.json")

    assert profiles[MATCHED_PAIR_PROFILE].enabled is False
    assert generate_feedback(_l1_reject(), profiles[MATCHED_PAIR_PROFILE]) is None


def test_feedback_on_is_structured_and_uses_priority_order():
    profiles = load_feedback_profiles(TASK / "config.json")
    feedback = generate_feedback(_l1_reject(), profiles[FEEDBACK_ON_PROFILE])

    assert feedback is not None
    assert feedback.terminal_rung == "L1"
    assert [signal.category for signal in feedback.signals] == ["slo", "drop_timeout"]
    assert "Suggested focus" in feedback.render_text()


def test_render_is_deterministic():
    narrative = FeedbackNarrative(
        schema="structured-placement-feedback-v1",
        profile=FEEDBACK_ON_PROFILE,
        terminal_rung="L2",
        summary="summary",
        signals=(FailureSignal("slo", 0.125, "evidence", "suggestion"),),
    )

    assert narrative.render_text() == narrative.render_text()
    assert narrative.to_dict()["signals"][0]["magnitude"] == 0.125
