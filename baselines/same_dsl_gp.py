"""Public Phase-7 entry for the exact Phase-5 same-DSL GP arm (TODO 7.2).

There is intentionally no second implementation here.  The baseline factory returns
``evolution.gp_operator.PlacementScriptGPOperator`` itself and the runner factory uses
``evolution.campaign.CampaignRunner`` itself.  This prevents representational or
downstream-harness drift between the named baseline and the G-ablate control arm.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from evolution.campaign import CampaignPlan, CampaignRunner
from evolution.feedback import FeedbackProfile
from evolution.gp_operator import GPOperatorConfig, PlacementScriptGPOperator


@dataclass(frozen=True)
class SameDSLGPArm:
    """Identity and configuration receipt for the never-descoped ablation arm."""

    config: GPOperatorConfig = dataclasses.field(default_factory=GPOperatorConfig)
    arm_name: str = "gp"
    operator_name: str = PlacementScriptGPOperator.name
    representation: str = "placementscript-ast"
    downstream_path: str = "evolution.campaign.CampaignRunner"

    def __post_init__(self) -> None:
        if self.arm_name != "gp":
            raise ValueError("same-DSL baseline arm is canonically named 'gp'")
        if self.operator_name != PlacementScriptGPOperator.name:
            raise ValueError("same-DSL baseline must use the Phase-5 operator identity")

    def operator(self) -> PlacementScriptGPOperator:
        return PlacementScriptGPOperator(self.config)

    def validate_matched_profile(
        self, plan: CampaignPlan, feedback: FeedbackProfile
    ) -> None:
        if self.arm_name not in plan.arms:
            raise ValueError("campaign plan does not contain the canonical GP arm")
        if plan.feedback_profile != "matched-llm-vs-gp":
            raise ValueError("same-DSL confirmatory arm requires matched-llm-vs-gp profile")
        if feedback.name != plan.feedback_profile or feedback.enabled:
            raise ValueError("simulator text feedback must be disabled for the matched GP arm")

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-same-dsl-gp-v1",
            "arm_name": self.arm_name,
            "operator_name": self.operator_name,
            "operator_class": (
                "evolution.gp_operator.PlacementScriptGPOperator"
            ),
            "representation": self.representation,
            "downstream_path": self.downstream_path,
            "config": dataclasses.asdict(self.config),
            "role": "confirmatory-operator-controlled-ablation",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {**payload, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def build_same_dsl_gp_runner(*, arm: SameDSLGPArm | None = None, **kwargs: Any) -> CampaignRunner:
    """Construct the shared Phase-5 runner with the canonical GP arm injected."""
    selected = arm or SameDSLGPArm()
    plan = kwargs.get("plan")
    feedback = kwargs.get("feedback_profile")
    if not isinstance(plan, CampaignPlan) or not isinstance(feedback, FeedbackProfile):
        raise TypeError("runner construction requires CampaignPlan and FeedbackProfile")
    selected.validate_matched_profile(plan, feedback)
    if "arm" in kwargs and kwargs["arm"] != selected.arm_name:
        raise ValueError("caller cannot relabel the same-DSL GP arm")
    if "operator" in kwargs:
        raise ValueError("caller cannot replace the same-DSL GP operator")
    return CampaignRunner(
        **kwargs,
        arm=selected.arm_name,
        operator=selected.operator(),
    )
