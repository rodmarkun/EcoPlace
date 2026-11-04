"""Structured simulator feedback as an explicitly separate ablation (TODO 5.5).

The matched LLM-vs-GP comparison never calls this generator.  The E5 feedback-on
contrast may render the same archived ladder evidence into a deterministic narrative;
no extra evaluation, hidden fixture, or model call is involved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim.energy import ConstraintSpec

from .ladder import LadderResult

MATCHED_PAIR_PROFILE = "matched-llm-vs-gp"
FEEDBACK_OFF_PROFILE = "e5-text-feedback-off"
FEEDBACK_ON_PROFILE = "e5-text-feedback-on"
ENHANCED_LLM_DEV_PROFILE = "enhanced-llm-dev"
LEGACY_REQUIRED_PROFILES = {
    MATCHED_PAIR_PROFILE: False,
    FEEDBACK_OFF_PROFILE: False,
    FEEDBACK_ON_PROFILE: True,
}
REQUIRED_PROFILES = {
    **LEGACY_REQUIRED_PROFILES,
    ENHANCED_LLM_DEV_PROFILE: True,
}
CATEGORY_NAMES = ("slo", "goodput", "drop_timeout", "disruption")


@dataclass(frozen=True)
class FeedbackProfile:
    name: str
    enabled: bool


@dataclass(frozen=True)
class FailureSignal:
    category: str
    magnitude: float | None
    evidence: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "magnitude": self.magnitude,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class FeedbackNarrative:
    schema: str
    profile: str
    terminal_rung: str
    summary: str
    signals: tuple[FailureSignal, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile,
            "terminal_rung": self.terminal_rung,
            "summary": self.summary,
            "signals": [signal.to_dict() for signal in self.signals],
        }

    def render_text(self) -> str:
        lines = [self.summary]
        for signal in self.signals:
            magnitude = "" if signal.magnitude is None else f" ({signal.magnitude:.6g})"
            lines.append(
                f"- {signal.category}{magnitude}: {signal.evidence} "
                f"Suggested focus: {signal.suggestion}"
            )
        return "\n".join(lines)


def load_feedback_profiles(path: str | Path) -> dict[str, FeedbackProfile]:
    raw = json.loads(Path(path).read_text()).get("feedback_profiles")
    if not isinstance(raw, dict):
        raise ValueError("task config must define feedback_profiles")
    if frozenset(raw) not in {
        frozenset(LEGACY_REQUIRED_PROFILES),
        frozenset(REQUIRED_PROFILES),
    }:
        raise ValueError(
            "feedback profiles must be exactly the sealed legacy set or the enhanced set"
        )
    profiles: dict[str, FeedbackProfile] = {}
    expected = (
        REQUIRED_PROFILES
        if ENHANCED_LLM_DEV_PROFILE in raw
        else LEGACY_REQUIRED_PROFILES
    )
    for name, required_enabled in expected.items():
        item = raw[name]
        if not isinstance(item, dict) or not isinstance(item.get("enabled"), bool):
            raise ValueError(f"feedback profile {name!r} needs an explicit enabled bool")
        if item["enabled"] is not required_enabled:
            raise ValueError(
                f"feedback profile {name!r} must set enabled={required_enabled}; "
                "the matched comparison and E5 contrast are predeclared"
            )
        profiles[name] = FeedbackProfile(name=name, enabled=required_enabled)
    return profiles


def _l0_signals(trace: dict[str, Any]) -> list[FailureSignal]:
    result = trace["result"]
    out: list[FailureSignal] = []
    for instance in result["instances"]:
        if instance["error"]:
            out.append(
                FailureSignal(
                    category="policy_runtime",
                    magnitude=None,
                    evidence=instance["error"],
                    suggestion="keep score arithmetic total and within PlacementScript limits.",
                )
            )
        for violation in instance["violations"]:
            prefix = violation.split(":", 1)[0]
            suggestions = {
                "unschedulable": "avoid concentrating allocations until no node can fit a pod.",
                "saturated": "spread replicas or favor nodes with more available CPU.",
                "no_replica": "preserve at least one serving placement for every active service.",
                "slo": "reduce path latency and queueing on the critical service chain.",
            }
            out.append(
                FailureSignal(
                    category=f"l0_{prefix}",
                    magnitude=None,
                    evidence=violation,
                    suggestion=suggestions.get(prefix, "inspect the reported L0 constraint."),
                )
            )
    if not out and result["mean_normalized_energy"] is not None:
        out.append(
            FailureSignal(
                category="l0_energy",
                magnitude=result["mean_normalized_energy"],
                evidence="the analytical normalized-energy promotion threshold was exceeded.",
                suggestion="reduce active non-cloud power and unnecessary cloud allocations.",
            )
        )
    return out


def _fitness_signals(summary: dict[str, Any], rung: str) -> list[FailureSignal]:
    vector = summary["infeasibility_vector"]
    suggestions = {
        "slo": "favor placements with shorter critical paths and lower queueing pressure.",
        "goodput": "retain serving replicas and avoid capacity-starved placements.",
        "drop_timeout": "leave headroom for bursts and reduce timeout-prone cross-tier paths.",
        "disruption": "make fewer/smaller moves and avoid slow wake or migration actions.",
    }
    out = [
        FailureSignal(
            category=category,
            magnitude=value,
            evidence=f"{rung} normalized {category} shortfall is non-zero.",
            suggestion=suggestions[category],
        )
        for category, value in zip(CATEGORY_NAMES, vector, strict=True)
        if value > 0.0
    ]
    if not out and summary["normalized_energy"] is not None:
        out.append(
            FailureSignal(
                category="normalized_energy",
                magnitude=summary["normalized_energy"],
                evidence=f"{rung} is feasible; energy is now the ordering objective.",
                suggestion="lower closed-boundary energy without creating any shortfall.",
            )
        )
    return out


def _constraint_signal(constraints: ConstraintSpec) -> FailureSignal:
    evidence = json.dumps(
        {
            "slo_targets": dict(constraints.slo_targets),
            "goodput_target": constraints.goodput_target,
            "drop_cap": constraints.drop_cap,
            "timeout_cap": constraints.timeout_cap,
            "disruption_caps": dict(constraints.disruption_caps),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return FailureSignal(
        category="binding_constraints",
        magnitude=None,
        evidence=evidence,
        suggestion="treat every listed target/cap as binding while reducing energy.",
    )


def _energy_signals(terminal: dict[str, Any]) -> list[FailureSignal]:
    evaluation = terminal.get("result", {}).get("evaluation")
    instances = evaluation.get("instances", []) if isinstance(evaluation, dict) else []
    components: dict[str, list[float]] = {"device_j": [], "cloud_j": [], "network_j": []}
    for instance in instances:
        energy = instance.get("episode", {}).get("energy", {})
        for category in components:
            value = energy.get(category)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                components[category].append(float(value))
    signals = []
    for category, values in components.items():
        if values:
            mean = sum(values) / len(values)
            signals.append(
                FailureSignal(
                    category=f"energy_{category[:-2]}",
                    magnitude=mean,
                    evidence=f"mean {category} across {len(values)} {terminal['rung']} instances.",
                    suggestion=f"reduce the {category[:-2]} component without violating caps.",
                )
            )
    return signals


def generate_feedback(
    result: LadderResult,
    profile: FeedbackProfile,
    *,
    constraints: ConstraintSpec | None = None,
    recent_rejections: tuple[str, ...] = (),
) -> FeedbackNarrative | None:
    """Render archived evidence only when the predeclared profile enables it."""
    if not profile.enabled:
        return None
    terminal = next(
        trace for trace in reversed(result.trace) if trace["rung"] in {"L0", "L1", "L2"}
    )
    if terminal["rung"] == "L0":
        signals = _l0_signals(terminal)
    else:
        signals = _fitness_signals(terminal["result"], terminal["rung"])
    if profile.name == ENHANCED_LLM_DEV_PROFILE:
        if constraints is None:
            raise ValueError("enhanced feedback requires the parent's binding constraints")
        signals = [_constraint_signal(constraints), *_energy_signals(terminal), *signals]
        signals.extend(
            FailureSignal(
                category="recent_rejected_attempt",
                magnitude=None,
                evidence=reason,
                suggestion="avoid repeating this locally validated failure.",
            )
            for reason in recent_rejections
        )
    disposition = "rejected" if not result.promoted_to_l2 else "L2 evaluated"
    return FeedbackNarrative(
        schema="structured-placement-feedback-v1",
        profile=profile.name,
        terminal_rung=terminal["rung"],
        summary=f"Candidate {disposition}; evidence comes from {terminal['rung']}.",
        signals=tuple(signals),
    )
