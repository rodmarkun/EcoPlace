"""Mechanical normalized-work matching rules shared by runner, summary, and tests."""

from __future__ import annotations

from typing import Any

ADAPTIVE_ARMS = ("same_dsl_gp", "classic_gphh", "random_ast", "llm")
EVALUATION_ONLY_ARMS = (
    "ppo_frozen",
    "latency_greedy",
    "energy_greedy",
    "first_fit",
    "least_allocated",
    "most_allocated",
    "network_overhead",
)


def validate_matching(
    accounting: dict[str, dict[str, Any]],
    *,
    matched_max_microwork: int,
    include_llm: bool,
) -> dict[str, Any]:
    """Fail closed unless adaptive budgets match and frozen references get none."""
    expected_adaptive = set(ADAPTIVE_ARMS if include_llm else ADAPTIVE_ARMS[:-1])
    expected = expected_adaptive | set(EVALUATION_ONLY_ARMS)
    if set(accounting) != expected:
        raise ValueError("arm accounting does not cover the exact requested roster")
    for arm in sorted(expected_adaptive):
        record = accounting[arm]
        if (
            record.get("mode") != "adaptive-normalized-work"
            or record.get("registered_max_microwork") != matched_max_microwork
        ):
            raise ValueError(f"adaptive arm {arm!r} is not work-matched")
        actual = record.get("actual_microwork")
        effective = record.get("effective_terminal_microwork")
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or not 0 <= actual <= matched_max_microwork
            or effective != matched_max_microwork
        ):
            raise ValueError(f"adaptive arm {arm!r} has invalid terminal work accounting")
    for arm in EVALUATION_ONLY_ARMS:
        record = accounting[arm]
        if (
            record.get("mode") != "frozen-evaluation-only"
            or record.get("registered_max_microwork") is not None
            or record.get("actual_microwork") != 0
            or record.get("effective_terminal_microwork") is not None
        ):
            raise ValueError(f"frozen reference {arm!r} was assigned adaptation work")
    return {
        "verified": True,
        "matched_max_microwork": matched_max_microwork,
        "matched_normalized_work": matched_max_microwork / 1_000_000,
        "adaptive_arms": sorted(expected_adaptive),
        "evaluation_only_arms": list(EVALUATION_ONLY_ARMS),
    }

