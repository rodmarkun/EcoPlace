"""Fail-closed evolution profiles separating matched science from dev guidance."""

from __future__ import annotations

from dataclasses import dataclass, replace

MATCHED_PROFILE = "matched-llm-vs-gp"
ENHANCED_PROFILE = "enhanced-llm-dev"
LEAN_DEEP_PROFILE = "lean-deep"
DEEP_SEARCH_PROFILE = "deep-search"
DEEP_SEARCH_FEEDBACK_PROFILE = "deep-search-feedback"
DEEP_SEARCH_NOREPAIR_PROFILE = "deep-search-norepair"


@dataclass(frozen=True)
class EvolutionProfile:
    name: str
    development_only: bool
    numeric_interleave_enabled: bool
    shared_numeric_probability: float
    grammar_aware_retry_enabled: bool
    prompt_exemplars_enabled: bool
    behavioral_novelty_enabled: bool
    behavioral_novelty_tolerance: float
    structural_rewrite_bias_enabled: bool
    proposal_mode_weights: tuple[float, float, float] | None
    diverse_inspirations_enabled: bool
    max_inspirations: int
    simulator_feedback_enabled: bool
    feedback_profile: str
    seeded_initial_archive_enabled: bool
    seeded_random_mutants: int
    # k-way tournament on the Phase-3 order when > 1; 1 reproduces the historical
    # uniform draw byte-for-byte. Applied in the shared runner, so both proposal
    # operators receive identical selection pressure.
    parent_selection_tournament_k: int = 1


_PROFILES = {
    MATCHED_PROFILE: EvolutionProfile(
        name=MATCHED_PROFILE,
        development_only=False,
        numeric_interleave_enabled=True,
        shared_numeric_probability=0.2,
        grammar_aware_retry_enabled=False,
        prompt_exemplars_enabled=False,
        behavioral_novelty_enabled=False,
        behavioral_novelty_tolerance=1e-9,
        structural_rewrite_bias_enabled=False,
        proposal_mode_weights=None,
        diverse_inspirations_enabled=False,
        max_inspirations=2,
        simulator_feedback_enabled=False,
        feedback_profile=MATCHED_PROFILE,
        seeded_initial_archive_enabled=False,
        seeded_random_mutants=0,
    ),
    ENHANCED_PROFILE: EvolutionProfile(
        name=ENHANCED_PROFILE,
        development_only=True,
        numeric_interleave_enabled=False,
        shared_numeric_probability=0.0,
        grammar_aware_retry_enabled=True,
        prompt_exemplars_enabled=True,
        behavioral_novelty_enabled=True,
        behavioral_novelty_tolerance=1e-9,
        structural_rewrite_bias_enabled=True,
        # Diagnostic: 10/11 improvements were large; rewrite/crossover generated all 11.
        proposal_mode_weights=(0.10, 0.60, 0.30),
        diverse_inspirations_enabled=True,
        max_inspirations=4,
        simulator_feedback_enabled=True,
        feedback_profile=ENHANCED_PROFILE,
        seeded_initial_archive_enabled=True,
        seeded_random_mutants=5,
    ),
    LEAN_DEEP_PROFILE: EvolutionProfile(
        name=LEAN_DEEP_PROFILE,
        development_only=True,
        numeric_interleave_enabled=False,
        shared_numeric_probability=0.0,
        grammar_aware_retry_enabled=False,
        prompt_exemplars_enabled=False,
        behavioral_novelty_enabled=True,
        behavioral_novelty_tolerance=1e-9,
        structural_rewrite_bias_enabled=False,
        proposal_mode_weights=None,
        diverse_inspirations_enabled=False,
        max_inspirations=2,
        simulator_feedback_enabled=False,
        feedback_profile=MATCHED_PROFILE,
        seeded_initial_archive_enabled=False,
        seeded_random_mutants=0,
    ),
    DEEP_SEARCH_PROFILE: EvolutionProfile(
        name=DEEP_SEARCH_PROFILE,
        development_only=True,
        numeric_interleave_enabled=False,
        shared_numeric_probability=0.0,
        grammar_aware_retry_enabled=True,
        prompt_exemplars_enabled=False,
        behavioral_novelty_enabled=True,
        behavioral_novelty_tolerance=1e-9,
        structural_rewrite_bias_enabled=False,
        proposal_mode_weights=None,
        diverse_inspirations_enabled=False,
        max_inspirations=2,
        simulator_feedback_enabled=False,
        feedback_profile=MATCHED_PROFILE,
        seeded_initial_archive_enabled=True,
        seeded_random_mutants=5,
        parent_selection_tournament_k=3,
    ),
    DEEP_SEARCH_NOREPAIR_PROFILE: EvolutionProfile(
        name=DEEP_SEARCH_NOREPAIR_PROFILE,
        development_only=True,
        numeric_interleave_enabled=False,
        shared_numeric_probability=0.0,
        grammar_aware_retry_enabled=False,
        prompt_exemplars_enabled=False,
        behavioral_novelty_enabled=True,
        behavioral_novelty_tolerance=1e-9,
        structural_rewrite_bias_enabled=False,
        proposal_mode_weights=None,
        diverse_inspirations_enabled=False,
        max_inspirations=2,
        simulator_feedback_enabled=False,
        feedback_profile=MATCHED_PROFILE,
        seeded_initial_archive_enabled=True,
        seeded_random_mutants=5,
        parent_selection_tournament_k=3,
    ),
}

# Keep this as a derived, single-variable ablation: even newly added dataclass fields
# inherit the frozen deep-search value unless explicitly changed here. The flag and
# named feedback profile are one conceptual variable; they must agree or preflight
# fails.
_PROFILES[DEEP_SEARCH_FEEDBACK_PROFILE] = replace(
    _PROFILES[DEEP_SEARCH_PROFILE],
    name=DEEP_SEARCH_FEEDBACK_PROFILE,
    simulator_feedback_enabled=True,
    feedback_profile="e5-text-feedback-on",
)


def evolution_profile(name: str) -> EvolutionProfile:
    """Return one exact profile; unknown labels never degrade to matched behavior."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown evolution profile {name!r}") from exc


__all__ = [
    "DEEP_SEARCH_FEEDBACK_PROFILE",
    "DEEP_SEARCH_NOREPAIR_PROFILE",
    "DEEP_SEARCH_PROFILE",
    "ENHANCED_PROFILE",
    "LEAN_DEEP_PROFILE",
    "MATCHED_PROFILE",
    "EvolutionProfile",
    "evolution_profile",
]
