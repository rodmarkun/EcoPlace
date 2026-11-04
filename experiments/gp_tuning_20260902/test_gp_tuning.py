"""Pinned unit tests for the GP-tuning grid and the prereg sign test."""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import analyze  # noqa: E402


def test_grid_is_the_brief_and_is_fixed():
    assert list(analyze.MIXES) == [
        "current",
        "numeric-heavy",
        "subtree-heavy",
        "crossover-heavy",
    ]
    assert analyze.MIXES["current"]["crossover_probability"] == 0.45
    assert analyze.MIXES["numeric-heavy"]["numeric_mutation_probability"] == 0.60
    assert analyze.MIXES["subtree-heavy"]["subtree_mutation_probability"] == 0.60
    assert analyze.MIXES["crossover-heavy"]["crossover_probability"] == 0.70
    assert analyze.ROOTS == tuple(range(800000801, 800000809))
    assert ("current", 56.0) in analyze.CELLS
    assert ("current", 224.0) in analyze.CELLS
    assert ("numeric-heavy", 112.0) not in analyze.CELLS


def test_current_56_reuses_runs_final_and_tuned_names_are_isolated():
    reused = analyze.cell_run_dir(800000801, "current", 56.0)
    assert reused == analyze.RUNS_FINAL / analyze.final_run_name(800000801)
    assert "--gp-mix-" not in reused.name
    tuned = analyze.cell_run_dir(800000801, "numeric-heavy", 56.0)
    assert tuned == analyze.RUNS_ROOT / analyze.tuned_run_name(
        800000801, "numeric-heavy", 56.0
    )
    assert tuned.name.endswith("--gp-mix-numeric-heavy")
    assert "non-llm" in tuned.name
    high = analyze.cell_run_dir(800000801, "current", 224.0)
    assert high.name.endswith("--gp-mix-current")
    assert "work-224000000" in high.name


def test_sign_test_matches_the_16_0_prereg_example():
    # 16 LLM wins, 0 GP wins: p = 2 / 2**16 = 3.0517578125e-05.
    assert math.isclose(analyze.exact_two_sided_sign_p(16, 0), 2 / 2**16)
    pairs = [
        {
            "gp_feasible": True,
            "gp_energy": 0.83,
            "llm_feasible": True,
            "llm_energy": 0.64,
        }
        for _ in range(8)
    ]
    result = analyze.sign_test(pairs)
    assert result["llm_wins"] == 8
    assert result["gp_wins"] == 0
    assert result["ties"] == 0
    assert result["attainable"] is True
    assert math.isclose(result["exact_p"], 2 / 2**8)


def test_lexicographic_feasibility_rule():
    assert analyze.lexicographic_direction(False, None, True, 0.6)[0] == 1
    assert analyze.lexicographic_direction(True, 0.9, False, None)[0] == -1
    assert analyze.lexicographic_direction(False, None, False, None)[0] == 0
    assert analyze.lexicographic_direction(True, 0.5, True, 0.5)[0] == 0
    assert analyze.lexicographic_direction(True, 0.7, True, 0.6)[1] == "both_feasible_llm"


def test_prekill_numeric_heavy_digests_are_pinned():
    assert analyze.PREKILL_NUMERIC_HEAVY_56[800000801].startswith("0cd4c7ed")
    assert len(analyze.PREKILL_NUMERIC_HEAVY_56) == 3


def test_token_distance_is_zero_on_identical_source():
    source = "def score():\n    return 1.0\n"
    distance = analyze.token_levenshtein(source, source)
    assert distance["levenshtein"] == 0
    assert distance["normalized"] == 0.0
