"""The scoped study-A design must be a verifiable strict subset of the frozen v1 design.

The scientific risk in shrinking a registered design is that something else changes with
it -- a re-drawn seed, a re-tuned operator, a shortened search -- and the reduction stops
being auditable. These tests pin that only two things changed (the study set/population,
and the source archive) and that everything the comparison depends on is byte-identical
to a freeze that predates any executed root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.prospective_20260917.statistics import power

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "prereg/CORRECTED_STUDIES_20260917_v1"
V2 = ROOT / "prereg/CORRECTED_STUDIES_20260919_v2"

pytestmark = pytest.mark.skipif(not V2.exists(), reason="v2 design not prepared")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def v1_freeze() -> dict:
    return read_sealed(V1 / "FREEZE.json")


def test_v1_was_never_executed_so_no_outcome_informed_the_reduction(v1_freeze) -> None:
    assert v1_freeze["launch_allowed"] is False
    assert v1_freeze["status"] == "design-frozen-execution-blocked"
    assert json.loads((V1 / "GATES.json").read_text())["launch_allowed"] is False
    # No campaign archive, and therefore no candidate, exists under the v1 design.
    assert not list(V1.rglob("events.jsonl"))


@pytest.mark.parametrize("name", ["task_A.json", "gp.json", "engine.json"])
def test_inherited_inputs_are_byte_identical_to_the_v1_freeze(name, v1_freeze) -> None:
    assert sha(V2 / name) == v1_freeze["files_sha256"][name] == sha(V1 / name)


def test_registered_roots_are_a_verbatim_balanced_prefix_of_v1(v1_freeze) -> None:
    v1_roots = json.loads((V1 / "SEEDS.json").read_text())["studies"]["A"]
    v2 = json.loads((V2 / "SEEDS.json").read_text())

    assert v2["roots"] == v1_roots[:12]
    assert v2["inherited_sha256"] == v1_freeze["files_sha256"]["SEEDS.json"]
    arms = [r["first_arm"] for r in v2["roots"]]
    assert arms.count("llm") == arms.count("gp") == 6
    assert len({r["root_seed"] for r in v2["roots"]}) == 12


def test_initialization_archives_are_byte_identical_to_v1(v1_freeze) -> None:
    archives = sorted((V2 / "initialization").glob("*.json"))
    assert [p.stem for p in archives] == [f"A{i:02d}" for i in range(12)]
    for path in archives:
        assert sha(path) == v1_freeze["files_sha256"][f"initialization/{path.name}"]


def test_search_depth_and_ladder_are_unchanged() -> None:
    v1 = json.loads((V1 / "campaign_A.json").read_text())
    v2 = json.loads((V2 / "campaign_A.json").read_text())

    assert v2["evaluator_work"]["max_microwork"] == v1["evaluator_work"]["max_microwork"]
    assert (
        v2["evaluator_work"]["checkpoints_microwork"]
        == v1["evaluator_work"]["checkpoints_microwork"]
    )
    for key in (
        "candidate_budget",
        "evolution_profile",
        "feedback_profile",
        "shared_numeric_probability",
        "proposals_per_call",
        "max_operator_attempts",
    ):
        assert v2[key] == v1[key], key
    # The population is the only campaign difference.
    assert v2["campaigns"] == v1["campaigns"][:12]


def test_only_the_population_differs_between_the_two_campaign_plans() -> None:
    v1 = json.loads((V1 / "campaign_A.json").read_text())
    v2 = json.loads((V2 / "campaign_A.json").read_text())
    differing = {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}

    assert differing == {"campaigns", "evaluator_work"}
    work_diff = {
        k
        for k in set(v1["evaluator_work"]) | set(v2["evaluator_work"])
        if v1["evaluator_work"].get(k) != v2["evaluator_work"].get(k)
    }
    assert work_diff == {"registration_source"}


def test_single_primary_test_does_not_reallocate_alpha() -> None:
    analysis = json.loads((V2 / "ANALYSIS.json").read_text())

    assert analysis["alpha"] == 0.05
    assert analysis["multiplicity"]["family_size"] == 1
    assert analysis["multiplicity"]["adjustment"] == "none"


def test_decision_rule_matches_an_independent_exact_binomial_computation() -> None:
    from scipy.stats import binomtest

    rule = json.loads((V2 / "ANALYSIS.json").read_text())["decision_rule"]
    for key, row in rule.items():
        if key == "note":
            continue
        n, needed = row["non_tied_roots"], row["wins_needed_to_reject"]
        if needed is None:
            # No win count on the advantage side can reach alpha.
            assert all(binomtest(w, n, 0.5).pvalue > 0.05 for w in range(n // 2 + 1, n + 1))
            continue
        assert 2 * needed > n
        assert binomtest(needed, n, 0.5).pvalue <= 0.05
        assert binomtest(needed - 1, n, 0.5).pvalue > 0.05


def test_registered_power_is_reported_honestly_for_a_small_design() -> None:
    reported = json.loads((V2 / "POWER.json").read_text())

    assert reported["registered_n"] == 12
    assert reported["registered_design_power"]["win_0.8_no_ties"] == power(12, 0.8, 0.0, 0.05)
    # The claim in PROTOCOL.md and in POWER.json's own interpretation.
    assert reported["registered_design_power"]["win_0.8_no_ties"] < 0.6
    assert power(12, 0.8, 0.0, 0.05) < power(30, 0.8, 0.0, 0.05)
    assert "does not establish equivalence" in reported["interpretation"]


def test_projected_cost_fits_the_authorized_ceiling_and_is_the_largest_that_does() -> None:
    r = json.loads((V2 / "MODEL_AND_RESOURCES.json").read_text())
    per_root = r["measured"]["usd_per_paired_root"]

    assert r["authorized_total_usd"] == 60.0
    assert r["projected_api_usd"] == pytest.approx(12 * per_root)
    assert r["projected_api_usd_with_40pct_variance_reserve"] + r["already_spent_usd"] <= 60.0
    assert r["fits_authorized_ceiling"] is True
    # Thirteen roots must not fit, or twelve would not be the largest affordable design.
    assert 13 * per_root * 1.4 + r["already_spent_usd"] > 60.0
    assert r["run_authorized"] is False


def test_launch_stays_blocked_and_failing_gates_are_not_marked_passing() -> None:
    gates = json.loads((V2 / "GATES.json").read_text())
    by_id = {g["id"]: g for g in gates["gates"]}

    assert gates["launch_allowed"] is False
    assert by_id["scorer-latency-gate"]["status"] == "failing"
    assert by_id["G0-capacity-authorization"]["status"] == "pending"
    assert by_id["G0-concurrency"]["status"] == "pending"
    assert by_id["P2"]["status"] == "pending"
    assert not any(g["status"].lower() in {"pass", "passed", "ok"} for g in gates["gates"])


def test_deviation_record_names_what_each_change_costs_in_claims() -> None:
    deviation = json.loads((V2 / "DEVIATION_FROM_V1.json").read_text())
    changes = {c["what"]: c for c in deviation["changes"]}

    assert deviation["supersedes"].endswith("CORRECTED_STUDIES_20260917_v1")
    assert set(changes) == {
        "study set",
        "paired roots",
        "multiplicity adjustment",
        "registered evaluation source",
    }
    for what, change in changes.items():
        assert change["why"] and change["cost_in_claims"], what
    assert "transfer" in changes["study set"]["cost_in_claims"].lower()
    assert "equivalence" in changes["paired roots"]["cost_in_claims"].lower()
