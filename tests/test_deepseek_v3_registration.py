"""The model change must not silently alter the inherited scientific comparisons."""

import json
from pathlib import Path

import pytest

from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.prospective_20260920.build import OUT, V1, sha


@pytest.mark.parametrize("study", ["A", "B"])
def test_all_original_roots_and_archives_retained(study):
    original = read_sealed(V1 / "FREEZE.json")
    for i in range(30):
        name = f"initialization/{study}{i:02d}.json"
        assert sha(OUT / name) == original["files_sha256"][name]
    plan = json.loads((OUT / f"campaign_{study}.json").read_text())
    assert len(plan["campaigns"]) == 30
    assert plan["evaluator_work"]["max_microwork"] == 56000000
    assert plan["proposals_per_call"] == 1


def test_seed_population_and_scenario_definitions_unchanged():
    for name in ("SEEDS.json", "task_A.json", "task_B.json", "gp.json", "POWER.json"):
        assert (OUT / name).read_bytes() == (V1 / name).read_bytes()


def test_two_primary_hypotheses_restore_multiplicity():
    analysis = json.loads((OUT / "ANALYSIS.json").read_text())
    assert len(analysis["primary_family"]) == 2
    assert analysis["multiplicity"] == "Holm, two tests"
    assert analysis["familywise_alpha"] == 0.05


def test_exact_provider_and_no_unsupported_seed():
    engine = json.loads((OUT / "ENGINE_PILOT.json").read_text())
    assert engine["model_id"] == "deepseek/deepseek-v4.1-flash"
    assert engine["generation"]["provider"]["only"] == ["deepseek"]
    assert engine["generation"]["provider"]["allow_fallbacks"] is False
    assert engine["generation"]["provider"]["require_parameters"] is True
    assert engine["reasoning_effort"] == "high"
    assert engine["max_attempts"] == 1
    assert engine["max_total_microusd"] == 600000
    metadata = json.loads(
        (Path(__file__).parents[1] / "experiments/deepseek_pilots_20260920/model.json").read_text()
    )
    assert engine["reasoning_effort"] in metadata["reasoning"]["supported_efforts"]


def test_elected_low_reasoning_treatment_before_freeze():
    engine = json.loads((OUT / "engine.json").read_text())
    pilot = json.loads((OUT / "ENGINE_PILOT.json").read_text())
    assert engine["reasoning_effort"] == "low"
    assert engine["max_output_tokens"] == 32000
    assert engine["max_total_microusd"] == 1500000
    # Only the elected treatment and the per-root ceiling differ from the pilot engine.
    changed = {k for k in engine if engine[k] != pilot[k]}
    assert changed == {"reasoning_effort", "max_output_tokens", "max_total_microusd"}
    amendment = json.loads((OUT / "TREATMENT_AMENDMENT.json").read_text())
    assert amendment["changes"]["reasoning_effort"] == ["high", "low"]
    assert amendment["pilot_summary_sha256"] == sha(OUT / "PILOT_SUMMARY.json")
    resources = json.loads((OUT / "MODEL_AND_RESOURCES.json").read_text())
    assert resources["fits_authorized_ceiling"] is True
    assert resources["run_authorized"] is False
    gates = json.loads((OUT / "GATES.json").read_text())
    assert gates["launch_allowed"] is False
