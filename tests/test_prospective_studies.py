"""Pre-outcome statistical and design-boundary tests; no simulation or paid API."""

from types import SimpleNamespace

import pytest

from experiments.allbaselines_harness.initialization import materialize_seed_archive
from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.prospective_20260917 import build
from experiments.prospective_20260917.statistics import (
    direction,
    holm_two,
    paired_report,
    power,
    sign_summary,
)


def outcome(feasible, energy=None):
    return {"status": "evaluated", "feasible": feasible, "energy": energy}


def test_feasibility_dominates_energy_and_missing_is_not_a_tie():
    assert direction(outcome(True, 100), outcome(False, 0)) == 1
    assert direction(outcome(False), outcome(False)) == 0
    assert direction({"status": "missing"}, outcome(True, 1)) is None
    assert direction(outcome(True, 1), outcome(True, 1 + 1e-10)) == 0
    assert direction(outcome(True, 1), outcome(True, 1.005), tolerance=0.01) == 0
    with pytest.raises(ValueError, match="finite"):
        direction(outcome(True, float("nan")), outcome(True, 1))


def test_exact_sign_test_and_holm_family():
    assert sign_summary([1] * 16 + [0] * 4)["p_two_sided"] == 2 / 2**16
    assert sign_summary([0] * 30)["p_two_sided"] == 1
    assert sign_summary([1] * 29 + [None])["p_two_sided"] is None
    assert holm_two([0.03, 0.001]) == [0.03, 0.002]
    assert holm_two([0.01, 0.015]) == [0.02, 0.02]
    assert holm_two([0.001, None]) == [None, None]


def test_root_denominators_and_conditional_energy():
    rows = [
        {"root_seed": 1, "left": outcome(True, 1), "right": outcome(True, 2)},
        {"root_seed": 2, "left": outcome(True, 1), "right": outcome(False)},
    ]
    result = paired_report(rows, [1, 2], bootstrap_seed=4)
    assert result["energy"]["jointly_feasible"] == 1
    assert result["energy"]["mean"] == 1
    assert result["energy"]["bootstrap_ci95_descriptive"] is None
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="complete unique"):
            paired_report(bad, [1, 2], bootstrap_seed=4)


def test_bootstrap_replay_and_reversed_root_input():
    rows = [
        {"root_seed": i, "left": outcome(True, 1), "right": outcome(True, 1 + i / 10)}
        for i in range(1, 5)
    ]
    assert paired_report(rows, [1, 2, 3, 4], bootstrap_seed=9) == paired_report(
        rows[::-1], [1, 2, 3, 4], bootstrap_seed=9
    )


def test_seed_roles_match_runner_and_collision_refused():
    from analysis.tracking import derive_seed

    seeds = build.seed_plan({"A": ["home"], "B": ["one", "two", "three", "four"]}, set())
    records = seeds["enumerated_seed_records"]
    assert len(records) == len({r["seed"] for r in records})
    for study in ("A", "B"):
        assert len(seeds["studies"][study]) == 30
        for row in seeds["studies"][study]:
            for sid, grid in row["grids"].items():
                assert grid["search"] == [
                    derive_seed(row["root_seed"], "sim", "scenario", sid, "replicate", 0)
                ]
                assert len(grid["final"]) == 8
    with pytest.raises(ValueError, match="seed collision"):
        build.seed_plan({"A": ["home"], "B": ["new"]}, {records[0]["seed"]})


def test_A_initialization_exactly_preserves_existing_generation(tmp_path):
    root = 12345678
    prepared = SimpleNamespace(
        population_parity=True,
        run_dir=tmp_path,
        harness_config={
            "seed_archive_reference_policies": build.NAMES,
            "classic_gphh_population_size": 12,
        },
        initial_path=build.INITIAL,
        root_seed=root,
        seed_archive_random_mutants=5,
    )
    expected = read_sealed(materialize_seed_archive(prepared))
    actual = build.archive(root, "A")
    assert [r["source"] for r in actual["entries"]] == [r["source"] for r in expected["entries"]]


def test_B_adds_strong_reference_and_four_mutants():
    import ast

    a = build.archive(87654321, "B")
    assert len(a["entries"]) == 12 and a["random_ast_mutants"] == 4
    strong = a["entries"][7]
    assert strong["name"] == "bottleneck_headroom"

    def functions(text):
        return [ast.dump(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]

    assert functions(strong["source"]) == functions(build.STRONG.read_text())


def test_power_uses_ties_and_more_stringent_holm_threshold():
    assert power(20, 0.8, 0.2, 0.05) == pytest.approx(0.6375, abs=0.00005)
    assert power(30, 0.8, 0.2, 0.025) < power(30, 0.8, 0.2, 0.05)


def test_missing_freeze_cannot_pass_verification(tmp_path):
    with pytest.raises(FileNotFoundError):
        build.verify(tmp_path)


def test_frozen_bundle_detects_resealed_extra_and_modified_sources(tmp_path, monkeypatch):
    import json

    source = tmp_path / "external_source.py"
    source.write_text("original")
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "CHECKS.json").write_text(json.dumps({"tests": 1, "failures": 0, "errors": 0}))
    monkeypatch.setattr(build, "source_paths", lambda: {"source.py": source})
    monkeypatch.setattr(build, "evaluation_snapshot", lambda: {"packages": [("example", "1.0")]})
    build.freeze(out)
    monkeypatch.setattr(build, "evaluation_snapshot", lambda: {"packages": [("example", "2.0")]})
    with pytest.raises(ValueError, match="execution environment changed"):
        build.verify(out)
    monkeypatch.setattr(build, "evaluation_snapshot", lambda: {"packages": [("example", "1.0")]})
    extra = out / "unexpected.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="extra or missing"):
        build.verify(out)
    extra.unlink()
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        build.verify(out)
