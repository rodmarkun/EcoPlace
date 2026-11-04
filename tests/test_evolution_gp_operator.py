"""Same-DSL grammar-guided GP proposer tests (TODO 5.2)."""

from __future__ import annotations

import random
from pathlib import Path

import placementscript as ps
import pytest
from shinka.database import Program

from evolution.gp_operator import (
    GPOperatorConfig,
    PlacementScriptGPOperator,
    load_gp_operator_config,
)

POLICIES = Path(__file__).resolve().parent.parent / "runtime" / "policies"


def program(name: str) -> Program:
    return Program(
        id=name,
        code=(POLICIES / f"{name}.ps").read_text(encoding="utf-8"),
        language="placementscript",
        correct=True,
    )


def test_gp_proposal_is_bitwise_reproducible():
    operator = PlacementScriptGPOperator()
    parent = program("least_allocated")
    inspirations = (program("energy_greedy"), program("latency_greedy"))
    a = operator(parent, inspirations, random.Random(20260721))
    b = operator(parent, inspirations, random.Random(20260721))
    assert a == b
    assert a.valid
    ps.Policy(a.source)


@pytest.mark.parametrize("seed", range(30))
def test_gp_always_returns_a_loadable_changed_bundle_for_seed_pool(seed):
    parent = program("least_allocated")
    result = PlacementScriptGPOperator()(
        parent,
        (program("most_allocated"), program("energy_greedy")),
        random.Random(seed),
    )
    assert result.valid, result.error
    assert result.source != parent.code
    ps.Policy(result.source)
    assert result.metadata["operator"] == "same_dsl_gp"
    assert result.metadata["attempts"] == result.metadata["retries"] + 1
    assert result.metadata["api_costs"] == 0.0


def test_numeric_mutation_preserves_tree_shape():
    config = GPOperatorConfig(
        crossover_probability=0.0,
        subtree_mutation_probability=0.0,
        numeric_mutation_probability=1.0,
    )
    parent = program("energy_greedy")
    result = PlacementScriptGPOperator(config)(parent, (), random.Random(4))
    assert result.valid
    assert result.metadata["operation"] == "numeric_mutation"
    assert len(ps.enumerate_subtrees(result.source)) == len(ps.enumerate_subtrees(parent.code))


def test_crossover_uses_inspiration_and_typed_sites():
    config = GPOperatorConfig(
        crossover_probability=1.0,
        subtree_mutation_probability=0.0,
        numeric_mutation_probability=0.0,
    )
    donor = program("most_allocated")
    result = PlacementScriptGPOperator(config)(
        program("least_allocated"), (donor,), random.Random(11)
    )
    assert result.valid, result.error
    assert result.metadata["operation"] == "crossover"
    assert result.metadata["detail"]["donor_sha256"]
    ps.Policy(result.source)


def test_invalid_parent_is_an_accounted_invalid_offspring():
    bad = Program(id="bad", code="def broken(", language="placementscript")
    result = PlacementScriptGPOperator()(bad, (), random.Random(1))
    assert not result.valid
    assert result.metadata["invalid_attempts"] == 1
    assert "invalid parent" in result.error


def test_rng_argument_is_required_and_typed():
    operator = PlacementScriptGPOperator()
    with pytest.raises(TypeError):
        operator(program("least_allocated"), ())
    with pytest.raises(TypeError, match="runner-owned"):
        operator(program("least_allocated"), (), object())


def test_configuration_rejects_uncontrolled_probabilities():
    with pytest.raises(ValueError, match="sum to 1"):
        GPOperatorConfig(
            crossover_probability=0.5,
            subtree_mutation_probability=0.5,
            numeric_mutation_probability=0.5,
        )


def test_load_gp_operator_config_round_trips_and_rejects_unknown_keys(tmp_path):
    path = tmp_path / "numeric-heavy.json"
    path.write_text(
        '{"label":"numeric-heavy","crossover_probability":0.2,'
        '"subtree_mutation_probability":0.2,"numeric_mutation_probability":0.6,'
        '"max_attempts":24}\n'
    )
    config, label = load_gp_operator_config(path)
    assert label == "numeric-heavy"
    assert config.crossover_probability == 0.2
    assert config.subtree_mutation_probability == 0.2
    assert config.numeric_mutation_probability == 0.6
    assert config.max_attempts == 24
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"crossover_probability":0.45,"subtree_mutation_probability":0.35,'
        '"numeric_mutation_probability":0.2,"max_attempts":24,"extra":1}\n'
    )
    with pytest.raises(ValueError, match="unknown"):
        load_gp_operator_config(bad)
