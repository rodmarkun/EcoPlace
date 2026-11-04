"""Classic GPHH must not bloat its trees past what Python can compile.

`baselines.gphh` was unbounded in two independent places, and the second one was the
one that actually bit.

**Generation.** `_generate_subtree` enforced `max_depth` only through its terminal
branch, and `bool` has no terminals in this primitive set — `and_`, `or_` and `not_` all
take booleans, so a boolean subtree could only stop by drawing `gt`/`lt`. That is a
branching process with no bound: a single generation-0 individual could be hundreds of
levels deep, and `gp.compile` then raised `SyntaxError: too many nested parentheses`.

**Variation.** Subtree mutation grafts a fresh subtree at an arbitrary point and subtree
crossover swaps arbitrary compatible subtrees, neither with a height check, so the
population could also bloat across generations.

That stayed hidden while the fixtures ran 65 s horizons and short searches. It surfaced
the first time the arm was given a population of 12 and a budget deep enough to reach
it — i.e. it would have surfaced during the confirmatory campaigns.
"""

from __future__ import annotations

import random

import pytest
from deap import gp

from baselines.gphh import (
    DEFAULT_MAX_TREE_HEIGHT,
    GPHHConfig,
    crossover_trees,
    generate_tree,
    mutate_tree,
    primitive_set,
)


def _deep_tree(pset, rng, height_target: int):
    """Grow a tree well past the cap by repeated unbounded-style mutation."""
    tree = generate_tree(pset, rng, min_depth=1, max_depth=4)
    for _ in range(400):
        if tree.height >= height_target:
            break
        # Mutate with a generous cap so the tree is free to grow for the fixture.
        tree = mutate_tree(tree, pset, rng, max_depth=4, max_height=10_000)
    return tree


def test_mutation_never_returns_a_tree_over_the_cap():
    pset = primitive_set()
    rng = random.Random(20260825)
    tree = generate_tree(pset, rng, min_depth=1, max_depth=4)
    for _ in range(300):
        tree = mutate_tree(tree, pset, rng, max_depth=4, max_height=DEFAULT_MAX_TREE_HEIGHT)
        assert tree.height <= DEFAULT_MAX_TREE_HEIGHT


def test_crossover_never_returns_a_tree_over_the_cap():
    pset = primitive_set()
    rng = random.Random(20260826)
    deep = _deep_tree(pset, rng, height_target=DEFAULT_MAX_TREE_HEIGHT + 6)
    shallow = generate_tree(pset, rng, min_depth=1, max_depth=2)
    assert deep.height > DEFAULT_MAX_TREE_HEIGHT, "fixture must actually be over-height"
    for _ in range(100):
        left, right = crossover_trees(
            shallow, deep, rng, max_height=DEFAULT_MAX_TREE_HEIGHT
        )
        # An over-height child is rejected in favour of its own parent, so the only way
        # to exceed the cap is to have started over it.
        assert left.height <= max(DEFAULT_MAX_TREE_HEIGHT, shallow.height)
        assert right.height <= max(DEFAULT_MAX_TREE_HEIGHT, deep.height)


def test_a_bounded_population_still_compiles():
    """The property that actually failed: every bounded tree must compile."""
    pset = primitive_set()
    rng = random.Random(20260827)
    tree = generate_tree(pset, rng, min_depth=1, max_depth=4)
    for _ in range(500):
        tree = mutate_tree(tree, pset, rng, max_depth=4, max_height=DEFAULT_MAX_TREE_HEIGHT)
    gp.compile(tree, pset)  # must not raise


def test_config_refuses_a_cap_below_its_own_initial_depth():
    GPHHConfig(max_depth=4, max_tree_height=4)
    with pytest.raises(ValueError, match="max_tree_height"):
        GPHHConfig(max_depth=4, max_tree_height=3)


def test_default_config_carries_the_cap():
    assert GPHHConfig().max_tree_height == DEFAULT_MAX_TREE_HEIGHT


def test_generation_terminates_boolean_subtrees():
    """`bool` has no terminals; past max_depth the draw must close on the next level.

    This is the defect that killed a campaign. Before the fix a single draw could produce
    a tree hundreds of levels deep purely from `and_`/`or_`/`not_` recursion.
    """
    pset = primitive_set()
    heights = [
        generate_tree(pset, random.Random(seed), min_depth=1, max_depth=4).height
        for seed in range(2000)
    ]
    assert max(heights) <= DEFAULT_MAX_TREE_HEIGHT
    # A depth-4 build should stay near depth 4, not merely under the safety cap.
    assert max(heights) <= 8


def test_every_generated_tree_compiles():
    pset = primitive_set()
    for seed in range(500):
        gp.compile(generate_tree(pset, random.Random(seed), min_depth=1, max_depth=4), pset)
