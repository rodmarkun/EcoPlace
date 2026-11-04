"""GPHH trees must round-trip through their string expression (held-out bridge, item 3)."""

from __future__ import annotations

import random

from deap import gp

from baselines.gphh import generate_tree, gphh_bundle, primitive_set


def test_generated_trees_round_trip_through_from_string() -> None:
    pset = primitive_set()
    rng = random.Random(20260905)
    for _ in range(25):
        tree = generate_tree(pset, rng, min_depth=1, max_depth=4)
        expression = str(tree)
        rebuilt = gp.PrimitiveTree.from_string(expression, pset)
        assert str(rebuilt) == expression
        # Identity is derived from the expression, so a rebuilt tree is the same policy.
        assert gphh_bundle(rebuilt, pset).identity_sha256 == gphh_bundle(tree, pset).identity_sha256
