from __future__ import annotations

import asyncio
import random
from pathlib import Path

import placementscript as ps
import pytest
from shinka.core import ProposalOffspring
from shinka.database import Program

from evolution.llm_operator import _COMMON_REJECTIONS, OpenWeightLLMOperator
from evolution.operators import (
    SharedNumericInterleaveOperator,
    apply_scoped_diff,
    validate_scoped_edit,
)
from evolution.profiles import DEEP_SEARCH_PROFILE, evolution_profile

INITIAL = Path(__file__).parents[1] / "evolution" / "task" / "initial.py"
FIXTURES = Path(__file__).parent / "fixtures"


class NeverCalled:
    name = "never_called"

    def __call__(self, parent, inspirations, rng):
        del parent, inspirations, rng
        raise AssertionError("primary operator should not have been called")


def _program() -> Program:
    return Program(id="p", code=INITIAL.read_text(), language="python")


def test_common_rejections_pairs_cover_measured_failure_classes() -> None:
    pairs = (
        ("`x = node.power_idle_w + 1`", "`x = node.power_idle_w + 1.0`"),
        ("`move_pod(pod.id, target)`", "`move = move_pod(pod.id, target)`"),
        ("REJECTED (too deep)", "ACCEPTED (flat)"),
        (
            "`node.cpu_allocated(pod.cpu_request)`",
            "`cpu_after = node.cpu_allocated + pod.cpu_request`",
        ),
        ("an unfenced rewrite", "one complete policy inside"),
        ("missing, duplicate, unmatched, nested", "exactly one numbered BEGIN/END pair"),
    )
    for rejected, accepted in pairs:
        assert rejected in _COMMON_REJECTIONS
        assert accepted in _COMMON_REJECTIONS
    assert "`q = 3 / 2`" in _COMMON_REJECTIONS
    assert "`q = 3 // 2`" in _COMMON_REJECTIONS
    assert "`float(i)` converts" in _COMMON_REJECTIONS
    assert "an int only" in _COMMON_REJECTIONS
    assert "An annotation must match its value" in _COMMON_REJECTIONS


def test_deep_search_prompt_composes_common_rejections_and_exact_batch_contract() -> None:
    prompt = OpenWeightLLMOperator._prompt(
        _program(),
        (),
        "rewrite",
        profile=evolution_profile(DEEP_SEARCH_PROFILE),
        proposals_per_call=3,
    )

    assert _COMMON_REJECTIONS in prompt
    assert "Return exactly 3 proposals" in prompt
    assert "<<<EVOLVED_CONTINUUM_PROPOSAL:1:BEGIN>>>" in prompt
    assert "<<<EVOLVED_CONTINUUM_PROPOSAL:2:END>>>" in prompt
    assert "Delimiter text must never appear inside a code body" in prompt


def test_shared_numeric_interleave_is_identical_for_both_primary_arms():
    a = SharedNumericInterleaveOperator(NeverCalled(), probability=1.0)
    b = SharedNumericInterleaveOperator(NeverCalled(), probability=1.0)

    left = asyncio.run(a(_program(), (), random.Random(20260721)))
    right = asyncio.run(b(_program(), (), random.Random(20260721)))

    assert left.valid and right.valid
    assert left.source == right.source
    assert left.metadata["shared_numeric_seed"] == right.metadata["shared_numeric_seed"]
    assert left.metadata["shared_route_draw"] == right.metadata["shared_route_draw"]


def test_seeded_interleave_isolates_route_numeric_and_primary_streams():
    operator = SharedNumericInterleaveOperator(NeverCalled(), probability=1.0)
    result = asyncio.run(
        operator.propose_seeded(
            _program(),
            (),
            route_seed=11,
            numeric_seed=22,
            primary_seed=33,
        )
    )

    assert result.valid
    assert result.metadata["shared_route_seed"] == 11
    assert result.metadata["shared_numeric_seed"] == 22
    assert result.metadata["arm_primary_seed"] == 33


def test_scoped_edit_rejects_identity_or_marker_changes():
    parent = INITIAL.read_text()
    valid = parent.replace("hi_u - lo_u < 0.25", "hi_u - lo_u < 0.3")
    validate_scoped_edit(parent, valid)

    with pytest.raises(ValueError, match="immutable"):
        changed_identity = valid.replace(
            'POLICY_VERSION: str = "1.0.0"', 'POLICY_VERSION: str = "9"'
        )
        validate_scoped_edit(parent, changed_identity)
    with pytest.raises(ValueError, match="exactly one"):
        validate_scoped_edit(parent, valid.replace("# EVOLVE-BLOCK-END", ""))


@pytest.mark.parametrize(
    ("parent_path", "patch_name"),
    (
        (INITIAL, "llm_diff_archived_single.patch"),
        (
            FIXTURES / "llm_diff_archived_multihunk_parent.ps",
            "llm_diff_archived_multihunk.patch",
        ),
    ),
)
def test_archived_diff_failures_load_without_touching_immutable_bytes(
    parent_path: Path, patch_name: str
) -> None:
    """Regression fixtures are exact responses from the 2026-08-20 evidence set."""
    parent = parent_path.read_text()
    child = apply_scoped_diff(parent, (FIXTURES / patch_name).read_text())

    ps.Policy(child)
    validate_scoped_edit(parent, child)
    assert child != parent
    assert child.endswith("# EVOLVE-BLOCK-END\n")


def test_scoped_diff_handles_indented_context_and_rejects_outside_anchors() -> None:
    parent = INITIAL.read_text()
    child = apply_scoped_diff(
        parent,
        """<<<<<<< SEARCH
u = cpu_util(node)
=======
u = cpu_util(node) + 0.0
>>>>>>> REPLACE""",
    )
    ps.Policy(child)
    validate_scoped_edit(parent, child)

    with pytest.raises(ValueError, match="outside EVOLVE-BLOCK|not found"):
        apply_scoped_diff(
            parent,
            """<<<<<<< SEARCH
POLICY_VERSION: str = "1.0.0"
=======
POLICY_VERSION: str = "2.0.0"
>>>>>>> REPLACE""",
        )


def test_primary_route_preserves_cost_ledger_and_metadata():
    ledger = object()

    class Primary:
        name = "primary"
        cost_ledger = ledger

        def __call__(self, parent, inspirations, rng):
            del inspirations
            return ProposalOffspring(
                source=parent.code.replace("0.25", "0.3"),
                metadata={"draw": rng.randrange(100)},
            )

    operator = SharedNumericInterleaveOperator(Primary(), probability=0.0)
    result = asyncio.run(operator(_program(), (), random.Random(1)))

    assert result.valid
    assert operator.cost_ledger is ledger
    assert result.metadata["route"] == "primary"
