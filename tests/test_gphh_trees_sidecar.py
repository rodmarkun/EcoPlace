"""The classic_gphh trees sidecar must create its directory (regression for 2026-09-05)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from experiments.allbaselines_harness.runner import write_gphh_trees_sidecar


def test_sidecar_creates_missing_arm_directory(tmp_path) -> None:
    candidates = [
        SimpleNamespace(proposal_seq=i, generation=0, tree=f"add(x{i}, 1.0)") for i in range(3)
    ]
    target = tmp_path / "run" / "arms" / "classic_gphh" / "trees.json"
    assert not target.parent.exists()
    write_gphh_trees_sidecar(target, candidates)
    payload = json.loads(target.read_text())
    assert [t["expression"] for t in payload["trees"]] == [
        "add(x0, 1.0)",
        "add(x1, 1.0)",
        "add(x2, 1.0)",
    ]
    # idempotent: a second call leaves the sealed sidecar alone
    write_gphh_trees_sidecar(target, [])
    assert len(json.loads(target.read_text())["trees"]) == 3
