"""Reproducibility checks for the Phase-5 ShinkaEvolve fork pin."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_shinkaevolve_patches_match_lock():
    lock = json.loads((REPO / "evolution" / "shinkaevolve.lock.json").read_text())
    assert lock["schema_version"] == 2
    assert len(lock["base_commit"]) == 40
    assert len(lock["fork_commit"]) == 40
    assert len(lock["base_tree"]) == 40
    assert len(lock["fork_tree"]) == 40
    assert len(lock["patches"]) == 2
    for item in lock["patches"]:
        patch = REPO / item["path"]
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == item["sha256"]
