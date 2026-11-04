"""Resource exceptions must not silently change the original scientific request grid."""

from __future__ import annotations

import pytest

from experiments.gp_tuning_20260902.recovery import digest
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.validation_20260909 import run_workload_resource_amendment as controller


def test_amendment_cannot_be_created_after_any_matrix_request(tmp_path):
    (tmp_path / "scores").mkdir()
    with pytest.raises(ValueError, match="before any comparative"):
        controller.freeze(tmp_path)


def test_amendment_rejects_unplanned_cap_and_changed_source(tmp_path, monkeypatch):
    plan = {"maximum_total_worker_seconds": 14400}
    review = {"pilot": "frozen"}
    write_sealed(tmp_path / "PILOT_REVIEW.json", review)
    amended = {
        "plan_sha256": digest(plan),
        "pilot_review_sha256": digest(review),
        "superseded_total_worker_seconds": 14400,
        "maximum_total_worker_seconds": 28801,
        "files": {},
    }
    write_sealed(tmp_path / "RESOURCE_AMENDMENT.json", amended)
    with pytest.raises(ValueError, match="explicit resource scope"):
        controller.verify_amendment(tmp_path, plan)
    amended["maximum_total_worker_seconds"] = 28800
    amended["files"] = {"controller.py": "expected"}
    write_sealed(tmp_path / "RESOURCE_AMENDMENT.json", amended)
    monkeypatch.setattr(controller, "sha", lambda p: "changed")
    with pytest.raises(ValueError, match="source changed"):
        controller.verify_amendment(tmp_path, plan)
