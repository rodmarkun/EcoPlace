"""Resource amendments preserve original identity and reject widened/unbound execution."""

from contextlib import nullcontext

import pytest

from experiments.gp_tuning_20260902.recovery import digest
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.validation_20260909 import run_independent_resource_amendment as controller


def test_freeze_refuses_any_calibration_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.runner, "shared_lock", lambda _: nullcontext())
    (tmp_path / "calibration").mkdir()
    with pytest.raises(ValueError, match="before any calibration"):
        controller.freeze(tmp_path)


@pytest.mark.parametrize("corruption", ["cap", "plan", "missing_source", "source"])
def test_amendment_rejects_resealed_scope_or_source_changes(tmp_path, monkeypatch, corruption):
    plan = {"calibration_total_seconds": 21600}
    review = {"fits_cap": False}
    write_sealed(tmp_path / "PILOT_REVIEW.json", review)
    a = {
        "plan_sha256": digest(plan),
        "pilot_review_sha256": digest(review),
        "superseded_total_worker_seconds": 21600,
        "maximum_total_worker_seconds": 43200,
        "files": {
            str(p.relative_to(controller.runner.ROOT)): "expected" for p in controller.SOURCE_PATHS
        },
    }
    if corruption == "cap":
        a["maximum_total_worker_seconds"] += 1
    elif corruption == "plan":
        a["plan_sha256"] = "wrong"
    elif corruption == "missing_source":
        a["files"] = {}
    monkeypatch.setattr(controller.runner, "sha", lambda _: "changed")
    write_sealed(tmp_path / "RESOURCE_AMENDMENT.json", a)
    with pytest.raises(ValueError, match="scope|source changed"):
        controller.verify_amendment(tmp_path, plan)


def test_complete_population_and_total_cap_required():
    plan = {"calibration_jobs": list(range(972))}
    usage = [{"elapsed_seconds": 1, "peak_rss_mib": 100}] * 972
    result = controller.completion(plan, {"amendment": True}, usage)
    assert result["requests"] == 972 and result["total_worker_seconds"] == 972
    with pytest.raises(ValueError, match="incomplete"):
        controller.completion(plan, {}, usage[:-1])
    with pytest.raises(ValueError, match="cap exceeded"):
        controller.completion(plan, {}, [{"elapsed_seconds": 45, "peak_rss_mib": 100}] * 972)


def test_run_preserves_requests_and_charges_cached_work(tmp_path, monkeypatch):
    plan = {"calibration_jobs": [{"id": 1}, {"id": 2}]}
    amendment = {"cap": 43200}
    monkeypatch.setattr(controller.runner, "shared_lock", lambda _: nullcontext())
    monkeypatch.setattr(controller, "original", lambda _: (plan, {}))
    monkeypatch.setattr(controller, "verify_amendment", lambda *_: amendment)
    monkeypatch.setattr(controller.runner, "verify_source", lambda _: plan)
    monkeypatch.setattr(controller.runner, "request", lambda p, j: {"plan": p, "job": j})
    seen = []

    def guard(out, req, p, remaining):
        assert p is plan and req["plan"] is plan
        seen.append((req["job"], remaining))
        return {"elapsed_seconds": 30, "peak_rss_mib": 100}

    monkeypatch.setattr(controller.runner, "guarded", guard)
    controller.run(tmp_path)
    assert seen == [(plan["calibration_jobs"][0], 43200), (plan["calibration_jobs"][1], 43170)]
    with pytest.raises(ValueError, match="already complete"):
        controller.run(tmp_path)
