"""Calibration controller tests with synthetic results; never launch the simulator."""

import copy
import json
import subprocess
import sys

import pytest

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.validation_20260909 import run_independent_calibration as runner


@pytest.fixture(scope="module")
def design():
    return json.loads((runner.design_module.OUT / "DESIGN.json").read_text())


@pytest.fixture
def plan(design):
    return {
        "design": design,
        "pilot_jobs": runner.jobs(design, "pilot"),
        "calibration_jobs": runner.jobs(design, "calibration"),
        "rss_limit_mib": 2048,
        "worker_timeout_seconds": 1200,
        "pilot_total_seconds": 3600,
        "calibration_total_seconds": 21600,
        "projection_reserve_factor": 1.30,
    }


def fake_result(req):
    j = req["job"]
    return {
        "request_sha256": digest(req),
        "effective_config_sha256": j["effective_config_sha256"],
        "status": "evaluated",
        "evaluation": {
            "fitness": {"n_instances": 1, "n_bundles": 1, "proposal_seq": 0, "feasible": True},
            "policy": {"artifact_sha256": j["source_sha256"]},
            "instances": [
                {"scenario_id": j["cell_id"], "seed": j["seed"], "bundle_id": j["bundle_id"]}
            ],
        },
    }


def test_full_grid_and_pilot_separation(plan):
    calibration, pilot = plan["calibration_jobs"], plan["pilot_jobs"]
    assert len(calibration) == len({digest(j) for j in calibration}) == 972
    assert len(pilot) == 3
    assert {j["label"] for j in pilot} == {"graph_headroom"}
    assert {j["cell_id"] for j in pilot} == {"edge_heavy-two_bursts-0.45"}
    assert {j["bundle_id"] for j in pilot} == {"dev-1", "dev-2", "dev-3"}
    assert not {j["seed"] for j in pilot} & {j["seed"] for j in calibration}
    assert len({j["cell_id"] for j in calibration}) == 12
    assert len({j["label"] for j in calibration}) == 9
    changed = {**calibration[0], "seed": -1}
    with pytest.raises(ValueError, match="outside"):
        runner.request(plan, changed)


def test_projection_reserve_and_pilot_total_cap(plan):
    review = runner.pilot_review(plan, [{"elapsed_seconds": 20.0}] * 3)
    assert review["linear_projection_seconds"] == 19440
    assert review["with_reserve_seconds"] == 25272
    assert review["fits_cap"] is False
    assert runner.pilot_review(plan, [{"elapsed_seconds": 10.0}] * 3)["fits_cap"] is True
    for durations in ([10, 10], [0, 10, 10], [float("nan"), 10, 10], [1201, 1201, 1201]):
        with pytest.raises(ValueError):
            runner.pilot_review(plan, [{"elapsed_seconds": d} for d in durations])


@pytest.mark.parametrize("change", ["seed", "source", "config", "request", "count", "status"])
def test_result_identity_mismatches_refused(plan, change):
    req = runner.request(plan, plan["pilot_jobs"][0])
    result = fake_result(req)
    runner.validate_result(req, result)
    if change == "seed":
        result["evaluation"]["instances"][0]["seed"] += 1
    if change == "source":
        result["evaluation"]["policy"]["artifact_sha256"] = "0" * 64
    if change == "config":
        result["effective_config_sha256"] = "0" * 64
    if change == "request":
        result["request_sha256"] = "0" * 64
    if change == "count":
        result["evaluation"]["fitness"]["n_instances"] = 3
    if change == "status":
        result["status"] = "resource-failure-as-infeasible"
    with pytest.raises(ValueError):
        runner.validate_result(req, result)


def test_fault_is_distinct_from_measured_infeasibility(plan):
    req = runner.request(plan, plan["pilot_jobs"][0])
    j = req["job"]
    result = {
        "request_sha256": digest(req),
        "effective_config_sha256": j["effective_config_sha256"],
        "status": "policy-fault",
        "failure": {
            "scenario_id": j["cell_id"],
            "seed": j["seed"],
            "bundle_id": j["bundle_id"],
            "message": "fuel",
        },
    }
    runner.validate_result(req, result)
    assert "evaluation" not in result
    result["failure"]["seed"] += 1
    with pytest.raises(ValueError):
        runner.validate_result(req, result)


def test_completed_cache_reuse_and_result_tamper(tmp_path, monkeypatch, plan):
    req = runner.request(plan, plan["pilot_jobs"][0])
    key = digest(req)
    folder = tmp_path / "pilot"
    folder.mkdir()
    result = fake_result(req)
    started = folder / f"{key}.started.json"
    result["offered_trace"] = runner.write_offered_trace(folder / f"{key}.records.jsonl.gz", [])
    runner.store_once(started, {"request_sha256": key, "pid": 1, "started_at_utc": "test-only"})
    usage = {
        "request_sha256": key,
        "result_sha256": digest(result),
        "exit_code": 0,
        "stop_reason": None,
        "elapsed_seconds": 10.0,
        "peak_rss_mib": 100.0,
        "timeout_seconds": 1200,
        "started_file_sha256": runner.sha(started),
    }
    for suffix, payload in [("request", req), ("result", result), ("usage", usage)]:
        runner.store_once(folder / f"{key}.{suffix}.json", payload)
    (folder / f"{key}.log").write_text("completed")
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("cache dispatched a worker")
    )
    assert runner.guarded(tmp_path, req, plan, 1200) == usage
    result["evaluation"]["fitness"]["feasible"] = False
    # Re-sealing one result cannot silently bypass its independent usage binding.
    (folder / f"{key}.result.json").unlink()
    runner.store_once(folder / f"{key}.result.json", result)
    with pytest.raises(ValueError, match="successful"):
        runner.guarded(tmp_path, req, plan, 1200)


def test_monitor_failure_reaps_owned_child_and_prevents_retry(tmp_path, monkeypatch, plan):
    class Child:
        pid = 12345
        killed = False
        waited = False

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self):
            self.waited = True
            return -9

    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)

    def denied(*a):
        raise PermissionError("monitor denied")

    monkeypatch.setattr(runner.psutil, "Process", denied)
    req = runner.request(plan, plan["pilot_jobs"][0])
    with pytest.raises(PermissionError):
        runner.guarded(tmp_path, req, plan, 1200)
    assert child.killed and child.waited
    usage = read_sealed(next((tmp_path / "pilot").glob("*.usage.json")))
    assert usage["exit_code"] == -9 and "PermissionError" in usage["stop_reason"]
    with pytest.raises(ValueError, match="incomplete"):
        runner.guarded(tmp_path, req, plan, 1200)


def test_exhausted_budget_and_exclusive_evidence(tmp_path, monkeypatch, plan):
    req = runner.request(plan, plan["pilot_jobs"][0])
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("bad budget dispatched")
    )
    for remaining in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="budget"):
            runner.guarded(tmp_path, req, plan, remaining)
    assert not list(tmp_path.rglob("*.request.json"))
    path = tmp_path / "receipt.json"
    runner.store_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        runner.store_once(path, {"original": False})
    assert read_sealed(path) == {"original": True}


def test_shared_lock_prevents_second_controller(tmp_path):
    path = tmp_path / "execution.lock"
    with runner.shared_lock(path):
        with pytest.raises(BlockingIOError):
            with runner.shared_lock(path):
                pytest.fail("second controller acquired shared lock")


def test_worker_uses_exact_single_episode_and_refuses_second_dispatch(tmp_path, monkeypatch, plan):
    req = runner.request(plan, plan["pilot_jobs"][0])
    key = digest(req)
    runner.store_once(tmp_path / "pilot" / f"{key}.request.json", req)
    policy = next(
        p for p in plan["design"]["fixed_references"] if p["label"] == req["job"]["label"]
    )
    folder = tmp_path / "policies"
    folder.mkdir()
    (folder / (policy["source_sha256"] + ".ps")).write_bytes(
        (runner.ROOT / policy["path"]).read_bytes()
    )
    monkeypatch.setattr(runner, "verify_source", lambda out: plan)
    called = []

    def evaluate(source, task, **kwargs):
        called.append((task, kwargs))

        class Evaluation:
            from types import SimpleNamespace

            instances = [SimpleNamespace(episode={"records": []})]

            def without_episode_ledgers(self):
                return self

            def to_dict(self):
                return copy.deepcopy(fake_result(req)["evaluation"])

        return Evaluation()

    monkeypatch.setattr(runner, "evaluate_policy_source", evaluate)
    runner.worker(tmp_path, key, "pilot")
    assert len(called) == 1
    task, kwargs = called[0]
    assert kwargs == {"proposal_seq": 0, "require_complete_development_set": False}
    assert task.bundle_ids == (req["job"]["bundle_id"],)
    assert task.scenarios[0].seeds == (req["job"]["seed"],)
    assert task.constraints.goodput_target == 0.95
    assert task.scenarios[0].path.name == req["job"]["cell_id"] + ".json"
    with pytest.raises(FileExistsError):
        runner.worker(tmp_path, key, "pilot")
    assert len(called) == 1


def test_worker_key_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        runner.worker(tmp_path, "../private", "pilot")


@pytest.mark.parametrize("limit", ["time", "memory"])
def test_real_dummy_child_is_terminated_and_not_retried(tmp_path, monkeypatch, plan, limit):
    real_popen = subprocess.Popen
    children = []

    def spawn_dummy(command, **kwargs):
        assert command[2] == "experiments.validation_20260909.run_independent_calibration"
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", spawn_dummy)
    bounded = {**plan, "rss_limit_mib": 1 if limit == "memory" else 2048}
    remaining = 0.05 if limit == "time" else 1200
    req = runner.request(bounded, bounded["pilot_jobs"][0])
    with pytest.raises(RuntimeError, match="cap exceeded"):
        runner.guarded(tmp_path, req, bounded, remaining)
    assert len(children) == 1 and children[0].poll() is not None
    usage = read_sealed(next((tmp_path / "pilot").glob("*.usage.json")))
    assert usage["exit_code"] != 0 and usage["stop_reason"] is not None
    with pytest.raises(ValueError, match="incomplete"):
        runner.guarded(tmp_path, req, bounded, 1200)
    assert len(children) == 1


def test_real_dummy_completion_binds_result_and_reuses(tmp_path, monkeypatch, plan):
    real_popen = subprocess.Popen
    req = runner.request(plan, plan["pilot_jobs"][0])
    key = digest(req)
    result = fake_result(req)
    trace_template = tmp_path / f"{key}.records.jsonl.gz"
    result["offered_trace"] = runner.write_offered_trace(trace_template, [])
    envelope = json.dumps({"payload": result, "payload_sha256": digest(result)})
    result_path = tmp_path / "pilot" / f"{key}.result.json"
    started_path = tmp_path / "pilot" / f"{key}.started.json"
    started = {"request_sha256": key, "pid": 1, "started_at_utc": "dummy-test-only"}
    started_envelope = json.dumps({"payload": started, "payload_sha256": digest(started)})
    children = []

    def spawn_dummy(command, **kwargs):
        (tmp_path / "pilot" / trace_template.name).write_bytes(trace_template.read_bytes())
        child = real_popen(
            [
                sys.executable,
                "-c",
                "import pathlib,sys,time; "
                "pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
                "pathlib.Path(sys.argv[3]).write_text(sys.argv[4]); time.sleep(.3)",
                str(result_path),
                envelope,
                str(started_path),
                started_envelope,
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", spawn_dummy)
    usage = runner.guarded(tmp_path, req, plan, 1200)
    assert usage["result_sha256"] == digest(result)
    assert usage["exit_code"] == 0 and usage["peak_rss_mib"] > 0
    assert runner.guarded(tmp_path, req, plan, 1200) == usage
    assert len(children) == 1 and children[0].poll() == 0


def test_freeze_stops_before_artifact_write_if_loader_repair_missing(tmp_path, monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(runner, "shared_lock", lambda path: nullcontext())
    monkeypatch.setattr(runner, "prerequisites", lambda: {})

    def broken_loader(raw):
        raise ValueError("typed-record loader repair missing")

    monkeypatch.setattr(runner, "episode_config_from_json", broken_loader)
    out = tmp_path / "unstarted"
    with pytest.raises(ValueError, match="repair missing"):
        runner.freeze(out)
    assert not out.exists()


def test_frozen_archive_round_trip_and_tamper_refusal(tmp_path, monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(runner, "shared_lock", lambda path: nullcontext())
    monkeypatch.setattr(runner, "prerequisites", lambda: {})
    out = tmp_path / "test-only-freeze"
    runner.freeze(out)
    plan = runner.verify_source(out)
    frozen = read_sealed(out / "snapshot.json")
    runner.verify_archive(out, frozen, plan["source_archive_sha256"])
    assert len(plan["calibration_jobs"]) == 972
    with pytest.raises(ValueError, match="new directory"):
        runner.freeze(out)
    with (out / "sources.tar.gz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="archive bytes"):
        runner.verify_archive(out, frozen, plan["source_archive_sha256"])


def test_prior_completion_rejects_duplicate_tail_replicates(tmp_path, monkeypatch):
    matrix, tails = tmp_path / "matrix", tmp_path / "tails"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MATRIX", matrix)
    monkeypatch.setattr(runner, "TAILS", tails)
    matrix_plan, amendment = {"test": "plan"}, {"test": "amendment"}
    jobs = [{"rate": 8.0, "seed": i, "horizon": 5000.0} for i in range(136)]
    tail_plan = {"matrix_plan_sha256": digest(matrix_plan), "jobs": jobs}
    complete = {
        "plan_sha256": digest(matrix_plan),
        "requests": 174,
        "resource_amendment_sha256": digest(amendment),
    }
    result = {"plan_sha256": digest(tail_plan), "replicates": jobs}
    for path, value in [
        (matrix / "PLAN.json", matrix_plan),
        (matrix / "COMPLETE.json", complete),
        (matrix / "RESOURCE_AMENDMENT.json", amendment),
        (tails / "PLAN.json", tail_plan),
        (tails / "RESULTS.json", result),
    ]:
        runner.store_once(path, value)
    assert len(runner.prerequisites()) == 5
    result["replicates"] = [jobs[0]] * 136
    (tails / "RESULTS.json").unlink()
    runner.store_once(tails / "RESULTS.json", result)
    with pytest.raises(ValueError, match="completion receipts"):
        runner.prerequisites()
