"""Reject changed/incomplete evidence and reap children after monitor failures."""

from __future__ import annotations

import pytest

from experiments.gp_tuning_20260902.recovery import canonical, digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.validation_20260909 import run_workload_sensitivity as runner


def test_early_terminal_policy_retained_without_claiming_completed_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "fitness_from_record",
        lambda raw: SimpleNamespace(proposal_seq=0, sort_key=lambda: (0, 0.6, 0)),
    )
    source = "fixed source\n"
    row = {
        "candidate_id": "selected",
        "proposal_seq": 0,
        "proposal": {"source": source},
        "evaluation": {"evaluation": {"fitness": {}}},
    }
    terminal = {
        "status": "unusable",
        "reason": "candidate-safety-ceiling-reached",
        "actual_microwork": 11600000,
        "registered_max_microwork": 56000000,
    }
    previous, lines = "0" * 64, []
    for index, (kind, payload) in enumerate(
        (("candidate_completed", row), ("campaign_terminal", terminal))
    ):
        event = {
            "event_seq": index,
            "previous_event_sha256": previous,
            "type": kind,
            "payload": payload,
        }
        event["event_sha256"] = digest(event)
        previous = event["event_sha256"]
        lines.append(canonical(event) + "\n")
    (tmp_path / "events.jsonl").write_text("".join(lines))
    (tmp_path / "candidates").mkdir()
    (tmp_path / "candidates/00000000.ps").write_text(source)
    found = runner.selected_source(tmp_path, {"candidate_id": "selected", "proposal_seq": 0})
    assert found["source"] == source
    assert found["terminal"] == terminal
    assert found["chain_head"] == previous


def test_monitor_exception_kills_and_reaps_owned_child(tmp_path, monkeypatch):
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
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: child)

    def denied(*args):
        raise PermissionError("monitor denied")

    monkeypatch.setattr(runner.psutil, "Process", denied)
    req = {"pilot": True}
    with pytest.raises(PermissionError):
        runner.guarded(tmp_path, req, {"worker_timeout_seconds": 1200}, 1200)
    assert child.killed and child.waited
    receipt = read_sealed(next((tmp_path / "pilot").glob("*.usage.json")))
    assert "PermissionError" in receipt["stop_reason"]
    with pytest.raises(ValueError, match="incomplete"):
        runner.guarded(tmp_path, req, {"worker_timeout_seconds": 1200}, 1200)


def test_completed_receipt_reuses_without_dispatch_and_rejects_changed_binding(
    tmp_path, monkeypatch
):
    req = {"pilot": False}
    folder = tmp_path / "scores"
    folder.mkdir()
    key = digest(req)
    write_sealed(folder / (key + ".request.json"), req)
    write_sealed(folder / (key + ".result.json"), {"request_sha256": key})
    usage = {"request_sha256": key, "exit_code": 0, "stop_reason": None}
    write_sealed(folder / (key + ".usage.json"), usage)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process")
    )
    assert runner.guarded(tmp_path, req, {}, 1200) == usage
    write_sealed(folder / (key + ".result.json"), {"request_sha256": "changed"})
    with pytest.raises(ValueError, match="not a complete"):
        runner.guarded(tmp_path, req, {}, 1200)


def test_selection_rejects_tampered_chain_before_using_source(tmp_path):
    event = {
        "event_seq": 0,
        "previous_event_sha256": "0" * 64,
        "type": "campaign_terminal",
        "payload": {"status": "complete"},
    }
    event["event_sha256"] = digest(event)
    event["payload"]["status"] = "tampered"
    (tmp_path / "events.jsonl").write_text(canonical(event) + "\n")
    with pytest.raises(ValueError, match="chain failure"):
        runner.selected_source(tmp_path, {"candidate_id": "x", "proposal_seq": 0})


def test_exact_source_hash_includes_comments(tmp_path):
    path = tmp_path / "policy.ps"
    path.write_text("body\n")
    before = runner.sha(path)
    path.write_text("# provenance\nbody\n")
    assert runner.sha(path) != before


def test_summary_missing_cells_is_not_complete(tmp_path):
    from experiments.validation_20260909.summarize_workload_sensitivity import summarize

    with pytest.raises(FileNotFoundError):
        summarize(tmp_path)


def test_comparison_keeps_fault_denominators_and_conditional_energy():
    from experiments.validation_20260909.summarize_workload_sensitivity import comparison

    good = {"feasible": True, "energy": 0.6}
    better = {"feasible": True, "energy": 0.5}
    bad = {"feasible": False, "energy": None}
    result = comparison([(good, better), (bad, good), (bad, bad), (good, good)])
    assert result["n"] == 4
    assert (result["arm_wins"], result["comparator_wins"], result["ties"]) == (0, 2, 2)
    assert result["jointly_feasible"] == 2
    assert result["mean_comparator_minus_arm_energy"] == pytest.approx(-0.05)
    assert comparison([(bad, bad)])["mean_comparator_minus_arm_energy"] is None
