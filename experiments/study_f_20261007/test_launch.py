"""Unit tests for the Study F quota path and selection helpers (no live calls, no simulation)."""

import json

import pytest

from evolution.codex_engine import CodexCLICompletionClient, CodexCLICompletionError
from experiments.study_f_20261007 import launch as F


def refusal(message):
    return CodexCLICompletionError("remote-error", message,
                                   archive_metadata={"raw_output_excerpt": message})


@pytest.mark.parametrize("message", [
    "You've hit your usage limit. Try again at 14:05.",
    "stream error: 429 Too Many Requests",
    "Rate limit reached for gpt-6.1-sol",
])
def test_quota_refusal_stops_arm_with_account_exit(tmp_path, monkeypatch, message):
    client = object.__new__(F.SolClient)
    client.pauses = tmp_path / "PAUSES.jsonl"

    def fail(self, **kwargs):
        raise refusal(message)

    monkeypatch.setattr(CodexCLICompletionClient, "complete", fail)
    with pytest.raises(F.QuotaStop) as stop:
        client.complete(model="m", prompt="p", generation={}, seed=1)
    assert stop.value.code == F.EXIT["account"]
    assert json.loads(client.pauses.read_text())["failure"] == "subscription-quota"


def test_model_failure_is_a_treatment_outcome(tmp_path, monkeypatch):
    client = object.__new__(F.SolClient)
    client.pauses = tmp_path / "PAUSES.jsonl"

    def fail(self, **kwargs):
        raise refusal("PlacementScript output could not be parsed")

    monkeypatch.setattr(CodexCLICompletionClient, "complete", fail)
    with pytest.raises(CodexCLICompletionError):
        client.complete(model="m", prompt="p", generation={}, seed=1)
    assert not client.pauses.exists()


def test_set_aside_moves_arm_and_pair(tmp_path):
    layout = F.Layout(out=tmp_path, smoke=True)
    launcher = F.Launcher(layout)
    root = F.f_root(layout, "T", 0)
    (root / "llm").mkdir(parents=True)
    (root / "pair.json").write_text("{}")
    launcher.set_aside("T", 0, 0)
    assert not (root / "llm").exists() and not (root / "pair.json").exists()
    assert (root / "superseded_quota_0/llm").is_dir()
    assert (root / "superseded_quota_0/pair.json").exists()


def rec(seq, key, source):
    return {"candidate_id": f"c{seq}", "proposal_seq": seq,
            "proposal": {"source": source},
            "evaluation": {"evaluation": {"fitness": {"sort_key": key}}}}


def test_top_k_distinct_in_search_order():
    rows = [rec(3, [0, 1.0], "a"), rec(1, [0, 0.5], "b"), rec(2, [0, 0.5], "a"),
            rec(4, [1, 0.1], "c")]
    picks = F.top_k(rows, 2)
    assert [p["proposal_seq"] for p in picks] == [1, 2]
    assert len({p["source_sha256"] for p in picks}) == 2


def test_validation_key_ranks_invalid_last_then_search_order():
    cand = {"search_fitness": {"sort_key": [0, 1.0]}, "proposal_seq": 5}
    good = {"valid": True, "trace": {"fitness_sort_key": [0, 2.0, 0]}}
    bad = {"valid": False, "error": "x"}
    assert F.validation_key(good, cand) < F.validation_key(bad, cand)


def test_f01_stop_tolerates_exited_unreaped_leader():
    import subprocess
    import time

    from evolution import codex_engine

    process = subprocess.Popen(["sh", "-c", "exit 0"], start_new_session=True)
    time.sleep(0.5)  # leader exited but not reaped: macOS killpg raises EPERM here
    codex_engine._stop_process_group(process)  # patched by the Study F launcher
    assert process.returncode == 0


def test_f02_refused_killpg_stops_a_live_leader_directly(monkeypatch):
    import subprocess

    from evolution import codex_engine

    process = subprocess.Popen(["sleep", "30"], start_new_session=True)

    def denied(process):
        raise PermissionError("denied")

    monkeypatch.setattr(F, "_original_stop_process_group", denied)
    try:
        codex_engine._stop_process_group(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_f02_raises_when_leader_cannot_be_stopped(monkeypatch):
    import subprocess

    from evolution import codex_engine

    class Stuck:
        def poll(self):
            return None

        def terminate(self):
            raise PermissionError("denied")

        kill = terminate

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("x", timeout)

    def denied(process):
        raise PermissionError("denied")

    monkeypatch.setattr(F, "_original_stop_process_group", denied)
    with pytest.raises(PermissionError):
        codex_engine._stop_process_group(Stuck())


def test_f01_host_failure_stops_arm_with_harness_exit(tmp_path, monkeypatch):
    client = object.__new__(F.SolClient)
    client.pauses = tmp_path / "PAUSES.jsonl"

    def fail(self, **kwargs):
        raise PermissionError("[Errno 1] Operation not permitted")

    monkeypatch.setattr(CodexCLICompletionClient, "complete", fail)
    with pytest.raises(F.HarnessStop) as stop:
        client.complete(model="m", prompt="p", generation={}, seed=1)
    assert stop.value.code == F.EXIT["harness"] == 70
    assert json.loads(client.pauses.read_text())["failure"] == "harness"


def test_set_aside_harness_kind(tmp_path):
    layout = F.Layout(out=tmp_path, smoke=True)
    root = F.f_root(layout, "T", 0)
    (root / "llm").mkdir(parents=True)
    F.Launcher(layout).set_aside("T", 0, 2, kind="harness")
    assert (root / "superseded_harness_2/llm").is_dir()
