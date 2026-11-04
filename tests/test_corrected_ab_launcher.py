"""The registered A+B launcher's budget, selection, scheduling and final-grid decisions."""

import json
from datetime import UTC, datetime

import pytest

from experiments.corrected_ab_20260925 import launch


def layout(tmp_path):
    return launch.Layout(out=tmp_path, studies=("A",), roots=1, smoke=True)


def test_global_budget_counts_prior_reservations_and_settlements(tmp_path):
    budget = launch.GlobalBudget(layout(tmp_path), prior_microusd=59_900_000)
    budget.reserve("A00", "k1", 50_000)
    budget.reserve("A00", "k1", 50_000)  # idempotent across restart
    assert budget.exposure() == 59_950_000
    with pytest.raises(launch.GlobalBudgetExhausted):
        budget.reserve("B00", "k2", 50_001 + 50_000)
    budget.settle("A00", "k1", 7_000)
    assert budget.exposure() == 59_907_000
    budget.reserve("B00", "k2", 50_000)
    assert budget.exposure() == 59_957_000
    assert not issubclass(launch.GlobalBudgetExhausted, Exception)


def row(seq, sort_key, promoted=True, source=None):
    evaluation = {"promoted_to_l2": promoted,
                  "evaluation": {"fitness": {"sort_key": sort_key}} if promoted else None}
    return {"candidate_id": f"c{seq}", "proposal_seq": seq, "evaluation": evaluation,
            "proposal": {"source": source or f"src{seq}"}}


def test_selection_uses_only_full_records_and_sequence_tie_break():
    rows = [
        row(0, [1, 0.2, 0, 0, 0, 1.0, 0]),
        row(3, [0, 0.9, 3]),
        row(5, [0, 0.4, 5]),
        row(7, [0, 0.4, 7]),
        row(9, None, promoted=False),  # L1 reject or audit-only: never selectable
    ]
    full = launch.full_records(rows)
    assert [r["proposal_seq"] for r in full] == [0, 3, 5, 7]
    assert launch.best(full)["proposal_seq"] == 5
    assert launch.best([]) is None


def test_peak_windows_are_waited_out_not_skipped():
    assert launch.seconds_until_off_peak(datetime(2026, 9, 25, 2, 0, tzinfo=UTC)) > 0
    assert launch.seconds_until_off_peak(datetime(2026, 9, 25, 12, 0, tzinfo=UTC)) == 0
    assert launch.seconds_until_off_peak(datetime(2026, 9, 26, 2, 0, tzinfo=UTC)) == 0  # Sat


def test_final_grid_is_the_registered_untouched_grid_for_b():
    task = launch.final_task(launch.SCIENTIFIC, "B", 0)
    seeds = json.loads((launch.V3 / "SEEDS.json").read_text())["studies"]["B"][0]["grids"]
    assert len(task.scenarios) == 4
    for scenario in task.scenarios:
        assert list(scenario.seeds) == seeds[scenario.scenario_id]["final"]
        assert len(scenario.seeds) == 8
        assert scenario.seeds[0] not in seeds[scenario.scenario_id]["search"]
    assert len(task.bundle_ids) == 3


def test_nine_references_compile_in_the_evolve_shell():
    import placementscript as ps

    references = launch.reference_sources()
    assert len(references) == 9
    for source in references.values():
        ps.Policy(source)


def test_interrupted_evaluation_is_moved_aside_and_repeated(tmp_path, monkeypatch):
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    (evaluations / "013-abc.pickle").write_bytes(b"partial")
    (evaluations / "012-done.result").write_bytes(b"other candidate")
    calls = []
    monkeypatch.setattr(launch, "guarded_evaluate",
                        lambda d, *a, **k: calls.append(sorted(p.name for p in
                                                        (d / "evaluations").glob("013-*"))) or "ok")
    monkeypatch.setattr(launch, "ladder_episodes", lambda task, ladder: 1)
    assert launch.RecoveringEvaluator(tmp_path)("src", None, None, 13) == "ok"
    assert calls == [[]]
    assert (evaluations / "012-done.result").exists()
    moved = list(evaluations.glob("interrupted-013-*/013-abc.pickle"))
    assert len(moved) == 1
    assert "013-abc.pickle" in (tmp_path / "RECOVERIES.jsonl").read_text()


def test_registered_episode_timeout_scales_with_the_ladder():
    from evolution.evaluation import load_task_evaluation_config
    from evolution.ladder import load_ladder_config

    counts = {}
    for study in "AB":
        path = launch.V3 / f"task_{study}.json"
        counts[study] = launch.ladder_episodes(load_task_evaluation_config(path),
                                               load_ladder_config(path))
    assert counts == {"A": 1 + 1 + 3, "B": 1 + 4 + 12}
    assert launch.EXIT["resource"] == 50
    assert not issubclass(launch.EvaluationResourceStop, Exception)


def test_amendment_chain_must_link_to_the_freeze(tmp_path):
    from experiments.gp_tuning_20260902.recovery import digest

    layout = launch.Layout(out=tmp_path, smoke=True)
    freeze = {"payload": {"x": 1}, "payload_sha256": digest({"x": 1})}
    launch.save_once(launch.p2_path(layout), freeze)
    bad = {"previous_sha256": "0" * 64}
    launch.save_once(tmp_path / "P2_AMENDMENT_01.json",
                     {"payload": bad, "payload_sha256": digest(bad)})
    with pytest.raises((ValueError, KeyError)):
        launch.verify_p2(layout)


def test_supersede_refuses_after_finals_and_keeps_the_old_selection(tmp_path):
    layout = launch.Layout(out=tmp_path, smoke=True)
    root = layout.runs / "A09"
    root.mkdir(parents=True)
    (root / "SELECTION.json").write_text("{}")
    launch.supersede(layout, ["A09"], "test")
    assert not (root / "SELECTION.json").exists()
    assert (root / "SELECTION_superseded_amendment02.json").exists()
    other = layout.runs / "A10"
    (other / "final").mkdir(parents=True)
    with pytest.raises(ValueError):
        launch.supersede(layout, ["A10"], "test")


def test_connectivity_wait_returns_immediately_when_reachable(tmp_path, monkeypatch):
    import socket

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: Conn())
    assert launch.wait_for_connectivity(tmp_path) == 0


def test_journal_exposure_counts_every_settlement_type(tmp_path):
    from experiments.gp_tuning_20260902.recovery import canonical, digest

    rows = [
        {"type": "configuration", "config_sha256": "x", "schema": "s"},
        {"type": "reserved", "request_sha256": "a", "reservation_microusd": 50000},
        {"type": "settled_unknown", "request_sha256": "a",
         "settlement": {"cost_microusd": 50000, "reason": "r"}},
        {"type": "reserved", "request_sha256": "b", "reservation_microusd": 50000},
        {"type": "settled_http_error", "request_sha256": "b",
         "settlement": {"status": 429, "cost_microusd": 50000}},
        {"type": "reserved", "request_sha256": "c", "reservation_microusd": 50000},
    ]
    previous, lines = "0" * 64, []
    for seq, payload in enumerate(rows):
        body = {"seq": seq, "previous_sha256": previous, "payload": payload}
        record = {**body, "sha256": digest(body)}
        lines.append(canonical(record))
        previous = record["sha256"]
    (tmp_path / "j.jsonl").write_text("\n".join(lines) + "\n")
    assert launch.journal_exposure(tmp_path / "j.jsonl") == 150000


def test_memory_back_pressure_waits_until_memory_is_free(tmp_path, monkeypatch):
    import psutil

    readings = iter([1 * 2**30, 1 * 2**30, 4 * 2**30])

    class Memory:
        def __init__(self):
            self.available = next(readings)

    monkeypatch.setattr(psutil, "virtual_memory", Memory)
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    assert launch.wait_for_memory(2048, tmp_path / "w.jsonl", "test") == 20
    assert "memory_wait_s" in (tmp_path / "w.jsonl").read_text()
    assert launch.MIN_AVAILABLE_MIB == 3072 and launch.EVALUATION_START_MIB == 2048
