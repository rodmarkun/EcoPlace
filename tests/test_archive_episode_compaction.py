"""Campaign archives must not inline horizon-scaled per-episode ledgers.

The 2026-08 artifact-size fix stopped *arm results* duplicating episode data. The
campaign archive still did, which stayed invisible while every fixture ran a 65 s
horizon. On the 420 s `dev-continuum-tight` episodes one archived candidate carried
22 MB and a 19-candidate arm reached **489 MB** — a full six-root matrix would have
filled the disk exactly as the 4 GB arm results did.

Each instance keeps its `episode_digest` and `certificate_seal`, so the episode stays
pinned; what is dropped is a copy of it.
"""

from __future__ import annotations

from evolution.campaign import CampaignRunner

compact = CampaignRunner._compact_episode_ledgers


def test_bulk_ledgers_are_replaced_by_their_length():
    payload = {
        "instances": [
            {
                "episode_digest": "a" * 64,
                "certificate_seal": "b" * 64,
                "energy_j": 1.5,
                "episode": {
                    "service_calls": [{"x": i} for i in range(1000)],
                    "records": [{"y": i} for i in range(50)],
                    "snapshots": [{"z": i} for i in range(12)],
                    "decision_log": [{"k": 1}],
                    "energy": {"device_j": 2.0},
                },
            }
        ]
    }
    out = compact(payload)
    episode = out["instances"][0]["episode"]
    assert episode["service_calls"] == {
        "omitted": "horizon-scaled episode ledger",
        "length": 1000,
        "pinned_by": "episode_digest + certificate_seal",
    }
    assert episode["records"]["length"] == 50
    assert episode["snapshots"]["length"] == 12
    # Small, bounded structures survive untouched.
    assert episode["decision_log"] == [{"k": 1}]
    assert episode["energy"] == {"device_j": 2.0}
    # The evidence that pins the omitted data is kept.
    assert out["instances"][0]["episode_digest"] == "a" * 64
    assert out["instances"][0]["certificate_seal"] == "b" * 64
    assert out["instances"][0]["energy_j"] == 1.5


def test_compaction_reaches_nested_ladder_traces():
    """L1/L2 rung results carry their own instances; both copies must be compacted."""
    payload = {
        "trace": [
            {
                "rung": "L2",
                "result": {
                    "evaluation": {
                        "instances": [{"episode": {"service_calls": [1, 2, 3]}}]
                    }
                },
            }
        ]
    }
    out = compact(payload)
    ledger = out["trace"][0]["result"]["evaluation"]["instances"][0]["episode"]["service_calls"]
    assert ledger["length"] == 3


def test_a_non_list_value_under_a_ledger_name_is_left_alone():
    """Only lists are ledgers; a scalar or dict of that name is someone else's field."""
    assert compact({"records": 7}) == {"records": 7}
    assert compact({"records": {"already": "compact"}}) == {"records": {"already": "compact"}}


def test_compaction_is_idempotent():
    once = compact({"episode": {"service_calls": list(range(5))}})
    assert compact(once) == once


def test_everything_else_passes_through_unchanged():
    payload = {"a": 1, "b": [1, 2, {"c": "d"}], "e": None, "f": 2.5, "g": True}
    assert compact(payload) == payload
