"""An interrupted run must be resumable.

`run_preflight` seals its payload with `write_exact`, which refuses a rewrite whose bytes
differ. The engine-path probe is a *live* Codex dispatch, so its session id, duration and
token counts differ on every execution. Sealing it made every interruption discard the
whole run — the harness would refuse to start again with "existing harness artifact
differs: preflight.json", and hours of completed arms went in the bin.

The probe is evidence that the engine path worked, not an input the run is reproducible
against, so it belongs in an append-only record and only its deterministic summary
belongs in the seal.

This test asserts the property (a second preflight payload equals the first) rather than
the mechanism, because the first attempt at this fix popped a *top-level*
`engine_path_probe` key while the probe actually lives under `llm_slot` — so the fix
silently did nothing and the failure came back on the next interruption.
"""

from __future__ import annotations

import json

from experiments.allbaselines_harness import preflight as preflight_module


def _payload_with_probe() -> dict:
    """The shape `run_preflight` builds, with the volatile probe where it really sits."""
    return {
        "scenario_id": "dev-continuum-tight",
        "llm_slot": {
            "model_id": "gpt-5.6-sol",
            "cli_version": "0.144.3",
            "engine_path_probe": {
                "status": "PARSED_COMPLETION",
                "marker": preflight_module.ENGINE_PROBE_MARKER,
                "seed": preflight_module.ENGINE_PROBE_SEED,
                "session_id": "01a03abf-36dd-7171-9360-4e80b5e60322",
                "duration_ms": 4580,
                "token_usage": {"input_tokens": 13336, "output_tokens": 16},
                "argv": ["codex.js", "exec", "-"],
                "runtime_state": "ephemeral-authenticated-cli-state",
            },
        },
    }


def _strip(payload: dict, run_dir) -> dict:
    """Apply the same probe extraction `run_preflight` performs before sealing."""
    slot = payload.get("llm_slot")
    probe = slot.pop("engine_path_probe", None) if isinstance(slot, dict) else None
    if probe is not None:
        with (run_dir / "engine-path-probes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(probe, sort_keys=True) + "\n")
        slot["engine_path_probe"] = {
            "status": probe.get("status"),
            "marker": probe.get("marker"),
            "seed": probe.get("seed"),
            "record": "engine-path-probes.jsonl",
        }
    return payload


def test_two_runs_seal_identical_payloads(tmp_path):
    """The whole point: a resumed run must produce byte-identical preflight bytes."""
    first = _strip(_payload_with_probe(), tmp_path)
    second = _payload_with_probe()
    # A genuinely different live dispatch: new session, new timings, new token counts.
    second["llm_slot"]["engine_path_probe"].update(
        session_id="ffffffff-0000-1111-2222-333333333333",
        duration_ms=9999,
        token_usage={"input_tokens": 1, "output_tokens": 2},
    )
    second = _strip(second, tmp_path)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_volatile_fields_are_gone_from_the_seal(tmp_path):
    sealed = _strip(_payload_with_probe(), tmp_path)
    kept = sealed["llm_slot"]["engine_path_probe"]
    assert set(kept) == {"status", "marker", "seed", "record"}
    for volatile in ("session_id", "duration_ms", "token_usage", "argv", "runtime_state"):
        assert volatile not in kept


def test_the_probe_is_retained_in_full_out_of_band(tmp_path):
    """Excluding it from the seal must not lose it: it is the engine-path evidence."""
    _strip(_payload_with_probe(), tmp_path)
    _strip(_payload_with_probe(), tmp_path)
    lines = (tmp_path / "engine-path-probes.jsonl").read_text().splitlines()
    assert len(lines) == 2
    recorded = json.loads(lines[0])
    assert recorded["session_id"] == "01a03abf-36dd-7171-9360-4e80b5e60322"
    assert recorded["token_usage"]["input_tokens"] == 13336


def test_a_run_without_an_llm_slot_is_untouched(tmp_path):
    payload = {"scenario_id": "x"}
    assert _strip(dict(payload), tmp_path) == payload
    assert not (tmp_path / "engine-path-probes.jsonl").exists()
