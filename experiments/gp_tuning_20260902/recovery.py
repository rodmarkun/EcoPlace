"""Read completed GP arms without requiring completion of unrelated baselines.

Never instantiate EventArchive here: its recovery constructor may modify evidence.
This reader verifies existing bytes and preserves the original selection semantics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evolution.campaign import EventArchive
from experiments.allbaselines_harness.summary import fitness_from_record


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_sealed(path: Path) -> dict:
    wrapped = json.loads(path.read_text())
    if set(wrapped) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid sealed envelope: {path}")
    if digest(wrapped["payload"]) != wrapped["payload_sha256"]:
        raise ValueError(f"payload seal mismatch: {path}")
    return wrapped["payload"]


def read_events(path: Path) -> list[dict]:
    events = []
    previous = "0" * 64
    with path.open() as stream:
        for index, line in enumerate(stream):
            event = json.loads(line)
            if line != canonical(event) + "\n":
                raise ValueError(f"noncanonical or partial event: {path}:{index}")
            expected = event["event_sha256"]
            body = {k: v for k, v in event.items() if k != "event_sha256"}
            if (
                digest(body) != expected
                or event["event_seq"] != index
                or event["previous_event_sha256"] != previous
            ):
                raise ValueError(f"event chain mismatch: {path}:{index}")
            events.append(event)
            previous = expected
    EventArchive._validate_events(events)
    if not events or events[-1]["type"] != "campaign_terminal":
        raise ValueError(f"archive has no final terminal event: {path}")
    return events


def recover_gp(run_dir: Path) -> dict:
    """Return a verified arm receipt; never imply whole-run/confirmatory completion."""
    archive = run_dir / "archives/same_dsl_gp"
    preflight = read_sealed(run_dir / "preflight.json")
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    campaign = manifest["campaign"]
    if campaign["arm"] != "gp" or campaign["root_seed"] != preflight["root_seed"]:
        raise ValueError("GP manifest identity differs from preflight")
    budget = preflight["matched_max_microwork"]
    if campaign["evaluator_work"]["max_microwork"] != budget:
        raise ValueError("GP manifest budget differs from preflight")
    for filename, key in (
        ("campaign_config.json", "campaign_config_sha256"),
        ("task_config.json", "task_config_sha256"),
        ("gp_operator_config.json", "gp_operator_config_sha256"),
    ):
        expected = preflight["inputs"].get(key)
        if expected is not None:
            actual = hashlib.sha256((run_dir / "inputs" / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"preflight input mismatch: {filename}")
    seeds = read_sealed(run_dir / "inputs/seed_archive.json")
    if digest(seeds) != manifest["provenance"]["seed_archive"]["payload_sha256"]:
        raise ValueError("initialization archive differs from manifest")

    events = read_events(archive / "events.jsonl")
    terminal = events[-1]["payload"]
    if (
        terminal["status"] != "complete"
        or terminal["reason"] != "normalized-work-budget-exhausted"
        or terminal["effective_terminal_microwork"] != budget
        or terminal["registered_max_microwork"] != budget
    ):
        raise ValueError("GP arm did not complete its registered work budget")
    cumulative = 0
    candidates = []
    expected_files = set()
    for event in events:
        if event["type"] != "candidate_completed":
            continue
        row = event["payload"]
        charge = row["evaluator_work"]["candidate_microwork"]
        if not isinstance(charge, int) or isinstance(charge, bool) or charge <= 0:
            raise ValueError("invalid candidate work charge")
        cumulative += charge
        if row["evaluator_work"]["cumulative_microwork"] != cumulative:
            raise ValueError("candidate cumulative work mismatch")
        source = row["proposal"].get("source")
        if source is not None:
            filename = f"{row['proposal_seq']:08d}.ps"
            expected_files.add(filename)
            if (archive / "candidates" / filename).read_text() != source:
                raise ValueError(f"candidate source mismatch: {filename}")
        evaluation = (row.get("evaluation") or {}).get("evaluation")
        if evaluation is not None:
            fitness = fitness_from_record(evaluation["fitness"])
            if fitness.proposal_seq != row["proposal_seq"] or source is None:
                raise ValueError("fitness/source candidate identity mismatch")
            candidates.append((fitness, row))
    if {p.name for p in (archive / "candidates").glob("*.ps")} != expected_files:
        raise ValueError("unexpected candidate files outside event archive")
    if (
        cumulative != terminal["actual_microwork"]
        or cumulative + terminal["remaining_microwork"] != budget
        or len([e for e in events if e["type"] == "candidate_completed"])
        != terminal["completed_candidates"]
    ):
        raise ValueError("terminal work/candidate accounting mismatch")
    if not candidates:
        raise ValueError("completed arm has no evaluated candidate")
    best_fitness, best = min(candidates, key=lambda pair: pair[0].sort_key())
    initial = [
        pair for pair in candidates
        if pair[1]["proposal"].get("metadata", {}).get("phase") == "initialization"
    ]
    initial_count = sum(
        e["type"] == "candidate_completed"
        and e["payload"]["proposal"].get("metadata", {}).get("phase") == "initialization"
        for e in events
    )
    if not initial or initial_count != preflight["initialization"]["population_size"]:
        raise ValueError("initialization population is incomplete")
    init_fitness, init_best = min(initial, key=lambda pair: pair[0].sort_key())
    energy = best_fitness.normalized_energy if best_fitness.feasible else None
    initial_energy = init_fitness.normalized_energy if init_fitness.feasible else None
    summary = {
        "best_absolute": energy,
        "best_at_initialization": initial_energy,
        "search_delta": (
            initial_energy - energy if initial_energy is not None and energy is not None else None
        ),
        "best_found_at": {
            "proposal_seq": best["proposal_seq"], "candidate_id": best["candidate_id"],
        },
        "initialization_best_found_at": {
            "proposal_seq": init_best["proposal_seq"], "candidate_id": init_best["candidate_id"],
        },
        "champion_from_initialization": best["proposal"].get("metadata", {}).get("phase")
        == "initialization",
    }
    report_agrees = None
    if (run_dir / "report.json").exists():
        old = read_sealed(run_dir / "report.json")["arms"]["same_dsl_gp"]
        for key in ("best_absolute", "best_at_initialization", "search_delta"):
            if old[key] != summary[key]:
                raise ValueError(f"recovered champion disagrees with report: {key}")
        for key in ("best_found_at", "initialization_best_found_at"):
            if any(old[key][field] != value for field, value in summary[key].items()):
                raise ValueError(f"recovered selection disagrees with report: {key}")
        report_agrees = True
    source = best["proposal"]["source"]
    return {
        "schema": "evolved-continuum-recovered-gp-arm-v1",
        "boundary": "Retrospective development analysis; no original confirmatory authorization.",
        "root_seed": campaign["root_seed"],
        "run": str(run_dir),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "preflight_payload_sha256": digest(preflight),
        "event_chain_head_sha256": events[-1]["event_sha256"],
        "terminal": terminal,
        "summary": summary,
        "champion_source": source,
        "champion_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "existing_report_agrees": report_agrees,
    }
