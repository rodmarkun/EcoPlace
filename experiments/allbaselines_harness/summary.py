"""One fail-closed report for adaptive arms and evaluation-only §6.2 references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baselines.handwritten import HANDWRITTEN_NAMES
from evolution.campaign import EventArchive, load_campaign_plan
from evolution.codex_engine import load_codex_engine_config
from evolution.openrouter_engine import load_openrouter_engine_config
from sim.energy import CandidateFitness

from .initialization import read_seed_archive, validate_parity_charging
from .matching import validate_matching
from .support import (
    BOUNDARY,
    PreparedRun,
    canonical_json,
    canonical_sha256,
    enforce_run_size,
    file_sha256,
    read_sealed,
    sealed_artifact_exists,
    size_guard,
    write_exact,
    write_sealed,
)


def fitness_from_record(raw: dict[str, Any]) -> CandidateFitness:
    fitness = CandidateFitness(
        feasible=raw["feasible"],
        infeasibility_vector=tuple(raw["infeasibility_vector"]),
        summed_primitive_shortfall=raw["summed_primitive_shortfall"],
        normalized_energy=raw["normalized_energy"],
        proposal_seq=raw["proposal_seq"],
        n_instances=raw["n_instances"],
        n_bundles=raw["n_bundles"],
    )
    if raw.get("sort_key") != list(fitness.sort_key()):
        raise ValueError("archived fitness differs from exact CandidateFitness.sort_key")
    return fitness


def trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, Any] | None = None
    points = []
    for row in sorted(rows, key=lambda item: item["proposal_seq"]):
        fitness = row.get("fitness")
        improved = fitness is not None and (
            best is None or fitness.sort_key() < best["fitness"].sort_key()
        )
        if improved:
            best = row
        points.append(
            {
                "candidate_id": row["candidate_id"],
                "proposal_seq": row["proposal_seq"],
                "cumulative_microwork": row["cumulative_microwork"],
                "candidate_sort_key": (
                    None if fitness is None else list(fitness.sort_key())
                ),
                "candidate_normalized_energy": (
                    None if fitness is None else fitness.normalized_energy
                ),
                "incumbent_improved": improved,
                "final_incumbent_id_so_far": (
                    None if best is None else best["candidate_id"]
                ),
                "final_incumbent_sort_key_so_far": (
                    None if best is None else list(best["fitness"].sort_key())
                ),
            }
        )
    return points


def plateau_histogram(
    rows: list[dict[str, Any]], *, tolerance: float
) -> dict[str, Any]:
    feasible = [
        row
        for row in rows
        if row.get("fitness") is not None and row["fitness"].feasible
    ]
    levels: list[float] = []
    for value in sorted(row["fitness"].normalized_energy for row in feasible):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
    buckets = []
    for level in levels:
        members = [
            row
            for row in feasible
            if abs(row["fitness"].normalized_energy - level) <= tolerance
        ]
        buckets.append(
            {
                "normalized_energy": level,
                "candidate_count": len(members),
                "first_proposal_seq": min(row["proposal_seq"] for row in members),
                "first_cumulative_normalized_work": min(
                    row["cumulative_microwork"] for row in members
                )
                / 1_000_000,
            }
        )
    return {
        "distinct_tolerance": tolerance,
        "l2_feasible_candidates": len(feasible),
        "distinct_feasible_energy_plateaus": len(buckets),
        "buckets": buckets,
    }


def _verify_run_sha(run: dict[str, Any]) -> None:
    claimed = run.get("sha256")
    body = {key: value for key, value in run.items() if key != "sha256"}
    if claimed != canonical_sha256(body):
        raise ValueError("adaptive run receipt digest mismatch")


def _direct_rows(run: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    rows = []
    for candidate in run["candidates"]:
        ladder = candidate["ladder"]
        evaluation = ladder.get("evaluation")
        raw = None if evaluation is None else evaluation.get("fitness")
        rows.append(
            {
                "candidate_id": candidate.get(
                    "candidate_id", f"{arm}:{candidate['proposal_seq']}"
                ),
                "proposal_seq": candidate["proposal_seq"],
                "generation": candidate.get("generation"),
                "cumulative_microwork": candidate["cumulative_microwork"],
                "candidate_microwork": candidate["candidate_microwork"],
                "trace": tuple(ladder["trace"]),
                "phase": candidate.get(
                    "phase", candidate.get("proposal_metadata", {}).get("phase")
                ),
                "identity": ladder.get("policy"),
                "fitness": None if raw is None else fitness_from_record(raw),
                "fitness_record": raw,
            }
        )
    return rows


def _read_usage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "journal_present": False,
            "dispatches_reserved": 0,
            "dispatches_settled": 0,
            "dispatches_abandoned": 0,
            "dispatches_pending": 0,
            "dispatch_progress_events": 0,
            "charged_wall_clock_seconds": 0.0,
        }
    encoded = path.read_bytes()
    if encoded and not encoded.endswith(b"\n"):
        raise ValueError("proposal-engine usage journal has an unterminated record")
    previous = "0" * 64
    reserved: set[int] = set()
    settled: set[int] = set()
    abandoned: set[int] = set()
    progress_events = 0
    charged_ms = 0
    for index, raw_line in enumerate(encoded.splitlines()):
        line = raw_line.decode()
        event = json.loads(line)
        if line != canonical_json(event):
            raise ValueError("proposal-engine usage journal is not canonical JSON")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if (
            body.get("event_seq") != index
            or body.get("previous_event_sha256") != previous
            or event.get("event_sha256") != canonical_sha256(body)
        ):
            raise ValueError("proposal-engine usage journal hash chain failed")
        payload = event["payload"]
        call_seq = payload["call_seq"]
        if event["type"] == "dispatch_reserved":
            if call_seq in reserved:
                raise ValueError("duplicate proposal-engine reservation")
            reserved.add(call_seq)
        elif event["type"] in {"dispatch_settled", "dispatch_abandoned"}:
            if call_seq not in reserved or call_seq in settled | abandoned:
                raise ValueError("proposal-engine settlement lacks one reservation")
            charged_ms += payload["duration_ms"]
            (settled if event["type"] == "dispatch_settled" else abandoned).add(call_seq)
        elif event["type"] == "dispatch_progress":
            if call_seq not in reserved or call_seq in settled | abandoned:
                raise ValueError("proposal-engine progress lacks one reservation")
            progress_events += 1
        else:
            raise ValueError("unknown proposal-engine usage event")
        previous = event["event_sha256"]
    return {
        "journal_present": True,
        "dispatches_reserved": len(reserved),
        "dispatches_settled": len(settled),
        "dispatches_abandoned": len(abandoned),
        "dispatches_pending": len(reserved - settled - abandoned),
        "dispatch_progress_events": progress_events,
        "charged_wall_clock_seconds": charged_ms / 1_000,
    }


def _campaign_rows(
    archive_path: Path,
    *,
    expected_campaign_arm: str,
    prepared: PreparedRun,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((archive_path / "manifest.json").read_text())
    archive = EventArchive(archive_path, manifest)
    campaign = manifest["campaign"]
    expected_identity = {
        "arm": expected_campaign_arm,
        "campaign_id": "h00",
        "root_seed": prepared.root_seed,
        "plan_status": "development-provisional",
        "feedback_profile": prepared.evolution_profile.feedback_profile,
        "evolution_profile": prepared.evolution_profile.name,
        "run_purpose": "scientific-full-work",
        "evaluation_seed_mode": "campaign-derived",
        "proposals_per_call": getattr(prepared, "proposals_per_call", 1),
        "max_microwork": prepared.max_microwork,
    }
    actual_identity = {
        key: campaign.get(key)
        for key in expected_identity
        if key != "max_microwork"
    }
    actual_identity["max_microwork"] = campaign.get("evaluator_work", {}).get(
        "max_microwork"
    )
    actual_identity["proposals_per_call"] = campaign.get("proposals_per_call", 1)
    divergence = {
        key: {"expected": expected, "actual": actual_identity[key]}
        for key, expected in expected_identity.items()
        if actual_identity[key] != expected
    }
    if divergence:
        raise ValueError(
            "adaptive archive differs from the requested development run: "
            f"{canonical_json(divergence)}"
        )
    archive_provenance = manifest.get("provenance", {}).get("seed_archive")
    if prepared.population_parity:
        fixed = read_seed_archive(
            prepared.run_dir / "inputs/seed_archive.json",
            expected_size=prepared.initialization_population_size,
            initial_path=prepared.initial_path,
        )
        if archive_provenance != {
            "payload_sha256": fixed["payload_sha256"],
            "population_size": prepared.initialization_population_size,
            "charging": "CampaignRunner evaluator-work ledger",
        }:
            raise ValueError("campaign archive is not bound to the fixed seed archive")
    elif archive_provenance is not None:
        raise ValueError("single-seed campaign unexpectedly declares a seed archive")
    rows = []
    terminal = None
    for event in archive.events:
        if event["type"] == "campaign_terminal":
            terminal = event["payload"]
            continue
        if event["type"] != "candidate_completed":
            continue
        payload = event["payload"]
        ladder = payload.get("evaluation")
        evaluation = None if ladder is None else ladder.get("evaluation")
        raw = None if evaluation is None else evaluation.get("fitness")
        rows.append(
            {
                "candidate_id": payload["candidate_id"],
                "proposal_seq": payload["proposal_seq"],
                "generation": None,
                "cumulative_microwork": payload["evaluator_work"][
                    "cumulative_microwork"
                ],
                "candidate_microwork": payload["evaluator_work"][
                    "candidate_microwork"
                ],
                "trace": () if ladder is None else tuple(ladder["trace"]),
                "phase": payload["proposal"].get("metadata", {}).get("phase"),
                "identity": None if ladder is None else ladder.get("policy"),
                "fitness": None if raw is None else fitness_from_record(raw),
                "fitness_record": raw,
            }
        )
    if terminal is None:
        raise ValueError(f"adaptive archive lacks a terminal event: {archive_path}")
    usage = _read_usage(archive_path / "proposal-engine-usage.jsonl")
    from .runner import _read_driver

    driver = _read_driver(archive_path / "driver-invocations.jsonl")
    started_events = [event for event in driver if event["type"] == "started"]
    finished_events = [event for event in driver if event["type"] == "finished"]
    if len(started_events) + len(finished_events) != len(driver):
        raise ValueError("adaptive driver journal contains an unknown event type")
    started = [event["payload"]["invocation_id"] for event in started_events]
    finished = [event["payload"]["invocation_id"] for event in finished_events]
    if (
        started != list(range(len(started)))
        or len(finished) != len(set(finished))
        or set(finished) - set(started)
    ):
        raise ValueError("adaptive driver journal invocation sequence is invalid")
    expected_driver_arm = archive_path.name
    command = None
    for event in started_events:
        payload = event["payload"]
        invocation_id = payload["invocation_id"]
        if (
            payload.get("arm") != expected_driver_arm
            or payload.get("resume") is not (invocation_id > 0)
            or not isinstance(payload.get("command"), list)
        ):
            raise ValueError("adaptive driver journal invocation identity is invalid")
        if command is None:
            command = payload["command"]
        elif payload["command"] != command:
            raise ValueError("adaptive driver command differs across resume")
    if any(
        event["payload"].get("arm") != expected_driver_arm
        for event in finished_events
    ):
        raise ValueError("adaptive driver finish arm differs from its archive")
    latest_invocation = started[-1] if started else None
    unfinished = set(started) - set(finished)
    superseded_unfinished = sorted(
        invocation_id
        for invocation_id in unfinished
        if latest_invocation is not None and invocation_id < latest_invocation
    )
    active_unfinished = sorted(unfinished - set(superseded_unfinished))
    nonzero = sum(
        event["payload"]["exit_code"] != 0
        for event in finished_events
    )
    latest_finished = next(
        (
            event
            for event in finished_events
            if event["payload"]["invocation_id"] == latest_invocation
        ),
        None,
    )
    latest_succeeded = (
        latest_finished is not None
        and latest_finished["payload"].get("exit_code") == 0
    )
    accounting = {
        "mode": "adaptive-normalized-work",
        "registered_max_microwork": terminal["registered_max_microwork"],
        "actual_microwork": terminal["actual_microwork"],
        "effective_terminal_microwork": terminal["effective_terminal_microwork"],
        "proposal_calls": usage["dispatches_reserved"],
        "proposal_wall_clock_seconds": usage["charged_wall_clock_seconds"],
    }
    health = {
        "complete": (
            terminal["status"] == "complete"
            and terminal["reason"] == "normalized-work-budget-exhausted"
        ),
        "archive_chain_verified": True,
        "archive_sha256": archive.digest(),
        "archive_chain_head_sha256": archive.events[-1]["event_sha256"],
        "terminal_status": terminal["status"],
        "terminal_reason": terminal["reason"],
        "usage": usage,
        "driver_journal_present": bool(driver),
        "driver_invocations": len(started),
        "driver_unfinished_invocations": active_unfinished,
        "driver_superseded_unfinished_invocations": superseded_unfinished,
        "driver_nonzero_exit_count": nonzero,
        "driver_latest_invocation_succeeded": latest_succeeded,
    }
    health["complete"] = (
        health["complete"]
        and health["driver_journal_present"]
        and not health["driver_unfinished_invocations"]
        and latest_succeeded
    )
    return rows, {
        "accounting": accounting,
        "mechanical_health": health,
        "terminal": terminal,
        "manifest": manifest,
    }


def _evaluation_arm(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_sealed(path)["payload"]
    raw = payload["evaluation"]["fitness"]
    row = {
        "candidate_id": payload["arm"],
        "proposal_seq": raw["proposal_seq"],
        "generation": None,
        "cumulative_microwork": 0,
        "candidate_microwork": 0,
        "trace": (),
        "phase": "evaluation-only",
        "identity": payload["evaluation"].get("policy"),
        "fitness": fitness_from_record(raw),
        "fitness_record": raw,
    }
    return [row], payload


def _adaptive_arm(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_sealed(path)["payload"]
    _verify_run_sha(payload["run"])
    return _direct_rows(payload["run"], payload["arm"]), payload


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["fitness"] is not None]
    return min(eligible, key=lambda row: row["fitness"].sort_key()) if eligible else None


def _row_record(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "candidate_id": row["candidate_id"],
        "proposal_seq": row["proposal_seq"],
        "generation": row.get("generation"),
        "sort_key": list(row["fitness"].sort_key()),
        "normalized_energy": row["fitness"].normalized_energy,
        "feasible": row["fitness"].feasible,
    }


def gphh_gen0_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generation_zero = [row for row in rows if row.get("generation") == 0]
    if not generation_zero:
        raise ValueError("classic GPHH run contains no generation-0 candidates")
    return generation_zero


def _best_found_at(arm: str, row: dict[str, Any] | None, *, role: str) -> dict[str, Any] | None:
    if row is None:
        return None
    if role.endswith("evaluation-only-reference") or role.startswith("hand-written"):
        coordinate = "evaluation_only"
        index = None
    elif arm in {"classic_gphh", "gphh_gen0"}:
        coordinate = "generation"
        index = row.get("generation")
    elif row["candidate_id"].endswith(":seeded-start"):
        coordinate = "seeded_start"
        index = 0
    else:
        coordinate = "proposal"
        index = row["proposal_seq"]
    return {
        "coordinate": coordinate,
        "index": index,
        "proposal_seq": row["proposal_seq"],
        "candidate_id": row["candidate_id"],
    }


def _feasible_energy(row: dict[str, Any] | None) -> float | None:
    if row is None or not row["fitness"].feasible:
        return None
    return row["fitness"].normalized_energy


def _comparison(reference: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict:
    reference_energy = None if reference is None else reference["normalized_energy"]
    candidate_energy = None if candidate is None else candidate["normalized_energy"]
    if (
        reference is None
        or candidate is None
        or not reference["feasible"]
        or not candidate["feasible"]
        or reference_energy is None
        or candidate_energy is None
    ):
        return {"direction": "not-comparable", "normalized_energy_reduction_pct": None}
    reduction = 100.0 * (reference_energy - candidate_energy) / reference_energy
    return {
        "direction": "better" if reduction > 0 else "worse" if reduction < 0 else "equal",
        "normalized_energy_reduction_pct": reduction,
    }


def arm_summary(
    arm: str,
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    tolerance: float,
    initialization_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    best = _best_row(rows)
    starting = rows if initialization_rows is None else initialization_rows
    best_at_initialization = _best_row(starting)
    by_id = {row["candidate_id"]: row for row in [*starting, *rows]}
    best_absolute = _best_row(list(by_id.values()))
    initial_energy = _feasible_energy(best_at_initialization)
    absolute_energy = _feasible_energy(best_absolute)
    search_delta = (
        None
        if initial_energy is None or absolute_energy is None
        else initial_energy - absolute_energy
    )
    points = trajectory(rows)
    final = best
    return {
        "arm": arm,
        "role": payload.get("role", "adaptive-search"),
        "best": _row_record(best),
        "best_absolute": absolute_energy,
        "best_at_initialization": initial_energy,
        "search_delta": search_delta,
        "best_found_at": _best_found_at(
            arm, best_absolute, role=payload.get("role", "adaptive-search")
        ),
        "initialization_best_found_at": _best_found_at(
            arm, best_at_initialization, role=payload.get("role", "adaptive-search")
        ),
        "search_improved": (
            None
            if best_absolute is None or best_at_initialization is None
            else best_absolute["fitness"].sort_key()
            < best_at_initialization["fitness"].sort_key()
        ),
        "final_incumbent": _row_record(final),
        "last_evaluated": _row_record(rows[-1] if rows and rows[-1]["fitness"] else None),
        "candidate_count": len(rows),
        "plateau_histogram": plateau_histogram(rows, tolerance=tolerance),
        "trajectory": points,
        "accounting": payload["accounting"],
        "mechanical_health": payload["mechanical_health"],
    }


def _level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    levels: dict[str, dict[str, int]] = {}
    for row in rows:
        for trace in row["trace"]:
            rung = trace["rung"]
            summary = levels.setdefault(
                rung,
                {"visited_candidates": 0, "passed_candidates": 0, "failed_candidates": 0},
            )
            summary["visited_candidates"] += 1
            decision = trace.get("decision")
            if trace.get("passed") is True or decision in {
                "promote",
                "fitness-of-record",
            }:
                summary["passed_candidates"] += 1
            elif trace.get("passed") is False or decision in {"reject", "invalid"}:
                summary["failed_candidates"] += 1
    return dict(sorted(levels.items()))


def _archive_references(
    prepared: PreparedRun, archive_name: str, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    archive_path = prepared.run_dir / "archives" / archive_name
    health = metadata["mechanical_health"]
    references = [
        {
            "role": "hash-chained-campaign-archive",
            "path": str(archive_path.relative_to(prepared.run_dir)),
            "sha256": health["archive_sha256"],
            "hash_chain_head_sha256": health["archive_chain_head_sha256"],
        }
    ]
    driver = archive_path / "driver-invocations.jsonl"
    if driver.exists():
        references.append(
            {
                "role": "driver-invocation-journal",
                "path": str(driver.relative_to(prepared.run_dir)),
                "sha256": file_sha256(driver),
            }
        )
    return references


def _write_archive_arm_result(
    prepared: PreparedRun,
    arm: str,
    archive_name: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    best = _best_row(rows)
    champion = None
    if best is not None:
        champion = {
            **_row_record(best),
            "identity": best.get("identity"),
            "fitness": best["fitness_record"],
            "energies": {
                "normalized_energy": best["fitness"].normalized_energy,
                "full_instance_evidence": "referenced-hash-chained-campaign-archive",
            },
        }
    payload = {
        "schema": "evolved-continuum-allbaselines-archive-arm-v1",
        "status": (
            "DEVELOPMENT_ADAPTIVE_COMPLETE"
            if metadata["mechanical_health"]["complete"]
            else "DEVELOPMENT_INCOMPLETE"
        ),
        "development_only": True,
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "arm": arm,
        "role": summary["role"],
        "champion": champion,
        "accounting": metadata["accounting"],
        "mechanical_health": metadata["mechanical_health"],
        "summaries": {
            "plateau_histogram": summary["plateau_histogram"],
            "ladder_levels": _level_summary(rows),
            "candidate_count": summary["candidate_count"],
        },
        "references": _archive_references(prepared, archive_name, metadata),
    }
    path = prepared.run_dir / "arms" / arm / "result.json"
    if sealed_artifact_exists(path):
        existing = read_sealed(path)
        if existing["payload_sha256"] != canonical_sha256(payload):
            raise ValueError(f"existing compact archive-arm result differs: {path}")
        return
    from .runner import _write_arm_result

    _write_arm_result(prepared, path, payload)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(wrapped: dict[str, Any]) -> str:
    report = wrapped["payload"]
    lines = [
        "# All-baselines development comparison",
        "",
        f"- Status: **{report['status']}**",
        f"- Payload SHA-256: `{wrapped['payload_sha256']}`",
        f"- Scenario: `{report['run']['scenario_id']}`",
        f"- Root seed: `{report['run']['root_seed']}`",
        f"- Matched normalized-work budget: `{report['run']['matched_normalized_work']}`",
    ]
    if report["run"].get("gp_mix_label") is not None:
        lines.append(
            f"- GP mix: `{report['run']['gp_mix_label']}` "
            f"(`{report['run']['gp_operator_config_sha256']}`)"
        )
    lines.extend(
        [
            "- Selection authority: exact Phase-3 `CandidateFitness.sort_key` (lower is better)",
            "",
            "## Initialization/search decomposition",
            "",
            (
                "| Arm | Role | Init best | Absolute best | Search delta | Best found at | "
                "Work | Calls | Healthy |"
            ),
            "|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for arm, data in report["arms"].items():
        accounting = data["accounting"]
        lines.append(
            f"| {arm} | {data['role']} | "
            f"{_format(data['best_at_initialization'])} | "
            f"{_format(data['best_absolute'])} | "
            f"{_format(data['search_delta'])} | "
            f"`{canonical_json(data['best_found_at'])}` | "
            f"{_format(accounting['actual_microwork'] / 1_000_000)} | "
            f"{accounting.get('proposal_calls', 0)} | "
            f"{data['mechanical_health']['complete']} |"
        )
    lines.extend(
        [
            "",
            "`gphh_gen0` is a derived random-population baseline. It performs no extra "
            "evaluation and exposes the best member of classic GPHH generation 0.",
        ]
    )
    if report["llm_slot"]["status"] == "SKIPPED_NOT_REQUESTED":
        lines.extend(
            [
                "",
                "The LLM slot was not requested for this run. Use `--include-llm` to create a "
                "separate all-arms run; preflight itself makes zero completion calls.",
            ]
        )
    lines.extend(["", "## Distinct feasible-energy plateaus", ""])
    for arm, data in report["arms"].items():
        histogram = data["plateau_histogram"]
        levels = [bucket["normalized_energy"] for bucket in histogram["buckets"]]
        lines.append(
            f"- `{arm}`: {histogram['distinct_feasible_energy_plateaus']} plateaus; "
            f"levels `{canonical_json(levels)}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            BOUNDARY,
            " Frozen PPO and hand-written policies receive no adaptation budget. This report "
            "supports development ranking only and carries no statistical or confirmatory claim.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(prepared: PreparedRun) -> dict[str, Any]:
    tolerance = prepared.harness_config["distinct_energy_tolerance"]
    codex_config = (
        load_codex_engine_config(prepared.engine_config)
        if prepared.engine == "codex"
        else None
    )
    openrouter_config = (
        load_openrouter_engine_config(prepared.engine_config)
        if prepared.engine == "openrouter"
        else None
    )
    arms: dict[str, dict[str, Any]] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    initialization_rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    initial_rows, initial_payload = _evaluation_arm(
        prepared.run_dir / "arms/initial/result.json"
    )
    initial = _row_record(_best_row(initial_rows))
    if initial is None:
        raise ValueError("common initial evaluation is absent")

    for arm in HANDWRITTEN_NAMES:
        rows, payload = _evaluation_arm(prepared.run_dir / f"arms/{arm}/result.json")
        rows_by_arm[arm] = rows
        initialization_rows_by_arm[arm] = rows
        arms[arm] = arm_summary(arm, rows, payload, tolerance=tolerance)
    rows, payload = _evaluation_arm(
        prepared.run_dir / "arms/ppo_frozen/result.json"
    )
    arms["ppo_frozen"] = arm_summary(
        "ppo_frozen", rows, payload, tolerance=tolerance
    )
    rows_by_arm["ppo_frozen"] = rows
    initialization_rows_by_arm["ppo_frozen"] = rows

    gphh_rows, gphh_payload = _adaptive_arm(
        prepared.run_dir / "arms/classic_gphh/result.json"
    )
    gphh_initial = gphh_gen0_rows(gphh_rows)
    rows_by_arm["classic_gphh"] = gphh_rows
    initialization_rows_by_arm["classic_gphh"] = gphh_initial
    arms["classic_gphh"] = arm_summary(
        "classic_gphh",
        gphh_rows,
        gphh_payload,
        tolerance=tolerance,
        initialization_rows=gphh_initial,
    )

    random_rows, random_payload = _adaptive_arm(
        prepared.run_dir / "arms/random_ast/result.json"
    )
    if prepared.population_parity:
        random_initial = [
            row
            for row in random_rows
            if row["proposal_seq"] < prepared.initialization_population_size
            and row.get("phase") == "initialization"
        ]
    else:
        random_seed = dict(initial_rows[0])
        random_seed["candidate_id"] = "random_ast:seeded-start"
        random_seed["phase"] = "initialization"
        random_initial = [random_seed]
    rows_by_arm["random_ast"] = random_rows
    initialization_rows_by_arm["random_ast"] = random_initial
    arms["random_ast"] = arm_summary(
        "random_ast",
        random_rows,
        random_payload,
        tolerance=tolerance,
        initialization_rows=random_initial,
    )

    for arm, archive_name in (("same_dsl_gp", "same_dsl_gp"),):
        rows, metadata = _campaign_rows(
            prepared.run_dir / "archives" / archive_name,
            expected_campaign_arm="gp",
            prepared=prepared,
        )
        payload = {
            "role": "adaptive-search",
            "accounting": metadata["accounting"],
            "mechanical_health": metadata["mechanical_health"],
        }
        initialization_rows = [
            row
            for row in rows
            if row["proposal_seq"] < prepared.initialization_population_size
            and (
                not prepared.population_parity or row.get("phase") == "initialization"
            )
        ]
        rows_by_arm[arm] = rows
        initialization_rows_by_arm[arm] = initialization_rows
        arms[arm] = arm_summary(
            arm,
            rows,
            payload,
            tolerance=tolerance,
            initialization_rows=initialization_rows,
        )
        _write_archive_arm_result(
            prepared,
            arm,
            archive_name,
            rows,
            metadata,
            arms[arm],
        )
    if prepared.include_llm:
        rows, metadata = _campaign_rows(
            prepared.run_dir / "archives/llm",
            expected_campaign_arm="llm",
            prepared=prepared,
        )
        payload = {
            "role": "adaptive-search-development-a1-subscription",
            "accounting": metadata["accounting"],
            "mechanical_health": metadata["mechanical_health"],
        }
        initialization_rows = [
            row
            for row in rows
            if row["proposal_seq"] < prepared.initialization_population_size
            and (
                not prepared.population_parity or row.get("phase") == "initialization"
            )
        ]
        rows_by_arm["llm"] = rows
        initialization_rows_by_arm["llm"] = initialization_rows
        arms["llm"] = arm_summary(
            "llm",
            rows,
            payload,
            tolerance=tolerance,
            initialization_rows=initialization_rows,
        )
        _write_archive_arm_result(
            prepared,
            "llm",
            "llm",
            rows,
            metadata,
            arms["llm"],
        )

    accounting = {arm: data["accounting"] for arm, data in arms.items()}
    try:
        matching = validate_matching(
            accounting,
            matched_max_microwork=prepared.max_microwork,
            include_llm=prepared.include_llm,
        )
    except ValueError as exc:
        matching = {
            "verified": False,
            "error": str(exc),
            "matched_max_microwork": prepared.max_microwork,
        }
    if prepared.population_parity:
        parity_rows = {
            arm: initialization_rows_by_arm[arm]
            for arm in (
                "classic_gphh",
                "same_dsl_gp",
                "random_ast",
                *(("llm",) if prepared.include_llm else ()),
            )
        }
        parity_charging = validate_parity_charging(
            parity_rows,
            expected_count=prepared.initialization_population_size,
            work=load_campaign_plan(prepared.campaign_config).evaluator_work,
        )
    else:
        parity_charging = {
            "enabled": False,
            "verified": None,
            "population_size": 1,
            "reason": "single-seed initialization mode",
        }

    derived_payload = {
        "role": "random-population-baseline-derived-no-additional-evaluation",
        "accounting": {
            "mode": "derived-view-no-additional-work",
            "registered_max_microwork": None,
            "actual_microwork": 0,
            "effective_terminal_microwork": None,
            "proposal_calls": 0,
            "source_arm": "classic_gphh",
            "source_generation": 0,
        },
        "mechanical_health": {
            "complete": True,
            "derived_from_verified_classic_gphh": True,
            "additional_evaluations": 0,
        },
    }
    arms["gphh_gen0"] = arm_summary(
        "gphh_gen0",
        gphh_initial,
        derived_payload,
        tolerance=tolerance,
        initialization_rows=gphh_initial,
    )
    references = {
        **{arm: arms[arm]["best"] for arm in HANDWRITTEN_NAMES},
        "ppo_frozen": arms["ppo_frozen"]["best"],
    }
    for data in arms.values():
        data["improvement_vs_initial"] = _comparison(initial, data["best"])
        data["improvement_vs_every_reference"] = {
            name: _comparison(reference, data["best"])
            for name, reference in references.items()
        }
    mechanically_healthy = all(
        data["mechanical_health"].get("complete") is True for data in arms.values()
    )
    status = (
        "DEVELOPMENT_ALL_ARMS_COMPLETE"
        if prepared.include_llm and mechanically_healthy and matching["verified"]
        else "DEVELOPMENT_NON_LLM_COMPLETE"
        if not prepared.include_llm and mechanically_healthy and matching["verified"]
        else "DEVELOPMENT_MECHANICAL_FAILURE"
    )
    report = {
        "schema": "evolved-continuum-allbaselines-report-v1",
        "status": status,
        "development_only": True,
        "confirmatory_ready": False,
        "acceptance_evidence": False,
        "interpretation_boundary": BOUNDARY,
        "selection_authority": "CandidateFitness.sort_key (lower-is-better)",
        "normalized_energy": "Phase-3 idle-floor-normalized closed-boundary energy",
        "run": {
            "scenario_id": prepared.scenario_id,
            "root_seed": prepared.root_seed,
            "matched_max_microwork": prepared.max_microwork,
            "matched_normalized_work": prepared.max_microwork / 1_000_000,
            "include_llm": prepared.include_llm,
            "initialization_mode": (
                "population-parity" if prepared.population_parity else "single-seed"
            ),
            "seed_archive_random_mutants": prepared.seed_archive_random_mutants,
            "initialization_population_size": prepared.initialization_population_size,
            **(
                {
                    "gp_mix_label": prepared.gp_mix_label,
                    "gp_operator_config_sha256": file_sha256(prepared.gp_config_path),
                }
                if prepared.gp_config_path is not None
                else {}
            ),
        },
        "initial": initial,
        "initial_policy": {
            "name": initial_payload["evaluation"]["policy"]["name"],
            "source_sha256": initial_payload["initial_source_sha256"],
        },
        "llm_slot": {
            "status": "RUN" if prepared.include_llm else "SKIPPED_NOT_REQUESTED",
            "engine": (
                prepared.harness_config["llm"]["engine"]
                if prepared.engine == "codex"
                else prepared.engine
            ),
            **(
                {
                    "hard_call_ceiling": codex_config.max_calls_per_campaign,
                    "hard_wall_clock_seconds": prepared.harness_config["llm"][
                        "hard_wall_clock_seconds"
                    ],
                }
                if codex_config is not None
                else {
                    "model_id": openrouter_config.model_id,
                    "max_total_microusd": openrouter_config.max_total_microusd,
                }
            ),
        },
        "matching": matching,
        "population_parity_charging": parity_charging,
        "arms": dict(sorted(arms.items())),
        "work_and_call_accounting": dict(sorted(accounting.items())),
        "mechanical_health": {
            "all_requested_arms_complete": mechanically_healthy,
            "matching_verified": matching["verified"],
            "sealed_arm_results_verified": True,
            "campaign_archive_chains_verified": True,
        },
        "limitations": [
            BOUNDARY,
            "one root and no statistical inference",
            "PPO is frozen evaluation-only and receives no per-trigger adaptation budget",
            "hand-written references are evaluation-only and receive no adaptation budget",
            "A1 Codex outputs are live-nondeterministic; retained archives are the audit record",
            (
                "Classic GPHH uses its native DEAP representation and is not "
                "PlacementScript fuel-metered; this experiment records but does not "
                "repair that compute-accounting asymmetry."
            ),
        ],
    }
    guard = size_guard(prepared.harness_config)
    wrapped = write_sealed(
        prepared.run_dir / "report.json",
        report,
        max_bytes=int(guard["max_artifact_bytes"]),
    )
    write_exact(
        prepared.run_dir / "REPORT.md",
        _markdown(wrapped),
        max_bytes=int(guard["max_artifact_bytes"]),
    )
    enforce_run_size(prepared.run_dir, prepared.harness_config)
    return wrapped
