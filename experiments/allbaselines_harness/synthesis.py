"""Seal the paired single-seed/population-parity search-isolation synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .support import BOUNDARY, ROOT, canonical_json, read_sealed, write_exact, write_sealed

SCENARIOS = (
    ("dev-alibaba-hier", "evolution/task/config_dev_alibaba_hier.json", 800_000_011),
    ("dev-medium-hot", "evolution/task/config_dev_medium_hot.json", 800_000_021),
    ("dev-alibaba-zoo", "evolution/task/config_dev_alibaba_zoo.json", 800_000_031),
)
ADAPTIVE_ARMS = ("classic_gphh", "same_dsl_gp", "random_ast")
DSL_ARMS = ("same_dsl_gp", "random_ast")
WORK_MICRO = 58_800_000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _run_name(scenario: str, seed: int, *, parity: bool) -> str:
    suffix = "--seed-archive-2" if parity else ""
    return f"{scenario}--seed-{seed}--work-{WORK_MICRO}--non-llm{suffix}"


def _load_report(path: Path, *, scenario: str, seed: int, parity: bool) -> dict[str, Any]:
    wrapped = read_sealed(path)
    report = wrapped["payload"]
    expected_mode = "population-parity" if parity else "single-seed"
    if (
        report.get("schema") != "evolved-continuum-allbaselines-report-v1"
        or report.get("status") != "DEVELOPMENT_NON_LLM_COMPLETE"
        or report.get("interpretation_boundary") != BOUNDARY
        or report.get("confirmatory_ready") is not False
        or report.get("acceptance_evidence") is not False
        or report["run"].get("scenario_id") != scenario
        or report["run"].get("root_seed") != seed
        or report["run"].get("matched_max_microwork") != WORK_MICRO
        or report["run"].get("include_llm") is not False
        or report["run"].get("initialization_mode") != expected_mode
        or report["mechanical_health"].get("all_requested_arms_complete") is not True
        or report["mechanical_health"].get("matching_verified") is not True
    ):
        raise ValueError(f"run report is not a complete matched development run: {path}")
    if parity and report["population_parity_charging"].get("verified") is not True:
        raise ValueError(f"population-parity charging is not verified: {path}")
    if not parity and report["population_parity_charging"].get("enabled") is not False:
        raise ValueError(f"single-seed report claims population parity: {path}")
    if "gphh_gen0" not in report["arms"]:
        raise ValueError(f"run report lacks the derived gphh_gen0 arm: {path}")
    return {**wrapped, "path": path, "report": report}


def _decomposition(report: dict[str, Any], arm: str) -> dict[str, Any]:
    data = report["arms"][arm]
    return {
        "best_at_initialization": data["best_at_initialization"],
        "best_absolute": data["best_absolute"],
        "search_delta": data["search_delta"],
        "search_improved": data["search_improved"],
        "best_found_at": data["best_found_at"],
    }


def _edge_decomposition(report: dict[str, Any], competitor: str) -> dict[str, Any]:
    gphh = _decomposition(report, "classic_gphh")
    other = _decomposition(report, competitor)
    values = (
        gphh["best_at_initialization"],
        gphh["best_absolute"],
        gphh["search_delta"],
        other["best_at_initialization"],
        other["best_absolute"],
        other["search_delta"],
    )
    if any(value is None for value in values):
        raise ValueError("edge decomposition requires feasible initialization and champions")
    initialization_edge = other["best_at_initialization"] - gphh["best_at_initialization"]
    net_search_contribution = gphh["search_delta"] - other["search_delta"]
    final_edge = other["best_absolute"] - gphh["best_absolute"]
    residual = final_edge - initialization_edge - net_search_contribution
    if abs(residual) > 1e-12:
        raise ValueError("initialization/search edge decomposition identity failed")
    return {
        "competitor": competitor,
        "positive_means_gphh_better": True,
        "initialization_edge": initialization_edge,
        "gphh_search_delta": gphh["search_delta"],
        "competitor_search_delta": other["search_delta"],
        "net_search_contribution": net_search_contribution,
        "final_edge": final_edge,
        "gphh_advantage_survives": final_edge > 0,
        "identity_residual": residual,
    }


def _commands(runs_root: Path) -> list[dict[str, Any]]:
    commands = []
    for scenario, task_config, seed in SCENARIOS:
        common = (
            "experiments/allbaselines_harness/run.sh "
            f"{scenario} {task_config} 58.8 {seed} "
            f"--runs-root {runs_root.as_posix()}"
        )
        commands.append(
            {
                "scenario_id": scenario,
                "mode": "single-seed",
                "command": f"{common} --include-llm",
            }
        )
        commands.append(
            {
                "scenario_id": scenario,
                "mode": "population-parity",
                "command": f"{common} --seed-archive 2 --include-llm",
            }
        )
    return commands


def _verify_legacy_default(
    new_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove opt-in parity did not alter the existing default campaign contract."""
    rows = []
    old_root = ROOT / "experiments/allbaselines_harness/runs"
    for scenario, _task_config, seed in SCENARIOS:
        old_path = old_root / (
            f"{scenario}--seed-{seed}--work-{WORK_MICRO}--all-arms/report.json"
        )
        old_wrapped = read_sealed(old_path)
        old = old_wrapped["payload"]
        new = new_reports[scenario]
        for arm in ADAPTIVE_ARMS:
            prior = old["arms"][arm]
            current = new["arms"][arm]
            matches = {
                "best_normalized_energy": (
                    prior["best"]["normalized_energy"]
                    == current["best"]["normalized_energy"]
                ),
                "best_proposal_seq": (
                    prior["best"]["proposal_seq"] == current["best"]["proposal_seq"]
                ),
                "candidate_count": prior["candidate_count"] == current["candidate_count"],
                "registered_max_microwork": (
                    prior["accounting"]["registered_max_microwork"]
                    == current["accounting"]["registered_max_microwork"]
                ),
                "effective_terminal_microwork": (
                    prior["accounting"]["effective_terminal_microwork"]
                    == current["accounting"]["effective_terminal_microwork"]
                ),
            }
            if not all(matches.values()):
                raise ValueError(f"default-mode legacy comparison failed for {scenario}/{arm}")
            rows.append(
                {
                    "scenario_id": scenario,
                    "arm": arm,
                    "all_required_fields_match": True,
                    "matches": matches,
                    "prior_report_payload_sha256": old_wrapped["payload_sha256"],
                    "prior_actual_microwork": prior["accounting"]["actual_microwork"],
                    "current_actual_microwork": current["accounting"]["actual_microwork"],
                }
            )
    return {
        "verified": True,
        "comparison": (
            "new default non-LLM reruns versus the three earlier sealed all-arms campaigns"
        ),
        "required_equalities": [
            "best_normalized_energy",
            "best_proposal_seq",
            "candidate_count",
            "registered_max_microwork",
            "effective_terminal_microwork",
        ],
        "rows": rows,
        "note": (
            "Population parity is opt-in and path-isolated. Actual microwork may differ "
            "when the read-only evolution runtime invalidates a trapping candidate at an "
            "earlier rung; the registered/effective 58.8 budget and champion are unchanged."
        ),
    }


def synthesize(runs_root: Path, output_dir: Path) -> dict[str, Any]:
    runs_root = runs_root.resolve()
    output_dir = output_dir.resolve()
    run_records = []
    decomposition = []
    edges = []
    single_seed_reports = {}
    for scenario, _task_config, seed in SCENARIOS:
        for parity in (False, True):
            mode = "population-parity" if parity else "single-seed"
            run_dir = runs_root / _run_name(scenario, seed, parity=parity)
            loaded = _load_report(
                run_dir / "report.json",
                scenario=scenario,
                seed=seed,
                parity=parity,
            )
            report = loaded["report"]
            if not parity:
                single_seed_reports[scenario] = report
            run_records.append(
                {
                    "scenario_id": scenario,
                    "mode": mode,
                    "root_seed": seed,
                    "report_path": str(loaded["path"]),
                    "report_payload_sha256": loaded["payload_sha256"],
                    "population_parity_charging": report["population_parity_charging"],
                }
            )
            for arm in ADAPTIVE_ARMS:
                decomposition.append(
                    {
                        "scenario_id": scenario,
                        "mode": mode,
                        "arm": arm,
                        **_decomposition(report, arm),
                    }
                )
            for competitor in DSL_ARMS:
                edges.append(
                    {
                        "scenario_id": scenario,
                        "mode": mode,
                        **_edge_decomposition(report, competitor),
                    }
                )

    parity_edges = [row for row in edges if row["mode"] == "population-parity"]
    survival = {
        scenario: all(
            row["gphh_advantage_survives"]
            for row in parity_edges
            if row["scenario_id"] == scenario
        )
        for scenario, _task, _seed in SCENARIOS
    }
    commands = _commands(runs_root)
    legacy_default_verification = _verify_legacy_default(single_seed_reports)
    payload = {
        "schema": "evolved-continuum-search-isolation-synthesis-v1",
        "status": "DEVELOPMENT_NON_LLM_SYNTHESIS_COMPLETE",
        "development_only": True,
        "confirmatory_ready": False,
        "acceptance_evidence": False,
        "interpretation_boundary": BOUNDARY,
        "live_llm_calls_made": 0,
        "matched_normalized_work": 58.8,
        "root_seeds": {scenario: seed for scenario, _task, seed in SCENARIOS},
        "runs": run_records,
        "decomposition": decomposition,
        "gphh_edge_decomposition": edges,
        "population_parity_gphh_edge_survives_all_non_llm_dsl_arms": survival,
        "legacy_default_verification": legacy_default_verification,
        "llm_slot": {
            "status": "UNRUN_SUPERVISOR_OPT_IN_REQUIRED",
            "commands": commands,
        },
        "accounting_caveat": (
            "Classic GPHH is native DEAP and is not PlacementScript fuel-metered; "
            "population evaluation counts and normalized evaluator-work charging are "
            "matched, but instruction-level compute is not."
        ),
        "interpretation_note": (
            "Edge identity: final GPHH edge = initialization edge + GPHH search delta "
            "- competitor search delta. Contributions are algebraic development-only "
            "descriptions, not causal or confirmatory estimates."
        ),
    }
    wrapped = write_sealed(output_dir / "SYNTHESIS.json", payload)
    write_exact(output_dir / "SYNTHESIS.md", _markdown(wrapped))
    return wrapped


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def _markdown(wrapped: dict[str, Any]) -> str:
    payload = wrapped["payload"]
    lines = [
        "# Search-isolation synthesis",
        "",
        f"**Status:** {payload['status']}.",
        "",
        BOUNDARY,
        " No live LLM calls were made; the LLM slot remains supervisor opt-in.",
        "",
        f"Sealed JSON payload SHA-256: `{wrapped['payload_sha256']}`.",
        "",
        "## Initialization versus search",
        "",
        "| Mode | Scenario | Arm | Init best | Absolute best | Search delta | Best found at |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload["decomposition"]:
        lines.append(
            f"| {row['mode']} | {row['scenario_id']} | {row['arm']} | "
            f"{_format(row['best_at_initialization'])} | "
            f"{_format(row['best_absolute'])} | {_format(row['search_delta'])} | "
            f"`{canonical_json(row['best_found_at'])}` |"
        )
    lines.extend(
        [
            "",
            "Positive search delta means the arm improved on its own initialization.",
            "",
            "## GPHH edge decomposition",
            "",
            (
                "| Mode | Scenario | Competitor | Initialization edge | GPHH search | "
                "Competitor search | Net search contribution | Final edge | Survives |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["gphh_edge_decomposition"]:
        lines.append(
            f"| {row['mode']} | {row['scenario_id']} | {row['competitor']} | "
            f"{_format(row['initialization_edge'])} | {_format(row['gphh_search_delta'])} | "
            f"{_format(row['competitor_search_delta'])} | "
            f"{_format(row['net_search_contribution'])} | {_format(row['final_edge'])} | "
            f"{row['gphh_advantage_survives']} |"
        )
    lines.extend(
        [
            "",
            "The edge is competitor energy minus GPHH energy, so positive means GPHH is better. "
            "The exact identity is `final edge = initialization edge + GPHH search - "
            "competitor search`; these are algebraic development-only contributions.",
            "",
            "## Population-parity verdict",
            "",
        ]
    )
    for scenario, survives in payload[
        "population_parity_gphh_edge_survives_all_non_llm_dsl_arms"
    ].items():
        lines.append(f"- `{scenario}`: GPHH edge survives both non-LLM DSL arms = **{survives}**.")
    lines.extend(
        [
            "",
            "The LLM comparison is intentionally pending, so this verdict covers same-DSL GP "
            "and Random AST only.",
            "",
            "## Exact supervisor commands for the LLM slot",
            "",
            "Each command creates a separate sealed all-arms run; it never mutates the sealed "
            "non-LLM run.",
            "",
            "```bash",
        ]
    )
    lines.extend(item["command"] for item in payload["llm_slot"]["commands"])
    lines.extend(
        [
            "```",
            "",
            "## Default-mode compatibility",
            "",
            (
                "**Verified.** The new default non-parity reruns exactly match the three "
                "earlier sealed campaigns on champion energy, champion proposal index, "
                "candidate count, registered work, and effective terminal work for every "
                "non-LLM adaptive arm. Population parity is opt-in and path-isolated."
            ),
            "",
            payload["legacy_default_verification"]["note"],
            "",
            "## Accounting boundary",
            "",
            payload["accounting_caveat"],
            " This asymmetry is recorded rather than repaired here.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    wrapped = synthesize(args.runs_root, args.output_dir)
    print(canonical_json({"payload_sha256": wrapped["payload_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
