# ruff: noqa: E501
"""GP operator-mix and budget-sweep analysis. Zero LLM calls.

Reads sealed confirmatory GP/LLM archives for the current mix at 56.0, new
harness runs under ``runs/`` for every other cell, and the sealed LLM held-out
table in ``FINALIST_RESCORE_runs_final.json``. Never writes into ``runs_final``.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/deep_search_20260825"))
sys.path.insert(0, str(ROOT / "experiments/transfer_20260825"))

from deep_search_analysis import seal_payload  # noqa: E402
from rescore_finalists import fresh_grid  # noqa: E402
from transfer import champions_from_run  # noqa: E402

from evolution.evaluation import (  # noqa: E402
    ScenarioGrid,
    evaluate_policy_source,
    load_task_evaluation_config,
)

BOUNDARY = (
    "Quarantined development-only. Zero LLM calls. Never acceptance, held-out, "
    "or confirmatory evidence."
)
ROOTS = tuple(range(800000801, 800000809))
MIXES: dict[str, dict[str, float | int]] = {
    "current": {
        "crossover_probability": 0.45,
        "subtree_mutation_probability": 0.35,
        "numeric_mutation_probability": 0.20,
        "max_attempts": 24,
    },
    "numeric-heavy": {
        "crossover_probability": 0.20,
        "subtree_mutation_probability": 0.20,
        "numeric_mutation_probability": 0.60,
        "max_attempts": 24,
    },
    "subtree-heavy": {
        "crossover_probability": 0.20,
        "subtree_mutation_probability": 0.60,
        "numeric_mutation_probability": 0.20,
        "max_attempts": 24,
    },
    "crossover-heavy": {
        "crossover_probability": 0.70,
        "subtree_mutation_probability": 0.20,
        "numeric_mutation_probability": 0.10,
        "max_attempts": 24,
    },
}
BUDGETS = (56.0, 112.0, 224.0)
CELLS = (
    *[(mix, 56.0) for mix in MIXES],
    ("current", 112.0),
    ("current", 224.0),
)
RUNS_ROOT = HERE / "runs"
EARLY_224_ROOT = HERE / "runs_budget224_early"
RUNS_FINAL = ROOT / "experiments/deep_search_20260825/runs_final"
LLM_RESCORE = ROOT / "experiments/deep_search_20260825/FINALIST_RESCORE_runs_final.json"
TASK_CONFIG = ROOT / "evolution/task/config_dev_continuum_tight_calibrated.json"
STARTING_POLICY = ROOT / "runtime/policies/graph_headroom.ps"
INIT_POP = 12
ALPHA = 0.05
MIN_SIGNIFICANT_M = 6
SCENARIO = "dev-continuum-tight"


def microwork(budget: float) -> int:
    return int(round(budget * 1_000_000))


def final_run_name(root: int) -> str:
    return (
        f"{SCENARIO}--seed-{root}--work-56000000--all-arms--"
        "deep-search-feedback--engine-openrouter--seed-archive-5--batch-3"
    )


def tuned_run_name(root: int, mix: str, budget: float) -> str:
    return (
        f"{SCENARIO}--seed-{root}--work-{microwork(budget)}--non-llm--"
        f"deep-search-feedback--engine-openrouter--seed-archive-5--batch-3--"
        f"gp-mix-{mix}"
    )


def cell_run_dir(root: int, mix: str, budget: float) -> Path:
    if mix == "current" and budget == 56.0:
        return RUNS_FINAL / final_run_name(root)
    primary = RUNS_ROOT / tuned_run_name(root, mix, budget)
    if (primary / "report.json").is_file():
        return primary
    # Author-side orchestrator started current@224 into a side runs-root while mix56
    # was still going, so the two launchers could not collide on the same directory.
    if mix == "current" and budget == 224.0:
        early = EARLY_224_ROOT / tuned_run_name(root, mix, budget)
        if (early / "report.json").is_file():
            return early
    return primary


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    """2 * the smaller Binomial(m, 1/2) tail, capped at one. Copied from the prereg."""
    m = positive + negative
    if m == 0:
        return 1.0
    tail_at = min(positive, negative)
    tail_count = sum(math.comb(m, value) for value in range(tail_at + 1))
    return min(1.0, 2.0 * tail_count / (2**m))


def lexicographic_direction(
    gp_feasible: bool,
    gp_energy: float | None,
    llm_feasible: bool,
    llm_energy: float | None,
) -> tuple[int, str, float | None]:
    """Positive favours the LLM. Both infeasible or equal energy is a tie."""
    if llm_feasible and gp_feasible:
        assert gp_energy is not None and llm_energy is not None
        delta = gp_energy - llm_energy
        if delta > 0:
            return 1, "both_feasible_llm", delta
        if delta < 0:
            return -1, "both_feasible_gp", delta
        return 0, "both_feasible_tie", 0.0
    if llm_feasible and not gp_feasible:
        return 1, "llm_only_feasible", None
    if gp_feasible and not llm_feasible:
        return -1, "gp_only_feasible", None
    return 0, "both_infeasible", None


def token_levenshtein(left: str, right: str) -> dict[str, float | int]:
    a = left.split()
    b = right.split()
    if not a and not b:
        return {"tokens_a": 0, "tokens_b": 0, "levenshtein": 0, "normalized": 0.0}
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        cur = [i]
        for j, token_b in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (token_a != token_b)
            cur.append(min(ins, delete, sub))
        prev = cur
    distance = prev[-1]
    return {
        "tokens_a": len(a),
        "tokens_b": len(b),
        "levenshtein": distance,
        "normalized": distance / max(len(a), len(b)),
    }


def _load_events(archive: Path) -> list[dict[str, Any]]:
    path = archive / "events.jsonl"
    compressed = archive / "events.jsonl.gz"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    elif compressed.is_file():
        import gzip

        text = gzip.open(compressed, "rt", encoding="utf-8").read()
    else:
        return []
    events = []
    for line in text.splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def archive_diagnostics(run_dir: Path, arm_archive: str) -> dict[str, Any] | None:
    archive = run_dir / "archives" / arm_archive
    if not archive.is_dir():
        return None
    events = _load_events(archive)
    if not events:
        return None
    created = sum(1 for event in events if event.get("type") == "candidate_created")
    completed = [event["payload"] for event in events if event.get("type") == "candidate_completed"]
    search = [row for row in completed if int(row.get("proposal_seq") or 0) >= INIT_POP]
    valid = 0
    accepted = 0
    l1_reject = 0
    l1_seen = 0
    promoted_l2 = 0
    attempts = 0
    invalid_attempts = 0
    operations: Counter[str] = Counter()
    energies: list[float] = []
    for row in completed:
        proposal = row.get("proposal") or {}
        meta = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
        if proposal.get("source") or proposal.get("valid"):
            valid += 1
        attempts += int(meta.get("attempts") or 0)
        invalid_attempts += int(meta.get("invalid_attempts") or 0)
        operation = meta.get("operation")
        if isinstance(operation, str):
            operations[operation] += 1
        decision = row.get("decision") or {}
        if isinstance(decision, dict) and (
            decision.get("entered_survivor")
            or (decision.get("elite_view") or {}).get("entered")
        ):
            accepted += 1
        evaluation = row.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            continue
        if evaluation.get("promoted_to_l2"):
            promoted_l2 += 1
        for rec in evaluation.get("trace") or []:
            if rec.get("rung") == "L1":
                l1_seen += 1
                if rec.get("decision") in {"reject", "invalid"}:
                    l1_reject += 1
        fitness = (evaluation.get("evaluation") or {}).get("fitness") or {}
        energy = fitness.get("normalized_energy")
        if fitness.get("feasible") and isinstance(energy, (int, float)):
            energies.append(float(energy))
    energies.sort()
    deciles = {}
    if energies:
        for k in range(1, 10):
            index = min(len(energies) - 1, max(0, round((k / 10) * (len(energies) - 1))))
            deciles[f"p{k}0"] = energies[index]
    champion_source = None
    champions = champions_from_run(run_dir)
    arm_name = "same_dsl_gp" if arm_archive == "same_dsl_gp" else "llm"
    if arm_name in champions:
        champion_source = champions[arm_name]
    starting = STARTING_POLICY.read_text(encoding="utf-8") if STARTING_POLICY.is_file() else ""
    distance = (
        token_levenshtein(starting, champion_source)
        if champion_source is not None
        else None
    )
    report_path = run_dir / "report.json"
    search_delta = None
    best_absolute = None
    best_seq = None
    if report_path.is_file():
        body = json.loads(report_path.read_text())
        body = body.get("payload", body)
        arm = (body.get("arms") or {}).get(arm_name) or {}
        search_delta = arm.get("search_delta")
        best_absolute = arm.get("best_absolute")
        found = arm.get("best_found_at") or {}
        best_seq = found.get("proposal_seq")
    return {
        "run": run_dir.name,
        "proposals_created": created,
        "candidates_completed": len(completed),
        "search_candidates": len(search),
        "offspring_valid": valid,
        "offspring_accepted": accepted,
        "l1_seen": l1_seen,
        "l1_rejections": l1_reject,
        "l1_rejection_share": (l1_reject / l1_seen) if l1_seen else None,
        "promoted_to_l2": promoted_l2,
        "operator_attempts": attempts,
        "invalid_attempts": invalid_attempts,
        "operations": dict(operations),
        "n_feasible_energies": len(energies),
        "feasible_energies": energies,
        "energy_deciles": deciles,
        "search_delta": search_delta,
        "best_absolute": best_absolute,
        "best_proposal_seq": best_seq,
        "champion_from_initialization": (
            None if best_seq is None else int(best_seq) < INIT_POP
        ),
        "starting_policy_token_distance": distance,
    }


def load_llm_heldout() -> dict[int, dict[str, Any]]:
    payload = json.loads(LLM_RESCORE.read_text())["payload"]
    out: dict[int, dict[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("arm") != "llm":
            continue
        seed = int(row["root_seed"])
        if seed in ROOTS:
            out[seed] = row
    return out


def load_current_gp_heldout() -> dict[int, dict[str, Any]]:
    payload = json.loads(LLM_RESCORE.read_text())["payload"]
    out: dict[int, dict[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("arm") != "same_dsl_gp":
            continue
        seed = int(row["root_seed"])
        if seed in ROOTS:
            out[seed] = row
    return out


def sign_test(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    positive = negative = ties = 0
    categories: Counter[str] = Counter()
    deltas: list[float] = []
    for pair in pairs:
        direction, category, delta = lexicographic_direction(
            pair["gp_feasible"],
            pair["gp_energy"],
            pair["llm_feasible"],
            pair["llm_energy"],
        )
        categories[category] += 1
        if direction > 0:
            positive += 1
        elif direction < 0:
            negative += 1
        else:
            ties += 1
        if delta is not None:
            deltas.append(delta)
    non_tied = positive + negative
    attainable = non_tied >= MIN_SIGNIFICANT_M
    exact_p = exact_two_sided_sign_p(positive, negative) if attainable else None
    return {
        "llm_wins": positive,
        "gp_wins": negative,
        "ties": ties,
        "non_tied": non_tied,
        "exact_p": exact_p,
        "significant": exact_p is not None and exact_p <= ALPHA,
        "attainable": attainable,
        "categories": dict(categories),
        "both_feasible_mean_d": statistics.fmean(deltas) if deltas else None,
        "n_roots": len(pairs),
    }


def rescore_champion(source: str, root: int, task) -> dict[str, Any]:
    scenario_id = task.scenarios[0].scenario_id
    seeds = fresh_grid(root, scenario_id)
    seeded = dataclasses.replace(
        task,
        scenarios=tuple(ScenarioGrid(item.scenario_id, item.path, seeds) for item in task.scenarios),
    )
    fresh = evaluate_policy_source(source, seeded, proposal_seq=980000 + (root % 1000))
    worst = min((min(item.slo_achieved.values()) for item in fresh.instances), default=1.0)
    return {
        "fresh_grid_energy": fresh.fitness.normalized_energy,
        "fresh_grid_feasible": fresh.fitness.feasible,
        "fresh_grid_worst_slo": worst,
        "fresh_seeds": list(seeds),
    }


def collect_cells(*, rescore: bool) -> dict[str, Any]:
    llm = load_llm_heldout()
    current_gp = load_current_gp_heldout()
    task = load_task_evaluation_config(TASK_CONFIG) if rescore else None
    cache_dir = HERE / "rescore_cache"
    cache_dir.mkdir(exist_ok=True)
    cells = []
    missing = []
    for mix, budget in CELLS:
        per_root = []
        diagnostics = []
        for root in ROOTS:
            run_dir = cell_run_dir(root, mix, budget)
            reused = mix == "current" and budget == 56.0
            complete = (run_dir / "report.json").is_file()
            if not complete:
                missing.append({"mix": mix, "budget": budget, "root": root, "run": str(run_dir)})
                continue
            diag = archive_diagnostics(run_dir, "same_dsl_gp")
            if diag is not None:
                diagnostics.append(diag)
            if reused:
                gp_row = current_gp[root]
                held = {
                    "fresh_grid_energy": gp_row.get("fresh_grid_energy"),
                    "fresh_grid_feasible": gp_row["fresh_grid_feasible"],
                    "fresh_grid_worst_slo": gp_row.get("fresh_grid_worst_slo"),
                    "source": "FINALIST_RESCORE_runs_final.json",
                }
            else:
                cache = cache_dir / f"{mix}-{budget}-{root}.json"
                if cache.is_file():
                    held = json.loads(cache.read_text())
                elif rescore:
                    champions = champions_from_run(run_dir)
                    source = champions.get("same_dsl_gp")
                    if source is None:
                        held = {
                            "fresh_grid_energy": None,
                            "fresh_grid_feasible": False,
                            "fresh_grid_worst_slo": None,
                            "source": "missing-champion",
                        }
                    else:
                        print(f"rescoring mix={mix} budget={budget} root={root}", flush=True)
                        held = rescore_champion(source, root, task)
                        held["source"] = "evaluate_policy_source"
                        cache.write_text(json.dumps(held, sort_keys=True) + "\n")
                else:
                    held = None
            llm_row = llm[root]
            pair = None
            if held is not None:
                direction, category, delta = lexicographic_direction(
                    bool(held["fresh_grid_feasible"]),
                    held.get("fresh_grid_energy"),
                    bool(llm_row["fresh_grid_feasible"]),
                    llm_row.get("fresh_grid_energy"),
                )
                pair = {
                    "root": root,
                    "gp_feasible": bool(held["fresh_grid_feasible"]),
                    "gp_energy": held.get("fresh_grid_energy"),
                    "llm_feasible": bool(llm_row["fresh_grid_feasible"]),
                    "llm_energy": llm_row.get("fresh_grid_energy"),
                    "d": delta,
                    "direction": direction,
                    "category": category,
                    "heldout_source": held.get("source"),
                }
                per_root.append(pair)
        cell = {
            "mix": mix,
            "budget": budget,
            "reused_runs_final": mix == "current" and budget == 56.0,
            "roots_complete": len(diagnostics),
            "comparison": sign_test(per_root) if per_root else None,
            "pairs": per_root,
            "diagnostics": _aggregate_diagnostics(diagnostics),
        }
        cells.append(cell)
    llm_diag = []
    for root in ROOTS:
        run_dir = RUNS_FINAL / final_run_name(root)
        if (run_dir / "report.json").is_file():
            item = archive_diagnostics(run_dir, "llm")
            if item is not None:
                llm_diag.append(item)
    oracle_pairs = _oracle_pairs(cells, llm)
    return {
        "schema": "evolved-continuum-gp-tuning-v1",
        "boundary": BOUNDARY,
        "roots": list(ROOTS),
        "mixes": MIXES,
        "cells": cells,
        "missing": missing,
        "llm_diagnostics": _aggregate_diagnostics(llm_diag),
        "headline_llm_56_vs_gp_current_224": _headline(cells, "current", 224.0),
        "gp_oracle_vs_llm_56": sign_test(oracle_pairs) if oracle_pairs else None,
        "gp_oracle_pairs": oracle_pairs,
        "current_56_reuse": (
            "Sealed runs_final archives and FINALIST_RESCORE_runs_final.json "
            "held-out rows for same_dsl_gp at 56.0; not re-run."
        ),
        "task_config": str(TASK_CONFIG),
        "task_config_sha256": hashlib.sha256(TASK_CONFIG.read_bytes()).hexdigest(),
        "llm_rescore": str(LLM_RESCORE),
        "llm_rescore_sha256": hashlib.sha256(LLM_RESCORE.read_bytes()).hexdigest(),
    }


def _aggregate_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    distances = []
    champion_energies = []
    from_init = 0
    all_energies: list[float] = []
    for item in items:
        if item.get("best_absolute") is not None:
            champion_energies.append(item["best_absolute"])
        if item.get("champion_from_initialization"):
            from_init += 1
        dist = item.get("starting_policy_token_distance") or {}
        if dist.get("normalized") is not None:
            distances.append(dist["normalized"])
        all_energies.extend(item.get("feasible_energies") or [])
    distinct = sorted({round(value, 9) for value in champion_energies})
    return {
        "roots": len(items),
        "proposals_created": sum(item["proposals_created"] for item in items),
        "offspring_valid": sum(item["offspring_valid"] for item in items),
        "offspring_accepted": sum(item["offspring_accepted"] for item in items),
        "l1_rejections": sum(item["l1_rejections"] for item in items),
        "l1_seen": sum(item["l1_seen"] for item in items),
        "l1_rejection_share": (
            sum(item["l1_rejections"] for item in items) / sum(item["l1_seen"] for item in items)
            if sum(item["l1_seen"] for item in items)
            else None
        ),
        "promoted_to_l2": sum(item["promoted_to_l2"] for item in items),
        "operator_attempts": sum(item["operator_attempts"] for item in items),
        "invalid_attempts": sum(item["invalid_attempts"] for item in items),
        "valid_rate": (
            sum(item["offspring_valid"] for item in items)
            / sum(item["candidates_completed"] for item in items)
            if sum(item["candidates_completed"] for item in items)
            else None
        ),
        "acceptance_rate": (
            sum(item["offspring_accepted"] for item in items)
            / sum(item["candidates_completed"] for item in items)
            if sum(item["candidates_completed"] for item in items)
            else None
        ),
        "pooled_energy_deciles": _deciles(all_energies),
        "distinct_champion_energies": distinct,
        "n_distinct_champion_energies": len(distinct),
        "champions_from_initialization": from_init,
        "mean_token_distance_from_graph_headroom": (
            statistics.fmean(distances) if distances else None
        ),
        "per_root": items,
    }


def _deciles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for k in range(1, 10):
        index = min(len(ordered) - 1, max(0, round((k / 10) * (len(ordered) - 1))))
        out[f"p{k}0"] = ordered[index]
    return out


def _oracle_pairs(cells: list[dict[str, Any]], llm: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    by_root: dict[int, list[dict[str, Any]]] = {root: [] for root in ROOTS}
    for cell in cells:
        for pair in cell.get("pairs") or []:
            by_root[pair["root"]].append({**pair, "mix": cell["mix"], "budget": cell["budget"]})
    oracle = []
    for root, options in by_root.items():
        if not options:
            continue
        feasible = [row for row in options if row["gp_feasible"]]
        if feasible:
            best = min(feasible, key=lambda row: row["gp_energy"])
        else:
            best = options[0]
        llm_row = llm[root]
        direction, category, delta = lexicographic_direction(
            best["gp_feasible"],
            best["gp_energy"],
            bool(llm_row["fresh_grid_feasible"]),
            llm_row.get("fresh_grid_energy"),
        )
        oracle.append(
            {
                "root": root,
                "gp_feasible": best["gp_feasible"],
                "gp_energy": best["gp_energy"],
                "llm_feasible": bool(llm_row["fresh_grid_feasible"]),
                "llm_energy": llm_row.get("fresh_grid_energy"),
                "d": delta,
                "direction": direction,
                "category": category,
                "chosen_mix": best.get("mix"),
                "chosen_budget": best.get("budget"),
            }
        )
    return oracle


def _headline(cells: list[dict[str, Any]], mix: str, budget: float) -> dict[str, Any] | None:
    for cell in cells:
        if cell["mix"] == mix and cell["budget"] == budget:
            return cell.get("comparison")
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# GP operator-mix and budget sweep",
        "",
        f"**Boundary:** {payload['boundary']}",
        "",
        payload["current_56_reuse"],
        "",
        "## Sign tests versus LLM at 56.0",
        "",
        "| mix | budget | roots | LLM wins | GP wins | ties | exact p | attainable |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in payload["cells"]:
        cmp_ = cell.get("comparison")
        if cmp_ is None:
            lines.append(
                f"| `{cell['mix']}` | {cell['budget']} | {cell['roots_complete']} | — | — | — | — | no |"
            )
            continue
        p_value = "—" if cmp_["exact_p"] is None else f"{cmp_['exact_p']:.6g}"
        lines.append(
            f"| `{cell['mix']}` | {cell['budget']} | {cmp_['n_roots']} | "
            f"{cmp_['llm_wins']} | {cmp_['gp_wins']} | {cmp_['ties']} | {p_value} | "
            f"{'yes' if cmp_['attainable'] else 'no'} |"
        )
    oracle = payload.get("gp_oracle_vs_llm_56")
    if oracle:
        p_value = "—" if oracle["exact_p"] is None else f"{oracle['exact_p']:.6g}"
        lines.extend(
            [
                "",
                f"GP oracle (best mix/budget per root) vs LLM at 56.0: "
                f"LLM {oracle['llm_wins']} – GP {oracle['gp_wins']}, "
                f"ties {oracle['ties']}, p = {p_value}.",
            ]
        )
    lines.extend(["", "## Operator health (pooled)", ""])
    lines.append(
        "| arm / cell | roots | created | valid | accepted | L1 reject share | "
        "distinct champion E | from init | mean token distance |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    llm_diag = payload.get("llm_diagnostics")
    if llm_diag:
        lines.append(
            "| `llm@56 runs_final` | "
            f"{llm_diag['roots']} | {llm_diag['proposals_created']} | "
            f"{llm_diag['offspring_valid']} | {llm_diag['offspring_accepted']} | "
            f"{_fmt(llm_diag['l1_rejection_share'])} | "
            f"{llm_diag['n_distinct_champion_energies']} | "
            f"{llm_diag['champions_from_initialization']} | "
            f"{_fmt(llm_diag['mean_token_distance_from_graph_headroom'])} |"
        )
    for cell in payload["cells"]:
        diag = cell.get("diagnostics")
        if not diag:
            continue
        lines.append(
            f"| `gp {cell['mix']}@{cell['budget']}` | "
            f"{diag['roots']} | {diag['proposals_created']} | "
            f"{diag['offspring_valid']} | {diag['offspring_accepted']} | "
            f"{_fmt(diag['l1_rejection_share'])} | "
            f"{diag['n_distinct_champion_energies']} | "
            f"{diag['champions_from_initialization']} | "
            f"{_fmt(diag['mean_token_distance_from_graph_headroom'])} |"
        )
    if payload.get("missing"):
        lines.extend(["", "## Missing cells", ""])
        for item in payload["missing"]:
            lines.append(
                f"- mix `{item['mix']}` budget {item['budget']} root {item['root']}"
            )
    lines.extend(["", "payload_sha256 is in `GP_TUNING.json`.", ""])
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


PREKILL_NUMERIC_HEAVY_56 = {
    800000801: "0cd4c7ed566cca2586f241603022a0dd9cb563d60551874256b72fb04a8353c1",
    800000802: "43d7bbbfdecee07770393b3351d0b70c30d828944b0258a6e53f9b6355958cb0",
    800000803: "025e483976f06c9eba0407f88c19f909c147e6d53ac5b0f998c3d842913196e3",
}


def _archive_digest(archive: Path) -> str | None:
    if not archive.is_dir():
        return None
    digest = hashlib.sha256()
    paths = [archive / "manifest.json"]
    events = archive / "events.jsonl"
    if events.is_file():
        paths.append(events)
    usage = archive / "proposal-engine-usage.jsonl"
    if usage.is_file():
        paths.append(usage)
    paths.extend(sorted((archive / "candidates").glob("*.ps")))
    for path in paths:
        if not path.is_file():
            return None
        content = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(b"\x00")
        digest.update(str(len(content)).encode())
        digest.update(b"\x00")
        digest.update(content)
    return digest.hexdigest()


def _arm_archive_sha256(run_dir: Path) -> str | None:
    result = run_dir / "arms/same_dsl_gp/result.json"
    if not result.is_file():
        return None
    body = json.loads(result.read_text())
    body = body.get("payload", body)
    health = body.get("mechanical_health") or {}
    digest = health.get("archive_sha256")
    return digest if isinstance(digest, str) else None


def verify_resume_digests() -> dict[str, Any]:
    rows = []
    for root, logged in PREKILL_NUMERIC_HEAVY_56.items():
        run_dir = cell_run_dir(root, "numeric-heavy", 56.0)
        current = _archive_digest(run_dir / "archives/same_dsl_gp")
        sealed = _arm_archive_sha256(run_dir)
        report = (run_dir / "report.json").is_file()
        chosen = sealed or current
        rows.append(
            {
                "root": root,
                "log_archive_sha256": logged,
                "current_archive_sha256": current,
                "sealed_arm_archive_sha256": sealed,
                "report_present": report,
                "matches_log": chosen == logged if chosen else None,
            }
        )
    return {
        "killed_at": "2026-09-02T14:19",
        "note": (
            "numeric-heavy roots 801-803 had GP campaign_terminal and a printed "
            "archive_sha256 before the host kill; report.json was missing. "
            "Resume is valid iff the digest still matches the log."
        ),
        "all_match": all(row["matches_log"] for row in rows if row["matches_log"] is not None),
        "roots": rows,
    }


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.7g}"


def _fmt_energy(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6f}"


def _category_label(category: str) -> str:
    return {
        "both_feasible_llm": "both feasible, LLM",
        "both_feasible_gp": "both feasible, GP",
        "both_feasible_tie": "both feasible, tie",
        "llm_only_feasible": "LLM-only feasible",
        "gp_only_feasible": "GP-only feasible",
        "both_infeasible": "both infeasible",
    }.get(category, category)


def _cell_note(cell: dict[str, Any]) -> str:
    mix, budget = cell["mix"], cell["budget"]
    n = cell["roots_complete"]
    if mix == "current" and budget == 56.0:
        return "reused `runs_final` + sealed rescore"
    if n >= 8:
        return "complete"
    if n == 0:
        if mix != "current" and budget == 56.0:
            return "queued or running"
        if budget == 224.0:
            return "headline 4× cell; after mix56"
        if budget == 112.0:
            return "after 224.0"
        return "not started"
    finished = sorted(pair["root"] % 1000 for pair in cell.get("pairs") or [])
    roots = ", ".join(f"{item:03d}" for item in finished)
    if n >= MIN_SIGNIFICANT_M:
        return f"incomplete: {n}/8 roots ({roots}); sign test attainable"
    return f"underpowered: {n}/8 roots ({roots})"


def _paper_verdict(payload: dict[str, Any]) -> list[str]:
    cells = { (cell["mix"], cell["budget"]): cell for cell in payload["cells"] }
    current56 = cells[("current", 56.0)]
    scored = [cell for cell in payload["cells"] if cell.get("comparison")]
    gp_wins = [
        cell
        for cell in scored
        if (cell["comparison"] or {}).get("gp_wins", 0) > 0
    ]
    all_complete = all(cell["roots_complete"] >= 8 for cell in payload["cells"])
    headline = payload.get("headline_llm_56_vs_gp_current_224")
    oracle = payload.get("gp_oracle_vs_llm_56")
    lines = [
        "**Is the GP control competent?** On the published mix and budget, it is a working "
        "operator that does not search. It is not a broken proposer: every completed "
        "candidate inspected so far is a valid, type-checked PlacementScript, a minority "
        "enter the archive, and L1 rejection is low. It is also not a search method that "
        "found better programs on `current`@56: 7 of 8 champions are initialization "
        "members, search delta is 0.000 on those seven, and held-out energy sits on the "
        "0.832–0.834 plateau (or is infeasible) on every root. The referee's sentence is "
        "slightly too strong — GP *did* move on root 803 at `current`@56 — and essentially right.",
        "",
    ]
    complete_mix56 = [
        cell
        for cell in scored
        if cell["budget"] == 56.0 and cell["roots_complete"] >= 8 and cell["mix"] != "current"
    ]
    remaining_mix = [
        mix
        for mix in ("numeric-heavy", "subtree-heavy", "crossover-heavy")
        if not any(cell["mix"] == mix and cell["roots_complete"] >= 8 for cell in scored)
    ]
    if gp_wins:
        details = ", ".join(
            f"`{cell['mix']}`@{cell['budget']} GP {cell['comparison']['gp_wins']}–"
            f"LLM {cell['comparison']['llm_wins']}"
            for cell in gp_wins
        )
        lines.append(
            f"**Does any GP configuration at any budget reach the LLM's held-out energy?** "
            f"Yes, at least on some roots: {details}. The paper must report those wins; "
            "the GP-oracle column exists so a referee cannot say the mix was chosen after "
            "the fact."
        )
    elif all_complete:
        lines.append(
            "**Does any GP configuration at any budget reach the LLM's held-out energy?** "
            "No. Every scored cell, including `current`@224 (4× work) and the GP oracle "
            "(best mix/budget per root), loses to LLM@56 on the fresh grid. The under-tuned-"
            "baseline objection is closed for this operator family on this scenario."
        )
    else:
        finished_mix = ", ".join(
            f"`{cell['mix']}`@56 lost {cell['comparison']['llm_wins']}–"
            f"{cell['comparison']['gp_wins']} (p = {_fmt_p(cell['comparison'].get('exact_p'))})"
            for cell in complete_mix56
            if cell.get("comparison")
        )
        leftover = ", ".join(f"`{mix}`" for mix in remaining_mix) or "none"
        extra = (
            f" Completed mix retuning so far: {finished_mix}. Still running: {leftover} "
            "and the 4× budget cell."
            if finished_mix
            else " Mix retuning and 4× budget are the cells that could still falsify that statement."
        )
        lines.append(
            "**Does any GP configuration at any budget reach the LLM's held-out energy?** "
            "Not in the cells scored so far." + extra + " Until every remaining cell finishes, "
            "the paper must not claim the objection is closed. Incomplete cells are labelled "
            "underpowered or incomplete."
        )
    lines.extend(["", "**How should outline §10 change, given what is in hand today?**", ""])
    lines.append(
        "- Keep the existing limitation that 8 of 9 GP held-out values sit at the "
        "starting-policy plateau. Add, from this work, that this is not a silent operator "
        "failure: validity rate 1.0 on `current`@56, archive acceptance 0.26, and a "
        "documented 7/8 initialization-champion rate on roots 801–808."
    )
    if all_complete and not gp_wins:
        lines.append(
            "- Add: GP was retuned over the fixed mix grid and given 4× budget and still "
            "lost to LLM@56, including under a GP-oracle that takes the best cell per root. "
            "The 16–0 result is then not an under-tuned-baseline artefact."
        )
    elif gp_wins:
        lines.append(
            "- Report every GP win and the configuration that produced it. Do not collapse "
            "the grid to the published mix after seeing a later cell succeed."
        )
    elif complete_mix56:
        names = ", ".join(f"`{cell['mix']}`@56" for cell in complete_mix56)
        lines.append(
            f"- {names} is complete and lost 8–0. Do **not** yet add the full sentence "
            "\"GP was retuned over the mix grid and given 4× budget and still lost\" until "
            "the remaining mixes and `current`@224 are scored."
        )
    else:
        lines.append(
            "- Do **not** yet add \"GP was retuned and given 4× budget and still lost.\" "
            "That sentence is the point of 2.2–2.3 and is not earned until those cells are scored."
        )
    if headline and headline.get("n_roots", 0) >= 8:
        lines.append(
            f"- Headline LLM@56 vs `current`@224: LLM {headline['llm_wins']} – "
            f"GP {headline['gp_wins']}, ties {headline['ties']}, p = {_fmt_p(headline.get('exact_p'))}."
        )
    else:
        lines.append(
            "- Headline LLM@56 vs `current`@224 is not yet available; treat the 4× test as "
            "underpowered until those eight roots finish."
        )
    if oracle:
        lines.append(
            f"- GP oracle vs LLM@56: LLM {oracle['llm_wins']} – GP {oracle['gp_wins']}, "
            f"ties {oracle['ties']}, p = {_fmt_p(oracle.get('exact_p'))}."
        )
    cmp56 = current56.get("comparison") or {}
    if cmp56.get("llm_wins") == 8:
        lines.append(
            "- Root 805's +0.000267 technical win is already in §10; it reappears here and "
            "does not change the sign test."
        )
    return lines


def render_results_md(payload: dict[str, Any]) -> str:
    resume = payload.get("resume_digests") or {}
    missing = payload.get("missing") or []
    n_missing = len(missing)
    n_complete = sum(cell["roots_complete"] for cell in payload["cells"])
    lines = [
        "# RESULTS: is the GP control competent?",
        "",
        "**Boundary:** quarantined development-only. Zero LLM calls. Never confirmatory.",
        "",
        "This is checklist item 2 of `docs/PAPER_CHECKLIST_20260902.md`. The referee "
        "objection is: eight of nine GP held-out values sit at the starting policy, so "
        "the 16–0 LLM win may be against an under-tuned operator that never moved. The "
        "test is whether GP, retuned over a fixed operator-mix grid and given up to 4× "
        "the work budget, can match the LLM at 1×.",
        "",
        "Either answer is the result. The grid was fixed before any new number existed.",
        "",
        "## (a) What was done",
        "",
        "1. **2.1 Config exposure.** Optional `--gp-config PATH` on `run.sh`, the harness "
        "runner, and `evolution/run_campaign.py`. JSON fields: `crossover_probability`, "
        "`subtree_mutation_probability`, `numeric_mutation_probability`, `max_attempts`, "
        "plus a `label`. When the flag is absent, `PlacementScriptGPOperator()` is still "
        "constructed with no arguments, run-directory names are unchanged, and no "
        "`gp_operator_config.json` is written. When it is present, the config is copied "
        "to `inputs/`, its digest is in the preflight receipt and the report, and the "
        "run directory gains `--gp-mix-<label>`. Re-verified green on resume: "
        "`tests/test_gp_operator_config_harness.py`, `tests/test_evolution_gp_operator.py`.",
        "2. **2.2 Mix grid.** Roots 800000801–808, budget 56.0, mixes `current` / "
        "`numeric-heavy` / `subtree-heavy` / `crossover-heavy`. **`current` at 56.0 is "
        "reused from sealed `runs_final`**, not re-run. The other three mixes are new "
        "non-LLM harness runs (the harness also runs classic GPHH and random-AST; that "
        "cost was accepted).",
        "3. **2.3 Budget sweep.** Same eight roots, mix `current`, budgets 224.0 then "
        "112.0. 224.0 is first because the headline is 1× LLM versus 4× GP.",
        "4. **2.4 Held-out.** `d = fresh_energy(GP) − fresh_energy(LLM at 56.0)` with the "
        "pre-registered lexicographic rule (feasible beats infeasible; both infeasible is "
        "a tie). Exact two-sided sign test, α = 0.05, m ≥ 6 non-tied roots. LLM numbers "
        "are never regenerated; they are the sealed rows in `FINALIST_RESCORE_runs_final.json`.",
        "5. **2.5 Diagnostics.** From archives: proposals created, valid offspring, archive "
        "acceptances, L1 rejection share, candidate energy deciles, distinct champion "
        "energies, whether the champion is an initialization member, token-Levenshtein "
        "distance to `graph_headroom.ps`. LLM arm from the same eight `runs_final` roots, "
        "side by side.",
        "6. **2.6 Existing evidence** is restated below, not recomputed.",
        "",
        f"Resume (second launch, 2026-09-02): first `mix56` wave was killed ~14:19 after "
        f"three `numeric-heavy` roots (801–803) finished the GP campaign but before "
        f"`report.json`. Grid relaunched via `daemonize.py` (`os.setsid` + exec of "
        f"`run_pipeline.sh`; macOS has no `setsid(1)`) so a third kill does not stop "
        f"the launcher. Complete cells {n_complete} root-runs; "
        f"{n_missing} still missing.",
        "",
        "## Resume digest check (numeric-heavy 801–803)",
        "",
        payload.get("resume_digests", {}).get(
            "note",
            "numeric-heavy 801-803 were partial at kill; see `resume_digests.json`.",
        ),
        "",
        "| root | log sha256 | current sha256 | matches log | report.json |",
        "|---|---|---|---|---|",
    ]
    for row in resume.get("roots") or []:
        current = row.get("sealed_arm_archive_sha256") or row.get("current_archive_sha256")
        match = row.get("matches_log")
        match_s = "yes" if match else ("—" if match is None else "NO — using new run")
        lines.append(
            f"| {row['root']} | `{row['log_archive_sha256']}` | "
            f"`{current or '—'}` | {match_s} | "
            f"{'yes' if row.get('report_present') else 'no'} |"
        )
    if not resume.get("roots"):
        lines.append("| — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## (b) The numbers that matter",
            "",
            "### Held-out, LLM at 56.0 versus each GP cell",
            "",
            "Positive `d` favours the LLM. `p = 0.0078125` is 2/2⁸, the smallest "
            "attainable value at 8 non-tied roots.",
            "",
            "| mix | budget | roots scored | LLM wins | GP wins | ties | exact p | notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in payload["cells"]:
        cmp_ = cell.get("comparison")
        note = _cell_note(cell)
        if cmp_ is None:
            lines.append(
                f"| `{cell['mix']}` | {cell['budget']} | {cell['roots_complete']} | "
                f"— | — | — | — | {note} |"
            )
            continue
        gp_wins = cmp_["gp_wins"]
        llm_wins = cmp_["llm_wins"]
        gp_mark = f"**{gp_wins}**" if gp_wins else str(gp_wins)
        llm_mark = f"**{llm_wins}**" if llm_wins else str(llm_wins)
        lines.append(
            f"| `{cell['mix']}` | {cell['budget']} | {cmp_['n_roots']} | "
            f"{llm_mark} | {gp_mark} | {cmp_['ties']} | {_fmt_p(cmp_.get('exact_p'))} | {note} |"
        )
    for cell in payload["cells"]:
        pairs = cell.get("pairs") or []
        if not pairs:
            continue
        lines.extend(
            [
                "",
                f"**Per-root `{cell['mix']}`@{cell['budget']} versus LLM@56** "
                "(fresh 8-seed grid, calibrated config):",
                "",
                "| root | LLM feas. | LLM energy | GP feas. | GP energy | category |",
                "|---|---|---:|---|---:|---|",
            ]
        )
        for pair in pairs:
            lines.append(
                f"| {pair['root'] % 1000} | "
                f"{'yes' if pair['llm_feasible'] else 'no'} | "
                f"{_fmt_energy(pair.get('llm_energy'))} | "
                f"{'yes' if pair['gp_feasible'] else 'no'} | "
                f"{_fmt_energy(pair.get('gp_energy'))} | "
                f"{_category_label(pair['category'])} |"
            )
        both = [
            pair["d"]
            for pair in pairs
            if pair.get("d") is not None and pair["category"].startswith("both_feasible")
        ]
        if both:
            lines.append(
                f"\nOn the {len(both)} both-feasible roots, mean `d` = {statistics.fmean(both):.3f}."
            )
    oracle = payload.get("gp_oracle_vs_llm_56")
    oracle_pairs = payload.get("gp_oracle_pairs") or []
    if oracle:
        lines.extend(
            [
                "",
                f"**GP oracle** (best mix/budget per root among scored cells): "
                f"LLM {oracle['llm_wins']} – GP {oracle['gp_wins']}, ties {oracle['ties']}, "
                f"p = {_fmt_p(oracle.get('exact_p'))}"
                + (
                    " (underpowered: not every cell is scored)"
                    if n_missing
                    else ""
                )
                + ".",
            ]
        )
        if oracle_pairs:
            lines.extend(
                [
                    "",
                    "| root | chosen mix | chosen budget | GP energy | LLM energy | category |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            for pair in oracle_pairs:
                lines.append(
                    f"| {pair['root'] % 1000} | `{pair.get('chosen_mix')}` | "
                    f"{pair.get('chosen_budget')} | {_fmt_energy(pair.get('gp_energy'))} | "
                    f"{_fmt_energy(pair.get('llm_energy'))} | "
                    f"{_category_label(pair['category'])} |"
                )
    headline = payload.get("headline_llm_56_vs_gp_current_224")
    lines.append("")
    if headline and headline.get("n_roots"):
        lines.append(
            f"**Headline LLM@56 versus GP `current`@224:** LLM {headline['llm_wins']} – "
            f"GP {headline['gp_wins']}, ties {headline['ties']}, "
            f"p = {_fmt_p(headline.get('exact_p'))}, n = {headline['n_roots']}."
            + (
                " Underpowered (not all eight roots)."
                if headline["n_roots"] < 8
                else ""
            )
        )
    else:
        lines.append(
            "**Headline LLM@56 versus GP `current`@224:** not yet available. "
            "Until those eight roots finish, the 4× test is underpowered."
        )
    lines.extend(
        [
            "",
            "### Operator health",
            "",
            "| arm / cell | roots | created | valid | accepted | L1 reject share | "
            "distinct champion E | from init | mean token distance |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    llm_diag = payload.get("llm_diagnostics")
    if llm_diag:
        lines.append(
            "| LLM@56 (`runs_final`) | "
            f"{llm_diag['roots']} | {llm_diag['proposals_created']} | "
            f"{llm_diag['offspring_valid']} | {llm_diag['offspring_accepted']} | "
            f"{_fmt(llm_diag['l1_rejection_share'])} | "
            f"{llm_diag['n_distinct_champion_energies']} | "
            f"{llm_diag['champions_from_initialization']} | "
            f"{_fmt(llm_diag['mean_token_distance_from_graph_headroom'])} |"
        )
    for cell in payload["cells"]:
        diag = cell.get("diagnostics")
        if not diag:
            continue
        lines.append(
            f"| GP `{cell['mix']}`@{cell['budget']} | "
            f"{diag['roots']} | {diag['proposals_created']} | "
            f"{diag['offspring_valid']} | {diag['offspring_accepted']} | "
            f"{_fmt(diag['l1_rejection_share'])} | "
            f"{diag['n_distinct_champion_energies']} | "
            f"{diag['champions_from_initialization']} | "
            f"{_fmt(diag['mean_token_distance_from_graph_headroom'])} |"
        )
    if llm_diag:
        gp56 = next(
            (
                cell.get("diagnostics")
                for cell in payload["cells"]
                if cell["mix"] == "current" and cell["budget"] == 56.0
            ),
            None,
        )
        if gp56:
            gp_dec = gp56.get("pooled_energy_deciles") or {}
            llm_dec = llm_diag.get("pooled_energy_deciles") or {}
            if gp_dec and llm_dec:
                lines.extend(
                    [
                        "",
                        "GP `current`@56 candidate-level feasible-energy deciles (pooled): "
                        f"p10 {_fmt(gp_dec.get('p10'))}, p50 {_fmt(gp_dec.get('p50'))}, "
                        f"p90 {_fmt(gp_dec.get('p90'))}. LLM: "
                        f"p10 {_fmt(llm_dec.get('p10'))}, p50 {_fmt(llm_dec.get('p50'))}, "
                        f"p90 {_fmt(llm_dec.get('p90'))}.",
                    ]
                )
    lines.extend(["", "## (c) What this means for the paper", ""])
    lines.extend(_paper_verdict(payload))
    lines.extend(
        [
            "",
            "## (d) Open problems and out-of-scope temptations",
            "",
            "- If wall-clock hits the eight-hour stop, leave the `daemonize.py` pipeline "
            "running and label unfinished cells underpowered rather than negative.",
            "- Classic GPHH is executed by every new cell and not used. Restricting the "
            "harness to the GP arm would have been cheaper; it was out of scope.",
            "- Distance to graph_headroom is the wrong \"did we move?\" metric when the seed "
            "archive already contains a better feasible policy. `champion_from_initialization` "
            "is the one to quote.",
            "- GPHH rescoring, SLO sensitivity, new scenarios, LLM prompt work: not this "
            "worker. See `NOTES.md`.",
            "",
            "## (e) Exact reproduction commands",
            "",
            "```bash",
            ".venv/bin/python -m pytest tests/test_gp_operator_config_harness.py "
            "tests/test_evolution_gp_operator.py experiments/gp_tuning_20260902/test_gp_tuning.py -q",
            "",
            "# Invariant check (no --include-llm, scratch runs-root, compare hashes to runs_final)",
            "# Evidence: experiments/gp_tuning_20260902/invariant_check/INVARIANT.json",
            "",
            "# macOS has no setsid(1); daemonize.py forks, os.setsid(), then exec.",
            "DAEMONIZE_PID_FILE=experiments/gp_tuning_20260902/logs/pipeline.pid \\",
            "DAEMONIZE_LOG=experiments/gp_tuning_20260902/logs/pipeline.log \\",
            "  .venv/bin/python experiments/gp_tuning_20260902/daemonize.py bash \\",
            "  experiments/gp_tuning_20260902/run_pipeline.sh",
            "",
            "# Or one phase at a time (same session-leader wrapper):",
            "DAEMONIZE_PID_FILE=experiments/gp_tuning_20260902/logs/grid-mix56.pid \\",
            "DAEMONIZE_LOG=experiments/gp_tuning_20260902/logs/grid-mix56.log \\",
            "  .venv/bin/python experiments/gp_tuning_20260902/daemonize.py bash \\",
            "  experiments/gp_tuning_20260902/run_grid.sh mix56",
            "",
            ".venv/bin/python experiments/gp_tuning_20260902/analyze.py --rescore",
            "```",
            "",
            "LLM held-out source: `experiments/deep_search_20260825/FINALIST_RESCORE_runs_final.json` "
            f"(`payload_sha256` `{payload.get('llm_rescore_sha256', '')}`, task config "
            "`config_dev_continuum_tight_calibrated.json`).",
            "",
            "## (f) 2.1 invariant check evidence",
            "",
            "Re-ran harness preflight for root 800000801 with every original flag except "
            "`--include-llm` (ground rule: zero model API calls; `inspect_cli=False`) and "
            "without `--gp-config`. Scratch runs-root: "
            "`experiments/gp_tuning_20260902/invariant_check/`. Diff against "
            "`runs_final/.../preflight.json`:",
            "",
            "| field | match |",
            "|---|---|",
            "| `same_dsl_gp` receipt (0.45 / 0.35 / 0.20, 24, sha256 "
            "`b7b0928eab3556a2f9562955057c07f35eb0bb0afe036934ebf15059855c7041`) | yes |",
            "| `campaign_config_sha256` "
            "`ba00ff63bd344855ea023327da162feb04c944754274eabc8e785b4983c8bb13` | yes |",
            "| `task_config_sha256` | yes |",
            "| `engine_config_sha256` (glm-5.3 OpenRouter pin) | yes |",
            "| `seed_archive_payload_sha256` | yes |",
            "| `rung_microwork`, `ppo_frozen`, profile, microwork, scenario, root | yes |",
            "| `--gp-mix-` in run-directory name | absent |",
            "| `inputs/gp_operator_config.json` | absent |",
            "",
            "Expected differences, because `--include-llm` was not passed: `include_llm` "
            "false vs true, preflight status `READY_NO_LLM_ENGINE_PROBE_REQUIRED` vs "
            "`READY_ENGINE_PROBE_PASSED`, run-directory mode `non-llm` vs `all-arms`. No "
            "other payload field used as an input hash changed.",
            "",
            "On resume, re-ran "
            "`pytest tests/test_gp_operator_config_harness.py tests/test_evolution_gp_operator.py "
            "experiments/gp_tuning_20260902/test_gp_tuning.py -q`: all green. Harness code "
            "was not edited in this launch.",
            "",
            "## (g) Existing evidence restated (2.6)",
            "",
            "Not recomputed.",
            "",
            "- `experiments/why_random_wins_20260820/ANALYSIS.md`: on `dev-medium-hot`, "
            "across three pooled roots, the LLM has lower energy at **all nine candidate "
            "deciles**, lower mean (1.3750 vs 1.4027), lower median (1.3818 vs 1.3967), "
            "and lower energy in 79.0% of LLM/random candidate pairs. That is a "
            "distributional LLM advantage against a different cheap proposer, on a "
            "different scenario, already in hand.",
            "- `docs/PAPER_OUTLINE.md` §8.4: classic GPHH's search delta is **0.044030** "
            "on average (exactly 0 on two of eight roots). Its apparent quality is "
            "initialization (gen-0 mean 0.690529), not search. Same-DSL GP at 56.0 on "
            "roots 801–808 is the same shape: initialization carries the champion, search "
            "does not.",
            "",
            "Together with the archive census above, the pattern is: cheap tree operators "
            "run, emit valid programs, and stall at their initial archive. The LLM arm on "
            "the same roots does not.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="evaluate new-run GP champions on the fresh 8-seed grid",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    lock_path = HERE / "logs" / "analyze.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _main_locked(argv)


def _main_locked(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = collect_cells(rescore=args.rescore)
    payload["resume_digests"] = verify_resume_digests()
    sealed = seal_payload(payload)
    (HERE / "GP_TUNING.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n"
    )
    (HERE / "GP_TUNING.md").write_text(render_markdown(payload))
    (HERE / "RESULTS.md").write_text(render_results_md(payload))
    (HERE / "resume_digests.json").write_text(
        json.dumps(payload["resume_digests"], indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "payload_sha256": sealed["payload_sha256"],
                "missing": len(payload["missing"]),
                "resume_digests_all_match": payload["resume_digests"]["all_match"],
                "cells": [
                    {
                        "mix": cell["mix"],
                        "budget": cell["budget"],
                        "roots": cell["roots_complete"],
                    }
                    for cell in payload["cells"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
