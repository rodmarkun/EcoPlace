# ruff: noqa: E501
"""Re-score each arm's champion on a fresh seed grid it was never selected on.

Why this exists
---------------
`EVAL_NOISE.md` measured the protocol this closes: select on one scenario seed (the
fixed-work optimum), then report the selected policy **once** on a fresh multi-seed grid
rather than reusing the selection grid. Reusing it is a winner's curse — the champion
looked best partly because that grid happened to flatter it. The measured size on the
2026-08 archives was a mean shift of 0.002339 normalized energy, against an LLM-arm
median top-two gap of 0.001203. In other words the bias is the same order as the margin
that decides which candidate wins, so a headline read off the selection grid is not
safe.

The campaigns select on the harness's derived replicate seeds
(`support.evaluation_seeds`, namespace `sim/scenario/<id>/replicate/<i>`). This module
derives a **disjoint** grid under its own namespace, so no fresh seed can coincide with
a selection seed.

What it answers: does the *ranking between arms* survive re-scoring, and how much did
each arm's headline number move? Zero model calls — champions are read from sealed
archives.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/transfer_20260825"))

from deep_search_analysis import seal_payload  # noqa: E402
from transfer import champions_from_run  # noqa: E402

from analysis.tracking import derive_seed  # noqa: E402
from evolution.evaluation import (  # noqa: E402
    ScenarioGrid,
    evaluate_policy_source,
    load_task_evaluation_config,
)

RUNS_ROOT = HERE / "runs"
TASK_CONFIG = "evolution/task/config_dev_continuum_tight.json"
# Eight seeds: EVAL_NOISE.md's "fresh 8-seed x three-bundle common-random-number grid".
FRESH_SEEDS = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help=f"completed campaign runs (default: {RUNS_ROOT})",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path(TASK_CONFIG),
        help=f"evaluation task config (default: {TASK_CONFIG})",
    )
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    runs_root = args.runs_root.resolve()
    task_config = args.task_config
    if not task_config.is_absolute():
        task_config = ROOT / task_config
    return runs_root, task_config.resolve()


def output_paths(runs_root: Path) -> tuple[Path, Path]:
    suffix = "" if runs_root.resolve() == RUNS_ROOT.resolve() else f"_{runs_root.name}"
    return HERE / f"FINALIST_RESCORE{suffix}.json", HERE / f"FINALIST_RESCORE{suffix}.md"


def input_provenance(runs_root: Path, task_config: Path) -> dict[str, str]:
    return {
        "runs_root": str(runs_root),
        "task_config": str(task_config),
        "task_config_sha256": hashlib.sha256(task_config.read_bytes()).hexdigest(),
    }


def fresh_grid(root_seed: int, scenario_id: str, count: int = FRESH_SEEDS) -> tuple[int, ...]:
    """Seeds under a namespace disjoint from the harness's selection replicates."""
    return tuple(
        derive_seed(root_seed, "finalist-rescore", scenario_id, "fresh-replicate", index)
        for index in range(count)
    )


def _root_seed(run_name: str) -> int | None:
    for part in run_name.split("--"):
        if part.startswith("seed-"):
            return int(part[len("seed-") :])
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root, task_config = resolve_inputs(args)
    json_output, markdown_output = output_paths(runs_root)
    runs = sorted(path for path in runs_root.glob("*") if (path / "report.json").is_file())
    if not runs:
        print(f"no completed runs under {runs_root}", file=sys.stderr)
        return 1

    provenance = input_provenance(runs_root, task_config)
    base = load_task_evaluation_config(task_config)
    rows = []
    for run in runs:
        root_seed = _root_seed(run.name)
        if root_seed is None:
            continue
        report = json.loads((run / "report.json").read_text())
        report = report.get("payload", report)
        champions = champions_from_run(run)
        if not champions:
            continue
        scenario_id = base.scenarios[0].scenario_id
        seeds = fresh_grid(root_seed, scenario_id)
        selection_seeds = set(
            report.get("run", {}).get("evaluation_seeds", [])
        ) or set()
        if selection_seeds & set(seeds):
            raise ValueError("fresh grid collided with a selection seed")
        task = dataclasses.replace(
            base,
            scenarios=tuple(
                ScenarioGrid(item.scenario_id, item.path, seeds) for item in base.scenarios
            ),
        )
        for index, (arm, source) in enumerate(sorted(champions.items())):
            selected = report.get("arms", {}).get(arm, {}).get("best_absolute")
            print(f"[{run.name.split('--')[1]}] re-scoring {arm} on {len(seeds)} fresh seeds ...", flush=True)
            try:
                fresh = evaluate_policy_source(source, task, proposal_seq=970000 + index)
            except Exception as exc:  # noqa: BLE001 - a champion that cannot run has no fresh score
                print(f"   {arm}: could not evaluate ({type(exc).__name__})", file=sys.stderr)
                continue
            worst = min(
                (min(item.slo_achieved.values()) for item in fresh.instances), default=1.0
            )
            rows.append(
                {
                    "run": run.name,
                    "root_seed": root_seed,
                    "arm": arm,
                    "selection_grid_energy": (
                        float(selected)
                        if isinstance(selected, (int, float)) and not isinstance(selected, bool)
                        else None
                    ),
                    "fresh_grid_energy": fresh.fitness.normalized_energy,
                    "fresh_grid_feasible": fresh.fitness.feasible,
                    "fresh_grid_worst_slo": worst,
                    "fresh_seeds": list(seeds),
                }
            )

    for row in rows:
        selected, fresh = row["selection_grid_energy"], row["fresh_grid_energy"]
        row["winners_curse"] = (
            None if selected is None or fresh is None else fresh - selected
        )

    by_arm: dict[str, list] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    summary = {}
    for arm, items in sorted(by_arm.items()):
        shifts = [item["winners_curse"] for item in items if item["winners_curse"] is not None]
        fresh = [
            item["fresh_grid_energy"] for item in items if item["fresh_grid_energy"] is not None
        ]
        summary[arm] = {
            "champions": len(items),
            "feasible_on_fresh_grid": sum(1 for item in items if item["fresh_grid_feasible"]),
            "mean_winners_curse": statistics.fmean(shifts) if shifts else None,
            "mean_fresh_grid_energy": statistics.fmean(fresh) if fresh else None,
            "median_fresh_grid_energy": statistics.median(fresh) if fresh else None,
        }

    payload = {
        "schema": "evolved-continuum-finalist-rescore-v1",
        "boundary": (
            "Quarantined development-only. Zero LLM calls. Never acceptance, held-out, "
            "or confirmatory evidence."
        ),
        **provenance,
        "fresh_seeds_per_root": FRESH_SEEDS,
        "seed_namespace": "finalist-rescore (disjoint from the harness selection replicates)",
        "rows": rows,
        "summary": summary,
    }
    json_output.write_text(
        json.dumps(seal_payload(payload), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Finalists re-scored on a grid they were never selected on",
        "",
        "**Boundary:** quarantined development-only; zero LLM calls; never confirmatory.",
        "",
        "Campaigns select on one scenario seed — the fixed-work optimum `EVAL_NOISE.md` "
        "measured. Reading the headline off that same grid is a winner's curse: the "
        "measured shift on the 2026-08 archives was 0.002339 normalized energy against an "
        "LLM-arm median top-two gap of 0.001203, so the bias is the same order as the "
        f"margin that decides the winner. Each champion is re-scored here on {FRESH_SEEDS} "
        "fresh seeds drawn from a namespace disjoint from the selection replicates.",
        "",
        "| arm | champions | feasible on fresh grid | mean winner's curse | mean fresh energy | median fresh energy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in summary.items():
        def fmt(value, signed=False):
            if value is None:
                return "—"
            return f"{value:+.6f}" if signed else f"{value:.6f}"

        lines.append(
            f"| `{arm}` | {row['champions']} | {row['feasible_on_fresh_grid']} | "
            f"{fmt(row['mean_winners_curse'], True)} | {fmt(row['mean_fresh_grid_energy'])} | "
            f"{fmt(row['median_fresh_grid_energy'])} |"
        )
    lines += [
        "",
        "A positive winner's curse means the champion looked better on its selection grid "
        "than it is. The number that matters for the paper is not the size of the shift but "
        "whether the **ranking between arms survives it**.",
        "",
        "## Every champion",
        "",
        "| run | arm | selection energy | fresh energy | shift | fresh feasible | fresh worst SLO |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        def cell(value, signed=False):
            if value is None:
                return "—"
            return f"{value:+.6f}" if signed else f"{value:.6f}"

        lines.append(
            f"| `{row['run'].split('--')[1]}` | `{row['arm']}` | "
            f"{cell(row['selection_grid_energy'])} | {cell(row['fresh_grid_energy'])} | "
            f"{cell(row['winners_curse'], True)} | "
            f"{'yes' if row['fresh_grid_feasible'] else 'no'} | {row['fresh_grid_worst_slo']:.4f} |"
        )
    lines += [
        "",
        "```bash",
        ".venv/bin/python experiments/deep_search_20260825/rescore_finalists.py",
        "```",
        "",
    ]
    markdown_output.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
