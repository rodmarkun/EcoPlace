#!/usr/bin/env python3
"""ShinkaEvolve evaluator entry point for the Phase-5 task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution.evaluation import load_task_evaluation_config
from evolution.feedback import MATCHED_PAIR_PROFILE, generate_feedback, load_feedback_profiles
from evolution.ladder import evaluate_through_ladder, load_ladder_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--task_config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument(
        "--proposal_seq",
        type=int,
        required=True,
        help="canonical candidate-creation sequence assigned by the campaign runner",
    )
    parser.add_argument(
        "--feedback_profile",
        default=MATCHED_PAIR_PROFILE,
        help="predeclared feedback ablation profile (matched pair defaults off)",
    )
    return parser.parse_args()


def _compatibility_score(record: dict) -> float:
    """Diagnostic Shinka scalar; never the survivor-selection authority.

    A finite scalar cannot preserve an arbitrary continuous lexicographic tuple.
    The campaign runner therefore ranks ``fitness.sort_key`` exactly.  This value is
    retained only for Shinka's dashboards and generic prompt formatting.
    """
    fitness = record["fitness"]
    if fitness["feasible"]:
        return 1.0 / (1.0 + fitness["normalized_energy"])
    return -1.0 - sum(fitness["infeasibility_vector"]) - fitness[
        "summed_primitive_shortfall"
    ]


def main() -> int:
    args = _arguments()
    if args.proposal_seq < 0:
        raise ValueError("--proposal_seq must be a non-negative integer")
    task = load_task_evaluation_config(args.task_config)
    ladder = load_ladder_config(args.task_config)
    profiles = load_feedback_profiles(args.task_config)
    if args.feedback_profile not in profiles:
        raise ValueError(f"unknown feedback profile {args.feedback_profile!r}")
    result = evaluate_through_ladder(
        args.program_path.read_text(),
        task,
        ladder,
        proposal_seq=args.proposal_seq,
    )
    record = result.to_dict()
    profile = profiles[args.feedback_profile]
    feedback = generate_feedback(result, profile)
    record["feedback"] = {
        "profile": profile.name,
        "enabled": profile.enabled,
        "narrative": None if feedback is None else feedback.to_dict(),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    (args.results_dir / "evaluation.json").write_text(canonical + "\n")
    metrics = {
        "combined_score": (
            _compatibility_score(record["evaluation"])
            if record["evaluation"] is not None
            else -1.0e300
        ),
        "public": record,
        "selection_authority": "public.evaluation.fitness.sort_key (lower-is-better)",
    }
    if feedback is not None:
        metrics["text_feedback"] = feedback.render_text()
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n"
    )
    correct = result.promoted_to_l2 and result.evaluation is not None
    error = "" if correct else "candidate rejected before L2 fitness-of-record"
    (args.results_dir / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
