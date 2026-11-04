"""Restore A+B with DeepSeek; preserve all earlier design bytes and launch guards."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from evolution.campaign import load_campaign_plan
from evolution.evaluation import load_task_evaluation_config
from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.allbaselines_harness.initialization import read_seed_archive
from experiments.gp_tuning_20260902.recovery import read_sealed
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot
from experiments.prospective_20260917 import build as original
from experiments.validation_20260909.run_independent_calibration import store_once

ROOT = original.ROOT
HERE = Path(__file__).parent
OUT = ROOT / "prereg/CORRECTED_STUDIES_20260920_v3"
V1 = ROOT / "prereg/CORRECTED_STUDIES_20260917_v1"
PILOT = ROOT / "experiments/deepseek_pilots_bounded_20260923/artifacts"
# The 20 September pilot is retained unchanged; its first request was blocked by account
# policy and the retry halted on a reasoning-only truncation. See the bounded pilot.
PILOT_20260920 = ROOT / "experiments/deepseek_pilots_20260920/artifacts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(out=OUT):
    if out.exists():
        raise ValueError("new directory required")
    v1 = read_sealed(V1 / "FREEZE.json")
    # Check inherited bytes against the old freeze, not against today's changed live source.
    inherited = [
        "SEEDS.json",
        "POWER.json",
        "task_A.json",
        "task_B.json",
        "gp.json",
        "campaign_A.json",
        "campaign_B.json",
    ]
    inherited += [f"initialization/{s}{i:02d}.json" for s in "AB" for i in range(30)]
    for name in inherited:
        if sha(V1 / name) != v1["files_sha256"][name]:
            raise ValueError(f"inherited input differs: {name}")
    out.mkdir()
    for name in inherited:
        (out / name).parent.mkdir(exist_ok=True)
        shutil.copyfile(V1 / name, out / name)
    write(
        out / "ANALYSIS.json",
        {
            "source": "September 17 v1 PROTOCOL.md, Endpoints and inference",
            "primary_family": ["A: LLM versus GP", "B: LLM versus initialization"],
            "test": "two-sided exact sign test on non-tied root directions",
            "familywise_alpha": 0.05,
            "multiplicity": "Holm, two tests",
            "missingness": "incomplete primary family blocks confirmatory inference",
            "effect_intervals": "exact 95% conditional win; descriptive root bootstrap",
            "bootstrap_resamples": 20000,
            "bootstrap_rng": "PCG64",
            "numerical_tie": 1e-9,
            "practical_energy_threshold": 0.01,
            "practical_feasibility_threshold": 0.10,
            "statistics_source_sha256": sha(original.HERE / "statistics.py"),
            "note": "Definitions, failures and secondary endpoints follow v1.",
        },
    )
    config = json.loads((PILOT_20260920 / "engine.json").read_text())
    # Scientific per-root allocation is assigned only after pilot capacity is measured.
    write(out / "ENGINE_PILOT.json", config)
    write(
        out / "INHERITANCE.json",
        {
            "v1_freeze_sha256": sha(V1 / "FREEZE.json"),
            "files_sha256": {n: sha(out / n) for n in inherited},
            "scientific_inputs_unchanged": True,
            "note": "Campaigns retain development labels; scientific launcher is pending.",
        },
    )
    write(
        out / "DEVIATION.json",
        {
            "date": "2026-09-20",
            "supersedes": "CORRECTED_STUDIES_20260919_v2",
            "authorization": "Investigator approval, 20 September 2026.",
            "restored": "A n=30 + B n=30; 56 work units; conditional on measured capacity",
            "model": "deepseek/deepseek-v4.1-flash",
            "provider": "DeepSeek",
            "changes": [
                "Restore both v1 studies and all v1 seeds/initializations",
                "DeepSeek V4.1 Flash; high reasoning, temperature .7, top_p .8",
                "DeepSeek provider, lower-rate price ceiling, no remote seed",
                "Use current evaluation source including ledger-retention remediation",
            ],
            "nonclaims": [
                "immutable hosted weights",
                "deterministic live completions",
                "GLM replication",
                "hardware energy",
                "carbon savings",
                "operational Kubernetes equivalence",
            ],
            "pilot_selection": "resource and compatibility only; no GP pair or final-grid outcomes",
        },
    )
    shutil.copyfile(HERE / "PROTOCOL.md", out / "PROTOCOL.md")
    for study in "AB":
        load_task_evaluation_config(out / f"task_{study}.json")
        load_campaign_plan(out / f"campaign_{study}.json")
        for i in range(30):
            read_seed_archive(
                out / f"initialization/{study}{i:02d}.json",
                expected_size=12,
                initial_path=original.INITIAL,
            )
    load_openrouter_engine_config(out / "ENGINE_PILOT.json")
    print("Prepared 60 unchanged root archives and two-study analysis; not frozen or launchable.")


def amend(out=OUT):
    """Adopt the elected low-reasoning treatment and measured pilot resources before freeze."""
    if (out / "FREEZE.json").exists():
        raise ValueError("already frozen")
    summary = json.loads((PILOT / "SUMMARY.json").read_text())
    if not summary["complete"] or not summary["projection"]["fits_60_usd_with_reserve"]:
        raise ValueError("bounded pilots must complete and fit the ceiling")
    engine = json.loads((PILOT / "engine.json").read_text())
    if (engine["reasoning_effort"], engine["max_output_tokens"]) != ("low", 32000):
        raise ValueError("pilot engine differs from the elected treatment")
    # Each root has its own durable journal; this is the per-root cost ceiling. The shared
    # project ceiling is enforced across journals by the (pending) scientific launcher.
    engine["max_total_microusd"] = 1500000
    write(out / "engine.json", engine)
    load_openrouter_engine_config(out / "engine.json")
    shutil.copyfile(PILOT / "SUMMARY.json", out / "PILOT_SUMMARY.json")
    write(
        out / "TREATMENT_AMENDMENT.json",
        {
            "date": "2026-09-23",
            "researcher": "low reasoning effort",
            "authorization": "Investigator approval: write low reasoning into v3 and freeze.",
            "changes": {
                "reasoning_effort": ["high", "low"],
                "max_output_tokens": [24000, 32000],
                "empty_completion_handling": "billed empty completions settle durably and fail "
                "only that proposal; unknown outcomes still stop",
                "peak_price_windows": "no dispatch within 10 minutes of DeepSeek weekday "
                "01-04 and 06-10 UTC windows",
            },
            "basis": "Resource/compatibility pilots only; no comparative outcome, GP pair or "
            "final grid was generated or inspected.",
            "pilot": "experiments/deepseek_pilots_bounded_20260923",
            "pilot_summary_sha256": sha(PILOT / "SUMMARY.json"),
            "supersedes_in": ["DEVIATION.json changes[1]", "PROTOCOL.md proposer settings"],
            "unchanged": "Roots, seeds, initialization archives, tasks, analysis, thresholds, "
            "temperature, top_p, provider routing and price caps.",
        },
    )
    projection = summary["projection"]
    write(
        out / "MODEL_AND_RESOURCES.json",
        {
            "model_id": engine["model_id"],
            "reasoning_effort": engine["reasoning_effort"],
            "max_output_tokens": engine["max_output_tokens"],
            "authorized_total_usd": 60.0,
            "prior_accounted_and_unresolved_usd": summary["prior_accounted_and_unresolved_usd"],
            "pilot_accounted_usd": summary["pilot_accounted_usd"],
            "projected_api_usd": projection["api_usd"],
            "project_total_with_40pct_reserve_usd": projection[
                "project_total_with_40pct_reserve_usd"
            ],
            "fits_authorized_ceiling": projection["fits_60_usd_with_reserve"],
            "roots_per_study": 30,
            "work_ceiling": 56,
            "per_root_api_ceiling_usd": 1.5,
            "projected_serial_search_hours": projection["serial_hours"],
            "projection_excludes": "final evaluation, scientific gates, analysis, peak-window waits",
            "evaluation_child_rss_mib": 2048,
            "single_episode_child_timeout_seconds": 1200,
            "campaign_process_family_rss_mib": 3072,
            "concurrent_roots_max": 3,
            "concurrency_basis": "Single-root pilots peak at 576-597 MiB process family on a "
            "16 GB machine. Concurrent behaviour is measured separately before launch; the "
            "launcher must gate on available memory.",
            "per_root_interrupt": "A root's own cost ceiling may end its search before 56 work "
            "units; that is a recorded outcome, not a retry condition.",
            "run_authorized": False,
        },
    )
    print("Adopted low-reasoning treatment and measured resources; not yet frozen.")


def freeze(out=OUT):
    if (out / "FREEZE.json").exists():
        raise ValueError("already frozen")
    summary = json.loads((PILOT / "SUMMARY.json").read_text())
    if not summary["complete"]:
        raise ValueError("both pilots must complete")
    if not summary["projection"]["fits_60_usd_with_reserve"]:
        raise ValueError("measured pilot projection does not admit this sample size")
    paths = original.source_paths()
    for folder in (
        HERE,
        ROOT / "experiments/deepseek_pilots_20260920",
        ROOT / "experiments/deepseek_pilots_bounded_20260923",
        ROOT / "experiments/matched_transport_20260909",
    ):
        for p in folder.glob("*.py"):
            paths[str(p.relative_to(ROOT))] = p
    hashes = {n: sha(p) for n, p in paths.items()}
    with tarfile.open(out / "sources.tar.gz", "x:gz") as tar:
        for name, path in sorted(paths.items()):
            tar.add(path, arcname=name, recursive=False)
    if hashes != {n: sha(p) for n, p in paths.items()}:
        raise ValueError("source drift during archive")
    store_once(
        out / "FREEZE.json",
        {
            "schema": "corrected-studies-design-freeze-v3",
            "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "design-frozen-execution-blocked",
            "launch_allowed": False,
            "files_sha256": {
                str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
            },
            "source_files_sha256": hashes,
            "source_locations": {n: str(p) for n, p in paths.items()},
            "environment": evaluation_snapshot(),
            "independent_public_timestamp": False,
            "pilot_summary_sha256": sha(PILOT / "SUMMARY.json"),
        },
    )
    original.verify(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "amend", "freeze", "verify"))
    args = parser.parse_args()
    if args.action == "verify":
        original.verify(OUT)
    else:
        {"prepare": prepare, "amend": amend, "freeze": freeze}[args.action]()
