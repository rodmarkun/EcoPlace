"""Bounded A/B resource pilots, not scientific comparisons or acceptance runs.

Amends the 20 September retry: its first request spent all 24,000 output tokens on
reasoning (high effort) and returned no content. See AMENDMENT.json.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cloudpickle as pickle
import psutil

from analysis.tracking import derive_seed
from evolution.campaign import _default_assess, load_campaign_plan
from evolution.evaluation import load_task_evaluation_config
from evolution.feedback import load_feedback_profiles
from evolution.islands import load_islands_config
from evolution.ladder import load_ladder_config
from evolution.llm_operator import OpenWeightLLMOperator
from evolution.model import ApiCostLedger
from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.allbaselines_harness.initialization import (
    read_seed_archive,
    seeded_campaign_runner,
)
from experiments.deepseek_pilots_20260920.client import PilotClient, PilotLimit
from experiments.gp_tuning_20260902.recovery import digest
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot
from experiments.prospective_20260917 import build
from experiments.validation_20260909.run_independent_calibration import store_once

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "artifacts"
PIN = ROOT / "prereg/CORRECTED_STUDIES_20260917_v1"
JOURNAL = OUT / "journal.jsonl"


# DeepSeek lists doubled weekday prices for 01:00-04:00 and 06:00-10:00 UTC; the pinned
# max_price would then reject routing. Stop cleanly with a margin rather than dispatch.
PEAK_UTC_WEEKDAY = ((1, 4), (6, 10))
PEAK_MARGIN = timedelta(minutes=10)


def in_peak_window(now=None):
    now = now or datetime.now(timezone.utc)
    for probe in (now, now + PEAK_MARGIN):
        if probe.weekday() < 5 and any(a <= probe.hour < b for a, b in PEAK_UTC_WEEKDAY):
            return True
    return False


class BoundedPilotClient(PilotClient):
    def complete(self, **kwargs):
        if in_peak_window():
            raise PilotLimit("DeepSeek weekday peak-price window; stopped before dispatch")
        return super().complete(**kwargs)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def check_sources():
    frozen = json.loads((OUT / "FREEZE.json").read_text())
    for name, h in frozen["source_hashes"].items():
        if sha(Path(name)) != h:
            raise ValueError(f"source changed: {name}")
    for name, h in frozen["inputs"].items():
        if sha(OUT / name) != h:
            raise ValueError(f"input changed: {name}")
    if digest(evaluation_snapshot()) != frozen["environment_digest"]:
        raise ValueError("evaluation environment changed")


def prepare():
    OUT.mkdir()
    seeds = json.loads((PIN / "SEEDS.json").read_text())
    occupied = {r["seed"] for r in seeds["enumerated_seed_records"]} | build.historical_seeds()
    grid_records = []
    for study in "AB":
        p = OUT / study
        p.mkdir()
        root = derive_seed(20260920, "deepseek-v41-resource-pilot", study)
        task = json.loads((PIN / f"task_{study}.json").read_text())
        save(p / "task.json", task)
        for scenario in load_task_evaluation_config(p / "task.json").scenarios:
            seed = derive_seed(root, "sim", "scenario", scenario.scenario_id, "replicate", 0)
            assert seed not in occupied
            occupied.add(seed)
            grid_records.append({"study": study, "scenario": scenario.scenario_id, "seed": seed})
        plan = json.loads((PIN / f"campaign_{study}.json").read_text())
        plan["campaigns"][0] = {
            "campaign_id": f"resource-pilot-{study}",
            "root_seed": root,
            "evaluation_replicates": 1,
        }
        plan["candidate_budget"] = 18
        save(p / "campaign.json", plan)
        payload = build.archive(root, study)
        store_once(p / "initialization.json", payload)
        read_seed_archive(p / "initialization.json", expected_size=12, initial_path=build.INITIAL)
        load_campaign_plan(p / "campaign.json")
    save(
        OUT / "PLAN.json",
        {
            "purpose": (
                "Two LLM-only adaptive resource pilots; no paired operator comparison, "
                "final grids, or scientific gate claim."
            ),
            "authorization": (
                "Investigator, 23 September 2026: fix the pilot and rerun A and B. $0.50 total "
                "reservation allocation within the existing $60 project ceiling."
            ),
            "model_id": "deepseek/deepseek-v4.1-flash",
            "new_api_ceiling_microusd": 500000,
            "requests_per_study": 6,
            "per_request_reservation_microusd": 50000,
            "project_ceiling_microusd": 60000000,
            "initialization_per_study": 12,
            "candidate_ceiling": 18,
            "repair_attempts_per_candidate": 3,
            "evaluation_child_rss_mib": 2048,
            "evaluation_child_wall_seconds": 1200,
            "study_process_family_rss_mib": 3072,
            "study_wall_seconds": 7200,
            "studies_run_sequentially": True,
            "new_search_seed_grid": grid_records,
            "unchanged": [
                "study scenarios and horizons",
                "ladder",
                "islands",
                "feedback and selection",
                "temperature 0.7, top_p 0.8; provider DeepSeek; no remote seed",
            ],
            "amended": [
                "reasoning effort high -> low; max output tokens 24000 -> 32000",
                "billed empty completions settle durably and fail only that proposal",
                "stop before dispatch within 10 minutes of a DeepSeek weekday peak-price window",
            ],
            "stopping": (
                "Stop at six requests including repairs, 18 candidates, source/integrity "
                "failure, uncertain dispatch, or resource limit. No automatic uncertain "
                "retries."
            ),
            "scope": (
                "Full starting archives on reserved resource roots; use actual seeded "
                "CampaignRunner in determinism-smoke mode. The resource-only pilot does "
                "not bypass or satisfy scientific-launch guards."
            ),
            "limitations": (
                "One root per study and at most six adaptive requests do not estimate "
                "full-run cost reliably or support effectiveness claims. No GP arm is "
                "needed to time the LLM path."
            ),
        },
    )
    config = json.loads((PIN / "engine.json").read_text())
    config.update(
        model_id="deepseek/deepseek-v4.1-flash",
        prompt_microusd_per_million=150000,
        completion_microusd_per_million=600000,
        max_call_reservation_microusd=50000,
        max_total_microusd=500000,
        reasoning_effort="low",
        max_output_tokens=32000,
    )
    config["generation"]["provider"].update(
        only=["deepseek"], max_price={"prompt": 0.15, "completion": 0.60}
    )
    save(OUT / "engine.json", config)
    paths = build.source_paths()
    for folder in (HERE, ROOT / "experiments/adaptive_budget_pilots_20260917"):
        for p in folder.glob("*.py"):
            paths[str(p.relative_to(ROOT))] = p
    for extra in (
        "experiments/deepseek_pilots_20260920/client.py",
        "experiments/glm_budget_pilot_20260917/run.py",
        "tests/test_matched_durable_transport.py",
        "tests/test_evolution_campaign.py",
    ):
        paths[extra] = ROOT / extra
    for name in ("endpoints.json", "model.json"):
        paths[str((HERE / name).relative_to(ROOT))] = HERE / name
    hashes = {str(p): sha(p) for p in paths.values()}
    with tarfile.open(OUT / "sources.tar.gz", "x:gz") as tar:
        for name, p in sorted(paths.items()):
            tar.add(p, arcname=name, recursive=False)
    save(
        OUT / "FREEZE.json",
        {
            "source_hashes": hashes,
            "inputs": {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file()},
            "environment_digest": digest(evaluation_snapshot()),
            "prior_global_journal_sha256": sha(
                ROOT / "experiments/api_budget_20260917/journal.jsonl"
            ),
        },
    )
    check_sources()
    print(
        "Prepared two verified 6-request pilots; total reservation ceiling $0.50.",
        flush=True,
    )


def eval_worker(request_path):
    import evolution.ladder as ladder_module

    measurements = []
    for name in ("_run_l0_adapter", "_run_l1_adapter", "evaluate_policy_adapter"):
        original = getattr(ladder_module, name)

        def measured(*args, _name=name, _original=original, **kwargs):
            start = time.monotonic()
            try:
                return _original(*args, **kwargs)
            finally:
                measurements.append({"component": _name, "seconds": time.monotonic() - start})

        setattr(ladder_module, name, measured)
    with request_path.open("rb") as f:
        request = pickle.load(f)
    start = time.monotonic()
    result = ladder_module.evaluate_through_ladder(
        request["source"],
        request["task"],
        request["ladder"],
        proposal_seq=request["seq"],
        retain_episode_ledgers=request.get("retain_episode_ledgers", True),
        **request["evaluation_options"],
    )
    with request_path.with_suffix(".result").open("xb") as f:
        pickle.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    save(
        request_path.with_suffix(".timing.json"),
        {"seconds": time.monotonic() - start, "components": measurements},
    )


def guarded_evaluate(study_dir, source, task, ladder, seq, **evaluation_options):
    directory = study_dir / "evaluations"
    directory.mkdir(exist_ok=True)
    request = {
        "source": source,
        "task": task,
        "ladder": ladder,
        "seq": seq,
        "evaluation_options": evaluation_options,
        # The guarded child returns the result by pickle; retaining per-episode ledgers
        # made one study-B result a 265 MB pickle and pushed the child past the 2 GiB
        # ceiling. See artifacts/RETENTION_REMEASUREMENT.json for the equivalence proof.
        "retain_episode_ledgers": False,
    }
    blob = pickle.dumps(request)
    key = hashlib.sha256(blob).hexdigest()
    path = directory / f"{seq:03d}-{key}.pickle"
    if path.exists():
        raise RuntimeError(
            "incomplete prior evaluation or duplicate request; explicit recovery required"
        )
    path.write_bytes(blob)
    command = [
        sys.executable,
        "-m",
        "experiments.deepseek_pilots_bounded_20260923.run",
        "eval-worker",
        str(path),
    ]
    start = time.monotonic()
    peak = 0
    with path.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    family = [
                        psutil.Process(process.pid),
                        *psutil.Process(process.pid).children(recursive=True),
                    ]
                    rss = sum(p.memory_info().rss for p in family)
                except psutil.NoSuchProcess:
                    rss = 0
                peak = max(peak, rss)
                if rss > 2048 * 2**20 or time.monotonic() - start > 1200:
                    raise RuntimeError("evaluation resource ceiling reached")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(
                    f"evaluation worker exit {process.returncode}; see {path.with_suffix('.log')}"
                )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    save(
        path.with_suffix(".resource.json"),
        {
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_bytes": peak,
            "request_sha256": key,
            "result_sha256": sha(path.with_suffix(".result")),
        },
    )
    with path.with_suffix(".result").open("rb") as f:
        result = pickle.load(f)
    print(
        json.dumps(
            {
                "study": study_dir.name,
                "evaluated_candidate": seq,
                "promoted_l2": result.promoted_to_l2,
                "seconds": round(time.monotonic() - start, 2),
                "rss_mib": round(peak / 2**20, 1),
            }
        ),
        flush=True,
    )
    return result


class GuardedEvaluator:
    supports_behavioral_novelty = True

    def __init__(self, study_dir):
        self.study_dir = study_dir

    def __call__(self, source, task, ladder, seq, **evaluation_options):
        return guarded_evaluate(self.study_dir, source, task, ladder, seq, **evaluation_options)


def worker(study):
    check_sources()
    directory = OUT / study
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "START.json").exists():
            raise RuntimeError("pilot already started; no automatic rerun")
        save(
            directory / "START.json",
            {"monotonic": time.monotonic(), "source_freeze_sha256": sha(OUT / "FREEZE.json")},
        )
        plan = load_campaign_plan(directory / "campaign.json")
        cfg = load_openrouter_engine_config(OUT / "engine.json")
        client = BoundedPilotClient(
            cfg,
            journal=JOURNAL,
            calls_dir=directory / "calls",
        )
        operator = OpenWeightLLMOperator(
            cfg,
            client,
            ApiCostLedger(150000),
            proposal_modes=plan.llm_proposal_modes,
            profile=plan.profile,
            proposals_per_call=1,
        )
        seed_archive = read_seed_archive(
            directory / "initialization.json", expected_size=12, initial_path=build.INITIAL
        )
        base = seeded_campaign_runner(seed_archive)

        class ResourceRunner(base):
            def _allocate(self, seq):
                if seq >= 12 and len(list((directory / "calls").glob("*.json"))) >= 6:
                    raise PilotLimit("six-model-request resource-pilot ceiling reached")
                return super()._allocate(seq)

        def assess(source, task, islands):
            start = time.monotonic()
            result = _default_assess(source, task, islands)
            with (directory / "assessments.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "seconds": time.monotonic() - start,
                        }
                    )
                    + "\n"
                )
            return result

        runner = ResourceRunner(
            definition=plan.campaigns[0],
            plan=plan,
            arm="llm",
            operator=operator,
            initial_source=build.INITIAL.read_text(),
            task=load_task_evaluation_config(directory / "task.json"),
            ladder=load_ladder_config(directory / "task.json"),
            islands=load_islands_config(directory / "task.json"),
            feedback_profile=load_feedback_profiles(directory / "task.json")[plan.feedback_profile],
            archive_dir=directory / "campaign",
            provenance={
                "purpose": "resource-pilot-only",
                "source_freeze_sha256": sha(OUT / "FREEZE.json"),
            },
            run_purpose="determinism-smoke",
            evaluate_fn=GuardedEvaluator(directory),
            assess_fn=assess,
        )
        started = time.monotonic()
        status = "candidate-target-complete"
        try:
            runner.run()
        except PilotLimit as exc:
            status = str(exc)
        save(
            directory / "COMPLETE.json",
            {
                "status": status,
                "seconds": time.monotonic() - started,
                "model_requests": len(list((directory / "calls").glob("*.json"))),
                "completed_candidates": len(runner._completed()),
                "evaluator_microwork": runner._spent_microwork(),
                "resource_pilot_only": True,
            },
        )
        check_sources()


def run():
    check_sources()
    for study in "AB":
        directory = OUT / study
        if (directory / "COMPLETE.json").exists():
            raise RuntimeError("existing pilot result; use separate explicit continuation")
        start = time.monotonic()
        peak = 0
        with (directory / "worker.log").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.deepseek_pilots_bounded_20260923.run",
                    "worker",
                    study,
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        family = [parent, *parent.children(recursive=True)]
                        rss = sum(p.memory_info().rss for p in family)
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    if rss > 3072 * 2**20 or time.monotonic() - start > 7200:
                        raise RuntimeError(f"{study} exceeded whole-study resource ceiling")
                    time.sleep(0.5)
                if process.returncode:
                    raise RuntimeError(
                        f"{study} stopped with exit {process.returncode}; see worker.log"
                    )
            finally:
                if process.poll() is None:
                    try:
                        children = psutil.Process(process.pid).children(recursive=True)
                        for child in children:
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                    process.kill()
                process.wait()
                save(
                    directory / "PROCESS_USAGE.json",
                    {
                        "seconds": time.monotonic() - start,
                        "peak_family_rss_bytes": peak,
                        "exit_code": process.returncode,
                    },
                )
        print(
            json.dumps(
                {
                    "study": study,
                    "status": "finished",
                    "seconds": time.monotonic() - start,
                    "peak_family_rss_mib": peak / 2**20,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        prepare()
    elif action == "run":
        run()
    elif action == "worker":
        worker(sys.argv[2])
    elif action == "eval-worker":
        eval_worker(Path(sys.argv[2]))
