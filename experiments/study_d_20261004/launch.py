"""Registered launcher for Study D (single-seed counterpart of Study C on C's roots).

A copy of experiments/study_c_20261003/launch.py. Differences: registration directory, study
"D", project ceiling $75 (investigator decision, 4 October 2026), prior spend that also
includes every Study C journal, and smoke seeds. Evaluation code paths are unchanged.

Executes the frozen v3 design (prereg/CORRECTED_STUDIES_20260920_v3) under the P2 freeze in
this directory. Scheduling, budget and transport amendments are listed in P2_FREEZE.json.

Per root pair, in fixed index order A00, B00, A01, B01, ... with three pair slots:
  first arm -> initialization barrier; second arm -> barrier; certify pair.json;
  first arm search; second arm search (registered `first_arm` order, one child at a time);
  seal SELECTION.json from completed full-evaluation (L2) records only;
  then final scoring of every label on the root's untouched final grid.

Monitoring (`status`) reports progress, terminal states and spend, never comparative outcomes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3 = ROOT / "prereg/STUDY_D_SINGLESEED_20261004"  # registration directory (name kept from A+B)
CEILING_MICROUSD = 75_000_000
SLOTS = 6
# Per-root API ceiling as in Study C. Prior spend ($43.85 after C) plus 30 x $0.95 = $72.35 fits
# the $75 project ceiling, so the global ceiling cannot bind and no root's budget depends on
# request order. Study A's most expensive (single-seed) LLM root cost $0.77.
PER_ROOT_CEILING_MICROUSD = 950_000
# P2 amendment 04: memory back-pressure. New stages/finals start only with this much free
# memory; every evaluation child additionally waits for EVALUATION_START_MIB.
MIN_AVAILABLE_MIB = 3072
EVALUATION_START_MIB = 2048
STAGE_FAMILY_RSS_MIB = 3072
FINAL_CHILD_RSS_MIB = 2048
FINAL_SECONDS_PER_EPISODE_GROUP = 1200
REFERENCE_FILES = (
    "latency_greedy.ps",
    "energy_greedy.ps",
    "first_fit.ps",
    "least_allocated.ps",
    "most_allocated.ps",
    "kube_least_allocated.ps",
    "kube_most_allocated.ps",
    "network_overhead.ps",
    "graph_headroom.ps",
)
# Journals of every earlier paid call; counted in full (unresolved reservations included).
PRIOR_JOURNALS = (
    "experiments/deepseek_pilots_20260920/artifacts/journal.jsonl",
    "experiments/deepseek_access_20260920/journal.jsonl",
    "experiments/deepseek_pilots_retry_20260920/artifacts/journal.jsonl",
    "experiments/deepseek_pilots_bounded_20260923/artifacts/journal.jsonl",
    "experiments/deepseek_concurrency_20260923/artifacts/A0/journal.jsonl",
    "experiments/deepseek_concurrency_20260923/artifacts/B0/journal.jsonl",
    "experiments/deepseek_concurrency_20260923/artifacts/B1/journal.jsonl",
    "experiments/g_ablate_commercial_20260924/artifacts/llm/journal.jsonl",
)
EXIT = {"complete": 0, "barrier": 10, "unusable": 20, "budget": 30, "unresolved": 40,
        "resource": 50}
# Registered single-episode child timeout (v1/v3 MODEL_AND_RESOURCES); the evaluation guard
# scales it by the episodes one ladder evaluation can run (P2 amendment 01).
EPISODE_TIMEOUT_S = 1200
EVALUATION_RSS_MIB = 2048


@dataclass(frozen=True)
class Layout:
    out: Path
    studies: tuple[str, ...] = ("D",)
    roots: int = 30
    smoke: bool = False

    @property
    def runs(self):
        return self.out / "runs"


SCIENTIFIC = Layout(out=HERE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def save_once(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_text() != text:
            raise ValueError(f"different sealed record already exists: {path}") from None


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------- budget and transport


class GlobalBudgetExhausted(SystemExit):
    """Stops a root outside operator retry handlers; recorded as a cost stop."""


class RootBudgetExhausted(GlobalBudgetExhausted):
    """The root's own registered cost ceiling ends its search (a recorded outcome)."""


class EvaluationResourceStop(SystemExit):
    """An evaluation exceeded its memory or registered time guard; stops only this arm."""


def ladder_episodes(task, ladder):
    """Upper bound on episodes in one ladder evaluation: L0 replay, L1 and full L2 grid."""
    l1 = len(ladder.l1.scenario_ids) * ladder.l1.seeds_per_scenario * len(ladder.l1.bundle_ids)
    l2 = sum(len(s.seeds) for s in task.scenarios) * len(task.bundle_ids)
    return 1 + l1 + l2


def journal_exposure(path):
    from experiments.matched_transport_20260909.client import DurableOpenRouterClient

    with Path(path).open("rb") as stream:
        events = DurableOpenRouterClient._read(stream)
    total, pending = 0, None
    for event in events[1:]:
        row = event["payload"]
        if row["type"] == "reserved":
            pending = row
        elif row["type"] == "completed":
            total += row["completion"]["cost_microusd"]
            pending = None
        elif row["type"] in {"settled_empty", "settled_http_error", "settled_unknown"}:
            total += row["settlement"]["cost_microusd"]
            pending = None
        else:
            raise ValueError("unknown journal event")
    return total + (pending["reservation_microusd"] if pending else 0)


def prior_exposure_microusd():
    from experiments.deepseek_pilots_20260920.client import LEGACY, legacy_exposure

    with LEGACY.open("rb") as stream:
        total = legacy_exposure(stream)
    smoke = sorted(HERE.glob("smoke*/runs/*/llm/journal.jsonl"))
    ab = ROOT / "experiments/corrected_ab_20260925"
    smoke += sorted(ab.glob("runs/*/llm/journal.jsonl"))
    smoke += sorted(ab.glob("smoke*/runs/*/llm/journal.jsonl"))
    study_c = ROOT / "experiments/study_c_20261003"
    smoke += sorted(study_c.glob("runs/*/llm/journal.jsonl"))
    smoke += sorted(study_c.glob("smoke*/runs/*/llm/journal.jsonl"))
    return total + sum(journal_exposure(p) for p in [*(ROOT / n for n in PRIOR_JOURNALS), *smoke])


class GlobalBudget:
    """Append-only reservation ledger enforcing the project ceiling across all journals."""

    def __init__(self, layout, prior_microusd):
        self.path = layout.out / "BUDGET.jsonl"
        self.prior = prior_microusd

    def _exposure(self, rows):
        reserved, settled = {}, {}
        for row in rows:
            (reserved if row["t"] == "reserve" else settled)[row["key"]] = row["amount"]
        return self.prior + sum(settled.get(k, v) for k, v in reserved.items())

    def _rows(self, stream):
        stream.seek(0)
        return [json.loads(line) for line in stream if line.strip()]

    def reserve(self, label, key, amount):
        with self.path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            rows = self._rows(stream)
            if any(r["t"] == "reserve" and r["key"] == key for r in rows):
                return
            if self._exposure(rows) + amount > CEILING_MICROUSD:
                raise GlobalBudgetExhausted("project API ceiling reached; root stopped")
            stream.write(json.dumps({"t": "reserve", "label": label, "key": key,
                                     "amount": amount, "at": now()}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def settle(self, label, key, amount):
        append_jsonl(self.path, {"t": "settle", "label": label, "key": key, "amount": amount,
                                 "at": now()})

    def exposure(self):
        if not self.path.exists():
            return self.prior
        with self.path.open() as stream:
            return self._exposure(self._rows(stream))


def seconds_until_off_peak(moment=None):
    from experiments.deepseek_pilots_bounded_20260923.run import in_peak_window

    moment = moment or datetime.now(UTC)
    waited = 0
    while in_peak_window(moment) and waited < 24 * 3600:
        moment = moment.replace(second=0, microsecond=0) + _minutes(5)
        waited += 300
    return waited


def _minutes(n):
    from datetime import timedelta

    return timedelta(minutes=n)


def wait_for_connectivity(directory, host="openrouter.ai", port=443):
    """Block (logged) until the provider accepts TCP connections (P2 amendment 02)."""
    import socket

    waited = 0
    while True:
        try:
            socket.create_connection((host, port), timeout=5).close()
            if waited:
                append_jsonl(directory / "PAUSES.jsonl", {"at": now(), "offline_wait_s": waited})
            return waited
        except OSError:
            time.sleep(60)
            waited += 60


def scientific_client(layout, label, directory, config):
    from experiments.deepseek_pilots_20260920.client import PilotClient
    from experiments.glm_budget_pilot_20260917.run import bound
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.matched_transport_20260909.client import (
        DurableOpenRouterClient,
        SettledEmptyCompletion,
        SettledHTTPError,
        SettledUnknownDispatch,
        UnresolvedDispatch,
        reconcile_unknown,
    )

    budget = GlobalBudget(layout, prior_exposure_microusd())

    class ScientificClient(PilotClient):
        consecutive_http_errors = 0

        def complete(self, *, model, prompt, generation, seed):
            bound(prompt, self.config)
            request = {"config_sha256": self.config_sha, "model": model, "prompt": prompt,
                       "generation": generation, "seed": seed}
            key = digest(request)
            with self._locked() as stream:
                completed = self._state(self._read(stream))
            known = key in completed or key in self._settled_empty or key in self._settled_http
            reservation = self.config.max_call_reservation_microusd
            ceiling = self.config.max_total_microusd
            if not known and self.ledger.spent_microusd + reservation > ceiling:
                raise RootBudgetExhausted("root API ceiling reached; root stopped")
            if not known:
                wait_for_connectivity(directory)
                wait = seconds_until_off_peak()
                if wait:
                    append_jsonl(directory / "PAUSES.jsonl", {"at": now(), "peak_wait_s": wait})
                    time.sleep(wait)
                budget.reserve(label, key, reservation)
            call_key = digest({"model": model, "prompt": prompt, "generation": generation,
                               "seed": seed})
            self.receipt_path = directory / f"{call_key}-provider.json"
            self.wire_path = directory / f"{call_key}-wire.json"
            self.error_path = directory / f"{call_key}-http-error.json"
            self.started = time.monotonic()
            try:
                result = DurableOpenRouterClient.complete(
                    self, model=model, prompt=prompt, generation=generation, seed=seed
                )
            except SettledEmptyCompletion as exc:
                if not known:
                    budget.settle(label, key, exc.cost_microusd)
                raise
            except (SettledHTTPError, SettledUnknownDispatch) as exc:
                if not known:
                    budget.settle(label, key, exc.cost_microusd)
                    self.consecutive_http_errors += 1
                    pause = min(60 * 2 ** (self.consecutive_http_errors - 1), 1800)
                    append_jsonl(directory / "PAUSES.jsonl",
                                 {"at": now(), "failure": type(exc).__name__,
                                  "http_status": getattr(exc, "status", None),
                                  "backoff_s": pause})
                    time.sleep(pause)
                raise
            if not known:
                budget.settle(label, key, result.cost_microusd)
                self.consecutive_http_errors = 0
            return result

    journal = directory / "journal.jsonl"
    try:
        return ScientificClient(config, journal=journal, calls_dir=directory / "calls")
    except UnresolvedDispatch:
        # A process died with a request in flight (reboot, kill, outage). Its outcome is
        # unknown: charge the full reservation by appending a settlement and continue
        # (P2 amendment 02). Earlier journal events are never edited.
        record = reconcile_unknown(journal, config, "pending reservation after interrupted process")
        if record is None:
            raise
        budget.settle(label, record["request_sha256"], record["cost_microusd"])
        append_jsonl(directory / "RECONCILIATIONS.jsonl", {"at": now(), **record})
        return ScientificClient(config, journal=journal, calls_dir=directory / "calls")


# ---------------------------------------------------------------- design inputs


def seeds_row(study, index):
    rows = json.loads((V3 / "SEEDS.json").read_text())["studies"][study]
    row = rows[index]
    assert row["campaign_id"] == f"{study}{index:02d}"
    return row


def smoke_row(study, index):
    from analysis.tracking import derive_seed
    from evolution.evaluation import load_task_evaluation_config

    root = derive_seed(20261004, "study-d-launcher-smoke", study, index)
    grids = {}
    for scenario in load_task_evaluation_config(V3 / f"task_{study}.json").scenarios:
        grids[scenario.scenario_id] = {
            "search": [derive_seed(root, "sim", "scenario", scenario.scenario_id, "replicate", j)
                       for j in range(1)],
            "final": [derive_seed(root, "smoke-final", scenario.scenario_id, 0)],
        }
    return {"campaign_id": f"S{study}{index:02d}", "root_seed": root, "grids": grids,
            "first_arm": "llm" if index % 2 == 0 else "gp", "evaluation_replicates": 1}


def root_row(layout, study, index):
    return smoke_row(study, index) if layout.smoke else seeds_row(study, index)


def root_dir(layout, study, index):
    return layout.runs / root_row(layout, study, index)["campaign_id"]


def prepare_smoke_inputs(layout, study, index):
    """Smoke roots use reserved seeds, a small candidate target and derived archives."""
    from evolution.campaign import load_campaign_plan
    from experiments.allbaselines_harness.initialization import read_seed_archive
    from experiments.prospective_20260917 import build
    from experiments.validation_20260909.run_independent_calibration import store_once

    row = smoke_row(study, index)
    directory = layout.runs / row["campaign_id"]
    directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads((V3 / f"campaign_{study}.json").read_text())
    plan["campaigns"][0] = {"campaign_id": row["campaign_id"], "root_seed": row["root_seed"],
                            "evaluation_replicates": 1}
    plan["candidate_budget"] = 14
    if not (directory / "campaign.json").exists():
        save_once(directory / "campaign.json", plan)
    load_campaign_plan(directory / "campaign.json")
    if not (directory / "initialization.json").exists():
        store_once(directory / "initialization.json", build.archive(row["root_seed"], study))
    read_seed_archive(directory / "initialization.json", expected_size=12,
                      initial_path=build.INITIAL)


def campaign_inputs(layout, study, index):
    from evolution.campaign import load_campaign_plan

    row = root_row(layout, study, index)
    if layout.smoke:
        directory = layout.runs / row["campaign_id"]
        plan = load_campaign_plan(directory / "campaign.json")
        return plan, plan.campaigns[0], directory / "initialization.json", "determinism-smoke"
    plan = load_campaign_plan(V3 / f"campaign_{study}.json")
    definition = next(c for c in plan.campaigns if c.campaign_id == row["campaign_id"])
    assert definition.root_seed == row["root_seed"]
    return plan, definition, V3 / "initialization" / f"{row['campaign_id']}.json", (
        "scientific-full-work"
    )


# ---------------------------------------------------------------- stages (child processes)


def stage(layout, study, index, arm):
    from experiments.allbaselines_harness.initialization import read_seed_archive
    from experiments.matched_control_20260909.runner import (
        InitializationPending,
        matched_campaign_runner,
    )
    from experiments.matched_transport_20260909.client import UnresolvedDispatch
    from experiments.prospective_20260917 import build

    verify_p2(layout)
    try:
        _stage(layout, study, index, arm, InitializationPending, matched_campaign_runner,
               build, read_seed_archive)
    except InitializationPending:
        raise SystemExit(EXIT["barrier"]) from None
    except GlobalBudgetExhausted:
        raise SystemExit(EXIT["budget"]) from None
    except UnresolvedDispatch:
        raise SystemExit(EXIT["unresolved"]) from None
    except EvaluationResourceStop:
        raise SystemExit(EXIT["resource"]) from None
    except RuntimeError as exc:
        text = str(exc)
        if ("proposal-engine ceiling" in text or "candidate safety ceiling" in text
                or "archive is unusable" in text):
            raise SystemExit(EXIT["unusable"]) from None
        raise
    raise SystemExit(EXIT["complete"])


def _stage(layout, study, index, arm, InitializationPending, matched_campaign_runner,
           build, read_seed_archive):
    from evolution.campaign import _default_assess
    from evolution.evaluation import load_task_evaluation_config
    from evolution.feedback import load_feedback_profiles
    from evolution.gp_operator import PlacementScriptGPOperator, load_gp_operator_config
    from evolution.islands import load_islands_config
    from evolution.ladder import load_ladder_config
    from evolution.llm_operator import OpenWeightLLMOperator
    from evolution.model import ApiCostLedger
    from evolution.openrouter_engine import load_openrouter_engine_config

    plan, definition, initialization, purpose = campaign_inputs(layout, study, index)
    directory = root_dir(layout, study, index) / arm
    directory.mkdir(parents=True, exist_ok=True)
    if arm == "gp":
        config, _label = load_gp_operator_config(V3 / "gp.json")
        operator = PlacementScriptGPOperator(config)
    else:
        engine = load_openrouter_engine_config(V3 / "engine.json")
        engine = replace(engine, max_total_microusd=PER_ROOT_CEILING_MICROUSD)
        client = scientific_client(layout, f"{study}{index:02d}", directory, engine)
        operator = OpenWeightLLMOperator(
            engine, client, ApiCostLedger(engine.max_total_microusd),
            proposal_modes=plan.llm_proposal_modes, profile=plan.profile,
            ablation_prompt_hint_probability=plan.ablation_prompt_hint_probability,
            proposals_per_call=plan.proposals_per_call,
        )
    task_path = V3 / f"task_{study}.json"
    seed_archive = read_seed_archive(initialization, expected_size=12,
                                     initial_path=build.INITIAL)
    runner = matched_campaign_runner(seed_archive, root_dir(layout, study, index) / "pair.json")(
        definition=definition, plan=plan, arm=arm, operator=operator,
        initial_source=build.INITIAL.read_text(),
        task=load_task_evaluation_config(task_path),
        ladder=load_ladder_config(task_path),
        islands=load_islands_config(task_path),
        feedback_profile=load_feedback_profiles(task_path)[plan.feedback_profile],
        archive_dir=directory / "campaign",
        provenance={"purpose": "corrected-ab-v3", "p2_freeze_sha256": sha(p2_path(layout)),
                    "arm": arm},
        run_purpose=purpose,
        evaluate_fn=RecoveringEvaluator(directory),
        assess_fn=_default_assess,
    )
    runner.run()


class RecoveringEvaluator:
    """Guarded subprocess evaluator that recovers an evaluation interrupted by a crash.

    The pilot evaluator refuses to overwrite an existing request pickle. After a crash, the
    runner re-requests the same candidate; its incomplete files are moved aside (retained) and
    the deterministic evaluation is repeated.
    """

    supports_behavioral_novelty = True

    def __init__(self, directory):
        self.directory = directory

    def __call__(self, source, task, ladder, seq, **evaluation_options):
        evaluations = self.directory / "evaluations"
        evaluations.mkdir(exist_ok=True)
        stale = sorted(evaluations.glob(f"{seq:03d}-*"))
        if stale:
            aside = evaluations / f"interrupted-{seq:03d}-{time.time_ns()}"
            aside.mkdir()
            for path in stale:
                path.rename(aside / path.name)
            append_jsonl(self.directory / "RECOVERIES.jsonl",
                         {"at": now(), "proposal_seq": seq, "moved": [p.name for p in stale],
                          "to": aside.name})
        return guarded_evaluate(self.directory, source, task, ladder, seq,
                                wall_s=EPISODE_TIMEOUT_S * ladder_episodes(task, ladder),
                                **evaluation_options)


def wait_for_memory(minimum_mib, log_path, what):
    """Block until the host has minimum_mib available; logged. Results are unaffected."""
    waited = 0
    while psutil.virtual_memory().available < minimum_mib * 2**20:
        time.sleep(10)
        waited += 10
    if waited:
        append_jsonl(log_path, {"at": now(), "memory_wait_s": waited, "for": what,
                                "minimum_mib": minimum_mib})
    return waited


def guarded_evaluate(study_dir, source, task, ladder, seq, *, wall_s, **evaluation_options):
    """The pilot's guarded subprocess evaluation with the registered time guard.

    Request, child worker and result files are byte-compatible with the pilot evaluator; only
    the wall-time limit differs, and a guard hit stops the arm instead of the whole launcher.
    """
    import cloudpickle as pickle

    evaluations = study_dir / "evaluations"
    evaluations.mkdir(exist_ok=True)
    request = {
        "source": source,
        "task": task,
        "ladder": ladder,
        "seq": seq,
        "evaluation_options": evaluation_options,
        "retain_episode_ledgers": False,
    }
    blob = pickle.dumps(request)
    key = hashlib.sha256(blob).hexdigest()
    path = evaluations / f"{seq:03d}-{key}.pickle"
    if path.exists():
        raise RuntimeError("incomplete prior evaluation or duplicate request")
    path.write_bytes(blob)
    command = [sys.executable, "-m", "experiments.deepseek_pilots_bounded_20260923.run",
               "eval-worker", str(path)]
    wait_for_memory(EVALUATION_START_MIB, study_dir / "MEMORY_WAITS.jsonl", f"evaluation {seq}")
    start = time.monotonic()
    peak = 0
    with path.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(True)])
                except psutil.NoSuchProcess:
                    rss = 0
                peak = max(peak, rss)
                elapsed = time.monotonic() - start
                if rss > EVALUATION_RSS_MIB * 2**20 or elapsed > wall_s:
                    append_jsonl(study_dir / "RESOURCE_STOPS.jsonl",
                                 {"at": now(), "proposal_seq": seq, "rss_mib": rss / 2**20,
                                  "elapsed_s": elapsed, "wall_limit_s": wall_s})
                    raise EvaluationResourceStop("evaluation resource guard reached")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"evaluation worker exit {process.returncode}; see "
                                   f"{path.with_suffix('.log')}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    save_once(path.with_suffix(".resource.json"), {
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_bytes": peak,
        "request_sha256": key,
        "result_sha256": sha(path.with_suffix(".result")),
        "wall_limit_s": wall_s,
    })
    with path.with_suffix(".result").open("rb") as stream:
        return pickle.load(stream)


def run_child(layout, args, log, rss_mib, wall_s=None):
    """Run a child with a process-family RSS guard. Returns its exit code."""
    wait_for_memory(MIN_AVAILABLE_MIB, layout.out / "MEMORY_WAITS.jsonl", " ".join(args))
    started = time.monotonic()
    with Path(log).open("a") as stream:
        stream.write(f"--- {now()} {' '.join(args)}\n")
        stream.flush()
        process = subprocess.Popen(
            [sys.executable, "-m", "experiments.study_d_20261004.launch", *args],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "STUDY_D_LAYOUT": str(layout.out),
                 "STUDY_D_SMOKE": "1" if layout.smoke else "0"},
        )
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(True)])
                except psutil.NoSuchProcess:
                    rss = 0
                if rss > rss_mib * 2**20:
                    raise RuntimeError(f"child exceeded {rss_mib} MiB: {args}")
                if wall_s is not None and time.monotonic() - started > wall_s:
                    raise RuntimeError(f"child exceeded {wall_s} s: {args}")
                time.sleep(1)
        finally:
            if process.poll() is None:
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
                process.kill()
            process.wait()
    return process.returncode


# ---------------------------------------------------------------- selection and finals


def completed_rows(archive):
    rows = []
    for line in (archive / "events.jsonl").open():
        event = json.loads(line)
        if event["type"] == "candidate_completed":
            rows.append(event["payload"])
    return rows


def terminal(archive):
    last = None
    for line in (archive / "events.jsonl").open():
        event = json.loads(line)
        if event["type"] == "campaign_terminal":
            last = event["payload"]
    return last


def full_records(rows):
    """Completed full-evaluation (L2) records; audits never count as selections."""
    out = []
    for row in rows:
        evaluation = row.get("evaluation") or {}
        if evaluation.get("promoted_to_l2") and evaluation.get("evaluation"):
            out.append(row)
    return out


def best(rows):
    if not rows:
        return None
    row = min(rows, key=lambda r: tuple(r["evaluation"]["evaluation"]["fitness"]["sort_key"]))
    return {
        "candidate_id": row["candidate_id"],
        "proposal_seq": row["proposal_seq"],
        "source": row["proposal"]["source"],
        "source_sha256": hashlib.sha256(row["proposal"]["source"].encode()).hexdigest(),
        "search_fitness": row["evaluation"]["evaluation"]["fitness"],
    }


def seal_selection(layout, study, index, exits):
    directory = root_dir(layout, study, index)
    arms = {}
    init_best = {}
    for arm in ("gp", "llm"):
        archive = directory / arm / "campaign"
        rows = completed_rows(archive)
        full = full_records(rows)
        arms[arm] = {
            "stage_exit": exits.get(arm),
            "terminal": terminal(archive),
            "completed_candidates": len(rows),
            "full_records": len(full),
            "champion": best(full),
        }
        init_best[arm] = best([r for r in full if r["proposal_seq"] < 12])
    a, b = init_best["gp"], init_best["llm"]
    if (a is None) != (b is None) or (a and a["source_sha256"] != b["source_sha256"]):
        raise ValueError("initialization-best identities differ after parity verification")
    record = {"study": study, "index": index, "sealed_at": now(), "arms": arms,
              "initialization_best": a,
              "rule": "CandidateFitness.sort_key over completed L2 records; proposal "
                      "sequence tie-break; no final-grid information"}
    save_once(directory / "SELECTION.json", record)
    return record


def reference_sources():
    from experiments.allbaselines_harness.initialization import _evolve_shell, _reference_body
    from experiments.prospective_20260917 import build

    prefix, suffix = _evolve_shell(build.INITIAL.read_text())
    out = {}
    for name in REFERENCE_FILES:
        body = _reference_body((ROOT / "runtime/policies" / name).read_text())
        out[f"reference:{name.removesuffix('.ps')}"] = prefix + body + suffix
    return out


def final_labels(layout, study, index):
    selection = json.loads((root_dir(layout, study, index) / "SELECTION.json").read_text())
    labels = {}
    for arm in ("llm", "gp"):
        champion = selection["arms"][arm]["champion"]
        if champion is not None:
            labels[f"{arm}_champion"] = champion["source"]
    if selection["initialization_best"] is not None:
        labels["initialization_best"] = selection["initialization_best"]["source"]
    labels.update(reference_sources())
    if study == "B":
        from experiments.allbaselines_harness.initialization import (
            _evolve_shell,
            _reference_body,
        )
        from experiments.prospective_20260917 import build

        prefix, suffix = _evolve_shell(build.INITIAL.read_text())
        labels["reference:cheapest_tier_bottleneck_headroom"] = (
            prefix + _reference_body(build.STRONG.read_text()) + suffix
        )
    return labels


def final_task(layout, study, index):
    from evolution.evaluation import ScenarioGrid, load_task_evaluation_config

    row = root_row(layout, study, index)
    task = load_task_evaluation_config(V3 / f"task_{study}.json")
    return replace(task, scenarios=tuple(
        ScenarioGrid(scenario_id=s.scenario_id, path=s.path,
                     seeds=tuple(row["grids"][s.scenario_id]["final"]))
        for s in task.scenarios
    ))


def final_child(layout, study, index, source_sha):
    from evolution.evaluation import evaluate_policy_source
    from evolution.ladder import _evaluation_trace

    verify_p2(layout)
    directory = root_dir(layout, study, index) / "final"
    source = (directory / f"{source_sha}.ps").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != source_sha:
        raise ValueError("final source changed")
    started = time.monotonic()
    try:
        evaluation = evaluate_policy_source(source, final_task(layout, study, index),
                                            proposal_seq=0, retain_episode_ledgers=False)
        record = {"valid": True, "trace": _evaluation_trace(evaluation)}
    except Exception as exc:
        import placementscript as ps

        from evolution.evaluation import PolicyExecutionError

        if not isinstance(exc, (PolicyExecutionError, ps.PlacementScriptError)):
            raise
        record = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    record.update(source_sha256=source_sha, seconds=time.monotonic() - started, at=now())
    save_once(directory / f"{source_sha}.json", record)


def run_finals(layout, study, index, log):
    directory = root_dir(layout, study, index) / "final"
    directory.mkdir(exist_ok=True)
    labels = final_labels(layout, study, index)
    by_source = {}
    for label, source in labels.items():
        key = hashlib.sha256(source.encode()).hexdigest()
        by_source.setdefault(key, []).append(label)
        path = directory / f"{key}.ps"
        if not path.exists():
            path.write_text(source)
    save_once(directory / "LABELS.json", {"labels": {k: v for k, v in sorted(by_source.items())}})
    groups = sum(len(s.seeds) for s in final_task(layout, study, index).scenarios)
    for key in sorted(by_source):
        if (directory / f"{key}.json").exists():
            continue
        code = run_child(layout, ["final-child", study, str(index), key], log,
                         FINAL_CHILD_RSS_MIB, wall_s=FINAL_SECONDS_PER_EPISODE_GROUP * groups)
        if code != 0:
            raise RuntimeError(f"final evaluation failed for {study}{index:02d} {key}")
    save_once(directory / "COMPLETE.json", {"at": now(), "sources": len(by_source)})


# ---------------------------------------------------------------- orchestration


class Launcher:
    def __init__(self, layout):
        self.layout = layout
        self.state = layout.out / "STATE.jsonl"
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def log(self, **event):
        append_jsonl(self.state, {"at": now(), **event})

    def pair(self, study, index):
        layout = self.layout
        row = root_row(layout, study, index)
        directory = root_dir(layout, study, index)
        directory.mkdir(parents=True, exist_ok=True)
        if layout.smoke:
            prepare_smoke_inputs(layout, study, index)
        log = directory / "stages.log"
        order = [row["first_arm"], "gp" if row["first_arm"] == "llm" else "llm"]
        from experiments.matched_control_20260909.runner import certify_pair

        if not (directory / "pair.json").exists():
            for arm in order:
                code = run_child(layout, ["stage", study, str(index), arm], log,
                                 STAGE_FAMILY_RSS_MIB)
                self.log(root=row["campaign_id"], arm=arm, phase="initialization", exit=code)
                if code in (EXIT["unresolved"], EXIT["resource"], EXIT["budget"]):
                    # The pair cannot be certified; the root stays in the ledger as missing.
                    save_once(directory / "ROOT_STOPPED.json",
                              {"at": now(), "arm": arm, "phase": "initialization",
                               "exit": code})
                    self.log(root=row["campaign_id"], phase="root-stopped", exit=code)
                    return
                if code != EXIT["barrier"]:
                    raise RuntimeError(f"{row['campaign_id']} {arm} initialization exit {code}")
            certify_pair(directory / "gp/campaign", directory / "llm/campaign",
                         directory / "pair.json")
            self.log(root=row["campaign_id"], phase="pair-certified")
        # Study C scheduling: both arms search concurrently after the certified barrier.
        # Their archives, journals and evaluations are independent, so results are unchanged.
        exits, errors = {}, []

        def search(arm):
            try:
                exits[arm] = run_child(layout, ["stage", study, str(index), arm], log,
                                       STAGE_FAMILY_RSS_MIB)
            except Exception as exc:  # recorded and re-raised below
                errors.append(exc)

        workers = [threading.Thread(target=search, args=(arm,)) for arm in order]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            raise errors[0]
        for arm in order:
            code = exits[arm]
            self.log(root=row["campaign_id"], arm=arm, phase="search", exit=code)
            if code not in (EXIT["complete"], EXIT["unusable"], EXIT["budget"],
                            EXIT["unresolved"], EXIT["resource"]):
                raise RuntimeError(f"{row['campaign_id']} {arm} search exit {code}")
        seal_selection(layout, study, index, exits)
        self.log(root=row["campaign_id"], phase="selection-sealed")

    def finals(self, study, index):
        row = root_row(self.layout, study, index)
        run_finals(self.layout, study, index, root_dir(self.layout, study, index) / "final.log")
        self.log(root=row["campaign_id"], phase="finals-complete")

    def jobs(self):
        order = [(s, i) for i in range(self.layout.roots) for s in self.layout.studies]
        return [("pair", s, i) for s, i in order] + [("finals", s, i) for s, i in order]

    def done(self, kind, study, index):
        directory = root_dir(self.layout, study, index)
        if (directory / "ROOT_STOPPED.json").exists():
            return True
        if kind == "pair":
            return (directory / "SELECTION.json").exists()
        return (directory / "final/COMPLETE.json").exists()

    def ready(self, kind, study, index):
        directory = root_dir(self.layout, study, index)
        return kind == "pair" or (directory / "SELECTION.json").exists() or (
            directory / "ROOT_STOPPED.json").exists()

    def run(self):
        verify_p2(self.layout)
        pending = [j for j in self.jobs() if not self.done(*j)]
        running = set()
        self.log(phase="launch", pending=len(pending), slots=SLOTS)

        def work(job):
            try:
                getattr(self, job[0])(job[1], job[2])
            except Exception as exc:  # integrity or unexpected failure: stop new work
                self.log(job=list(job), phase="failed", error=f"{type(exc).__name__}: {exc}")
                self.stop.set()
            finally:
                with self.lock:
                    running.discard(job)

        threads = []
        while (pending or running) and not self.stop.is_set():
            with self.lock:
                free = SLOTS - len(running)
                startable = [j for j in pending if self.ready(*j)]
            for job in startable[:free]:
                pending.remove(job)
                with self.lock:
                    running.add(job)
                thread = threading.Thread(target=work, args=(job,), daemon=False)
                thread.start()
                threads.append(thread)
            time.sleep(5)
        for thread in threads:
            thread.join()
        self.log(phase="stopped" if self.stop.is_set() else "all-complete")


# ---------------------------------------------------------------- P2 freeze and status


def p2_path(layout):
    return layout.out / "P2_FREEZE.json"


def source_closure():
    from experiments.prospective_20260917 import build

    paths = build.source_paths()
    for folder in (
        HERE,
        ROOT / "experiments/matched_control_20260909",
        ROOT / "experiments/matched_transport_20260909",
        ROOT / "experiments/deepseek_pilots_bounded_20260923",
        ROOT / "experiments/deepseek_pilots_20260920",
        ROOT / "experiments/prospective_20260920",
    ):
        for p in folder.glob("*.py"):
            paths[str(p.relative_to(ROOT))] = p
    paths["experiments/glm_budget_pilot_20260917/run.py"] = (
        ROOT / "experiments/glm_budget_pilot_20260917/run.py"
    )
    for name in REFERENCE_FILES:
        paths[f"runtime/policies/{name}"] = ROOT / "runtime/policies" / name
    return {name: str(p) for name, p in sorted(paths.items())}


def amendment_paths(layout):
    return sorted(layout.out.glob("P2_AMENDMENT_*.json"))


def amend(layout, text):
    """Seal a technical amendment chained to the P2 freeze (or the previous amendment)."""
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    existing = amendment_paths(layout)
    previous = existing[-1] if existing else p2_path(layout)
    number = len(existing) + 1
    locations = source_closure()
    payload = {
        "schema": "corrected-ab-p2-amendment-v1",
        "number": number,
        "sealed_at_utc": now(),
        "previous_sha256": sha(previous),
        "text": text,
        "source_locations": locations,
        "source_files_sha256": {name: sha(path) for name, path in locations.items()},
        "environment_digest": digest(evaluation_snapshot()),
    }
    path = layout.out / f"P2_AMENDMENT_{number:02d}.json"
    save_once(path, {"payload": payload, "payload_sha256": digest(payload)})
    verify_p2(layout)
    print(json.dumps({"amendment": str(path), "sources": len(locations)}))


def verify_p2(layout):
    from experiments.gp_tuning_20260902.recovery import digest, read_sealed
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    frozen = read_sealed(p2_path(layout))
    current, previous = frozen, sha(p2_path(layout))
    for path in amendment_paths(layout):
        amendment = read_sealed(path)
        if amendment["previous_sha256"] != previous:
            raise ValueError(f"P2 amendment chain broken at {path.name}")
        current, previous = amendment, sha(path)
    for name, h in frozen["v3_files_sha256"].items():
        if sha(V3 / name) != h:
            raise ValueError(f"v3 input changed: {name}")
    for name, location in current["source_locations"].items():
        if sha(location) != current["source_files_sha256"][name]:
            raise ValueError(f"source changed since P2: {name}")
    if digest(evaluation_snapshot()) != current["environment_digest"]:
        raise ValueError("execution environment changed since P2")
    if frozen["smoke"] != layout.smoke:
        raise ValueError("P2 freeze belongs to a different layout")
    return frozen


AMENDMENTS = {
    "scheduling": "Six concurrent pair slots; after the paired initialization certificate both "
    "arms search concurrently. Completed evaluations are deterministic; execution limits (wall-time "
    "guards, provider failures) are handled per arm as in the A+B launcher.",
    "budget": "Per-root API ceiling $0.95 as in Study C. The investigator raised the project "
    "ceiling from $60 to $75, so prior spend plus 30 roots ($28.50) fits and the global ceiling "
    "cannot bind; a root's ceiling depends only on that root.",
    "inherited": "Settled empty/HTTP/no-response requests, connectivity waits, peak-window waits, "
    "memory back-pressure, registered per-episode evaluation guard, crash recovery and "
    "reconciliation exactly as in experiments/corrected_ab_20260925 (P2 amendments 01-04).",
    "registration": "prereg/STUDY_D_SINGLESEED_20261004 (frozen); Study C's roots, archives and "
    "final seeds with its single registered change reversed: evaluation_replicates 4 -> 1.",
}


def freeze(layout, authorization):
    from experiments.gp_tuning_20260902.recovery import digest, read_sealed
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot
    from experiments.prospective_20260917 import build

    v3 = read_sealed(V3 / "FREEZE.json")
    locations = source_closure()
    receipts = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            *sorted((ROOT / "experiments/scientific_gates_20260920").glob("*.json")),
            *sorted((ROOT / "experiments/scientific_gates_20260920").glob("*.md")),
            ROOT / "experiments/scientific_gates_20260920/des_v2_attempt_20260924/GATE_RESULT.json",
            ROOT / "experiments/ladder_fidelity_20260924/artifacts/SUMMARY.json",
            ROOT / "experiments/g_ablate_commercial_20260924/artifacts/RESULT.json",
            ROOT / "experiments/deepseek_concurrency_20260923/REPORT.md",
        ]
    }
    payload = {
        "schema": "corrected-ab-p2-freeze-v1",
        "frozen_at_utc": now(),
        "smoke": layout.smoke,
        "authorization": authorization,
        "v3_freeze_sha256": sha(V3 / "FREEZE.json"),
        "v3_files_sha256": v3["files_sha256"],
        "source_locations": locations,
        "source_files_sha256": {name: sha(path) for name, path in locations.items()},
        "environment_digest": digest(evaluation_snapshot()),
        "gate_receipts_sha256": receipts,
        "prior_exposure_microusd": prior_exposure_microusd(),
        "ceiling_microusd": CEILING_MICROUSD,
        "slots": SLOTS,
        "amendments": AMENDMENTS,
        "retired_des_v1_seeds_occupied": True,
        "initial_sha256": sha(build.INITIAL),
    }
    layout.out.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": digest(payload)}
    save_once(p2_path(layout), envelope)
    verify_p2(layout)
    print(json.dumps({"p2": str(p2_path(layout)), "sources": len(locations),
                      "prior_exposure_usd": payload["prior_exposure_microusd"] / 1e6}))


def supersede(layout, roots, reason):
    """Move sealed selections of re-opened roots aside; only before any final evaluation."""
    for campaign_id in roots:
        directory = layout.runs / campaign_id
        if (directory / "final").exists():
            raise ValueError(f"{campaign_id} already has final evaluations; cannot supersede")
        selection = directory / "SELECTION.json"
        if selection.exists():
            target = directory / "SELECTION_superseded_amendment02.json"
            if target.exists():
                raise ValueError(f"{campaign_id} was already superseded")
            selection.rename(target)
        append_jsonl(layout.out / "STATE.jsonl", {"at": now(), "phase": "selection-superseded",
                                                  "root": campaign_id, "reason": reason})


def status(layout):
    state = layout.out / "STATE.jsonl"
    rows = [json.loads(line) for line in state.open()] if state.exists() else []
    budget = GlobalBudget(layout, prior_exposure_microusd())
    sealed = len(list(layout.runs.glob("*/SELECTION.json")))
    finals = len(list(layout.runs.glob("*/final/COMPLETE.json")))
    exits = {}
    for row in rows:
        if row.get("phase") == "search":
            exits[str(row["exit"])] = exits.get(str(row["exit"]), 0) + 1
    print(json.dumps({
        "selections_sealed": sealed, "finals_complete": finals,
        "search_stage_exits": exits,
        "failures": [r for r in rows if r.get("phase") == "failed"],
        "roots_stopped": [r["root"] for r in rows if r.get("phase") == "root-stopped"],
        "project_exposure_usd": budget.exposure() / 1e6,
        "last_event": rows[-1] if rows else None,
    }, indent=1))


def current_layout():
    out = Path(os.environ.get("STUDY_D_LAYOUT", HERE))
    smoke = os.environ.get("STUDY_D_SMOKE") == "1"
    if smoke:
        return Layout(out=out, studies=("D",), roots=1, smoke=True)
    return SCIENTIFIC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action")
    parser.add_argument("rest", nargs="*")
    args = parser.parse_args()
    layout = current_layout()
    if args.action == "stage":
        stage(layout, args.rest[0], int(args.rest[1]), args.rest[2])
    elif args.action == "final-child":
        final_child(layout, args.rest[0], int(args.rest[1]), args.rest[2])
    elif args.action == "run":
        Launcher(layout).run()
    elif args.action == "status":
        status(layout)
    elif args.action == "freeze":
        freeze(layout, " ".join(args.rest))
    elif args.action == "amend":
        amend(layout, " ".join(args.rest))
    elif args.action == "supersede":
        supersede(layout, args.rest[1:], args.rest[0])
    else:
        raise SystemExit(f"unknown action {args.action}")


if __name__ == "__main__":
    main()
