"""Registered launcher for Study E (multi-family confirmatory study).

A copy of experiments/study_d_20261004/launch.py. Differences: registration directory, three
families T/W/R interleaved by index, project ceiling $145 and per-root caps $0.75 (evolving LLM
arm) and $0.20 (zero-shot) set by the investigator on 5 October 2026, prior spend that also
includes every Study D journal, a zero-shot LLM stage, the llm_champion_atN and zero-shot labels,
descriptive transfer finals of family-T labels on the drift scenario, a disk-space guard, and
fixes from the Codex pre-launch review (per-grid final resume, settlement of replayed requests,
crash-safe zero-shot log and seal).
Evaluation code paths of the evolving arms are unchanged.

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
V3 = ROOT / "prereg/STUDY_E_MULTIFAMILY_20261005"  # registration directory (name kept from A+B)
CEILING_MICROUSD = 145_000_000
SLOTS = 6
# Per-root API ceilings. Prior spend ($58.37 after D) plus 90 x ($0.75 + $0.20) = $143.87 fits the
# $145 project ceiling, so the global ceiling cannot bind and no root's budget depends on request
# order. Study C's most expensive (four-seed) LLM root cost $0.56.
PER_ROOT_CEILING_MICROUSD = 750_000
ZEROSHOT_CEILING_MICROUSD = 200_000
ZEROSHOT_PROPOSALS = 20
MIN_FREE_DISK_GIB = 8
DRIFT_ID = "dev-continuum-drift"
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
        "resource": 50, "account": 60}
# Registered single-episode child timeout (v1/v3 MODEL_AND_RESOURCES); the evaluation guard
# scales it by the episodes one ladder evaluation can run (P2 amendment 01).
EPISODE_TIMEOUT_S = 1200
EVALUATION_RSS_MIB = 2048


@dataclass(frozen=True)
class Layout:
    out: Path
    studies: tuple[str, ...] = ("T", "W", "R")
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


class ProviderAccountStop(SystemExit):
    """HTTP 403 account refusal (e.g. workspace budget): halts the launcher (amendment E-01)."""


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
    for name in ("study_c_20261003", "study_d_20261004"):
        earlier = ROOT / "experiments" / name
        smoke += sorted(earlier.glob("runs/*/llm/journal.jsonl"))
        smoke += sorted(earlier.glob("smoke*/runs/*/llm/journal.jsonl"))
    smoke += sorted(HERE.glob("smoke*/runs/*/zeroshot/journal.jsonl"))
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

    def settle_if_open(self, label, key, amount):
        """Settle a replayed request whose process died after the journal but before the
        global settlement, so stale full reservations cannot accumulate (Codex E review)."""
        with self.path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            rows = self._rows(stream)
            reserved = any(r["t"] == "reserve" and r["key"] == key for r in rows)
            settled = any(r["t"] == "settle" and r["key"] == key for r in rows)
            if reserved and not settled:
                stream.write(json.dumps({"t": "settle", "label": label, "key": key,
                                         "amount": amount, "at": now(),
                                         "replayed": True}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

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
            known = (key in completed or key in self._settled_empty or key in self._settled_http
                     or key in self._settled_unknown)
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
                else:
                    budget.settle_if_open(label, key, exc.cost_microusd)
                raise
            except (SettledHTTPError, SettledUnknownDispatch) as exc:
                if not known:
                    budget.settle(label, key, exc.cost_microusd)
                    if getattr(exc, "status", None) == 403:
                        # Account-level refusal (E-01): never a treatment outcome; stop everything.
                        append_jsonl(directory / "PAUSES.jsonl",
                                     {"at": now(), "failure": "ProviderAccountStop",
                                      "http_status": 403})
                        raise ProviderAccountStop(EXIT["account"]) from exc
                    self.consecutive_http_errors += 1
                    pause = min(60 * 2 ** (self.consecutive_http_errors - 1), 1800)
                    append_jsonl(directory / "PAUSES.jsonl",
                                 {"at": now(), "failure": type(exc).__name__,
                                  "http_status": getattr(exc, "status", None),
                                  "backoff_s": pause})
                    time.sleep(pause)
                else:
                    budget.settle_if_open(label, key, exc.cost_microusd)
                raise
            if not known:
                budget.settle(label, key, result.cost_microusd)
                self.consecutive_http_errors = 0
            else:
                budget.settle_if_open(label, key, result.cost_microusd)
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

    root = derive_seed(20261005, "study-e-launcher-smoke", study, index)
    grids = {}
    for scenario in load_task_evaluation_config(V3 / f"task_{study}.json").scenarios:
        grids[scenario.scenario_id] = {
            "search": [derive_seed(root, "sim", "scenario", scenario.scenario_id, "replicate", j)
                       for j in range(4)],
            "final": [derive_seed(root, "smoke-final", scenario.scenario_id, 0)],
        }
    row = {"campaign_id": f"S{study}{index:02d}", "root_seed": root, "grids": grids,
           "first_arm": "llm" if index % 2 == 0 else "gp", "evaluation_replicates": 4,
           "zeroshot_seed": derive_seed(root, "smoke-zeroshot")}
    if study == "T":
        row["transfer_final"] = {DRIFT_ID: [derive_seed(root, "smoke-transfer", 0)]}
    return row


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
                            "evaluation_replicates": 4}
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


# ---------------------------------------------------------------- zero-shot LLM stage


def zeroshot_stage(layout, study, index):
    """Up to ZEROSHOT_PROPOSALS one-step LLM proposals from the initialization archive only.

    Same operator, engine, proposal modes, retry and grammar-aware repair as the evolving LLM
    arm; parents and inspirations are drawn uniformly from the 12 initialization programs, built
    from the LLM arm's initialization records exactly as the campaign builds them. No proposal or
    score of this stage is ever shown to the model. Valid proposals go through the same ladder on
    the root's campaign-derived search grid, rotating islands (k mod n) as the campaign does, and
    the same archive assessment. Behavioural-novelty screening is not applied (disclosed; it can
    only let more zero-shot candidates reach L2).
    """
    import random

    from analysis.tracking import derive_seed
    from evolution.campaign import _default_assess
    from evolution.evaluation import ScenarioGrid, load_task_evaluation_config
    from evolution.islands import load_islands_config
    from evolution.ladder import load_ladder_config
    from evolution.llm_operator import OpenWeightLLMOperator, ValidationRepair
    from evolution.model import ApiCostLedger
    from evolution.openrouter_engine import load_openrouter_engine_config
    from evolution.operators import validate_scoped_edit
    from shinka.database import Program

    import placementscript as ps

    verify_p2(layout)
    plan, definition, _initialization, _purpose = campaign_inputs(layout, study, index)
    row = root_row(layout, study, index)
    base = root_dir(layout, study, index)
    directory = base / "zeroshot"
    directory.mkdir(parents=True, exist_ok=True)
    seal_path = directory / "ZEROSHOT.json"
    if seal_path.exists():
        try:
            json.loads(seal_path.read_text())
            raise SystemExit(EXIT["complete"])
        except json.JSONDecodeError:  # torn seal from a crash: rebuild it from the log
            seal_path.rename(directory / f"ZEROSHOT.torn-{time.time_ns()}.json")
    init_rows = []
    for line in (base / "llm" / "campaign" / "events.jsonl").read_text().splitlines():
        try:  # the LLM arm appends concurrently; a torn final line is skipped
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event["type"] == "candidate_completed" and event["payload"]["proposal_seq"] < 12:
            init_rows.append(event["payload"])
    init_rows.sort(key=lambda r: r["proposal_seq"])
    if [r["proposal_seq"] for r in init_rows] != list(range(12)):
        raise RuntimeError("zero-shot requires the LLM arm's complete initialization")

    def program(payload):  # evolution.campaign.CampaignRunner._program, verbatim fields
        evaluation = payload.get("evaluation")
        public = evaluation["evaluation"] if evaluation is not None else {}
        return Program(id=payload["candidate_id"], code=payload["proposal"]["source"],
                       language="python", parent_id=payload["parent_id"],
                       island_idx=payload["island_index"], generation=payload["proposal_seq"],
                       timestamp=0.0, public_metrics=public or {},
                       text_feedback=payload.get("feedback_text", ""),
                       correct=evaluation is not None,
                       metadata=payload["proposal"].get("metadata", {}))

    programs = [program(r) for r in init_rows]
    task_path = V3 / f"task_{study}.json"
    task = load_task_evaluation_config(task_path)
    task = replace(task, scenarios=tuple(
        ScenarioGrid(scenario_id=s.scenario_id, path=s.path, seeds=tuple(
            derive_seed(definition.root_seed, "sim", "scenario", s.scenario_id, "replicate", j)
            for j in range(definition.evaluation_replicates)))
        for s in task.scenarios))
    islands = load_islands_config(task_path)
    base_task = task
    ladder = replace(load_ladder_config(task_path), audit_root_seed=derive_seed(
        definition.root_seed, "evolution", "ladder-audit"))
    engine = load_openrouter_engine_config(V3 / "engine.json")
    engine = replace(engine, max_total_microusd=ZEROSHOT_CEILING_MICROUSD)
    client = scientific_client(layout, f"{row['campaign_id']}-zs", directory, engine)
    operator = OpenWeightLLMOperator(
        engine, client, ApiCostLedger(engine.max_total_microusd),
        proposal_modes=plan.llm_proposal_modes, profile=plan.profile,
        ablation_prompt_hint_probability=plan.ablation_prompt_hint_probability,
        proposals_per_call=plan.proposals_per_call,
    )
    evaluator = RecoveringEvaluator(directory)
    log_path = directory / "PROPOSALS.jsonl"

    def logged():
        if not log_path.exists():
            return []
        text = log_path.read_text()
        if text and not text.endswith("\n"):  # torn final line: terminate it so appends parse
            with log_path.open("a") as stream:
                stream.write("\n")
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    previous = logged()
    done = {r["k"] for r in previous}
    stop = next((r["stop"] for r in previous if r.get("stop")), None)
    proposals_n = 3 if layout.smoke else ZEROSHOT_PROPOSALS  # smoke checks mechanics only
    for k in range(proposals_n):
        if stop:
            break
        if k in done:
            continue
        # Mirror the campaign's island rotation: proposal k is scored under island k mod n.
        profile = islands.profiles[k % len(islands.profiles)]
        task = replace(base_task, constraints=profile.constraints(base_task.constraints))
        rng = random.Random(derive_seed(row["zeroshot_seed"], "proposal", k))
        parent = programs[rng.randrange(len(programs))]
        others = [p for p in programs if p is not parent]
        inspirations = tuple(rng.sample(others, min(plan.profile.max_inspirations, len(others))))
        source, attempts, repair, mode = None, [], None, None
        try:
            for attempt in range(plan.max_operator_attempts):
                arng = random.Random(derive_seed(row["zeroshot_seed"], "proposal", k,
                                                 "attempt", attempt))
                if repair is not None:
                    offspring = operator.propose_with_repair(parent, inspirations, arng,
                                                             mode=mode, repair=repair)
                else:
                    offspring = operator(parent, inspirations, arng)
                attempts.append({"attempt": attempt, "valid": offspring.valid,
                                 "error": offspring.error})
                if not offspring.valid:
                    if (offspring.error or "").startswith("proposal cost budget exhausted"):
                        stop = "zero-shot root API ceiling reached"
                        break
                    failure = offspring.metadata.get("validation_failure")
                    repair, mode = None, None
                    if (plan.profile.grammar_aware_retry_enabled and isinstance(failure, dict)
                            and isinstance(failure.get("error"), str)
                            and isinstance(failure.get("offending_excerpt"), str)):
                        try:
                            repair = ValidationRepair(error=failure["error"],
                                                      offending_excerpt=failure["offending_excerpt"])
                            mode = offspring.metadata.get("proposal_mode")
                        except ValueError:
                            repair = None
                    continue
                try:
                    ps.Policy(offspring.source)
                    validate_scoped_edit(parent.code, offspring.source)
                except (ps.PlacementScriptError, ValueError) as exc:
                    attempts[-1].update(valid=False, error=f"runner validation: {exc}")
                    continue
                source = offspring.source
                break
        except GlobalBudgetExhausted:
            stop = "zero-shot root API ceiling reached"
        record = {"k": k, "parent": parent.id, "inspirations": [p.id for p in inspirations],
                  "island": profile.island_id, "attempts": attempts, "valid": source is not None}
        if source is not None:
            try:
                ladder_result = evaluator(source, task, ladder, 1000 + k)
            except EvaluationResourceStop:
                stop = "evaluation resource guard reached"
                ladder_result = None
            if ladder_result is not None:
                result = ladder_result.to_dict()
                promoted = bool(result["promoted_to_l2"]) and result["candidate_failure"] is None
                if promoted:
                    try:  # the campaign's archive assessment can still invalidate a candidate
                        _default_assess(source, task, islands)
                    except ps.PlacementScriptError as exc:
                        promoted = False
                        record["assessment_failure"] = str(exc)
                fitness = (result.get("evaluation") or {}).get("fitness") if promoted else None
                record.update(source=source,
                              source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                              promoted_to_l2=promoted, search_fitness=fitness)
        if stop:
            record["stop"] = stop
        append_jsonl(log_path, record)
    proposals = logged()
    pool = [{"candidate_id": r["candidate_id"], "proposal_seq": r["proposal_seq"],
             "source": r["proposal"]["source"],
             "source_sha256": hashlib.sha256(r["proposal"]["source"].encode()).hexdigest(),
             "search_fitness": r["evaluation"]["evaluation"]["fitness"]}
            for r in full_records(init_rows)]
    pool += [{"candidate_id": f"zeroshot-{r['k']}", "proposal_seq": 1000 + r["k"],
              "source": r["source"], "source_sha256": r["source_sha256"],
              "search_fitness": r["search_fitness"]}
             for r in proposals if r.get("promoted_to_l2") and r.get("search_fitness")]
    champion = (min(pool, key=lambda r: (tuple(r["search_fitness"]["sort_key"]),
                                         r["proposal_seq"])) if pool else None)
    seal = {
        "sealed_at": now(), "proposals": len(proposals),
        "valid": sum(r["valid"] for r in proposals),
        "promoted_to_l2": sum(bool(r.get("promoted_to_l2")) for r in proposals),
        "stop": stop, "spent_microusd": client.ledger.spent_microusd,
        "champion": champion,
        "rule": "CandidateFitness.sort_key over initialization L2 records and zero-shot L2 "
                "records; proposal sequence tie-break; no final-grid information"}
    tmp = seal_path.with_suffix(".tmp")  # atomic seal: a crash never leaves a torn file
    tmp.write_text(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, seal_path)
    raise SystemExit(EXIT["complete"])


def wait_for_disk(log_path, what):
    """Block new children while free disk is below MIN_FREE_DISK_GIB; logged. Results unaffected."""
    import shutil

    waited = 0
    while shutil.disk_usage(ROOT).free < MIN_FREE_DISK_GIB * 2**30:
        time.sleep(60)
        waited += 60
    if waited:
        append_jsonl(log_path, {"at": now(), "disk_wait_s": waited, "for": what,
                                "minimum_gib": MIN_FREE_DISK_GIB})
    return waited


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
    wait_for_disk(layout.out / "DISK_WAITS.jsonl", " ".join(args))
    started = time.monotonic()
    with Path(log).open("a") as stream:
        stream.write(f"--- {now()} {' '.join(args)}\n")
        stream.flush()
        process = subprocess.Popen(
            [sys.executable, "-m", "experiments.study_e_20261005.launch", *args],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "STUDY_E_LAYOUT": str(layout.out),
                 "STUDY_E_SMOKE": "1" if layout.smoke else "0"},
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


def initialized(archive):
    """True if the archive holds all 12 initialization completions (E-01 reuse of GP arms)."""
    path = archive / "events.jsonl"
    if not path.exists():
        return False
    seqs = set()
    for line in path.open():
        event = json.loads(line)
        if event["type"] == "candidate_completed" and event["payload"]["proposal_seq"] < 12:
            seqs.add(event["payload"]["proposal_seq"])
    return seqs == set(range(12))


def terminal(archive):
    last = None
    if not (archive / "events.jsonl").exists():
        return None
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
    zeroshot = json.loads((directory / "zeroshot" / "ZEROSHOT.json").read_text())
    slots = zeroshot["proposals"]  # equal proposal opportunities for the K2 comparator
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
        if arm == "llm":
            arms[arm]["champion_atN"] = best(
                [r for r in full if r["proposal_seq"] < 12 + slots])
            arms[arm]["atN_slots"] = slots
    a, b = init_best["gp"], init_best["llm"]
    if (a is None) != (b is None) or (a and a["source_sha256"] != b["source_sha256"]):
        raise ValueError("initialization-best identities differ after parity verification")
    record = {"study": study, "index": index, "sealed_at": now(), "arms": arms,
              "initialization_best": a, "zeroshot": zeroshot,
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
    if selection["arms"]["llm"].get("champion_atN") is not None:
        labels["llm_champion_atN"] = selection["arms"]["llm"]["champion_atN"]["source"]
    if selection["zeroshot"]["champion"] is not None:
        labels["zeroshot_champion"] = selection["zeroshot"]["champion"]["source"]
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


def final_task(layout, study, index, grid="final"):
    from evolution.evaluation import ScenarioGrid, load_task_evaluation_config

    row = root_row(layout, study, index)
    if grid == "transfer":  # family-T labels on fresh drift seeds (descriptive)
        task = load_task_evaluation_config(V3 / "task_R.json")
        return replace(task, scenarios=tuple(
            ScenarioGrid(scenario_id=s.scenario_id, path=s.path,
                         seeds=tuple(row["transfer_final"][s.scenario_id]))
            for s in task.scenarios
        ))
    task = load_task_evaluation_config(V3 / f"task_{study}.json")
    return replace(task, scenarios=tuple(
        ScenarioGrid(scenario_id=s.scenario_id, path=s.path,
                     seeds=tuple(row["grids"][s.scenario_id]["final"]))
        for s in task.scenarios
    ))


GRID_DIRS = {"final": "final", "transfer": "final_transfer_drift"}


def final_child(layout, study, index, source_sha, grid="final"):
    from evolution.evaluation import evaluate_policy_source
    from evolution.ladder import _evaluation_trace

    verify_p2(layout)
    directory = root_dir(layout, study, index) / GRID_DIRS[grid]
    source = (directory / f"{source_sha}.ps").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != source_sha:
        raise ValueError("final source changed")
    started = time.monotonic()
    try:
        evaluation = evaluate_policy_source(source, final_task(layout, study, index, grid),
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


def run_finals(layout, study, index, log, grid="final"):
    directory = root_dir(layout, study, index) / GRID_DIRS[grid]
    if (directory / "COMPLETE.json").exists():  # grids resume independently
        return
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
    groups = sum(len(s.seeds) for s in final_task(layout, study, index, grid).scenarios)
    for key in sorted(by_source):
        if (directory / f"{key}.json").exists():
            continue
        code = run_child(layout, ["final-child", study, str(index), key, grid], log,
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
                if initialized(directory / arm / "campaign"):
                    self.log(root=row["campaign_id"], arm=arm,
                             phase="initialization-reused")  # E-01 re-certification
                    continue
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

        def zeroshot():
            try:
                exits["zeroshot"] = run_child(layout, ["stage-zeroshot", study, str(index)], log,
                                              STAGE_FAMILY_RSS_MIB)
            except Exception as exc:  # recorded and re-raised below
                errors.append(exc)

        searching = []
        for arm in order:
            if terminal(directory / arm / "campaign") is not None:  # E-01: finished arm reused
                exits[arm] = EXIT["complete"]
                self.log(root=row["campaign_id"], arm=arm, phase="search-reused")
            else:
                searching.append(arm)
        workers = [threading.Thread(target=search, args=(arm,)) for arm in searching]
        if not (directory / "zeroshot" / "ZEROSHOT.json").exists():
            workers.append(threading.Thread(target=zeroshot))
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            raise errors[0]
        if "zeroshot" in exits:
            self.log(root=row["campaign_id"], arm="zeroshot", phase="search",
                     exit=exits["zeroshot"])
            if exits["zeroshot"] != EXIT["complete"]:
                raise RuntimeError(f"{row['campaign_id']} zeroshot exit {exits['zeroshot']}")
        for arm in order:
            code = exits[arm]
            if arm in searching:
                self.log(root=row["campaign_id"], arm=arm, phase="search", exit=code)
            if code not in (EXIT["complete"], EXIT["unusable"], EXIT["budget"],
                            EXIT["unresolved"], EXIT["resource"]):
                raise RuntimeError(f"{row['campaign_id']} {arm} search exit {code}")
        seal_selection(layout, study, index, exits)
        self.log(root=row["campaign_id"], phase="selection-sealed")

    def finals(self, study, index):
        row = root_row(self.layout, study, index)
        run_finals(self.layout, study, index, root_dir(self.layout, study, index) / "final.log")
        if study == "T":
            run_finals(self.layout, study, index,
                       root_dir(self.layout, study, index) / "final.log", grid="transfer")
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
        done = (directory / "final/COMPLETE.json").exists()
        if study == "T":
            done = done and (directory / "final_transfer_drift/COMPLETE.json").exists()
        return done

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
    "scheduling": "Six concurrent pair slots over families T, W, R interleaved by index; after the "
    "paired initialization certificate the two evolving arms and the zero-shot stage run "
    "concurrently. Completed evaluations are deterministic; execution limits are handled per arm.",
    "budget": "Per-root API ceilings $0.75 (evolving LLM arm) and $0.20 (zero-shot). The "
    "investigator set the project ceiling to $145, so prior spend plus 90 x $0.95 fits and the "
    "global ceiling cannot bind; a root's ceilings depend only on that root.",
    "disk": "New children wait while free disk is below 8 GiB (after the Study D disk-full fault).",
    "inherited": "Settled empty/HTTP/no-response requests, connectivity waits, peak-window waits, "
    "memory back-pressure, registered per-episode evaluation guard, crash recovery and "
    "reconciliation exactly as in experiments/corrected_ab_20260925 (P2 amendments 01-04).",
    "registration": "prereg/STUDY_E_MULTIFAMILY_20261005 (frozen); three families, Study C "
    "treatments, zero-shot arm and transfer finals as in PROTOCOL.md.",
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
        if (directory / "final").exists() or (directory / "final_transfer_drift").exists():
            raise ValueError(f"{campaign_id} already has final evaluations; cannot supersede")
        selection = directory / "SELECTION.json"
        if selection.exists():
            target = directory / "SELECTION_superseded_amendment02.json"
            if target.exists():
                raise ValueError(f"{campaign_id} was already superseded")
            selection.rename(target)
        append_jsonl(layout.out / "STATE.jsonl", {"at": now(), "phase": "selection-superseded",
                                                  "root": campaign_id, "reason": reason})


def supersede_e01(layout):
    """Amendment E-01: set aside LLM/zero-shot arms that met an HTTP 403 account refusal.

    Only before any final evaluation of the root. The arm directories, the selection and (when
    the LLM arm is set aside) the pair receipt move to <root>/superseded_E01/; GP arms stay. The
    launcher then re-certifies the fresh LLM arm against the unchanged GP arm and reruns."""
    moved = []
    for directory in sorted(layout.runs.iterdir()):
        arms = [arm for arm in ("llm", "zeroshot")
                if '"http_status": 403' in ((directory / arm / "PAUSES.jsonl").read_text()
                                            if (directory / arm / "PAUSES.jsonl").exists() else "")]
        if not arms:
            continue
        if (directory / "final").exists() or (directory / "final_transfer_drift").exists():
            raise ValueError(f"{directory.name} already has final evaluations")
        aside = directory / "superseded_E01"
        aside.mkdir()
        items = list(arms) + ["SELECTION.json"] + (["pair.json"] if "llm" in arms else [])
        for item in items:
            if (directory / item).exists():
                (directory / item).rename(aside / item)
        moved.append({"root": directory.name, "moved": items})
        append_jsonl(layout.out / "STATE.jsonl", {"at": now(), "phase": "superseded-E01",
                                                  "root": directory.name, "moved": items})
    print(json.dumps(moved, indent=1))


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
    out = Path(os.environ.get("STUDY_E_LAYOUT", HERE))
    smoke = os.environ.get("STUDY_E_SMOKE") == "1"
    if smoke:
        return Layout(out=out, studies=("T", "W", "R"), roots=1, smoke=True)
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
        final_child(layout, args.rest[0], int(args.rest[1]), args.rest[2], *args.rest[3:4])
    elif args.action == "stage-zeroshot":
        zeroshot_stage(layout, args.rest[0], int(args.rest[1]))
    elif args.action == "run":
        Launcher(layout).run()
    elif args.action == "status":
        status(layout)
    elif args.action == "freeze":
        freeze(layout, " ".join(args.rest))
    elif args.action == "amend":
        amend(layout, " ".join(args.rest))
    elif args.action == "supersede-e01":
        supersede_e01(layout)
    elif args.action == "supersede":
        supersede(layout, args.rest[1:], args.rest[0])
    else:
        raise SystemExit(f"unknown action {args.action}")


if __name__ == "__main__":
    main()
