"""Registered launcher for Study F (engine-strength study with validation re-selection).

Study E's 90 roots, inputs and treatments are reused unchanged (experiments/study_e_20261005,
prereg/STUDY_E_MULTIFAMILY_20261005). Differences, all fixed in prereg/STUDY_F_SOL_20261007:
  * only the evolving LLM arm is run, with gpt-6.1-sol through the Codex CLI subscription
    (reasoning effort medium, isolated empty read-only workspace); it is paired to E's GP arm by
    re-certifying the initialization barrier exactly as E's amendment E-01 did;
  * validation re-selection for three arms: Sol (this study), GP and DeepSeek (E's archives,
    read-only). Each arm's top-5 distinct L2 programs by search fitness are scored on fresh
    validation seeds; the validated champion is sealed before any final grid;
  * final scoring of sol_champion, sol_validated, gp_validated and deepseek_validated on E's
    final grid; a program E already scored on that grid is copied byte-for-byte.
Subscription usage-limit refusals are never treatment outcomes: the arm is set aside unchanged
and rerun from scratch after the quota recovers (as E-01).

Monitoring (`status`) reports progress and terminal states, never comparative outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import psutil

from evolution import codex_engine
from evolution.codex_engine import (
    _EXPECTED_GENERATION,
    _VERSION_PATTERN,
    CODEX_ENGINE_NAME,
    CodexCLICompletionClient,
    CodexCLICompletionError,
    CodexEngineConfig,
)
from experiments.study_e_20261005 import launch as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = ROOT / "prereg/STUDY_F_SOL_20261007"
E_DIR = ROOT / "experiments/study_e_20261005"
MODULE = "experiments.study_f_20261007.launch"
MODEL_ID = "gpt-6.1-sol"
CLI_SERIES = (0, 159)
REASONING_EFFORT = "medium"
SOL_SLOTS = 10
VALIDATION_SLOTS = 3
TOP_K = 5
VALIDATION_SEEDS = 8
SMOKE_TOP_K = 2
SMOKE_VALIDATION_SEEDS = 1
QUOTA_PROBE_INTERVAL_S = 1800
MAX_QUOTA_RESTARTS = 6
EXIT = {**E.EXIT, "harness": 70}
QUOTA = re.compile(r"rate[ _-]?limit|usage[ _-]?limit|too many requests|quota|"
                   r"limit (?:has been )?reached|try again (?:at|in)", re.I)
VALIDATION_ARMS = ("sol", "gp", "deepseek")
FINAL_LABELS = ("sol_champion", "sol_validated", "gp_validated", "deepseek_validated")


@dataclass(frozen=True)
class Layout:
    out: Path
    smoke: bool = False

    @property
    def runs(self):
        return self.out / "runs"

    @property
    def roots(self):
        return 1 if self.smoke else 30

    @property
    def e(self):
        """Study E's layout for the same roots (smoke roots are E's reserved smoke roots)."""
        if self.smoke:
            return E.Layout(out=E_DIR / "smoke", studies=("T", "W", "R"), roots=1, smoke=True)
        return E.Layout(out=E_DIR, studies=("T", "W", "R"), roots=30, smoke=False)


def current_layout():
    smoke = os.environ.get("STUDY_F_SMOKE") == "1"
    return Layout(out=Path(os.environ.get("STUDY_F_LAYOUT", HERE / ("smoke" if smoke else ""))),
                  smoke=smoke)


def campaign_id(layout, study, index):
    return E.root_row(layout.e, study, index)["campaign_id"]


def f_root(layout, study, index):
    return layout.runs / campaign_id(layout, study, index)


def e_root(layout, study, index):
    return E.root_dir(layout.e, study, index)


# ---------------------------------------------------------------- engine


@dataclass(frozen=True)
class SolEngineConfig(CodexEngineConfig):
    reasoning_effort: str = REASONING_EFFORT

    def __post_init__(self):
        # Study F's registration replaces A1's development-only identity pins (A1 option b).
        if (self.status != "study-f-registered" or self.engine != CODEX_ENGINE_NAME
                or self.model_id != MODEL_ID or self.required_cli_series != "0.159.x"
                or self.reasoning_effort != REASONING_EFFORT
                or self.generation != _EXPECTED_GENERATION
                or self.a1_reference != "prereg/STUDY_F_SOL_20261007/PROTOCOL.md"):
            raise ValueError("Study F engine configuration differs from the registration")


def load_engine():
    raw = json.loads((REG / "engine.json").read_text())
    if raw.pop("schema") != "evolved-continuum-codex-cli-engine-v1":
        raise ValueError("engine schema")
    return SolEngineConfig(**raw)


class QuotaStop(SystemExit):
    pass


class HarnessStop(SystemExit):
    """A host-side failure of the proposal transport (amendment F-01): never a treatment outcome."""


_original_stop_process_group = codex_engine._stop_process_group


def _stop_process_group(process):
    """F-01/F-02: on macOS, killpg can raise EPERM while the CLI's process group is exiting.

    The CLI often exits right after `turn.completed`, racing the harness's cleanup. When killpg
    is refused, the leader is signalled directly (terminate, then kill) and reaped; the error is
    re-raised only if the leader cannot be stopped.
    """
    try:
        _original_stop_process_group(process)
    except PermissionError:
        for send in (process.terminate, process.kill):
            if process.poll() is not None:
                break
            try:
                send()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                continue
        if process.poll() is None:
            raise


codex_engine._stop_process_group = _stop_process_group


class SolClient(CodexCLICompletionClient):
    def __init__(self, config, pauses):
        self.pauses = pauses
        super().__init__(config)
        self.engine_descriptor.update(development_only=False,
                                      reasoning_effort=config.reasoning_effort,
                                      registration=config.a1_reference)

    def _read_and_validate_version(self):
        with tempfile.TemporaryDirectory(prefix="ec-codex-version-") as workspace:
            result = subprocess.run([self.executable, "--version"], cwd=workspace, check=False,
                                    capture_output=True, text=True, timeout=10.0,
                                    env=self._subprocess_environment)
        match = _VERSION_PATTERN.search((result.stdout or result.stderr).strip())
        if result.returncode != 0 or match is None:
            raise RuntimeError("unable to read a semantic version from Codex CLI")
        if tuple(int(p) for p in match.groups()[:2]) != CLI_SERIES:
            raise RuntimeError(f"Codex CLI {match.group(0)!r} is outside series 0.159.x")
        return ".".join(match.groups())

    def _normalized_command(self):
        command = super()._normalized_command()
        at = command.index("--sandbox")
        return command[:at] + ["-c", f"model_reasoning_effort={self.config.reasoning_effort}"] + (
            command[at:])

    def complete(self, **kwargs):
        try:
            return super().complete(**kwargs)
        except CodexCLICompletionError as exc:
            text = f"{exc} {json.dumps(exc.archive_metadata, default=str)}"
            if QUOTA.search(text):
                E.append_jsonl(self.pauses, {"at": E.now(), "failure": "subscription-quota",
                                             "category": exc.category, "message": str(exc)[:500]})
                raise QuotaStop(EXIT["account"]) from exc
            raise
        except Exception as exc:
            # F-01: a host failure (not a classified CLI outcome) stops the arm; it is set aside
            # and rerun from scratch. Without this, the unsettled reservation made every later
            # attempt fail instantly ("already has a pending dispatch").
            import traceback

            E.append_jsonl(self.pauses, {"at": E.now(), "failure": "harness",
                                         "error": f"{type(exc).__name__}: {exc}"[:500],
                                         "traceback": traceback.format_exc()[-4000:]})
            raise HarnessStop(EXIT["harness"]) from exc


def probe_quota():
    """A trivial call outside any campaign; True when the subscription answers."""
    with tempfile.TemporaryDirectory(prefix="ec-codex-probe-") as workspace:
        result = subprocess.run(
            ["codex", "exec", "-", "-m", MODEL_ID, "-c", "model_reasoning_effort=low",
             "--sandbox", "read-only", "-C", workspace, "--skip-git-repo-check", "--ephemeral",
             "--ignore-user-config", "--color", "never", "--json"],
            input="Reply with the single word OK.", capture_output=True, text=True,
            timeout=300, check=False)
    return result.returncode == 0 and '"turn.completed"' in result.stdout


# ---------------------------------------------------------------- Sol stage (child process)


def stage(layout, study, index):
    from experiments.matched_control_20260909.runner import InitializationPending

    verify_freeze(layout)
    try:
        _stage(layout, study, index)
    except InitializationPending:
        raise SystemExit(EXIT["barrier"]) from None
    except E.EvaluationResourceStop:
        raise SystemExit(EXIT["resource"]) from None
    except RuntimeError as exc:
        text = str(exc)
        if ("proposal-engine ceiling" in text or "candidate safety ceiling" in text
                or "archive is unusable" in text):
            raise SystemExit(EXIT["unusable"]) from None
        raise
    raise SystemExit(EXIT["complete"])


def _stage(layout, study, index):
    from evolution.campaign import _default_assess
    from evolution.evaluation import load_task_evaluation_config
    from evolution.feedback import load_feedback_profiles
    from evolution.islands import load_islands_config
    from evolution.ladder import load_ladder_config
    from evolution.llm_operator import OpenWeightLLMOperator
    from evolution.model import ApiCostLedger
    from experiments.allbaselines_harness.initialization import read_seed_archive
    from experiments.matched_control_20260909.runner import matched_campaign_runner
    from experiments.prospective_20260917 import build

    plan, definition, initialization, purpose = E.campaign_inputs(layout.e, study, index)
    directory = f_root(layout, study, index) / "llm"
    directory.mkdir(parents=True, exist_ok=True)
    config = load_engine()
    client = SolClient(config, directory / "PAUSES.jsonl")
    operator = OpenWeightLLMOperator(
        config, client, ApiCostLedger(0),
        proposal_modes=plan.llm_proposal_modes, profile=plan.profile,
        ablation_prompt_hint_probability=plan.ablation_prompt_hint_probability,
        proposals_per_call=plan.proposals_per_call,
    )
    task_path = E.V3 / f"task_{study}.json"
    seed_archive = read_seed_archive(initialization, expected_size=12,
                                     initial_path=build.INITIAL)
    runner = matched_campaign_runner(seed_archive, f_root(layout, study, index) / "pair.json")(
        definition=definition, plan=plan, arm="llm", operator=operator,
        initial_source=build.INITIAL.read_text(),
        task=load_task_evaluation_config(task_path),
        ladder=load_ladder_config(task_path),
        islands=load_islands_config(task_path),
        feedback_profile=load_feedback_profiles(task_path)[plan.feedback_profile],
        archive_dir=directory / "campaign",
        provenance={"purpose": "study-f-sol", "freeze_sha256": E.sha(freeze_path(layout)),
                    "arm": "llm"},
        run_purpose=purpose,
        evaluate_fn=E.RecoveringEvaluator(directory),
        assess_fn=_default_assess,
    )
    runner.run()


# ---------------------------------------------------------------- selection and validation


def arm_archive(layout, study, index, arm):
    if arm == "sol":
        return f_root(layout, study, index) / "llm/campaign"
    return e_root(layout, study, index) / {"gp": "gp", "deepseek": "llm"}[arm] / "campaign"


def top_k(rows, k):
    ordered = sorted(rows, key=lambda r: (tuple(r["evaluation"]["evaluation"]["fitness"]
                                                ["sort_key"]), r["proposal_seq"]))
    out, seen = [], set()
    for row in ordered:
        rec = E.best([row])
        if rec["source_sha256"] not in seen:
            seen.add(rec["source_sha256"])
            out.append(rec)
        if len(out) == k:
            break
    return out


def candidates(layout, study, index, arm):
    full = E.full_records(E.completed_rows(arm_archive(layout, study, index, arm)))
    return top_k(full, SMOKE_TOP_K if layout.smoke else TOP_K)


def validation_task(layout, study, index):
    from analysis.tracking import derive_seed
    from evolution.evaluation import ScenarioGrid, load_task_evaluation_config

    row = E.root_row(layout.e, study, index)
    label = "study-f-smoke-validation" if layout.smoke else "study-f-validation"
    count = SMOKE_VALIDATION_SEEDS if layout.smoke else VALIDATION_SEEDS
    task = load_task_evaluation_config(E.V3 / f"task_{study}.json")
    return replace(task, scenarios=tuple(
        ScenarioGrid(scenario_id=s.scenario_id, path=s.path,
                     seeds=tuple(derive_seed(row["root_seed"], label, s.scenario_id, j)
                                 for j in range(count)))
        for s in task.scenarios
    ))


def evaluate_child(layout, study, index, directory, source_sha, grid):
    """Score one program on the validation or final grid (child process)."""
    from evolution.evaluation import evaluate_policy_source
    from evolution.ladder import _evaluation_trace

    verify_freeze(layout)
    directory = Path(directory)
    source = (directory / f"{source_sha}.ps").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != source_sha:
        raise ValueError("source changed")
    task = (validation_task(layout, study, index) if grid == "validation"
            else E.final_task(layout.e, study, index))
    started = time.monotonic()
    try:
        evaluation = evaluate_policy_source(source, task, proposal_seq=0,
                                            retain_episode_ledgers=False)
        record = {"valid": True, "trace": _evaluation_trace(evaluation)}
    except Exception as exc:
        import placementscript as ps

        from evolution.evaluation import PolicyExecutionError

        if not isinstance(exc, (PolicyExecutionError, ps.PlacementScriptError)):
            raise
        record = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    record.update(source_sha256=source_sha, grid=grid, seconds=time.monotonic() - started,
                  at=E.now())
    E.save_once(directory / f"{source_sha}.json", record)


def score(layout, study, index, directory, sources, grid, log):
    """Score {sha: source} into directory with guarded children; resumable."""
    directory.mkdir(parents=True, exist_ok=True)
    groups = sum(len(s.seeds) for s in (validation_task(layout, study, index) if grid ==
                                        "validation" else E.final_task(layout.e, study,
                                                                       index)).scenarios)
    for sha, source in sorted(sources.items()):
        if (directory / f"{sha}.json").exists():
            continue
        path = directory / f"{sha}.ps"
        if not path.exists():
            path.write_text(source)
        code = run_child(layout, ["evaluate-child", study, str(index), str(directory), sha, grid],
                         log, E.FINAL_CHILD_RSS_MIB,
                         wall_s=E.FINAL_SECONDS_PER_EPISODE_GROUP * groups)
        if code != 0:
            raise RuntimeError(f"{grid} evaluation failed for {directory} {sha}")


def validation_key(record, candidate):
    if record["valid"]:
        key = (0, tuple(record["trace"]["fitness_sort_key"]))
    else:
        key = (1, ())
    return key + (tuple(candidate["search_fitness"]["sort_key"]), candidate["proposal_seq"])


def validate_arm(layout, study, index, arm):
    """Score one arm's top-k on the validation grid and seal its validated champion."""
    root = f_root(layout, study, index)
    seal = root / "validation" / f"{arm}.json"
    if seal.exists():
        return
    if arm == "sol" and E.terminal(arm_archive(layout, study, index, arm)) is None:
        raise RuntimeError("Sol arm not terminal")
    picks = candidates(layout, study, index, arm)
    directory = root / "validation" / arm
    score(layout, study, index, directory, {c["source_sha256"]: c["source"] for c in picks},
          "validation", root / "validation.log")
    validated = None
    if picks:
        records = {c["source_sha256"]: json.loads((directory / f"{c['source_sha256']}.json")
                                                  .read_text()) for c in picks}
        best = min(picks, key=lambda c: validation_key(records[c["source_sha256"]], c))
        record = records[best["source_sha256"]]
        validated = {**best, "validation_sort_key": (record["trace"]["fitness_sort_key"]
                                                     if record["valid"] else None)}
    E.save_once(seal, {"arm": arm, "sealed_at": E.now(), "top_k": picks, "validated": validated,
                       "rule": "top-k distinct L2 programs by search sort_key; validated "
                               "champion = best validation fitness_sort_key (invalid last), "
                               "then search sort_key, then proposal sequence"})


def seal_selection(layout, study, index, exits):
    root = f_root(layout, study, index)
    arms, validated = {}, {}
    for arm in VALIDATION_ARMS:
        archive = arm_archive(layout, study, index, arm)
        rows = E.completed_rows(archive)
        full = E.full_records(rows)
        v = json.loads((root / "validation" / f"{arm}.json").read_text())
        arms[arm] = {"terminal": E.terminal(archive), "completed_candidates": len(rows),
                     "full_records": len(full), "champion": E.best(full), "top_k": v["top_k"],
                     "archive": str(archive.relative_to(ROOT))}
        validated[arm] = v["validated"]
    arms["sol"]["stage_exit"] = exits.get("sol")
    e_selection = json.loads((e_root(layout, study, index) / "SELECTION.json").read_text())
    for arm, e_arm in (("gp", "gp"), ("deepseek", "llm")):
        mine, theirs = arms[arm]["champion"], e_selection["arms"][e_arm]["champion"]
        if (mine and mine["source_sha256"]) != (theirs and theirs["source_sha256"]):
            raise ValueError(f"{arm} search champion differs from E's sealed selection")
    sol_init = E.best([r for r in E.full_records(E.completed_rows(arm_archive(
        layout, study, index, "sol"))) if r["proposal_seq"] < 12])
    if (sol_init and e_selection["initialization_best"]
            and sol_init["source_sha256"] != e_selection["initialization_best"]["source_sha256"]):
        raise ValueError("initialization-best identities differ from E")
    row = E.root_row(layout.e, study, index)
    E.save_once(root / "SELECTION.json", {
        "study": study, "index": index, "campaign_id": row["campaign_id"], "sealed_at": E.now(),
        "arms": arms, "validated": validated, "initialization_best": sol_init,
        "e_selection_sha256": E.sha(e_root(layout, study, index) / "SELECTION.json"),
        "rule": "search champion: CandidateFitness.sort_key over completed L2 records, "
                "proposal-sequence tie-break; validated champion: see validation/<arm>.json; "
                "no final-grid information"})


# ---------------------------------------------------------------- finals


def run_finals(layout, study, index):
    root = f_root(layout, study, index)
    final = root / "final"
    if (final / "COMPLETE.json").exists():
        return
    final.mkdir(exist_ok=True)
    selection = json.loads((root / "SELECTION.json").read_text())
    picks = {"sol_champion": selection["arms"]["sol"]["champion"],
             "sol_validated": selection["validated"]["sol"],
             "gp_validated": selection["validated"]["gp"],
             "deepseek_validated": selection["validated"]["deepseek"]}
    by_sha, sources = {}, {}
    for label, rec in picks.items():
        if rec is not None:
            by_sha.setdefault(rec["source_sha256"], []).append(label)
            sources[rec["source_sha256"]] = rec["source"]
    E.save_once(final / "LABELS.json", {"labels": dict(sorted(by_sha.items()))})
    e_final = e_root(layout, study, index) / "final"
    e_labels = json.loads((e_final / "LABELS.json").read_text())["labels"]
    copied = {}
    for sha in sorted(sources):
        if sha in e_labels and (e_final / f"{sha}.json").exists():
            if not (final / f"{sha}.json").exists():
                shutil.copyfile(e_final / f"{sha}.json", final / f"{sha}.json")
            copied[sha] = str((e_final / f"{sha}.json").relative_to(ROOT))
    remaining = {s: src for s, src in sources.items() if s not in copied}
    score(layout, study, index, final, remaining, "final", root / "final.log")
    E.save_once(final / "REUSED.json", {"copied": copied})
    E.save_once(final / "COMPLETE.json", {"at": E.now(), "sources": len(sources),
                                          "copied": len(copied)})


# ---------------------------------------------------------------- children and orchestration


def run_child(layout, args, log, rss_mib, wall_s=None):
    E.wait_for_memory(E.MIN_AVAILABLE_MIB, layout.out / "MEMORY_WAITS.jsonl", " ".join(args))
    E.wait_for_disk(layout.out / "DISK_WAITS.jsonl", " ".join(args))
    started = time.monotonic()
    with Path(log).open("a") as stream:
        stream.write(f"--- {E.now()} {' '.join(args)}\n")
        stream.flush()
        process = subprocess.Popen(
            [sys.executable, "-m", MODULE, *args], cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "STUDY_F_LAYOUT": str(layout.out),
                 "STUDY_F_SMOKE": "1" if layout.smoke else "0"})
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


class Launcher:
    def __init__(self, layout):
        self.layout = layout
        self.state = layout.out / "STATE.jsonl"
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.quota_lock = threading.Lock()
        self.quota_clear = threading.Event()
        self.quota_clear.set()

    def log(self, **event):
        E.append_jsonl(self.state, {"at": E.now(), **event})

    def quota_pause(self, root):
        self.quota_clear.clear()
        with self.quota_lock:
            if self.quota_clear.is_set():
                return
            self.log(root=root, phase="quota-pause")
            while not self.stop.is_set():
                time.sleep(QUOTA_PROBE_INTERVAL_S)
                if probe_quota():
                    break
            self.log(root=root, phase="quota-resume")
            self.quota_clear.set()

    def set_aside(self, study, index, n, kind="quota"):
        root = f_root(self.layout, study, index)
        aside = root / f"superseded_{kind}_{n}"
        aside.mkdir()
        for item in ("llm", "pair.json"):
            if (root / item).exists():
                (root / item).rename(aside / item)
        self.log(root=root.name, phase=f"superseded-{kind}", to=aside.name)

    def sol(self, study, index):
        layout = self.layout
        root = f_root(layout, study, index)
        root.mkdir(parents=True, exist_ok=True)
        cid = root.name
        log = root / "stages.log"
        e = e_root(layout, study, index)
        if not E.initialized(e / "gp/campaign") or E.terminal(e / "gp/campaign") is None:
            raise RuntimeError(f"{cid}: E's GP arm is not complete")
        from experiments.matched_control_20260909.runner import certify_pair

        for attempt in range(MAX_QUOTA_RESTARTS + 1):
            self.quota_clear.wait()
            code = None
            if not (root / "pair.json").exists():
                if not E.initialized(root / "llm/campaign"):
                    code = run_child(layout, ["stage", study, str(index)], log,
                                     E.STAGE_FAMILY_RSS_MIB)
                    self.log(root=cid, arm="sol", phase="initialization", exit=code)
                    if code not in (EXIT["barrier"], EXIT["account"], EXIT["harness"]):
                        if code in (EXIT["resource"], EXIT["unresolved"]):
                            E.save_once(root / "ROOT_STOPPED.json",
                                        {"at": E.now(), "phase": "initialization", "exit": code})
                            return
                        raise RuntimeError(f"{cid} initialization exit {code}")
                if code not in (EXIT["account"], EXIT["harness"]):
                    certify_pair(e / "gp/campaign", root / "llm/campaign", root / "pair.json")
                    self.log(root=cid, phase="pair-certified")
            if code not in (EXIT["account"], EXIT["harness"]):
                if E.terminal(root / "llm/campaign") is not None:
                    code = EXIT["complete"]
                else:
                    code = run_child(layout, ["stage", study, str(index)], log,
                                     E.STAGE_FAMILY_RSS_MIB)
                    self.log(root=cid, arm="sol", phase="search", exit=code)
            if code == EXIT["account"]:
                self.set_aside(study, index, attempt)
                self.quota_pause(cid)
                continue
            if code == EXIT["harness"]:
                self.set_aside(study, index, attempt, kind="harness")
                continue
            break
        else:
            E.save_once(root / "ROOT_STOPPED.json", {"at": E.now(), "phase": "quota"})
            return
        if code not in (EXIT["complete"], EXIT["unusable"], EXIT["resource"]):
            raise RuntimeError(f"{cid} search exit {code}")
        if code == EXIT["resource"]:
            E.save_once(root / "ROOT_STOPPED.json", {"at": E.now(), "phase": "search",
                                                     "exit": code})
            return
        E.save_once(root / "SOL_EXIT.json", {"exit": code, "at": E.now()})

    def validate(self, study, index, arm):
        validate_arm(self.layout, study, index, arm)
        self.log(root=campaign_id(self.layout, study, index), arm=arm, phase="validated")

    def seal(self, study, index):
        root = f_root(self.layout, study, index)
        exit_code = json.loads((root / "SOL_EXIT.json").read_text())["exit"]
        seal_selection(self.layout, study, index, {"sol": exit_code})
        self.log(root=root.name, phase="selection-sealed")

    def finals(self, study, index):
        run_finals(self.layout, study, index)
        self.log(root=campaign_id(self.layout, study, index), phase="finals-complete")

    def order(self):
        return [(s, i) for i in range(self.layout.roots) for s in ("T", "W", "R")]

    def stopped(self, s, i):
        return (f_root(self.layout, s, i) / "ROOT_STOPPED.json").exists()

    def done(self, job):
        kind, s, i, *arm = job
        root = f_root(self.layout, s, i)
        if self.stopped(s, i):
            return True
        return {"sol": lambda: (root / "SOL_EXIT.json").exists(),
                "validate": lambda: (root / "validation" / f"{arm[0]}.json").exists(),
                "seal": lambda: (root / "SELECTION.json").exists(),
                "finals": lambda: (root / "final/COMPLETE.json").exists()}[kind]()

    def ready(self, job):
        kind, s, i, *arm = job
        root = f_root(self.layout, s, i)
        if kind == "sol" or (kind == "validate" and arm[0] != "sol"):
            return True
        if kind == "validate":
            return (root / "SOL_EXIT.json").exists()
        if kind == "seal":
            return all((root / "validation" / f"{a}.json").exists() for a in VALIDATION_ARMS)
        return (root / "SELECTION.json").exists()

    def slots(self, job):
        return "sol" if job[0] == "sol" else "eval"

    def run(self):
        verify_freeze(self.layout)
        jobs = []
        for s, i in self.order():
            jobs += [("sol", s, i), ("validate", s, i, "sol"), ("seal", s, i), ("finals", s, i)]
        # E's archives need no Sol: their validation is queued behind Sol's own so that a
        # finished Sol root is never starved, and fills idle evaluation slots meanwhile.
        for s, i in self.order():
            jobs += [("validate", s, i, "gp"), ("validate", s, i, "deepseek")]
        pending = [j for j in jobs if not self.done(j)]
        running = {"sol": set(), "eval": set()}
        limits = {"sol": SOL_SLOTS, "eval": VALIDATION_SLOTS}
        self.log(phase="launch", pending=len(pending), sol_slots=SOL_SLOTS,
                 eval_slots=VALIDATION_SLOTS)

        def work(job):
            try:
                getattr(self, job[0])(*job[1:])
            except Exception as exc:  # integrity or unexpected failure: stop new work
                self.log(job=list(job), phase="failed", error=f"{type(exc).__name__}: {exc}")
                self.stop.set()
            finally:
                with self.lock:
                    running[self.slots(job)].discard(job)

        threads = []
        while (pending or any(running.values())) and not self.stop.is_set():
            with self.lock:
                startable = []
                for j in pending:
                    if self.stopped(*j[1:3]):
                        continue
                    pool = self.slots(j)
                    if len(running[pool]) < limits[pool] and self.ready(j):
                        startable.append(j)
                        running[pool].add(j)
                pending = [j for j in pending if j not in startable and not self.stopped(*j[1:3])]
            for job in startable:
                thread = threading.Thread(target=work, args=(job,), daemon=False)
                thread.start()
                threads.append(thread)
            time.sleep(5)
        for thread in threads:
            thread.join()
        self.log(phase="stopped" if self.stop.is_set() else "all-complete")


# ---------------------------------------------------------------- freeze and status


def freeze_path(layout):
    return layout.out / "FREEZE.json"


def source_closure(smoke=False):
    paths = dict(E.source_closure())
    for p in (*HERE.glob("*.py"), *REG.glob("*")):
        if smoke and p.name in ("analyze.py", "test_analyze.py", "PROTOCOL.md"):
            continue  # the smoke exercises execution only; these are frozen with the study
        if p.is_file():
            paths[str(p.relative_to(ROOT))] = str(p)
    for rel in ("evolution/codex_engine.py", "evolution/llm_operator.py",
                "evolution/campaign.py", "analysis/tracking/seeds.py"):
        paths[rel] = str(ROOT / rel)
    return dict(sorted(paths.items()))


def freeze(layout, authorization):
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    e_frozen = E.verify_p2(Layout(out=HERE).e)  # E's registered sources/environment unchanged
    locations = source_closure(layout.smoke)
    payload = {
        "schema": "study-f-freeze-v1", "frozen_at_utc": E.now(), "smoke": layout.smoke,
        "authorization": authorization,
        "e_registration_freeze_sha256": E.sha(E.V3 / "FREEZE.json"),
        "e_analysis_primary_sha256": E.sha(E_DIR / "analysis/PRIMARY.json"),
        "source_locations": locations,
        "source_files_sha256": {n: E.sha(p) for n, p in locations.items()},
        "environment_digest": digest(evaluation_snapshot()),
        "e_environment_digest": e_frozen.get("environment_digest"),
        "engine": {"model_id": MODEL_ID, "cli_series": "0.159.x",
                   "reasoning_effort": REASONING_EFFORT},
        "design": {"top_k": SMOKE_TOP_K if layout.smoke else TOP_K,
                   "validation_seeds_per_scenario": (SMOKE_VALIDATION_SEEDS if layout.smoke
                                                     else VALIDATION_SEEDS),
                   "sol_slots": SOL_SLOTS, "validation_slots": VALIDATION_SLOTS},
    }
    layout.out.mkdir(parents=True, exist_ok=True)
    E.save_once(freeze_path(layout), {"payload": payload, "payload_sha256": digest(payload)})
    verify_freeze(layout)
    print(json.dumps({"freeze": str(freeze_path(layout)), "sources": len(locations)}))


def amendment_paths(layout):
    return sorted(layout.out.glob("AMENDMENT_*.json"))


def amend(layout, text):
    """Seal an amendment chained to the freeze (or the previous amendment) with new sources."""
    from experiments.gp_tuning_20260902.recovery import digest
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    existing = amendment_paths(layout)
    previous = existing[-1] if existing else freeze_path(layout)
    locations = source_closure(layout.smoke)
    payload = {"schema": "study-f-amendment-v1", "number": len(existing) + 1,
               "sealed_at_utc": E.now(), "previous_sha256": E.sha(previous), "text": text,
               "source_locations": locations,
               "source_files_sha256": {n: E.sha(p) for n, p in locations.items()},
               "environment_digest": digest(evaluation_snapshot())}
    path = layout.out / f"AMENDMENT_{payload['number']:02d}.json"
    E.save_once(path, {"payload": payload, "payload_sha256": digest(payload)})
    verify_freeze(layout)
    print(json.dumps({"amendment": str(path), "sources": len(locations)}))


def verify_freeze(layout):
    from experiments.gp_tuning_20260902.recovery import digest, read_sealed
    from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot

    frozen = read_sealed(freeze_path(layout))
    current, previous = frozen, E.sha(freeze_path(layout))
    for path in amendment_paths(layout):
        amendment = read_sealed(path)
        if amendment["previous_sha256"] != previous:
            raise ValueError(f"amendment chain broken at {path.name}")
        current, previous = amendment, E.sha(path)
    for name, location in current["source_locations"].items():
        if E.sha(location) != current["source_files_sha256"][name]:
            raise ValueError(f"source changed since freeze: {name}")
    if digest(evaluation_snapshot()) != frozen["environment_digest"]:
        raise ValueError("execution environment changed since freeze")
    if frozen["smoke"] != layout.smoke:
        raise ValueError("freeze belongs to a different layout")
    return frozen


def check_seeds():
    """Validation seeds must be disjoint from every registered seed of every root."""
    layout = Layout(out=HERE)
    used = set()
    for study, rows in json.loads((E.V3 / "SEEDS.json").read_text())["studies"].items():
        for row in rows:
            used |= {row["root_seed"], row.get("zeroshot_seed")}
            for grids in row["grids"].values():
                for seeds in grids.values():
                    used |= set(seeds)
            for seeds in (row.get("transfer_final") or {}).values():
                used |= set(seeds)
    generated = []
    for s, i in [(s, i) for i in range(30) for s in ("T", "W", "R")]:
        for scenario in validation_task(layout, s, i).scenarios:
            generated += list(scenario.seeds)
    new = set(generated)
    clash = used & new
    print(json.dumps({"validation_seeds": len(new), "registered_seeds": len(used),
                      "overlap": len(clash)}))
    if clash or len(new) != len(generated):
        raise SystemExit("validation seeds are not disjoint or not distinct")


def status(layout):
    state = layout.out / "STATE.jsonl"
    rows = [json.loads(line) for line in state.open()] if state.exists() else []
    print(json.dumps({
        "sol_exits": sum(1 for _ in layout.runs.glob("*/SOL_EXIT.json")),
        "validated": {a: sum(1 for _ in layout.runs.glob(f"*/validation/{a}.json"))
                      for a in VALIDATION_ARMS},
        "selections_sealed": sum(1 for _ in layout.runs.glob("*/SELECTION.json")),
        "finals_complete": sum(1 for _ in layout.runs.glob("*/final/COMPLETE.json")),
        "roots_stopped": sorted(p.parent.name for p in layout.runs.glob("*/ROOT_STOPPED.json")),
        "quota_events": [r for r in rows if r.get("phase", "").startswith("quota")],
        "failures": [r for r in rows if r.get("phase") == "failed"],
        "last_event": rows[-1] if rows else None,
    }, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action")
    parser.add_argument("rest", nargs="*")
    args = parser.parse_args()
    layout = current_layout()
    if args.action == "stage":
        stage(layout, args.rest[0], int(args.rest[1]))
    elif args.action == "evaluate-child":
        evaluate_child(layout, args.rest[0], int(args.rest[1]), *args.rest[2:5])
    elif args.action == "run":
        Launcher(layout).run()
    elif args.action == "status":
        status(layout)
    elif args.action == "freeze":
        freeze(layout, " ".join(args.rest))
    elif args.action == "amend":
        amend(layout, " ".join(args.rest))
    elif args.action == "check-seeds":
        check_seeds()
    else:
        raise SystemExit(f"unknown action {args.action}")


if __name__ == "__main__":
    main()
