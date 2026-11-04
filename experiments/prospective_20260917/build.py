"""Materialize and seal prospective choices without simulations or API calls."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import random
import tarfile
from pathlib import Path

import placementscript as ps
import shinka
from shinka.database import Program

from analysis.tracking import derive_seed
from baselines.random_ast import RandomASTConfig, RandomASTFloor
from evolution.campaign import load_campaign_plan
from evolution.evaluation import load_task_evaluation_config
from evolution.gp_operator import GPOperatorConfig
from evolution.islands import load_islands_config
from evolution.ladder import load_ladder_config
from evolution.openrouter_engine import load_openrouter_engine_config
from experiments.allbaselines_harness.initialization import (
    _evolve_shell,
    _reference_body,
    _reference_sources,
    read_seed_archive,
)
from experiments.allbaselines_harness.support import BOUNDARY
from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot
from experiments.matched_control_20260909.real_pilot import historical_seed_inventory
from experiments.prospective_20260917.statistics import power
from experiments.validation_20260909 import independent_scenarios as independent
from experiments.validation_20260909.run_independent_calibration import store_once

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = ROOT / "prereg/CORRECTED_STUDIES_20260917_v1"
CALIBRATION = ROOT / "experiments/validation_20260909/independent_calibration_20260909_v1"
TASK = ROOT / "evolution/task/config_dev_continuum_tight_calibrated.json"
INITIAL = ROOT / "evolution/task/initial_tight.py"
STRONG = ROOT / "experiments/baseline_check_20260905/policies/cheapest_tier_bottleneck_headroom.py"
NAMES = [
    "latency_greedy",
    "energy_greedy",
    "first_fit",
    "least_allocated",
    "most_allocated",
    "network_overhead",
    "graph_headroom",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def historical_seeds() -> set[int]:
    seeds = set(historical_seed_inventory()["seeds"]) | set(range(800000801, 800000821))
    for filename in (
        "workload_sensitivity_20260909/PLAN.json",
        "analytical_tails_20260909/PLAN.json",
    ):
        plan = read_sealed(independent.HERE / filename)
        if "seeds" in plan:
            seeds.update(plan["seeds"])
            seeds.add(plan["pilot_seed"])
        else:
            seeds.update(j["seed"] for j in plan["jobs"])
    design = json.loads((independent.OUT / "DESIGN.json").read_text())
    for values in design["seed_design"]["roles"].values():
        seeds.update(values)
    seeds.update(
        json.loads((ROOT / "sim/fixtures/MANIFEST.json").read_text())["acceptance_seed_matrix"]
    )
    seeds.update((3246458936443790985, 9113512321037399042))
    return seeds


def seed_plan(ids: dict[str, list[str]], excluded: set[int]) -> dict:
    used = set(excluded)
    records = []

    def keep(value, role):
        if value in used:
            raise ValueError(f"seed collision in {role}; no automatic redraw")
        used.add(value)
        records.append({"seed": value, "role": role})
        return value

    studies = {}
    for study in ("A", "B"):
        roots = []
        for i in range(30):
            root = keep(
                derive_seed(20260917, "corrected-v1", study, "root", i), f"{study}/root/{i}"
            )
            grids = {}
            for sid in ids[study]:
                search = keep(
                    derive_seed(root, "sim", "scenario", sid, "replicate", 0),
                    f"{study}/search/{i}/{sid}",
                )
                final = [
                    keep(
                        derive_seed(root, "corrected-v1", "final", sid, j),
                        f"{study}/final/{i}/{sid}/{j}",
                    )
                    for j in range(8)
                ]
                grids[sid] = {"search": [search], "final": final}
            roots.append(
                {
                    "campaign_id": f"{study}{i:02d}",
                    "root_seed": root,
                    "evaluation_replicates": 1,
                    "grids": grids,
                    "first_arm": "llm" if i % 2 == 0 else "gp",
                }
            )
        studies[study] = roots
    analysis = {
        name: keep(derive_seed(20260917, "corrected-v1", "analysis", name), f"analysis/{name}")
        for name in ("A", "B", "descriptive")
    }
    pilots = [
        keep(derive_seed(20260917, "corrected-v1", "resource-pilot", i), f"pilot/{i}")
        for i in range(3)
    ]
    return {
        "studies": studies,
        "analysis": analysis,
        "resource_pilot_roots": pilots,
        "enumerated_seed_records": records,
        "excluded_named_inventory": sorted(excluded),
        "scope": (
            "Named historical evaluation, calibration, sensitivity, tail and "
            "acceptance-metadata grids; not every historic RNG draw."
        ),
    }


def archive(root: int, study: str) -> dict:
    records = _reference_sources(NAMES, INITIAL)
    prefix, suffix = _evolve_shell(INITIAL.read_text())
    if study == "B":
        source = prefix + _reference_body(STRONG.read_text()) + suffix
        ps.Policy(source)
        records.append(
            {
                "kind": "handwritten_reference",
                "name": "bottleneck_headroom",
                "source_path": str(STRONG.relative_to(ROOT)),
                "source_path_sha256": sha(STRONG),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "source": source,
                "identity_note": "Posthoc reference included prospectively in both B arms.",
            }
        )
    reference_count = len(records)
    seen = {r["source_sha256"] for r in records}
    floor = RandomASTFloor(RandomASTConfig())
    for index in range(12 - reference_count):
        attempts = []
        for attempt in range(64):
            seed = derive_seed(
                root, "allbaselines", "seed-archive", "random-ast-mutant", index, attempt
            )
            rng = random.Random(seed)
            parent = rng.choice(records[:reference_count])
            offspring = floor.operator(
                Program(
                    id=f"seed-archive-parent-{index}-{attempt}",
                    code=parent["source"],
                    language="placementscript",
                    correct=True,
                ),
                (),
                rng,
            )
            attempts.append(
                {
                    "seed": seed,
                    "attempt": attempt,
                    "valid": offspring.valid,
                    "parent_source_sha256": parent["source_sha256"],
                }
            )
            if not offspring.valid:
                continue
            source_sha = hashlib.sha256(offspring.source.encode()).hexdigest()
            if source_sha in seen:
                continue
            ps.Policy(offspring.source)
            if _evolve_shell(offspring.source) != (prefix, suffix):
                raise ValueError("initial mutant changed immutable shell")
            records.append(
                {
                    "kind": "seeded_random_ast_mutant",
                    "name": f"random_ast_mutant_{index}",
                    "seed": seed,
                    "source": offspring.source,
                    "source_sha256": source_sha,
                    "proposal_metadata": offspring.metadata,
                    "generation_attempts": attempts,
                }
            )
            seen.add(source_sha)
            break
        else:
            raise ValueError("archive construction exhausted attempts")
    return {
        "schema": "evolved-continuum-seed-archive-v1",
        "status": "development-only",
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "root_seed": root,
        "random_ast_mutants": 12 - reference_count,
        "population_size": 12,
        "gphh_generation0_population_size": 12,
        "entries": records,
    }


def prepare(out: Path) -> None:
    if out.exists():
        raise ValueError("new design directory required; never overwrite a prepared/frozen version")
    admission = read_sealed(CALIBRATION / "OPERATING_LEVEL_RECEIPT.json")
    calibrated = read_sealed(CALIBRATION / "SUMMARY.json")
    if (
        admission["summary_sha256"] != digest(calibrated)
        or admission["admission"] != calibrated["admission"]
    ):
        raise ValueError("calibration admission binding differs")
    design = json.loads((independent.OUT / "DESIGN.json").read_text())
    selected = [
        c
        for c in design["cells"]
        if c["nominal_average_edge_cpu_pressure"]
        == admission["admission"]["topologies"][c["topology"]]["selected_pressure"]
    ]
    if len(selected) != 4:
        raise ValueError("expected exactly four admitted conditions")
    tasks = {"A": json.loads(TASK.read_text())}
    tasks["B"] = json.loads(TASK.read_text())
    tasks["B"].update(
        task_variant="independent-four-cell-joint-search",
        task_variant_note=(
            "Synthetic independent design; prospective execution prerequisites remain pending."
        ),
        scenarios=[
            {
                "id": c["id"],
                "path": str((independent.OUT / (c["id"] + ".json")).relative_to(ROOT)),
                "seeds": [1],
            }
            for c in selected
        ],
        constraints=independent.CONSTRAINTS,
    )
    tasks["B"]["boundary_bounds"]["network_bytes_bound"] = 20 * 1024**3
    tasks["B"]["ladder"]["l1"].update(scenario_ids=[c["id"] for c in selected], horizon_s=900)
    seeds = seed_plan(
        {k: [s["id"] for s in t["scenarios"]] for k, t in tasks.items()}, historical_seeds()
    )
    out.mkdir(parents=True)
    (out / "PROTOCOL.md").write_bytes((HERE / "PROTOCOL.md").read_bytes())
    for study, task in tasks.items():
        # Task seeds are replaced with campaign-derived seeds at execution.
        for s in task["scenarios"]:
            s["seeds"] = seeds["studies"][study][0]["grids"][s["id"]]["search"]
        write(out / f"task_{study}.json", task)
        for load in (load_task_evaluation_config, load_ladder_config, load_islands_config):
            load(out / f"task_{study}.json")
        config = json.loads((ROOT / "evolution/campaigns.json").read_text())
        config.update(
            candidate_budget=562,
            evolution_profile="deep-search-feedback",
            feedback_profile="e5-text-feedback-on",
            shared_numeric_probability=0.0,
            proposals_per_call=1,
            campaigns=[
                {k: r[k] for k in ("campaign_id", "root_seed", "evaluation_replicates")}
                for r in seeds["studies"][study]
            ],
        )
        config["evaluator_work"].update(
            max_microwork=56000000,
            checkpoints_microwork=[0, 14000000, 28000000, 42000000, 56000000],
            registration_source=(
                "Prospective design freeze; G0 and scientific execution gate outstanding."
            ),
        )
        write(out / f"campaign_{study}.json", config)
        load_campaign_plan(out / f"campaign_{study}.json")
        for row in seeds["studies"][study]:
            path = out / "initialization" / (row["campaign_id"] + ".json")
            payload = archive(row["root_seed"], study)
            store_once(path, payload)
            read_seed_archive(path, expected_size=12, initial_path=INITIAL)
    write(out / "SEEDS.json", seeds)
    write(out / "gp.json", dataclasses.asdict(GPOperatorConfig()))
    engine = json.loads((ROOT / "evolution/openrouter_engine_glm53.json").read_text())
    engine.update(max_attempts=1, reasoning_effort="high")
    engine["generation"]["provider"] = {
        "only": ["z-ai"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 1.4, "completion": 4.4},
    }
    write(out / "engine.json", engine)
    load_openrouter_engine_config(out / "engine.json")
    write(
        out / "MODEL_AND_RESOURCES.json",
        {
            "budget_authorization": "pending-researcher-response",
            "run_authorized": False,
            "model_id": engine["model_id"],
            "per_llm_run_usd": 2.5,
            "llm_study_runs": 60,
            "pilot_runs": 3,
            "pilot_total_usd": 7.5,
            "allocated_max_usd": 157.5,
            "proposed_total_envelope_usd": 225,
            "unallocated_reserve_fraction": 0.30,
            "automatic_reserve_spend": False,
            "capacity_certified": False,
            "final_scoring_label_episode_ceiling": {"A": 30 * 12 * 8 * 3, "B": 30 * 12 * 4 * 8 * 3},
            "labels_per_root": (
                "LLM champion, GP champion, common initialization best, nine fixed references; "
                "deduplicate exact source/grid requests only"
            ),
            "simulation_child_rss_mib": 2048,
            "single_episode_child_timeout_seconds": 1200,
            "campaign_process_family_rss_mib_proposed": 3072,
            "cpu_storage_wall_researcher_time_ceilings": (
                "require measured capacity and researcher resource allocation "
                "before execution freeze"
            ),
            "model_pilot_requirements": [
                "provider slug and parameter support",
                "prompt/token reservation coverage",
                "complete successful generation",
                "cost and latency",
                "failure/uncertain-dispatch handling",
            ],
        },
    )
    gates = [
        ("calibration", "complete", "OPERATING_LEVEL_RECEIPT.json binds 972 verified episodes"),
        (
            "initialization-transport-engineering",
            "historical-witness-only",
            "real initialization and local HTTPS pilots; not current scientific acceptance",
        ),
        (
            "G0",
            "pending",
            (
                "live proposer/rung timings, failure inflation, memory, explicit "
                "per-ceiling capacity and monetary authorization"
            ),
        ),
        (
            "P1-DES",
            "pending",
            "clean source, exact criteria/fixtures, one-shot registration and acceptance",
        ),
        (
            "P1-G2",
            "pending",
            "real scheduler trace conformance or explicit permitted claim narrowing",
        ),
        (
            "P1-G3-cal",
            "pending",
            (
                "external face-validity evidence or explicit permitted workload "
                "claim narrowing; tail precision unresolved"
            ),
        ),
        (
            "P1-G-ladder",
            "pending",
            (
                "prospective A/B rung fidelity/reject audits or justified common "
                "alternative before study"
            ),
        ),
        (
            "P1-G-ablate-commercial",
            "pending",
            "new commercial/replay criteria version and actual live-model witness",
        ),
        (
            "scientific-launcher",
            "pending",
            (
                "commercial deep-search profile presently development-only; "
                "implement/verify new registered path without bypassing old "
                "guards"
            ),
        ),
        (
            "P2",
            "pending",
            (
                "complete gate receipts, clean release/tag, capacity, public "
                "timestamp requirement, authorized execution"
            ),
        ),
    ]
    write(
        out / "GATES.json",
        {
            "launch_allowed": False,
            "gates": [{"id": g, "status": s, "requirement": r} for g, s, r in gates],
        },
    )
    write(
        out / "POWER.json",
        [
            {
                "n": n,
                "win_probability": w,
                "tie_probability": t,
                "alpha": a,
                "power": power(n, w, t, a),
            }
            for n in (20, 30, 40)
            for w, t in ((0.7, 0), (0.8, 0), (0.8, 0.2), (0.8, 0.4))
            for a in (0.05, 0.025)
        ],
    )
    print(
        "Prepared 60 paired roots, 720 initialization programs, two tasks "
        "and explicit launch blockers."
    )


def source_paths() -> dict[str, Path]:
    snap = evaluation_snapshot()
    paths = {name: ROOT / name for name in snap["files"]}
    for p in independent.HERE.glob("*.py"):
        paths[str(p.relative_to(ROOT))] = p
    for p in independent.OUT.glob("*.json"):
        paths[str(p.relative_to(ROOT))] = p
    for folder in (
        HERE,
        ROOT / "tests",
        ROOT / "experiments/matched_control_20260909",
        ROOT / "experiments/matched_transport_20260909",
        ROOT / "experiments/allbaselines_harness",
        ROOT / "runtime/crates",
    ):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".rs", ".toml"):
                paths[str(p.relative_to(ROOT))] = p
    for name in (
        "pyproject.toml",
        "uv.lock",
        "runtime/Cargo.toml",
        "runtime/Cargo.lock",
        "ContinuumPlacementScopedDown.md",
        "docs/CLAIM_EVIDENCE_LEDGER.md",
        "sim/fixtures/MANIFEST.json",
    ):
        paths[name] = ROOT / name
    for p in ROOT.joinpath("evolution/shinkaevolve-patches").glob("*.patch"):
        paths[str(p.relative_to(ROOT))] = p
    for p in (
        CALIBRATION / "OPERATING_LEVEL_RECEIPT.json",
        CALIBRATION / "SUMMARY.json",
        STRONG,
        independent.OUT / "DESIGN.json",
    ):
        paths[str(p.relative_to(ROOT))] = p
    for i, name in enumerate(snap["native"]):
        paths[f"native/{i}-{Path(name).name}"] = Path(name)
    package = Path(shinka.__file__).parent
    for p in package.rglob("*.py"):
        paths["dependencies/shinka/" + str(p.relative_to(package))] = p
    return paths


def freeze(out: Path) -> None:
    if (out / "FREEZE.json").exists():
        raise ValueError("already frozen")
    checks = json.loads((out / "CHECKS.json").read_text())
    if checks["failures"] or checks["errors"] or not checks["tests"]:
        raise ValueError("passing checks required")
    paths = source_paths()
    hashes = {n: sha(p) for n, p in paths.items()}
    with tarfile.open(out / "sources.tar.gz", "x:gz") as tar:
        for name, p in sorted(paths.items()):
            tar.add(p, arcname=name, recursive=False)
    if hashes != {n: sha(p) for n, p in paths.items()}:
        raise ValueError("source drift during archive creation")
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    files.update({str(p.relative_to(out)): sha(p) for p in (out / "initialization").glob("*.json")})
    store_once(
        out / "FREEZE.json",
        {
            "schema": "corrected-studies-design-freeze-v1",
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "design-frozen-execution-blocked",
            "launch_allowed": False,
            "files_sha256": files,
            "source_files_sha256": hashes,
            "source_locations": {n: str(p) for n, p in paths.items()},
            "environment": evaluation_snapshot(),
            "independent_public_timestamp": False,
            "original_P1_P2_satisfied": False,
        },
    )
    verify(out)


def verify(out: Path) -> None:
    frozen = read_sealed(out / "FREEZE.json")
    actual = {
        str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "FREEZE.json"
    }
    if actual != set(frozen["files_sha256"]):
        raise ValueError("extra or missing frozen input")
    for name, h in frozen["files_sha256"].items():
        if sha(out / name) != h:
            raise ValueError(f"frozen input changed: {name}")
    for name, h in frozen["source_files_sha256"].items():
        if sha(Path(frozen["source_locations"][name])) != h:
            raise ValueError(f"source changed: {name}")
    with tarfile.open(out / "sources.tar.gz") as tar:
        members = tar.getmembers()
        if len(members) != len(frozen["source_files_sha256"]):
            raise ValueError("archive member count differs")
        if {m.name for m in members} != set(frozen["source_files_sha256"]):
            raise ValueError("archive membership differs")
        for m in members:
            if (
                hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                != frozen["source_files_sha256"][m.name]
            ):
                raise ValueError("source archive member changed")
    if digest(evaluation_snapshot()) != digest(frozen["environment"]):
        raise ValueError("execution environment changed")
    print(
        f"Verified design freeze: {len(frozen['files_sha256'])} inputs, "
        f"{len(frozen['source_files_sha256'])} source/native members. Execution blocked."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("action", choices=("prepare", "freeze", "verify", "check-launch"))
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.action == "freeze":
        freeze(args.out)
    else:
        verify(args.out)
        if args.action == "check-launch":
            raise SystemExit(
                "BLOCKED: design freeze is not scientific execution authorization; see GATES.json"
            )
