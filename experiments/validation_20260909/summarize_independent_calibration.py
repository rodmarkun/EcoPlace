"""Close every calibration request and apply the already declared admission rule.

No partial report or adaptive dispatch. Compressed offered cohorts are recomputed;
arrival streams are regenerated without running a policy or the event simulator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from analysis.tracking import spawn_rng
from evolution.evaluation import config_for_development_bundle, policy_identity_for_source
from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.validation_20260909 import run_independent_calibration as runner
from sim.arrivals import iter_arrival_times
from sim.energy import build_development_bundles, load_pinned_sources
from sim.energy.boundary import (
    BoundaryBounds,
    BoundaryWeights,
    VerifiedInstanceEnergy,
    certify_boundary,
)
from sim.energy.fitness import ConstraintSpec, InstanceOutcome, evaluate_candidate
from sim.spec import config_sha256, load_episode_config

STATUSES = ("completed", "timeout", "dropped_no_replica", "dropped_queue_full", "in_flight")


def line(value) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def expected_arrivals(config, seed: int) -> dict:
    expected = {}
    for template in config.workload.templates:
        rng = spawn_rng(seed, "sim", "arrivals", template.id)
        count, h = 0, hashlib.sha256()
        for i, at in enumerate(iter_arrival_times(template.arrival, config.cutoff_s, rng)):
            h.update(line([i, at]))
            count += 1
        expected[template.name] = {"offered": count, "arrival_sha256": h.hexdigest()}
    return expected


def cohort(path: Path, metadata: dict, config) -> dict:
    if metadata["columns"] != ["rid", "template", "arrival_s", "end_s", "status"]:
        raise ValueError("offered trace schema differs")
    raw = hashlib.sha256()
    templates = config.workload.templates
    totals = [Counter() for _ in templates]
    on_time, arrival_hashes, previous = (
        [0] * len(templates),
        [hashlib.sha256() for _ in templates],
        -math.inf,
    )
    count = 0
    with gzip.open(path, "rb") as stream:
        for i, encoded in enumerate(stream):
            raw.update(encoded)
            row = json.loads(encoded)
            if not isinstance(row, list) or len(row) != 5 or encoded != line(row):
                raise ValueError("malformed or noncanonical offered row")
            rid, tid, arrival, end, status = row
            if (
                type(rid) is not int
                or rid != i
                or type(tid) is not int
                or not 0 <= tid < len(templates)
            ):
                raise ValueError("request IDs or template identity differ")
            if (
                isinstance(arrival, bool)
                or not isinstance(arrival, (int, float))
                or not math.isfinite(arrival)
                or arrival < previous
                or not templates[tid].arrival.start_s <= arrival < config.cutoff_s
            ):
                raise ValueError("offered arrival outside its ordered window")
            previous = arrival
            if status not in STATUSES:
                raise ValueError("unknown request terminal status")
            if status == "in_flight":
                if end is not None:
                    raise ValueError("in-flight request has a terminal time")
            elif (
                isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not math.isfinite(end)
                or not arrival <= end <= config.horizon_s
            ):
                raise ValueError("terminal request time is invalid")
            index = sum(totals[tid].values())
            arrival_hashes[tid].update(line([index, arrival]))
            totals[tid][status] += 1
            on_time[tid] += status == "completed" and end - arrival <= templates[tid].slo_latency_s
            count += 1
    if (
        type(metadata["offered_records"]) is not int
        or count != metadata["offered_records"]
        or raw.hexdigest() != metadata["uncompressed_sha256"]
    ):
        raise ValueError("offered trace count or uncompressed digest differs")
    per = {}
    all_status = sum(totals, Counter())
    for template, statuses, ok, h in zip(templates, totals, on_time, arrival_hashes, strict=True):
        n = sum(statuses.values())
        per[template.name] = {
            "offered": n,
            "on_time_completed": ok,
            "statuses": dict(statuses),
            "slo_achieved": ok / n if n else 1.0,
            "arrival_sha256": h.hexdigest(),
        }
    return {
        "offered": count,
        "per_template": per,
        "statuses": dict(all_status),
        "goodput": all_status["completed"] / count if count else 1.0,
        "drop_rate": sum(v for k, v in all_status.items() if k.startswith("dropped"))
        / max(1, count),
        "timeout_rate": all_status["timeout"] / max(1, count),
    }


def close_number(actual, expected, label: str) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
        or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise ValueError(f"{label} differs from recomputation")


def verify_measured(req: dict, result: dict, config, design: dict, observed: dict) -> dict:
    """Reconstruct certificate/token, energy arithmetic and fitness without simulation."""
    runner.validate_result(req, result)
    job = req["job"]
    instance = result["evaluation"]["instances"][0]
    episode = instance["episode"]
    if episode["config_sha256"] != config_sha256(config) or episode["seed"] != job["seed"]:
        raise ValueError("retained episode identity differs")
    close_number(episode["horizon_s"], config.horizon_s, "horizon")
    if set(instance["slo_achieved"]) != set(observed["per_template"]):
        raise ValueError("SLO template coverage differs")
    for name, values in observed["per_template"].items():
        # Exact equality preserves the published threshold rule, without a tolerance waiver.
        if instance["slo_achieved"][name] != values["slo_achieved"]:
            raise ValueError("SLO attainment differs from offered trace")
    for field in ("goodput", "drop_rate", "timeout_rate"):
        if instance[field] != observed[field]:
            raise ValueError(f"{field} differs from offered trace")
    if (
        episode["records"].get("length") != observed["offered"]
        or episode["ledger"]["requests_injected"] != observed["offered"]
    ):
        raise ValueError("offered count differs from compact ledger")
    for status in STATUSES[:-1]:
        if episode["ledger"]["requests_" + status] != observed["statuses"].get(status, 0):
            raise ValueError("terminal status count differs from ledger")
    weights, bounds = (
        BoundaryWeights(**design["boundary_weights"]),
        BoundaryBounds(**design["boundary_bounds"]),
    )
    cert = certify_boundary(
        config.cluster,
        config.energy,
        config.horizon_s,
        weights,
        bounds,
        scenario_id=job["cell_id"],
        seed=job["seed"],
        bundle_id=job["bundle_id"],
    )
    if (
        instance["boundary_certificate"] != {**cert.payload(), "seal": cert.seal}
        or instance["certificate_seal"] != cert.seal
    ):
        raise ValueError("energy boundary certificate differs from effective config")
    token = VerifiedInstanceEnergy(**instance["verified_energy"])
    token.verify_seal()
    for name in (
        "scenario_id",
        "seed",
        "bundle_id",
        "cluster_sha256",
        "coefficients_sha256",
        "horizon_s",
        "idle_floor_j",
        "ceiling_j",
    ):
        if getattr(token, name) != getattr(cert, name):
            raise ValueError("verified energy token boundary identity differs")
    if token.certificate_seal != cert.seal or len(token.transition_counts) != cert.n_nodes:
        raise ValueError("token certificate or transition coverage differs")
    if (
        any(
            type(v) is not int or not 0 <= v <= bounds.max_transitions_per_node
            for v in token.transition_counts
        )
        or not 0 <= token.network_bytes <= bounds.network_bytes_bound
        or not 0 <= token.energy_j <= token.ceiling_j
    ):
        raise ValueError("token exceeds finite energy/transition/network bounds")
    energy = episode["energy"]
    for name in ("device_j", "cloud_j", "network_j", "cloud_vcpu_hours", "network_bytes"):
        value = energy[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("energy accounting bases must be finite and nonnegative")
    for name in ("cluster_sha256", "coefficients_sha256"):
        if energy[name] != getattr(cert, name):
            raise ValueError("meter identity differs")
    if energy["transition_counts"] != list(token.transition_counts):
        raise ValueError("meter transitions differ")
    close_number(energy["network_bytes"], token.network_bytes, "network bytes")
    for field in ("device_j_per_node", "transition_j_per_node"):
        if len(energy[field]) != cert.n_nodes or any(
            not math.isfinite(v) or v < 0 for v in energy[field]
        ):
            raise ValueError("invalid per-node energy vector")
    if any(
        a > b
        for a, b in zip(energy["transition_j_per_node"], energy["device_j_per_node"], strict=True)
    ):
        raise ValueError("transition energy exceeds node energy")
    close_number(energy["device_j"], math.fsum(energy["device_j_per_node"]), "device total")
    close_number(
        energy["cloud_j"],
        energy["cloud_vcpu_hours"] * config.energy.cloud_j_per_vcpu_hour,
        "cloud allocation",
    )
    close_number(
        energy["network_j"],
        energy["network_bytes"] * config.energy.network_j_per_byte,
        "network allocation",
    )
    close_number(
        token.energy_j,
        energy["device_j"] * weights.device
        + energy["cloud_j"] * weights.cloud
        + energy["network_j"] * weights.network,
        "closed energy",
    )
    for field in ("energy_j", "idle_floor_j", "ceiling_j"):
        close_number(instance[field], getattr(token, field), field)
    for name in design["constraints"]["disruption_caps"]:
        if instance["disruption"][name] != episode["rebalancer"][name]:
            raise ValueError("disruption ledger differs")
    outcome = InstanceOutcome(
        verified=token,
        slo_achieved=instance["slo_achieved"],
        goodput=observed["goodput"],
        drop_rate=observed["drop_rate"],
        timeout_rate=observed["timeout_rate"],
        disruption=instance["disruption"],
    )
    fitness = evaluate_candidate(
        ConstraintSpec(**design["constraints"]),
        [outcome],
        development_bundle_ids=(job["bundle_id"],),
        instance_keys=((job["cell_id"], job["seed"]),),
        expected_certificate_seals={(job["cell_id"], job["seed"], job["bundle_id"]): cert.seal},
        proposal_seq=0,
    )
    saved = result["evaluation"]["fitness"]
    expected = {
        "feasible": fitness.feasible,
        "infeasibility_vector": list(fitness.infeasibility_vector),
        "summed_primitive_shortfall": fitness.summed_primitive_shortfall,
        "normalized_energy": fitness.normalized_energy,
        "proposal_seq": 0,
        "n_instances": 1,
        "n_bundles": 1,
        "sort_key": list(fitness.sort_key()),
        "direction": "lower-is-better",
    }
    if type(saved["feasible"]) is not bool or saved != expected:
        raise ValueError("fitness differs from reconstructed offered-cohort outcome")
    return {
        "status": "evaluated",
        "feasible": fitness.feasible,
        "normalized_energy": fitness.normalized_energy,
        "offered": observed["offered"],
        "cohort": observed,
        "slo_achieved": instance["slo_achieved"],
        "goodput": observed["goodput"],
        "drop_rate": observed["drop_rate"],
        "timeout_rate": observed["timeout_rate"],
        "disruption": instance["disruption"],
        "energy_j": token.energy_j,
        "episode_digest": instance["episode_digest"],
        "certificate_seal": cert.seal,
    }


def admission(design: dict, rows: list[dict]) -> dict:
    expected_jobs = runner.jobs(design, "calibration")
    indexed = {digest(row["job"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {digest(j) for j in expected_jobs}:
        raise ValueError("admission requires the complete unique 972-episode grid")
    for row in rows:
        if row["status"] not in ("evaluated", "policy-fault"):
            raise ValueError("technical failure is not a calibration outcome")
        if row["status"] == "evaluated" and type(row["feasible"]) is not bool:
            raise ValueError("measured feasibility must be boolean")
        if row["status"] == "policy-fault" and (
            row["feasible"] is not None or row["normalized_energy"] is not None
        ):
            raise ValueError("policy fault must not invent measured feasibility or energy")
    topologies = {}
    for topology in design["topologies"]:
        levels = {}
        for pressure in (0.15, 0.30, 0.45):
            cells = {
                c["id"]
                for c in design["cells"]
                if c["topology"] == topology and c["nominal_average_edge_cpu_pressure"] == pressure
            }
            qualifiers = []
            for policy in design["fixed_references"]:
                subset = [
                    r
                    for r in rows
                    if r["job"]["cell_id"] in cells and r["job"]["label"] == policy["label"]
                ]
                if len(subset) != 18:
                    raise ValueError("qualification block is not 18 episodes")
                if all(r["status"] == "evaluated" and r["feasible"] for r in subset):
                    qualifiers.append(policy["label"])
            levels[f"{pressure:.2f}"] = qualifiers
        chosen = next((p for p in (0.30, 0.15) if levels[f"{p:.2f}"]), None)
        topologies[topology] = {
            "selected_pressure": chosen,
            "qualifying_references_by_pressure": levels,
        }
    return {
        "topologies": topologies,
        "status": "qualified-for-prospective-study-design"
        if all(t["selected_pressure"] is not None for t in topologies.values())
        else "stop-design-amendment-required",
        "energy_used_for_selection": False,
        "stress_pressure": 0.45,
        "automatic_adaptive_launch": False,
    }


def verify_phase(out: Path, plan: dict, phase: str) -> dict:
    """Verify a complete declared phase; caller holds locks and verifies frozen sources."""
    if phase not in ("pilot", "calibration") or plan[phase + "_jobs"] != runner.jobs(
        plan["design"], phase
    ):
        raise ValueError("phase is not the complete prespecified grid")
    design = plan["design"]
    bundles = {b.bundle_id: b for b in build_development_bundles(load_pinned_sources())}
    configs, arrivals, identities = {}, {}, {}
    for policy in design["fixed_references"]:
        path = out / "policies" / (policy["source_sha256"] + ".ps")
        if runner.sha(path) != policy["source_sha256"]:
            raise ValueError("copied source changed")
        identities[policy["label"]] = policy_identity_for_source(path.read_text())
    files = {}
    rows, usages, expected_files = [], [], set()
    for job in plan[phase + "_jobs"]:
        req = runner.request(plan, job)
        key = digest(req)
        folder = out / phase
        paths = {
            suffix: folder / (key + suffix)
            for suffix in (
                ".request.json",
                ".started.json",
                ".result.json",
                ".usage.json",
                ".log",
            )
        }
        saved, start, result, usage = [
            read_sealed(paths[s])
            for s in (".request.json", ".started.json", ".result.json", ".usage.json")
        ]
        if (
            saved != req
            or start["request_sha256"] != key
            or usage["started_file_sha256"] != runner.sha(paths[".started.json"])
        ):
            raise ValueError("dispatch or start receipt identity differs")
        runner.validate_usage(req, result, usage, plan)
        runner.validate_trace_file(folder, key, result)
        usages.append(usage)
        if result["status"] == "policy-fault":
            row = {
                "job": job,
                "status": "policy-fault",
                "feasible": None,
                "normalized_energy": None,
                "failure": result["failure"],
            }
        else:
            if result["evaluation"]["policy"] != identities[job["label"]]:
                raise ValueError("policy identity record differs")
            config_key = (job["cell_id"], job["bundle_id"])
            if config_key not in configs:
                base = load_episode_config(runner.design_module.OUT / (job["cell_id"] + ".json"))
                configs[config_key] = config_for_development_bundle(base, bundles[job["bundle_id"]])
            config = configs[config_key]
            if config_sha256(config) != job["effective_config_sha256"]:
                raise ValueError("effective config changed")
            trace_path = folder / (key + ".records.jsonl.gz")
            paths[".records.jsonl.gz"] = trace_path
            observed = cohort(trace_path, result["offered_trace"], config)
            arrival_key = (job["cell_id"], job["seed"])
            if arrival_key not in arrivals:
                arrivals[arrival_key] = expected_arrivals(config, job["seed"])
            actual = {
                n: {k: v[k] for k in ("offered", "arrival_sha256")}
                for n, v in observed["per_template"].items()
            }
            if actual != arrivals[arrival_key]:
                raise ValueError("offered stream differs from declared seed/profile")
            row = {"job": job, **verify_measured(req, result, config, design, observed)}
        rows.append(row)
        for path in paths.values():
            expected_files.add(path.name)
            files[str(path.relative_to(out))] = runner.sha(path)
    if {p.name for p in (out / phase).iterdir()} != expected_files:
        raise ValueError("extra or missing phase artifact")
    return {"rows": rows, "usages": usages, "files_sha256": files}


def pilot_closure(plan: dict, phase: dict) -> dict:
    if len(phase["rows"]) != 3 or any(r["status"] != "evaluated" for r in phase["rows"]):
        raise ValueError("resource pilot requires three measured episodes, not policy faults")
    return {
        "schema": "independent-calibration-pilot-closure-v1",
        "plan_sha256": digest(plan),
        "requests": 3,
        "files_sha256": phase["files_sha256"],
        "offered_stream_accounting_fitness_verified": True,
        "quality_used_for_design": False,
    }


def verify(out: Path) -> dict:
    complete = read_sealed(out / "COMPLETE.json")  # refuse before reading any partial outcomes
    plan = runner.verify_source(out)
    if complete["plan_sha256"] != digest(plan) or complete["requests"] != 972:
        raise ValueError("calibration completion identity or count differs")
    runner.verify_archive(out, read_sealed(out / "snapshot.json"), plan["source_archive_sha256"])
    design = plan["design"]
    checked = {phase: verify_phase(out, plan, phase) for phase in ("pilot", "calibration")}
    phases = {p: data["rows"] for p, data in checked.items()}
    usages_by_phase = {p: data["usages"] for p, data in checked.items()}
    files = {name: sha for data in checked.values() for name, sha in data["files_sha256"].items()}
    if read_sealed(out / "PILOT_CLOSURE.json") != pilot_closure(plan, checked["pilot"]):
        raise ValueError("pilot semantic closure differs")
    review = runner.pilot_review(plan, usages_by_phase["pilot"])
    if read_sealed(out / "PILOT_REVIEW.json") != review or review["fits_cap"] is not True:
        raise ValueError("pilot reserve gate differs")
    usage = usages_by_phase["calibration"]
    expected_complete = {
        "plan_sha256": digest(plan),
        "requests": 972,
        "usage_sha256": [digest(u) for u in usage],
        "total_worker_seconds": sum(u["elapsed_seconds"] for u in usage),
        "peak_rss_mib": max(u["peak_rss_mib"] for u in usage),
        "no_model_calls": True,
    }
    if (
        complete != expected_complete
        or complete["total_worker_seconds"] > plan["calibration_total_seconds"]
    ):
        raise ValueError("completion usage total or membership differs")
    rows = phases["calibration"]
    groups = []
    for cell in design["cells"]:
        for policy in design["fixed_references"]:
            group = [
                r
                for r in rows
                if r["job"]["cell_id"] == cell["id"] and r["job"]["label"] == policy["label"]
            ]
            feasible = [r for r in group if r["status"] == "evaluated" and r["feasible"]]
            groups.append(
                {
                    "cell_id": cell["id"],
                    "reference": policy["label"],
                    "episodes": len(group),
                    "feasible": len(feasible),
                    "policy_faults": sum(r["status"] == "policy-fault" for r in group),
                    "mean_energy_among_feasible": mean(r["normalized_energy"] for r in feasible)
                    if feasible
                    else None,
                }
            )
    runner.verify_source(out)
    return {
        "schema": "independent-calibration-closure-v1",
        "plan_sha256": digest(plan),
        "complete_sha256": digest(complete),
        "analysis_source_sha256": runner.sha(Path(__file__)),
        "files_sha256": files,
        "pilot_rows": phases["pilot"],
        "rows": rows,
        "groups": groups,
        "admission": admission(design, rows),
        "resource_usage": complete,
        "scope": "Development calibration. Offered traces, seeded arrival streams, accounting "
        "and fitness are recomputed. Arrival/energy/fitness routines are shared model components, "
        "not independent physical validation. Full service-call/snapshot ledgers are omitted; "
        "their episode digest cannot be rederived here. Public content hashes and energy seals "
        "detect mistakes, not malicious forgery. No adaptive study is launched.",
    }


def render(report: dict) -> str:
    lines = [
        "# Independent synthetic-family calibration",
        "",
        report["scope"],
        "",
        "All 972 calibration episodes and three separate pilot episodes passed closure checks.",
        "",
        "## Prespecified operating-level decision",
        "",
        report["admission"]["status"],
        "",
        "| Topology | Selected pressure | Qualifying references at selected pressure |",
        "|---|---:|---|",
    ]
    for name, data in report["admission"]["topologies"].items():
        p = data["selected_pressure"]
        labels = (
            data["qualifying_references_by_pressure"].get(f"{p:.2f}", []) if p is not None else []
        )
        lines.append(
            f"| {name} | {p if p is not None else 'none'} | {', '.join(labels) or 'none'} |"
        )
    lines += [
        "",
        "A single reference must qualify across all 18 episodes. Preference is 0.30, then 0.15; "
        "0.45 remains stress. Energy does not select conditions.",
        "",
        "## Every calibration cell and reference",
        "",
        "| Cell | Reference | Feasible /9 | Policy faults /9 | "
        "Mean normalized energy among feasible episodes |",
        "|---|---|---:|---:|---:|",
    ]
    for g in report["groups"]:
        e = g["mean_energy_among_feasible"]
        lines.append(
            f"| {g['cell_id']} | {g['reference']} | {g['feasible']} | {g['policy_faults']} | "
            f"{e if e is not None else 'undefined'} |"
        )
    lines += [
        "",
        "Energy means use potentially different feasible subsets and are not paired energy "
        "advantages. Faults have unknown QoS, not invented infeasibility metrics. All 972 "
        "individual rows and all pressure-level qualifiers remain in SUMMARY.json.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=runner.OUT)
    out = parser.parse_args().out
    with (
        runner.shared_lock(runner.MATRIX / "execution.lock"),
        runner.shared_lock(out / "execution.lock"),
    ):
        report = verify(out)
        runner.preserve_or_match(out / "SUMMARY.json", report)
        text = render(report)
        path = out / "REPORT.md"
        if path.exists() and path.read_text() != text:
            raise ValueError("existing report differs")
        if not path.exists():
            path.write_text(text)
    print(
        json.dumps(
            {"verified_calibration_episodes": len(report["rows"]), "admission": report["admission"]}
        )
    )


if __name__ == "__main__":
    main()
