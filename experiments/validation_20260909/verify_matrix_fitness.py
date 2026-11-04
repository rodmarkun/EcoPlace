"""Post-completion matrix accounting and independent fitness arithmetic review.

No simulator calls. SLO attainment remains a retained metric: root/service traces
were omitted by this run and cannot be reconstructed from compact evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from evolution.evaluation import config_for_development_bundle, load_task_evaluation_config
from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.validation_20260909 import run_workload_resource_amendment as amendment
from experiments.validation_20260909 import run_workload_sensitivity as runner
from experiments.validation_20260909.run_independent_calibration import preserve_or_match, sha
from sim.energy import build_development_bundles, load_pinned_sources
from sim.energy.boundary import VerifiedInstanceEnergy, certify_boundary
from sim.spec import config_sha256, load_episode_config


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    require(
        type(a) in (int, float)
        and math.isfinite(a)
        and math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-9),
        "accounting arithmetic differs",
    )


def fitness(spec, instances, expected_ids):
    indexed = {(i["scenario_id"], i["seed"], i["bundle_id"]): i for i in instances}
    require(
        len(indexed) == len(instances) and set(indexed) == set(expected_ids), "fitness grid differs"
    )
    groups = defaultdict(list)
    for i in instances:
        require(
            set(i["slo_achieved"]) == set(spec.slo_targets)
            and set(i["disruption"]) == set(spec.disruption_caps),
            "constraint components differ",
        )
        require(
            all(
                type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                for v in [
                    *i["slo_achieved"].values(),
                    i["goodput"],
                    i["drop_rate"],
                    i["timeout_rate"],
                ]
            ),
            "invalid delivery metric",
        )
        require(
            all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in i["disruption"].values()
            ),
            "invalid disruption metric",
        )
        caps = [
            ("drop", spec.drop_cap, i["drop_rate"]),
            ("timeout", spec.timeout_cap, i["timeout_rate"]),
        ]
        caps += [(n, c, i["disruption"][n]) for n, c in sorted(spec.disruption_caps.items())]
        upper = [max(0.0, (v - c) / (c or spec.zero_cap_scales[n])) for n, c, v in caps]
        slo = [
            max(0.0, (t - i["slo_achieved"][n]) / t) for n, t in sorted(spec.slo_targets.items())
        ]
        groups[(i["scenario_id"], i["seed"])].append(
            slo + [max(0.0, (spec.goodput_target - i["goodput"]) / spec.goodput_target)] + upper
        )
    vectors, totals = [], []
    n_slo = len(spec.slo_targets)
    for values in groups.values():
        maxima = [max(col) for col in zip(*values, strict=True)]
        vectors.append(
            [
                max(maxima[:n_slo]),
                maxima[n_slo],
                max(maxima[n_slo + 1 : n_slo + 3]),
                max(maxima[n_slo + 3 :]),
            ]
        )
        totals.append(math.fsum(maxima))
    vector = [math.fsum(col) / len(groups) for col in zip(*vectors, strict=True)]
    total = math.fsum(totals)
    feasible = total == 0
    energy = (
        math.fsum(i["energy_j"] / i["idle_floor_j"] for i in instances) / len(instances)
        if feasible
        else None
    )
    return {
        "feasible": feasible,
        "infeasibility_vector": vector,
        "summed_primitive_shortfall": total,
        "normalized_energy": energy,
        "proposal_seq": 0,
        "n_instances": len(groups),
        "n_bundles": len({k[2] for k in indexed}),
        "direction": "lower-is-better",
        "sort_key": [0, energy, 0] if feasible else [1, *vector, total, 0],
    }


def instance_check(i, config, task):
    episode = i["episode"]
    require(
        episode["config_sha256"] == config_sha256(config) and episode["seed"] == i["seed"],
        "effective episode identity differs",
    )
    close(episode["horizon_s"], config.horizon_s)
    cert = certify_boundary(
        config.cluster,
        config.energy,
        config.horizon_s,
        task.weights,
        task.bounds,
        scenario_id=i["scenario_id"],
        seed=i["seed"],
        bundle_id=i["bundle_id"],
    )
    require(
        i["boundary_certificate"] == {**cert.payload(), "seal": cert.seal}
        and i["certificate_seal"] == cert.seal,
        "effective boundary certificate differs",
    )
    token = VerifiedInstanceEnergy(**i["verified_energy"])
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
        require(getattr(token, name) == getattr(cert, name), "token/config binding differs")
    require(
        token.certificate_seal == cert.seal
        and len(token.transition_counts) == cert.n_nodes
        and all(
            type(n) is int and 0 <= n <= task.bounds.max_transitions_per_node
            for n in token.transition_counts
        )
        and 0 <= token.network_bytes <= task.bounds.network_bytes_bound
        and 0 <= token.energy_j <= token.ceiling_j,
        "energy bound differs",
    )
    e = episode["energy"]
    require(e["transition_counts"] == list(token.transition_counts), "transition ledger differs")
    for key in ("cluster_sha256", "coefficients_sha256"):
        require(e[key] == getattr(cert, key), "meter identity differs")
    for name in ("device_j", "cloud_j", "network_j", "cloud_vcpu_hours", "network_bytes"):
        require(
            type(e[name]) in (int, float) and math.isfinite(e[name]) and e[name] >= 0,
            "invalid energy base",
        )
    for key in ("device_j_per_node", "transition_j_per_node"):
        require(
            len(e[key]) == cert.n_nodes
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in e[key]),
            "invalid node energy",
        )
    require(
        all(
            a <= b for a, b in zip(e["transition_j_per_node"], e["device_j_per_node"], strict=True)
        ),
        "transition energy exceeds node energy",
    )
    close(e["device_j"], math.fsum(e["device_j_per_node"]))
    close(e["cloud_j"], e["cloud_vcpu_hours"] * config.energy.cloud_j_per_vcpu_hour)
    close(e["network_j"], e["network_bytes"] * config.energy.network_j_per_byte)
    close(e["network_bytes"], token.network_bytes)
    close(
        token.energy_j,
        e["device_j"] * task.weights.device
        + e["cloud_j"] * task.weights.cloud
        + e["network_j"] * task.weights.network,
    )
    for key in ("energy_j", "idle_floor_j", "ceiling_j"):
        close(i[key], getattr(token, key))
    ledger = episode["ledger"]
    statuses = [
        "completed",
        "timeout",
        "dropped_no_replica",
        "dropped_queue_full",
        "in_flight_at_end",
    ]
    counts = [ledger["requests_" + s] for s in statuses]
    offered = ledger["requests_injected"]
    require(
        all(type(n) is int and n >= 0 for n in [offered, *counts])
        and sum(counts) == offered
        and offered == episode["records"]["length"]
        and ledger["scorer_errors"] == 0,
        "request conservation differs",
    )
    require(
        i["goodput"] == (counts[0] / offered if offered else 1.0)
        and i["timeout_rate"] == counts[1] / max(1, offered)
        and i["drop_rate"] == (counts[2] + counts[3]) / max(1, offered),
        "delivery ledger differs",
    )
    require(
        all(
            i["disruption"][n] == episode["rebalancer"][n] for n in task.constraints.disruption_caps
        ),
        "disruption ledger differs",
    )


def main():
    out = runner.DEFAULT_OUT
    plan, _ = amendment.verify_original(out)
    amendment.verify_amendment(out, plan)
    replay = read_sealed(out / "INTEGRITY_REPLAY_20260910.json")
    require(
        replay["complete_sha256"] == digest(read_sealed(out / "COMPLETE.json")),
        "completion changed",
    )
    for path, expected in replay["files_sha256"].items():
        require(sha(out / path) == expected, "integrity-replayed result changed")
    task = load_task_evaluation_config(runner.TASK)
    bundles = {b.bundle_id: b for b in build_development_bundles(load_pinned_sources())}
    configs = {}
    checks = []
    for cell in plan["cells"]:
        for grid in task.scenarios:
            for bid in plan["bundles"]:
                base = config_for_development_bundle(load_episode_config(grid.path), bundles[bid])
                configs[(cell["id"], grid.scenario_id, bid)] = runner.transform(
                    base, shape=cell["shape"], fraction=cell["fraction"], fits=plan["fits"]
                )
        for source in sorted({p["source_sha256"] for p in plan["policies"]}):
            key = digest(runner.request(plan, source, cell, pilot=False))
            result = read_sealed(out / "scores" / (key + ".result.json"))
            require(
                result["status"] == "evaluated", "this completed matrix was recorded as fault-free"
            )
            ev = result["evaluation"]
            expected = [
                (g.scenario_id, s, b)
                for g in task.scenarios
                for s in plan["seeds"]
                for b in plan["bundles"]
            ]
            rebuilt = fitness(task.constraints, ev["instances"], expected)
            saved = ev["fitness"]
            require(
                saved.keys() == rebuilt.keys()
                and type(saved["feasible"]) is bool
                and saved["feasible"] == rebuilt["feasible"],
                "fitness fields or feasibility differ",
            )
            for name in rebuilt:
                a, b = saved[name], rebuilt[name]
                if type(b) is float:
                    close(a, b)
                elif isinstance(b, list):
                    require(len(a) == len(b), "fitness vector length differs")
                    for x, y in zip(a, b, strict=True):
                        close(x, y)
                else:
                    require(a == b, "fitness metadata differs")
            for i in ev["instances"]:
                instance_check(i, configs[(cell["id"], i["scenario_id"], i["bundle_id"])], task)
            checks.append(
                {
                    "request_sha256": key,
                    "instances": len(ev["instances"]),
                    "feasible": rebuilt["feasible"],
                }
            )
    amendment.verify_original(out)
    payload = {
        "schema": "matrix-accounting-fitness-review-v1",
        "integrity_replay_sha256": digest(replay),
        "verifier_sha256": sha(Path(__file__)),
        "requests": len(checks),
        "instances": sum(r["instances"] for r in checks),
        "checks": checks,
        "scope": "Independent fitness aggregation and energy/category arithmetic match "
        "retained evidence. Shared config/certificate/token routines verify modeled "
        "boundaries. Goodput/drop/timeout recomputed from aggregate ledgers. SLO "
        "attainment, event-level energy integrals and full episode digests cannot be "
        "independently rederived because request/service/snapshot traces were "
        "omitted. No simulations or empirical-validity claim.",
    }
    preserve_or_match(out / "FITNESS_ACCOUNTING_REVIEW_20260911.json", payload)
    print({k: v for k, v in payload.items() if k != "checks"}, flush=True)


if __name__ == "__main__":
    main()
