"""Report all frozen workload cells only after the matrix is complete."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import mean

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.validation_20260909.run_workload_sensitivity import (
    DEFAULT_OUT,
    PILOT_BASELINE,
    request,
)


def comparison(pairs: list[tuple[dict, dict]]) -> dict:
    wins = losses = ties = 0
    differences = []
    for arm, comparator in pairs:
        if arm["feasible"] and comparator["feasible"]:
            delta = comparator["energy"] - arm["energy"]
            differences.append(delta)
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
        elif arm["feasible"]:
            wins += 1
        elif comparator["feasible"]:
            losses += 1
        else:
            ties += 1
    return {
        "n": len(pairs),
        "arm_wins": wins,
        "comparator_wins": losses,
        "ties": ties,
        "jointly_feasible": len(differences),
        "mean_comparator_minus_arm_energy": mean(differences) if differences else None,
    }


def summarize(out: Path) -> None:
    plan, complete = read_sealed(out / "PLAN.json"), read_sealed(out / "COMPLETE.json")
    if complete["plan_sha256"] != digest(plan):
        raise ValueError("completion refers to a different plan")
    sources = {p["source_sha256"] for p in plan["policies"]}
    if complete["requests"] != len(sources) * len(plan["cells"]):
        raise ValueError("matrix is incomplete")
    rows, expected_keys = [], set()
    for cell in plan["cells"]:
        results = {}
        for source_hash in sources:
            req = request(plan, source_hash, cell, pilot=False)
            key = digest(req)
            expected_keys.add(key)
            base = out / "scores" / key
            saved = read_sealed(base.with_suffix(".request.json"))
            result = read_sealed(base.with_suffix(".result.json"))
            usage = read_sealed(base.with_suffix(".usage.json"))
            if (
                saved != req
                or result["request_sha256"] != key
                or usage["request_sha256"] != key
                or usage["exit_code"] != 0
                or usage["stop_reason"] is not None
            ):
                raise ValueError("request/result/usage binding mismatch")
            if result["status"] == "policy-fault":
                results[source_hash] = {
                    "status": "policy-fault",
                    "feasible": False,
                    "energy": None,
                    "worst_slo": None,
                    "failure": result["failure"],
                }
                continue
            if result["status"] != "evaluated":
                raise ValueError("unexpected result status")
            evaluation = result["evaluation"]
            fitness = evaluation["fitness"]
            instances = evaluation["instances"]
            actual = [(r["seed"], r["bundle_id"]) for r in instances]
            expected = [(s, b) for s in plan["seeds"] for b in plan["bundles"]]
            if sorted(actual) != sorted(expected):
                raise ValueError("instance grid is incomplete or changed")
            results[source_hash] = {
                "status": "evaluated",
                "feasible": fitness["feasible"],
                "energy": fitness["normalized_energy"] if fitness["feasible"] else None,
                "worst_slo": min(v for r in instances for v in r["slo_achieved"].values()),
            }
        labels = {p["label"]: {**p, **results[p["source_sha256"]]} for p in plan["policies"]}
        groups = {
            arm: [labels[f"{arm}-{root}"] for root in range(800000801, 800000821)]
            for arm in ("llm", "same_dsl_gp")
        }
        reference = labels[PILOT_BASELINE]
        rows.append(
            {
                "cell": cell,
                "labels": labels,
                "feasible": {
                    arm: sum(p["feasible"] for p in group) for arm, group in groups.items()
                },
                "policy_faults": {
                    arm: sum(p["status"] == "policy-fault" for p in group)
                    for arm, group in groups.items()
                },
                "llm_vs_gp": comparison(
                    list(zip(groups["llm"], groups["same_dsl_gp"], strict=True))
                ),
                "versus_strong_reference": {
                    arm: comparison([(p, reference) for p in group])
                    for arm, group in groups.items()
                },
            }
        )
    if {
        p.name.removesuffix(".result.json") for p in (out / "scores").glob("*.result.json")
    } != expected_keys:
        raise ValueError("extra or missing scores outside frozen matrix")
    payload = {
        "plan_sha256": digest(plan),
        "rows": rows,
        "usage": complete,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": plan["boundary"],
    }
    write_sealed(out / "SUMMARY.json", payload)
    lines = [
        "# Fixed-policy workload sensitivity",
        "",
        plan["boundary"],
        "",
        "All counts retain the original 20 champion labels per arm. Identical source bytes",
        "share results; the eight environmental seeds are common across policies and cells.",
        "These are descriptive counts, not independent replications or significance tests.",
        "",
        "| Cell | LLM feasible /20 | GP feasible /20 | Strong reference feasible | "
        "LLM–GP wins/losses/ties | Jointly feasible | Mean GP−LLM energy |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        c = row["llm_vs_gp"]
        delta = c["mean_comparator_minus_arm_energy"]
        delta_text = "undefined" if delta is None else f"{delta:.6f}"
        lines.append(
            f"| {row['cell']['id']} | {row['feasible']['llm']} | "
            f"{row['feasible']['same_dsl_gp']} | {row['labels'][PILOT_BASELINE]['feasible']} | "
            f"{c['arm_wins']}/{c['comparator_wins']}/{c['ties']} | "
            f"{c['jointly_feasible']} | {delta_text} |"
        )
    lines += [
        "",
        "Positive paired energy favors LLM only within the jointly feasible subset.",
        "",
        "| Cell | Reference | Feasible | Normalized energy if feasible | Worst SLO |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        for label in row["labels"].values():
            if label["group"] not in ("fixed-reference", "posthoc-reference"):
                continue
            energy = "undefined" if label["energy"] is None else f"{label['energy']:.6f}"
            slo = "undefined" if label["worst_slo"] is None else f"{label['worst_slo']:.6f}"
            lines.append(
                f"| {row['cell']['id']} | {label['label']} | "
                f"{label['feasible']} | {energy} | {slo} |"
            )
    lines += [
        "",
        "All 49 labels per cell, fault details and arm-versus-strong-reference comparisons",
        "are retained in SUMMARY.json. Both infeasible counts as a descriptive tie;",
        "faults remain explicitly identified. No infeasible energy is replaced with zero.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    summarize(parser.parse_args().out)
