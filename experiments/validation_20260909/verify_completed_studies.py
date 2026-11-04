"""Post-completion integrity replay; no new episodes, source edits or model calls.

This uses the frozen analysis routines. Independent statistical/model validation
and matrix fitness reconstruction are separate remaining reviews.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.validation_20260909 import run_analytical_tails as tails
from experiments.validation_20260909 import run_independent_calibration as common
from experiments.validation_20260909 import run_workload_resource_amendment as amendment
from experiments.validation_20260909 import run_workload_sensitivity as matrix
from experiments.validation_20260909 import summarize_workload_sensitivity as report


def check(condition, message):
    if not condition:
        raise ValueError(message)


def archive(out):
    frozen = read_sealed(out / "snapshot.json")
    common.verify_archive(out, frozen, common.sha(out / "sources.tar.gz"))
    return len(frozen["files"]) + len(frozen["native"])


def receipts(out, folder, req, plan, *, trace=False):
    key = digest(req)
    paths = {
        s: out / folder / (key + s)
        for s in (".request.json", ".result.json", ".usage.json", ".log")
    }
    saved, result, usage = [
        read_sealed(paths[s]) for s in (".request.json", ".result.json", ".usage.json")
    ]
    check(
        saved == req and result["request_sha256"] == key and usage["request_sha256"] == key,
        "request binding differs",
    )
    check(usage["exit_code"] == 0 and usage["stop_reason"] is None, "unsuccessful request")
    elapsed, rss = usage["elapsed_seconds"], usage["peak_rss_mib"]
    check(
        math.isfinite(elapsed)
        and 0 < elapsed <= plan["worker_timeout_seconds"]
        and math.isfinite(rss)
        and 0 <= rss <= plan["rss_limit_mib"],
        "resource bounds differ",
    )
    if trace:
        paths[".records.jsonl.gz"] = out / folder / (key + ".records.jsonl.gz")
    hashes = {str(p.relative_to(out)): common.sha(p) for p in paths.values()}
    return result, usage, hashes


def verify_matrix():
    out = matrix.DEFAULT_OUT
    plan, _ = amendment.verify_original(out)
    amended = amendment.verify_amendment(out, plan)
    members = archive(out)
    complete = read_sealed(out / "COMPLETE.json")
    sources = sorted({p["source_sha256"] for p in plan["policies"]})
    check(len(sources) == 29 and len(plan["policies"]) == 49, "source/label count differs")
    usages, files, faults = [], {}, []
    for source in sources:
        check(common.sha(out / "policies" / (source + ".ps")) == source, "policy copy differs")
    for cell in plan["cells"]:
        for source in sources:
            req = matrix.request(plan, source, cell, pilot=False)
            result, usage, hashes = receipts(out, "scores", req, plan)
            usages.append(usage)
            files.update(hashes)
            if result["status"] == "policy-fault":
                faults.append({"request": digest(req), "failure": result["failure"]})
            else:
                check(result["status"] == "evaluated", "unknown matrix outcome")
                evaluation = result["evaluation"]
                check(evaluation["policy"]["artifact_sha256"] == source, "policy identity differs")
                check(
                    sorted((i["seed"], i["bundle_id"]) for i in evaluation["instances"])
                    == sorted((s, b) for s in plan["seeds"] for b in plan["bundles"]),
                    "episode grid differs",
                )
    check(
        {"scores/" + p.name for p in (out / "scores").iterdir()} == set(files),
        "matrix file set differs",
    )
    expected = {
        "plan_sha256": digest(plan),
        "resource_amendment_sha256": digest(amended),
        "maximum_total_worker_seconds": amendment.CAP_SECONDS,
        "requests": 174,
        "total_worker_seconds": sum(u["elapsed_seconds"] for u in usages),
        "peak_rss_mib": max(u["peak_rss_mib"] for u in usages),
        "no_model_calls": True,
    }
    check(
        complete == expected and expected["total_worker_seconds"] <= amendment.CAP_SECONDS,
        "completion/resource amendment differs",
    )
    report.summarize(out)
    return out, {
        "requests": 174,
        "archived_members_verified": members,
        "faults": faults,
        "complete_sha256": digest(complete),
        "files_sha256": files,
        "scope": "Source/native archive, current frozen environment, amendment, exact "
        "request/grid/source and resource bindings verified. Frozen summarizer "
        "replayed. Full independent fitness/energy reconstruction remains open.",
    }


def verify_tails():
    out = tails.OUT
    plan = tails.verify_source(out)
    members = archive(out)
    saved = read_sealed(out / "RESULTS.json")
    results, usages, files = [], [], {}
    for job in plan["jobs"]:
        req = {"plan_sha256": digest(plan), "job": job}
        result, usage, hashes = receipts(out, "runs", req, plan, trace=True)
        check(all(result[k] == v for k, v in job.items()), "tail job identity differs")
        path = out / "runs" / (digest(req) + ".records.jsonl.gz")
        raw = gzip.decompress(path.read_bytes())
        check(
            common.sha(path) == result["records_file_sha256"]
            and hashlib.sha256(raw).hexdigest() == result["records_uncompressed_sha256"],
            "trace digest differs",
        )
        rows = [json.loads(line) for line in raw.splitlines()]
        check(len(rows) == result["all_offered_records"], "offered trace count differs")
        check(
            all(
                type(r[0]) is int
                and r[0] == i
                and math.isfinite(r[1])
                and 0 <= r[1] < 4600
                and (i == 0 or r[1] >= rows[i - 1][1])
                and (r[2] is None or math.isfinite(r[2]) and r[1] <= r[2] <= job["horizon"])
                for i, r in enumerate(rows)
            ),
            "invalid offered records",
        )
        records = [SimpleNamespace(rid=r[0], arrival_s=r[1], end_s=r[2], status=r[3]) for r in rows]
        for name, window in (
            ("primary", tails.PRIMARY_WINDOW),
            ("early", tails.EARLY_WINDOW),
            ("late", tails.LATE_WINDOW),
        ):
            check(
                result[name] == tails.cohort(records, window, job["rate"]), "cohort replay differs"
            )
        results.append(result)
        usages.append(usage)
        files.update(hashes)
    check(
        {"runs/" + p.name for p in (out / "runs").iterdir()} == set(files), "tail file set differs"
    )
    horizon = [
        {
            "rate": short["rate"],
            "seed": short["seed"],
            "primary_cohort_identical": short["primary"]
            == next(
                long["primary"]
                for long in results
                if long["rate"] == short["rate"]
                and long["seed"] == short["seed"]
                and long["horizon"] == 5000
            ),
        }
        for short in results
        if short["horizon"] == 4800
    ]
    rates = [
        tails.analyze_rate([r for r in results if r["rate"] == rate and r["horizon"] == 5000], rate)
        for rate in tails.ARRIVAL_RATES
    ]
    expected = {
        "plan_sha256": digest(plan),
        "rates": rates,
        "horizon_checks": horizon,
        "replicates": results,
        "all_fixed_checks_pass": all(c["primary_cohort_identical"] for c in horizon)
        and all(r["status"] == "within-fixed-tolerances" for r in rates),
        "total_worker_seconds": sum(u["elapsed_seconds"] for u in usages),
        "peak_rss_mib": max(u["peak_rss_mib"] for u in usages),
        "boundary": plan["boundary"],
    }
    check(
        saved == expected
        and len(results) == 136
        and len(horizon) == 8
        and saved["total_worker_seconds"] <= plan["maximum_total_worker_seconds"],
        "tail completion replay differs",
    )
    return out, {
        "requests": 136,
        "archived_members_verified": members,
        "files_sha256": files,
        "results_sha256": digest(saved),
        "all_fixed_checks_pass": saved["all_fixed_checks_pass"],
        "scope": "Source/native archive and current frozen environment verified; every "
        "offered trace and frozen cohort/interval/horizon analysis replayed exactly. "
        "Shared analysis routines are not an independent statistical implementation "
        "or empirical validation.",
    }


def main():
    with (
        common.shared_lock(matrix.DEFAULT_OUT / "execution.lock"),
        common.shared_lock(tails.OUT / "execution.lock"),
    ):
        for verify in (verify_matrix, verify_tails):
            out, payload = verify()
            payload["verifier_sha256"] = common.sha(Path(__file__))
            common.preserve_or_match(out / "INTEGRITY_REPLAY_20260910.json", payload)
            print(out.name, {k: v for k, v in payload.items() if k != "files_sha256"}, flush=True)


if __name__ == "__main__":
    main()
