"""Independent arithmetic review of the completed tail witness; no simulator calls.

Uses NumPy reductions and scipy.special inverse-t instead of the frozen
cohort/statistics implementation. Shared libraries do not imply independent
physical validation or exact Monte Carlo coverage.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import stdtrit

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.validation_20260909.run_independent_calibration import preserve_or_match, sha

OUT = Path(__file__).parent / "analytical_tails_20260909"
WINDOWS = {"primary": (600, 4600), "early": (300, 2300), "late": (2600, 4600)}
QUANTILES = (0.95, 0.99)
TOLERANCES = (0.005, 0.0025)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), "statistic fields differ")
        for k, v in expected.items():
            equal(actual[k], v)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "statistic lengths differ")
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    elif type(expected) is float:
        require(
            math.isclose(actual, expected, abs_tol=2e-14, rel_tol=2e-13),
            "numeric statistic differs",
        )
    else:
        require(actual == expected, "statistic differs")


def cohort(rows, window, rate):
    selected = [r for r in rows if window[0] <= r[1] < window[1]]
    counts = {status: sum(r[3] == status for r in selected) for status in {r[3] for r in selected}}
    valid = bool(selected) and all(
        r[3] == "completed" and r[2] is not None and math.isfinite(r[2]) and r[2] >= r[1]
        for r in selected
    )
    result = {
        "window": list(window),
        "offered": len(selected),
        "status_counts": counts,
        "all_offered_completed": valid,
        "cohort_sha256": digest(selected),
    }
    if valid:
        latency = np.array([r[2] - r[1] for r in selected], dtype=float)
        result["mean_sojourn_s"] = float(np.mean(latency))
        result["tails"] = []
        for q in QUANTILES:
            threshold = -math.log1p(-q) / (12.5 - rate)
            result["tails"].append(
                {
                    "q": q,
                    "theoretical_quantile_s": threshold,
                    "expected_exceedance": 1 - q,
                    "observed_exceedance": float(
                        np.count_nonzero(latency > threshold) / len(latency)
                    ),
                    "observed_inverse_ecdf_quantile_s": float(
                        np.quantile(latency, q, method="inverted_cdf")
                    ),
                }
            )
    return result


def interval(values):
    require(len(values) == 64 and np.isfinite(values).all(), "64 finite replicate values required")
    average = float(np.average(values))
    se = float(np.sqrt(np.var(values, ddof=1) / 64))
    critical = float(stdtrit(63, 1 - 0.05 / 16))
    return {
        "n_independent_runs": 64,
        "mean": average,
        "standard_error": se,
        "critical_t": critical,
        "nominal_family_confidence": 0.95,
        "lower": average - critical * se,
        "upper": average + critical * se,
    }


def main():
    plan = read_sealed(OUT / "PLAN.json")
    retained = read_sealed(OUT / "RESULTS.json")
    replay = read_sealed(OUT / "INTEGRITY_REPLAY_20260910.json")
    require(
        replay["results_sha256"] == digest(retained) and retained["plan_sha256"] == digest(plan),
        "completion identity differs",
    )
    for name, expected in replay["files_sha256"].items():
        require(sha(OUT / name) == expected, "integrity-replayed evidence changed")
    jobs = plan["jobs"]
    identities = [(j["rate"], j["seed"], j["horizon"]) for j in jobs]
    require(len(identities) == len(set(identities)) == 136, "job grid differs")
    rows_by_job, results = {}, []
    for job in jobs:
        req = {"plan_sha256": digest(plan), "job": job}
        key = digest(req)
        saved = read_sealed(OUT / "runs" / (key + ".result.json"))
        raw = gzip.decompress((OUT / "runs" / (key + ".records.jsonl.gz")).read_bytes())
        require(
            hashlib.sha256(raw).hexdigest() == saved["records_uncompressed_sha256"], "trace differs"
        )
        rows = [json.loads(line) for line in raw.splitlines()]
        require(len(rows) == saved["all_offered_records"], "offered count differs")
        current = {name: cohort(rows, window, job["rate"]) for name, window in WINDOWS.items()}
        for name, result in current.items():
            equal(saved[name], result)
        results.append({**job, **current})
        rows_by_job[(job["rate"], job["seed"], job["horizon"])] = [
            r for r in rows if 600 <= r[1] < 4600
        ]
    checks = []
    for rate in (8.0, 10.625):
        group = [r for r in results if r["rate"] == rate and r["horizon"] == 5000]
        require(len(group) == len({r["seed"] for r in group}) == 64, "replicate grid differs")
        require(
            all(r[w]["all_offered_completed"] for r in group for w in WINDOWS),
            "cohort integrity failed",
        )
        saved_rate = next(r for r in retained["rates"] if r["rate"] == rate)
        for i, (q, tol) in enumerate(zip(QUANTILES, TOLERANCES, strict=True)):
            primary = interval([r["primary"]["tails"][i]["observed_exceedance"] for r in group])
            delta = interval(
                [
                    r["late"]["tails"][i]["observed_exceedance"]
                    - r["early"]["tails"][i]["observed_exceedance"]
                    for r in group
                ]
            )
            endpoint = saved_rate["endpoints"][i]
            equal(endpoint["primary"], primary)
            equal(endpoint["late_minus_early"], delta)
            primary_ok = primary["lower"] >= 1 - q - tol and primary["upper"] <= 1 - q + tol
            delta_ok = delta["lower"] >= -tol and delta["upper"] <= tol
            require(
                primary_ok == endpoint["primary_within_tolerance"]
                and delta_ok == endpoint["window_difference_within_tolerance"],
                "tolerance decision differs",
            )
            checks.append(
                {
                    "rate": rate,
                    "q": q,
                    "primary": primary,
                    "late_minus_early": delta,
                    "primary_pass": primary_ok,
                    "difference_pass": delta_ok,
                }
            )
    horizon = []
    for rate, seed, h in identities:
        if h == 4800:
            horizon.append(rows_by_job[(rate, seed, h)] == rows_by_job[(rate, seed, 5000)])
    require(len(horizon) == 8 and all(horizon), "paired raw primary records differ")
    overall = all(c["primary_pass"] and c["difference_pass"] for c in checks)
    require(overall == retained["all_fixed_checks_pass"], "overall conclusion differs")
    payload = {
        "schema": "independent-tail-arithmetic-review-v1",
        "results_sha256": digest(retained),
        "integrity_replay_sha256": digest(replay),
        "verifier_sha256": sha(Path(__file__)),
        "cohorts_verified": 408,
        "horizon_raw_record_checks": 8,
        "endpoints": checks,
        "overall_fixed_checks_pass": overall,
        "scope": "Independent NumPy/scipy.special arithmetic reproduces all cohort metrics and "
        "nominal intervals; all offered records retained, raw paired horizon cohorts identical. "
        "No new simulations. Not empirical microservice validity, exact confidence coverage "
        "or stationarity proof.",
    }
    preserve_or_match(OUT / "INDEPENDENT_STATISTICS_20260910.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "endpoints"}), flush=True)


if __name__ == "__main__":
    main()
