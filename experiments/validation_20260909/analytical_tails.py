"""Development M/M/1 tail witness with independent-replicate uncertainty.

This does not access acceptance outcomes or validate empirical request tails.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from experiments.gp_tuning_20260902.recovery import digest
from sim.spec import EpisodeConfig, load_episode_config

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "sim/fixtures/dev/mm1.json"
ARRIVAL_RATES = (8.0, 10.625)
SERVICE_RATE = 12.5
REPLICATES = 64
HORIZONS = (4800.0, 5000.0)
PRIMARY_WINDOW = (600.0, 4600.0)
EARLY_WINDOW = (300.0, 2300.0)
LATE_WINDOW = (2600.0, 4600.0)
QUANTILES = (0.95, 0.99)
ABSOLUTE_TOLERANCES = {0.95: 0.005, 0.99: 0.0025}
INTERVAL_FAMILY_SIZE = 8


def theoretical_quantile(rate: float, q: float) -> float:
    if not 0 < rate < SERVICE_RATE or not 0 < q < 1:
        raise ValueError("stationary single-server queue and interior quantile required")
    return -math.log1p(-q) / (SERVICE_RATE - rate)


def episode(rate: float, horizon: float) -> EpisodeConfig:
    if rate not in ARRIVAL_RATES or horizon not in HORIZONS:
        raise ValueError("condition outside the fixed analytical witness")
    base = load_episode_config(BASE)
    if len(base.workload.services) != 1 or len(base.workload.templates) != 1:
        raise ValueError("M/M/1 witness requires one service and one request template")
    service, template = base.workload.services[0], base.workload.templates[0]
    if (
        service.concurrency != 1
        or service.initial_replicas != 1
        or service.cpu_demand != 0
        or service.queue_capacity is not None
        or service.backend is not None
        or service.service_time.kind != "exponential"
        or service.service_time.params != {"mean": 1 / SERVICE_RATE}
        or template.root.children
        or template.slo_latency_s is not None
        or template.arrival.kind != "poisson"
        or template.arrival.rate_schedule
        or base.hpa.enabled
        or base.rebalancer.enabled
        or base.power.suspend_after_idle_s is not None
        or len(base.cluster.nodes) != 1
        or base.cluster.latency_ms != ((0.0,),)
    ):
        raise ValueError("development fixture no longer satisfies the analytical assumptions")
    workload = dataclasses.replace(
        base.workload,
        templates=(
            dataclasses.replace(
                template, arrival=dataclasses.replace(template.arrival, rate_per_s=rate)
            ),
        ),
    )
    return dataclasses.replace(base, workload=workload, arrival_cutoff_s=4600.0, horizon_s=horizon)


def cohort(records, window: tuple[float, float], rate: float) -> dict:
    selected = [r for r in records if window[0] <= r.arrival_s < window[1]]
    counts = dict(Counter(r.status for r in selected))
    retained = [[r.rid, r.arrival_s, r.end_s, r.status] for r in selected]
    valid = bool(selected) and all(
        r.status == "completed"
        and r.end_s is not None
        and math.isfinite(r.end_s)
        and r.end_s >= r.arrival_s
        for r in selected
    )
    result = {
        "window": list(window),
        "offered": len(selected),
        "status_counts": counts,
        "all_offered_completed": valid,
        "cohort_sha256": digest(retained),
    }
    if not valid:
        return result
    latencies = sorted(r.end_s - r.arrival_s for r in selected)
    result["mean_sojourn_s"] = mean(latencies)
    result["tails"] = [
        {
            "q": q,
            "theoretical_quantile_s": theoretical_quantile(rate, q),
            "expected_exceedance": 1 - q,
            "observed_exceedance": sum(x > theoretical_quantile(rate, q) for x in latencies)
            / len(latencies),
            "observed_inverse_ecdf_quantile_s": latencies[math.ceil(q * len(latencies)) - 1],
        }
        for q in QUANTILES
    ]
    return result


def replicate_interval(values: list[float]) -> dict:
    """Nominal Bonferroni-adjusted t interval using runs as sampling units."""
    if len(values) != REPLICATES or any(not math.isfinite(x) for x in values):
        raise ValueError("all 64 finite independent replicate summaries must be retained")
    center = mean(values)
    sem = stdev(values) / math.sqrt(len(values))
    critical = float(t.ppf(1 - 0.05 / (2 * INTERVAL_FAMILY_SIZE), len(values) - 1))
    half_width = critical * sem
    return {
        "n_independent_runs": len(values),
        "mean": center,
        "standard_error": sem,
        "critical_t": critical,
        "nominal_family_confidence": 0.95,
        "lower": center - half_width,
        "upper": center + half_width,
    }


def within_tolerance(interval: dict, *, target: float, tolerance: float) -> bool:
    # Containment requires adequate precision; merely including the target is not enough.
    return interval["lower"] >= target - tolerance and interval["upper"] <= target + tolerance


def analyze_rate(rows: list[dict], rate: float) -> dict:
    if len(rows) != REPLICATES or len({r["seed"] for r in rows}) != REPLICATES:
        raise ValueError("rate does not retain all 64 distinct simulation replicates")
    if any(r["rate"] != rate or r["horizon"] != 5000.0 for r in rows):
        raise ValueError("main analysis requires the registered rate and long horizon")
    valid = all(r[w]["all_offered_completed"] for r in rows for w in ("primary", "early", "late"))
    if not valid:
        return {"rate": rate, "status": "failed-cohort-integrity", "all_replicates_retained": True}
    endpoints = []
    for q in QUANTILES:

        def exceed(row, window, q=q):
            return next(v["observed_exceedance"] for v in row[window]["tails"] if v["q"] == q)

        primary = replicate_interval([exceed(row, "primary") for row in rows])
        warmup = replicate_interval([exceed(row, "late") - exceed(row, "early") for row in rows])
        target, tol = 1 - q, ABSOLUTE_TOLERANCES[q]
        endpoints.append(
            {
                "q": q,
                "target_tail_mass": target,
                "absolute_tolerance": tol,
                "primary": primary,
                "late_minus_early": warmup,
                "primary_within_tolerance": within_tolerance(primary, target=target, tolerance=tol),
                "window_difference_within_tolerance": within_tolerance(
                    warmup, target=0, tolerance=tol
                ),
            }
        )
    return {
        "rate": rate,
        "rho": rate / SERVICE_RATE,
        "endpoints": endpoints,
        "status": "within-fixed-tolerances"
        if all(
            e["primary_within_tolerance"] and e["window_difference_within_tolerance"]
            for e in endpoints
        )
        else "not-established-at-fixed-precision",
        "interval_boundary": "Nominal Monte Carlo intervals across independent runs; "
        "not exact finite-sample coverage or proof of stationarity.",
    }
