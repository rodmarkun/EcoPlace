"""Fixed-policy development stress test of interval-mean workload assumptions.

Only service-time distributions change. Neither shape is a validated per-request
distribution; the CPU attribution fractions are stress settings, not estimates.
"""

from __future__ import annotations

import dataclasses
import math
from statistics import NormalDist

from sim.spec import DistributionSpec, EpisodeConfig

SHAPES = ("original", "observed-mean-p95")
FRACTIONS = (0.30, 0.45, 0.60)


def matched_lognormal(mean: float, p95: float, original_sigma: float) -> tuple[float, float]:
    """Match arithmetic mean and p95, choosing the root nearest the old sigma."""
    if any(not math.isfinite(x) or x <= 0 for x in (mean, p95, original_sigma)):
        raise ValueError("mean, p95 and original sigma must be finite and positive")
    z = NormalDist().inv_cdf(0.95)
    discriminant = z * z - 2 * math.log(p95 / mean)
    if discriminant < 0:
        raise ValueError("mean and p95 cannot be represented by a lognormal")
    roots = [s for s in (z - math.sqrt(discriminant), z + math.sqrt(discriminant)) if s > 0]
    sigma = min(roots, key=lambda s: (abs(s - original_sigma), s))
    return math.log(mean) - sigma * sigma / 2, sigma


def exact_observation(bounds: list[float]) -> float:
    if len(bounds) != 2 or bounds[0] != bounds[1] or not math.isfinite(bounds[0]):
        raise ValueError("this design requires exact uncensored mean and p95 observations")
    return bounds[0]


def transform(
    config: EpisodeConfig, *, shape: str, fraction: float, fits: list[dict]
) -> EpisodeConfig:
    if shape not in SHAPES or fraction not in FRACTIONS or isinstance(fraction, bool):
        raise ValueError("setting is outside the fixed six-cell grid")
    by_name = {"ali-" + row["service"][:12]: row for row in fits}
    if len(by_name) != len(fits) or set(by_name) != {s.name for s in config.workload.services}:
        raise ValueError("fit inventory does not exactly match the service population")
    services = []
    for service in config.workload.services:
        if service.service_time.kind != "lognormal":
            raise ValueError("expected the historical lognormal service family")
        fit = by_name[service.name]
        if not fit["seal_matches"] or not fit["historical_parameters_and_counts_match"]:
            raise ValueError("unverified fit record")
        mu, sigma = (service.service_time.params[k] for k in ("mu", "sigma"))
        if shape == "observed-mean-p95":
            mean = exact_observation(fit["observed_mean_bounds_s"])
            p95 = exact_observation(
                next(q["observed_bounds_s"] for q in fit["quantiles"] if q["q"] == 0.95)
            )
            mu, sigma = matched_lognormal(mean * 0.45, p95 * 0.45, sigma)
        if fraction != 0.45:
            mu += math.log(fraction / 0.45)
        distribution = DistributionSpec("lognormal", {"mu": mu, "sigma": sigma})
        services.append(dataclasses.replace(service, service_time=distribution))
    return dataclasses.replace(
        config, workload=dataclasses.replace(config.workload, services=tuple(services))
    )
