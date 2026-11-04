"""Shared, deterministic offered-arrival stream generation.

The DES and the Kubernetes fixture exporter intentionally call this same iterator.
Keeping cumulative-intensity inversion in one place is what makes their environmental
stimulus byte-for-byte comparable for a fixed EpisodeConfig and episode seed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from analysis.tracking import spawn_rng
from sim.spec import ArrivalSpec


def iter_arrival_times(
    spec: ArrivalSpec,
    cutoff_s: float,
    rng: Any | None = None,
) -> Iterator[float]:
    """Yield exact offered times before ``cutoff_s``.

    Realized scenario artifacts need no RNG. Unsealed Poisson streams require the
    caller's explicitly derived generator; deterministic streams do not consume it.
    """
    if spec.arrivals_realized:
        yield from (at_s for at_s in spec.realized_times_s if at_s < cutoff_s)
        return
    if spec.kind == "poisson" and rng is None:
        raise ValueError("an explicit RNG is required for an unrealized Poisson stream")

    target = 0.0
    deterministic_index = 0
    while True:
        if spec.kind == "poisson":
            target += float(rng.exponential(1.0))
        else:
            deterministic_index += 1
            target = float(deterministic_index)
        at_s = spec.time_at_integrated_rate(target, cutoff_s)
        if at_s is None or at_s >= cutoff_s:
            return
        yield at_s


def episode_arrival_times(
    spec: ArrivalSpec,
    cutoff_s: float,
    episode_seed: int,
    template_id: int,
) -> tuple[float, ...]:
    """Materialize the sanctioned ``sim/arrivals/<template>`` episode substream."""
    rng = None
    if not spec.arrivals_realized and spec.kind == "poisson":
        rng = spawn_rng(episode_seed, "sim", "arrivals", template_id)
    return tuple(iter_arrival_times(spec, cutoff_s, rng))
