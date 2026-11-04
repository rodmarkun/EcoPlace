"""Check analytical targets, censoring preservation and replicate-level inference."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from experiments.validation_20260909.analytical_tails import (
    ARRIVAL_RATES,
    SERVICE_RATE,
    cohort,
    episode,
    replicate_interval,
    theoretical_quantile,
    within_tolerance,
)


@pytest.mark.parametrize("rate", ARRIVAL_RATES)
@pytest.mark.parametrize("q", [0.95, 0.99])
def test_tail_target_agrees_with_geometric_erlang_mixture(rate, q):
    # Independently sum the stationary geometric mixture of Erlang survival functions.
    threshold = theoretical_quantile(rate, q)
    rho, x = rate / SERVICE_RATE, SERVICE_RATE * threshold
    poisson_term, erlang_survival, tail = math.exp(-x), 0.0, 0.0
    for n in range(1000):
        if n:
            poisson_term *= x / n
        erlang_survival += poisson_term
        tail += (1 - rho) * rho**n * erlang_survival
    assert tail == pytest.approx(1 - q, rel=1e-12, abs=1e-14)


def test_horizon_changes_only_drain_time_and_preserves_arrival_cutoff():
    a, b = episode(8.0, 4800.0), episode(8.0, 5000.0)
    import dataclasses

    assert dataclasses.replace(a, horizon_s=b.horizon_s) == b
    assert a.arrival_cutoff_s == b.arrival_cutoff_s == 4600.0


@pytest.mark.parametrize(
    "status,end", [("in_flight", None), ("timeout", 601.0), ("dropped_queue_full", 600.0)]
)
def test_loss_or_censoring_stays_in_offered_denominator(status, end):
    records = [
        SimpleNamespace(rid=1, arrival_s=600.0, end_s=end, status=status),
        SimpleNamespace(rid=2, arrival_s=601.0, end_s=601.1, status="completed"),
    ]
    result = cohort(records, (600.0, 4600.0), 8.0)
    assert result["offered"] == 2
    assert not result["all_offered_completed"]
    assert "tails" not in result


def test_independent_run_count_cannot_be_replaced_with_request_count():
    with pytest.raises(ValueError, match="64"):
        replicate_interval([0.05] * 20000)
    ci = replicate_interval([0.04, 0.06] * 32)
    assert ci["n_independent_runs"] == 64
    assert ci["lower"] < 0.05 < ci["upper"]


def test_interval_containing_target_does_not_establish_equivalence():
    assert not within_tolerance({"lower": 0, "upper": 0.1}, target=0.05, tolerance=0.005)
    assert within_tolerance({"lower": 0.047, "upper": 0.053}, target=0.05, tolerance=0.005)
