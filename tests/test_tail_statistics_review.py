"""Adversarial checks for the separate post-completion arithmetic review."""
import math

import pytest

from experiments.validation_20260909.verify_tail_statistics import cohort, interval


def test_censored_request_keeps_cohort_invalid():
    rows = [[0, 600.0, 600.1, "completed"], [1, 601.0, None, "in_flight"]]
    result = cohort(rows, (600, 4600), 8.0)
    assert result["offered"] == 2
    assert result["all_offered_completed"] is False
    assert "tails" not in result


def test_window_boundaries_and_inverse_ecdf():
    rows = [[i, 600.0, 600.0 + i / 10, "completed"] for i in range(20)]
    rows += [[20, 4600.0, 4600.1, "completed"]]
    result = cohort(rows, (600, 4600), 8.0)
    assert result["offered"] == 20
    assert result["tails"][0]["observed_inverse_ecdf_quantile_s"] == pytest.approx(1.8)
    assert result["tails"][1]["observed_inverse_ecdf_quantile_s"] == pytest.approx(1.9)
    assert result["tails"][0]["theoretical_quantile_s"] == pytest.approx(math.log(20) / 4.5)


def test_equal_weight_runs_and_sample_variance():
    values = [0.0] * 32 + [1.0] * 32
    result = interval(values)
    assert result["mean"] == 0.5
    assert result["standard_error"] == pytest.approx(math.sqrt(16 / 63 / 64))
    assert result["critical_t"] > 2.7
    with pytest.raises(ValueError, match="64 finite"):
        interval(values * 2)
    with pytest.raises(ValueError, match="64 finite"):
        interval([float("nan")] + values[1:])
