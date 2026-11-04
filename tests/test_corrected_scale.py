"""Scale must preserve offered demand/capacity ratios and autoscaler opportunities."""
from __future__ import annotations

from collections import Counter

import pytest

from experiments.validation_20260909.scale import scale_episode, scaled_nodes
from sim.scenarios.dev_tight import DEV_CONTINUUM_TIGHT_PATH
from sim.spec import config_sha256, load_episode_config


def test_factor_one_is_exact_identity():
    assert config_sha256(scale_episode(1)) == config_sha256(
        load_episode_config(DEV_CONTINUUM_TIGHT_PATH)
    )


@pytest.mark.parametrize('factor,total,cloud', [(2, 30, 4), (4, 60, 8)])
def test_all_tiers_and_density_scale_together(factor, total, cloud):
    base, scaled = scale_episode(1), scale_episode(factor)
    assert len(scaled.cluster.nodes) == total
    assert sum(n.tier == 0 for n in scaled.cluster.nodes) == cloud
    original_counts = Counter(
        (n.node_class, n.tier, n.cpu_capacity, n.mem_capacity) for n in base.cluster.nodes
    )
    scaled_counts = Counter(
        (n.node_class, n.tier, n.cpu_capacity, n.mem_capacity) for n in scaled.cluster.nodes
    )
    assert scaled_counts == Counter({key: count * factor for key, count in original_counts.items()})
    assert len({row[5] for row in scaled_nodes(factor) if row[5] >= 0}) == 4 * factor
    def initial_cpu(workload):
        return sum(s.cpu_request * s.initial_replicas for s in workload.services)

    assert initial_cpu(scaled.workload) == factor * initial_cpu(base.workload)
    for old, new in zip(base.workload.services, scaled.workload.services, strict=True):
        assert new.max_replicas == factor * old.max_replicas
        assert new.min_replicas == factor * old.min_replicas
        assert new.queue_capacity == old.queue_capacity
        assert new.service_time == old.service_time
    for old, new in zip(base.workload.templates, scaled.workload.templates, strict=True):
        assert new.root == old.root
        assert new.slo_latency_s == old.slo_latency_s
        assert new.arrival.rate_per_s == factor * old.arrival.rate_per_s
        assert new.arrival.rate_schedule == tuple(
            (at, rate * factor) for at, rate in old.arrival.rate_schedule
        )
    assert scaled.hpa == base.hpa
    assert scaled.rebalancer.budget_per_window == base.rebalancer.budget_per_window * factor
    assert scaled.rebalancer.max_total_moves == base.rebalancer.max_total_moves * factor
    assert scaled.horizon_s == base.horizon_s


def test_noninteger_factor_is_rejected():
    with pytest.raises(ValueError):
        scale_episode(True)
