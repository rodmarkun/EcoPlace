"""Structural checks of new synthetic designs; no policy scores or simulations."""

import math
from collections import Counter

import pytest

from evolution.evaluation import config_for_development_bundle
from experiments.validation_20260909.independent_scenarios import (
    DEADLINES,
    PRESSURES,
    TOPOLOGIES,
    counts,
    episode,
    packing_witness,
    profile_segments,
    seed_design,
    topology,
)
from sim.energy import build_development_bundles, load_pinned_sources
from sim.spec import config_sha256, episode_config_from_json, episode_config_to_json


@pytest.fixture(scope="module")
def bundles():
    return build_development_bundles(load_pinned_sources())


@pytest.mark.parametrize(
    "name,expected", [("distributed", {0: 3, 1: 6, 2: 9}), ("edge_heavy", {0: 2, 1: 4, 2: 12})]
)
def test_network_geometry_and_distinct_tier_proportions(name, expected, bundles):
    cluster, sites, links = topology(name, bundles)
    assert Counter(n.tier for n in cluster.nodes) == expected
    assert len(sites) == len(cluster.nodes) == 18
    assert len(links) == 18 + TOPOLOGIES[name]["sites"]
    d = cluster.latency_ms
    for i in range(18):
        assert d[i][i] == 0
        for j in range(18):
            assert d[i][j] == d[j][i]
            if i != j:
                assert d[i][j] > 0
            for k in range(18):
                assert d[i][j] <= d[i][k] + d[k][j]
    for i, a in enumerate(sites):
        for j, b in enumerate(sites):
            if i != j and a["site"] == b["site"] and a["site"] is not None:
                expected_delay = sum(1 if cluster.nodes[n].tier == 1 else 2 for n in (i, j))
                assert d[i][j] == expected_delay


@pytest.mark.parametrize("name", TOPOLOGIES)
@pytest.mark.parametrize("pressure", PRESSURES)
def test_equal_offered_cpu_work_and_burst_volume(name, pressure, bundles):
    steady = episode(name, "steady", pressure, bundles)
    bursts = episode(name, "two_bursts", pressure, bundles)
    edge_capacity = sum(n.cpu_capacity for n in steady.cluster.nodes if n.tier == 2)
    total_cpu_work = 0.0
    for s, b in zip(steady.workload.templates, bursts.workload.templates, strict=True):
        integral = s.arrival.integrated_rate(steady.cutoff_s)
        assert math.isclose(integral, b.arrival.integrated_rate(bursts.cutoff_s), rel_tol=1e-14)
        assert b.arrival.rate_at(300) / s.arrival.rate_at(300) == pytest.approx(2.5)
        assert b.arrival.rate_at(600) / s.arrival.rate_at(600) == pytest.approx(1.625)
        assert b.arrival.rate_at(400) / s.arrival.rate_at(400) == pytest.approx(0.75)
        for service_id, n in counts(s.root).items():
            service = steady.workload.services[service_id]
            params = service.service_time.params
            mean = math.exp(params["mu"] + params["sigma"] ** 2 / 2)
            total_cpu_work += integral * n * mean * service.cpu_demand
    assert total_cpu_work / (840 * edge_capacity) == pytest.approx(pressure)
    assert tuple(t.slo_latency_s for t in steady.workload.templates) == DEADLINES
    assert tuple(t.slo_latency_s for t in bursts.workload.templates) == DEADLINES
    assert config_sha256(episode_config_from_json(episode_config_to_json(steady))) == config_sha256(
        steady
    )


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_packing_and_bundle_invariants(name, bundles):
    base = episode(name, "two_bursts", 0.30, bundles)
    # Independently sum every assignment; witness must place each max-replica pod once.
    witness = packing_witness(base)
    assert len(witness) == 56
    assert len({(a["service"], a["replica"]) for a in witness}) == 56
    for bundle in bundles:
        config = config_for_development_bundle(base, bundle)
        assert config.workload == base.workload
        assert config.cluster.latency_ms == base.cluster.latency_ms
        assert config.hpa == base.hpa
        for node in config.cluster.nodes:
            assignments = [a for a in witness if a["node"] == node.id]
            assert (
                sum(config.workload.services[a["service"]].cpu_request for a in assignments)
                <= node.cpu_capacity
            )
            assert (
                sum(config.workload.services[a["service"]].mem_request for a in assignments)
                <= node.mem_capacity
            )
            if assignments:
                assert node.tier == 2


def test_declared_multiplicities_and_finite_grid(bundles):
    config = episode("distributed", "steady", 0.30, bundles)
    assert [dict(counts(t.root)) for t in config.workload.templates] == [
        {0: 1, 1: 1, 2: 1, 3: 1},
        {0: 1, 4: 1, 5: 1},
        {0: 1, 1: 1, 2: 1, 3: 2, 4: 1},
    ]
    with pytest.raises(ValueError, match="finite grid"):
        episode("distributed", "steady", 0.31, bundles)
    with pytest.raises(ValueError, match="unknown demand"):
        profile_segments("outcome_selected_burst")


def test_seeds_are_disjoint_and_final_grid_not_invented():
    design = seed_design()
    roles = design["roles"]
    seeds = [s for values in roles.values() for s in values]
    assert len(set(seeds)) == len(seeds) == 4
    assert not set(seeds) & set(design["historical_inventory"]["seeds"])
    assert design["search_and_final_seeds"] is None
