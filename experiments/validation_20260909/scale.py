"""Construct corrected density-matched scale probes without editing sealed scenarios.

Construction only. No policy comparison, calibration outcome or generalization claim.
The old 28/54-node probes and their results remain historical artifacts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from experiments.gp_tuning_20260902.run_recovered import write_sealed
from sim.scenarios.dev_tight import _TIGHT_NODES, DEV_CONTINUUM_TIGHT_PATH, _cluster
from sim.spec import EpisodeConfig, config_sha256, load_episode_config, save_episode_config

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'scale_construction'
TASK = ROOT / 'evolution/task/config_dev_continuum_tight_calibrated.json'


def scaled_nodes(factor: int):
    if type(factor) is not int or factor not in (1, 2, 4):
        raise ValueError('scale factor must be 1, 2 or 4')
    site_count = len({row[5] for row in _TIGHT_NODES if row[5] >= 0})
    nodes = []
    # Interleave copies to avoid grouping one entire cluster behind stable-ID ties.
    for _node_id, name, tier, cpu, mem, site in _TIGHT_NODES:
        for copy in range(factor):
            nodes.append((len(nodes), name, tier, cpu, mem,
                          site if site < 0 else site + copy * site_count))
    return tuple(nodes)


def scale_episode(factor: int) -> EpisodeConfig:
    nodes = scaled_nodes(factor)
    base = load_episode_config(DEV_CONTINUUM_TIGHT_PATH)
    if base.workload.backends or base.node_churn or base.rebalancer.excluded_pods:
        raise ValueError('this scale transform requires the unchanged tight base semantics')
    services = tuple(dataclasses.replace(
        service, initial_replicas=service.initial_replicas * factor,
        min_replicas=service.min_replicas * factor, max_replicas=service.max_replicas * factor,
    ) for service in base.workload.services)
    templates = []
    for template in base.workload.templates:
        arrival = template.arrival
        if arrival.arrivals_realized or arrival.realized_times_s:
            raise ValueError('do not scale a finite realized-arrival sample as an arrival process')
        templates.append(dataclasses.replace(template, arrival=dataclasses.replace(
            arrival, rate_per_s=arrival.rate_per_s * factor,
            rate_schedule=tuple((at, rate * factor) for at, rate in arrival.rate_schedule),
        )))
    rebalancer = base.rebalancer
    caps = {field: None if getattr(rebalancer, field) is None
            else getattr(rebalancer, field) * factor
            for field in ('max_total_moves', 'max_total_bytes_mb', 'max_total_unavailability_s')}
    return dataclasses.replace(
        base, cluster=_cluster(nodes),
        workload=dataclasses.replace(base.workload, services=services, templates=tuple(templates)),
        rebalancer=dataclasses.replace(
            rebalancer, budget_per_window=rebalancer.budget_per_window * factor, **caps
        ),
    )


def scale_task(factor: int, *, episode_path: Path, seeds: tuple[int, ...]) -> dict:
    """Bind an explicitly chosen grid; no implicit reuse of old evaluation seeds."""
    scaled_nodes(factor)  # Validate the same factor domain as the episode transform.
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)):
        raise ValueError('provide a nonempty, unique, explicit evaluation seed grid')
    path = episode_path.resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('scaled episode must live in the separate construction directory')
    if config_sha256(load_episode_config(path)) != config_sha256(scale_episode(factor)):
        raise ValueError('task episode differs from the corrected scale construction')
    raw = json.loads(TASK.read_text())
    scenario_id = f'dev-continuum-tight-density-x{factor}'
    raw['task_variant'] = scenario_id
    raw['task_variant_note'] = (
        'Corrected proportional scale probe; retained home SLO targets, not new calibration. '
        'Total disruption caps and network-byte boundary scale with workload and capacity. '
        'The caller must supply the grid from its prospective calibration/validation protocol.'
    )
    raw['scenarios'] = [
        {'id': scenario_id, 'path': str(path.relative_to(ROOT)), 'seeds': list(seeds)}
    ]
    raw['ladder']['l1']['scenario_ids'] = [scenario_id]
    raw['constraints']['disruption_caps'] = {
        key: value * factor for key, value in raw['constraints']['disruption_caps'].items()
    }
    raw['boundary_bounds']['network_bytes_bound'] *= factor
    return raw


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = scale_episode(1)
    if config_sha256(base) != config_sha256(load_episode_config(DEV_CONTINUUM_TIGHT_PATH)):
        raise ValueError('factor-one transform must preserve the base episode exactly')
    rows = []
    for factor in (2, 4):
        episode = scale_episode(factor)
        path = OUT / f'tight-density-x{factor}.json'
        save_episode_config(episode, path)
        rows.append({'factor': factor, 'path': str(path.relative_to(ROOT)),
                     'config_sha256': config_sha256(episode),
                     'node_counts_by_tier': {
                         str(tier): sum(n.tier == tier for n in episode.cluster.nodes)
                         for tier in (0, 1, 2)
                     },
                     'initial_replicas': sum(s.initial_replicas for s in episode.workload.services),
                     'max_replicas': sum(s.max_replicas for s in episode.workload.services)})
    write_sealed(OUT / 'MANIFEST.json', {
        'schema': 'evolved-continuum-corrected-scale-construction-v1',
        'base_config_sha256': config_sha256(base),
        'sources_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), DEV_CONTINUUM_TIGHT_PATH, TASK,
                         ROOT / 'sim/scenarios/dev_tight.py', ROOT / 'sim/spec.py')
        },
        'variants': rows,
        'boundary': 'Structural construction only; no calibration or policy comparison. '
                    'Shared service pools have scaled min/initial/max replicas; arrival intensity, '
                    'all node tiers and per-window/episode disruption allowances scale together. '
                    'Service-time distributions, per-replica queues and '
                    'HPA utilization/timing stay fixed.',
    })
    print(rows)


if __name__ == '__main__':
    main()
