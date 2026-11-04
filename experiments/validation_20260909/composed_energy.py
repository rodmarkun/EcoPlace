"""Development witness: real actuation, startup, requests and closed energy boundary.

Hand arithmetic is specified independently of EnergyMeter. No acceptance fixtures,
held-out data, adaptive search or provider calls are used.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import sys
import tarfile
from pathlib import Path

from sim import BoundaryWeights, CallableScorer, Simulation
from sim.actuation import CallableProposer
from sim.spec import (
    ArrivalSpec,
    CallEdgeSpec,
    CallSpec,
    ClusterSpec,
    EnergyCoefficients,
    EpisodeConfig,
    NodeClassSpec,
    NodeSpec,
    PowerParams,
    RebalancerParams,
    SchedulerParams,
    ServiceSpec,
    TemplateSpec,
    WorkloadSpec,
    config_sha256,
    det,
    episode_config_to_json,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 9113512321037399043


def scenario(horizon: float) -> EpisodeConfig:
    cloud = NodeClassSpec(name='cloud', power_idle_w=50., power_max_w=100.,
                          image_pull_mib_per_s=2.)
    edge = NodeClassSpec(name='edge', power_idle_w=3., power_max_w=8.,
                         image_pull_mib_per_s=2., suspended_power_w=.5,
                         suspend_latency_s=2., suspend_energy_j=5.,
                         wake_latency_s=3., wake_energy_j=8.)
    cluster = ClusterSpec(
        classes=(cloud, edge),
        nodes=(NodeSpec(0, 'cloud', 0, 1000, 4096),
               NodeSpec(1, 'edge', 2, 1000, 4096)),
        latency_ms=((0., 100.), (100., 0.)),
    )
    services = tuple(ServiceSpec(
        id=i, name=f's{i}', app=0, cpu_request=500, cpu_demand=500,
        mem_request=128, service_time=det(float(i + 1)), image_mb=1,
        cold_start=det(.25), initial_replicas=1, min_replicas=1, max_replicas=1,
        activation_s=0. if i == 0 else 9.25,
    ) for i in range(2))
    template = TemplateSpec(
        id=0, name='single-blocking-request',
        root=CallSpec(service=0, children=(CallEdgeSpec(
            child=CallSpec(service=1), request_kb=2., response_kb=3.),)),
        arrival=ArrivalSpec('deterministic', 1., start_s=12., realized_times_s=(12.,),
                            arrivals_realized=True),
    )
    return EpisodeConfig(
        cluster=cluster, workload=WorkloadSpec(services=services, templates=(template,)),
        horizon_s=horizon, arrival_cutoff_s=13.,
        scheduler=SchedulerParams(bind_latency=det(.25)),
        power=PowerParams(suspend_after_idle_s=1.),
        rebalancer=RebalancerParams(enabled=True, window_s=6.),
        energy=EnergyCoefficients(
            version='composed-energy-development-witness-v1',
            cloud_j_per_vcpu_hour=3600., network_j_per_byte=1e-6,
            source='hand-specified synthetic arithmetic; no physical calibration',
        ),
    )


def expected(horizon: float) -> dict:
    if horizon not in (14., 20.):
        raise ValueError('only the two predeclared horizon cases are supported')
    # Empty edge: idle 0–1, suspending 1–3, suspended 3–6, waking 6–9,
    # awake 9–horizon. Child executes 13.1–15.1 at half utilization.
    # Transition windows use idle draw, with separate pulses of 5 + 8 J.
    busy = min(horizon, 15.1) - 13.1
    device = 3. * (horizon - 3.) + .5 * 3. + 5. + 8. + (8. - 3.) * .5 * busy
    # Cloud allocation (not busy CPU) is 0.5 vCPU from bind at t=.25.
    cloud = .5 * (horizon - .25) / 3600. * 3600.
    network_bytes = 2 * 1024**2 + 2000 + (3000 if horizon == 20. else 0)
    network = network_bytes * 1e-6
    return {'device_j': device, 'cloud_j': cloud, 'network_j': network,
            'network_bytes': network_bytes, 'transition_j': 13.,
            'closed_boundary_j': device + cloud + network,
            'request_status': 'completed' if horizon == 20. else 'in_flight',
            'request_latency_s': 3.2 if horizon == 20. else None,
            'cloud_bound_s': .25, 'cloud_ready_s': 1.,
            'edge_bound_s': 9.5, 'edge_ready_s': 10.25}


def run_case(horizon: float) -> dict:
    config = scenario(horizon)
    # Placement follows service identity, independently of any energy accounting.
    scorer = CallableScorer('force-service-to-node',
                             lambda pod, node, ctx: float(node['id'] == pod['service']))
    calls = 0

    def wake_once(state, graph, budget):
        nonlocal calls
        calls += 1
        return [{'kind': 1, 'pod': -1, 'dest': 1}] if calls == 1 else []

    simulation = Simulation(config, SEED, scorer, CallableProposer('wake-once', wake_once))
    states = []

    def observe():
        for at in (0.5, 1.5, 3.5, 6.5, 9.1, 10.5, 13.5):
            yield simulation.env.timeout(at - simulation.env.now)
            node = simulation.cluster.nodes[1]
            states.append({'t': at, 'power_state': node.power_state,
                           'pods': len(node.pods)})

    simulation.env.process(observe())
    result = simulation.run()
    energy = result.energy
    observed = {
        'device_j': energy.device_j, 'cloud_j': energy.cloud_j,
        'network_j': energy.network_j, 'network_bytes': energy.network_bytes,
        'transition_j': sum(energy.transition_j_per_node),
        'closed_boundary_j': energy.closed_boundary_j(BoundaryWeights())[0],
        'request_status': result.records[0].status,
        'request_latency_s': result.records[0].latency_s if result.records[0].terminal else None,
        'cloud_bound_s': result.pods[0].bound_s, 'cloud_ready_s': result.pods[0].ready_s,
        'edge_bound_s': result.pods[1].bound_s, 'edge_ready_s': result.pods[1].ready_s,
    }
    errors = []
    for key, target in expected(horizon).items():
        actual = observed[key]
        if isinstance(target, (float, int)):
            agrees = (isinstance(actual, (float, int))
                      and math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-10))
        else:
            agrees = actual == target
        if not agrees:
            errors.append(f'{key}: expected {target!r}, got {actual!r}')
    target_states = ['awake', 'suspending', 'suspended', 'waking', 'awake', 'awake', 'awake']
    if [item['power_state'] for item in states] != target_states:
        errors.append('power-state sequence differs')
    if len(result.records) != 1 or result.ledger['rebalancer_wakes_executed'] != 1:
        errors.append('request/wake count differs')
    if energy.device_j_per_node[0] != 0.:
        errors.append('cloud energy leaked into device category')
    return {'horizon_s': horizon, 'config_sha256': config_sha256(config),
            'passed': not errors, 'errors': errors, 'expected': expected(horizon),
            'observed': observed, 'states': states, 'ledger': result.ledger,
            'requests': [dataclasses.asdict(row) for row in result.records],
            'service_calls': [dataclasses.asdict(row) for row in result.service_calls]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), *sorted((ROOT / 'sim').rglob('*.py')),
             ROOT / 'analysis/tracking/seeds.py', ROOT / 'uv.lock']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    plan = {'schema': 'composed-energy-development-witness-v1', 'seed': SEED,
            'horizons': [14., 20.], 'source_sha256': hashes,
            'python': sys.version, 'platform': platform.platform(),
            'expected': {str(h): expected(h) for h in (14., 20.)},
            'boundary': 'Development deterministic composed accounting witness, no P1 gate, '
                        'physical measurements, held-out data or policy comparison.'}
    (args.out / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    with tarfile.open(args.out / 'sources.tar.gz', 'w:gz') as archive:
        for p in files:
            archive.add(p, arcname=p.relative_to(ROOT).as_posix())
    try:
        for h in plan['horizons']:
            (args.out / f'config-{int(h)}.json').write_text(
                episode_config_to_json(scenario(h)) + '\n')
        rows = [run_case(h) for h in plan['horizons']]
    except Exception as exc:
        (args.out / 'FAILURE.json').write_text(json.dumps(
            {'error_type': type(exc).__name__, 'error': str(exc)}, indent=2) + '\n')
        raise
    if hashes != {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in hashes}:
        raise ValueError('source drift during witness')
    result = {'complete': True, 'passed': all(row['passed'] for row in rows), 'rows': rows}
    (args.out / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
