"""Bounded real-simulator initialization witness; zero adaptive search or API calls."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import psutil

from analysis.tracking import derive_seed
from evolution.campaign import _default_evaluate, load_campaign_plan
from evolution.evaluation import load_task_evaluation_config
from evolution.feedback import load_feedback_profiles
from evolution.islands import load_islands_config
from evolution.ladder import load_ladder_config
from experiments.allbaselines_harness.initialization import read_seed_archive
from experiments.gp_tuning_20260902.recovery import canonical, digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.gphh_holdout_20260905.score_recovered import snapshot as evaluation_snapshot

from .runner import InitializationPending, _jsonable, certify_pair, matched_campaign_runner

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / 'real_pilot_20260909'
TASK = ROOT / 'evolution/task/config_dev_continuum_tight_calibrated.json'
INITIAL = ROOT / 'evolution/task/initial_tight.py'
SEED_ARCHIVE = next((ROOT / 'experiments/deep_search_20260825/runs_final').glob(
    '*--seed-800000801--*/inputs/seed_archive.json'))
PILOT_ROOT = derive_seed(20260909, 'matched-control-real-initialization-pilot')


def historical_seed_inventory() -> dict:
    manifests = sorted((ROOT / 'experiments/deep_search_20260825/runs_final').glob(
        '*/archives/*/manifest.json'))
    seeds = set()
    for path in manifests:
        raw = json.loads(path.read_text())
        for grid in raw.get('campaign', {}).get('evaluation_seeds', {}).values():
            seeds.update(grid)
    finalists = ROOT / 'experiments/deep_search_20260825/FINALIST_RESCORE_runs_final.json'
    for row in read_sealed(finalists)['rows']:
        seeds.update(row['fresh_seeds'])
    for scenario in load_task_evaluation_config(TASK).scenarios:
        seeds.update(scenario.seeds)
    return {'seeds': sorted(seeds), 'files': {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*manifests, finalists]
    }}


def snapshot() -> dict:
    state = evaluation_snapshot()
    for path in [*HERE.glob('*.py'),
                 *(ROOT / 'experiments/matched_transport_20260909').glob('*.py'),
                 ROOT / 'evolution/campaigns.json', INITIAL, SEED_ARCHIVE]:
        state['files'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = historical_seed_inventory()
    state['files'].update(inventory['files'])
    state['historical_evaluation_seeds'] = inventory['seeds']
    # Package inventory contains tuples; normalize once to its on-disk JSON representation.
    return json.loads(canonical(state))


class ForbiddenOperator:
    name = 'forbidden-initialization-pilot-operator'

    def __init__(self, out: Path):
        self.out = out

    def __call__(self, *args, **kwargs):
        with (self.out / 'UNEXPECTED_OPERATOR_CALL').open('a') as stream:
            stream.write('adaptive dispatch attempted\n')
        raise AssertionError('initialization must not call a proposal operator')


def pilot_plan():
    plan = load_campaign_plan(ROOT / 'evolution/campaigns.json')
    definition = dataclasses.replace(plan.campaigns[0], campaign_id='real-init-pilot',
                                     root_seed=PILOT_ROOT, evaluation_replicates=1)
    # Preserve the plan's ten-definition invariant; only the explicit pilot definition runs.
    plan = dataclasses.replace(plan, campaigns=(definition, *plan.campaigns[1:]),
                               candidate_budget=13, proposals_per_call=1)
    return definition, plan


def worker(out: Path, arm: str, interrupt: bool) -> int:
    frozen = read_sealed(out / 'snapshot.json')
    if snapshot() != frozen:
        raise ValueError('pilot source/environment changed')
    seed_archive = read_seed_archive(SEED_ARCHIVE, expected_size=12, initial_path=INITIAL)
    if len(set(seed_archive['sources'])) != 12:
        raise ValueError('pilot requires twelve distinct real initialization policies')
    definition, plan = pilot_plan()
    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(source, task, ladder, proposal_seq):
        actual_seeds = {seed for scenario in task.scenarios for seed in scenario.seeds}
        if actual_seeds.intersection(frozen['historical_evaluation_seeds']):
            raise ValueError('pilot evaluation overlaps a historical search or finalist grid')
        if interrupt and proposal_seq == 5:
            write_sealed(out / 'interruption.json', {
                'arm': arm, 'proposal_seq': proposal_seq,
                'point': 'after durable allocation, before evaluator invocation',
            })
            os._exit(17)  # Deliberate process interruption, not a caught evaluator exception.
        request = {'proposal_seq': proposal_seq,
                   'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                   'task_sha256': digest(_jsonable(task)),
                   'ladder_sha256': digest(_jsonable(ladder))}
        with (arm_dir / 'evaluation_calls.jsonl').open('a') as stream:
            stream.write(canonical(request) + '\n')
            stream.flush()
        return _default_evaluate(source, task, ladder, proposal_seq)

    runner = matched_campaign_runner(seed_archive, out / 'pair.json')(
        definition=definition, plan=plan, arm=arm, operator=ForbiddenOperator(out),
        initial_source=INITIAL.read_text(), task=load_task_evaluation_config(TASK),
        ladder=load_ladder_config(TASK), islands=load_islands_config(TASK),
        feedback_profile=load_feedback_profiles(TASK)[plan.feedback_profile],
        archive_dir=arm_dir, provenance={'pilot_snapshot_sha256': digest(frozen)},
        run_purpose='determinism-smoke', evaluation_seed_mode='campaign-derived',
        evaluate_fn=evaluate,
    )
    try:
        runner.run()
    except InitializationPending:
        if snapshot() != frozen:
            raise ValueError('pilot source/environment changed during evaluation') from None
        return 10
    raise ValueError('pilot must stop at the paired initialization barrier')


def guarded(out: Path, arm: str, *, interrupt: bool, expected_exit: int) -> dict:
    name = f'{arm}-' + ('interrupt' if interrupt else 'initialize-or-resume')
    command = [sys.executable, '-m', 'experiments.matched_control_20260909.real_pilot',
               '--out', str(out), '--worker', arm]
    if interrupt:
        command.append('--interrupt')
    start, peak = time.monotonic(), 0
    with (out / f'{name}.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            observed = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    peak = max(peak, observed.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak > 2048 * 1024**2 or time.monotonic() - start > 1200:
                    raise RuntimeError('pilot exceeded 2 GiB RSS or 1200-second worker bound')
                time.sleep(1)
            code = child.wait()
            if code != expected_exit:
                raise RuntimeError(f'pilot exit {code}, expected {expected_exit}; see {log.name}')
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
    result = {'arm': arm, 'interrupt': interrupt, 'exit_code': code,
              'peak_rss_bytes': peak, 'elapsed_seconds': time.monotonic() - start,
              'rss_limit_mib': 2048, 'timeout_seconds': 1200}
    write_sealed(out / f'{name}.json', result)
    return result


def run(out: Path) -> None:
    # Never accidentally continue past a previously certified barrier.
    if out.exists():
        raise ValueError('pilot output already exists; inspect it before any explicit recovery')
    out.mkdir(parents=True)
    frozen = snapshot()
    write_sealed(out / 'snapshot.json', frozen)
    with tarfile.open(out / 'sources.tar.gz', 'w:gz') as archive:
        for name in sorted(frozen['files']):
            archive.add(ROOT / name, arcname=name)
        for i, name in enumerate(frozen['native']):
            archive.add(name, arcname=f'native/{i}-{Path(name).name}')
    usage = []
    for arm, interrupted, code in (('gp', False, 10), ('llm', True, 17), ('llm', False, 10)):
        print(f'{arm}: ' + ('interrupt before candidate 5' if interrupted else 'reach barrier'),
              flush=True)
        usage.append(guarded(out, arm, interrupt=interrupted, expected_exit=code))
    pair = certify_pair(out / 'gp', out / 'llm', out / 'pair.json')
    calls = [(out / arm / 'evaluation_calls.jsonl').read_text() for arm in ('gp', 'llm')]
    if calls[0] != calls[1] or len(calls[0].splitlines()) != 12:
        raise ValueError('real evaluator calls differ or interruption repeated a completed call')
    if (out / 'UNEXPECTED_OPERATOR_CALL').exists():
        raise ValueError('initialization dispatched a proposal')
    if snapshot() != frozen:
        raise ValueError('pilot source/environment changed')
    seeds = {s for receipt in pair['arms'] for values in
             receipt['context']['effective_campaign']['evaluation_seeds'].values() for s in values}
    write_sealed(out / 'RESULT.json', {
        'schema': 'evolved-continuum-real-initialization-pilot-v1', 'complete': True,
        'pilot_root_seed': PILOT_ROOT, 'evaluation_seeds': sorted(seeds),
        'snapshot_sha256': digest(frozen), 'pair_receipt_sha256': digest(pair),
        'distinct_sources': 12, 'real_evaluations_per_arm': 12, 'proposal_calls': 0,
        'interruption_recovered': True, 'exact_evaluator_inputs_match': True,
        'disjoint_from_historical_evaluation_seeds': True,
        'initialization_records_match': True, 'usage': usage,
        'boundary': 'Real simulator initialization/replay witness; no adaptive search, '
                    'no API calls, no scientific or cost-journal certification.',
    })
    print('real initialization pair verified; zero proposal calls', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--worker', choices=('gp', 'llm'))
    parser.add_argument('--interrupt', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args.out, args.worker, args.interrupt)
    run(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
