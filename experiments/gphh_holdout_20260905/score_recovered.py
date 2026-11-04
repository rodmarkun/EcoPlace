"""Score verified historical GPHH trees with bounded, checkpointed child processes."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import psutil
from deap import gp

from baselines.gphh import gphh_bundle, primitive_set
from evolution.evaluation import load_task_evaluation_config
from experiments.allbaselines_harness.runner import _compact_evaluation
from experiments.gp_tuning_20260902 import analyze
from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import evaluation_snapshot, write_sealed

from .recover_trees import OUT as RECOVERED
from .recover_trees import ROOT

OUT = Path(__file__).parent / 'validated_20260909'


def snapshot() -> dict:
    result = evaluation_snapshot()
    extras = [*Path(__file__).parent.glob('*.py'),
              *(ROOT / 'experiments/allbaselines_harness').glob('*.py'),
              *(ROOT / 'experiments/deep_search_20260825').glob('*.py'),
              *(ROOT / 'experiments/transfer_20260825').glob('*.py')]
    for path in sorted(extras):
        result['files'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def worker(request_path: Path, output: Path) -> None:
    request = read_sealed(request_path)
    if digest(snapshot()) != request['snapshot_sha256']:
        raise ValueError('evaluation source/environment changed before scoring')
    if hashlib.sha256(request['expression'].encode()).hexdigest() != request['tree_sha256']:
        raise ValueError('tree expression does not match request digest')
    if hashlib.sha256(analyze.TASK_CONFIG.read_bytes()).hexdigest() != request['task_sha256']:
        raise ValueError('evaluation task differs from request')
    pset = primitive_set()
    tree = gp.PrimitiveTree.from_string(request['expression'], pset)
    if str(tree) != request['expression']:
        raise ValueError('tree reconstruction changes expression')
    task = load_task_evaluation_config(analyze.TASK_CONFIG)
    if len(task.scenarios) != 1:
        raise ValueError('this historical comparison expects one scenario')
    seeds = list(analyze.fresh_grid(request['root_seed'], task.scenarios[0].scenario_id))
    if seeds != request['fresh_seeds']:
        raise ValueError('fresh evaluation seed namespace changed')
    task = dataclasses.replace(task, scenarios=(
        dataclasses.replace(task.scenarios[0], seeds=tuple(seeds)),
    ))
    bundle = gphh_bundle(tree, pset)
    if bundle.artifact_sha256 != request['tree_sha256']:
        raise ValueError('adapter artifact identity differs from request')
    evaluation = bundle.evaluate(task, proposal_seq=970000 + request['root_seed'] % 1000)
    compact = _compact_evaluation(evaluation)
    if digest(snapshot()) != request['snapshot_sha256']:
        raise ValueError('evaluation source/environment changed during scoring')
    result = {'fresh_grid_feasible': evaluation.fitness.feasible,
              'fresh_grid_energy': evaluation.fitness.normalized_energy,
              'fresh_grid_worst_slo': min(
                  min(i.slo_achieved.values()) for i in evaluation.instances
              ),
              'fresh_seeds': seeds, 'evaluation': compact}
    write_sealed(output, {'request_sha256': digest(request), 'result': result})


def guarded_score(request_path: Path, output: Path, *, max_rss_mb=2048, timeout_seconds=1200):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'experiments.gphh_holdout_20260905.score_recovered',
               '--worker', str(request_path), str(output)]
    start, peak = time.monotonic(), 0
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            observed = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, observed.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak > max_rss_mb * 1024**2:
                    raise RuntimeError(f'GPHH evaluator exceeded {max_rss_mb} MiB RSS')
                if time.monotonic() - start > timeout_seconds:
                    raise RuntimeError(f'GPHH evaluator exceeded {timeout_seconds} seconds')
                time.sleep(1)
            code = process.wait()
            if code:
                raise RuntimeError(f'GPHH evaluator failed ({code}); see {log.name}')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return {'peak_rss_bytes': peak, 'elapsed_seconds': time.monotonic() - start,
            'rss_limit_mib': max_rss_mb, 'timeout_seconds': timeout_seconds}


def run(out: Path, *, limit_requests: int = 0) -> None:
    current = snapshot()
    sha = digest(current)
    snapshot_path = out / 'evaluation_snapshot.json'
    if snapshot_path.exists() and digest(read_sealed(snapshot_path)) != sha:
        raise ValueError('source/environment drift; use a new output directory')
    write_sealed(snapshot_path, current)
    archive = out / 'evaluation_sources.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as bundle:
            for name in sorted(current['files']):
                bundle.add(ROOT / name, arcname=name)
            for index, name in enumerate(current['native']):
                bundle.add(name, arcname=f'native/{index}-{Path(name).name}')
    index = read_sealed(RECOVERED / 'INDEX.json')
    if [row['root_seed'] for row in index['roots']] != list(range(800000801, 800000821)):
        raise ValueError('historical recovery root grid differs')
    index_path = out / 'inputs/recovery_index.json'
    if index_path.exists() and read_sealed(index_path) != index:
        raise ValueError('historical recovery index changed; refuse mixed output')
    write_sealed(index_path, index)
    results = {'schema': 'evolved-continuum-recovered-gphh-holdout-v1', 'complete': False,
               'snapshot_sha256': sha, 'recovery_index_sha256': digest(index), 'rows': [],
               'boundary': 'Retrospective development comparison; historical champions and '
                           'generation-zero bests on shared fresh grids. '
                           'No new search or LLM calls.'}
    launched = 0
    for row in index['roots']:
        root = row['root_seed']
        recovery = read_sealed(RECOVERED / f'{root}.json')
        if digest(recovery) != row['payload_sha256']:
            raise ValueError('recovered trees changed after indexing')
        write_sealed(out / 'inputs' / f'{root}.json', recovery)
        for which, field in (('champion', 'champion_proposal_seq'),
                             ('gen0_best', 'gen0_best_proposal_seq')):
            tree = recovery['trees'][recovery[field]]
            request = {'root_seed': root, 'expression': tree['expression'],
                       'tree_sha256': tree['sha256'], 'snapshot_sha256': sha,
                       'task_sha256': current['files'][str(analyze.TASK_CONFIG.relative_to(ROOT))],
                       'fresh_seeds': list(analyze.fresh_grid(root, analyze.SCENARIO))}
            request_sha = digest(request)
            score_path = out / 'scores' / f'{request_sha}.json'
            if not score_path.exists():
                if limit_requests and launched >= limit_requests:
                    write_sealed(out / 'RESULTS.json', results)
                    return
                request_path = out / 'requests' / f'{request_sha}.json'
                write_sealed(request_path, request)
                print(f'scoring {root} {which}', flush=True)
                usage = guarded_score(request_path, score_path)
                write_sealed(out / 'usage' / f'{request_sha}.json', usage)
                launched += 1
                print(f"scored {root} {which}, peak RSS {usage['peak_rss_bytes']/1024**2:.1f} MiB",
                      flush=True)
            score = read_sealed(score_path)
            if score['request_sha256'] != request_sha:
                raise ValueError('score/request binding differs')
            if score['result']['fresh_seeds'] != request['fresh_seeds']:
                raise ValueError('score seed grid differs from request')
            results['rows'].append({'root_seed': root, 'which': which,
                                    'candidate_id': tree['candidate_id'],
                                    'request_sha256': request_sha, **score['result']})
            write_sealed(out / 'RESULTS.json', results)
    results['complete'] = len(results['rows']) == 40
    write_sealed(out / 'RESULTS.json', results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--limit-requests', type=int, default=0)
    parser.add_argument('--worker', nargs=2, type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'process.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.out, limit_requests=args.limit_requests)
    print(json.dumps({'complete': read_sealed(args.out / 'RESULTS.json')['complete']}))


if __name__ == '__main__':
    main()
