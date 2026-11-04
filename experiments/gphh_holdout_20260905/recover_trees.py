"""Recover historical GPHH expressions through digest-checked proposal replay.

Archived evaluations drive selection and work accounting. No simulator or model is
called, and no fitness is recomputed. Every generated tree must match its historical
digest before that candidate's recorded evaluation can be consumed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from analysis.tracking import derive_seed
from baselines.gphh import ClassicGPHH, GPHHConfig
from evolution.campaign import load_campaign_plan
from evolution.evaluation import PolicyEvaluation, load_task_evaluation_config
from evolution.ladder import LadderResult, load_ladder_config
from experiments.allbaselines_harness.summary import fitness_from_record
from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'experiments/deep_search_20260825/runs_final'
OUT = Path(__file__).parent / 'recovered_20260909'


def recorded_ladder(raw: dict) -> LadderResult:
    """Only identity, fitness, promotion and trace are needed for proposal replay."""
    identity = raw['policy']
    fields = {'policy_name': identity['name'], 'policy_version': identity['version'],
              'policy_identity_sha256': identity['identity_sha256'],
              'policy_artifact_sha256': identity['artifact_sha256']}
    evaluation = raw['evaluation']
    if evaluation is not None and evaluation['policy'] != identity:
        raise ValueError('recorded evaluation identity differs from ladder identity')
    return LadderResult(
        promoted_to_l2=raw['promoted_to_l2'], trace=tuple(raw['trace']),
        budget_registration_status=raw['budget_registration_status'],
        budget_registration_source=raw['budget_registration_source'],
        candidate_failure=raw['candidate_failure'],
        evaluation=None if evaluation is None else PolicyEvaluation(
            fitness=fitness_from_record(evaluation['fitness']), instances=(), **fields
        ), **fields,
    )


def recover(run_dir: Path) -> dict:
    result_path = run_dir / 'arms/classic_gphh/result.json'
    payload = read_sealed(result_path)
    preflight = read_sealed(run_dir / 'preflight.json')
    original = payload['run']
    if digest({k: v for k, v in original.items() if k != 'sha256'}) != original['sha256']:
        raise ValueError('inner GPHH run seal differs')
    if payload['arm'] != 'classic_gphh' or payload['status'] != 'DEVELOPMENT_ADAPTIVE_COMPLETE':
        raise ValueError('expected a completed development GPHH arm')
    for reference in payload['references']:
        path = (run_dir / reference['path']).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError('arm input reference escapes run directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError(f"GPHH input changed: {reference['path']}")
    for name in ('task_config', 'campaign_config'):
        actual = hashlib.sha256((run_dir / f'inputs/{name}.json').read_bytes()).hexdigest()
        if actual != preflight['inputs'][f'{name}_sha256']:
            raise ValueError(f'GPHH input differs from preflight: {name}')
    root = preflight['root_seed']
    expected_seed = derive_seed(root, 'allbaselines', 'arm', 'classic-gphh')
    if original['root_seed'] != expected_seed:
        raise ValueError('GPHH root seed differs from paired study root')
    # Require the recorded cap explicitly; a current default must never rewrite history.
    if 'max_tree_height' not in original['config']:
        raise ValueError('historical tree-height semantics require explicit reconstruction')
    config = GPHHConfig(**original['config'])
    plan = load_campaign_plan(run_dir / 'inputs/campaign_config.json')
    task = load_task_evaluation_config(run_dir / 'inputs/task_config.json')
    ladder = load_ladder_config(run_dir / 'inputs/task_config.json')
    work = plan.evaluator_work
    if (work.max_microwork != preflight['matched_max_microwork']
            or work.rung_microwork != preflight['rung_microwork']):
        raise ValueError('GPHH work configuration differs from preflight')
    recorded = original['candidates']
    visited = []

    def replay(bundle, task, ladder, proposal_seq):
        del task, ladder
        if proposal_seq != len(visited) or proposal_seq >= len(recorded):
            raise ValueError('replay candidate sequence differs')
        row = recorded[proposal_seq]
        identity = {'name': bundle.name, 'version': bundle.version,
                    'identity_sha256': bundle.identity_sha256,
                    'artifact_sha256': bundle.artifact_sha256}
        if (row['proposal_seq'] != proposal_seq
                or row['candidate_id'] != f'classic_gphh:{proposal_seq}'
                or row['ladder']['policy'] != identity
                or row['representation'] != {'kind': 'deap-primitive-tree',
                                              'sha256': bundle.artifact_sha256}):
            raise ValueError(f'GPHH tree/identity replay mismatch at candidate {proposal_seq}')
        visited.append(proposal_seq)
        return recorded_ladder(row['ladder'])

    replayed = ClassicGPHH(config, evaluate_fn=replay).run(
        task, ladder, work, root_seed=expected_seed
    )
    if len(replayed.candidates) != len(recorded):
        raise ValueError('replay did not recover every historical candidate')
    trees = []
    for candidate, row in zip(replayed.candidates, recorded, strict=True):
        expression = str(candidate.tree)
        for field in ('proposal_seq', 'generation', 'candidate_microwork', 'cumulative_microwork'):
            if getattr(candidate, field) != row[field]:
                raise ValueError(f'GPHH replay differs: {field}')
        tree_sha = hashlib.sha256(expression.encode()).hexdigest()
        if (list(candidate.selection_key) != row['selection_key']
                or tree_sha != row['representation']['sha256']):
            raise ValueError('GPHH replay selection key or final expression differs')
        trees.append({'candidate_id': row['candidate_id'], 'proposal_seq': candidate.proposal_seq,
                      'generation': candidate.generation, 'expression': expression,
                      'sha256': row['representation']['sha256']})
    if ([list(generation) for generation in replayed.population_proposal_seqs_by_generation]
            != original['population_proposal_seqs_by_generation']
            or replayed.spent_microwork != original['spent_microwork']
            or replayed.stop_reason != original['stop_reason']):
        raise ValueError('GPHH replay population history or termination differs')
    champion_seq = None if replayed.champion is None else replayed.champion.proposal_seq
    if champion_seq != original['champion_proposal_seq']:
        raise ValueError('GPHH replay champion differs')
    if (payload['accounting']['actual_microwork'] != replayed.spent_microwork
            or payload['accounting']['registered_max_microwork'] != work.max_microwork
            or payload['accounting']['effective_terminal_microwork'] != work.max_microwork
            or replayed.stop_reason != 'normalized-work-budget-exhausted'):
        raise ValueError('GPHH arm work accounting differs')
    champion_fitness = (None if champion_seq is None
                        else recorded[champion_seq]['ladder']['evaluation']['fitness'])
    if (champion_seq is None or payload['champion'] != original['champion']
            or payload['champion']['proposal_seq'] != champion_seq
            or payload['champion']['fitness'] != champion_fitness):
        raise ValueError('GPHH arm champion summary differs')
    champion_record = recorded[champion_seq]
    expected_champion = {
        'candidate_id': champion_record['candidate_id'], 'proposal_seq': champion_seq,
        'generation': champion_record['generation'],
        'identity': champion_record['ladder']['policy'],
        'fitness': champion_fitness,
        'energies': champion_record['ladder']['evaluation']['energies'],
    }
    if payload['champion'] != expected_champion:
        raise ValueError('GPHH champion summary does not match selected candidate')
    gen0 = min((candidate for candidate in replayed.candidates if candidate.generation == 0),
               key=lambda candidate: candidate.selection_key)
    return {'schema': 'evolved-continuum-recovered-gphh-trees-v1', 'root_seed': root,
            'run': str(run_dir.resolve()), 'original_result_payload_sha256': digest(payload),
            'preflight_payload_sha256': digest(preflight), 'recorded_config': original['config'],
            'champion_proposal_seq': champion_seq, 'gen0_best_proposal_seq': gen0.proposal_seq,
            'trees': trees, 'boundary': 'Historical proposal replay using recorded evaluations; '
            'every tree identity, selection key, work charge and population history verified. '
            'No simulation rerun, new search, or historical source-timestamp claim.'}


def main() -> None:
    receipts = []
    for root in range(800000801, 800000821):
        runs = list(RUNS.glob(f'*--seed-{root}--*'))
        if len(runs) != 1:
            raise ValueError(f'expected one historical run for root {root}')
        recovered = recover(runs[0])
        write_sealed(OUT / f'{root}.json', recovered)
        receipts.append({'root_seed': root, 'trees': len(recovered['trees']),
                         'payload_sha256': digest(recovered),
                         'champion_proposal_seq': recovered['champion_proposal_seq'],
                         'gen0_best_proposal_seq': recovered['gen0_best_proposal_seq']})
        print(receipts[-1], flush=True)
    write_sealed(OUT / 'INDEX.json', {
        'schema': 'evolved-continuum-gphh-recovery-index-v1',
        'roots': receipts, 'total_trees': sum(r['trees'] for r in receipts),
        'replay_source_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT / 'baselines/gphh.py',
                         ROOT / 'evolution/campaign.py', ROOT / 'evolution/ladder.py')
        },
    })


if __name__ == '__main__':
    main()
