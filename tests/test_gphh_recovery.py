"""Historical tree replay must be complete, source preserving, and fail closed."""
from __future__ import annotations

import hashlib
import shutil

import pytest

from experiments.gp_tuning_20260902.recovery import digest, read_sealed
from experiments.gp_tuning_20260902.run_recovered import write_sealed
from experiments.gphh_holdout_20260905.recover_trees import RUNS, recover


@pytest.fixture
def copied_run(tmp_path):
    original = next(RUNS.glob('*--seed-800000801--*'))
    # The task loader requires a checkout; replay never opens the scenario files.
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "replay-test"\n')
    (tmp_path / 'sim').mkdir()
    shutil.copytree(original / 'inputs', tmp_path / 'inputs')
    shutil.copy2(original / 'preflight.json', tmp_path / 'preflight.json')
    arm = tmp_path / 'arms/classic_gphh'
    arm.mkdir(parents=True)
    shutil.copy2(original / 'arms/classic_gphh/result.json', arm / 'result.json')
    return tmp_path


def reseal_mutation(run, mutate):
    path = run / 'arms/classic_gphh/result.json'
    payload = read_sealed(path)
    mutate(payload['run'])
    inner = payload['run']
    inner['sha256'] = digest({k: v for k, v in inner.items() if k != 'sha256'})
    write_sealed(path, payload)


def test_replay_preserves_sources_and_never_simulates(copied_run, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('tree recovery must not call a simulator')

    monkeypatch.setattr('baselines.gphh.PolicyBundleAdapter.evaluate_through_ladder', forbidden)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in copied_run.rglob('*') if p.is_file()}
    result = recover(copied_run)
    assert len(result['trees']) == 71
    assert result['champion_proposal_seq'] == 70
    assert result['gen0_best_proposal_seq'] == 5
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in copied_run.rglob('*') if p.is_file()}
    assert before == after


def test_resealed_wrong_tree_digest_is_rejected(copied_run):
    reseal_mutation(copied_run, lambda run: run['candidates'][4]['representation'].update(
        sha256='0' * 64
    ))
    with pytest.raises(ValueError, match='tree/identity replay mismatch at candidate 4'):
        recover(copied_run)


def test_resealed_wrong_population_history_is_rejected(copied_run):
    reseal_mutation(
        copied_run, lambda run: run['population_proposal_seqs_by_generation'][1].reverse()
    )
    with pytest.raises(ValueError, match='population history'):
        recover(copied_run)


def test_missing_historical_height_cap_does_not_use_current_default(copied_run):
    reseal_mutation(copied_run, lambda run: run['config'].pop('max_tree_height'))
    with pytest.raises(ValueError, match='height semantics'):
        recover(copied_run)
