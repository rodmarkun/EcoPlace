"""The real CLI must not permit an unjournaled corrected OpenRouter campaign."""
from __future__ import annotations

import subprocess
import sys

import pytest

from experiments.matched_control_20260909.real_pilot import INITIAL, ROOT, SEED_ARCHIVE


@pytest.mark.parametrize('engine_args', [
    ['--proposal_engine', 'openrouter'], ['--proposal_engine=openrouter'],
])
def test_openrouter_requires_durable_journal_before_campaign_dispatch(tmp_path, engine_args):
    result = subprocess.run([
        sys.executable, '-m', 'experiments.matched_control_20260909.run_campaign',
        '--pair-receipt', str(tmp_path / 'pair.json'),
        '--seed-archive', str(SEED_ARCHIVE), '--initial', str(INITIAL), *engine_args,
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'require --proposal-journal' in result.stderr
    assert not list(tmp_path.iterdir())
