"""Index existing development checks; do not rerun or certify scientific gates."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.gp_tuning_20260902.run_recovered import write_sealed

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'experiments/quality_20260909'
MODULES = ('test_sim_invariants', 'test_sim_microscenarios', 'test_sim_energy',
           'test_alibaba_workload_builder', 'test_baseline_assignment_oracles',
           'test_g2_conformance', 'test_energy_gsa')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = EVIDENCE / 'python-source-manifest.json'
    junit_path = EVIDENCE / 'python.xml'
    manifest = json.loads(manifest_path.read_text())
    cases = list(ET.parse(junit_path).iter('testcase'))
    rows = []
    for module in MODULES:
        path = ROOT / 'tests' / f'{module}.py'
        selected = [case for case in cases if case.get('classname') == f'tests.{module}'
                    or case.get('classname', '').startswith(f'tests.{module}.')]
        if not selected:
            raise ValueError(f'no recorded tests for {module}')
        failures = sum(any(case.find(tag) is not None for tag in ('failure', 'error'))
                       for case in selected)
        skips = sum(case.find('skipped') is not None for case in selected)
        rows.append({'module': module, 'cases': len(selected), 'failures': failures,
                     'skipped': skips, 'source_sha256': sha(path),
                     'source_matches_test_run': sha(path) == manifest[str(path.relative_to(ROOT))]})
    sim_sources = {name: source_sha for name, source_sha in manifest.items()
                   if name.startswith('sim/')}
    drift = [name for name, source_sha in sim_sources.items() if sha(ROOT / name) != source_sha]
    write_sealed(Path(__file__).parent / 'SIMULATOR_INVENTORY.json', {
        'schema': 'evolved-continuum-simulator-development-inventory-v1',
        'source_manifest_sha256': sha(manifest_path), 'junit_sha256': sha(junit_path),
        'inventory_source_sha256': sha(Path(__file__)), 'test_modules': rows,
        'sim_source_files_checked': len(sim_sources), 'sim_source_drift': drift,
        'boundary': 'Existing development-test evidence and source comparison only. '
                    'Not P1 acceptance evidence, real-scheduler conformance, workload '
                    'tail calibration, physical energy validation or current '
                    'full-tree certification.',
    })
    print(json.dumps({'test_cases': sum(r['cases'] for r in rows), 'modules': rows,
                      'sim_source_files_checked': len(sim_sources), 'source_drift': drift},
                     indent=2))


if __name__ == '__main__':
    main()
