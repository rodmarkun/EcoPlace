"""Close and summarize a complete retrospective calibration-fit diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from experiments.validation_20260909.fit_diagnostics import ROOT, sha_file, write


def unseal(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    raw = json.dumps(envelope['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(raw.encode()).hexdigest() != envelope['payload_sha256']:
        raise ValueError(f'seal mismatch: {path}')
    return envelope['payload']


def summarize(out: Path) -> dict:
    plan = unseal(out / 'PLAN.json')
    result = unseal(out / 'RESULTS.json')
    usage = json.loads((out / 'USAGE.json').read_text())
    if not result['complete'] or usage['exit_code'] != 0 or usage['stop_reason'] is not None:
        raise ValueError('diagnostic did not complete within its resource guards')
    rows = result['rows']
    if len(rows) != plan['expected_fit_count']:
        raise ValueError('incomplete fit inventory')
    checkpoints = [json.loads(line) for line in (out / 'fits.jsonl').read_text().splitlines()]
    if checkpoints != rows:
        raise ValueError('checkpoint/results mismatch')
    with tarfile.open(out / 'sources.tar.gz') as archive:
        if (len(archive.getmembers()) != len(plan['source_sha256'])
                or set(archive.getnames()) != set(plan['source_sha256'])):
            raise ValueError('source archive inventory differs')
        for name, digest in plan['source_sha256'].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError(f'source archive member mismatch: {name}')
    if result['scan']['content_sha256'] != plan['parser_input_sha256']:
        raise ValueError('complete-member digest mismatch')
    by_seal = {row['historical_seal']: row for row in rows}
    if len(by_seal) != len(rows):
        raise ValueError('duplicate fit identity')
    scenarios = []
    for path in sorted((ROOT / 'sim/scenarios/dev').glob('*.manifest.json')):
        manifest = json.loads(path.read_text())
        builder = manifest.get('workload_builder', {})
        seals = {pair[1] for pair in builder.get('selected_fit_seals', [])}
        if 'closure_json' in builder:
            closure = json.loads(builder['closure_json'])
            seals.update(e['artifact_sha256'] for e in closure['evidence']
                         if e['kind'] == 'interval-mean-rt-fit')
        if not seals:
            raise ValueError(f'no explicit fit membership in {path}')
        if any(seal not in by_seal for seal in seals):
            raise ValueError('scenario refers to a fit outside the recovered diagnostic')
        if sha_file(ROOT / manifest['file']['path']) != manifest['file']['sha256']:
            raise ValueError(f'scenario file hash differs from manifest: {path}')
        scenarios.append({'scenario_id': manifest['scenario_id'],
                          'manifest': path.relative_to(ROOT).as_posix(),
                          'manifest_file_sha256': sha_file(path),
                          'selected_fit_seals': sorted(seals),
                          'selected_fit_errors': [by_seal[seal] for seal in sorted(seals)]})
    errors = sorted(r['cdf_sup_error_at_or_above_1ms'] for r in rows)
    summary = {
        'fits': len(rows), 'positive_cells': sum(r['positive_cells'] for r in rows),
        'censored_cells': sum(r['censored_cells'] for r in rows),
        'all_historical_parameters_and_counts_match':
            result['all_historical_parameters_and_counts_match'],
        'matching_seals': result['matching_seals'],
        'cdf_error_unweighted_by_fit': {
            'min': errors[0], 'median': statistics.median(errors),
            'p90_nearest_rank': errors[(9 * len(errors) + 9) // 10 - 1], 'max': errors[-1],
            'count_above_0_1': sum(e > .1 for e in errors),
            'count_above_0_2': sum(e > .2 for e in errors),
        },
        'scenarios': scenarios, 'usage': usage,
        'verified_archived_source_members': len(plan['source_sha256']),
        'analysis_source_sha256': sha_file(Path(__file__)),
        'scope': result['scope'],
        'membership_scope': 'Fit membership declared by all six retained development scenario '
                            'manifests, with referenced config file hashes verified. Not a claim '
                            'that other historical consumers never accessed these or other fits.',
    }
    write(out / 'SUMMARY.json', summary)
    lines = ['# Retrospective Alibaba fit diagnostics', '', summary['scope'], '',
             f"Fits: **{len(rows)}**. Exact historical parameter/count agreement: "
             f"**{summary['all_historical_parameters_and_counts_match']}**; identical fit seals: "
             f"**{result['matching_seals']}**.", '',
             'The CDF error is the largest absolute difference between the fitted and empirical',
             'cumulative distribution for interval means at or above 1 ms. It is a descriptive',
             'in-sample discrepancy, not a KS test, validation score or acceptance threshold.',
             'Repeated cells within a service/instance are not assumed independent. Quantile',
             'and mean bounds retain left-censoring uncertainty.', '',
             f"Across fits (unweighted): median error {statistics.median(errors):.4f}, "
             f"90th percentile {summary['cdf_error_unweighted_by_fit']['p90_nearest_rank']:.4f}, "
             f"maximum {errors[-1]:.4f}. Errors >0.1: {sum(e>.1 for e in errors)}/{len(rows)}; "
             f">0.2: {sum(e>.2 for e in errors)}/{len(rows)}. These cutoffs are descriptive.", '',
             '## Fits used by the tight scenario', '',
             '| Service prefix | Metric | Active cells | CDF error | '
             'Observed p95 mean-RT (ms) | Fitted p95 mean-RT (ms) |',
             '|---|---|---:|---:|---:|---:|']
    tight = next(s for s in scenarios if s['scenario_id'] == 'dev-continuum-tight')
    for r in tight['selected_fit_errors']:
        q = r['quantiles'][1]
        lo, hi = q['observed_bounds_s']
        obs = f'{1000*lo:.3f}' if lo == hi else f'[{1000*lo:.3f}, {1000*hi:.3f}]'
        lines.append(f"| {r['service'][:12]} | {r['metric']} | {r['active_cells']} | "
                     f"{r['cdf_sup_error_at_or_above_1ms']:.4f} | {obs} | "
                     f"{1000*q['model_s']:.3f} |")
    lines += ['', 'All per-fit errors, censoring bounds and full service identities are in',
              '`RESULTS.json`; all six development-manifest memberships are in `SUMMARY.json`.',
              'The historical 0.45 response-time-to-service-time attribution and later demand/SLO',
              'transforms remain synthetic assumptions. Fit quality does not identify that mapping',
              'or validate request-level tails. No protected v2022/TeaStore data was opened.', '',
              f"Completed within guards: {usage['elapsed_seconds']:.1f} s, "
              f"{usage['peak_rss_mib']:.1f} MiB peak RSS; owned scratch removed. "
              f"Verified {len(plan['source_sha256'])} archived source/data/lock members.",
              'The first scan was explicitly stopped after a sandbox monitoring failure; its',
              'partial attempt remains in the neighboring unversioned output directory.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    answer = summarize(arguments.output)
    print(json.dumps({k: v for k, v in answer.items() if k != 'scenarios'}, indent=2))
