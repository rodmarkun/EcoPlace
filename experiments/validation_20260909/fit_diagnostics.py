"""Retrospective fit errors for the registered calibration corpus, never held-out data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import time
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from sim.scenarios import alibaba_stream
from sim.scenarios.alibaba import METRICS_COMPONENT, V2021
from sim.scenarios.sources import load_source_registry

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / 'experiments/alibaba_ingestion_20260809/calibration.json'
CENSOR_S = .001


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    envelope = {'payload': payload, 'payload_sha256': hashlib.sha256(raw.encode()).hexdigest()}
    with path.open('x') as stream:
        stream.write(json.dumps(envelope, indent=2, allow_nan=False) + '\n')


def diagnostics(positive_s: list[float], censored: int, mu: float, sigma: float) -> dict:
    """Describe fitted-data errors; no KS p-value or independent validation claim.

    Censored values lie in [0, .001). Quantile/mean bounds replace them by either
    endpoint; no unobserved value is silently imputed as an exact observation.
    """
    values = np.sort(np.asarray(positive_s, dtype=float))
    if (values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values))
            or np.any(values <= 0) or type(censored) is not int or censored < 0
            or not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError('invalid diagnostic inputs')
    n = len(values) + censored
    total = math.fsum(values)
    mean_bounds = [total / n, (total + censored * CENSOR_S) / n]
    model_mean = math.exp(mu + sigma * sigma / 2.)

    def empirical_quantile_bounds(q: float) -> list[float]:
        index = math.ceil(q * n) - 1  # inverse empirical CDF; no interpolation
        lower = 0. if index < censored else float(values[index - censored])
        below = int(np.searchsorted(values, CENSOR_S, side='left'))
        if index < below:
            upper = float(values[index])
        elif index < below + censored:
            upper = CENSOR_S
        else:
            upper = float(values[index - censored])
        return [lower, upper]

    quantiles = []
    for q in (.5, .95, .99):
        bounds = empirical_quantile_bounds(q)
        prediction = math.exp(mu + sigma * NormalDist().inv_cdf(q))
        quantiles.append({'q': q, 'observed_bounds_s': bounds, 'model_s': prediction,
                          'model_minus_observed_bounds_s':
                              [prediction - bounds[1], prediction - bounds[0]]})

    # For x >= censor threshold every censored value is known to be below x.
    # Evaluate both sides of every strictly larger empirical jump, and the
    # right-continuous CDF at the threshold. Group ties correctly.
    unique, first, counts = np.unique(values, return_index=True, return_counts=True)
    mask = unique > CENSOR_S
    f = ndtr((np.log(unique[mask]) - mu) / sigma)
    left = (first[mask] + censored) / n
    right = (first[mask] + counts[mask] + censored) / n
    at_threshold = (int(np.searchsorted(values, CENSOR_S, side='right')) + censored) / n
    model_threshold = float(ndtr((math.log(CENSOR_S) - mu) / sigma))
    discrepancies = [abs(at_threshold - model_threshold)]
    if len(f):
        discrepancies.extend([float(np.max(np.abs(left - f))),
                              float(np.max(np.abs(right - f)))])
    return {
        'active_cells': n, 'positive_cells': len(values), 'censored_cells': censored,
        'observed_mean_bounds_s': mean_bounds, 'model_mean_s': model_mean,
        'model_minus_observed_mean_bounds_s':
            [model_mean - mean_bounds[1], model_mean - mean_bounds[0]],
        'model_over_observed_mean_bounds':
            [model_mean / mean_bounds[1], model_mean / mean_bounds[0]],
        'quantiles': quantiles,
        'cdf_sup_error_at_or_above_1ms': max(discrepancies),
        'empirical_cdf_at_1ms': at_threshold, 'model_cdf_at_1ms': model_threshold,
    }


def observed_scanner(callback, *args, **kwargs):
    """Observe the production fitter's exact selected values; return its original fit."""
    original = alibaba_stream.seal_rt_fit

    def observe(**fit_args):
        fit = original(**fit_args)
        callback(fit, fit_args['positive_s'], fit_args['censored'])
        return fit

    with patch.object(alibaba_stream, 'seal_rt_fit', observe):
        return alibaba_stream.scan_v2021_metrics(*args, **kwargs)


def run(out: Path, scratch: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    registry = load_source_registry()
    component = registry.get(V2021).component(METRICS_COMPONENT)
    archive_path = ROOT / component.archive_path
    historical = json.loads(HISTORICAL.read_text())
    old = {(r['service'], r['metric']): r for r in historical['rt_fits']}
    if len(old) != len(historical['rt_fits']):
        raise ValueError('duplicate historical fit identity')
    source_paths = [Path(__file__).resolve(), HISTORICAL, ROOT / 'uv.lock',
                    ROOT / 'sim/scenarios/data/source_split.json',
                    *sorted((ROOT / 'sim').rglob('*.py')),
                    *sorted((ROOT / 'analysis').rglob('*.py'))]
    sources = {p.relative_to(ROOT).as_posix(): sha_file(p) for p in source_paths}
    plan = {
        'schema': 'retrospective-alibaba-fit-diagnostics-v1',
        'written_unix_seconds': time.time(), 'source_sha256': sources,
        'archive': str(archive_path.relative_to(ROOT)), 'archive_sha256': component.archive_sha256,
        'parser_input': component.parser_input, 'parser_input_sha256': component.sha256,
        'minimum_active_cells': 20, 'shards': 256, 'expected_fit_count': len(old),
        'metrics': ['mean and inverse-ECDF p50/p95/p99 with censoring bounds',
                    'supremum CDF error restricted to x>=1ms'],
        'rule': 'All eligible service/metric fits retained. Compare exact parameters and counts '
                'to historical fits; report seal equality separately. No outcome-based selection, '
                'parameter retuning, goodness-of-fit p-value or acceptance verdict.',
        'boundary': 'In-sample descriptive fit errors for service-instance means over 60s '
                    'intervals in registered v2021 chunk0. No per-request tail, CPU service-time, '
                    'independent validation, whole-trace or physical-energy inference.',
    }
    write(out / 'PLAN.json', plan)
    with tarfile.open(out / 'sources.tar.gz', 'w:gz') as tar:
        for path in source_paths:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    if sha_file(archive_path) != component.archive_sha256:
        raise ValueError('registered metric archive hash differs')
    rows = []

    def consume(fit, positive, censored):
        key = (fit.service, fit.metric)
        prior = old[key]
        parameters = dict(fit.params_seconds)
        matches = (parameters == prior['params_seconds']
                   and len(positive) == prior['positive_count']
                   and censored == prior['censored_count'])
        rows.append({'service': fit.service, 'metric': fit.metric,
                     'historical_parameters_and_counts_match': matches,
                     'historical_seal': prior['seal'], 'current_seal': fit.seal,
                     'seal_matches': fit.seal == prior['seal'],
                     **diagnostics(positive, censored, parameters['mu'], parameters['sigma'])})
        with (out / 'fits.jsonl').open('a') as checkpoint:
            checkpoint.write(json.dumps(rows[-1], sort_keys=True, allow_nan=False) + '\n')

    result = observed_scanner(
        consume, archive_path, component.parser_input, registry,
        workspace=scratch, minimum_count=20, shards=256, progress=True,
    )
    if {(r['service'], r['metric']) for r in rows} != set(old) or len(rows) != len(old):
        raise ValueError('diagnostic fit inventory differs from historical')
    if any(sha_file(ROOT / p) != value for p, value in sources.items()):
        raise ValueError('source drift during diagnostic')
    write(out / 'RESULTS.json', {
        'complete': True, 'rows': rows,
        'all_historical_parameters_and_counts_match':
            all(r['historical_parameters_and_counts_match'] for r in rows),
        'matching_seals': sum(r['seal_matches'] for r in rows),
        'scan': result.scan._payload(),
        'scope': plan['boundary'],
    })
    print(json.dumps({'complete': True, 'fits': len(rows),
                      'parameter_count_matches': sum(r['historical_parameters_and_counts_match']
                                                     for r in rows),
                      'seal_matches': sum(r['seal_matches'] for r in rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.scratch)


if __name__ == '__main__':
    main()
