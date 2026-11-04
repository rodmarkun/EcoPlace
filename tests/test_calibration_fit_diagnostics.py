"""Independent arithmetic and input-preservation checks for retrospective fit errors."""
import hashlib
import io
import json
import math
import tarfile

import pytest

from experiments.validation_20260909.fit_diagnostics import diagnostics, observed_scanner
from sim.scenarios.alibaba import V2021
from sim.scenarios.alibaba_stream import scan_v2021_metrics
from sim.scenarios.sources import DATA_PATH, _load_source_registry_bytes


def test_censored_mean_and_quantiles_use_bounds_not_exact_zero_imputation():
    row = diagnostics([.001, .001, .01], 1, math.log(.001), 1.)
    assert row['observed_mean_bounds_s'] == pytest.approx([.003, .00325])
    assert row['quantiles'][0]['observed_bounds_s'] == [.001, .001]
    assert row['quantiles'][1]['observed_bounds_s'] == [.01, .01]
    assert row['cdf_sup_error_at_or_above_1ms'] == pytest.approx(.25)
    assert row['empirical_cdf_at_1ms'] == .75


def test_positive_observations_below_censor_threshold_do_not_get_misordered():
    row = diagnostics([.0001, .002], 2, math.log(.001), 1.)
    assert row['quantiles'][0]['observed_bounds_s'] == [0., .001]
    assert row['quantiles'][2]['observed_bounds_s'] == [.002, .002]
    assert row['observed_mean_bounds_s'] == pytest.approx([.000525, .001025])


def test_restricted_cdf_includes_the_threshold_when_all_values_are_below_it():
    row = diagnostics([.0001, .0005], 1, math.log(.001), 1.)
    assert row['cdf_sup_error_at_or_above_1ms'] == pytest.approx(.5)


@pytest.mark.parametrize('positive,censored,mu,sigma', [
    ([], 0, 0., 1.), ([0.], 0, 0., 1.), ([float('nan')], 0, 0., 1.),
    ([1.], -1, 0., 1.), ([1.], True, 0., 1.), ([1.], 0, 0., 0.),
])
def test_invalid_data_refused(positive, censored, mu, sigma):
    with pytest.raises(ValueError):
        diagnostics(positive, censored, mu, sigma)


def test_observer_preserves_production_selection_fit_parameters_and_seals(tmp_path):
    rows = [',timestamp,msname,msinstanceid,metric,value']
    for i in range(24):
        for metric, value in [('PROVIDERRPC_RT', 0 if i < 3 else i + 1),
                              ('PROVIDERRPC_MCR', 2)]:
            rows.append(f'{len(rows)},{i * 60000},svc,instance,{metric},{value}')
    # Unpaired RT service is ineligible and must not reach the diagnostic callback.
    rows.append(f'{len(rows)},0,unpaired,instance,PROVIDERRPC_RT,5')
    payload = ('\n'.join(rows) + '\n').encode()
    archive = tmp_path / 'metrics.tar.gz'
    member = 'metrics.csv'
    with tarfile.open(archive, 'w:gz') as tar:
        item = tarfile.TarInfo(member)
        item.size = len(payload)
        tar.addfile(item, io.BytesIO(payload))
    data = json.loads(DATA_PATH.read_text())
    for source in data['sources']:
        if source['source_id'] == V2021:
            for component in source['payload_components']:
                if component['component'] == 'metrics':
                    component.update(parser_input=member, bytes=len(payload),
                                     sha256=hashlib.sha256(payload).hexdigest(),
                                     archive_path=str(archive),
                                     archive_bytes=archive.stat().st_size,
                                     archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    registry = _load_source_registry_bytes(json.dumps(data).encode())
    observed = []

    def consume(fit, positive, censored):
        observed.append((fit.service, list(positive), censored))

    ordinary = scan_v2021_metrics(archive, member, registry, workspace=tmp_path, shards=2)
    witnessed = observed_scanner(consume, archive, member, registry, workspace=tmp_path, shards=2)
    assert ordinary.fits == witnessed.fits
    assert ordinary.scan == witnessed.scan
    assert ordinary.cells == witnessed.cells
    assert observed == [('svc', [i / 1000 for i in range(4, 25)], 3)]
