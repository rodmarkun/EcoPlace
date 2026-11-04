"""All eight tight-scenario fits; descriptive errors with no acceptance threshold."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from experiments.validation_20260909.summarize_fits import unseal  # noqa: E402


def plot(out: Path) -> None:
    paths = [out / 'tight-fit-errors.png', out / 'tight-fit-errors.svg']
    if any(p.exists() for p in paths):
        raise ValueError('figure already exists')
    summary = unseal(out / 'SUMMARY.json')
    tight = next(s for s in summary['scenarios'] if s['scenario_id'] == 'dev-continuum-tight')
    rows = sorted(tight['selected_fit_errors'],
                  key=lambda r: r['cdf_sup_error_at_or_above_1ms'], reverse=True)
    labels = [r['service'][:12] for r in rows]
    cdf = [r['cdf_sup_error_at_or_above_1ms'] for r in rows]
    biases = []
    for r in rows:
        q = r['quantiles'][1]
        lo, hi = q['observed_bounds_s']
        if lo != hi or lo <= 0:
            raise ValueError('this plot requires identified positive empirical p95 values')
        biases.append(100 * (q['model_s'] / lo - 1))
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.7), sharey=True)
    y = list(range(len(rows)))
    axes[0].barh(y, cdf, color='#355b79', height=.63)
    axes[1].barh(y, biases, color='#a76238', height=.63)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, .45)
    axes[1].set_xlim(-15, 37)
    axes[0].set_title('Cumulative-distribution error', loc='left', pad=16)
    axes[1].set_title('Fitted p95 relative to observed p95', loc='left', pad=16)
    axes[0].set_xlabel('Maximum absolute CDF error, x ≥ 1 ms')
    axes[1].set_xlabel('Difference (%)')
    axes[1].axvline(0, color='#555555', lw=.7)
    for ax in axes:
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dddddd', lw=.6)
    for i, value in enumerate(cdf):
        axes[0].text(value + .007, i, f'{value:.3f}', va='center', fontsize=9)
    for i, value in enumerate(biases):
        axes[1].text(value + (1 if value >= 0 else -1), i, f'{value:+.1f}%',
                     va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    fig.suptitle('Calibration-fit errors in the tight scenario', x=.04, ha='left',
                 fontsize=15, fontweight='bold')
    fig.text(.04, .035, 'All eight selected fits • Service-instance mean response times over '
             '60-second intervals\nIn-sample diagnostics; not per-request latency or '
             'independent model validation.', fontsize=9, color='#444444', va='bottom')
    fig.tight_layout(rect=(.02, .13, 1, .94), w_pad=3)
    for path in paths:
        fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    plot(parser.parse_args().output)
