"""Read back the preserved Gaussian selection panel without refitting populations."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    import numpy as np
    from scipy.special import logsumexp
    from scripts.check_reduced_spin_feasibility import sha
    from src.boundary_regularized_population import BoundaryRegularizedSpin
    from src.independent_spin_model import SpinParameters
    from src.sample_selection_audit import parse_months, assign_months, log_ratio_reference_sd
    from src.reduced_selection_reference import replay_covariance_references

    started = time.monotonic()
    source = args.summary.resolve()
    if not source.is_relative_to(ROOT/'results/preflight'):
        raise ValueError('Read back only workspace evidence')
    output = source.parent/('readback_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    report = dict(status='INCOMPLETE', errors=[], worker_budget_seconds=900,
        actual_campaign_variance_validated=False, new_population_fit_run=False,
        original_selection_point_estimates_modified=False, manuscripts_modified=False)
    bindings = {}

    def bind(path, expected=None):
        if time.monotonic()-started > 900:
            raise TimeoutError('Fixed selection readback budget exceeded')
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Consume only workspace evidence')
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('Consumed evidence changed: '+str(path))
        bindings[str(path.relative_to(ROOT))] = digest
        return path

    try:
        worker = json.loads(bind(source).read_text())
        if worker['status'] != 'COMPLETED_REDUCED_SELECTION_REFERENCE_REVIEW_REQUIRED' or worker['errors'] or worker['changed_inputs']:
            raise ValueError('Successful fixed-panel worker required')
        for name, digest in worker['preserved_hashes'].items():
            bind(ROOT/name, digest)
        entry = worker['arrays']
        with np.load(bind(ROOT/entry['path'], entry['sha256']), allow_pickle=False) as a:
            settings, logs = a['settings'].copy(), a['log_weights'].copy()
            constants, rows, times = a['log_constants'].copy(), a['row_index'].copy(), a['time_geocenter'].copy()
            strata, n, r = a['strata'].copy(), a['generated_by_month'].copy(), a['accepted_by_month_inferred'].copy()
            evaluated = a['evaluated'].copy()
        np.testing.assert_array_equal(settings, [s['settings'] for s in worker['settings']])
        if logs.shape != (504666, 21) or int((~evaluated).sum()) != 18634 or worker['total_generated_cumulative'] != 1568035640:
            raise ValueError('Complete original selection count ledger required')
        kernel = json.loads((ROOT/'results/preflight/reduced_kernel_preflight_20261012T132102172011Z/summary.json').read_text())
        cache = kernel['selection_cache']
        with np.load(bind(ROOT/cache['path'], cache['sha256']), allow_pickle=False) as a:
            np.testing.assert_array_equal(rows, a['row_index'])
            np.testing.assert_array_equal(evaluated, a['evaluated'])
            x, base = a['x'].copy(), a['log_base'].copy()
        old_selection = json.loads((ROOT/'results/preflight/o4b_selection_2026-09-17.json').read_text())
        old = old_selection['arrays']
        with np.load(bind(ROOT/old['path'], old['sha256']), allow_pickle=False) as a:
            np.testing.assert_array_equal(rows, a['row_index'])
            np.testing.assert_array_equal(times, a['time_geocenter'])
        readme = (ROOT/'data_manifests/sample_match_sources_2026-09-17/o4_injection_README.md').read_text()
        months = parse_months(readme)
        if worker['months'] != months:
            raise ValueError('Published monthly intervals changed')
        np.testing.assert_array_equal(assign_months(times, months), strata)
        np.testing.assert_array_equal(n, [m['generated'] for m in months])
        np.testing.assert_array_equal(r, [(m['end']-m['start'])//6 for m in months])
        np.testing.assert_array_equal(np.bincount(strata, minlength=len(n)), worker['recovered_by_month'])
        prototype = SpinParameters(**kernel['fixed_models']['spin_profiles']['mixed'])
        maximum = constant_maximum = 0.
        for k, (mu, ls) in enumerate(settings):
            model = BoundaryRegularizedSpin(replace(prototype, mu=float(mu), log10_sigma=float(ls), xi_low=0, xi_high=0), a_max=.99, tail_width=.05)
            c = float(model.gaussian_logpdf(mu))-math.log(model.gaussian_norm)-math.log(.99)
            constant_maximum = max(constant_maximum, abs(c-constants[k]))
            # Canonical normalized spin density versus the stored windowed kernel.
            # Original base includes the window; rearrange the Gaussian exponent.
            expected = base+c-((x-mu)**2)/(2*10.**(2*ls))
            np.testing.assert_array_equal(np.isfinite(expected), np.isfinite(logs[:, k]))
            finite = np.isfinite(expected)
            maximum = max(maximum, float(np.max(abs(expected[finite]-logs[finite, k]))))
        if max(maximum, constant_maximum) > 2e-8:
            raise ArithmeticError('Canonical complete weight replay fails')
        replay = replay_covariance_references(logs, strata, n, r)
        fixed, stopped = np.array(worker['fixed_reference']['relative_covariance']), np.array(worker['stopped_reference']['relative_covariance'])
        np.testing.assert_allclose(fixed, replay['fixed_relative_covariance'], atol=1e-16, rtol=2e-10)
        np.testing.assert_allclose(stopped, replay['stopped_relative_covariance'], atol=1e-16, rtol=2e-10)
        np.testing.assert_allclose(worker['stopped_reference']['point_multiplier_vs_published'], replay['point_multipliers'], atol=2e-10, rtol=0)
        log_alpha = logsumexp(logs, axis=0)-math.log(1568035640)
        cases = []
        for k in range(21):
            saved = worker['cases'][k]
            np.testing.assert_allclose([saved['log_selection'], saved['log_selection_ratio_vs_column0']], [log_alpha[k], log_alpha[k]-log_alpha[0]], atol=2e-8, rtol=0)
            fsd, ssd = log_ratio_reference_sd(fixed, k, 0), log_ratio_reference_sd(stopped, k, 0)
            np.testing.assert_allclose([saved['fixed_log_ratio_reference_sd'], saved['stopped_log_ratio_reference_sd'], saved['fixed_catalog_shape_reference_sd'], saved['stopped_catalog_shape_reference_sd']], [fsd, ssd, 104*fsd, 104*ssd], atol=1e-12, rtol=0)
            cases.append(dict(index=k, log_selection=float(log_alpha[k]),
                fixed_catalog_shape_reference_sd=104*fsd, stopped_catalog_shape_reference_sd=104*ssd,
                fixed_relative_sd=float(np.sqrt(fixed[k,k])), stopped_relative_sd=float(np.sqrt(stopped[k,k]))))
        report.update(status='COMPLETED_REDUCED_SELECTION_REFERENCE_READBACK', cases=cases,
            rows=504666, columns=21, complete_weight_values_replayed=504666*21,
            all_row_weight_max_abs_log_residual=maximum, normalization_max_abs_log_residual=constant_maximum,
            fixed_covariance_max_abs_residual=float(np.max(abs(fixed-replay['fixed_relative_covariance']))),
            stopped_covariance_max_abs_residual=float(np.max(abs(stopped-replay['stopped_relative_covariance']))),
            monthly_count_assumption_retained=True)
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if not (ROOT/name).exists() or sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['elapsed_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output))), flush=True)
    return int(report['status'] != 'COMPLETED_REDUCED_SELECTION_REFERENCE_READBACK')


if __name__ == '__main__':
    raise SystemExit(main())
