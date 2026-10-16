"""Locked Gaussian selection panel and explicitly conditional covariance references."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

FINAL = 'results/preflight/reduced_integrated_row_sensitivity_20261012T203612852061Z/readback_20261013T095415077510Z.json'
FINAL_SHA = 'd732ba8f0cf434bc4970c5841435f949e9319e8c5aff17606b6df1ba7163cb6c'
PANEL = 'results/preflight/reduced_posterior_region_weights_20261012T193833154571Z/summary.json'
PANEL_REVIEW = 'results/preflight/reduced_posterior_region_weights_20261012T193833154571Z/readback_20261012T194102020537Z.json'
KERNEL = 'results/preflight/reduced_kernel_preflight_20261012T132102172011Z/summary.json'
STOPPED = 'results/preflight/stopped_campaign_reference_2026-09-18.json'
STOPPED_SHA = '6eb5aa4d2d7ce7b8bb7c6a969bea3a156a4fdd7ddd74845520840345164f38d7'
README = 'data_manifests/sample_match_sources_2026-09-17/o4_injection_README.md'


def main():
    import numpy as np
    from scipy.special import logsumexp
    from scripts.check_reduced_spin_feasibility import sha, weight_summary
    from src.sample_selection_audit import parse_months, assign_months, fixed_stratum_reference, log_ratio_reference_sd
    from src.stopped_campaign_reference import stopped_stratum_reference
    from src.reduced_selection_reference import replay_covariance_references
    from src.reduced_physical_moments import physical_moments
    from src.reduced_gaussian_grid import kernel_log_factors
    from src.boundary_regularized_population import BoundaryRegularizedSpin, log_boundary_window
    from src.independent_spin_model import SpinParameters

    started = time.monotonic()
    output = ROOT/'results/preflight'/('reduced_selection_reference_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    report = dict(status='INCOMPLETE', errors=[], worker_budget_seconds=1800,
        new_population_fit_run=False, event_exclusions=0, retained_events_per_arm=104,
        original_selection_point_estimates_modified=False, actual_campaign_variance_validated=False,
        posterior_sampling_accuracy_validated=False, shared_arm_contrast_calibrated=False,
        hyperprior_changed=False, manuscripts_modified=False,
        monthly_accepted_counts_independently_observed=False, cases=[])
    bindings = {}
    print('Results: '+str(output), flush=True)

    def budget():
        if time.monotonic()-started > 1800:
            raise TimeoutError('Fixed selection-reference budget exceeded; retain evidence')

    def bind(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Use only preserved workspace evidence')
        name = str(path.relative_to(ROOT)).replace('\\', '/')
        budget()
        digest = sha(path)
        if expected and expected != digest:
            raise ValueError('Preserved evidence changed: '+name)
        if name in bindings and bindings[name] != digest:
            raise ValueError('Conflicting evidence binding: '+name)
        bindings[name] = digest
        return path

    def read(name, expected=None):
        result = json.loads(bind(ROOT/name, expected).read_text())
        if not result['status'].startswith('COMPLETED') or result['errors'] or result.get('changed_inputs'):
            raise ValueError('Successful predecessor required: '+name)
        return result

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['preserved_hashes'] = bindings.copy()
        temporary = output/'summary.pending.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'summary.json')

    try:
        save()
        final = read(FINAL, FINAL_SHA)
        for name in (PANEL, PANEL_REVIEW, KERNEL):
            bind(ROOT/name, final['preserved_hashes'][name])
        panel, panel_review, kernel = read(PANEL), read(PANEL_REVIEW), read(KERNEL)
        if panel_review['preserved_hashes'][PANEL] != bindings[PANEL]:
            raise ValueError('Settings differ from reviewed diagnostic')
        stopped_old = read(STOPPED, STOPPED_SHA)
        # Recheck the old campaign/count contract, not every unrelated event file.
        for name in (README, 'src/stopped_campaign_reference.py', 'src/sample_selection_audit.py',
                     'results/preflight/o4_injection_metadata_2026-09-18.json',
                     'results/preflight/o4b_selection_2026-09-17.json'):
            bind(ROOT/name, stopped_old['preserved_hashes'][name])
        for name in ('protocol/reduced_selection_reference_lock_2026-10-13.md',
                     'src/reduced_selection_reference.py', 'tests/test_reduced_selection_reference.py',
                     'scripts/check_reduced_selection_reference.py', 'scripts/review_reduced_selection_reference.py',
                     'src/reduced_physical_moments.py', 'src/reduced_gaussian_grid.py',
                     'src/boundary_regularized_population.py', 'src/independent_spin_model.py',
                     'scripts/check_reduced_spin_feasibility.py'):
            bind(ROOT/name, final['preserved_hashes'].get(name))
        origins = [dict(s, panel='fitted_region') for s in panel['settings']]
        origins += [dict(index=8+k, panel='broad_prior_stress', original_index=k, settings=s) for k, s in enumerate(kernel['settings'])]
        settings = np.array([s['settings'] for s in origins])
        if settings.shape != (21, 2) or len(panel['selection_cases']) != 8:
            raise ValueError('Unchanged eight-plus-thirteen panel required')
        report['settings'] = origins
        report['baseline_column'] = 0
        report['fixed_models'] = kernel['fixed_models']
        old_selection = read('results/preflight/o4b_selection_2026-09-17.json')
        if old_selection['fixed_models'] != kernel['fixed_models']:
            raise ValueError('Fixed target differs from selection archive')
        cache, original = kernel['selection_cache'], old_selection['arrays']
        bind(ROOT/cache['path'], cache['sha256']); bind(ROOT/original['path'], original['sha256'])
        with np.load(ROOT/cache['path'], allow_pickle=False) as a, np.load(ROOT/original['path'], allow_pickle=False) as b:
            x, base, row_index, evaluated = a['x'].copy(), a['log_base'].copy(), a['row_index'].copy(), a['evaluated'].copy()
            times = b['time_geocenter'].copy()
            np.testing.assert_array_equal(row_index, b['row_index'])
            np.testing.assert_array_equal(x, b['chi_eff']/.99)
            np.testing.assert_array_equal(evaluated, b['evaluated'].astype(bool))
            expected_base = np.full(len(x), -np.inf)
            keep = evaluated.astype(bool)
            expected_base[keep] = (b['log_mass_redshift'][keep]-np.log(b['mass_1_source'][keep])+b['log_r0'][keep]
                -b['log_p0'][keep]-b['log_draw'][keep]+b['log_mixture'][keep]+log_boundary_window(x[keep], .05/.99))
            np.testing.assert_allclose(base, expected_base, atol=2e-8, rtol=0)
        if len(x) != 504666 or int(evaluated.sum()) != 486032 or cache['total_generated'] != 1568035640:
            raise ValueError('Complete selection/support count ledger differs')
        report['selection_rows'] = len(x); report['support_zero_rows'] = int((~evaluated).sum())
        report['total_generated_cumulative'] = cache['total_generated']
        readme = (ROOT/README).read_text()
        rows = re.findall(r'\* (\w+ \d{4}) \((\d+) - (\d+)\) : (\d+)', readme)
        all_months = [dict(month=name, start=int(a), end=int(b), generated=int(n),
                          accepted_inferred_from_cadence=(int(b)-int(a))//6) for name, a, b, n in rows]
        if len(all_months) != 20 or any((m['end']-m['start']) % 24 for m in all_months):
            raise ValueError('Published cadence/count table changed')
        if all_months != stopped_old['metadata_reconciliation']['all_twenty_months']:
            raise ValueError('Monthly counts differ from reconciled predecessor')
        attrs = stopped_old['metadata_reconciliation']['recorded_root_attributes']
        if sum(m['generated'] for m in all_months) != attrs['total_generated'] or sum(m['accepted_inferred_from_cadence'] for m in all_months) != attrs['num_accepted']:
            raise ValueError('Full metadata count reconciliation failed')
        months = parse_months(readme)
        strata = assign_months(times, months)
        n = np.array([m['generated'] for m in months], dtype=np.int64)
        r = np.array([(m['end']-m['start'])//6 for m in months], dtype=np.int64)
        report['months'] = months; report['generated_by_month'] = n.tolist(); report['accepted_by_month_inferred'] = r.tolist()
        report['recovered_by_month'] = np.bincount(strata, minlength=len(months)).tolist()
        report['monthly_count_basis'] = 'Full published monthly counts; accepted duration/6 inferred, not independently observed'
        constants = -np.log(.99*physical_moments(settings, budget=budget)[:, 0])
        canonical = []
        prototype = SpinParameters(**kernel['fixed_models']['spin_profiles']['mixed'])
        for mu, ls in settings:
            budget()
            model = BoundaryRegularizedSpin(replace(prototype, mu=float(mu), log10_sigma=float(ls), xi_low=0, xi_high=0), a_max=.99, tail_width=.05)
            canonical.append(float(model.gaussian_logpdf(mu))-math.log(model.gaussian_norm)-math.log(.99))
        norm_residual = float(np.max(abs(constants-canonical)))
        if norm_residual > 2e-8:
            raise ArithmeticError('Independent/canonical Gaussian normalization differs')
        logs = np.empty((len(x), 21))
        for k, (mu, ls) in enumerate(settings):
            budget()
            logs[:, k] = base-.5*((x-mu)/10.**ls)**2+constants[k]
        kernel_sums = logsumexp(logs-constants, axis=0)-math.log(cache['total_generated'])
        expected = np.array([c['summary']['log_factor'] for c in panel['selection_cases']]+cache['kernel_log_factors'])
        residual = float(np.max(abs(kernel_sums-expected)))
        if residual > 2e-8:
            raise ArithmeticError('Selection sums differ from preserved panel')
        path = output/'selection_panel.npz'
        np.savez_compressed(path, settings=settings, log_weights=logs, log_constants=constants,
            row_index=row_index, time_geocenter=times, strata=strata, evaluated=evaluated,
            generated_by_month=n, accepted_by_month_inferred=r)
        report['arrays'] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
        report['normalization_max_abs_log_residual'] = norm_residual
        report['predecessor_kernel_sum_max_abs_log_residual'] = residual
        report['fixed_reference'] = fixed_stratum_reference(logs, strata, n)
        report['stopped_reference'] = stopped_stratum_reference(logs, strata, n, r)
        replay = replay_covariance_references(logs, strata, n, r)
        fixed = np.array(report['fixed_reference']['relative_covariance'])
        stopped = np.array(report['stopped_reference']['relative_covariance'])
        np.testing.assert_allclose(fixed, replay['fixed_relative_covariance'], atol=1e-16, rtol=2e-10)
        np.testing.assert_allclose(stopped, replay['stopped_relative_covariance'], atol=1e-16, rtol=2e-10)
        np.testing.assert_allclose(report['stopped_reference']['point_multiplier_vs_published'], replay['point_multipliers'], atol=2e-10, rtol=0)
        report['covariance_replay_residuals'] = dict(fixed=float(np.max(abs(fixed-replay['fixed_relative_covariance']))),
            stopped=float(np.max(abs(stopped-replay['stopped_relative_covariance']))),
            point_multiplier=float(np.max(abs(np.array(report['stopped_reference']['point_multiplier_vs_published'])-replay['point_multipliers']))))
        log_alpha = logsumexp(logs, axis=0)-math.log(cache['total_generated'])
        for k in range(21):
            fsd = log_ratio_reference_sd(fixed, k, 0)
            ssd = log_ratio_reference_sd(stopped, k, 0)
            report['cases'].append(dict(index=k, log_selection=float(log_alpha[k]),
                log_selection_ratio_vs_column0=float(log_alpha[k]-log_alpha[0]),
                concentration=weight_summary(logs[:, k], cache['total_generated']),
                fixed_reference_relative_sd=float(math.sqrt(fixed[k, k])),
                stopped_reference_relative_sd=float(math.sqrt(stopped[k, k])),
                fixed_log_ratio_reference_sd=fsd, stopped_log_ratio_reference_sd=ssd,
                fixed_catalog_shape_reference_sd=104*fsd, stopped_catalog_shape_reference_sd=104*ssd,
                hypothetical_stopped_point_multiplier=report['stopped_reference']['point_multiplier_vs_published'][k],
                first_order_reference_not_calibrated=True))
        report['status'] = 'COMPLETED_REDUCED_SELECTION_REFERENCE_REVIEW_REQUIRED'
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['changed_inputs'] = [name for name, digest in bindings.items() if not (ROOT/name).exists() or sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output/'summary.json'), elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(report['status'] != 'COMPLETED_REDUCED_SELECTION_REFERENCE_REVIEW_REQUIRED')


if __name__ == '__main__':
    raise SystemExit(main())
