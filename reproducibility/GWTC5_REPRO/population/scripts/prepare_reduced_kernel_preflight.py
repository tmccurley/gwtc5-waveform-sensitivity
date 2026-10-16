"""Verify exact normalization cancellation and cache all original row factors."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    import numpy as np
    from src.reduced_gaussian_grid import kernel_log_factors
    from src.boundary_regularized_population import BoundaryRegularizedSpin, log_boundary_window
    from src.independent_spin_model import SpinParameters
    from scripts.check_reduced_catalog_weights import fixed_settings, checked_report, checked_array
    from scripts.check_reduced_spin_feasibility import sha
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_kernel_preflight_'+stamp)
    output.mkdir(exist_ok=False)
    bindings = {}
    report = dict(status='INCOMPLETE', errors=[], products=[], original_event_rows=0,
                  worker_budget_seconds=1800, real_population_fit_run=False,
                  combined_fixed_point_identity_checked=False, event_exclusions=0,
                  native_evolution_calls=0, posterior_sampling_accuracy_validated=False,
                  common_spin_frequency_verified=False, production_ready=False)
    print('Results: '+str(output), flush=True)
    try:
        parent = ROOT/'results/preflight/reduced_catalog_weights_20261012T054553412584Z'
        diagnostic = checked_report(parent/'summary.json', bindings)
        readback = checked_report(parent/'readback_20261012T055258491571Z.json', bindings)
        manifest = checked_report(ROOT/'results/preflight/reduced_catalog_reference_20261012T040844544750Z/summary.json', bindings)
        selection = checked_report(ROOT/'results/preflight/o4b_selection_2026-09-17.json', bindings)
        for name, digest in readback['preserved_hashes'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Consumed diagnostic evidence changed: '+name)
            bindings[name] = digest
        fixed = manifest['fixed_models']
        if fixed != diagnostic['fixed_models'] or fixed != selection['fixed_models']:
            raise ValueError('Fixed target differs')
        names = ['protocol/reduced_fit_lock_2026-10-12.md', 'scripts/prepare_reduced_kernel_preflight.py',
                 'src/reduced_gaussian_grid.py', 'tests/test_reduced_gaussian_grid.py']
        names += [str(Path(m.__file__).resolve().relative_to(ROOT)).replace('\\', '/')
                  for m in tuple(sys.modules.values()) if getattr(m, '__file__', None)
                  and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix == '.py']
        bindings.update({name: sha(ROOT/name) for name in set(names)})
        settings = np.array(fixed_settings())
        base = SpinParameters(**fixed['spin_profiles']['mixed'])
        constants = []
        for mu, ls in settings:
            model = BoundaryRegularizedSpin(replace(base, mu=float(mu), log10_sigma=float(ls), xi_low=0, xi_high=0),
                                           a_max=fixed['a_max'], tail_width=fixed['tail_width'])
            constants.append(float(model.gaussian_logpdf(mu))-math.log(model.gaussian_norm)-math.log(model.a_max))
        constants = np.array(constants)
        report['fixed_models'] = fixed
        report['settings'] = settings.tolist()
        report['log_spin_normalization_constants'] = constants.tolist()
        expected_entries = {(p['event'], p['label']): p for p in diagnostic['products']}
        if len(expected_entries) != 208 or len(manifest['products']) != 208:
            raise ValueError('Full catalog required')
        window_width = fixed['tail_width']/fixed['a_max']
        with checked_array(selection['arrays'], bindings) as s:
            mask = s['evaluated'].astype(bool)
            if mask.size != 504666 or selection['total_generated'] != 1568035640:
                raise ValueError('Injection count changed')
            x = s['chi_eff']/fixed['a_max']
            b = np.full(mask.shape, -np.inf)
            b[mask] = (s['log_mass_redshift'][mask]-np.log(s['mass_1_source'][mask])
                +s['log_r0'][mask]-s['log_p0'][mask]-s['log_draw'][mask]+s['log_mixture'][mask]
                +log_boundary_window(x[mask], window_width))
            begin = time.monotonic()
            kernel_selection = kernel_log_factors(x, b, settings, selection['total_generated'])
            report['selection_compute_seconds'] = time.monotonic()-begin
            expected = np.array([c['summary']['log_factor'] for c in diagnostic['selection_cases']])
            np.testing.assert_allclose(kernel_selection+constants, expected, rtol=0, atol=2e-8)
            report['selection_max_abs_log_residual'] = float(np.max(abs(kernel_selection+constants-expected)))
            checkpoint = output/'selection_kernel.npz'
            np.savez_compressed(checkpoint, x=x, log_base=b, row_index=s['row_index'], evaluated=mask)
            report['selection_cache'] = dict(path=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint),
                recovered_rows=mask.size, total_generated=selection['total_generated'],
                kernel_log_factors=kernel_selection.tolist())
        sums = {mode: {arm: np.zeros(13) for arm in ('ST', 'XPNR')}
                for mode in ('continuous', 'bilby_2.6.0_grid_reference')}
        old_sums = {mode: {arm: np.zeros(13) for arm in ('ST', 'XPNR')}
                    for mode in sums}
        counts = dict(ST=0, XPNR=0)
        for number, product in enumerate(manifest['products']):
            if time.monotonic()-started > 1800:
                raise TimeoutError('Fixed kernel preflight budget exceeded')
            key = product['event'], product['label']
            old_entry = expected_entries[key]
            old_path = ROOT/old_entry['checkpoint']
            if sha(old_path) != old_entry['sha256']:
                raise ValueError('Diagnostic product changed')
            bindings[old_entry['checkpoint']] = old_entry['sha256']
            old = json.loads(old_path.read_text())
            arm = 'ST' if key[1].endswith('SpinTaylor') else 'XPNR'
            if arm == 'XPNR' and not key[1].endswith('IMRPhenomXPNR'):
                raise ValueError('Unexpected arm')
            counts[arm] += 1
            with checked_array(product['arrays'], bindings) as a:
                n = product['rows']
                np.testing.assert_array_equal(a['row_index'], np.arange(n))
                x = a['chi_eff']/fixed['a_max']
                window = log_boundary_window(x, window_width)
                saved = dict(event=key[0], label=key[1], arm=arm, rows=n, modes={})
                arrays = dict(x=x, row_index=a['row_index'])
                for mode in sums:
                    b = a['log_mass_redshift']-a['log_pe_'+mode]-a['log_spin_reference']+window
                    arrays['log_base_'+mode] = b
                    begin = time.monotonic()
                    factors = kernel_log_factors(x, b, settings, n)
                    seconds = time.monotonic()-begin
                    expected = np.array([c['routes'][mode]['summary']['log_factor'] for c in old['cases']])
                    np.testing.assert_allclose(factors+constants, expected, rtol=0, atol=2e-8)
                    saved['modes'][mode] = dict(max_abs_log_residual=float(np.max(abs(factors+constants-expected))),
                        compute_seconds=seconds, kernel_log_factors=factors.tolist())
                    sums[mode][arm] += factors
                    old_sums[mode][arm] += expected
                checkpoint = output/f'product_{number:03d}.npz'
                np.savez_compressed(checkpoint, **arrays)
                saved['cache'] = dict(path=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint))
                report['products'].append(saved)
                report['original_event_rows'] += n
            if (number+1) % 32 == 0 or number == 207:
                print(f'{number+1}/208 exact kernel caches checked', flush=True)
        if counts != dict(ST=104, XPNR=104) or report['original_event_rows'] != 3457688:
            raise ValueError('Original frame changed')
        expected_selection = np.array([c['summary']['log_factor'] for c in diagnostic['selection_cases']])
        report['combined_identity_residuals'] = {}
        for mode in sums:
            report['combined_identity_residuals'][mode] = {}
            for arm in counts:
                canonical = sums[mode][arm]-104*kernel_selection
                normalized = old_sums[mode][arm]-104*expected_selection
                residual = float(np.max(abs(canonical-normalized)))
                if residual > 5e-6:
                    raise ArithmeticError('Combined common-normalization cancellation differs')
                report['combined_identity_residuals'][mode][arm] = residual
        report['combined_fixed_point_identity_checked'] = True
        event_compute = sum(v['compute_seconds'] for p in report['products'] for v in p['modes'].values())
        report['all_thirteen_settings_compute_seconds'] = event_compute+report['selection_compute_seconds']
        report['linear_extrapolation_only_seconds_by_grid_order'] = {
            str(order): report['all_thirteen_settings_compute_seconds']*order*order/13 for order in (32, 64, 128, 256)}
        report['extrapolation_excludes_io_and_refinement_checks'] = True
        report['status'] = 'COMPLETED_EXACT_KERNEL_PREFLIGHT_REVIEW_REQUIRED'
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        report['status'] = 'STOPPED_NEEDS_REVIEW'
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['elapsed_seconds'] = time.monotonic()-started
    path = output/'summary.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(path),
                         products_completed=len(report['products']), elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(bool(report['errors'] or report['changed_inputs']))


if __name__ == '__main__':
    raise SystemExit(main())
