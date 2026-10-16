"""Thirteen fixed catalog diagnostics; no population posterior or prior change."""

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

from scripts.check_reduced_spin_feasibility import sha, weight_summary


def fixed_settings():
    return [(mu, math.log10(sigma)) for mu in (-.2, 0., .2)
            for sigma in (.05, .1, .2)] + [(.1, -.7), (-1., -2.), (1., -2.), (0., 1.)]


def partition_summary(values):
    """Describe fixed row partitions without treating them as IID replicates."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 4:
        raise ValueError('At least four ordered rows required')
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('Invalid partition weights')
    report = {'independent_replicates': False, 'integration_error_validated': False}
    for name, groups in [('contiguous', np.array_split(values, 4)),
                         ('interleaved', [values[i::4] for i in range(4)])]:
        summaries = []
        for group in groups:
            if not np.isfinite(group).any():
                summaries.append(dict(status='NO_POSITIVE_WEIGHT_FACTOR_UNAVAILABLE',
                                      rows=int(group.size), log_factor=None))
            else:
                item = weight_summary(group, group.size)
                item['status'] = 'DESCRIPTIVE_GROUP_AVERAGE'
                summaries.append(item)
        logs = [s['log_factor'] for s in summaries]
        report[name] = dict(groups=summaries,
            all_groups_positive=all(x is not None for x in logs),
            log_average_span=max(logs)-min(logs) if all(x is not None for x in logs) else None)
    return report


def checked_report(path, bindings):
    relative = str(path.resolve().relative_to(ROOT)).replace('\\', '/')
    bindings[relative] = sha(path)
    saved = json.loads(path.read_text())
    if not saved['status'].startswith('COMPLETED') or saved['errors'] or saved.get('changed_inputs'):
        raise ValueError('Incomplete or failed predecessor: '+relative)
    return saved


def checked_array(entry, bindings):
    import numpy as np
    path = ROOT/entry['path']
    if sha(path) != entry['sha256']:
        raise ValueError('Array binding differs: '+entry['path'])
    bindings[entry['path']] = entry['sha256']
    return np.load(path, allow_pickle=False)


def compare_weights(actual, expected):
    import numpy as np
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    finite = np.isfinite(actual)
    if not finite.any():
        raise ArithmeticError('Replay has no positive weights')
    return float(np.max(np.abs(actual[finite]-expected[finite])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('readback', type=Path)
    args = parser.parse_args()
    import numpy as np
    from src.boundary_regularized_population import BoundaryRegularizedSpin
    from src.independent_spin_model import SpinParameters
    from src.declared_event_ratio import factorized_log_ratio
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_catalog_weights_'+stamp)
    output.mkdir(exist_ok=False)
    report = dict(status='INCOMPLETE', errors=[], products=[], selection_cases=[],
        baseline_checks=[], worker_budget_seconds=1800, population_fit_run=False,
        combined_likelihood_computed=False, event_exclusions=0, native_evolution_calls=0,
        hyperprior_adopted=False, integration_accuracy_validated=False,
        common_spin_frequency_verified=False, production_ready=False)
    bindings = {}
    print('Results: '+str(output), flush=True)

    def budget():
        if time.monotonic()-started > report['worker_budget_seconds']:
            raise TimeoutError('Fixed catalog diagnostic budget exceeded')

    try:
        for p in (args.manifest.resolve(), args.readback.resolve()):
            if not p.is_relative_to(ROOT/'results/preflight'):
                raise ValueError('Use catalog evidence inside preflight')
        worker = checked_report(args.manifest, bindings)
        review = checked_report(args.readback, bindings)
        if worker['status'] != 'COMPLETED_CATALOG_REFERENCE_PREPARATION_REVIEW_REQUIRED':
            raise ValueError('A complete catalog manifest is required')
        manifest_name = str(args.manifest.resolve().relative_to(ROOT)).replace('\\', '/')
        if review['preserved_hashes'][manifest_name] != bindings[manifest_name]:
            raise ValueError('Readback does not bind this manifest')
        for name, digest in review['preserved_hashes'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Readback evidence changed: '+name)
            bindings[name] = digest
        keys = [(p['event'], p['label']) for p in worker['products']]
        reviewed = {(p['event'], p['label']): p['rows'] for p in review['products']}
        if len(keys) != 208 or len(set(keys)) != 208 or len({k[0] for k in keys}) != 104:
            raise ValueError('Catalog membership differs')
        if set(keys) != set(reviewed) or sum(p['rows'] for p in worker['products']) != 3457688:
            raise ValueError('Readback coverage or catalog row count differs')
        selection = checked_report(ROOT/'results/preflight/o4b_selection_2026-09-17.json', bindings)
        pilot = checked_report(ROOT/'results/preflight/event_coverage_2026-09-17.json', bindings)
        feasibility = checked_report(ROOT/'results/preflight/reduced_spin_feasibility_20261012T034434295802Z/summary.json', bindings)
        fixed = worker['fixed_models']
        if fixed != selection['fixed_models'] or fixed != pilot['fixed_models'] or fixed != feasibility['fixed_models']:
            raise ValueError('Fixed event, selection or feasibility models differ')
        settings = fixed_settings()
        if [(c['mu'], c['log10_sigma']) for c in feasibility['cases']] != settings:
            raise ValueError('Declared settings differ from normalization checks')
        if any(c['status'] != 'COMPUTED_DIAGNOSTIC_ONLY' for c in feasibility['cases']):
            raise ValueError('Prior normalization check failed')
        report['fixed_models'] = fixed
        report['settings'] = [dict(index=i, mu=mu, log10_sigma=ls,
                                  density_check=feasibility['cases'][i]['density_check'])
                              for i, (mu, ls) in enumerate(settings)]
        s = checked_array(selection['arrays'], bindings)
        e = checked_array(pilot['arrays'], bindings)
        mask = s['evaluated'].astype(bool)
        if mask.size != 504666 or selection['total_generated'] != 1568035640:
            raise ValueError('Injection counts differ from declared plan')
        common = (s['log_mass_redshift'][mask]-np.log(s['mass_1_source'][mask])
                  +s['log_r0'][mask]-s['log_p0'][mask]-s['log_draw'][mask]+s['log_mixture'][mask])

        def selection_weights(model):
            weights = np.full(mask.shape, -np.inf)
            weights[mask] = common+model.logpdf(s['mass_1_source'][mask], s['chi_eff'][mask])
            return weights

        base = SpinParameters(**fixed['spin_profiles']['mixed'])
        models = [BoundaryRegularizedSpin(replace(base, mu=mu, log10_sigma=ls, xi_low=0, xi_high=0),
                   a_max=fixed['a_max'], tail_width=fixed['tail_width']) for mu, ls in settings]
        profiles = {name: BoundaryRegularizedSpin(SpinParameters(**pars),
                    a_max=fixed['a_max'], tail_width=fixed['tail_width'])
                    for name, pars in fixed['spin_profiles'].items()}
        names = ['scripts/check_reduced_catalog_weights.py', 'tests/test_reduced_catalog_weights.py',
                 'protocol/reduced_catalog_weight_plan_2026-10-11.md']
        names += [str(Path(m.__file__).resolve().relative_to(ROOT)).replace('\\', '/')
                  for m in tuple(sys.modules.values()) if getattr(m, '__file__', None)
                  and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix == '.py']
        bindings.update({name: sha(ROOT/name) for name in set(names)})
        for name, model in profiles.items():
            old = next(c for c in selection['cases'] if c['profile'] == name and c['displacement'] == 0)
            residual = compare_weights(selection_weights(model), s[old['array_key']])
            report['baseline_checks'].append(dict(kind='selection', profile=name, max_abs_log_residual=residual))
        for i, model in enumerate(models):
            budget()
            weights = selection_weights(model)
            item = dict(index=i, summary=weight_summary(weights, selection['total_generated']),
                        recovered_row_partition_diagnostic=partition_summary(weights))
            old_log = feasibility['cases'][i]['selection']['log_factor']
            if abs(item['summary']['log_factor']-old_log) > 2e-8:
                raise ArithmeticError('Selection diagnostic replay differs')
            report['selection_cases'].append(item)
        pilot_products = {(p['event'], p['label']): p for p in pilot['products']}
        for number, product in enumerate(worker['products']):
            budget()
            begin = time.monotonic()
            key = product['event'], product['label']
            if product['rows'] != reviewed[key]:
                raise ValueError('Product rows differ from readback')
            with checked_array(product['arrays'], bindings) as a:
                n = product['rows']
                np.testing.assert_array_equal(a['row_index'], np.arange(n))
                if any(a[name].shape != (n,) for name in a.files):
                    raise ValueError('Product array shape differs')
                def event_weights(model, mode):
                    return factorized_log_ratio(a['log_mass_redshift'],
                        model.logpdf(a['mass_1_source'], a['chi_eff']),
                        a['log_pe_'+mode], a['log_spin_reference'])
                if key in pilot_products:
                    old_product = pilot_products[key]
                    for name, model in profiles.items():
                        old_case = next(c for c in old_product['cases'] if c['spin_profile'] == name
                                        and c['displacement'] == 0 and c['distance_mode'] == 'continuous')
                        residual = compare_weights(event_weights(model, 'continuous'), e[old_case['array_key']])
                        report['baseline_checks'].append(dict(kind='event', event=key[0], label=key[1],
                            profile=name, max_abs_log_residual=residual))
                result = dict(event=key[0], label=key[1], rows=n, cases=[])
                for i, model in enumerate(models):
                    budget()
                    routes = {}
                    for mode in ('continuous', 'bilby_2.6.0_grid_reference'):
                        weights = event_weights(model, mode)
                        routes[mode] = dict(summary=weight_summary(weights, n),
                                            row_partition_diagnostic=partition_summary(weights))
                    result['cases'].append(dict(index=i, routes=routes,
                        bilby_minus_continuous_log_factor=routes['bilby_2.6.0_grid_reference']['summary']['log_factor']
                            -routes['continuous']['summary']['log_factor']))
                result['elapsed_seconds'] = time.monotonic()-begin
                checkpoint = output/f'product_{number:03d}.json'
                checkpoint.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                report['products'].append(dict(event=key[0], label=key[1], rows=n,
                    checkpoint=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint)))
            if (number+1) % 16 == 0 or number == 207:
                print(f'{number+1}/208 product diagnostics retained', flush=True)
        if len(report['baseline_checks']) != 14:
            raise ValueError('Both archived profiles were not replayed for every pilot product')
        report['status'] = 'COMPLETED_CATALOG_WEIGHT_DIAGNOSTIC_REVIEW_REQUIRED'
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        report['status'] = 'STOPPED_NEEDS_REVIEW'
    finally:
        report['preserved_hashes'] = bindings
        report['changed_inputs'] = [name for name, digest in bindings.items() if sha(ROOT/name) != digest]
        if report['changed_inputs']:
            report['status'] = 'STOPPED_INPUT_CHANGED'
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        path = output/'summary.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(path),
                             products_completed=len(report['products']), elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(bool(report['errors'] or report['changed_inputs']))


if __name__ == '__main__':
    raise SystemExit(main())
