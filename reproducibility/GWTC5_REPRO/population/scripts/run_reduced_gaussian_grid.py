"""Checkpointed provisional two-parameter grid, with explicit refinement stops."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel_preflight', type=Path)
    parser.add_argument('controls', type=Path)
    args = parser.parse_args()
    import numpy as np
    from src.reduced_gaussian_grid import prior_grid, kernel_log_factors, normalized_grid, refinement_check
    from scripts.check_reduced_spin_feasibility import sha
    from scripts.check_reduced_catalog_weights import checked_report, checked_array
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_observed_grid_'+stamp)
    output.mkdir(exist_ok=False)
    bindings = {}
    report = dict(status='INCOMPLETE', errors=[], levels=[], checkpoints=[],
        worker_budget_seconds=21600, event_exclusions=0, real_population_fit_run=False,
        prior={'mu': [-1, 1], 'log10_sigma': [-2, 1], 'joint_density': 1/6},
        empirical_quadrature_refinement_pass=False, scientific_result_validated=False,
        posterior_sampling_accuracy_validated=False, selection_design_variance_validated=False,
        common_spin_frequency_verified=False, physical_moments_computed=False,
        manuscripts_modified=False, production_ready=False)
    print('Results: '+str(output), flush=True)

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def budget():
        if time.monotonic()-started > 21600:
            raise TimeoutError('Fixed observed-grid budget exceeded; completed tiles retained')

    try:
        for path in (args.kernel_preflight.resolve(), args.controls.resolve()):
            if not path.is_relative_to(ROOT/'results/preflight'):
                raise ValueError('Use preflight evidence inside the workspace')
        preflight = checked_report(args.kernel_preflight, bindings)
        controls = checked_report(args.controls, bindings)
        report['kernel_preflight'] = str(args.kernel_preflight.resolve().relative_to(ROOT))
        report['controls'] = str(args.controls.resolve().relative_to(ROOT))
        if preflight['status'] != 'COMPLETED_EXACT_KERNEL_PREFLIGHT_REVIEW_REQUIRED' or not preflight['combined_fixed_point_identity_checked']:
            raise ValueError('Exact kernel preflight is not complete')
        if controls['status'] != 'COMPLETED_REDUCED_GRID_NUMERICAL_CONTROLS_REVIEW_REQUIRED':
            raise ValueError('Known-answer integration controls are not complete')
        for evidence in (preflight, controls):
            for name, digest in evidence['preserved_hashes'].items():
                if sha(ROOT/name) != digest:
                    raise ValueError('Consumed preflight evidence changed: '+name)
                if name in bindings and bindings[name] != digest:
                    raise ValueError('Preflight source bindings disagree')
                bindings[name] = digest
        names = ['scripts/run_reduced_gaussian_grid.py', 'src/reduced_gaussian_grid.py',
                 'protocol/reduced_fit_lock_2026-10-12.md', 'tests/test_reduced_gaussian_grid.py',
                 'protocol/reduced_grid_compute_budget_amendment_2026-10-12.md']
        bindings.update({name: sha(ROOT/name) for name in names})
        products = []
        for entry in preflight['products']:
            budget()
            with checked_array(entry['cache'], bindings) as a:
                n = entry['rows']
                np.testing.assert_array_equal(a['row_index'], np.arange(n))
                item = dict(event=entry['event'], label=entry['label'], arm=entry['arm'], rows=n,
                            x=a['x'].copy(), bases={})
                for mode in ('continuous', 'bilby_2.6.0_grid_reference'):
                    item['bases'][mode] = a['log_base_'+mode].copy()
                if any(v.shape != (n,) for v in [item['x'], *item['bases'].values()]):
                    raise ValueError('Kernel cache row shape changed')
                products.append(item)
        identities = {(p['event'], p['label']) for p in products}
        if len(products) != 208 or len(identities) != 208 or sum(p['rows'] for p in products) != 3457688:
            raise ValueError('Original catalog coverage changed')
        if any(sum(p['arm'] == arm for p in products) != 104 for arm in ('ST', 'XPNR')):
            raise ValueError('Arm membership changed')
        with checked_array(preflight['selection_cache'], bindings) as a:
            selection_x, selection_base = a['x'].copy(), a['log_base'].copy()
        generated = preflight['selection_cache']['total_generated']
        if selection_x.size != 504666 or generated != 1568035640:
            raise ValueError('Selection counts changed')
        report['fixed_models'] = preflight['fixed_models']
        report['real_population_fit_run'] = True
        previous = None
        columns = [('continuous', 'ST'), ('continuous', 'XPNR'),
                   ('bilby_2.6.0_grid_reference', 'ST'), ('bilby_2.6.0_grid_reference', 'XPNR')]
        report['likelihood_columns'] = [list(c) for c in columns]
        for order in (32, 64, 128, 256):
            budget()
            begin = time.monotonic()
            nodes, prior = prior_grid(order)
            likelihoods = np.full((len(nodes), 4), np.nan)
            level = dict(order=order, nodes=len(nodes), completed_nodes=0, status='INCOMPLETE')
            report['levels'].append(level)
            for first in range(0, len(nodes), 128):
                budget()
                batch = nodes[first:first+128]
                selection_values = kernel_log_factors(selection_x, selection_base, batch, generated)
                factors = np.empty((len(products), len(batch), 2))
                accumulated = np.zeros((len(batch), 4))
                for j, p in enumerate(products):
                    budget()
                    for k, mode in enumerate(('continuous', 'bilby_2.6.0_grid_reference')):
                        values = kernel_log_factors(p['x'], p['bases'][mode], batch, p['rows'])
                        factors[j, :, k] = values
                        column = columns.index((mode, p['arm']))
                        accumulated[:, column] += values
                values = accumulated-104*selection_values[:, None]
                if not np.isfinite(values).all():
                    raise ArithmeticError('Invalid combined likelihood; no zero substituted')
                likelihoods[first:first+len(batch)] = values
                checkpoint = output/f'order_{order:03d}_tile_{first:05d}.npz'
                np.savez_compressed(checkpoint, settings=batch, prior_weights=prior[first:first+len(batch)],
                                    selection_kernel_log_factors=selection_values,
                                    event_kernel_log_factors=factors, catalog_log_likelihoods=values)
                report['checkpoints'].append(dict(order=order, first_node=first, nodes=len(batch),
                    path=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint)))
                level['completed_nodes'] += len(batch)
                if (first//128+1) % 8 == 0 or level['completed_nodes'] == len(nodes):
                    save()
                    print(f'order {order}: {level["completed_nodes"]}/{len(nodes)} nodes retained', flush=True)
            summaries, mass_columns = [], []
            for i in range(4):
                summary, masses = normalized_grid(nodes, prior, likelihoods[:, i])
                summaries.append(summary); mass_columns.append(masses)
            level.update(status='COMPLETE_PROVISIONAL_GRID_LEVEL', summaries=summaries,
                         compute_seconds=time.monotonic()-begin,
                         refinement=[refinement_check(previous[i], summaries[i]) for i in range(4)] if previous else None)
            checkpoint = output/f'order_{order:03d}_complete.npz'
            np.savez_compressed(checkpoint, settings=nodes, prior_weights=prior,
                                catalog_log_likelihoods=likelihoods, posterior_quadrature_masses=np.column_stack(mass_columns))
            level['arrays'] = dict(path=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint))
            save()
            if order >= 128 and all(c['empirical_refinement_pass'] for c in level['refinement']):
                report['empirical_quadrature_refinement_pass'] = True
                break
            previous = summaries
        report['status'] = ('COMPLETED_PROVISIONAL_REDUCED_GRID_REVIEW_REQUIRED'
                            if report['empirical_quadrature_refinement_pass'] else 'STOPPED_REFINEMENT_UNRESOLVED')
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        report['status'] = 'STOPPED_NEEDS_REVIEW'
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output/'summary.json'),
                         elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(bool(report['errors'] or report['changed_inputs'] or not report['empirical_quadrature_refinement_pass']))


if __name__ == '__main__':
    raise SystemExit(main())
