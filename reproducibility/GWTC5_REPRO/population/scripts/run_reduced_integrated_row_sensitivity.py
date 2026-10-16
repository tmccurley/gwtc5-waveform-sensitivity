"""Bounded, checkpointed integrated posterior-row sensitivity, not MC calibration."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

# Completed evidence is fixed before this diagnostic. No latest-file discovery
# can accidentally select a different fit, model, or development panel.
EVIDENCE = {
    'grid': ('reduced_observed_grid_20261012T132719333655Z/summary.json', '70db7d331bc3bfb841c0fd64b84b0e3bd3c809552582a16dd419f087fee72d31'),
    'grid_readback': ('reduced_observed_grid_20261012T132719333655Z/readback_20261012T185545544529Z.json', '02b7dd925a4f9eecb2823423a16e50c83b85ad5e8ac9a545ce8c66810f1883ab'),
    'physical': ('reduced_physical_moments_20261012T193150884287Z/summary.json', 'c5ba84d950b3b2f82b633a1f193e598f051daac1b848c34dfcaeca71616410d5'),
    'physical_readback': ('reduced_physical_moments_20261012T193150884287Z/readback_20261012T193424563832Z.json', '0989eb56d97fbdf9f4c6b68aa9d17d6731330b644d22489ea5faee0ff721a1fd'),
    'group_preflight': ('reduced_row_sensitivity_preflight_20261012T194558901805Z/summary.json', '9f47b797f41c27d7c60472ca1554eb0030f03f995a595b067ca2de71c91e21f0'),
    'group_comparison': ('reduced_row_sensitivity_fast_20261012T195120703429Z/summary.json', '26dfdbd3012cfbc4abba2b31ebc717f19e1d5b5d6ecb03032d2bfca59d66ad3c'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Verify inputs, without any integrated subset fit')
    args = parser.parse_args()
    import numpy as np
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid, normalized_grid, refinement_check
    from src.reduced_physical_moments import summarize_physical, physical_refinement
    from src.reduced_row_group_kernels import grouped_kernel_log_factors, GROUP_NAMES, group_geometry
    from src.reduced_integrated_row_sensitivity import ROUTES, COLUMNS, assemble_catalog, quarter_residuals

    started = time.monotonic()
    # The operating system releases this lock on exit/crash. The pathname stays
    # as a harmless marker; file existence alone never means a worker is alive.
    lock = (ROOT/'results/preflight/reduced_integrated_row_sensitivity.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('An integrated row-sensitivity worker already holds the lock; no duplicate launched.', flush=True)
        return 2
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_integrated_row_sensitivity_'+stamp)
    output.mkdir(exist_ok=False)
    bindings = {}
    report = dict(status='PREPARING', errors=[], levels=[], checkpoints=[], products=[],
        worker_budget_seconds=108000, pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        validate_only=args.validate_only, subset_population_fit_run=False, event_exclusions=0,
        hyperprior_changed=False, selection_held_full=True, selection_recovered_rows=504666,
        selection_generated_denominator=1568035640, posterior_sampling_accuracy_validated=False,
        selection_design_variance_validated=False, scientific_result_validated=False,
        paired_contrast_significance_calibrated=False, common_spin_frequency_verified=False,
        manuscripts_modified=False, group_names=list(GROUP_NAMES), likelihood_columns=[list(c) for c in COLUMNS])
    print('Results: '+str(output), flush=True)

    def budget():
        if time.monotonic()-started > 108000:
            raise TimeoutError('Thirty-hour diagnostic budget exceeded; checkpoints preserved')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['preserved_hashes'] = bindings.copy()
        temporary = output/'summary.pending.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'summary.json')

    def bind(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Consumed evidence must remain in the research workspace')
        name = str(path.relative_to(ROOT))
        if name in bindings:
            if expected and expected != bindings[name]:
                raise ValueError('Conflicting preserved evidence hashes: '+name)
            return path
        budget()
        digest = sha(path)
        if expected and expected != digest:
            raise ValueError('Consumed scientific evidence changed: '+name)
        bindings[name] = digest
        return path

    def array(entry):
        return np.load(bind(ROOT/entry['path'], entry['sha256']), allow_pickle=False)

    try:
        save()
        free = shutil.disk_usage(output).free
        report['free_disk_bytes_at_start'] = free
        if free < 10*1024**3:
            raise OSError('Less than ten GiB available; do not begin the large archive')
        evidence = {}
        for key, (relative, digest) in EVIDENCE.items():
            path = bind(ROOT/'results/preflight'/relative, digest)
            item = json.loads(path.read_text())
            if not item['status'].startswith('COMPLETED') or item['errors'] or item.get('changed_inputs'):
                raise ValueError('Successful predecessor required: '+key)
            evidence[key] = item
            for name, expected in item['preserved_hashes'].items():
                bind(ROOT/name, expected)
        grid, physical = evidence['grid'], evidence['physical']
        if grid['status'] != 'COMPLETED_PROVISIONAL_REDUCED_GRID_REVIEW_REQUIRED' or not grid['empirical_quadrature_refinement_pass']:
            raise ValueError('Completed primary grid required')
        if physical['status'] != 'COMPLETED_PHYSICAL_MOMENTS_REVIEW_REQUIRED' or not physical['empirical_physical_refinement_pass']:
            raise ValueError('Verified physical moment arrays required')
        if grid['fixed_models']['a_max'] != .99 or grid['fixed_models']['tail_width'] != .05:
            raise ValueError('Locked physical model changed')
        report['fixed_models'] = grid['fixed_models']
        report['evidence'] = {key: str(Path('results/preflight')/value[0]) for key, value in EVIDENCE.items()}
        for name in ('scripts/run_reduced_integrated_row_sensitivity.py',
                     'scripts/review_reduced_integrated_row_sensitivity.py',
                     'src/reduced_integrated_row_sensitivity.py', 'tests/test_reduced_integrated_row_sensitivity.py',
                     'src/reduced_row_group_kernels.py', 'src/reduced_gaussian_grid.py',
                     'src/reduced_physical_moments.py',
                     'protocol/reduced_integrated_row_sensitivity_lock_2026-10-12.md',
                     'protocol/reduced_integrated_row_sensitivity_proposal_2026-10-12.md'):
            bind(ROOT/name)

        kernel_path = bind(ROOT/grid['kernel_preflight'], grid['preserved_hashes'][grid['kernel_preflight']])
        kernel = json.loads(kernel_path.read_text())
        if kernel['selection_cache']['recovered_rows'] != 504666 or kernel['selection_cache']['total_generated'] != 1568035640:
            raise ValueError('Complete primary selection counts differ')
        products = []
        for entry in kernel['products']:
            with array(entry['cache']) as a:
                n = entry['rows']
                np.testing.assert_array_equal(a['row_index'], np.arange(n))
                item = dict(event=entry['event'], label=entry['label'], arm=entry['arm'], rows=n,
                            x=a['x'].copy(), bases={route: a['log_base_'+route].copy() for route in ROUTES})
            if any(v.shape != (n,) for v in [item['x'], *item['bases'].values()]):
                raise ValueError('Original row shape changed')
            products.append(item)
            report['products'].append({k: item[k] for k in ('event', 'label', 'arm', 'rows')})
        if len(products) != 208 or len({(p['event'], p['label']) for p in products}) != 208 or len({p['event'] for p in products}) != 104:
            raise ValueError('Full original catalog identities changed')
        if sum(p['rows'] for p in products) != 3457688 or any(sum(p['arm'] == arm for p in products) != 104 for arm in ('ST', 'XPNR')):
            raise ValueError('Original row counts or arm membership changed')
        report['group_row_counts'] = [group_geometry(p['rows'], np.array([], dtype=int))[1].tolist() for p in products]

        prepared = {}
        for order in (128, 256):
            nodes, prior = prior_grid(order)
            original = next(level for level in grid['levels'] if level['order'] == order)
            with array(original['arrays']) as a:
                np.testing.assert_array_equal(a['settings'], nodes)
                np.testing.assert_array_equal(a['prior_weights'], prior)
                primary = a['catalog_log_likelihoods'].copy()
            selection = np.full(len(nodes), np.nan)
            covered = np.zeros(len(nodes), bool)
            for entry in grid['checkpoints']:
                if entry['order'] != order:
                    continue
                first, end = entry['first_node'], entry['first_node']+entry['nodes']
                if first < 0 or end > len(nodes) or covered[first:end].any():
                    raise ValueError('Primary selection tiles overlap or leave the grid')
                with array(entry) as a:
                    np.testing.assert_array_equal(a['settings'], nodes[first:end])
                    np.testing.assert_array_equal(a['prior_weights'], prior[first:end])
                    np.testing.assert_array_equal(a['catalog_log_likelihoods'], primary[first:end])
                    selection[first:end] = a['selection_kernel_log_factors']
                covered[first:end] = True
            if not covered.all() or not np.isfinite(selection).all() or primary.shape != (len(nodes), 4):
                raise ValueError('Complete matching primary selection required')
            plevel = next(level for level in physical['levels'] if level['order'] == order)
            with array(plevel['arrays']) as a:
                np.testing.assert_array_equal(a['settings'], nodes)
                moments = a['physical_values'].copy()
            prepared[order] = (nodes, prior, selection, primary, moments)
        report['input_verification_complete'] = True
        save()
        print('Inputs verified: all 104 events per arm, full selection, both complete prior grids.', flush=True)

        if args.validate_only:
            report['status'] = 'COMPLETED_INTEGRATED_ROW_INPUT_CHECK'
        else:
            report['status'] = 'RUNNING'
            report['subset_population_fit_run'] = True
            previous = previous_physical = None
            arms = [p['arm'] for p in products]
            for order in (128, 256):
                nodes, prior, selection, primary, moments = prepared[order]
                likelihoods = np.full((len(nodes), 44), np.nan)
                level = dict(order=order, nodes=len(nodes), completed_nodes=0, status='INCOMPLETE',
                             max_full_control_abs_log_residual=0., max_quarter_abs_log_residual=0.)
                report['levels'].append(level)
                save()
                for first in range(0, len(nodes), 128):
                    budget()
                    batch = nodes[first:first+128]
                    end = first+len(batch)
                    factors = np.empty((208, len(batch), 11, 2))
                    quarter_max = 0.
                    for j, product in enumerate(products):
                        for route_index, route in enumerate(ROUTES):
                            values, counts = grouped_kernel_log_factors(product['x'], product['bases'][route], batch, budget=budget)
                            np.testing.assert_array_equal(counts, report['group_row_counts'][j])
                            factors[j, :, :, route_index] = values
                        quarter_max = max(quarter_max, *quarter_residuals(factors[j], counts, product['rows']))
                    catalogs = assemble_catalog(factors, arms, selection[first:end])
                    residual = float(np.max(abs(catalogs[:, :4]-primary[first:end])))
                    if residual > 5e-6:
                        raise ArithmeticError('Full control differs from original primary grid at a node')
                    likelihoods[first:end] = catalogs
                    path = output/f'order_{order:03d}_tile_{first:05d}.npz'
                    np.savez_compressed(path, settings=batch, prior_weights=prior[first:end],
                        event_group_kernel_log_factors=factors, group_row_counts=np.array(report['group_row_counts']),
                        selection_kernel_log_factors=selection[first:end], catalog_log_likelihoods=catalogs)
                    report['checkpoints'].append(dict(order=order, first_node=first, nodes=len(batch),
                        path=str(path.relative_to(ROOT)), sha256=sha(path),
                        full_control_abs_log_residual=residual, quarter_abs_log_residual=quarter_max))
                    level['completed_nodes'] += len(batch)
                    level['max_full_control_abs_log_residual'] = max(level['max_full_control_abs_log_residual'], residual)
                    level['max_quarter_abs_log_residual'] = max(level['max_quarter_abs_log_residual'], quarter_max)
                    save()
                    print(f'order {order}: {level["completed_nodes"]}/{len(nodes)} nodes retained for all 44 columns', flush=True)
                summaries, masses, physical_summaries = [], [], []
                for column in range(44):
                    budget()
                    summary, mass = normalized_grid(nodes, prior, likelihoods[:, column])
                    summaries.append(summary); masses.append(mass)
                    physical_summaries.append(summarize_physical(moments, mass))
                path = output/f'order_{order:03d}_complete.npz'
                np.savez_compressed(path, settings=nodes, prior_weights=prior,
                    catalog_log_likelihoods=likelihoods, posterior_quadrature_masses=np.column_stack(masses))
                level.update(status='COMPLETE_DIAGNOSTIC_GRID_LEVEL', summaries=summaries,
                    physical_summaries=physical_summaries, arrays=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
                    refinement=[refinement_check(a, b) for a, b in zip(previous, summaries)] if previous else None,
                    physical_refinement=[physical_refinement(a, b) for a, b in zip(previous_physical, physical_summaries)] if previous else None)
                previous, previous_physical = summaries, physical_summaries
                save()
            coordinate_pass = all(item['empirical_refinement_pass'] for item in report['levels'][-1]['refinement'])
            physical_pass = all(item['empirical_physical_refinement_pass'] for item in report['levels'][-1]['physical_refinement'])
            report['empirical_quadrature_refinement_pass'] = coordinate_pass
            report['empirical_physical_refinement_pass'] = physical_pass
            report['status'] = ('COMPLETED_INTEGRATED_ROW_SENSITIVITY_REVIEW_REQUIRED' if coordinate_pass and physical_pass else 'STOPPED_ROW_REFINEMENT_UNRESOLVED')
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['changed_inputs'] = []
    for name, digest in bindings.items():
        if not (ROOT/name).exists() or sha(ROOT/name) != digest:
            report['changed_inputs'].append(name)
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output/'summary.json'), elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(not report['status'].startswith('COMPLETED'))


if __name__ == '__main__':
    raise SystemExit(main())
