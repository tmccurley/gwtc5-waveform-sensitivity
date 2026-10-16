"""Checkpointed full-prior conditional reference diagnostic, without refitting."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
PREFLIGHT = 'results/preflight/reduced_selection_grid_preflight_20261013T134844855276Z/summary.json'
PREFLIGHT_SHA = 'fb80d6f29af1631c9ddfa286872d41f655208f29610be18ea50a11ca471bd46a'


def main():
    import numpy as np
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid
    from src.reduced_selection_grid_reference import paired_selection_reference
    from src.reduced_selection_reference_summary import reference_rms, reference_rms_refinement
    started = time.monotonic()
    lock = (ROOT/'results/preflight/reduced_selection_grid_reference.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('An existing selection-grid worker holds the lock; no duplicate started.', flush=True)
        return 2
    output = ROOT/'results/preflight'/('reduced_selection_grid_reference_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    report = dict(status='PREPARING', errors=[], levels=[], checkpoints=[], pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), worker_budget_seconds=21600,
        actual_campaign_variance_validated=False, new_population_fit_run=False,
        stochastic_selection_field_adopted=False, original_selection_estimator_modified=False,
        hyperprior_changed=False, manuscripts_modified=False, event_exclusions=0, retained_events_per_arm=104,
        selection_rows=504666, total_generated_cumulative=1568035640,
        reference_columns=['fixed_N', 'cadence_conditioned_stopped'], empirical_reference_rms_refinement_pass=False)
    bindings = {}
    print('Results: '+str(output), flush=True)

    def budget():
        if time.monotonic()-started > 21600:
            raise TimeoutError('Six-hour selection-grid diagnostic cap exceeded; tiles preserved')

    def bind(name, expected=None):
        budget()
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Consume only workspace evidence')
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('Preserved evidence changed: '+name)
        if name in bindings and bindings[name] != digest:
            raise ValueError('Conflicting evidence binding: '+name)
        bindings[name] = digest
        return path

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['preserved_hashes'] = bindings.copy()
        temp = output/'summary.pending.json'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(output/'summary.json')

    try:
        save()
        preflight = json.loads(bind(PREFLIGHT, PREFLIGHT_SHA).read_text())
        if preflight['status'] != 'COMPLETED_SELECTION_GRID_REFERENCE_PREFLIGHT' or preflight['errors'] or preflight['changed_inputs']:
            raise ValueError('Completed pair arithmetic/timing preflight required')
        for name, digest in preflight['preserved_hashes'].items():
            bind(name, digest)
        for name in ('scripts/run_reduced_selection_grid_reference.py', 'scripts/review_reduced_selection_grid_reference.py',
                     'src/reduced_selection_reference_summary.py', 'tests/test_reduced_selection_reference_summary.py',
                     'protocol/reduced_selection_grid_reference_lock_2026-10-13.md'):
            bind(name)
        point_name = 'results/preflight/reduced_selection_reference_20261013T133416581483Z/summary.json'
        point = json.loads((ROOT/point_name).read_text())
        kernel_name = 'results/preflight/reduced_kernel_preflight_20261012T132102172011Z/summary.json'
        kernel = json.loads((ROOT/kernel_name).read_text())
        cache = kernel['selection_cache']; panel = point['arrays']
        with np.load(bind(cache['path'], cache['sha256']), allow_pickle=False) as a:
            x, base, rows = a['x'].copy(), a['log_base'].copy(), a['row_index'].copy()
            if int((~a['evaluated']).sum()) != 18634:
                raise ValueError('Support-zero ledger changed')
        with np.load(bind(panel['path'], panel['sha256']), allow_pickle=False) as a:
            np.testing.assert_array_equal(rows, a['row_index'])
            strata, n, r = a['strata'].copy(), a['generated_by_month'].copy(), a['accepted_by_month_inferred'].copy()
            anchor_settings = a['settings'][0].copy()
        if x.shape != (504666,) or cache['total_generated'] != 1568035640:
            raise ValueError('Original selection counts changed')
        np.testing.assert_array_equal(anchor_settings, preflight['anchor_settings'])
        anchor = base-.5*((x-anchor_settings[0])/10.**anchor_settings[1])**2
        anchor_diagonals = np.array([point['fixed_reference']['relative_covariance'][0][0], point['stopped_reference']['relative_covariance'][0][0]])
        grid_name = 'results/preflight/reduced_observed_grid_20261012T132719333655Z/summary.json'
        grid = json.loads((ROOT/grid_name).read_text())
        bind('src/reduced_gaussian_grid.py', grid['preserved_hashes']['src/reduced_gaussian_grid.py'])
        report['grid_summary'] = grid_name
        report['likelihood_columns'] = grid['likelihood_columns']
        report['anchor_settings'] = anchor_settings.tolist()
        report['monthly_accepted_counts_independently_observed'] = False
        report['monthly_count_basis'] = point['monthly_count_basis']
        report['status'] = 'RUNNING'
        previous = None
        for order in (128, 256):
            original = next(v for v in grid['levels'] if v['order'] == order)
            entry = original['arrays']
            nodes, prior = prior_grid(order)
            with np.load(bind(entry['path'], entry['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(nodes, a['settings'])
                np.testing.assert_array_equal(prior, a['prior_weights'])
                masses = a['posterior_quadrature_masses'].copy()
            if masses.shape != (len(nodes), 4):
                raise ValueError('Four complete primary posterior columns required')
            shapes = np.full((len(nodes), 2), np.nan)
            level = dict(order=order, nodes=len(nodes), completed_nodes=0, status='INCOMPLETE', max_anchor_diagonal_residual=0.)
            report['levels'].append(level)
            save()
            for first in range(0, len(nodes), 128):
                batch = nodes[first:first+128]
                matrices = np.empty((len(batch), 2, 2, 2))
                values = np.empty((len(batch), 2))
                for k, pars in enumerate(batch):
                    budget()
                    pair = paired_selection_reference(x, base, pars, anchor, strata, n, r)
                    matrices[k] = [pair['fixed_relative_covariance'], pair['stopped_relative_covariance']]
                    values[k] = [pair['fixed_catalog_shape_reference_sd'], pair['stopped_catalog_shape_reference_sd']]
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ArithmeticError('Invalid complete-node reference values')
                residual = float(np.max(abs(matrices[:, :, 0, 0]-anchor_diagonals)))
                if residual > 1e-14:
                    raise ArithmeticError('Anchor covariance diagonal changed between pairs')
                shapes[first:first+len(batch)] = values
                path = output/f'order_{order:03d}_tile_{first:05d}.npz'
                np.savez_compressed(path, settings=batch, relative_covariances=matrices, catalog_shape_reference_sds=values)
                report['checkpoints'].append(dict(order=order, first_node=first, nodes=len(batch),
                    path=str(path.relative_to(ROOT)), sha256=sha(path), max_anchor_diagonal_residual=residual))
                level['completed_nodes'] += len(batch)
                level['max_anchor_diagonal_residual'] = max(level['max_anchor_diagonal_residual'], residual)
                save()
                if len(report['checkpoints']) % 4 == 0 or level['completed_nodes'] == len(nodes):
                    print(f'order {order}: {level["completed_nodes"]}/{len(nodes)} selection-reference nodes retained', flush=True)
            summaries = [reference_rms(shapes, masses[:, column]).tolist() for column in range(4)]
            path = output/f'order_{order:03d}_complete.npz'
            np.savez_compressed(path, settings=nodes, prior_weights=prior,
                posterior_quadrature_masses=masses, catalog_shape_reference_sds=shapes)
            level.update(status='COMPLETE_CONDITIONAL_REFERENCE_LEVEL', reference_rms_by_posterior_column=summaries,
                arrays=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
                refinement=[reference_rms_refinement(a,b) for a,b in zip(previous,summaries)] if previous else None)
            previous = summaries
            save()
        passed = all(v['numerical_refinement_pass'] for v in report['levels'][-1]['refinement'])
        report['empirical_reference_rms_refinement_pass'] = passed
        report['status'] = 'COMPLETED_SELECTION_GRID_REFERENCE_REVIEW_REQUIRED' if passed else 'STOPPED_SELECTION_REFERENCE_REFINEMENT_UNRESOLVED'
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['changed_inputs'] = [name for name,digest in bindings.items() if not (ROOT/name).exists() or sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output/'summary.json'))), flush=True)
    return int(report['status'] != 'COMPLETED_SELECTION_GRID_REFERENCE_REVIEW_REQUIRED')


if __name__ == '__main__':
    raise SystemExit(main())
