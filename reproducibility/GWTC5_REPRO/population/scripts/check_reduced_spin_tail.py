"""Full-prior tail probabilities and conditional empirical-factor sensitivity.

Consumes completed grids only; fits no parameters and modifies no old files.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    import numpy as np
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid
    from src.reduced_physical_moments import physical_moments
    from src.reduced_spin_tail import log_tail_kernel, adaptive_log_tail, tail_summary
    started = time.monotonic()
    output = ROOT/'results/preflight'/('reduced_spin_tail_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    bindings = {}
    report = dict(status='INCOMPLETE', errors=[], controls=[], levels=[], worker_budget_seconds=900,
                  threshold_physical=.965, event_count=104, population_refit_run=False,
                  certified_likelihood_error_bound=False, selection_variance_validated=False,
                  manuscripts_modified=False)

    def budget():
        if time.monotonic()-started > 900:
            raise TimeoutError('Fixed tail diagnostic budget exceeded')

    def bind(name, expected=None):
        budget()
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Only workspace evidence permitted')
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('Preserved input changed: '+name)
        bindings[name] = digest
        return path

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        tmp = output/'summary.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(output/'summary.json')

    try:
        for name in ('scripts/check_reduced_spin_tail.py', 'scripts/review_reduced_spin_tail.py',
                     'src/reduced_spin_tail.py', 'tests/test_reduced_spin_tail.py',
                     'protocol/reduced_tail_measure_audit_2026-10-13.md'):
            bind(name)
        inputs = [
            ('results/preflight/reduced_observed_grid_20261012T132719333655Z/summary.json', '70db7d331bc3bfb841c0fd64b84b0e3bd3c809552582a16dd419f087fee72d31'),
            ('results/preflight/reduced_observed_grid_20261012T132719333655Z/readback_20261012T185545544529Z.json', '02b7dd925a4f9eecb2823423a16e50c83b85ad5e8ac9a545ce8c66810f1883ab'),
            ('results/preflight/reduced_physical_moments_20261012T193150884287Z/summary.json', 'c5ba84d950b3b2f82b633a1f193e598f051daac1b848c34dfcaeca71616410d5'),
            ('results/preflight/reduced_physical_moments_20261012T193150884287Z/readback_20261012T193424563832Z.json', '0989eb56d97fbdf9f4c6b68aa9d17d6731330b644d22489ea5faee0ff721a1fd')]
        grid, checked_grid, physical, checked_physical = [json.loads(bind(name, digest).read_text()) for name, digest in inputs]
        statuses = ('COMPLETED_PROVISIONAL_REDUCED_GRID_REVIEW_REQUIRED', 'COMPLETED_PROVISIONAL_REDUCED_GRID_READBACK',
                    'COMPLETED_PHYSICAL_MOMENTS_REVIEW_REQUIRED', 'COMPLETED_PHYSICAL_MOMENT_READBACK')
        for obj, status in zip((grid, checked_grid, physical, checked_physical), statuses):
            if obj['status'] != status or obj['errors'] or obj['changed_inputs']:
                raise ValueError('Completed unchanged grid and physical readback required')
            for name, digest in obj.get('preserved_hashes', {}).items():
                bind(name, digest)
        report['likelihood_columns'] = grid['likelihood_columns']
        controls = np.array([(m, ell) for m in (-1., -.97, 0., .97, 1.) for ell in (-2., -.7, 1.)])
        norms = physical_moments(controls, budget=budget)[:, 0]
        for setting, j0 in zip(controls, norms):
            budget()
            reference = adaptive_log_tail(setting)
            values = [float(log_tail_kernel([setting], order=n)[0]-np.log(j0)) for n in (64, 128)]
            residual = max(abs(v-reference) for v in values)
            report['controls'].append(dict(settings=setting.tolist(), adaptive_log_tail=reference,
                                          quadrature_log_tails=values, max_absolute_log_residual=residual))
            if residual > 1e-8:
                raise ArithmeticError('Declared adaptive tail control failed')
        previous = None
        for order in (128, 256):
            nodes, _ = prior_grid(order)
            level = next(v for v in grid['levels'] if v['order'] == order)
            phys = next(v for v in physical['levels'] if v['order'] == order)
            with np.load(bind(level['arrays']['path'], level['arrays']['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(nodes, a['settings'])
                masses = a['posterior_quadrature_masses'].copy()
            with np.load(bind(phys['arrays']['path'], phys['arrays']['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(nodes, a['settings'])
                np.testing.assert_array_equal(masses, a['posterior_quadrature_masses'])
                j0 = a['physical_values'][:, 0].copy()
            if not np.isfinite(j0).all() or np.any(j0 <= 0) or masses.shape != (order**2, 4):
                raise ValueError('Complete positive normalization and four posterior columns required')
            factors = np.full(len(nodes), np.nan)
            covered = np.zeros(len(nodes), bool)
            for tile in grid['checkpoints']:
                if tile['order'] != order:
                    continue
                first, count = tile['first_node'], tile['nodes']
                if np.any(covered[first:first+count]):
                    raise ValueError('Overlapping original grid tiles')
                with np.load(bind(tile['path'], tile['sha256']), allow_pickle=False) as a:
                    np.testing.assert_array_equal(nodes[first:first+count], a['settings'])
                    factors[first:first+count] = a['selection_kernel_log_factors']
                covered[first:first+count] = True
            if not covered.all() or not np.isfinite(factors).all():
                raise ValueError('Complete original selection-factor tiles required')
            # Reinsert the common Gaussian constant; never compare a normalized
            # tail probability to an unnormalized likelihood kernel sum.
            log_selection = factors-np.log(.99*j0)
            coarse = log_tail_kernel(nodes, order=64, budget=budget)-np.log(j0)
            fine = log_tail_kernel(nodes, order=128, budget=budget)-np.log(j0)
            change = abs(coarse-fine)
            summaries, penalty_columns = [], []
            for column in range(4):
                summary, penalty = tail_summary(fine, log_selection, masses[:, column])
                summaries.append(summary)
                penalty_columns.append(penalty)
            filename = output/f'tail_order_{order}.npz'
            np.savez_compressed(filename, settings=nodes, log_tail_probability=fine,
                                log_tail_probability_order64=coarse, empirical_log_selection=log_selection,
                                plugin_log_penalty=np.column_stack(penalty_columns),
                                posterior_quadrature_masses=masses)
            entry = dict(order=order, nodes=len(nodes), max_absolute_log_tail_refinement=float(change.max()),
                         spin_refinement_pass=bool(np.all(change <= 1e-8)), summaries=summaries,
                         arrays=dict(path=str(filename.relative_to(ROOT)), sha256=sha(filename)))
            if previous:
                refinement = []
                for a, b in zip(previous, summaries):
                    tail_change = float(-np.expm1(-abs(a['posterior_log_mean_tail_probability']-b['posterior_log_mean_tail_probability'])))
                    scale = max(a['posterior_mean_plugin_log_penalty'], b['posterior_mean_plugin_log_penalty'])
                    penalty_change = abs(a['posterior_mean_plugin_log_penalty']-b['posterior_mean_plugin_log_penalty'])/scale if scale else 0.
                    refinement.append(dict(relative_weighted_tail_change=tail_change,
                                           relative_weighted_plugin_penalty_change=penalty_change,
                                           numerical_refinement_pass=bool(tail_change <= .01 and penalty_change <= .01)))
                entry['refinement'] = refinement
            report['levels'].append(entry)
            previous = summaries
            save()
            print(f'Tail grid {order}: all {len(nodes)} nodes retained', flush=True)
        passed = all(v['spin_refinement_pass'] for v in report['levels']) and all(v['numerical_refinement_pass'] for v in report['levels'][-1]['refinement'])
        report['status'] = 'COMPLETED_TAIL_DIAGNOSTIC_REVIEW_REQUIRED' if passed else 'STOPPED_TAIL_REFINEMENT_UNRESOLVED'
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        report['status'] = 'STOPPED_NEEDS_REVIEW'
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    save()
    print(json.dumps(dict(status=report['status'], report=str(output/'summary.json'), errors=report['errors'])), flush=True)
    return int(report['status'] != 'COMPLETED_TAIL_DIAGNOSTIC_REVIEW_REQUIRED')


if __name__ == '__main__':
    raise SystemExit(main())
