"""Independent saved-array readback and adaptive tail/normalization replay."""

import argparse
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
    from scipy.special import logsumexp, ndtr
    from scripts.check_reduced_spin_feasibility import sha
    from src.boundary_regularized_population import BoundaryRegularizedSpin
    from src.independent_spin_model import SpinParameters
    from src.reduced_spin_tail import adaptive_log_tail
    started, bindings = time.monotonic(), {}
    source = args.summary.resolve()
    if not source.is_relative_to(ROOT/'results/preflight'):
        raise ValueError('Workspace summary required')
    output = source.parent/('readback_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    report = dict(status='INCOMPLETE', errors=[], levels=[], adaptive_replays=[], worker_budget_seconds=900,
                  certified_likelihood_error_bound=False, population_refit_run=False)

    def bind(name, expected=None):
        if time.monotonic()-started > 900:
            raise TimeoutError('Tail readback budget exceeded')
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Workspace input required')
        digest = sha(path)
        if expected and expected != digest:
            raise ValueError('Preserved input changed: '+name)
        bindings[name] = digest
        return path

    try:
        worker = json.loads(bind(str(source.relative_to(ROOT))).read_text())
        if worker['status'] not in ('COMPLETED_TAIL_DIAGNOSTIC_REVIEW_REQUIRED', 'STOPPED_TAIL_REFINEMENT_UNRESOLVED') or worker['errors'] or worker['changed_inputs']:
            raise ValueError('Both complete unchanged tail levels required')
        for name, digest in worker['preserved_hashes'].items():
            bind(name, digest)
        grid = json.loads((ROOT/'results/preflight/reduced_observed_grid_20261012T132719333655Z/summary.json').read_text())
        physical = json.loads((ROOT/'results/preflight/reduced_physical_moments_20261012T193150884287Z/summary.json').read_text())
        for level in worker['levels']:
            order, entry = level['order'], level['arrays']
            if order not in (128, 256) or level['nodes'] != order**2:
                raise ValueError('Complete full-prior grid required')
            with np.load(bind(entry['path'], entry['sha256']), allow_pickle=False) as a:
                settings, logs, coarse, sel, masses, penalty = [a[k].copy() for k in
                    ('settings', 'log_tail_probability', 'log_tail_probability_order64', 'empirical_log_selection',
                     'posterior_quadrature_masses', 'plugin_log_penalty')]
            old = next(v for v in grid['levels'] if v['order'] == order)['arrays']
            phy = next(v for v in physical['levels'] if v['order'] == order)['arrays']
            with np.load(bind(old['path'], old['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(settings, a['settings'])
                np.testing.assert_array_equal(masses, a['posterior_quadrature_masses'])
            with np.load(bind(phy['path'], phy['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(settings, a['settings'])
                j0 = a['physical_values'][:, 0].copy()
            covered = np.zeros(len(settings), bool)
            for tile in grid['checkpoints']:
                if tile['order'] != order:
                    continue
                first, count = tile['first_node'], tile['nodes']
                if covered[first:first+count].any():
                    raise ValueError('Repeated source tile')
                with np.load(bind(tile['path'], tile['sha256']), allow_pickle=False) as a:
                    np.testing.assert_allclose(sel[first:first+count]+np.log(.99*j0[first:first+count]),
                                               a['selection_kernel_log_factors'], atol=2e-11, rtol=0)
                covered[first:first+count] = True
            if not covered.all() or not np.isfinite(logs).all() or np.any(logs > 1e-10):
                raise ValueError('Complete finite log probabilities required')
            derived_penalty = 104*np.log1p(np.exp(-abs(logs-sel)))+104*np.maximum(logs-sel, 0)
            np.testing.assert_allclose(penalty, derived_penalty[:, None]*np.ones((1, 4)), atol=1e-9, rtol=1e-12)
            for column, saved in enumerate(level['summaries']):
                q = masses[:, column]
                np.testing.assert_allclose(q.sum(), 1, atol=1e-10, rtol=0)
                with np.errstate(divide='ignore'):
                    actual_log = float(logsumexp(np.log(q)+logs))
                actual_mean = float(np.sum(q*derived_penalty))
                np.testing.assert_allclose(actual_log, saved['posterior_log_mean_tail_probability'], atol=1e-11, rtol=0)
                np.testing.assert_allclose(actual_mean, saved['posterior_mean_plugin_log_penalty'], atol=1e-12, rtol=1e-12)
                for cutoff, value in saved['posterior_mass_above_plugin_penalty'].items():
                    np.testing.assert_allclose(q[derived_penalty > float(cutoff)].sum(), value, atol=1e-12, rtol=0)
            max_residual = 0.
            for index in np.linspace(0, len(settings)-1, 16, dtype=int):
                pars = settings[index]
                reference = adaptive_log_tail(pars)
                residual = abs(reference-float(logs[index]))
                if residual > 1e-8:
                    raise ArithmeticError('Adaptive original-node replay failed')
                spin = BoundaryRegularizedSpin(SpinParameters(float(pars[0]), float(pars[1]), 12, 20, 45, 0, 0, -.6, .8, -1, 1), a_max=.99, tail_width=.05)
                sigma = 10.**pars[1]
                truncation = ndtr((1-pars[0])/sigma)-ndtr((-1-pars[0])/sigma)
                canonical_j0 = spin.gaussian_norm*sigma*math.sqrt(2*math.pi)*truncation
                norm_residual = abs(math.log(canonical_j0)-math.log(j0[index]))
                if norm_residual > 1e-8:
                    raise ArithmeticError('Canonical full-support normalization replay failed')
                max_residual = max(max_residual, residual)
                report['adaptive_replays'].append(dict(order=order, node=int(index), settings=pars.tolist(),
                    log_tail_residual=residual, canonical_log_normalizer_residual=norm_residual))
            report['levels'].append(dict(order=order, all_nodes_checked=len(settings),
                max_absolute_log_tail_refinement=float(abs(logs-coarse).max()),
                max_adaptive_replay_residual=max_residual))
        report['worker_refinement_status_preserved'] = worker['status']
        report['status'] = 'COMPLETED_TAIL_DIAGNOSTIC_READBACK'
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['elapsed_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], report=str(output), errors=report['errors'])), flush=True)
    return int(report['status'] != 'COMPLETED_TAIL_DIAGNOSTIC_READBACK')


if __name__ == '__main__':
    raise SystemExit(main())
