"""Saved-array checks and independently arranged full-node curve replay."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    import numpy as np
    from scipy.special import ndtr
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid
    from src.boundary_regularized_population import BoundaryRegularizedSpin, log_boundary_window
    from src.independent_spin_model import SpinParameters
    source = args.summary.resolve()
    if not source.is_relative_to(ROOT/'results/preflight'):
        raise ValueError('Workspace worker summary required')
    started, bindings = time.monotonic(), {}
    report = dict(status='INCOMPLETE', errors=[], levels=[], point_replays=[], normalizer_replays=[],
                  worker_budget_seconds=900, new_fit_run=False, actual_campaign_variance_validated=False)

    def bind(name, expected=None):
        if time.monotonic()-started > 900:
            raise TimeoutError('Density readback budget exceeded')
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Workspace evidence required')
        value = sha(path)
        if expected and value != expected:
            raise ValueError('Preserved input changed: '+name)
        bindings[name] = value
        return path

    try:
        worker = json.loads(bind(str(source.relative_to(ROOT))).read_text())
        if worker['status'] != 'COMPLETED_PREDICTIVE_DENSITY_REVIEW_REQUIRED' or worker['errors'] or worker['changed_inputs']:
            raise ValueError('Successful completed worker required')
        for name, value in worker['preserved_hashes'].items():
            bind(name, value)
        grid = json.loads((ROOT/worker['grid_summary']).read_text())
        physical = json.loads((ROOT/worker['physical_summary']).read_text())
        previous = None
        for level in worker['levels']:
            order, entry = level['order'], level['arrays']
            with np.load(bind(entry['path'], entry['sha256']), allow_pickle=False) as a:
                nodes, prior, q, j0, x, y, coarse, fine = [a[k].copy() for k in
                    ('settings', 'prior_weights', 'posterior_quadrature_masses', 'j0', 'chi_coarse', 'chi_fine', 'curves_coarse', 'curves_fine')]
            original_nodes, original_prior = prior_grid(order)
            np.testing.assert_array_equal(nodes, original_nodes)
            np.testing.assert_array_equal(prior, original_prior)
            np.testing.assert_array_equal(x, y[::2])
            np.testing.assert_array_equal(y, np.linspace(-.99, .99, 4097))
            original = next(v for v in grid['levels'] if v['order'] == order)['arrays']
            phy = next(v for v in physical['levels'] if v['order'] == order)
            with np.load(bind(original['path'], original['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(q, a['posterior_quadrature_masses'])
            with np.load(bind(phy['arrays']['path'], phy['arrays']['sha256']), allow_pickle=False) as a:
                np.testing.assert_array_equal(j0, a['physical_values'][:, 0])
            for z, curves, saved in ((x, coarse, level['coarse_moments']), (y, fine, level['fine_moments'])):
                if curves.shape != (len(z), 4) or not np.isfinite(curves).all() or np.any(curves < 0):
                    raise ValueError('Four complete finite density columns required')
                np.testing.assert_array_equal(curves[[0, -1]], 0)
                dz = np.diff(z)[:, None]
                integral = np.sum((curves[1:]+curves[:-1])*dz/2, axis=0)
                mean = np.sum((z[1:, None]*curves[1:]+z[:-1, None]*curves[:-1])*dz/2, axis=0)
                centered = (z[:, None]-mean)**2*curves
                sd = np.sqrt(np.sum((centered[1:]+centered[:-1])*dz/2, axis=0))
                for actual, key in ((integral, 'integrals'), (mean, 'predictive_means'), (sd, 'predictive_sds')):
                    np.testing.assert_allclose(actual, saved[key], atol=1e-11, rtol=0)
                if len(z) == 4097:
                    expected_mean = np.array([s['predictive_mean'] for s in phy['summaries']])
                    expected_sd = np.array([s['predictive_sd'] for s in phy['summaries']])
                    if np.any(abs(integral-1) > 1e-6) or np.any(abs(mean-expected_mean) > 1e-6) or np.any(abs(sd-expected_sd) > 1e-6):
                        raise ArithmeticError('Fine curve moment readback failed')
            indices = np.linspace(0, len(y)-1, 16, dtype=int)
            chi = y[indices]
            exponent = -(chi[None, :]-.99*nodes[:, 0, None])**2/(2*(.99*10.**nodes[:, 1, None])**2)
            density = np.exp(exponent+log_boundary_window(chi/.99, .05/.99)[None, :])/(.99*j0[:, None])
            replay = np.sum(density[:, :, None]*q[:, None, :], axis=0)
            peak_scaled = float(np.max(abs(replay-fine[indices])/fine.max(axis=0)))
            if peak_scaled > 1e-8:
                raise ArithmeticError('Full-node independent curve replay failed')
            report['point_replays'].append(dict(order=order, physical_indices=indices.tolist(), nodes_per_point=len(nodes),
                                                max_peak_scaled_residual=peak_scaled))
            for index in (0, len(nodes)//2, len(nodes)-1):
                mu, ell = nodes[index]
                spin = BoundaryRegularizedSpin(SpinParameters(float(mu), float(ell), 12, 20, 45, 0, 0, -.6, .8, -1, 1), a_max=.99, tail_width=.05)
                sigma = 10.**ell
                reference = spin.gaussian_norm*sigma*math.sqrt(2*math.pi)*(ndtr((1-mu)/sigma)-ndtr((-1-mu)/sigma))
                residual = abs(math.log(reference/j0[index]))
                if residual > 1e-8:
                    raise ArithmeticError('Canonical normalizer mismatch')
                report['normalizer_replays'].append(dict(order=order, node=index, absolute_log_residual=residual))
            peak = np.maximum(coarse.max(axis=0), fine.max(axis=0))
            pointwise = np.max(abs(coarse-fine[::2]), axis=0)/peak
            interpolated = np.column_stack([np.interp(y, x, coarse[:, c]) for c in range(4)])
            l1 = np.trapz(abs(interpolated-fine), y, axis=0)
            if np.any(pointwise > .01) or np.any(l1 > .01):
                raise ArithmeticError('Physical grid refinement readback failed')
            entry = dict(order=order, nodes=len(nodes), physical_pointwise_changes=pointwise.tolist(), physical_l1_changes=l1.tolist())
            if previous is not None:
                pointwise = np.max(abs(previous-fine), axis=0)/np.maximum(previous.max(axis=0), fine.max(axis=0))
                l1 = np.trapz(abs(previous-fine), y, axis=0)
                if np.any(pointwise > .01) or np.any(l1 > .01):
                    raise ArithmeticError('Hyperparameter grid refinement readback failed')
                entry['hyperparameter_pointwise_changes'] = pointwise.tolist()
                entry['hyperparameter_l1_changes'] = l1.tolist()
            report['levels'].append(entry)
            previous = fine
        if len(report['levels']) != 2:
            raise ValueError('Both complete predictive levels required')
        report['status'] = 'COMPLETED_PREDICTIVE_DENSITY_READBACK'
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, value in bindings.items() if sha(ROOT/name) != value]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['elapsed_seconds'] = time.monotonic()-started
    path = source.parent/('readback_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(path))), flush=True)
    return int(report['status'] != 'COMPLETED_PREDICTIVE_DENSITY_READBACK')


if __name__ == '__main__':
    raise SystemExit(main())
