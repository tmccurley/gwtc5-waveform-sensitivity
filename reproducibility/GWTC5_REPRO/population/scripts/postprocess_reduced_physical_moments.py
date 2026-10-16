"""Checked physical-moment postprocessing of the completed provisional grid."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('readback', type=Path)
    args = parser.parse_args()
    import numpy as np
    from scipy.integrate import quad, IntegrationWarning
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid
    from src.reduced_physical_moments import physical_moments, summarize_physical, physical_refinement

    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_physical_moments_'+stamp)
    output.mkdir(exist_ok=False)
    report = dict(status='INCOMPLETE', errors=[], controls=[], levels=[],
                  worker_budget_seconds=900, new_population_fit_run=False,
                  event_exclusions=0, hyperprior_changed=False, manuscripts_modified=False,
                  scientific_result_validated=False, finite_posterior_accuracy_validated=False,
                  selection_variance_validated=False, physical_quantiles_certified=False,
                  density_curves_certified=False, physical_moments_computed=False)
    bindings = {}
    print('Results: '+str(output), flush=True)

    def budget():
        if time.monotonic()-started > 900:
            raise TimeoutError('Fixed physical-moment worker budget exceeded; retain results')

    def bind(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Consumed input must be inside research workspace')
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('Consumed evidence changed: '+str(path))
        bindings[str(path.relative_to(ROOT))] = digest
        return path

    def reference(mu, logsigma):
        # Independent scalar polynomial window and adaptive integration.
        # Use full support, not a fixed Gaussian-tail cutoff.
        sigma, width, a = 10.**logsigma, .05/.99, .99
        points = sorted(set(x for x in [-1+width, 1-width,
                            *(mu+k*sigma for k in (-8, -3, -1, 0, 1, 3, 8))] if -1 < x < 1))

        def kernel(x):
            t = (1-abs(x))/width
            window = 0. if t <= 0 else (1. if t >= 1 else t**3*(10-15*t+6*t*t))
            return window*math.exp(-.5*((x-mu)/sigma)**2)

        def integrate(func):
            with warnings.catch_warnings():
                warnings.simplefilter('error', IntegrationWarning)
                value, error = quad(func, -1, 1, points=points, epsabs=1e-13, epsrel=1e-11, limit=400)
            if not math.isfinite(value) or not math.isfinite(error) or error > 5*max(1e-13, 1e-11*abs(value)):
                raise ArithmeticError('Adaptive moment reference failed')
            return value

        j0 = integrate(kernel)
        if j0 <= 0:
            raise ArithmeticError('Adaptive normalization unavailable')
        m = integrate(lambda x: x*kernel(x))/j0
        v = integrate(lambda x: (x-m)**2*kernel(x))/j0
        if v <= 0:
            raise ArithmeticError('Adaptive variance unavailable')
        return np.array([j0, a*m, a*math.sqrt(v)])

    try:
        summary_path, readback_path = bind(args.summary), bind(args.readback)
        if not summary_path.is_relative_to(ROOT/'results/preflight') or readback_path.parent != summary_path.parent:
            raise ValueError('Use matching saved grid and readback inside preflight')
        worker = json.loads(summary_path.read_text())
        checked = json.loads(readback_path.read_text())
        if worker['status'] != 'COMPLETED_PROVISIONAL_REDUCED_GRID_REVIEW_REQUIRED' or worker['errors'] or worker['changed_inputs']:
            raise ValueError('Completed provisional grid required')
        if checked['status'] != 'COMPLETED_PROVISIONAL_REDUCED_GRID_READBACK' or checked['errors'] or checked['changed_inputs']:
            raise ValueError('Successful independent readback required')
        if checked['preserved_hashes'].get(str(summary_path.relative_to(ROOT))) != bindings[str(summary_path.relative_to(ROOT))]:
            raise ValueError('Grid summary differs from verified readback')
        for name in ('src/boundary_regularized_population.py', 'src/independent_spin_model.py', 'src/reduced_gaussian_grid.py'):
            bind(ROOT/name, worker['preserved_hashes'][name])
        for name in ('protocol/reduced_physical_moments_lock_2026-10-12.md',
                     'protocol/reduced_post_grid_checks_plan_2026-10-12.md',
                     'src/reduced_physical_moments.py', 'tests/test_reduced_physical_moments.py',
                     'scripts/postprocess_reduced_physical_moments.py'):
            bind(ROOT/name)
        if worker['fixed_models']['a_max'] != .99 or worker['fixed_models']['tail_width'] != .05:
            raise ValueError('Physical model settings differ from locked postprocessing')
        report['grid_summary'] = str(summary_path.relative_to(ROOT))
        report['grid_readback'] = str(readback_path.relative_to(ROOT))
        report['likelihood_columns'] = worker['likelihood_columns']

        specs = [(mu, ls) for mu in (-1., -.97, -.2, 0., .1, .97, 1.) for ls in (-2., -.7, 1.)]+[(0., 0.)]
        numerical = physical_moments(specs, order=64, budget=budget)
        for spec, actual in zip(specs, numerical):
            budget()
            expected = reference(*spec)
            errors = [abs(actual[0]-expected[0])/expected[0], *abs(actual[1:]-expected[1:]).tolist()]
            report['controls'].append(dict(settings=list(spec), actual=actual.tolist(),
                                          reference=expected.tolist(), discrepancies=errors))
            if max(errors) > 1e-9:
                raise ArithmeticError('Independent physical-moment control failed')
        print('All 22 independent adaptive physical-moment controls passed.', flush=True)

        previous = None
        for order in (128, 256):
            level = next(v for v in worker['levels'] if v['order'] == order)
            array_path = bind(ROOT/level['arrays']['path'], level['arrays']['sha256'])
            with np.load(array_path, allow_pickle=False) as arrays:
                settings, weights = prior_grid(order)
                np.testing.assert_array_equal(settings, arrays['settings'])
                np.testing.assert_array_equal(weights, arrays['prior_weights'])
                masses = arrays['posterior_quadrature_masses'].copy()
            if masses.shape != (len(settings), 4):
                raise ValueError('Four full-node posterior columns required')
            fine = physical_moments(settings, order=64, budget=budget)
            coarse = physical_moments(settings, order=32, budget=budget)
            discrepancies = dict(kernel_integral_relative=float(np.max(abs(fine[:, 0]-coarse[:, 0])/fine[:, 0])),
                                 mean_absolute=float(np.max(abs(fine[:, 1]-coarse[:, 1]))),
                                 sd_absolute=float(np.max(abs(fine[:, 2]-coarse[:, 2]))))
            entry = dict(order=order, nodes=len(settings), spin_quadrature_discrepancies=discrepancies, summaries=[])
            report['levels'].append(entry)
            data_path = output/f'physical_order_{order:03d}.npz'
            np.savez_compressed(data_path, settings=settings, physical_values=fine,
                                physical_values_order32=coarse, posterior_quadrature_masses=masses)
            entry['arrays'] = dict(path=str(data_path.relative_to(ROOT)), sha256=sha(data_path))
            if max(discrepancies.values()) > 1e-9:
                raise ArithmeticError('Full-node spin integration refinement unresolved')
            summaries = [summarize_physical(fine, masses[:, k]) for k in range(4)]
            entry['summaries'] = summaries
            if previous:
                entry['refinement'] = [physical_refinement(a, b) for a, b in zip(previous, summaries)]
            previous = summaries
            print(f'Physical moments retained for all {len(settings)} order-{order} nodes.', flush=True)
        report['physical_moments_computed'] = True
        passed = all(v['empirical_physical_refinement_pass'] for v in report['levels'][-1]['refinement'])
        report['empirical_physical_refinement_pass'] = passed
        report['status'] = ('COMPLETED_PHYSICAL_MOMENTS_REVIEW_REQUIRED' if passed else 'STOPPED_PHYSICAL_REFINEMENT_UNRESOLVED')
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        report['status'] = 'STOPPED_NEEDS_REVIEW'
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['elapsed_seconds'] = time.monotonic()-started
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output/'summary.json'),
                         elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(report['status'] != 'COMPLETED_PHYSICAL_MOMENTS_REVIEW_REQUIRED')


if __name__ == '__main__':
    raise SystemExit(main())
