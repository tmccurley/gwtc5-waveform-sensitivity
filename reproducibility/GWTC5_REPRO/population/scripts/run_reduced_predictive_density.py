"""Numerical representation of existing conditional predictive densities."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'  # Process-local compute setting, not a model change.
HANDOFF = 'results/preflight/reduced_evidence_handoff_20261014T010712696648Z/summary.json'
HANDOFF_SHA = '14d8239fc15be7f29aac834113d1365dcaae189e9916b054e520d73ed0769ce9'


def main():
    import numpy as np
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_predictive_density import predictive_curve, curve_moments, curve_checks, curve_refinement
    from src.reduced_physical_moments import physical_moments
    from src.boundary_regularized_population import BoundaryRegularizedSpin
    from src.independent_spin_model import SpinParameters
    from src.independent_mass_redshift import checked_quad
    started, bindings = time.monotonic(), {}
    output = ROOT/'results/preflight'/('reduced_predictive_density_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    report = dict(status='INCOMPLETE', errors=[], controls=[], levels=[], worker_budget_seconds=900,
                  new_fit_run=False, prior_changed=False, manuscripts_modified=False,
                  actual_campaign_variance_validated=False, physical_population_recovery_validated=False,
                  credible_bands_computed=False, quantiles_computed=False, curve_renormalized=False,
                  native_reference=True, process_blas_threads=1)

    def budget():
        if time.monotonic()-started > 900:
            raise TimeoutError('Predictive-density worker budget exceeded')

    def bind(name, expected=None):
        budget()
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Workspace evidence required')
        value = sha(path)
        if expected and value != expected:
            raise ValueError('Preserved evidence changed: '+name)
        bindings[name] = value
        return path

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        temp = output/'summary.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(output/'summary.json')

    try:
        for name in ('scripts/run_reduced_predictive_density.py', 'scripts/review_reduced_predictive_density.py',
                     'src/reduced_predictive_density.py', 'src/reduced_physical_moments.py',
                     'src/boundary_regularized_population.py', 'src/independent_spin_model.py',
                     'protocol/reduced_predictive_density_plan_2026-10-13.md', 'tests/test_reduced_predictive_density.py'):
            bind(name)
        handoff = json.loads(bind(HANDOFF, HANDOFF_SHA).read_text())
        if handoff['status'] != 'COMPLETED_REDUCED_EVIDENCE_HANDOFF_AUDIT' or handoff['errors'] or handoff['changed_reports']:
            raise ValueError('Completed unchanged handoff required')
        evidence = {}
        for entry in handoff['evidence']:
            evidence[entry['key']] = json.loads(bind(entry['path'], entry['sha256']).read_text())
        grid, physical = evidence['grid'], evidence['physical']
        report['likelihood_columns'] = physical['likelihood_columns']
        report['grid_summary'] = next(v['path'] for v in handoff['evidence'] if v['key'] == 'grid')
        report['physical_summary'] = next(v['path'] for v in handoff['evidence'] if v['key'] == 'physical')
        coarse_x, fine_x = np.linspace(-.99, .99, 2049), np.linspace(-.99, .99, 4097)
        settings = np.array([(mu, ell) for mu in (-1., -.97, 0., .1, .97, 1.) for ell in (-2., -.7, 1.)])
        moments = physical_moments(settings, budget=budget)
        # Each control column is a single normalized density, not an observation.
        a = predictive_curve(settings, moments[:, 0], np.eye(len(settings)), coarse_x, budget=budget)
        b = predictive_curve(settings, moments[:, 0], np.eye(len(settings)), fine_x, budget=budget)
        for index, pars in enumerate(settings):
            budget()
            spin = BoundaryRegularizedSpin(SpinParameters(float(pars[0]), float(pars[1]), 12, 20, 45, 0, 0, -.6, .8, -1, 1), a_max=.99, tail_width=.05)
            canonical = np.exp(spin.logpdf(20., fine_x))
            points = sorted({p for p in [-.94, .94, *(.99*(pars[0]+k*10.**pars[1]) for k in (-8, -3, -1, 0, 1, 3, 8))] if -.99 < p < .99})
            mean = checked_quad(lambda chi: chi*float(np.exp(spin.logpdf(20., chi))), -.99, .99, points=points, epsabs=1e-12, epsrel=1e-11)
            variance = checked_quad(lambda chi: (chi-mean)**2*float(np.exp(spin.logpdf(20., chi))), -.99, .99, points=points, epsabs=1e-12, epsrel=1e-11)
            coarse_check = curve_checks(curve_moments(coarse_x, a[:, index:index+1]), [mean], [np.sqrt(variance)])
            fine_check = curve_checks(curve_moments(fine_x, b[:, index:index+1]), [mean], [np.sqrt(variance)])
            residual = float(abs(canonical-b[:, index]).max()/canonical.max())
            report['controls'].append(dict(settings=pars.tolist(), canonical_peak_scaled_density_residual=residual,
                                          coarse_check=coarse_check, fine_check=fine_check))
            if not fine_check['numerical_moment_pass'] or residual > 1e-8:
                raise ArithmeticError('Fine-grid canonical density control failed')
        report['control_grid_refinement'] = curve_refinement(coarse_x, a, fine_x, b)
        if not report['control_grid_refinement']['numerical_curve_refinement_pass']:
            raise ArithmeticError('Control physical-grid refinement failed')
        controls_path = output/'controls.npz'
        np.savez_compressed(controls_path, settings=settings, chi_coarse=coarse_x, chi_fine=fine_x,
                            control_curves_coarse=a, control_curves_fine=b)
        report['control_arrays'] = dict(path=str(controls_path.relative_to(ROOT)), sha256=sha(controls_path))
        previous = None
        for order in (128, 256):
            level = next(v for v in grid['levels'] if v['order'] == order)
            phy = next(v for v in physical['levels'] if v['order'] == order)
            with np.load(bind(level['arrays']['path'], level['arrays']['sha256']), allow_pickle=False) as arr:
                nodes, prior, q = arr['settings'].copy(), arr['prior_weights'].copy(), arr['posterior_quadrature_masses'].copy()
            with np.load(bind(phy['arrays']['path'], phy['arrays']['sha256']), allow_pickle=False) as arr:
                np.testing.assert_array_equal(nodes, arr['settings'])
                np.testing.assert_array_equal(q, arr['posterior_quadrature_masses'])
                j0 = arr['physical_values'][:, 0].copy()
            if len(nodes) != order**2 or q.shape != (order**2, 4):
                raise ValueError('Complete prior grids and four original columns required')
            coarse = predictive_curve(nodes, j0, q, coarse_x, budget=budget)
            fine = predictive_curve(nodes, j0, q, fine_x, budget=budget)
            expected_mean = np.array([s['predictive_mean'] for s in phy['summaries']])
            expected_sd = np.array([s['predictive_sd'] for s in phy['summaries']])
            check = curve_checks(curve_moments(fine_x, fine), expected_mean, expected_sd)
            physical_refinement = curve_refinement(coarse_x, coarse, fine_x, fine)
            path = output/f'predictive_order_{order}.npz'
            np.savez_compressed(path, settings=nodes, prior_weights=prior, posterior_quadrature_masses=q,
                j0=j0, chi_coarse=coarse_x, chi_fine=fine_x, curves_coarse=coarse, curves_fine=fine)
            entry = dict(order=order, nodes=len(nodes), arrays=dict(path=str(path.relative_to(ROOT)), sha256=sha(path)),
                coarse_moments=curve_moments(coarse_x, coarse), fine_moments=curve_moments(fine_x, fine),
                coarse_check=curve_checks(curve_moments(coarse_x, coarse), expected_mean, expected_sd),
                fine_check=check, physical_grid_refinement=physical_refinement)
            if previous is not None:
                entry['hyperparameter_grid_refinement'] = curve_refinement(fine_x, previous, fine_x, fine)
            report['levels'].append(entry)
            previous = fine
            save()
            print(f'Predictive order {order}: all {len(nodes)} original nodes represented', flush=True)
            if not check['numerical_moment_pass'] or not physical_refinement['numerical_curve_refinement_pass']:
                raise ArithmeticError('Observed predictive representation screen failed; arrays retained')
        if not report['levels'][-1]['hyperparameter_grid_refinement']['numerical_curve_refinement_pass']:
            raise ArithmeticError('Predictive hyperparameter-grid refinement unresolved')
        report['status'] = 'COMPLETED_PREDICTIVE_DENSITY_REVIEW_REQUIRED'
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, value in bindings.items() if sha(ROOT/name) != value]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    save()
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output/'summary.json'))), flush=True)
    return int(report['status'] != 'COMPLETED_PREDICTIVE_DENSITY_REVIEW_REQUIRED')


if __name__ == '__main__':
    raise SystemExit(main())
