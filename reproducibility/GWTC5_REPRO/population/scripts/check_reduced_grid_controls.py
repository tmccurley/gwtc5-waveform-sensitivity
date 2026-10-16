"""Known-answer two-coordinate integration and a labeled Gaussian toy catalog."""

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
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import ndtr, betaln
    from scipy.stats import truncnorm
    from src.reduced_gaussian_grid import prior_grid, normalized_grid, refinement_check
    from scripts.check_reduced_spin_feasibility import sha
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_grid_controls_'+stamp)
    output.mkdir(exist_ok=False)
    report = dict(status='INCOMPLETE', errors=[], controls=[], all_data_synthetic=True,
                  real_event_rows_read=False, real_selection_rows_read=False,
                  physical_windowed_model_recovery_validated=False, master_seed=2026101201,
                  worker_budget_seconds=900, real_population_fit_run=False)
    names = ['protocol/reduced_fit_lock_2026-10-12.md', 'scripts/check_reduced_grid_controls.py',
             'src/reduced_gaussian_grid.py', 'tests/test_reduced_gaussian_grid.py',
             'scripts/check_reduced_spin_feasibility.py']
    bindings = {name: sha(ROOT/name) for name in names}
    print('Results: '+str(output), flush=True)

    def budget():
        if time.monotonic()-started > 900:
            raise TimeoutError('Fixed synthetic-control budget exceeded')

    def errors(actual, reference):
        return [abs(actual['log_normalization']-reference['log_normalization']),
                *abs(np.asarray(actual['mean'])-reference['mean']).tolist(),
                *abs(np.asarray(actual['sd'])-reference['sd']).tolist()]

    def gaussian_reference(center, scales):
        bounds = [(-1., 1.), (-2., 1.)]
        means, sds, logz = [], [], -math.log(6)
        for c, s, (lo, hi) in zip(center, scales, bounds):
            aa, bb = (lo-c)/s, (hi-c)/s
            prob = ndtr(bb)-ndtr(aa)
            m, var = truncnorm.stats(aa, bb, loc=c, scale=s, moments='mv')
            means.append(float(m)); sds.append(math.sqrt(float(var)))
            logz += math.log(s*math.sqrt(2*math.pi)*prob)
        return dict(log_normalization=logz, mean=means, sd=sds)

    try:
        control_specs = [('uniform', None), ('polynomial_beta', None),
                         ('broad_gaussian', ([.13, -.8], [.08, .17])),
                         ('concentrated_gaussian', ([.17, -.85], [.025, .05]))]
        for name, pars in control_specs:
            budget()
            if name == 'uniform':
                reference = dict(log_normalization=0., mean=[0., -.5], sd=[1/math.sqrt(3), math.sqrt(.75)])
            elif name == 'polynomial_beta':
                shapes = [(2., 4.), (3., 2.)]
                mean = [a/(a+b) for a, b in shapes]
                sd = [math.sqrt(a*b/((a+b)**2*(a+b+1))) for a, b in shapes]
                reference = dict(log_normalization=0., mean=[2*mean[0]-1, 3*mean[1]-2], sd=[2*sd[0], 3*sd[1]])
            else:
                reference = gaussian_reference(*pars)
            saved = dict(name=name, reference=reference, levels=[])
            report['controls'].append(saved)
            previous = None
            for order in (32, 64, 128, 256):
                nodes, weights = prior_grid(order)
                if name == 'uniform':
                    logs = np.zeros(len(nodes))
                elif name == 'polynomial_beta':
                    unit = np.column_stack(((nodes[:, 0]+1)/2, (nodes[:, 1]+2)/3))
                    logs = sum((a-1)*np.log(unit[:, i])+(b-1)*np.log1p(-unit[:, i])-betaln(a, b)
                               for i, (a, b) in enumerate(shapes))
                else:
                    logs = -.5*np.sum(((nodes-pars[0])/pars[1])**2, axis=1)
                actual, _ = normalized_grid(nodes, weights, logs)
                saved['levels'].append(dict(order=order, summary=actual,
                    absolute_reference_errors=errors(actual, reference),
                    refinement=refinement_check(previous, actual) if previous else None))
                previous = actual
            if max(saved['levels'][-1]['absolute_reference_errors']) > 1e-8:
                raise ArithmeticError('Final known-answer control failed: '+name)
            print('Known-answer control retained: '+name, flush=True)

        # A grid can miss a much narrower peak. Preserve this failure as a
        # negative control instead of claiming global mode certification.
        center, scales = [.123456, -.765432], [.0007, .0007]
        reference = gaussian_reference(center, scales)
        nodes, weights = prior_grid(256)
        negative = dict(name='deliberately_underresolved_gaussian', reference=reference)
        report['negative_control'] = negative
        try:
            actual, _ = normalized_grid(nodes, weights, -.5*np.sum(((nodes-center)/scales)**2, axis=1))
            residual = errors(actual, reference)
            negative.update(summary=actual, absolute_reference_errors=residual,
                            status='EXPECTED_UNDERRESOLUTION_RETAINED' if max(residual) > 1e-8 else 'UNEXPECTED_PASS')
        except ArithmeticError as error:
            negative.update(status='EXPECTED_UNDERRESOLUTION_RETAINED', error=str(error))
        if negative['status'] != 'EXPECTED_UNDERRESOLUTION_RETAINED':
            raise ArithmeticError('Declared under-resolution control unexpectedly passed')

        rng = np.random.Generator(np.random.PCG64(2026101201))
        latent = rng.normal(.1, .2, 48)
        observed = latent+rng.normal(0., .08, 48)
        report['toy'] = dict(latent=latent.tolist(), observed=observed.tolist(),
            generating_mu=.1, generating_sigma=.2, observation_sd=.08,
            unbounded_latent_normal=True, certain_detection=True, levels=[])
        n, average = observed.size, float(observed.mean())
        sse = float(np.sum((observed-average)**2))

        def log_mu_integral(ls):
            variance = 10.**(2*ls)+.08**2
            scale = math.sqrt(variance/n)
            probability = ndtr((1-average)/scale)-ndtr((-1-average)/scale)
            return -.5*n*math.log(2*math.pi*variance)-sse/(2*variance)+math.log(scale*math.sqrt(2*math.pi)*probability)

        anchor = log_mu_integral(-.7)
        quadrature_errors = []
        def integrand(ls, moment):
            variance = 10.**(2*ls)+.08**2
            scale = math.sqrt(variance/n)
            conditional_mean, conditional_var = truncnorm.stats((-1-average)/scale, (1-average)/scale,
                                                                loc=average, scale=scale, moments='mv')
            value = [1., float(conditional_mean), float(conditional_var+conditional_mean**2), ls, ls*ls][moment]
            return math.exp(log_mu_integral(ls)-anchor)*value/6
        integrals = []
        for moment in range(5):
            value, error = quad(lambda ls: integrand(ls, moment), -2, 1,
                                epsabs=1e-12, epsrel=1e-11, points=[-1., -.7, -.4])
            integrals.append(value); quadrature_errors.append(error)
        z, mu1, mu2, ls1, ls2 = integrals
        reference = dict(log_normalization=anchor+math.log(z), mean=[mu1/z, ls1/z],
                         sd=[math.sqrt(mu2/z-(mu1/z)**2), math.sqrt(ls2/z-(ls1/z)**2)])
        report['toy']['adaptive_reference'] = reference
        report['toy']['adaptive_absolute_error_estimates'] = quadrature_errors
        previous = None
        for order in (32, 64, 128, 256):
            budget()
            nodes, weights = prior_grid(order)
            variance = 10.**(2*nodes[:, 1])+.08**2
            logs = -.5*n*np.log(2*math.pi*variance)-(sse+n*(nodes[:, 0]-average)**2)/(2*variance)
            actual, _ = normalized_grid(nodes, weights, logs)
            report['toy']['levels'].append(dict(order=order, summary=actual,
                absolute_reference_errors=errors(actual, reference),
                refinement=refinement_check(previous, actual) if previous else None))
            previous = actual
        if max(report['toy']['levels'][-1]['absolute_reference_errors']) > 1e-6:
            raise ArithmeticError('Synthetic toy does not match its independent reference')
        report['status'] = 'COMPLETED_REDUCED_GRID_NUMERICAL_CONTROLS_REVIEW_REQUIRED'
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
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(path), elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(bool(report['errors'] or report['changed_inputs']))


if __name__ == '__main__':
    raise SystemExit(main())
