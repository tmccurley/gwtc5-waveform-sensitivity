"""Bounded saved-array feasibility check; no fit or native waveform calls."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def weight_summary(values, denominator):
    """Preserve support zeros and distinguish concentration from accuracy."""
    import numpy as np
    from scipy.special import logsumexp
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or denominator <= 0:
        raise ValueError('Nonempty log-weight vector and positive denominator required')
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('Invalid log weight')
    finite = np.isfinite(values)
    if not finite.any():
        raise ArithmeticError('All weights have zero support; no factor substituted')
    total = float(logsumexp(values))
    # Center before both reductions to avoid cancellation for extreme tails.
    centered = values[finite] - values[finite].max()
    lsum = float(logsumexp(centered))
    log_kish = 2 * lsum - float(logsumexp(2 * centered))
    return dict(rows=int(values.size), positive_weight_rows=int(finite.sum()),
                support_zero_rows=int((~finite).sum()),
                log_factor=total-math.log(denominator),
                largest_weight_share=math.exp(-lsum),
                kish_weight_concentration=math.exp(log_kish),
                finite_log_weight_range=[float(values[finite].min()), float(values[finite].max())],
                integration_error_validated=False, sampler_ess_validated=False)


def main():
    import numpy as np
    from scipy.integrate import quad
    from src.boundary_regularized_population import BoundaryRegularizedSpin, log_boundary_window
    from src.independent_spin_model import SpinParameters
    from src.declared_event_ratio import factorized_log_ratio

    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'results/preflight' / ('reduced_spin_feasibility_' + stamp)
    output.mkdir(exist_ok=False)
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='INCOMPLETE',
                  errors=[], cases=[], baseline_checks=[], catalog_inventory=[],
                  worker_budget_seconds=180, population_fit_run=False,
                  combined_likelihood_computed=False, native_evolution_calls=0,
                  event_exclusions=0, hyperprior_adopted=False,
                  posterior_recycling_accuracy_validated=False,
                  common_spin_frequency_verified=False, production_ready=False)
    print('Results: ' + str(output), flush=True)
    bindings = {}
    try:
        event_name = 'results/preflight/event_coverage_2026-09-17.json'
        selection_name = 'results/preflight/o4b_selection_2026-09-17.json'
        contract_name = 'results/preflight/released_density_contract_2026-09-16.json'
        names = [event_name, selection_name, contract_name,
                 'results/preflight/event_coverage_review_2026-09-17.json',
                 'results/preflight/o4b_selection_review_2026-09-17.json',
                 'protocol/reduced_spin_population_scope_2026-10-11.md',
                 'protocol/reduced_spin_feasibility_plan_2026-10-11.md',
                 'scripts/check_reduced_spin_feasibility.py',
                 'tests/test_reduced_spin_feasibility.py']
        names += [str(Path(module.__file__).resolve().relative_to(ROOT)).replace('\\', '/')
                  for module in tuple(sys.modules.values())
                  if getattr(module, '__file__', None)
                  and Path(module.__file__).resolve().is_relative_to(ROOT)
                  and Path(module.__file__).suffix == '.py']
        bindings = {name: sha(ROOT/name) for name in set(names)}
        event = json.loads((ROOT/event_name).read_text())
        selection = json.loads((ROOT/selection_name).read_text())
        for name, saved in [(event_name, event), (selection_name, selection)]:
            if saved['errors'] or saved.get('changed_inputs'):
                raise ValueError('Failed predecessor: ' + name)
            array = saved['arrays']
            if sha(ROOT/array['path']) != array['sha256']:
                raise ValueError('Saved array binding failed: ' + name)
            bindings[array['path']] = array['sha256']
        for name in names[3:5]:
            review = json.loads((ROOT/name).read_text())
            if not review['status'].startswith('COMPLETED') or review['errors'] or review.get('changed_inputs'):
                raise ValueError('Predecessor review failed: ' + name)
        fixed = event['fixed_models']
        if fixed != selection['fixed_models']:
            raise ValueError('Event and selection fixed models differ')
        report['fixed_models'] = fixed
        contract = json.loads((ROOT/contract_name).read_text())
        if contract['errors'] or contract.get('changed_inputs'):
            raise ValueError('Catalog contract failed')
        for entry in contract['posterior_contract']['events']:
            for label, metadata in entry['labels'].items():
                if label.endswith(('IMRPhenomXPHM-SpinTaylor', 'IMRPhenomXPNR')):
                    report['catalog_inventory'].append(dict(event=entry['event'], label=label,
                        rows=metadata['posterior_rows'],
                        complete_analytic_reference=metadata['complete_analytic_reference'],
                        reference_binding_issues=metadata['reference_binding_issues'],
                        historically_proven=metadata['released_row_prior_historically_proven']))
        report['catalog_events'] = len({p['event'] for p in report['catalog_inventory']})
        if report['catalog_events'] != 104 or len(report['catalog_inventory']) != 208:
            raise ValueError('Catalog membership differs from declared frame')
        report['catalog_total_posterior_rows'] = sum(p['rows'] for p in report['catalog_inventory'])
        report['catalog_flagged_products'] = [p for p in report['catalog_inventory']
                                             if not p['complete_analytic_reference'] or p['reference_binding_issues']]
        e = np.load(ROOT/event['arrays']['path'], allow_pickle=False)
        s = np.load(ROOT/selection['arrays']['path'], allow_pickle=False)
        mask = s['evaluated'].astype(bool)
        if s['chi_eff'].size != selection['processed_recovered_rows']:
            raise ValueError('Injection row count changed')
        common_selection = (s['log_mass_redshift'][mask]-np.log(s['mass_1_source'][mask])
            +s['log_r0'][mask]-s['log_p0'][mask]-s['log_draw'][mask]+s['log_mixture'][mask])

        def selection_weights(model):
            result = np.full(mask.shape, -np.inf)
            result[mask] = common_selection + model.logpdf(s['mass_1_source'][mask], s['chi_eff'][mask])
            return result

        def event_weights(product, model):
            prefix = product['array_prefix']
            return factorized_log_ratio(e[prefix+'_log_mass_redshift'],
                model.logpdf(e[prefix+'_mass_1_source'], e[prefix+'_chi_eff']),
                e[prefix+'_log_pe_continuous'], e[prefix+'_log_spin_reference'])

        def compare(actual, expected):
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
            keep = np.isfinite(actual)
            return float(np.max(np.abs(actual[keep]-expected[keep])))

        for profile, pars in fixed['spin_profiles'].items():
            model = BoundaryRegularizedSpin(SpinParameters(**pars), a_max=fixed['a_max'], tail_width=fixed['tail_width'])
            saved_case = next(c for c in selection['cases'] if c['profile'] == profile and c['displacement'] == 0)
            residual = compare(selection_weights(model), s[saved_case['array_key']])
            event_residuals = []
            for product in event['products']:
                case = next(c for c in product['cases'] if c['spin_profile'] == profile
                            and c['displacement'] == 0 and c['distance_mode'] == 'continuous')
                event_residuals.append(compare(event_weights(product, model), e[case['array_key']]))
            report['baseline_checks'].append(dict(profile=profile,
                selection_max_abs_log_residual=residual, event_max_abs_log_residuals=event_residuals))
        print('Both archived profiles reconstructed for all six pilot products and recovered injections.', flush=True)
        settings = [(mu, math.log10(sigma)) for mu in (-.2, 0., .2) for sigma in (.05, .1, .2)]
        settings += [(.1, -.7), (-1., -2.), (1., -2.), (0., 1.)]
        base = SpinParameters(**fixed['spin_profiles']['mixed'])
        report['pilot_products'] = [dict(event=p['event'], label=p['label'], rows=p['rows']) for p in event['products']]
        for index, (mu, logsigma) in enumerate(settings):
            if time.monotonic()-started > 180:
                raise TimeoutError('Fixed feasibility worker budget exceeded')
            begin = time.monotonic()
            case = dict(index=index, mu=mu, log10_sigma=logsigma, status='INCOMPLETE')
            report['cases'].append(case)
            try:
                model = BoundaryRegularizedSpin(replace(base, mu=mu, log10_sigma=logsigma, xi_low=0, xi_high=0),
                                                a_max=fixed['a_max'], tail_width=fixed['tail_width'])
                sigma = 10**logsigma
                points = sorted({x for x in [-1+model.width, 1-model.width,
                                *(mu+k*sigma for k in (-8, -3, -1, 0, 1, 3, 8))] if -1 < x < 1})
                def kernel(x):
                    return math.exp(-.5*((x-mu)/sigma)**2 + float(log_boundary_window(x, model.width)))
                normalizer, qerror = quad(kernel, -1, 1, points=points, epsabs=0, epsrel=1e-10)
                if not math.isfinite(normalizer) or normalizer <= 0 or qerror/normalizer > 1e-8:
                    raise ArithmeticError('Direct kernel normalization failed')
                nodes = np.array([-.999, -.8, -.2, 0., .2, .8, .999])
                direct = -.5*((nodes-mu)/sigma)**2 + log_boundary_window(nodes, model.width) - math.log(normalizer*model.a_max)
                residual = compare(model.logpdf(20., model.a_max*nodes), direct)
                mean, _ = quad(lambda x: x*kernel(x)/normalizer, -1, 1, points=points, epsabs=1e-10, epsrel=1e-10)
                variance, _ = quad(lambda x: (x-mean)**2*kernel(x)/normalizer, -1, 1, points=points, epsabs=1e-12, epsrel=1e-10)
                negative, _ = quad(kernel, -1, 0, points=[x for x in points if x < 0], epsabs=1e-12, epsrel=1e-10)
                case['density_check'] = dict(max_abs_log_residual=residual, quadrature_relative_error=qerror/normalizer,
                    actual_mean=model.a_max*mean, actual_sd=model.a_max*math.sqrt(variance),
                    negative_probability_float=negative/normalizer if negative > 0 else None,
                    negative_probability_unresolved_in_float=negative == 0,
                    mathematical_negative_probability_positive=True)
                case['events'] = [weight_summary(event_weights(p, model), p['rows']) for p in event['products']]
                case['selection'] = weight_summary(selection_weights(model), selection['total_generated'])
                case['status'] = 'COMPUTED_DIAGNOSTIC_ONLY'
            except Exception as error:
                case['status'] = 'FAILED_NO_FACTOR_SUBSTITUTED'
                case['error'] = type(error).__name__ + ': ' + str(error)
                report['errors'].append('setting ' + str(index) + ': ' + case['error'])
            case['elapsed_seconds'] = time.monotonic()-begin
            print(f'{index+1}/{len(settings)}: {case["status"]}', flush=True)
        report['status'] = 'COMPLETED_REDUCED_FEASIBILITY_DIAGNOSTIC_REVIEW_REQUIRED' if not report['errors'] else 'STOPPED_NEEDS_REVIEW'
    except Exception as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
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
                             elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return 0 if not report['errors'] and not report['changed_inputs'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
