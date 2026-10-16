"""Independent saved-array assembly and original-cache subset replay."""

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
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    import numpy as np
    from scipy.special import logsumexp
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid, kernel_log_factors, refinement_check
    from src.reduced_physical_moments import physical_refinement

    started = time.monotonic()
    bindings = {}
    source = args.summary.resolve()
    if not source.is_relative_to(ROOT/'results/preflight'):
        raise ValueError('Review only workspace evidence')
    output = source.parent/('readback_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    report = dict(status='INCOMPLETE', errors=[], levels=[], direct_replays=[],
        worker_budget_seconds=3600, independent_product_contrast_used=False,
        posterior_sampling_accuracy_validated=False, selection_design_variance_validated=False,
        scientific_result_validated=False, manuscripts_modified=False)

    def budget():
        if time.monotonic()-started > 3600:
            raise TimeoutError('Fixed readback budget exceeded; evidence preserved')

    def bind(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Consume only workspace evidence')
        name = str(path.relative_to(ROOT))
        if name in bindings:
            if expected and bindings[name] != expected:
                raise ValueError('Conflicting evidence hashes: '+name)
            return path
        budget()
        digest = sha(path)
        if expected and expected != digest:
            raise ValueError('Saved evidence changed: '+name)
        bindings[name] = digest
        return path

    def array(entry):
        return np.load(bind(ROOT/entry['path'], entry['sha256']), allow_pickle=False)

    def close(actual, expected, tolerance=1e-10):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)

    try:
        worker = json.loads(bind(source).read_text())
        if worker['status'] not in ('COMPLETED_INTEGRATED_ROW_SENSITIVITY_REVIEW_REQUIRED', 'STOPPED_ROW_REFINEMENT_UNRESOLVED') or worker['errors'] or worker['changed_inputs']:
            raise ValueError('Two complete diagnostic levels without input errors required')
        if worker['validate_only'] or len(worker['levels']) != 2:
            raise ValueError('Complete integrated diagnostic required, not an input check')
        for name, digest in worker['preserved_hashes'].items():
            bind(ROOT/name, digest)
        grid = json.loads((ROOT/worker['evidence']['grid']).read_text())
        physical = json.loads((ROOT/worker['evidence']['physical']).read_text())
        kernel = json.loads((ROOT/grid['kernel_preflight']).read_text())
        products = kernel['products']
        if worker['products'] != [{k: p[k] for k in ('event', 'label', 'arm', 'rows')} for p in products]:
            raise ValueError('Product identities or row counts changed')
        if len(products) != 208 or sum(p['rows'] for p in products) != 3457688 or any(sum(p['arm'] == arm for p in products) != 104 for arm in ('ST', 'XPNR')):
            raise ValueError('Full original coverage required')
        groups = ['full', 'prefix_half', 'prefix_three_quarters']+[f'contiguous_{i}' for i in range(4)]+[f'interleaved_{i}' for i in range(4)]
        routes = ['continuous', 'bilby_2.6.0_grid_reference']
        columns = [[g, r, a] for g in groups for r in routes for a in ['ST', 'XPNR']]
        if worker['likelihood_columns'] != columns or worker['group_names'] != groups:
            raise ValueError('All 44 declared columns required')
        counts = []
        for product in products:
            n = product['rows']
            quarters = [n//4+(i < n % 4) for i in range(4)]
            counts.append([n, n//2, 3*n//4, *quarters, *quarters])
        counts = np.array(counts)
        np.testing.assert_array_equal(counts, worker['group_row_counts'])
        previous = previous_physical = None
        for level in worker['levels']:
            budget()
            order = level['order']
            if order not in (128, 256) or level['completed_nodes'] != order**2:
                raise ValueError('Both complete full-prior levels required')
            nodes, prior = prior_grid(order)
            with array(level['arrays']) as a:
                np.testing.assert_array_equal(a['settings'], nodes)
                np.testing.assert_array_equal(a['prior_weights'], prior)
                likelihoods, saved_masses = a['catalog_log_likelihoods'].copy(), a['posterior_quadrature_masses'].copy()
            if likelihoods.shape != (len(nodes), 44) or saved_masses.shape != likelihoods.shape:
                raise ValueError('Complete 44-column arrays required')
            original = next(v for v in grid['levels'] if v['order'] == order)
            with array(original['arrays']) as a:
                np.testing.assert_array_equal(a['settings'], nodes)
                primary = a['catalog_log_likelihoods'].copy()
            plevel = next(v for v in physical['levels'] if v['order'] == order)
            with array(plevel['arrays']) as a:
                np.testing.assert_array_equal(a['settings'], nodes)
                moments = a['physical_values'].copy()
            selection = np.full(len(nodes), np.nan)
            for entry in grid['checkpoints']:
                if entry['order'] == order:
                    first, end = entry['first_node'], entry['first_node']+entry['nodes']
                    with array(entry) as a:
                        selection[first:end] = a['selection_kernel_log_factors']
            coverage = np.zeros(len(nodes), bool)
            max_quarter = max_assembly = max_full = 0.
            replay_indices = [0, len(nodes)//2, len(nodes)-1]
            for entry in worker['checkpoints']:
                if entry['order'] != order:
                    continue
                budget()
                first, end = entry['first_node'], entry['first_node']+entry['nodes']
                if first < 0 or end > len(nodes) or coverage[first:end].any():
                    raise ValueError('Overlapping or out-of-range diagnostic tiles')
                with array(entry) as a:
                    np.testing.assert_array_equal(a['settings'], nodes[first:end])
                    np.testing.assert_array_equal(a['prior_weights'], prior[first:end])
                    np.testing.assert_array_equal(a['group_row_counts'], counts)
                    np.testing.assert_array_equal(a['selection_kernel_log_factors'], selection[first:end])
                    values = a['event_group_kernel_log_factors']
                    if values.shape != (208, end-first, 11, 2) or not np.isfinite(values).all():
                        raise ValueError('Complete finite product/group tensor required')
                    recombined = np.empty((end-first, 44))
                    for col, (_, _, arm) in enumerate(columns):
                        gi, ri = col//4, (col % 4)//2
                        total = np.zeros(end-first)
                        for j, product in enumerate(products):
                            if product['arm'] == arm:
                                total += values[j, :, gi, ri]
                        recombined[:, col] = total-104*selection[first:end]
                    error = float(np.max(abs(recombined-likelihoods[first:end])))
                    max_assembly = max(max_assembly, error)
                    close(recombined, likelihoods[first:end], 5e-8)
                    close(recombined, a['catalog_log_likelihoods'], 5e-8)
                    full_error = float(np.max(abs(recombined[:, :4]-primary[first:end])))
                    max_full = max(max_full, full_error)
                    if full_error > 5e-6:
                        raise ArithmeticError('Full control fails primary likelihood screen')
                    for quarter in (slice(3, 7), slice(7, 11)):
                        rebuilt = logsumexp(values[:, :, quarter, :]+np.log(counts[:, quarter]/counts[:, 0, None])[:, None, :, None], axis=2)
                        quarter_error = float(np.max(abs(rebuilt-values[:, :, 0, :])))
                        max_quarter = max(max_quarter, quarter_error)
                        if quarter_error > 2e-8:
                            raise ArithmeticError('Quarter reconstruction fails')
                    for index in replay_indices:
                        if not first <= index < end:
                            continue
                        for j in (0, 207):
                            product = products[j]
                            n = product['rows']; row = np.arange(n)
                            edges = np.r_[0, np.cumsum(counts[j, 3:7])]
                            masks = [np.ones(n, bool), row < n//2, row < 3*n//4]
                            masks += [(row >= edges[k]) & (row < edges[k+1]) for k in range(4)]
                            masks += [row % 4 == k for k in range(4)]
                            with array(product['cache']) as cache:
                                np.testing.assert_array_equal(cache['row_index'], row)
                                replay = np.empty((11, 2))
                                for g, mask in enumerate(masks):
                                    for r, route in enumerate(routes):
                                        replay[g, r] = kernel_log_factors(cache['x'][mask], cache['log_base_'+route][mask], nodes[index:index+1], int(mask.sum()), batch_size=1, row_chunk=32768)[0]
                            replay_error = float(np.max(abs(replay-values[j, index-first])))
                            if replay_error > 2e-8:
                                raise ArithmeticError('Direct original-cache subset replay fails')
                            report['direct_replays'].append(dict(order=order, node=index, product=j, max_abs_log_residual=replay_error))
                coverage[first:end] = True
            if not coverage.all():
                raise ValueError('Missing saved diagnostic tiles')
            summaries, psummaries = [], []
            for col in range(44):
                logz = float(logsumexp(likelihoods[:, col]+np.log(prior)))
                masses = np.exp(likelihoods[:, col]+np.log(prior)-logz)
                close(masses, saved_masses[:, col], 1e-12)
                mean = np.sum(masses[:, None]*nodes, axis=0)
                sd = np.sqrt(np.sum(masses[:, None]*(nodes-mean)**2, axis=0))
                summary = dict(log_normalization=logz, mean=mean.tolist(), sd=sd.tolist())
                saved = level['summaries'][col]
                close(logz, saved['log_normalization']); close(mean, saved['mean']); close(sd, saved['sd'])
                summaries.append(summary)
                masses = masses/masses.sum()
                pm = np.sum(masses[:, None]*moments[:, 1:], axis=0)
                pv = np.sum(masses[:, None]*(moments[:, 1:]-pm)**2, axis=0)
                average_variance = float(np.sum(masses*moments[:, 2]**2))
                prediction = float(np.sqrt(average_variance+pv[0]))
                psummary = dict(posterior_expectations=pm.tolist(), posterior_sds=np.sqrt(pv).tolist(), predictive_mean=float(pm[0]), predictive_sd=prediction)
                saved = level['physical_summaries'][col]
                close(pm, saved['posterior_expectations']); close(np.sqrt(pv), saved['posterior_sds'])
                close(average_variance, saved['posterior_average_intrinsic_variance']); close(prediction, saved['predictive_sd'])
                psummaries.append(psummary)
            entry = dict(order=order, nodes=len(nodes), summaries=summaries, physical_summaries=psummaries,
                         max_tile_assembly_abs_log_residual=max_assembly, max_full_control_abs_log_residual=max_full,
                         max_quarter_abs_log_residual=max_quarter)
            if previous:
                entry['refinement'] = [refinement_check(a, b) for a, b in zip(previous, summaries)]
                entry['physical_refinement'] = [physical_refinement(a, b) for a, b in zip(previous_physical, psummaries)]
                if [x['empirical_refinement_pass'] for x in entry['refinement']] != [x['empirical_refinement_pass'] for x in level['refinement']]:
                    raise ValueError('Coordinate refinement classification changed')
                if [x['empirical_physical_refinement_pass'] for x in entry['physical_refinement']] != [x['empirical_physical_refinement_pass'] for x in level['physical_refinement']]:
                    raise ValueError('Physical refinement classification changed')
            report['levels'].append(entry)
            previous, previous_physical = summaries, psummaries
            print(f'order {order}: all {len(nodes)} nodes, 44 posteriors and original-cache replays reviewed', flush=True)
        report['empirical_quadrature_refinement_pass'] = all(x['empirical_refinement_pass'] for x in report['levels'][-1]['refinement'])
        report['empirical_physical_refinement_pass'] = all(x['empirical_physical_refinement_pass'] for x in report['levels'][-1]['physical_refinement'])
        final = report['levels'][-1]
        differences = []
        for col, (group, route, arm) in enumerate(columns):
            control = col % 4
            p = final['physical_summaries'][col]; full = final['physical_summaries'][control]
            differences.append(dict(group=group, route=route, arm=arm,
                physical_expectation_difference_from_full=(np.array(p['posterior_expectations'])-full['posterior_expectations']).tolist(),
                descriptive_difference_in_full_posterior_sd=((np.array(p['posterior_expectations'])-full['posterior_expectations'])/full['posterior_sds']).tolist()))
        report['descriptive_subset_changes'] = differences
        report['status'] = 'COMPLETED_INTEGRATED_ROW_SENSITIVITY_READBACK'
    except Exception as error:
        report['status'] = 'STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['preserved_hashes'] = bindings
    report['changed_inputs'] = [name for name, digest in bindings.items() if not (ROOT/name).exists() or sha(ROOT/name) != digest]
    if report['changed_inputs']:
        report['status'] = 'STOPPED_INPUT_CHANGED'
    report['elapsed_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output))), flush=True)
    return int(report['status'] != 'COMPLETED_INTEGRATED_ROW_SENSITIVITY_READBACK')


if __name__ == '__main__':
    raise SystemExit(main())
