"""Independent covariance/variance assembly and original-row pair replay."""

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
    from scripts.check_reduced_spin_feasibility import sha
    from src.reduced_gaussian_grid import prior_grid
    from src.sample_selection_audit import fixed_stratum_reference
    from src.stopped_campaign_reference import stopped_stratum_reference
    from src.reduced_selection_reference_summary import reference_rms_refinement
    started = time.monotonic()
    source = args.summary.resolve()
    if not source.is_relative_to(ROOT/'results/preflight'):
        raise ValueError('Consume only workspace evidence')
    output = source.parent/('readback_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    bindings = {}
    report = dict(status='INCOMPLETE', errors=[], levels=[], raw_replays=[], worker_budget_seconds=1800,
                  actual_campaign_variance_validated=False, new_population_fit_run=False,
                  stochastic_selection_field_adopted=False, manuscripts_modified=False)

    def bind(name, expected=None):
        if time.monotonic()-started > 1800:
            raise TimeoutError('Fixed selection-grid review budget exceeded')
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Use only workspace inputs')
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError('Preserved input changed: '+name)
        bindings[name] = digest
        return path

    try:
        worker = json.loads(bind(str(source.relative_to(ROOT))).read_text())
        if worker['status'] not in ('COMPLETED_SELECTION_GRID_REFERENCE_REVIEW_REQUIRED', 'STOPPED_SELECTION_REFERENCE_REFINEMENT_UNRESOLVED') or worker['errors'] or worker['changed_inputs']:
            raise ValueError('Two completed diagnostic levels required')
        for name, digest in worker['preserved_hashes'].items():
            bind(name, digest)
        grid = json.loads((ROOT/worker['grid_summary']).read_text())
        point = json.loads((ROOT/'results/preflight/reduced_selection_reference_20261013T133416581483Z/summary.json').read_text())
        kernel = json.loads((ROOT/'results/preflight/reduced_kernel_preflight_20261012T132102172011Z/summary.json').read_text())
        cache, panel = kernel['selection_cache'], point['arrays']
        with np.load(bind(cache['path'], cache['sha256']), allow_pickle=False) as a:
            x, base, rows = a['x'].copy(), a['log_base'].copy(), a['row_index'].copy()
        with np.load(bind(panel['path'], panel['sha256']), allow_pickle=False) as a:
            np.testing.assert_array_equal(rows, a['row_index'])
            strata, n, r, anchor_setting = a['strata'].copy(), a['generated_by_month'].copy(), a['accepted_by_month_inferred'].copy(), a['settings'][0].copy()
        np.testing.assert_array_equal(anchor_setting, worker['anchor_settings'])
        if len(x) != 504666 or cache['total_generated'] != 1568035640:
            raise ValueError('Full original selection required')
        anchor_logs = base-(x-anchor_setting[0])**2/(2*10.**(2*anchor_setting[1]))
        previous = None
        for level in worker['levels']:
            order = level['order']
            if order not in (128,256) or level['completed_nodes'] != order**2:
                raise ValueError('Both complete full-prior levels required')
            nodes, prior = prior_grid(order)
            entry = level['arrays']
            with np.load(bind(entry['path'],entry['sha256']),allow_pickle=False) as a:
                np.testing.assert_array_equal(a['settings'],nodes)
                np.testing.assert_array_equal(a['prior_weights'],prior)
                saved_shapes, masses = a['catalog_shape_reference_sds'].copy(), a['posterior_quadrature_masses'].copy()
            original = next(v for v in grid['levels'] if v['order']==order)['arrays']
            with np.load(bind(original['path'],original['sha256']),allow_pickle=False) as a:
                np.testing.assert_array_equal(masses,a['posterior_quadrature_masses'])
                np.testing.assert_array_equal(nodes,a['settings'])
            replay_indices = np.linspace(0,len(nodes)-1,16,dtype=int)
            covered = np.zeros(len(nodes),bool)
            derived_shapes = np.full((len(nodes),2),np.nan)
            for tile in worker['checkpoints']:
                if tile['order'] != order:
                    continue
                first,end=tile['first_node'],tile['first_node']+tile['nodes']
                if first<0 or end>len(nodes) or covered[first:end].any():
                    raise ValueError('Overlapping/out-of-range tile')
                with np.load(bind(tile['path'],tile['sha256']),allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['settings'],nodes[first:end])
                    c=a['relative_covariances']
                    if c.shape!=(end-first,2,2,2) or not np.isfinite(c).all():
                        raise ValueError('Complete two-reference pair covariance matrices required')
                    np.testing.assert_allclose(c,np.swapaxes(c,-1,-2),atol=1e-14,rtol=0)
                    if np.linalg.eigvalsh(c).min() < -1e-14:
                        raise ArithmeticError('Non-PSD pair covariance')
                    variance=c[:,:,1,1]+c[:,:,0,0]-2*c[:,:,1,0]
                    if variance.min() < -1e-14:
                        raise ArithmeticError('Negative genuine shape variance')
                    values=104*np.sqrt(np.maximum(0.,variance))
                    np.testing.assert_allclose(values,a['catalog_shape_reference_sds'],atol=1e-10,rtol=0)
                    np.testing.assert_allclose(values,saved_shapes[first:end],atol=1e-10,rtol=0)
                    derived_shapes[first:end]=values
                    for index in replay_indices:
                        if not first<=index<end:
                            continue
                        mu,ls=nodes[index]
                        logs=np.column_stack((anchor_logs,base-(x-mu)**2/(2*10.**(2*ls))))
                        expected=np.array([fixed_stratum_reference(logs,strata,n)['relative_covariance'],stopped_stratum_reference(logs,strata,n,r)['relative_covariance']])
                        actual=c[index-first]
                        np.testing.assert_allclose(actual,expected,atol=1e-16,rtol=2e-10)
                        report['raw_replays'].append(dict(order=order,node=int(index),max_covariance_abs_residual=float(np.max(abs(actual-expected)))))
                covered[first:end]=True
            if not covered.all():
                raise ValueError('Missing tiles')
            summaries=np.sqrt(masses.T@(derived_shapes**2))
            np.testing.assert_allclose(summaries,level['reference_rms_by_posterior_column'],atol=1e-10,rtol=0)
            entry=dict(order=order,nodes=len(nodes),reference_rms_by_posterior_column=summaries.tolist())
            if previous is not None:
                entry['refinement']=[reference_rms_refinement(a,b) for a,b in zip(previous,summaries)]
                if [v['numerical_refinement_pass'] for v in entry['refinement']] != [v['numerical_refinement_pass'] for v in level['refinement']]:
                    raise ValueError('Refinement classifications differ')
            report['levels'].append(entry)
            previous=summaries
            print(f'order {order}: all {len(nodes)} reference nodes and posterior-weighted summaries reviewed',flush=True)
        if len(report['levels'])!=2:
            raise ValueError('Two complete levels required')
        report['empirical_reference_rms_refinement_pass']=all(v['numerical_refinement_pass'] for v in report['levels'][-1]['refinement'])
        report['status']='COMPLETED_SELECTION_GRID_REFERENCE_READBACK'
    except Exception as error:
        report['status']='STOPPED_NEEDS_REVIEW'
        report['errors'].append(type(error).__name__+': '+str(error))
    report['preserved_hashes']=bindings
    report['changed_inputs']=[name for name,digest in bindings.items() if not (ROOT/name).exists() or sha(ROOT/name)!=digest]
    if report['changed_inputs']:
        report['status']='STOPPED_INPUT_CHANGED'
    report['elapsed_seconds']=time.monotonic()-started
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],errors=report['errors'],report=str(output))),flush=True)
    return int(report['status']!='COMPLETED_SELECTION_GRID_REFERENCE_READBACK')


if __name__=='__main__':
    raise SystemExit(main())
