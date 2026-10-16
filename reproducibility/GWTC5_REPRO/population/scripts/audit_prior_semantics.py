"""Bind mathematical prior references to checked metadata, without fitting rows.

Fresh analytic strings must match the pinned full-shape audit. Read-only
software/history/spin metadata and column names narrow provenance questions.
No posterior values, prior draws or stored log_prior values are read.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_population_prior_metadata import safe_node
from scripts.audit_population_prior_bindings import analytic_inventory, candidate_binding


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def small_metadata(group, path, h5py):
    node = safe_node(group, path, h5py)
    if node is None:
        return None
    if not isinstance(node, h5py.Dataset) or node.size > 16 or node.external or node.is_virtual:
        raise ValueError('Unsupported metadata representation: '+path)
    def plain(value):
        if hasattr(value, 'tolist'): return plain(value.tolist())
        if isinstance(value, list): return [plain(v) for v in value]
        if isinstance(value, bytes): return value.decode('utf-8')
        if isinstance(value, (str,int,float,bool,type(None))): return value
        raise ValueError('Unsupported scalar metadata')
    result = plain(node[()])
    if len(json.dumps(result).encode()) > 65536:
        raise ValueError('Oversize metadata')
    return result


def mass_support_from_metadata(analytic):
    from src.pe_prior_reference import DetectorMassSupport
    def bounds(name, constructor):
        parsed = analytic.get(name, {}).get('parsed', {})
        if parsed.get('constructor') != constructor:
            raise ValueError('Missing or unsupported mass prior: '+name)
        p = parsed.get('shape_parameters', {})
        return tuple(p[k] for k in ('minimum','maximum'))
    supported = {'mass_1','mass_2','total_mass'}
    constraints = {k for k,v in analytic.items() if v.get('parsed',{}).get('constructor','').split('.')[-1] == 'Constraint'}
    if constraints-supported:
        raise ValueError('Unimplemented constraint: '+','.join(sorted(constraints-supported)))
    return DetectorMassSupport(
        bounds('chirp_mass','bilby.gw.prior.UniformInComponentsChirpMass'),
        bounds('mass_ratio','bilby.gw.prior.UniformInComponentsMassRatio'),
        bounds('mass_1','Constraint'), bounds('mass_2','Constraint'),
        bounds('total_mass','Constraint') if 'total_mass' in analytic else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.dependency_dir: sys.path.insert(0,str(args.dependency_dir.resolve()))
    import h5py
    import numpy as np
    from src.pe_prior_reference import FlatDetectorMassPrior, SourceFrameDistancePrior
    source = root/'results/preflight/population_prior_bindings_fullshape_2026-09-12.json'
    previous = json.loads(source.read_text())
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root/'results/preflight'):
        raise ValueError('Use a new report inside preflight')
    records, failures, distance_cache, mass_cache = [], [], {}, {}
    versions, spin_methods, column_counts = Counter(), Counter(), Counter()
    paths = ['version', 'meta_data/other/loaded_modules/bilby', 'meta_data/other/loaded_modules/astropy',
             'meta_data/other/reweighting_configuration', 'meta_data/meta_data/backward_spin_evolution',
             'meta_data/meta_data/forward_spin_evolution', 'meta_data/meta_data/f_ref',
             'meta_data/meta_data/cosmology']
    for event in previous['events']:
        record = {'event':event['event'],'path':event['path'],'labels':{}}
        try:
            with h5py.File(event['path'],'r') as handle:
                record['pesummary_version'] = small_metadata(handle,'version/pesummary',h5py)
                record['history_command'] = small_metadata(handle,'history/command_line',h5py)
                versions[json.dumps(record['pesummary_version'])] += 1
                for label, old in event['labels'].items():
                    group = safe_node(handle,label,h5py)
                    analytic = analytic_inventory(group,h5py)
                    if {k:v['raw'] for k,v in analytic.items()} != {k:v['raw'] for k,v in old['analytic_priors'].items()}:
                        raise ValueError('Prior strings changed since full-shape audit: '+label)
                    data = {'metadata':{p:small_metadata(group,p,h5py) for p in paths},
                            'analytic_metadata_sha256':fingerprint(analytic), 'analytic_strings_unchanged':True}
                    node = safe_node(group,'posterior_samples',h5py)
                    if not isinstance(node,h5py.Dataset) or node.external or node.is_virtual:
                        raise ValueError('Unsupported posterior schema')
                    fields = node.dtype.names or ()
                    data['spin_columns'] = [k for k in fields if k.startswith(('chi_eff','tilt_1_infinity','tilt_2_infinity'))]
                    for k in data['spin_columns']: column_counts[k]+=1
                    spin_methods[json.dumps(data['metadata']['meta_data/meta_data/backward_spin_evolution'])]+=1
                    issues = [x for x in candidate_binding(analytic)['recognition_issues'] if x != 'additional_constraint:total_mass']
                    data['reference_binding_issues'] = issues
                    data['reference_normalization_computed'] = False
                    data['released_row_prior_validated'] = False
                    if not issues:
                        support = mass_support_from_metadata(analytic)
                        if support not in mass_cache: mass_cache[support] = FlatDetectorMassPrior(support)
                        mass = mass_cache[support]
                        bounds = tuple(analytic['luminosity_distance']['parsed']['shape_parameters'][k] for k in ('minimum','maximum'))
                        if bounds not in distance_cache:
                            grid = SourceFrameDistancePrior(bounds,mode='bilby_2.6.0_grid_reference')
                            smooth = SourceFrameDistancePrior(bounds,mode='continuous')
                            # Fixed log-spaced coordinate grid, not observed posterior locations.
                            z = np.geomspace(grid.zmin*(1+1e-8),grid.zmax*(1-1e-8),2048)
                            continuous = np.exp(smooth.redshift_logpdf(z))
                            tabulated = np.exp(grid.redshift_logpdf(z))
                            relative = np.abs(tabulated/continuous-1)
                            index = int(np.argmax(relative))
                            distance_cache[bounds] = {'bounds_Mpc':bounds, 'z_bounds':[grid.zmin,grid.zmax],
                                'continuous_volume_time_normalizer_Mpc3_per_sr':grid.continuous_normalizer,
                                'grid_raw_normalizer_Mpc3_per_sr':grid.grid_normalizer,
                                'grid_node_pdf_integral':float(np.trapezoid(grid.distance_density_nodes,grid.distance_nodes)),
                                'comparison_grid_points':2048, 'max_grid_sampled_relative_difference':float(relative[index]),
                                'distance_at_max_relative_difference_Mpc':float(grid.cosmology.luminosity_distance(z[index]).value),
                                'grid_approximate_total_variation':float(.5*np.trapezoid(np.abs(tabulated-continuous),z)),
                                'difference_is_not_posterior_weight_error_bound':True}
                        data['reference_normalization_computed'] = True
                        data['mass_support'] = {k:getattr(support,k) for k in ('chirp_mass','mass_ratio','mass_1','mass_2','total_mass')}
                        data['detector_component_mass_area_solar_mass_squared'] = mass.area
                        data['distance_reference'] = distance_cache[bounds]
                        data['reference_coordinate_status'] = 'PE sampling-frequency reference only; no common-frequency mapping validated'
                    record['labels'][label] = data
        except (OSError,ValueError,TypeError,KeyError) as error:
            failures.append({'event':event['event'],'error':str(error)})
        records.append(record)
    labels = [d for e in records for d in e['labels'].values()]
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'prior definitions, numerical normalizers, software/history metadata and spin-column schema only',
              'source_prior_audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'summary':{'events':len(records),'products':len(labels),
                         'reference_normalizations_computed':sum(d['reference_normalization_computed'] for d in labels),
                         'unique_mass_supports':len(mass_cache),'unique_distance_bounds':len(distance_cache),
                         'pesummary_version_counts_by_file':dict(versions),
                         'backward_evolution_counts_by_product':dict(spin_methods),'spin_column_counts':dict(column_counts)},
              'posterior_values_read':False,'prior_draws_read':False,'full_PE_hashes_recomputed':False,
              'population_outcomes_computed':False,'released_row_prior_validated':False,
              'source_code_imported_or_executed':False,'errors':failures,'events':records}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'summary':report['summary'],'errors':failures},indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
