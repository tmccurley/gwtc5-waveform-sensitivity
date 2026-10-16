"""Audit all explicit mass/distance/spin rows without a population likelihood."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.audit_population_prior_metadata import safe_node
from scripts.audit_population_prior_bindings import analytic_inventory
from scripts.audit_prior_semantics import fingerprint,small_metadata,mass_support_from_metadata


def read_coordinates(group,h5py):
    from src.released_row_checks import BASE_FIELDS,OPTIONAL_FIELDS
    node=safe_node(group,'posterior_samples',h5py)
    if not isinstance(node,h5py.Dataset) or node.external or node.is_virtual:
        raise ValueError('Unsupported or externally stored posterior dataset')
    if node.ndim!=1 or not 0<node.size<=250000 or node.dtype.names is None:
        raise ValueError('Posterior schema outside declared one-product memory limit')
    if any(k not in node.dtype.names for k in BASE_FIELDS):
        raise ValueError('Missing coordinate field')
    fields=[*BASE_FIELDS,*(k for k in OPTIONAL_FIELDS if k in node.dtype.names)]
    if any(node.dtype[k].kind not in 'if' for k in fields):
        raise ValueError('Coordinate columns must be real numeric scalars')
    return node.fields(fields)[:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dependency-dir',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.dependency_dir: sys.path.insert(0,str(args.dependency_dir.resolve()))
    import h5py
    import astropy
    from src.pe_prior_reference import FlatDetectorMassPrior,SourceFrameDistancePrior
    from src.released_row_checks import BASE_FIELDS,audit_coordinate_rows,selected_value_digest
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(root/'results/preflight'):
        raise ValueError('Use a new report inside preflight')
    source=root/'results/preflight/prior_semantics_and_reference_2026-09-12.json'
    prior=json.loads(source.read_text())
    plan=root/'protocol/released_row_audit_plan_2026-09-12.md'
    progress=output.with_suffix('.progress.json')
    if progress.exists(): raise ValueError('Existing progress record; choose a new output name')
    records,errors=[],[]; masses={}; distances={}; row_count=0
    started=datetime.now(timezone.utc).isoformat()
    for number,event in enumerate(prior['events'],1):
        record={'event':event['event'],'path':event['path'],'labels':{}}
        path=Path(event['path'])
        try:
            before=path.stat()
            with h5py.File(path,'r') as handle:
                for label,old in event['labels'].items():
                    group=safe_node(handle,label,h5py)
                    analytic=analytic_inventory(group,h5py)
                    if fingerprint(analytic)!=old['analytic_metadata_sha256']:
                        raise ValueError('Prior metadata changed: '+label)
                    raw=small_metadata(group,'meta_data/meta_data/cosmology',h5py)
                    if raw!=old['metadata']['meta_data/meta_data/cosmology']:
                        raise ValueError('Conversion cosmology metadata changed: '+label)
                    name=raw[0] if isinstance(raw,list) and len(raw)==1 else raw
                    mass_ref,distance_refs=None,None
                    if old['reference_normalization_computed']:
                        support=mass_support_from_metadata(analytic)
                        if support not in masses: masses[support]=FlatDetectorMassPrior(support)
                        mass_ref=masses[support]
                        bounds=tuple(old['distance_reference']['bounds_Mpc'])
                        if bounds not in distances:
                            distances[bounds]=tuple(SourceFrameDistancePrior(bounds,mode=m) for m in SourceFrameDistancePrior.MODES)
                        distance_refs=distances[bounds]
                    rows=read_coordinates(group,h5py)
                    result=audit_coordinate_rows(rows,analytic,cosmology_name=name,
                                                  distance_references=distance_refs,mass_reference=mass_ref)
                    result['selected_fields']=list(rows.dtype.names)
                    result['selected_value_sha256']=selected_value_digest(rows)
                    result['selected_value_hash_encoding']='JSON sorted-key field-name/count header followed by each entire column as little-endian float64 bytes, row order preserved'
                    result['analytic_metadata_sha256']=old['analytic_metadata_sha256']
                    result['frequency_metadata']=old['metadata']['meta_data/meta_data/f_ref']
                    record['labels'][label]=result
                    row_count+=rows.size
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Container changed during read')
            record['container_stat_unchanged_during_read']=True
            record['container_bytes']=before.st_size
            record['container_mtime_ns']=before.st_mtime_ns
        except (OSError,ValueError,TypeError,KeyError) as error:
            errors.append({'event':event['event'],'error':str(error)})
        records.append(record)
        state={'started_utc':started,'updated_utc':datetime.now(timezone.utc).isoformat(),
               'state':'RUNNING','events_processed':number,'products_processed':sum(len(e['labels']) for e in records),
               'rows_processed':row_count,'errors':errors,'population_outcomes_computed':False}
        progress.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        if number%10==0: print(json.dumps(state),flush=True)
    products=[d for e in records for d in e['labels'].values()]
    report={'started_utc':started,'created_utc':datetime.now(timezone.utc).isoformat(),
            'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
            'source_metadata_audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'all selected O4b product rows; within-product identities, support and spin evolution only',
            'events':records,'errors':errors,'summary':{'events':len(records),'products':len(products),'rows':row_count,
              'products_with_native_nonfinite_values':sum(any(d['nonfinite_by_field'][k] for k in BASE_FIELDS) for d in products),
              'products_with_algebraic_screen_exceedances':sum(any(v['rows_above_screen'] for v in d['algebraic_identities'].values()) for d in products),
              'products_with_distance_screen_exceedances':sum(d['cosmology'].get('distance_identity',{}).get('rows_above_screen',0)>0 for d in products),
              'products_with_bound_screen_exceedances':sum(any(v['below_beyond_screen'] or v['above_beyond_screen'] for v in d['recorded_bounds'].values()) for d in products)},
            'posterior_values_read':True,'prior_draws_read':False,'population_outcomes_computed':False,
            'ST_XPNR_effect_computed':False,'released_prior_validated':False,
            'common_spin_coordinate_validated':False,'full_PE_container_hashes_recomputed':False,
            'inputs_modified':False,'astropy_version':astropy.__version__,'h5py_version':h5py.__version__}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    state.update(state='COMPLETE' if not errors else 'COMPLETE_WITH_ERRORS',output=str(output))
    progress.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':report['summary'],'errors':errors},indent=2))
    return 1 if errors else 0


if __name__=='__main__': raise SystemExit(main())
