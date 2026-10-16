"""Checkpoint unchanged declared-reference factors, without fitting a population."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
KEYS = ('mass_1_source', 'mass_ratio', 'redshift', 'chi_eff', 'log_spin_reference',
        'spin_reference_relative_quadrature_error', 'log_detector_component_distance_jacobian',
        'log_mass_redshift', 'log_pe_continuous', 'log_pe_bilby_2.6.0_grid_reference', 'row_index')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_path(value):
    """Translate recorded Windows paths only when running inside Linux."""
    if sys.platform != 'win32' and len(value) > 2 and value[1] == ':':
        return Path('/mnt')/value[0].lower()/value[3:].replace('\\', '/')
    return Path(value)


def numpy_compatibility():
    """Expose the old trapezoidal rule under its new name, only in this process."""
    import numpy as np
    if np.__version__ != '1.26.4':
        raise ValueError('Use the pinned NumPy 1.26.4 runtime')
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    if np.trapezoid is not np.trapz:
        raise ValueError('Unexpected trapezoidal integration implementation')
    return dict(numpy_version=np.__version__, process_local_alias='numpy.trapezoid = numpy.trapz',
                implementation_path=np.trapz.__module__, installed_files_changed=False)


def validate_arrays(arrays, count):
    import numpy as np
    if not isinstance(count, int) or count <= 0:
        raise ValueError('Positive original posterior row count required')
    if set(arrays) != set(KEYS):
        raise ValueError('Unexpected prepared-factor columns')
    for key, values in arrays.items():
        if values.shape != (count,):
            raise ValueError('Prepared row shape differs: ' + key)
        if key == 'log_mass_redshift':
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError('Invalid target factor')
        elif not np.isfinite(values).all():
            raise ValueError('Invalid reference or coordinate: ' + key)
    np.testing.assert_array_equal(arrays['row_index'], np.arange(count))
    if np.any(np.abs(arrays['chi_eff']) >= .99):
        raise ValueError('Native spin outside declared interior support')
    if np.any(arrays['spin_reference_relative_quadrature_error'] < 0) or np.any(arrays['spin_reference_relative_quadrature_error'] > 5e-6):
        raise ValueError('Induced-spin quadrature screen failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-new-products', type=int, default=2)
    parser.add_argument('--resume-manifest', type=Path)
    parser.add_argument('--worker-budget-seconds', type=int, default=900)
    args = parser.parse_args()
    if not 0 <= args.max_new_products <= 208 or not 0 < args.worker_budget_seconds <= 10800:
        raise ValueError('Use bounded product and worker budgets')
    import h5py
    import numpy as np
    compatibility = numpy_compatibility()
    from scripts.audit_population_prior_metadata import safe_node
    from scripts.audit_population_prior_bindings import analytic_inventory
    from scripts.audit_prior_semantics import fingerprint, mass_support_from_metadata
    from scripts.audit_released_posterior_rows import read_coordinates
    from src.released_row_checks import selected_value_digest
    from src.analysis_coordinates import AnalysisCosmology, MassDistancePriorInAnalysisCoordinates
    from src.pe_prior_reference import FlatDetectorMassPrior, SourceFrameDistancePrior
    from src.chirp_ratio_event_adapter import bind_chirp_ratio_event
    from src.independent_mass_redshift import MassParameters, RedshiftModel
    from src.cached_mass_normalizer import CachedTaperMassModel
    from src.declared_event_ratio import prepare_event_reference

    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/preflight'/('reduced_catalog_reference_'+stamp)
    output.mkdir(exist_ok=False)
    print('Results: '+str(output), flush=True)
    report = dict(status='INCOMPLETE', started_utc=datetime.now(timezone.utc).isoformat(),
        errors=[], products=[], worker_budget_seconds=args.worker_budget_seconds,
        numpy_compatibility=compatibility,
        max_new_products=args.max_new_products, population_fit_run=False,
        historical_released_row_density_certified=False, native_evolution_calls=0,
        events_excluded=[], production_ready=False, interrupted_product=None)
    bindings = {}
    try:
        def load(name, *, legacy_row_ledger=False):
            bindings[name] = sha(ROOT/name)
            saved = json.loads((ROOT/name).read_text())
            complete = saved.get('status', '').startswith('COMPLETED')
            if legacy_row_ledger:
                complete = ('status' not in saved and saved.get('inputs_modified') is False
                            and saved.get('posterior_values_read') is True and len(saved.get('events', [])) == 104)
            if saved.get('errors') or saved.get('changed_inputs') or not complete:
                raise ValueError('Incomplete predecessor: '+name)
            return saved
        contract = load('results/preflight/released_density_contract_2026-09-16.json')
        ledger = load('results/preflight/released_posterior_row_audit_2026-09-12.json', legacy_row_ledger=True)
        load('results/preflight/prior_and_stopping_review_2026-09-18.json')
        load('results/preflight/declared_event_ratio_review_2026-09-17.json')
        load('results/preflight/event_coverage_review_2026-09-17.json')
        load('results/preflight/catalog_component_coverage_2026-09-23.json')
        priors = load('results/preflight/prior_and_stopping_2026-09-18.json')
        if not priors['prior_reference_family_exception_resolved']:
            raise ValueError('Recorded-dictionary exception not resolved')
        sources = [load('results/preflight/'+name+'_2026-09-'+day+'.json')
                   for name, day in [('declared_event_ratio','17'), ('event_coverage','17'), ('chirp_ratio_event','18')]]
        fixed = sources[0]['fixed_models']
        if any(source['fixed_models'] != fixed for source in sources):
            raise ValueError('Reused products have different fixed models')
        report['fixed_models'] = fixed
        cache = {}
        for source in sources:
            array = source['arrays']
            if sha(ROOT/array['path']) != array['sha256']:
                raise ValueError('Prior factor arrays changed')
            bindings[array['path']] = array['sha256']
            data = np.load(ROOT/array['path'], allow_pickle=False)
            for product in source['products']:
                key = (product.get('event', source.get('event')), product['label'])
                cache[key] = (product, {name: data[product['array_prefix']+'_'+name] for name in KEYS}, array['path'])
        if args.resume_manifest:
            path = args.resume_manifest.resolve()
            if not path.is_relative_to(ROOT/'results/preflight'):
                raise ValueError('Resume only a local prepared-factor manifest')
            bindings[str(path.relative_to(ROOT))] = sha(path)
            previous = json.loads(path.read_text())
            if previous['errors'] or previous['changed_inputs'] or previous['fixed_models'] != fixed:
                raise ValueError('Resume manifest failed or models changed')
            for product in previous['products']:
                array = product['arrays']
                if sha(ROOT/array['path']) != array['sha256']:
                    raise ValueError('Prepared checkpoint changed')
                bindings[array['path']] = array['sha256']
                cache[(product['event'], product['label'])] = (product,
                    dict(np.load(ROOT/array['path'], allow_pickle=False)), array['path'])
        events = {e['event']: e for e in contract['posterior_contract']['events']}
        old = {e['event']: e for e in ledger['events']}
        queue = [(event, label) for event in sorted(events) for label in sorted(events[event]['labels'])
                 if label.endswith(('IMRPhenomXPHM-SpinTaylor', 'IMRPhenomXPNR'))]
        if len(queue) != 208 or len(events) != 104:
            raise ValueError('Declared catalog membership changed')
        # Reusable references first, then all other products in fixed order.
        queue.sort(key=lambda key: (key not in cache, key))
        report['expected_products'] = 208
        report['expected_events'] = 104
        report['expected_rows'] = sum(events[event]['labels'][label]['posterior_rows'] for event, label in queue)
        mass = CachedTaperMassModel(MassParameters(**fixed['mass']))
        redshift = RedshiftModel(fixed['kappa'], z_max=fixed['z_max'], H0=fixed['H0'], omega_m=fixed['omega_m'])
        cosmo = AnalysisCosmology('Planck15_LAL', inversion_zmax=16)
        mass_cache, distance_cache = {}, {}
        names = ['scripts/prepare_reduced_spin_catalog.py', 'protocol/reduced_catalog_preparation_plan_2026-10-11.md',
                 'tests/test_reduced_catalog_preparation.py']
        names += [str(Path(module.__file__).resolve().relative_to(ROOT)) for module in tuple(sys.modules.values())
                  if getattr(module, '__file__', None) and Path(module.__file__).suffix == '.py'
                  and Path(module.__file__).resolve().is_relative_to(ROOT)]
        bindings.update({name: sha(ROOT/name) for name in set(names)})
        newly_prepared = 0
        for event, label in queue:
            if (event, label) not in cache and newly_prepared >= args.max_new_products:
                break
            if time.monotonic()-started > args.worker_budget_seconds:
                raise TimeoutError('Fixed preparation budget exceeded; completed checkpoints retained')
            begin = time.monotonic()
            report['interrupted_product'] = dict(event=event, label=label)
            record = events[event]
            path = local_path(record['path'])
            before = path.stat()
            if (before.st_size, before.st_mtime_ns) != (old[event]['container_bytes'], old[event]['container_mtime_ns']):
                raise ValueError('Posterior container changed: '+event)
            with h5py.File(path, 'r') as handle:
                group = safe_node(handle, label, h5py)
                analytic = analytic_inventory(group, h5py)
                digest = fingerprint(analytic)
                if digest != record['labels'][label]['analytic_metadata_sha256']:
                    raise ValueError('Analytic reference metadata changed')
                rows = read_coordinates(group, h5py)
                selected_digest = selected_value_digest(rows)
                if selected_digest != old[event]['labels'][label]['selected_value_sha256']:
                    raise ValueError('Ordered posterior coordinates changed')
                q = rows['mass_2']/rows['mass_1']
                native = (rows['a_1']*np.cos(rows['tilt_1'])+q*rows['a_2']*np.cos(rows['tilt_2']))/(1+q)
                np.testing.assert_allclose(native, rows['chi_eff'], atol=1e-12, rtol=1e-10)
                if any(np.any((rows[name] < 0) | (rows[name] > .99)) for name in ('a_1','a_2')):
                    raise ValueError('Component spin support changed')
                if event == 'GW240925_005809':
                    adapters = {mode: bind_chirp_ratio_event(analytic, distance_mode=mode,
                        analysis_cosmology=cosmo, pe_cosmology=cosmo,
                        recorded_dictionary_has_no_extra_mass_constraints=True).analysis_prior
                        for mode in SourceFrameDistancePrior.MODES}
                    family = 'recorded_chirp_ratio_only_NO_extra_constraints'
                else:
                    support = mass_support_from_metadata(analytic)
                    if support not in mass_cache:
                        mass_cache[support] = FlatDetectorMassPrior(support)
                    bounds = tuple(analytic['luminosity_distance']['parsed']['shape_parameters'][name]
                                   for name in ('minimum','maximum'))
                    adapters = {}
                    for mode in SourceFrameDistancePrior.MODES:
                        if (bounds, mode) not in distance_cache:
                            distance_cache[bounds, mode] = SourceFrameDistancePrior(bounds, mode=mode)
                        adapters[mode] = MassDistancePriorInAnalysisCoordinates(mass_cache[support],
                            distance_cache[bounds, mode], analysis_cosmology=cosmo, pe_cosmology=cosmo)
                    family = 'recorded_component_constraints'
                reused = (event, label) in cache
                if reused:
                    prior_product, arrays, origin = cache[event, label]
                    if prior_product['selected_input_sha256'] != selected_digest or prior_product['analytic_metadata_sha256'] != digest:
                        raise ValueError('Reusable factor identities differ')
                else:
                    chunks = {name: [] for name in KEYS if name != 'row_index'}
                    for offset in range(0, len(rows), 128):
                        if time.monotonic()-started > args.worker_budget_seconds:
                            raise TimeoutError('Fixed preparation budget exceeded mid-product')
                        sub = rows[offset:offset+128]
                        prepared = prepare_event_reference(sub['mass_1'], sub['mass_2'], sub['luminosity_distance'],
                            sub['chi_eff'], prior=adapters['continuous'], a_max=fixed['a_max'])
                        for name in chunks:
                            if hasattr(prepared, name):
                                chunks[name].append(getattr(prepared, name))
                        chunks['log_mass_redshift'].append(mass.logpdf(prepared.mass_1_source, prepared.mass_ratio)
                                                         +redshift.logpdf(prepared.redshift))
                        chunks['log_pe_continuous'].append(prepared.log_pe_mass_distance)
                        chunks['log_pe_bilby_2.6.0_grid_reference'].append(
                            adapters['bilby_2.6.0_grid_reference'].detector_component_logpdf(
                                sub['mass_1'], sub['mass_2'], sub['luminosity_distance'])
                            +prepared.log_detector_component_distance_jacobian)
                    arrays = {name: np.concatenate(values) for name, values in chunks.items()}
                    arrays['row_index'] = np.arange(len(rows), dtype=np.int64)
                    origin = 'fresh_unchanged_reference_evaluation'
                    newly_prepared += 1
                validate_arrays(arrays, len(rows))
                np.testing.assert_array_equal(arrays['chi_eff'], rows['chi_eff'])
                indices = np.unique(np.linspace(0, len(rows)-1, 17, dtype=int))
                residuals = {}
                for mode, adapter in adapters.items():
                    reference = adapter.source_mqz_logpdf(arrays['mass_1_source'][indices],
                        arrays['mass_ratio'][indices], arrays['redshift'][indices])
                    np.testing.assert_allclose(reference, arrays['log_pe_'+mode][indices], atol=2e-8, rtol=0)
                    residuals[mode] = float(np.max(np.abs(reference-arrays['log_pe_'+mode][indices])))
                py = mass.logpdf(arrays['mass_1_source'][indices], arrays['mass_ratio'][indices])+redshift.logpdf(arrays['redshift'][indices])
                np.testing.assert_allclose(py, arrays['log_mass_redshift'][indices], atol=2e-8, rtol=0)
                if selected_value_digest(rows) != selected_digest:
                    raise ValueError('Source arrays changed during calculation')
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source container changed during calculation')
            array_path = output/f'product_{len(report["products"]):03d}.npz'
            with array_path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            item = dict(event=event, label=label, rows=len(rows), prior_family=family,
                selected_input_sha256=selected_digest, analytic_metadata_sha256=digest,
                reused=reused, origin=origin, elapsed_seconds=time.monotonic()-begin,
                fixed_density_check_indices=indices.tolist(), coordinate_log_residuals=residuals,
                max_spin_reference_relative_quadrature_error=float(arrays['spin_reference_relative_quadrature_error'].max()),
                positive_mass_redshift_rows=int(np.isfinite(arrays['log_mass_redshift']).sum()),
                arrays=dict(path=str(array_path.relative_to(ROOT)), sha256=sha(array_path)))
            report['products'].append(item)
            report['interrupted_product'] = None
            report['newly_prepared_products'] = newly_prepared
            report['completed_rows'] = sum(p['rows'] for p in report['products'])
            print(f'{len(report["products"])}/208 retained: {event} {label}, {len(rows)} rows; '+
                  ('reused' if reused else 'fresh')+f'; {item["elapsed_seconds"]:.2f} s', flush=True)
            (output/f'checkpoint_{len(report["products"]):03d}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        report['status'] = ('COMPLETED_CATALOG_REFERENCE_PREPARATION_REVIEW_REQUIRED' if len(report['products']) == 208
                            else 'COMPLETED_BOUNDED_REFERENCE_PREPARATION_REVIEW_REQUIRED')
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
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
            completed_products=len(report['products']), elapsed_seconds=report['elapsed_seconds'])), flush=True)
    return int(bool(report['errors'] or report['changed_inputs']))


if __name__ == '__main__':
    raise SystemExit(main())
