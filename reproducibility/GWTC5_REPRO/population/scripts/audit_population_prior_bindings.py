"""Read complete analytic-prior metadata and compare ST/XPNR bindings.

No posterior arrays or stored log_prior values are loaded. Constructor strings
are parsed as data, never executed. Matching metadata is not a validated
population denominator or proof that release-level reweighting was absent.
"""

import argparse
import ast
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_population_prior_metadata import parse_prior, safe_node, small_value


def parse_shape_prior(text):
    """Include Gaussian mu/sigma, fixed peaks and all nondisplay scalar keywords."""
    parsed = parse_prior(text)
    shape = {}
    for keyword in ast.parse(text, mode='eval').body.keywords:
        if keyword.arg in ('name', 'latex_label', 'unit'):
            continue
        value = ast.literal_eval(keyword.value)
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError('Nonscalar shape keyword requires a dedicated prior evaluator')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite prior-shape metadata')
        shape[keyword.arg] = value
    parsed['shape_parameters'] = shape
    return parsed


def analytic_inventory(group, h5py):
    analytic = safe_node(group, 'priors/analytic', h5py)
    if analytic is None:
        return {}
    if not isinstance(analytic, h5py.Group):
        raise ValueError('Analytic priors are not a metadata group')
    result = {}
    for name in sorted(analytic.keys()):
        node = safe_node(analytic, name, h5py)
        if isinstance(node, h5py.Dataset) and (node.external or node.is_virtual):
            raise ValueError('External or virtual prior metadata refused')
        raw = small_value(node, h5py)
        entry = {'raw': raw}
        if isinstance(raw, str):
            try:
                entry['parsed'] = parse_shape_prior(raw)
            except (ValueError, SyntaxError) as error:
                entry['parse_error'] = str(error)
        result[name] = entry
    return result


def candidate_binding(analytic):
    """Recognize a proposed analytic family without certifying released rows."""
    def spec(name):
        return analytic.get(name, {}).get('parsed', {})
    def bounds(name):
        values = spec(name).get('literal_metadata', {})
        return [values.get('minimum'), values.get('maximum')]
    expected = {'chirp_mass': 'bilby.gw.prior.UniformInComponentsChirpMass',
                'mass_ratio': 'bilby.gw.prior.UniformInComponentsMassRatio',
                'mass_1': 'Constraint', 'mass_2': 'Constraint',
                'luminosity_distance': 'bilby.gw.prior.UniformSourceFrame',
                'a_1': 'Uniform', 'a_2': 'Uniform', 'tilt_1': 'Sine', 'tilt_2': 'Sine',
                'phi_12': 'Uniform', 'phi_jl': 'Uniform'}
    issues = []
    for name, constructor in expected.items():
        if spec(name).get('constructor') != constructor:
            issues.append('missing_or_unexpected_prior:'+name)
        low, high = bounds(name)
        if not (isinstance(low, (int, float)) and not isinstance(low, bool)
                and isinstance(high, (int, float)) and not isinstance(high, bool)
                and math.isfinite(low) and math.isfinite(high) and low < high):
            issues.append('missing_or_invalid_bounds:'+name)
    for name in ['a_1', 'a_2']:
        if bounds(name) != [0, 0.99]:
            issues.append('different_component_spin_support:'+name)
    for name in ['tilt_1', 'tilt_2']:
        if bounds(name) != [0, math.pi]:
            issues.append('unverified_isotropic_tilt:'+name)
    for name in ['phi_12', 'phi_jl']:
        if bounds(name) != [0, 2*math.pi]:
            issues.append('unverified_spin_azimuth:'+name)
    for name in ['chirp_mass', 'mass_1', 'mass_2', 'mass_ratio']:
        low, high = bounds(name)
        if isinstance(low, (int, float)) and low <= 0:
            issues.append('nonpositive_mass_or_ratio_bound:'+name)
    ratio_upper = bounds('mass_ratio')[1]
    if isinstance(ratio_upper, (int, float)) and ratio_upper > 1:
        issues.append('unordered_mass_ratio_support')
    constraints = [name for name in analytic if spec(name).get('constructor', '').split('.')[-1] == 'Constraint']
    for name in constraints:
        if name not in ['mass_1', 'mass_2']:
            issues.append('additional_constraint:'+name)
    if any('parsed' not in item for item in analytic.values()):
        issues.append('unparsed_analytic_metadata')
    cosmology = spec('luminosity_distance').get('literal_metadata', {}).get('cosmology')
    if not isinstance(cosmology, str) or cosmology.lower() != 'planck15_lal':
        issues.append('unresolved_distance_prior_cosmology')
    return {'family': 'flat_detector_component_masses__UniformSourceFrame__isotropic_uniform_magnitudes',
            'support': {name: bounds(name) for name in expected}, 'distance_prior_cosmology': cosmology,
            'all_analytic_prior_keys': list(analytic), 'constraint_keys': constraints,
            'recognition_issues': issues, 'selected_family_metadata_recognized': not issues,
            'released_row_prior_validated': False}


def comparison(left, right):
    """Compare parsed definitions, ignoring display labels but retaining bounds."""
    names = sorted(set(left) | set(right))
    differences = []
    for name in names:
        def representation(item):
            parsed = item.get('parsed')
            if parsed is None:
                return {'unparsed_raw': item.get('raw')}
            return {'constructor': parsed['constructor'], 'shape_parameters': parsed.get('shape_parameters', parsed['literal_metadata'])}
        a, b = representation(left.get(name, {})), representation(right.get(name, {}))
        if a != b:
            differences.append({'prior': name, 'ST': a, 'XPNR': b})
    return differences


def above_roundoff(difference, *, relative_tolerance=1e-12, absolute_tolerance=1e-14):
    """Screen scalar differences, not a scientific equivalence/acceptance rule."""
    a, b = difference['ST'], difference['XPNR']
    if a.get('constructor') != b.get('constructor') or 'shape_parameters' not in a or 'shape_parameters' not in b:
        return True
    left, right = a['shape_parameters'], b['shape_parameters']
    if set(left) != set(right):
        return True
    for key, value in left.items():
        other = right[key]
        if isinstance(value, (float, int)) and not isinstance(value, bool) and isinstance(other, (float, int)) and not isinstance(other, bool):
            if not math.isclose(value, other, rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
                return True
        elif value != other:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    import h5py
    source = root/'results/preflight/population_prior_metadata_2026-09-11.json'
    previous = json.loads(source.read_text())
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root/'results/preflight'):
        raise ValueError('Use a new JSON within results/preflight')
    events, errors = [], []
    frequencies, issues, keys = Counter(), Counter(), Counter()
    for item in previous['events']:
        event = {'event': item['event'], 'path': item['path'], 'labels': {}}
        try:
            with h5py.File(item['path'], 'r') as handle:
                for label, old in item['labels'].items():
                    group = safe_node(handle, label, h5py)
                    if not isinstance(group, h5py.Group):
                        raise ValueError('Previously inventoried label missing: '+label)
                    analytic = analytic_inventory(group, h5py)
                    binding = candidate_binding(analytic)
                    reference = old['conversion_metadata'].get('f_ref')
                    config_reference = old['configuration'].get('reference-frequency')
                    try:
                        ref_numeric = float(reference)
                    except (TypeError, ValueError):
                        ref_numeric = None
                    data = {'analytic_priors': analytic, 'candidate_binding': binding,
                            'configuration': old['configuration'], 'conversion_metadata': old['conversion_metadata'],
                            'reference_frequency_hz': ref_numeric,
                            'configuration_reference_frequency': config_reference,
                            'differs_from_documented_injection_16Hz': None if ref_numeric is None else ref_numeric != 16.,
                            'source_previous_selected_metadata_sha256': old['selected_metadata_sha256']}
                    event['labels'][label] = data
                    frequencies[str(ref_numeric)] += 1
                    issues.update(binding['recognition_issues'])
                    keys.update(analytic.keys())
            st = [label for label in event['labels'] if label.endswith(':IMRPhenomXPHM-SpinTaylor')]
            xp = [label for label in event['labels'] if label.endswith(':IMRPhenomXPNR')]
            if len(st) == len(xp) == 1:
                a, b = event['labels'][st[0]], event['labels'][xp[0]]
                event['pair'] = {'ST_label': st[0], 'XPNR_label': xp[0],
                                 'analytic_prior_differences': comparison(a['analytic_priors'], b['analytic_priors']),
                                 'raw_prior_keys_with_differences': [name for name in sorted(set(a['analytic_priors']) | set(b['analytic_priors']))
                                                                   if a['analytic_priors'].get(name, {}).get('raw') != b['analytic_priors'].get(name, {}).get('raw')],
                                 'reference_frequency_equal': a['reference_frequency_hz'] is not None and a['reference_frequency_hz'] == b['reference_frequency_hz'],
                                 'both_candidate_families_recognized': a['candidate_binding']['selected_family_metadata_recognized'] and b['candidate_binding']['selected_family_metadata_recognized'],
                                 'pair_ready_for_population_recycling': False}
                event['pair']['differences_above_roundoff_screen'] = [d for d in event['pair']['analytic_prior_differences'] if above_roundoff(d)]
            else:
                event['pair_error'] = 'Require exactly one ST and one XPNR product'
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append({'event': item['event'], 'error': str(error)})
        events.append(event)
    pairs = [event['pair'] for event in events if 'pair' in event]
    summary = {'events': len(events), 'labels': sum(len(e['labels']) for e in events), 'pairs': len(pairs),
               'pairs_without_parsed_prior_differences': sum(not p['analytic_prior_differences'] for p in pairs),
               'pairs_without_raw_prior_differences': sum(not p['raw_prior_keys_with_differences'] for p in pairs),
               'pairs_with_shape_differences_above_roundoff_screen': sum(bool(p['differences_above_roundoff_screen']) for p in pairs),
               'pairs_with_equal_reference_frequency': sum(p['reference_frequency_equal'] for p in pairs),
               'pairs_with_both_candidate_families_recognized': sum(p['both_candidate_families_recognized'] for p in pairs),
               'reference_frequency_counts': dict(frequencies), 'recognition_issue_counts': dict(issues),
               'analytic_prior_key_counts': dict(keys)}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'summary': summary, 'errors': errors,
              'previous_metadata_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'full_input_hashes_recomputed': False, 'posterior_values_read': False, 'prior_samples_read': False,
              'population_outcomes_computed': False, 'released_row_prior_validated': False,
              'shape_comparison_excludes_display_keywords': ['name', 'latex_label', 'unit'],
              'roundoff_screen': {'relative_tolerance': 1e-12, 'absolute_tolerance': 1e-14, 'scientific_equivalence_certified': False},
              'note': 'New analytic-prior metadata; configuration and conversion metadata are explicitly linked to the preceding audit. No event exclusion or replacement is made.',
              'events': events}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'summary': summary, 'errors': errors}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    # Direct execution needs the project root for sibling script imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
