"""Inventory selected PE-prior definitions without loading posterior values.

Prior strings are parsed as syntax and literal keyword values, never executed.
This is evidence collection, not certification of a reduced PE-prior density.
"""

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PRIOR_KEYS = ('chirp_mass', 'mass_ratio', 'mass_1', 'mass_2', 'a_1', 'a_2',
              'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'luminosity_distance')
CONFIG_KEYS = ('cosmology', 'conversion-function', 'default-prior', 'prior-dict',
               'prior-file', 'reference-frequency', 'reweighting-configuration',
               'reweight-nested-samples')
META_KEYS = ('cosmology', 'parameter_conversion', 'backward_spin_evolution',
             'forward_spin_evolution', 'f_ref')
LITERAL_KEYS = ('minimum', 'maximum', 'cosmology', 'equal_mass', 'boundary')


def function_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return function_name(node.value) + '.' + node.attr
    raise ValueError('Prior constructor must have a plain qualified name')


def parse_prior(text):
    """Retain class and scalar metadata without importing a constructor."""
    node = ast.parse(text, mode='eval').body
    if not isinstance(node, ast.Call) or node.args:
        raise ValueError('Expected a prior constructor with keyword arguments')
    name = function_name(node.func)
    values = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError('Expanded keyword arguments are not permitted')
        if keyword.arg in LITERAL_KEYS:
            if keyword.arg in values:
                raise ValueError('Duplicate prior keyword')
            value = ast.literal_eval(keyword.value)
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError('Expected a scalar literal')
            values[keyword.arg] = value
    return {'constructor': name, 'literal_metadata': values,
            'evaluator_executed': False, 'population_prior_validated': False}


def safe_node(group, relative, h5py):
    """Follow explicit hard-link paths only; never external or soft links."""
    node = group
    for part in relative.split('/'):
        if not isinstance(node, h5py.Group) or part not in node:
            return None
        if not isinstance(node.get(part, getlink=True), h5py.HardLink):
            raise ValueError(f'Non-hard link refused at {relative}')
        node = node[part]
    return node


def small_value(node, h5py):
    if node is None:
        return None
    if not isinstance(node, h5py.Dataset) or node.size != 1:
        return {'not_read': True, 'reason': 'not a scalar metadata dataset'}
    value = node[()]
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if not isinstance(value, (str, int, float, bool, type(None))):
        return {'not_read': True, 'reason': 'unsupported scalar representation'}
    if isinstance(value, str) and len(value.encode('utf-8')) > 65536:
        raise ValueError('Metadata string exceeds allowed size')
    return value


def label_metadata(group, h5py):
    result = {'analytic_priors': {}, 'configuration': {}, 'conversion_metadata': {}}
    for key in PRIOR_KEYS:
        raw = small_value(safe_node(group, 'priors/analytic/' + key, h5py), h5py)
        entry = {'raw': raw}
        if isinstance(raw, str):
            try:
                entry['parsed'] = parse_prior(raw)
            except (ValueError, SyntaxError) as error:
                entry['parse_error'] = str(error)
        result['analytic_priors'][key] = entry
    for key in CONFIG_KEYS:
        result['configuration'][key] = small_value(
            safe_node(group, 'config_file/config/' + key, h5py), h5py)
    for key in META_KEYS:
        result['conversion_metadata'][key] = small_value(
            safe_node(group, 'meta_data/meta_data/' + key, h5py), h5py)
    canonical = json.dumps(result, sort_keys=True, allow_nan=False).encode('utf-8')
    result['selected_metadata_sha256'] = hashlib.sha256(canonical).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / 'results/preflight') or output.exists():
        raise ValueError('Use a new JSON filename within results/preflight')
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    import h5py
    inventory = root / 'data_manifests/part_I_references/full_catalog_input_inventory.csv'
    with inventory.open(encoding='utf-8-sig', newline='') as stream:
        items = list(csv.DictReader(stream))
    if len({item['event'] for item in items}) != len(items):
        raise ValueError('Duplicate event identity in inventory')
    events, errors = [], []
    constructors, spin_limits, cosmologies = Counter(), Counter(), Counter()
    for item in items:
        event = {'event': item['event'], 'path': item['actual_path'], 'labels': {}}
        try:
            with h5py.File(item['actual_path'], 'r') as handle:
                for label in handle.keys():
                    if not (label.endswith((':IMRPhenomXPHM-SpinTaylor', ':IMRPhenomXPNR', ':NRSur7dq4'))
                            or 'mixed' in label.lower()):
                        continue
                    group = safe_node(handle, label, h5py)
                    if not isinstance(group, h5py.Group):
                        continue
                    data = label_metadata(group, h5py)
                    event['labels'][label] = data
                    for key, entry in data['analytic_priors'].items():
                        parsed = entry.get('parsed', {})
                        constructors[key + ': ' + parsed.get('constructor', 'MISSING_OR_UNPARSED')] += 1
                        if key in ('a_1', 'a_2'):
                            spin_limits[key + ': ' + json.dumps(parsed.get('literal_metadata'), sort_keys=True)] += 1
                    cosmologies[str(data['configuration'].get('cosmology'))] += 1
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
            errors.append({'event': item['event'], 'error': str(error)})
        events.append(event)
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'selected local O4b prior/configuration metadata only; not final population membership',
        'posterior_values_loaded': False, 'prior_samples_loaded': False,
        'population_outcomes_computed': False, 'arbitrary_code_executed': False,
        'full_input_hashes_recomputed': False, 'reduced_PE_prior_validated': False,
        'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
        'event_count': len(events), 'label_count': sum(len(event['labels']) for event in events),
        'prior_constructor_counts': dict(constructors), 'component_spin_prior_counts': dict(spin_limits),
        'configuration_cosmology_counts': dict(cosmologies), 'errors': errors, 'events': events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in record.items() if key != 'events'}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
