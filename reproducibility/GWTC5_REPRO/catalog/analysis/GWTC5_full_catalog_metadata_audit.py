"""Audit every available XPHM-XPNR pair without reading posterior values.

The audit reuses the strict settings logic applied to the prospective H3
replication. It inspects posterior schema and row counts, then compares
configuration, metadata, sampler, and analytic-prior records. Outcome metrics
cannot be calculated by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


XPHM_FAMILY = "IMRPhenomXPHM-SpinTaylor"
XPNR_FAMILY = "IMRPhenomXPNR"
H3_PARAMETERS = ("chi_eff", "luminosity_distance")
PRIMARY_PARAMETERS = (
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "luminosity_distance",
)
WEIGHT_NAMES = {
    "weight",
    "weights",
    "posterior_weight",
    "posterior_weights",
    "log_weight",
    "log_weights",
}

CONFIG_KEYS = [
    "waveform-approximant",
    "waveform-arguments-dict",
    "sampler",
    "sampler-kwargs",
    "detectors",
    "minimum-frequency",
    "maximum-frequency",
    "reference-frequency",
    "duration",
    "sampling-frequency",
    "likelihood-type",
    "calibration-marginalization",
    "calibration-model",
    "distance-marginalization",
    "phase-marginalization",
    "time-marginalization",
    "time-reference",
    "reference-frame",
    "cosmology",
    "prior-dict",
    "psd-dict",
    "channel-dict",
    "trigger-time",
    "generation-seed",
    "sampling-seed",
]

METADATA_KEYS = [
    "IFOs",
    "approximant",
    "f_low",
    "f_start",
    "f_ref",
    "f_final",
    "duration",
    "sampling_frequency",
    "delta_f",
    "start_time",
    "calibration_marginalization",
    "distance_marginalization",
    "phase_marginalization",
    "time_marginalization",
    "time_reference",
    "reference_frame",
    "cosmology",
    "frequency_domain_source_model",
    "parameter_conversion",
]

SAMPLER_KEYS = [
    "pe_algorithm",
    "nsamples",
    "ln_evidence",
    "ln_evidence_error",
    "ln_noise_evidence",
    "ln_bayes_factor",
]

PERMITTED_WAVEFORM_DIFFERENCES = {
    ("top_level", "approximant"),
    ("top_level", "description"),
    ("config", "waveform-approximant"),
    ("config", "waveform-arguments-dict"),
    ("metadata", "approximant"),
}

PERMITTED_RUN_OUTPUT_DIFFERENCES = {
    ("sampler", "nsamples"),
    ("sampler", "ln_evidence"),
    ("sampler", "ln_evidence_error"),
    ("sampler", "ln_bayes_factor"),
}

NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Metadata-only full-catalog XPHM-XPNR audit."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=root / "results" / "inventory" / "full_catalog_input_inventory.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "metadata_audit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known audit outputs from a documented rerun.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    expected = {
        "full_catalog_metadata_event_summary.csv",
        "full_catalog_pair_metadata_comparison.csv",
        "full_catalog_metadata_audit_manifest.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.is_file()]
    unexpected = [path for path in existing if path.name not in expected]
    if unexpected:
        raise FileExistsError(
            "Unexpected file(s) in audit output directory:\n"
            + "\n".join(str(path) for path in unexpected)
        )
    if existing and not overwrite:
        raise FileExistsError(
            f"Audit output already exists in {output_dir}; use --overwrite intentionally."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def convert_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and not isinstance(value, (str, list, tuple, dict)):
        try:
            return convert_value(value.item())
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return convert_value(value.tolist())
    if isinstance(value, tuple):
        return [convert_value(item) for item in value]
    if isinstance(value, list):
        return [convert_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): convert_value(item) for key, item in value.items()}
    return value


def read_dataset(source: h5py.File, path: str) -> Any:
    if path not in source or not isinstance(source[path], h5py.Dataset):
        return None
    return convert_value(source[path][()])


def read_named_datasets(
    source: h5py.File, base_path: str, names: list[str]
) -> dict[str, Any]:
    return {
        name: value
        for name in names
        if (value := read_dataset(source, f"{base_path}/{name}")) is not None
    }


def read_all_small_datasets(source: h5py.File, group_path: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if group_path not in source or not isinstance(source[group_path], h5py.Group):
        return output

    def visitor(relative_name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.size <= 1000:
            output[relative_name] = convert_value(obj[()])

    source[group_path].visititems(visitor)
    return output


def numeric_text_equivalent(left: str, right: str) -> bool:
    left_compact = re.sub(r"\s+", "", left)
    right_compact = re.sub(r"\s+", "", right)
    if NUMBER_TOKEN.split(left_compact) != NUMBER_TOKEN.split(right_compact):
        return False
    left_numbers = NUMBER_TOKEN.findall(left_compact)
    right_numbers = NUMBER_TOKEN.findall(right_compact)
    if len(left_numbers) != len(right_numbers):
        return False
    return all(
        math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-15)
        for x, y in zip(left_numbers, right_numbers, strict=True)
    )


def equivalent(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(left, str) and isinstance(right, str):
        return numeric_text_equivalent(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent(x, y) for x, y in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equivalent(left[key], right[key]) for key in left
        )
    return left == right


def comparison_category(section: str, setting: str, same: bool) -> str:
    key = (section, setting)
    if same:
        return "identical_or_numeric_serialization"
    if key in PERMITTED_WAVEFORM_DIFFERENCES:
        return "permitted_waveform_difference"
    if key in PERMITTED_RUN_OUTPUT_DIFFERENCES:
        return "permitted_run_output_difference"
    return "nonpermitted_metadata_difference"


def model_metadata(source: h5py.File, label: str) -> dict[str, dict[str, Any]]:
    base = f"/{label}"
    return {
        "top_level": {
            "approximant": read_dataset(source, f"{base}/approximant"),
            "description": read_dataset(source, f"{base}/description"),
        },
        "config": read_named_datasets(
            source, f"{base}/config_file/config", CONFIG_KEYS
        ),
        "metadata": read_named_datasets(
            source, f"{base}/meta_data/meta_data", METADATA_KEYS
        ),
        "sampler": read_named_datasets(
            source, f"{base}/meta_data/sampler", SAMPLER_KEYS
        ),
        "analytic_priors": read_all_small_datasets(
            source, f"{base}/priors/analytic"
        ),
    }


def discover_model_labels(source: h5py.File) -> tuple[str, str, str] | None:
    """Find the unique event-specific XPHM/XPNR pair and common C-tag.

    The frozen catalog summary records model families across 104 events, but
    one official event stores its pair under C01 rather than C00. Requiring a
    unique pair with the same prefix preserves the intended family comparison
    without silently excluding that event.
    """

    xphm_labels = [
        key
        for key in source.keys()
        if key.endswith(f":{XPHM_FAMILY}")
        and isinstance(source[key], h5py.Group)
        and "posterior_samples" in source[key]
    ]
    xpnr_labels = [
        key
        for key in source.keys()
        if key.endswith(f":{XPNR_FAMILY}")
        and isinstance(source[key], h5py.Group)
        and "posterior_samples" in source[key]
    ]
    if len(xphm_labels) != 1 or len(xpnr_labels) != 1:
        return None
    xphm_label = xphm_labels[0]
    xpnr_label = xpnr_labels[0]
    xphm_tag = xphm_label.split(":", 1)[0]
    xpnr_tag = xpnr_label.split(":", 1)[0]
    if xphm_tag != xpnr_tag:
        return None
    return xphm_label, xpnr_label, xphm_tag


def audit_event(
    inventory_row: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event = inventory_row["event"]
    path = Path(inventory_row["actual_path"])
    summary: dict[str, Any] = {
        "event": event,
        "prior_role": inventory_row["prior_role"],
        "screening_score_tertile_104": inventory_row[
            "screening_score_tertile_104"
        ],
        "snr_tertile_104": inventory_row["snr_tertile_104"],
        "source_path": str(path),
        "file_opened": False,
        "configuration_tag": "",
        "xphm_label": "",
        "xpnr_label": "",
        "xphm_present": False,
        "xpnr_present": False,
        "xphm_sample_count": "",
        "xpnr_sample_count": "",
        "xphm_h3_parameters_present": False,
        "xpnr_h3_parameters_present": False,
        "h3_parameters_present_both": False,
        "primary_parameters_present_both": False,
        "explicit_weight_fields": "",
        "nonpermitted_difference_count": "",
        "strict_pair_status": "audit_error",
        "all_pair_outcome_usable": False,
        "error": "",
    }
    comparison_rows: list[dict[str, Any]] = []

    try:
        with h5py.File(path, "r") as source:
            summary["file_opened"] = True
            metadata: dict[str, dict[str, dict[str, Any]]] = {}
            schema_by_model: dict[str, set[str]] = {}

            discovered = discover_model_labels(source)
            if discovered is None:
                summary["strict_pair_status"] = "missing_or_ambiguous_target_pair"
                return summary, comparison_rows
            xphm_label, xpnr_label, configuration_tag = discovered
            summary["configuration_tag"] = configuration_tag
            summary["xphm_label"] = xphm_label
            summary["xpnr_label"] = xpnr_label

            for prefix, label in (("xphm", xphm_label), ("xpnr", xpnr_label)):
                present = label in source and isinstance(source[label], h5py.Group)
                summary[f"{prefix}_present"] = present
                if not present or "posterior_samples" not in source[label]:
                    continue
                posterior = source[label]["posterior_samples"]
                if not isinstance(posterior, h5py.Dataset):
                    continue
                names = set(posterior.dtype.names or [])
                schema_by_model[label] = names
                summary[f"{prefix}_sample_count"] = int(posterior.shape[0])
                summary[f"{prefix}_h3_parameters_present"] = all(
                    parameter in names for parameter in H3_PARAMETERS
                )
                metadata[label] = model_metadata(source, label)

            summary["h3_parameters_present_both"] = bool(
                summary["xphm_h3_parameters_present"]
                and summary["xpnr_h3_parameters_present"]
            )
            summary["primary_parameters_present_both"] = bool(
                set(schema_by_model) == {xphm_label, xpnr_label}
                and all(
                    set(PRIMARY_PARAMETERS).issubset(schema_by_model[label])
                    for label in (xphm_label, xpnr_label)
                )
            )
            weights = sorted(
                {
                    f"{label}:{name}"
                    for label, names in schema_by_model.items()
                    for name in names.intersection(WEIGHT_NAMES)
                }
            )
            summary["explicit_weight_fields"] = ";".join(weights)

            if set(metadata) != {xphm_label, xpnr_label}:
                summary["strict_pair_status"] = "missing_or_ambiguous_target_pair"
                return summary, comparison_rows

            for section in sorted(
                {
                    key
                    for label in (xphm_label, xpnr_label)
                    for key in metadata[label]
                }
            ):
                settings = sorted(
                    {
                        key
                        for label in (xphm_label, xpnr_label)
                        for key in metadata[label].get(section, {})
                    }
                )
                for setting in settings:
                    left = metadata[xphm_label].get(section, {}).get(setting)
                    right = metadata[xpnr_label].get(section, {}).get(setting)
                    same = equivalent(left, right)
                    category = comparison_category(section, setting, same)
                    comparison_rows.append(
                        {
                            "event": event,
                            "section": section,
                            "setting": setting,
                            "comparison_category": category,
                            "xphm_value_json": json.dumps(
                                left, ensure_ascii=False, sort_keys=True, default=str
                            ),
                            "xpnr_value_json": json.dumps(
                                right, ensure_ascii=False, sort_keys=True, default=str
                            ),
                        }
                    )

            nonpermitted_count = sum(
                row["comparison_category"] == "nonpermitted_metadata_difference"
                for row in comparison_rows
            )
            summary["nonpermitted_difference_count"] = nonpermitted_count
            enough_rows = all(
                int(summary[f"{prefix}_sample_count"] or 0) >= 4
                for prefix in ("xphm", "xpnr")
            )
            summary["all_pair_outcome_usable"] = bool(
                summary["h3_parameters_present_both"]
                and not weights
                and enough_rows
            )

            if not summary["h3_parameters_present_both"]:
                summary["strict_pair_status"] = "missing_required_h3_parameter"
            elif weights:
                summary["strict_pair_status"] = "explicit_weights_require_rule"
            elif not enough_rows:
                summary["strict_pair_status"] = "too_few_posterior_rows"
            elif nonpermitted_count:
                summary["strict_pair_status"] = "non_tier_A_metadata_difference"
            else:
                summary["strict_pair_status"] = "strict_tier_A_automated"

    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"

    return summary, comparison_rows


def main() -> None:
    args = parse_args()
    ensure_output(args.output_dir, args.overwrite)
    inventory = read_csv(args.inventory)
    verified = [
        row for row in inventory if row["inventory_status"] == "verified"
    ]

    event_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for number, row in enumerate(verified, start=1):
        print(
            f"[{number:03d}/{len(verified):03d}] metadata audit {row['event']}",
            flush=True,
        )
        summary, comparisons = audit_event(row)
        event_summaries.append(summary)
        comparison_rows.extend(comparisons)

    summary_fields = [
        "event",
        "prior_role",
        "screening_score_tertile_104",
        "snr_tertile_104",
        "source_path",
        "file_opened",
        "configuration_tag",
        "xphm_label",
        "xpnr_label",
        "xphm_present",
        "xpnr_present",
        "xphm_sample_count",
        "xpnr_sample_count",
        "xphm_h3_parameters_present",
        "xpnr_h3_parameters_present",
        "h3_parameters_present_both",
        "primary_parameters_present_both",
        "explicit_weight_fields",
        "nonpermitted_difference_count",
        "strict_pair_status",
        "all_pair_outcome_usable",
        "error",
    ]
    comparison_fields = [
        "event",
        "section",
        "setting",
        "comparison_category",
        "xphm_value_json",
        "xpnr_value_json",
    ]
    write_csv(
        args.output_dir / "full_catalog_metadata_event_summary.csv",
        event_summaries,
        summary_fields,
    )
    write_csv(
        args.output_dir / "full_catalog_pair_metadata_comparison.csv",
        comparison_rows,
        comparison_fields,
    )

    status_counts: dict[str, int] = {}
    for row in event_summaries:
        status = str(row["strict_pair_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "purpose": "outcome-blind full-catalog matched-settings audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "h5py": h5py.__version__,
        "catalog_rows_in_inventory": len(inventory),
        "checksum_verified_events_audited": len(verified),
        "strict_status_counts": status_counts,
        "all_pair_outcome_usable_count": sum(
            bool(row["all_pair_outcome_usable"]) for row in event_summaries
        ),
        "posterior_values_loaded": False,
        "posterior_metrics_calculated": False,
        "hypothesis_tests_calculated": False,
    }
    (args.output_dir / "full_catalog_metadata_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
